from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from simulate_causal_service_commitments import (
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


POLICY = (
    "tenant_equal_actual_service40_gain16"
)
BURST_MODES = (
    "burst_4",
    "burst_16",
)
WINDOW_CAP_FRACTION = 0.90


def pace_arrivals_window(
    *,
    original_arrivals: np.ndarray,
    requested_window_ms: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if requested_window_ms < 0.0:
        raise ValueError(
            "Pacing window must be nonnegative"
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
    effective_windows = []
    sizes = []
    capped = 0
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
        cap_ms = (
            WINDOW_CAP_FRACTION
            * period_ms
        )
        window_ms = min(
            requested_window_ms,
            cap_ms,
        )
        capped += int(
            requested_window_ms > cap_ms
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
        effective_windows.append(window_ms)
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
        "requested_pacing_window_ms": (
            requested_window_ms
        ),
        "mean_effective_pacing_window_ms": (
            float(
                np.mean(
                    effective_windows
                )
            )
        ),
        "maximum_effective_pacing_window_ms": (
            float(
                np.max(
                    effective_windows
                )
            )
        ),
        "mean_burst_period_ms": float(
            np.mean(periods)
        ),
        "mean_burst_size": float(
            np.mean(sizes)
        ),
        "maximum_burst_size": float(
            max(sizes)
        ),
        "capped_burst_fraction": (
            capped / len(sizes)
        ),
    }


def e61_records(
    root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = (
        root
        / "results"
        / "raw"
        / "burst_ingress_pacing"
        / "burst_ingress_pacing_full.json"
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
            record["burst_mode"],
            record["pacing_fraction"],
            record["policy"],
        ): record
        for record in result["records"]
    }


def build_slo_frontier(
    *,
    records: list[dict[str, Any]],
    slos_ms: list[float],
) -> list[dict[str, Any]]:
    frontier = []
    prefix_modes = sorted(
        {
            record["prefix_mode"]
            for record in records
        }
    )
    offered_loads = sorted(
        {
            record[
                "offered_load_vs_decode_capacity"
            ]
            for record in records
        }
    )
    burst_modes = sorted(
        {
            record["burst_mode"]
            for record in records
        }
    )
    for prefix_mode in prefix_modes:
        for offered_load in offered_loads:
            for burst_mode in burst_modes:
                members = [
                    record
                    for record in records
                    if record[
                        "prefix_mode"
                    ]
                    == prefix_mode
                    and record[
                        "offered_load_vs_decode_capacity"
                    ]
                    == offered_load
                    and record[
                        "burst_mode"
                    ]
                    == burst_mode
                ]
                baseline = next(
                    record
                    for record in members
                    if record[
                        "requested_pacing_window_ms"
                    ]
                    == 0.0
                )
                for slo_ms in slos_ms:
                    feasible = [
                        record
                        for record in members
                        if record["metrics"][
                            "p99_ttft_ms"
                        ]["maximum"]
                        <= slo_ms + 1e-9
                    ]
                    if not feasible:
                        frontier.append(
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
                                "ttft_slo_ms": (
                                    slo_ms
                                ),
                                "status": (
                                    "infeasible"
                                ),
                            }
                        )
                        continue
                    selected = max(
                        feasible,
                        key=lambda record: (
                            record["metrics"][
                                "throughput_output_tokens_s"
                            ]["median"],
                            -record["metrics"][
                                "p99_ttft_ms"
                            ]["maximum"],
                            -record[
                                "requested_pacing_window_ms"
                            ],
                        ),
                    )
                    metrics = selected[
                        "metrics"
                    ]
                    baseline_metrics = (
                        baseline["metrics"]
                    )
                    frontier.append(
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
                            "ttft_slo_ms": (
                                slo_ms
                            ),
                            "status": (
                                "feasible"
                            ),
                            "requested_pacing_window_ms": (
                                selected[
                                    "requested_pacing_window_ms"
                                ]
                            ),
                            "effective_pacing_window_ms": (
                                metrics[
                                    "mean_effective_"
                                    "pacing_window_ms"
                                ]["median"]
                            ),
                            "throughput_output_tokens_s": (
                                metrics[
                                    "throughput_output_"
                                    "tokens_s"
                                ]["median"]
                            ),
                            "throughput_vs_unpaced": (
                                metrics[
                                    "throughput_output_"
                                    "tokens_s"
                                ]["median"]
                                / baseline_metrics[
                                    "throughput_output_"
                                    "tokens_s"
                                ]["median"]
                            ),
                            "accepted_fraction": (
                                metrics[
                                    "accepted_fraction"
                                ]["median"]
                            ),
                            "acceptance_delta": (
                                metrics[
                                    "accepted_fraction"
                                ]["median"]
                                - baseline_metrics[
                                    "accepted_fraction"
                                ]["median"]
                            ),
                            "median_p99_ttft_ms": (
                                metrics[
                                    "p99_ttft_ms"
                                ]["median"]
                            ),
                            "maximum_p99_ttft_ms": (
                                metrics[
                                    "p99_ttft_ms"
                                ]["maximum"]
                            ),
                            "median_p99_server_ttft_ms": (
                                metrics[
                                    "p99_server_ttft_ms"
                                ]["median"]
                            ),
                            "median_p99_ingress_delay_ms": (
                                metrics[
                                    "p99_ingress_delay_ms"
                                ]["median"]
                            ),
                            "causal_target_l1_error": (
                                metrics[
                                    "causal_target_l1_error"
                                ]["median"]
                            ),
                            "mean_admission_batch_size": (
                                metrics[
                                    "mean_admission_"
                                    "batch_size"
                                ]["median"]
                            ),
                        }
                    )
    return frontier


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
    requested_windows_ms = [
        float(value)
        for value
        in args.pacing_windows_ms.split(",")
    ]
    slos_ms = [
        float(value)
        for value in args.ttft_slos_ms.split(",")
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
                for requested_window_ms in (
                    requested_windows_ms
                ):
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
                            ) = pace_arrivals_window(
                                original_arrivals=(
                                    original_arrivals
                                ),
                                requested_window_ms=(
                                    requested_window_ms
                                ),
                            )
                            metrics = simulate_policy(
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
                                policy_name=POLICY,
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
                            "requested_pacing_window_ms": (
                                requested_window_ms
                            ),
                            "policy": POLICY,
                            "metrics": (
                                aggregate_metrics(
                                    scenarios
                                )
                            ),
                            "scenarios": scenarios,
                        }
                    )

    reference = e61_records(args.root)
    equivalence = []
    equivalence_enabled = (
        args.num_requests == 2_000
        and args.warmup_requests == 200
        and args.tail_guard_requests == 100
        and args.capacity_requests == 1_500
        and args.seeds == 3
        and args.seed == 40_000
        and 0.4 in offered_loads
        and 0.0 in requested_windows_ms
        and set(burst_modes)
        == set(BURST_MODES)
    )
    if equivalence_enabled:
        for record in records:
            if (
                record[
                    "requested_pacing_window_ms"
                ]
                != 0.0
                or record[
                    "offered_load_vs_decode_capacity"
                ]
                != 0.4
            ):
                continue
            expected_record = reference[
                (
                    record["prefix_mode"],
                    0.4,
                    record["burst_mode"],
                    0.0,
                    POLICY,
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
                    and metric in scenario
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
                        "E61 zero-window "
                        "equivalence failed: "
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
                        "burst_mode": (
                            record[
                                "burst_mode"
                            ]
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
            "slo_ingress_pacing_simulation"
        ),
        "model": (
            "Absolute ingress pacing windows "
            "with robust maximum-replicate p99 "
            "TTFT SLO selection; end-to-end "
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
        "policy": POLICY,
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
            "requested_pacing_windows_ms": (
                requested_windows_ms
            ),
            "window_cap_fraction": (
                WINDOW_CAP_FRACTION
            ),
            "ttft_slos_ms": slos_ms,
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
        "slo_frontier": build_slo_frontier(
            records=records,
            slos_ms=slos_ms,
        ),
        "validation": {
            "scenario_count": (
                scenario_count
            ),
            "e61_equivalence_enabled": (
                equivalence_enabled
            ),
            "e61_equivalence_checks": (
                equivalence
            ),
            "e61_maximum_absolute_difference": (
                max(
                    item[
                        "maximum_absolute_difference"
                    ]
                    for item in equivalence
                )
                if equivalence
                else None
            ),
            "slo_selection": (
                "A window is feasible only when "
                "the maximum per-scenario p99 "
                "client TTFT is within the SLO. "
                "The feasible window with maximum "
                "median output-token throughput "
                "is selected."
            ),
        },
        "records": records,
        "limitations": [
            "The study inherits E61's fitted-service, synthetic-tenant, immediate-rejection, fixed-burst, and single-replica limitations.",
            "Absolute windows are capped at 90% of the measured inter-burst period to preserve monotonic request order.",
            "The frontier uses a finite candidate grid rather than continuous token-bucket optimization.",
            "Robust SLO feasibility uses the maximum p99 TTFT across two service replicates and three seeds, but not a production confidence interval.",
            "Only the equal-share actual-service controller is evaluated because E61 found little incremental value from commitments after pacing.",
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
        "--burst-modes",
        default="burst_4,burst_16",
    )
    parser.add_argument(
        "--pacing-windows-ms",
        default=(
            "0,50,100,150,200,300,400,600"
        ),
    )
    parser.add_argument(
        "--ttft-slos-ms",
        default="650,800,1000",
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
