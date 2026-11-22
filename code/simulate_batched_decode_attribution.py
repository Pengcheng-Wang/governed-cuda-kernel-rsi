from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np

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
from simulate_prefill_decode_scheduling import (
    PREFIX_MODES,
    load_prefill_service_model,
)
from simulate_prefill_stall_admission import (
    remaining_prefill_service_ms,
)
from simulate_tenant_reservation_fairness import (
    TENANTS,
    estimate_tenant_capacity,
    make_tenant_workload,
    simulate_tenant_policy,
    standalone_decode_lookup,
    tenant_specifications,
)


POLICIES = (
    "prompt_equal_prefill_service40",
    "tenant_equal_e2e_service320",
    "tenant_equal_e2e_service320_gain8",
    "tenant_reserved_e2e_service320",
)


def make_decode_observer(
    *,
    num_requests: int,
    decode_model: dict[str, Any],
    decode_service_models: dict[str, Any],
) -> tuple[
    Callable[[dict[str, Any]], None],
    dict[str, Any],
]:
    causal_ms = np.zeros(
        num_requests,
        dtype=np.float64,
    )
    equal_ms = np.zeros(
        num_requests,
        dtype=np.float64,
    )
    state: dict[str, Any] = {
        "causal_ms": causal_ms,
        "equal_ms": equal_ms,
        "step_count": 0,
        "batch_sizes": [],
        "observed_total_ms": 0.0,
        "maximum_step_error_ms": 0.0,
    }

    def observe(event: dict[str, Any]) -> None:
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
        if (
            requests.size < 1
            or contexts.size != requests.size
        ):
            raise RuntimeError(
                "Invalid decode observer event"
            )
        batch = int(requests.size)
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
        causal_allocation = (
            common_per_request
            + LAYERS
            * coefficients[
                "context_token_ms"
            ]
            * contexts
        )
        equal_allocation = np.full(
            batch,
            service_ms / batch,
            dtype=np.float64,
        )
        causal_error = abs(
            float(np.sum(causal_allocation))
            - service_ms
        )
        equal_error = abs(
            float(np.sum(equal_allocation))
            - service_ms
        )
        step_error = max(
            causal_error,
            equal_error,
        )
        if step_error > 1e-9:
            raise RuntimeError(
                "Decode allocation did not sum "
                f"to the step: {step_error=}"
            )
        causal_ms[requests] += (
            causal_allocation
        )
        equal_ms[requests] += (
            equal_allocation
        )
        state["step_count"] += 1
        state["batch_sizes"].append(batch)
        state["observed_total_ms"] += (
            service_ms
        )
        state[
            "maximum_step_error_ms"
        ] = max(
            state[
                "maximum_step_error_ms"
            ],
            step_error,
        )

    return observe, state


def l1_share_error(
    first: list[float],
    second: list[float],
) -> float:
    return 0.5 * sum(
        abs(left - right)
        for left, right in zip(
            first,
            second,
        )
    )


def attribution_metrics(
    *,
    workload: dict[str, np.ndarray],
    observer_state: dict[str, Any],
    base_metrics: dict[str, float],
    decode_lookup: dict[
        tuple[int, int],
        float,
    ],
    prefill_model: dict[str, Any],
    analysis_start: int,
    analysis_end: int,
) -> dict[str, float]:
    causal_all = observer_state[
        "causal_ms"
    ]
    equal_all = observer_state["equal_ms"]
    observer_difference = abs(
        observer_state["observed_total_ms"]
        - base_metrics["decode_busy_ms"]
    )
    if observer_difference > 1e-8:
        raise RuntimeError(
            "Observer did not cover decode busy "
            f"time: {observer_difference=}"
        )
    window = np.arange(
        analysis_start,
        analysis_end,
        dtype=np.int64,
    )
    selected = window[
        causal_all[window] > 0.0
    ]
    if selected.size < 1:
        raise RuntimeError(
            "Attribution found no accepted "
            "analysis requests"
        )
    prompt_tokens = workload[
        "prompt_tokens"
    ]
    output_tokens = workload[
        "output_tokens"
    ]
    cached_tokens = workload[
        "cached_tokens"
    ]
    uncached_tokens = workload[
        "uncached_tokens"
    ]
    tenant_ids = workload["tenant_ids"]
    standalone_decode = np.asarray(
        [
            decode_lookup[
                (
                    int(prompt_tokens[request]),
                    int(output_tokens[request]),
                )
            ]
            for request in selected
        ],
        dtype=np.float64,
    )
    causal_decode = causal_all[selected]
    equal_decode = equal_all[selected]
    prefill_service = np.asarray(
        [
            remaining_prefill_service_ms(
                remaining_tokens=int(
                    uncached_tokens[request]
                ),
                prefix_tokens=int(
                    cached_tokens[request]
                ),
                budget_ms=100.0,
                prefill_model=prefill_model,
            )
            for request in selected
        ],
        dtype=np.float64,
    )
    standalone_total = (
        prefill_service
        + standalone_decode
    )
    causal_total = (
        prefill_service + causal_decode
    )
    equal_total = (
        prefill_service + equal_decode
    )
    decode_ratios = (
        standalone_decode / causal_decode
    )
    total_ratios = (
        standalone_total / causal_total
    )
    batch_sizes = np.asarray(
        observer_state["batch_sizes"],
        dtype=np.float64,
    )
    metrics = {
        "observer_step_count": float(
            observer_state["step_count"]
        ),
        "observer_mean_batch_size": float(
            np.mean(batch_sizes)
        ),
        "observer_p95_batch_size": float(
            np.percentile(
                batch_sizes,
                95,
            )
        ),
        "observer_decode_busy_difference_ms": (
            observer_difference
        ),
        "observer_maximum_step_error_ms": float(
            observer_state[
                "maximum_step_error_ms"
            ]
        ),
        "accepted_attribution_requests": float(
            selected.size
        ),
        "standalone_decode_over_causal_ratio": (
            float(
                np.sum(standalone_decode)
                / np.sum(causal_decode)
            )
        ),
        "standalone_total_over_causal_ratio": (
            float(
                np.sum(standalone_total)
                / np.sum(causal_total)
            )
        ),
        "mean_standalone_decode_ms": float(
            np.mean(standalone_decode)
        ),
        "mean_causal_decode_ms": float(
            np.mean(causal_decode)
        ),
        "mean_equal_decode_ms": float(
            np.mean(equal_decode)
        ),
        "p50_standalone_to_causal_decode_ratio": (
            float(
                np.percentile(
                    decode_ratios,
                    50,
                )
            )
        ),
        "p95_standalone_to_causal_decode_ratio": (
            float(
                np.percentile(
                    decode_ratios,
                    95,
                )
            )
        ),
        "p50_standalone_to_causal_total_ratio": (
            float(
                np.percentile(
                    total_ratios,
                    50,
                )
            )
        ),
        "p95_standalone_to_causal_total_ratio": (
            float(
                np.percentile(
                    total_ratios,
                    95,
                )
            )
        ),
        "equal_decode_l1_fraction": float(
            np.sum(
                np.abs(
                    equal_decode
                    - causal_decode
                )
            )
            / np.sum(causal_decode)
        ),
    }
    tenant_standalone_decode_shares = []
    tenant_causal_decode_shares = []
    tenant_equal_decode_shares = []
    tenant_standalone_total_shares = []
    tenant_causal_total_shares = []
    tenant_equal_total_shares = []
    for tenant in TENANTS:
        tenant_id = int(tenant["id"])
        name = tenant["name"]
        mask = (
            tenant_ids[selected] == tenant_id
        )
        if not np.any(mask):
            raise RuntimeError(
                f"No accepted tenant {name}"
            )
        standalone_decode_share = float(
            np.sum(standalone_decode[mask])
            / np.sum(standalone_decode)
        )
        causal_decode_share = float(
            np.sum(causal_decode[mask])
            / np.sum(causal_decode)
        )
        equal_decode_share = float(
            np.sum(equal_decode[mask])
            / np.sum(equal_decode)
        )
        standalone_total_share = float(
            np.sum(standalone_total[mask])
            / np.sum(standalone_total)
        )
        causal_total_share = float(
            np.sum(causal_total[mask])
            / np.sum(causal_total)
        )
        equal_total_share = float(
            np.sum(equal_total[mask])
            / np.sum(equal_total)
        )
        metrics[
            f"tenant_{name}_standalone_decode_share"
        ] = standalone_decode_share
        metrics[
            f"tenant_{name}_causal_decode_share"
        ] = causal_decode_share
        metrics[
            f"tenant_{name}_equal_decode_share"
        ] = equal_decode_share
        metrics[
            f"tenant_{name}_standalone_total_share"
        ] = standalone_total_share
        metrics[
            f"tenant_{name}_causal_total_share"
        ] = causal_total_share
        metrics[
            f"tenant_{name}_equal_total_share"
        ] = equal_total_share
        metrics[
            f"tenant_{name}_standalone_decode_share_error"
        ] = (
            standalone_decode_share
            - causal_decode_share
        )
        metrics[
            f"tenant_{name}_equal_decode_share_error"
        ] = (
            equal_decode_share
            - causal_decode_share
        )
        metrics[
            f"tenant_{name}_standalone_total_share_error"
        ] = (
            standalone_total_share
            - causal_total_share
        )
        metrics[
            f"tenant_{name}_equal_total_share_error"
        ] = (
            equal_total_share
            - causal_total_share
        )
        tenant_ratios = (
            standalone_decode[mask]
            / causal_decode[mask]
        )
        metrics[
            f"tenant_{name}_mean_standalone_to_causal_decode_ratio"
        ] = float(
            np.mean(tenant_ratios)
        )
        metrics[
            f"tenant_{name}_p95_standalone_to_causal_decode_ratio"
        ] = float(
            np.percentile(
                tenant_ratios,
                95,
            )
        )
        tenant_standalone_decode_shares.append(
            standalone_decode_share
        )
        tenant_causal_decode_shares.append(
            causal_decode_share
        )
        tenant_equal_decode_shares.append(
            equal_decode_share
        )
        tenant_standalone_total_shares.append(
            standalone_total_share
        )
        tenant_causal_total_shares.append(
            causal_total_share
        )
        tenant_equal_total_shares.append(
            equal_total_share
        )
    metrics[
        "standalone_decode_share_l1_error"
    ] = l1_share_error(
        tenant_standalone_decode_shares,
        tenant_causal_decode_shares,
    )
    metrics[
        "equal_decode_share_l1_error"
    ] = l1_share_error(
        tenant_equal_decode_shares,
        tenant_causal_decode_shares,
    )
    metrics[
        "standalone_total_share_l1_error"
    ] = l1_share_error(
        tenant_standalone_total_shares,
        tenant_causal_total_shares,
    )
    metrics[
        "equal_total_share_l1_error"
    ] = l1_share_error(
        tenant_equal_total_shares,
        tenant_causal_total_shares,
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
                    decode_lookup = (
                        standalone_decode_lookup(
                            decode_model=decode_model,
                            decode_service_models=(
                                decode_service_models
                            ),
                        )
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
                        (
                            observer,
                            observer_state,
                        ) = make_decode_observer(
                            num_requests=(
                                args.num_requests
                            ),
                            decode_model=decode_model,
                            decode_service_models=(
                                decode_service_models
                            ),
                        )
                        metrics = simulate_tenant_policy(
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
                            decode_service_observer=(
                                observer
                            ),
                        )
                        metrics.update(
                            attribution_metrics(
                                workload=workload,
                                observer_state=(
                                    observer_state
                                ),
                                base_metrics=metrics,
                                decode_lookup=(
                                    decode_lookup
                                ),
                                prefill_model=(
                                    prefill_model
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
            "batched_decode_service_"
            "attribution_simulation"
        ),
        "model": (
            "E57 tenant policies with exact "
            "additive attribution of each fitted "
            "batched decode step, compared with "
            "standalone batch-1 and equal-share "
            "decode accounting."
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
        "policies": POLICIES,
        "attribution": {
            "causal_rule": (
                "Each active request receives "
                "LAYERS * (batch_ms + "
                "context_token_ms * context + "
                "(attention_intercept_ms + "
                "mlp_ms(batch)) / batch)."
            ),
            "equal_rule": (
                "Each active request receives "
                "decode_step_ms / batch."
            ),
            "standalone_rule": (
                "Sum measured-service batch-1 "
                "decode steps for the request's "
                "prompt and output lengths."
            ),
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
                "The observer covers every decode "
                "step without changing policy "
                "metrics; exact and equal "
                "allocations each sum to every "
                "step and the observed total equals "
                "decode_busy_ms."
            ),
        },
        "records": records,
        "limitations": [
            "The study inherits all E57 limitations and audits a fitted service model rather than a live serving-engine trace.",
            "The exact causal attribution is exact only for the affine fitted attention model plus batch-dependent MLP term; real kernels have nonlinear interactions and scheduler overhead.",
            "Standalone batch-1 service is an offered-work proxy, while causal attribution exists only for accepted requests that actually execute.",
            "The observer attributes GPU step time, not CPU scheduling, sampling, networking, or queue occupancy.",
            "The audit does not feed causal service back into admission; it measures the error in E57 accounting before designing a new controller.",
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
