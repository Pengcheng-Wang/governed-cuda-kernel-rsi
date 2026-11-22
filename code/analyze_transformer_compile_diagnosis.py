from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


REPETITION_FILES = (
    "rep1_transformer_compile_diagnosis.json",
    "rep2_transformer_compile_diagnosis.json",
    "rep3_transformer_compile_diagnosis.json",
)

ABLATION_SHAPES = (
    [4, 1024, 1024, 16, 2816],
    [1, 4096, 1024, 16, 2816],
)

FRONTIER_SHAPES = (
    [1, 512, 1024, 16, 2816],
    [4, 1024, 1024, 16, 2816],
    [1, 4096, 1024, 16, 2816],
)

ABLATION_PROTOCOLS = (
    ("legacy_cold_cache_eager_first", "torch_eager", "cold"),
    ("legacy_cold_cache_eager_first", "compile_default", "cold"),
    ("legacy_warm_cache_eager_first", "torch_eager", "warm"),
    ("legacy_warm_cache_eager_first", "compile_default", "warm"),
    ("idle_warm_cache_compile_first", "compile_default", "warm"),
    ("idle_warm_cache_compile_first", "torch_eager", "warm"),
    ("unlocked_sustained_abba", "torch_eager", "warm"),
    ("unlocked_sustained_abba", "compile_default", "warm"),
    ("locked_base_clock_abba", "torch_eager", "warm"),
    ("locked_base_clock_abba", "compile_default", "warm"),
    ("locked_boost_clock_abba", "torch_eager", "n/a"),
    ("locked_boost_clock_abba", "compile_default", "warm"),
    ("locked_boost_clock_abba", "compile_default_cold_artifact", "cold"),
)

FRONTIER_IMPLEMENTATIONS = (
    "torch_eager",
    "explicit_cuda_graph_eager",
    "compile_default",
    "compile_max_autotune_no_cudagraphs",
    "compile_reduce_overhead",
    "compile_max_autotune",
    "explicit_cuda_graph_compiled",
)

KERNEL_IMPLEMENTATIONS = (
    "torch_eager",
    "compile_default",
    "compile_max_autotune_no_cudagraphs",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def shape_label(shape: list[int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def relative_spread(values: list[float]) -> float:
    med = statistics.median(values)
    if med == 0.0:
        return 0.0
    return (max(values) - min(values)) / med


def find_ablation_record(
    payload: dict[str, Any],
    shape: list[int],
    protocol: str,
    implementation: str,
    cache_state: str,
) -> dict[str, Any]:
    return next(
        rec
        for rec in payload["records"]
        if rec["experiment"] == "dvfs_cache_order_ablation"
        and rec["shape"] == shape
        and rec["protocol"] == protocol
        and rec["implementation"] == implementation
        and rec["inductor_cache_state"] == cache_state
    )


def find_frontier_record(
    payload: dict[str, Any],
    shape: list[int],
    implementation: str,
) -> dict[str, Any]:
    return next(
        rec
        for rec in payload["records"]
        if rec["experiment"] == "compiler_mode_frontier"
        and rec["shape"] == shape
        and rec["implementation"] == implementation
    )


def find_kernel_record(
    payload: dict[str, Any],
    shape: list[int],
    implementation: str,
) -> dict[str, Any]:
    return next(
        rec
        for rec in payload["kernel_attribution_records"]
        if rec["shape"] == shape and rec["implementation"] == implementation
    )


def summarize_ablation(
    reps: list[dict[str, Any]],
    e7_original: dict[str, Any],
    e110_summary: dict[str, Any],
) -> dict[str, Any]:
    by_shape: dict[str, Any] = {}
    for shape in ABLATION_SHAPES:
        label = shape_label(shape)
        proto_rows: list[dict[str, Any]] = []
        lookup: dict[tuple[str, str, str], dict[str, Any]] = {}

        for protocol, impl, cache_state in ABLATION_PROTOCOLS:
            matched = [
                find_ablation_record(rep, shape, protocol, impl, cache_state)
                for rep in reps
            ]
            latencies = [float(r["median_ms"]) for r in matched]
            throughputs = [float(r["tokens_per_second"]) for r in matched]
            setup_s = [float(r["setup_seconds"]) for r in matched]
            min_clks = [int(r["min_sm_clock_mhz"]) for r in matched]
            max_clks = [int(r["max_sm_clock_mhz"]) for r in matched]
            med_clks = [float(r["median_sm_clock_mhz"]) for r in matched]
            powers = [float(r["mean_power_w"]) for r in matched]
            entry = {
                "protocol": protocol,
                "clock_policy": matched[0]["clock_policy"],
                "execution_order": matched[0]["execution_order"],
                "inductor_cache_state": cache_state,
                "implementation": impl,
                "median_ms": statistics.median(latencies),
                "repetition_medians_ms": latencies,
                "repeat_spread": relative_spread(latencies),
                "tokens_per_second_median": statistics.median(throughputs),
                "setup_seconds_median": statistics.median(setup_s),
                "sm_clock_min_across_reps_mhz": min(min_clks),
                "sm_clock_max_across_reps_mhz": max(max_clks),
                "sm_clock_repetition_ranges_mhz": [
                    [mn, mx] for mn, mx in zip(min_clks, max_clks, strict=True)
                ],
                "median_sm_clock_mhz": statistics.median(med_clks),
                "mean_power_w_median": statistics.median(powers),
                "loss_abs_error_max": max(float(r["loss_abs_error"]) for r in matched),
                "gradient_l2_relative_error_max": max(
                    float(r["gradient_l2_relative_error"]) for r in matched
                ),
                "gradient_elementwise_l2_relative_error_max": max(
                    float(r["gradient_elementwise_l2_relative_error"])
                    for r in matched
                ),
            }
            proto_rows.append(entry)
            lookup[(protocol, impl, cache_state)] = entry

        def pair_speedups(
            proto: str,
            eager_cache: str,
            comp_impl: str,
            comp_cache: str,
        ) -> dict[str, Any]:
            per_rep = [
                float(
                    find_ablation_record(rep, shape, proto, "torch_eager", eager_cache)[
                        "median_ms"
                    ]
                )
                / float(
                    find_ablation_record(rep, shape, proto, comp_impl, comp_cache)[
                        "median_ms"
                    ]
                )
                for rep in reps
            ]
            return {
                "median_speedup": statistics.median(per_rep),
                "min_speedup": min(per_rep),
                "max_speedup": max(per_rep),
                "repetition_speedups": per_rep,
            }

        e7_eager = float(
            next(
                r["median_ms"]
                for r in e7_original["records"]
                if r["shape"] == shape and r["implementation"] == "torch_eager"
            )
        )
        e7_comp = float(
            next(
                r["median_ms"]
                for r in e7_original["records"]
                if r["shape"] == shape and r["implementation"] == "torch_compile"
            )
        )
        e110_block = e110_summary["headline"]["representative"][
            "transformer_training"
        ][label]

        locked_1140_eager = lookup[("locked_base_clock_abba", "torch_eager", "warm")][
            "median_ms"
        ]
        locked_1140_comp = lookup[
            ("locked_base_clock_abba", "compile_default", "warm")
        ]["median_ms"]
        locked_1410_eager = lookup[("locked_boost_clock_abba", "torch_eager", "n/a")][
            "median_ms"
        ]
        locked_1410_warm = lookup[
            ("locked_boost_clock_abba", "compile_default", "warm")
        ]["median_ms"]
        locked_1410_cold = lookup[
            ("locked_boost_clock_abba", "compile_default_cold_artifact", "cold")
        ]["median_ms"]

        by_shape[label] = {
            "shape": shape,
            "protocols": proto_rows,
            "protocol_speedups": {
                "legacy_cold_cache_eager_first": pair_speedups(
                    "legacy_cold_cache_eager_first",
                    "cold",
                    "compile_default",
                    "cold",
                ),
                "legacy_warm_cache_eager_first": pair_speedups(
                    "legacy_warm_cache_eager_first",
                    "warm",
                    "compile_default",
                    "warm",
                ),
                "idle_warm_cache_compile_first": pair_speedups(
                    "idle_warm_cache_compile_first",
                    "warm",
                    "compile_default",
                    "warm",
                ),
                "unlocked_sustained_abba": pair_speedups(
                    "unlocked_sustained_abba",
                    "warm",
                    "compile_default",
                    "warm",
                ),
                "locked_1140mhz_abba": pair_speedups(
                    "locked_base_clock_abba",
                    "warm",
                    "compile_default",
                    "warm",
                ),
                "locked_1410mhz_abba": pair_speedups(
                    "locked_boost_clock_abba",
                    "n/a",
                    "compile_default",
                    "warm",
                ),
            },
            "causal_decomposition": {
                "locked_1140mhz_eager_ms": locked_1140_eager,
                "locked_1140mhz_compile_ms": locked_1140_comp,
                "locked_1140mhz_isoclock_speedup": locked_1140_eager
                / locked_1140_comp,
                "locked_1410mhz_eager_ms": locked_1410_eager,
                "locked_1410mhz_compile_warm_ms": locked_1410_warm,
                "locked_1410mhz_compile_cold_ms": locked_1410_cold,
                "locked_1410mhz_isoclock_speedup": locked_1410_eager
                / locked_1410_warm,
                "cold_vs_warm_artifact_relative_diff_at_1410mhz": abs(
                    locked_1410_cold - locked_1410_warm
                )
                / locked_1410_warm,
                "eager_clock_scaling_1140_over_1410": locked_1140_eager
                / locked_1410_eager,
                "compile_clock_scaling_1140_over_1410": locked_1140_comp
                / locked_1410_warm,
                "synthetic_cross_clock_speedup_1140_eager_over_1410_compile": (
                    locked_1140_eager / locked_1410_warm
                ),
                "synthetic_reverse_clock_ratio_1410_eager_over_1140_compile": (
                    locked_1410_eager / locked_1140_comp
                ),
                "e7_original_eager_ms": e7_eager,
                "e7_original_compile_ms": e7_comp,
                "e7_original_speedup": e7_eager / e7_comp,
                "e110_six_run_median_speedup": float(
                    e110_block["new_median_speedup"]
                ),
                "e110_six_run_min_speedup": float(
                    e110_block["new_minimum_speedup"]
                ),
                "e110_six_run_max_speedup": float(
                    e110_block["new_maximum_speedup"]
                ),
            },
        }
    return by_shape


def summarize_mode_frontier(reps: list[dict[str, Any]]) -> dict[str, Any]:
    by_shape: dict[str, Any] = {}
    for shape in FRONTIER_SHAPES:
        label = shape_label(shape)
        eager_meds = [
            float(find_frontier_record(rep, shape, "torch_eager")["median_ms"])
            for rep in reps
        ]
        eager_median_ms = statistics.median(eager_meds)
        rows: list[dict[str, Any]] = []

        for impl in FRONTIER_IMPLEMENTATIONS:
            matched = [find_frontier_record(rep, shape, impl) for rep in reps]
            latencies = [float(r["median_ms"]) for r in matched]
            throughputs = [float(r["tokens_per_second"]) for r in matched]
            setups = [float(r["setup_seconds"]) for r in matched]
            memories = [int(r["peak_incremental_memory_bytes"]) for r in matched]
            per_rep_speedups = [
                e_ms / c_ms for e_ms, c_ms in zip(eager_meds, latencies, strict=True)
            ]
            med_ms = statistics.median(latencies)
            rows.append(
                {
                    "implementation": impl,
                    "clock_policy": matched[0]["clock_policy"],
                    "median_ms": med_ms,
                    "repetition_medians_ms": latencies,
                    "repeat_spread": relative_spread(latencies),
                    "speedup_over_eager_median": statistics.median(per_rep_speedups),
                    "repetition_speedups_over_eager": per_rep_speedups,
                    "tokens_per_second_median": statistics.median(throughputs),
                    "setup_seconds_median": statistics.median(setups),
                    "peak_incremental_memory_bytes_median": int(
                        statistics.median(memories)
                    ),
                    "peak_incremental_memory_mib_median": statistics.median(memories)
                    / (1024.0**2),
                    "sm_clock_min_mhz": min(
                        int(r["min_sm_clock_mhz"]) for r in matched
                    ),
                    "sm_clock_max_mhz": max(
                        int(r["max_sm_clock_mhz"]) for r in matched
                    ),
                    "loss_abs_error_max": max(
                        float(r["loss_abs_error"]) for r in matched
                    ),
                    "gradient_l2_relative_error_max": max(
                        float(r["gradient_l2_relative_error"]) for r in matched
                    ),
                    "gradient_elementwise_l2_relative_error_max": max(
                        float(r["gradient_elementwise_l2_relative_error"])
                        for r in matched
                    ),
                }
            )

        best_row = min(rows, key=lambda item: item["median_ms"])
        by_shape[label] = {
            "shape": shape,
            "tokens_per_step": shape[0] * shape[1],
            "eager_median_ms": eager_median_ms,
            "best_implementation": best_row["implementation"],
            "best_median_ms": best_row["median_ms"],
            "best_speedup_over_eager": best_row["speedup_over_eager_median"],
            "implementations": rows,
        }
    return by_shape


def summarize_kernel_attribution(
    reps: list[dict[str, Any]],
    frontier_summary: dict[str, Any],
) -> dict[str, Any]:
    by_shape: dict[str, Any] = {}
    for shape in FRONTIER_SHAPES:
        label = shape_label(shape)
        impl_map: dict[str, Any] = {}
        for impl in KERNEL_IMPLEMENTATIONS:
            matched = [find_kernel_record(rep, shape, impl) for rep in reps]
            tot_us = statistics.median(
                float(r["total_cuda_kernel_us_per_step"]) for r in matched
            )
            gemm_us = statistics.median(
                float(r["gemm_us_per_step"]) for r in matched
            )
            attn_us = statistics.median(
                float(r["attention_us_per_step"]) for r in matched
            )
            npl_us = statistics.median(
                float(r["norm_pointwise_loss_us_per_step"]) for r in matched
            )
            kernels_cnt = statistics.median(
                float(r["cuda_kernels_per_step"]) for r in matched
            )
            launches_cnt = statistics.median(
                float(r["cpu_launch_calls_per_step"]) for r in matched
            )

            # Separate Triton template GEMM+epilogue kernels (`triton_tem_fused`) visible in top kernels
            tem_us_reps: list[float] = []
            for rec in matched:
                tem_us_reps.append(
                    sum(
                        float(k["cuda_us_per_step"])
                        for k in rec["top_cuda_kernels"]
                        if "triton_tem_fused" in k["kernel"]
                    )
                )
            triton_template_top_us = statistics.median(tem_us_reps)

            wall_ms = next(
                item["median_ms"]
                for item in frontier_summary[label]["implementations"]
                if item["implementation"] == impl
            )
            wall_us = wall_ms * 1000.0
            non_kernel_overhead_us = max(0.0, wall_us - tot_us)

            impl_map[impl] = {
                "total_cuda_kernel_us_per_step": round(tot_us, 3),
                "gemm_us_per_step": round(gemm_us, 3),
                "attention_us_per_step": round(attn_us, 3),
                "norm_pointwise_loss_and_triton_epilogue_us_per_step": round(
                    npl_us, 3
                ),
                "triton_template_gemm_epilogue_top8_us_per_step": round(
                    triton_template_top_us, 3
                ),
                "gemm_plus_attention_share": round((gemm_us + attn_us) / tot_us, 6),
                "fusible_non_gemm_attn_share": round(npl_us / tot_us, 6),
                "cuda_kernels_per_step": round(kernels_cnt, 2),
                "cpu_launch_calls_per_step": round(launches_cnt, 2),
                "wall_clock_step_us": round(wall_us, 3),
                "non_kernel_host_or_gap_us": round(non_kernel_overhead_us, 3),
                "non_kernel_host_or_gap_fraction_of_wall": round(
                    non_kernel_overhead_us / wall_us, 6
                ),
                "top_cuda_kernels_rep1": matched[0]["top_cuda_kernels"][:6],
            }

        eager_tot = impl_map["torch_eager"]["total_cuda_kernel_us_per_step"]
        def_tot = impl_map["compile_default"]["total_cuda_kernel_us_per_step"]
        max_tot = impl_map["compile_max_autotune_no_cudagraphs"][
            "total_cuda_kernel_us_per_step"
        ]
        def_wall_speedup = next(
            item["speedup_over_eager_median"]
            for item in frontier_summary[label]["implementations"]
            if item["implementation"] == "compile_default"
        )
        max_wall_speedup = next(
            item["speedup_over_eager_median"]
            for item in frontier_summary[label]["implementations"]
            if item["implementation"] == "compile_max_autotune_no_cudagraphs"
        )

        by_shape[label] = {
            "shape": shape,
            "implementations": impl_map,
            "amdahl_closure": {
                "eager_vendor_gemm_plus_attention_share": impl_map["torch_eager"][
                    "gemm_plus_attention_share"
                ],
                "eager_fusible_non_gemm_attn_share": impl_map["torch_eager"][
                    "fusible_non_gemm_attn_share"
                ],
                "compile_default_non_gemm_attn_reduction_fraction": round(
                    1.0
                    - (
                        impl_map["compile_default"][
                            "norm_pointwise_loss_and_triton_epilogue_us_per_step"
                        ]
                        / impl_map["torch_eager"][
                            "norm_pointwise_loss_and_triton_epilogue_us_per_step"
                        ]
                    ),
                    6,
                ),
                "compile_default_kernel_sum_speedup": round(eager_tot / def_tot, 6),
                "compile_default_wall_clock_speedup": round(def_wall_speedup, 6),
                "compile_default_kernel_vs_wall_relative_gap": round(
                    abs((eager_tot / def_tot) - def_wall_speedup) / def_wall_speedup,
                    6,
                ),
                "max_autotune_kernel_sum_speedup": round(eager_tot / max_tot, 6),
                "max_autotune_wall_clock_speedup": round(max_wall_speedup, 6),
            },
        }
    return by_shape


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.root / args.input_dir
    smoke = load_json(input_dir / "smoke_transformer_compile_diagnosis.json")
    reps = [load_json(input_dir / name) for name in REPETITION_FILES]
    e7_original = load_json(
        args.root / "results/raw/full/transformer_training_full.json"
    )
    e110_summary = load_json(
        args.root
        / "results/raw/independent_a100_reproduction/independent_a100_reproduction_full.json"
    )

    ablation = summarize_ablation(reps, e7_original, e110_summary)
    frontier = summarize_mode_frontier(reps)
    kernels = summarize_kernel_attribution(reps, frontier)

    all_full_records = [rec for rep in reps for rec in rep["records"]]
    all_smoke_records = list(smoke["records"])

    env0 = reps[0]["environment"]
    e7_env = e7_original["environment"]
    env_fields = (
        "gpu_name",
        "gpu_memory_bytes",
        "compute_capability",
        "torch",
        "torch_cuda",
        "triton",
        "flash_attn",
        "cuda_driver",
    )
    env_match = all(
        rep["environment"].get(k) == e7_env.get(k)
        for rep in reps
        for k in env_fields
    )

    locked_1140_records = [
        rec for rec in all_full_records if rec.get("clock_policy") == "locked_1140mhz"
    ]
    locked_1410_records = [
        rec for rec in all_full_records if rec.get("clock_policy") == "locked_1410mhz"
    ]
    locked_clocks_stable = (
        all(
            int(r["min_sm_clock_mhz"]) == 1140 and int(r["max_sm_clock_mhz"]) == 1140
            for r in locked_1140_records
        )
        and all(
            float(r["median_sm_clock_mhz"]) >= 1395.0
            and int(r["min_sm_clock_mhz"]) >= 1380
            and int(r["max_sm_clock_mhz"]) == 1410
            for r in locked_1410_records
        )
        and all(
            item["repeat_spread"] <= 0.01
            for label in ("4x1024", "1x4096")
            for item in frontier[label]["implementations"]
        )
    )

    unlocked_restored = all(
        int(rep["environment"]["initial_telemetry"]["sm_clock_mhz"]) <= 1140
        for rep in reps
    )

    cache_diff_max = max(
        ablation[label]["causal_decomposition"][
            "cold_vs_warm_artifact_relative_diff_at_1410mhz"
        ]
        for label in ("4x1024", "1x4096")
    )

    # Check that cross-clock ratio explains E7 & E110 high outliers while iso-clock matches E110 6-run median
    dvfs_closure_ok = True
    for label in ("4x1024", "1x4096"):
        cd = ablation[label]["causal_decomposition"]
        if not (
            cd["synthetic_cross_clock_speedup_1140_eager_over_1410_compile"]
            >= cd["e7_original_speedup"] * 0.98
        ):
            dvfs_closure_ok = False
        if (
            abs(cd["locked_1410mhz_isoclock_speedup"] - cd["e110_six_run_median_speedup"])
            / cd["e110_six_run_median_speedup"]
            > 0.02
        ):
            dvfs_closure_ok = False

    amdahl_closure_ok = all(
        kernels[label]["amdahl_closure"]["compile_default_kernel_vs_wall_relative_gap"]
        <= 0.01
        for label in ("4x1024", "1x4096")
    )

    small_graph_speedup = frontier["1x512"]["best_speedup_over_eager"]
    regime_crossover_ok = (
        frontier["1x512"]["best_implementation"] == "explicit_cuda_graph_compiled"
        and small_graph_speedup >= 3.0
        and frontier["4x1024"]["best_implementation"]
        == "compile_max_autotune_no_cudagraphs"
        and frontier["1x4096"]["best_implementation"]
        == "compile_max_autotune_no_cudagraphs"
    )

    max_loss_err_full = max(float(r["loss_abs_error"]) for r in all_full_records)
    max_grad_l2_rel_full = max(
        float(r["gradient_l2_relative_error"]) for r in all_full_records
    )
    max_grad_elem_rel_full = max(
        float(r["gradient_elementwise_l2_relative_error"]) for r in all_full_records
    )
    max_loss_err_smoke = max(float(r["loss_abs_error"]) for r in all_smoke_records)
    max_grad_elem_rel_smoke = max(
        float(r["gradient_elementwise_l2_relative_error"]) for r in all_smoke_records
    )

    validation = {
        "smoke_present_and_valid": (
            len(all_smoke_records) == 20
            and len(smoke["kernel_attribution_records"]) == 3
            and max_loss_err_smoke <= 1e-4
            and max_grad_elem_rel_smoke <= 1e-2
        ),
        "three_full_process_repetitions": (
            len(reps) == 3
            and all(len(rep["records"]) == 47 for rep in reps)
            and all(len(rep["kernel_attribution_records"]) == 9 for rep in reps)
        ),
        "environment_parity_with_e110": env_match,
        "correctness_gate_passed": (
            max_loss_err_full <= 1e-4
            and max_grad_l2_rel_full <= 1e-3
            and max_grad_elem_rel_full <= 1e-2
        ),
        "locked_clock_stability_gate_passed": locked_clocks_stable and unlocked_restored,
        "inductor_cache_invariance_verified": cache_diff_max <= 0.005,
        "dvfs_explains_e7_and_e110_spread": dvfs_closure_ok,
        "amdahl_kernel_closure_verified": amdahl_closure_ok,
        "regime_crossover_verified": regime_crossover_ok,
    }
    validation["all_gates_passed"] = all(validation.values())
    if not validation["all_gates_passed"]:
        failed = [k for k, v in validation.items() if not v]
        raise RuntimeError(f"E111 validation gates failed: {failed}")

    return {
        "experiment": "E111_transformer_compile_diagnosis",
        "suite": "transformer_compile_diagnosis",
        "hardware": {
            "gpu_name": env0["gpu_name"],
            "compute_capability": env0["compute_capability"],
            "gpu_memory_bytes": env0["gpu_memory_bytes"],
            "cuda_driver": env0["cuda_driver"],
            "torch": env0["torch"],
            "torch_cuda": env0["torch_cuda"],
            "triton": env0["triton"],
            "flash_attn": env0["flash_attn"],
            "nvml_max_clocks": env0["nvml_max_clocks"],
            "repetition_count": len(reps),
        },
        "correctness_summary": {
            "full_max_loss_abs_error": max_loss_err_full,
            "full_max_gradient_l2_relative_error": max_grad_l2_rel_full,
            "full_max_gradient_elementwise_l2_relative_error": max_grad_elem_rel_full,
            "smoke_max_loss_abs_error": max_loss_err_smoke,
            "smoke_max_gradient_elementwise_l2_relative_error": max_grad_elem_rel_smoke,
        },
        "dvfs_cache_order_ablation": ablation,
        "compiler_mode_frontier_1410mhz": frontier,
        "cuda_kernel_attribution_1410mhz": kernels,
        "key_findings": {
            "root_cause_of_e7_e110_gap": (
                "Unlocked A100 DVFS transitions between 1140 MHz default application clock "
                "and 1410 MHz max boost clock (1.237x SM frequency ratio) during 3-step warmup, "
                "not TorchInductor persistent-cache codegen regression."
            ),
            "cold_vs_warm_cache_max_diff_at_1410mhz": cache_diff_max,
            "isoclock_1410mhz_compile_default_speedups": {
                "1x512": frontier["1x512"]["implementations"][2][
                    "speedup_over_eager_median"
                ],
                "4x1024": ablation["4x1024"]["causal_decomposition"][
                    "locked_1410mhz_isoclock_speedup"
                ],
                "1x4096": ablation["1x4096"]["causal_decomposition"][
                    "locked_1410mhz_isoclock_speedup"
                ],
            },
            "best_1410mhz_mode_speedups": {
                label: {
                    "winner": frontier[label]["best_implementation"],
                    "speedup_over_eager": frontier[label]["best_speedup_over_eager"],
                    "median_ms": frontier[label]["best_median_ms"],
                }
                for label in ("1x512", "4x1024", "1x4096")
            },
        },
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw/transformer_compile_diagnosis"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/raw/transformer_compile_diagnosis/"
            "transformer_compile_diagnosis_full.json"
        ),
    )
    args = parser.parse_args()
    payload = run_analysis(args)
    output_path = (
        args.output if args.output.is_absolute() else (args.root / args.output)
    )
    write_json(output_path, payload)


if __name__ == "__main__":
    main()
