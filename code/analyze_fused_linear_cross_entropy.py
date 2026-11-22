from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


REPETITION_FILES = (
    "rep1_fused_linear_cross_entropy.json",
    "rep2_fused_linear_cross_entropy.json",
    "rep3_fused_linear_cross_entropy.json",
)

SCALING_SHAPES = (
    [4096, 1024, 128256],
    [8192, 1024, 128256],
    [16384, 1024, 128256],
    [8192, 2048, 128256],
    [8192, 4096, 128256],
)

SCALING_IMPLEMENTATIONS = (
    "eager_fp32_logits_ce",
    "eager_bf16_logits_ce",
    "torch_compile_full_ce",
    "checkpointed_chunked_compile_ce",
    "triton_fused_linear_ce_bf16_dw",
    "triton_fused_linear_ce_fp32_dw",
)

CHUNK_SHAPES = (
    [8192, 1024, 128256],
    [8192, 4096, 128256],
)

CHUNK_SIZES = (128, 256, 512, 1024, 2048, 4096, 8192)

KERNEL_IMPLEMENTATIONS = (
    "eager_fp32_logits_ce",
    "torch_compile_full_ce",
    "checkpointed_chunked_compile_ce",
    "triton_fused_linear_ce_bf16_dw",
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
    return f"{shape[0]}x{shape[1]}x{shape[2]}"


def relative_spread(values: list[float]) -> float:
    med = statistics.median(values)
    if med == 0.0:
        return 0.0
    return (max(values) - min(values)) / med


def find_scaling_record(
    payload: dict[str, Any],
    shape: list[int],
    implementation: str,
) -> dict[str, Any]:
    return next(
        rec
        for rec in payload["records"]
        if rec["experiment"] == "slm_llm_vocab_scaling"
        and rec["shape_bt_h_v"] == shape
        and rec["implementation"] == implementation
    )


def find_chunk_record(
    payload: dict[str, Any],
    shape: list[int],
    implementation: str,
    chunk_size: int,
) -> dict[str, Any]:
    return next(
        rec
        for rec in payload["records"]
        if rec["experiment"] == "chunk_size_frontier"
        and rec["shape_bt_h_v"] == shape
        and rec["implementation"] == implementation
        and int(rec["chunk_size"]) == chunk_size
    )


def find_kernel_record(
    payload: dict[str, Any],
    shape: list[int],
    implementation: str,
) -> dict[str, Any]:
    return next(
        rec
        for rec in payload["kernel_attribution_records"]
        if rec["shape_bt_h_v"] == shape and rec["implementation"] == implementation
    )


def summarize_scaling(reps: list[dict[str, Any]]) -> dict[str, Any]:
    by_shape: dict[str, Any] = {}
    for shape in SCALING_SHAPES:
        label = shape_label(shape)
        bt, hidden, vocab = shape
        eager_fp32_rows = [
            find_scaling_record(rep, shape, "eager_fp32_logits_ce") for rep in reps
        ]
        eager_ms_reps = [float(r["median_ms"]) for r in eager_fp32_rows]
        eager_trans_mib = statistics.median(
            float(r["peak_transient_workspace_bytes"]) / (1024.0**2)
            for r in eager_fp32_rows
        )
        eager_incr_mib = statistics.median(
            float(r["peak_incremental_memory_bytes"]) / (1024.0**2)
            for r in eager_fp32_rows
        )

        impl_rows: list[dict[str, Any]] = []
        lookup: dict[str, dict[str, Any]] = {}
        for impl in SCALING_IMPLEMENTATIONS:
            matched = [find_scaling_record(rep, shape, impl) for rep in reps]
            latencies = [float(r["median_ms"]) for r in matched]
            tflops = [float(r["effective_3gemm_tflop_s"]) for r in matched]
            tok_s = [float(r["tokens_per_second"]) for r in matched]
            trans_mib_reps = [
                float(r["peak_transient_workspace_bytes"]) / (1024.0**2)
                for r in matched
            ]
            incr_mib_reps = [
                float(r["peak_incremental_memory_bytes"]) / (1024.0**2)
                for r in matched
            ]
            grad_mib_reps = [
                float(r["persistent_grad_bytes"]) / (1024.0**2) for r in matched
            ]
            speedup_reps = [
                e_ms / m_ms for e_ms, m_ms in zip(eager_ms_reps, latencies, strict=True)
            ]

            med_trans_mib = statistics.median(trans_mib_reps)
            med_incr_mib = statistics.median(incr_mib_reps)
            entry = {
                "implementation": impl,
                "chunk_size": int(matched[0]["chunk_size"]),
                "median_ms": statistics.median(latencies),
                "repetition_medians_ms": latencies,
                "repeat_spread": relative_spread(latencies),
                "speedup_over_eager_fp32_median": statistics.median(speedup_reps),
                "repetition_speedups_over_eager_fp32": speedup_reps,
                "effective_3gemm_tflop_s_median": statistics.median(tflops),
                "tokens_per_second_median": statistics.median(tok_s),
                "peak_transient_workspace_mib_median": med_trans_mib,
                "peak_incremental_memory_mib_median": med_incr_mib,
                "persistent_grad_mib_median": statistics.median(grad_mib_reps),
                "transient_memory_reduction_vs_eager_fp32": eager_trans_mib
                / max(med_trans_mib, 1e-9),
                "incremental_memory_reduction_vs_eager_fp32": eager_incr_mib
                / max(med_incr_mib, 1e-9),
                "loss_abs_error_vs_fp32_max": max(
                    float(r["loss_abs_error_vs_fp32"]) for r in matched
                ),
                "dx_l2_rel_error_vs_fp32_max": max(
                    float(r["dx_l2_rel_error_vs_fp32"]) for r in matched
                ),
                "dw_l2_rel_error_vs_fp32_max": max(
                    float(r["dw_l2_rel_error_vs_fp32"]) for r in matched
                ),
                "scaled_backward_chain_rule_rel_error_max": max(
                    float(r["scaled_backward_chain_rule_rel_error"]) for r in matched
                ),
                "sm_clock_min_mhz": min(int(r["min_sm_clock_mhz"]) for r in matched),
                "sm_clock_max_mhz": max(int(r["max_sm_clock_mhz"]) for r in matched),
                "mean_power_w_median": statistics.median(
                    float(r["mean_power_w"]) for r in matched
                ),
            }
            impl_rows.append(entry)
            lookup[impl] = entry

        expected_full_bf16_logits_mib = (bt * vocab * 2) / (1024.0**2)
        expected_chunk_bf16_logits_mib = (1024 * vocab * 2) / (1024.0**2)

        by_shape[label] = {
            "shape_bt_h_v": shape,
            "vocab_to_hidden_ratio": vocab / hidden,
            "theoretical_full_bf16_logits_mib": expected_full_bf16_logits_mib,
            "theoretical_chunk1024_bf16_logits_mib": expected_chunk_bf16_logits_mib,
            "implementations": impl_rows,
            "comparisons": {
                "triton_bf16_dw_speedup_over_eager_fp32": lookup[
                    "triton_fused_linear_ce_bf16_dw"
                ]["speedup_over_eager_fp32_median"],
                "triton_bf16_dw_speedup_over_checkpointed_chunk": lookup[
                    "checkpointed_chunked_compile_ce"
                ]["median_ms"]
                / lookup["triton_fused_linear_ce_bf16_dw"]["median_ms"],
                "triton_bf16_dw_latency_ratio_vs_unchunked_compile": lookup[
                    "triton_fused_linear_ce_bf16_dw"
                ]["median_ms"]
                / lookup["torch_compile_full_ce"]["median_ms"],
                "triton_bf16_dw_transient_reduction_vs_unchunked_compile": lookup[
                    "torch_compile_full_ce"
                ]["peak_transient_workspace_mib_median"]
                / lookup["triton_fused_linear_ce_bf16_dw"][
                    "peak_transient_workspace_mib_median"
                ],
                "triton_bf16_dw_transient_reduction_vs_eager_fp32": lookup[
                    "triton_fused_linear_ce_bf16_dw"
                ]["transient_memory_reduction_vs_eager_fp32"],
                "eager_bf16_over_triton_loss_error_ratio": lookup[
                    "eager_bf16_logits_ce"
                ]["loss_abs_error_vs_fp32_max"]
                / max(
                    lookup["triton_fused_linear_ce_bf16_dw"][
                        "loss_abs_error_vs_fp32_max"
                    ],
                    1e-12,
                ),
            },
        }
    return by_shape


def summarize_chunk_frontier(reps: list[dict[str, Any]]) -> dict[str, Any]:
    by_shape: dict[str, Any] = {}
    for shape in CHUNK_SHAPES:
        label = shape_label(shape)
        _, hidden, vocab = shape
        mode_map: dict[str, Any] = {}

        for impl in (
            "triton_fused_linear_ce_bf16_dw",
            "triton_fused_linear_ce_fp32_dw",
        ):
            rows: list[dict[str, Any]] = []
            for cs in CHUNK_SIZES:
                matched = [find_chunk_record(rep, shape, impl, cs) for rep in reps]
                latencies = [float(r["median_ms"]) for r in matched]
                tflops = [float(r["effective_3gemm_tflop_s"]) for r in matched]
                trans_mib = statistics.median(
                    float(r["peak_transient_workspace_bytes"]) / (1024.0**2)
                    for r in matched
                )
                dw_errs = [float(r["dw_l2_rel_error_vs_fp32"]) for r in matched]
                rows.append(
                    {
                        "chunk_size": cs,
                        "num_chunks": int(matched[0]["num_chunks"]),
                        "median_ms": statistics.median(latencies),
                        "repetition_medians_ms": latencies,
                        "repeat_spread": relative_spread(latencies),
                        "effective_3gemm_tflop_s_median": statistics.median(tflops),
                        "peak_transient_workspace_mib_median": trans_mib,
                        "theoretical_tile_plus_accum_mib": (cs * vocab * 2) / (1024.0**2)
                        + (
                            (vocab * hidden * 4) / (1024.0**2)
                            if impl.endswith("fp32_dw")
                            else 0.0
                        ),
                        "dw_l2_rel_error_vs_fp32_median": statistics.median(dw_errs),
                        "dx_l2_rel_error_vs_fp32_max": max(
                            float(r["dx_l2_rel_error_vs_fp32"]) for r in matched
                        ),
                        "loss_abs_error_vs_fp32_max": max(
                            float(r["loss_abs_error_vs_fp32"]) for r in matched
                        ),
                    }
                )

            unchunked_ms = rows[-1]["median_ms"]
            for r in rows:
                r["throughput_retention_vs_c8192"] = unchunked_ms / r["median_ms"]

            dw_series = [r["dw_l2_rel_error_vs_fp32_median"] for r in rows]
            mode_map[impl] = {
                "rows": rows,
                "dw_rel_error_span_c128_to_c8192": max(dw_series) - min(dw_series),
                "c1024_throughput_retention_vs_c8192": next(
                    r["throughput_retention_vs_c8192"]
                    for r in rows
                    if r["chunk_size"] == 1024
                ),
                "c2048_throughput_retention_vs_c8192": next(
                    r["throughput_retention_vs_c8192"]
                    for r in rows
                    if r["chunk_size"] == 2048
                ),
            }

        by_shape[label] = {
            "shape_bt_h_v": shape,
            "modes": mode_map,
        }
    return by_shape


def summarize_kernel_attribution(reps: list[dict[str, Any]]) -> dict[str, Any]:
    by_shape: dict[str, Any] = {}
    for shape in CHUNK_SHAPES:
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
            ce_us = statistics.median(
                float(r["ce_softmax_cast_us_per_step"]) for r in matched
            )
            krn_cnt = statistics.median(
                float(r["cuda_kernels_per_step"]) for r in matched
            )
            cpu_cnt = statistics.median(
                float(r["cpu_launch_calls_per_step"]) for r in matched
            )
            impl_map[impl] = {
                "total_cuda_kernel_us_per_step": round(tot_us, 3),
                "gemm_us_per_step": round(gemm_us, 3),
                "ce_softmax_cast_us_per_step": round(ce_us, 3),
                "gemm_share": round(gemm_us / tot_us, 6),
                "ce_softmax_cast_share": round(ce_us / tot_us, 6),
                "cuda_kernels_per_step": round(krn_cnt, 2),
                "cpu_launch_calls_per_step": round(cpu_cnt, 2),
                "top_cuda_kernels_rep1": matched[0]["top_cuda_kernels"][:6],
            }

        by_shape[label] = {
            "shape_bt_h_v": shape,
            "implementations": impl_map,
            "ce_kernel_speedup_triton_over_eager": round(
                impl_map["eager_fp32_logits_ce"]["ce_softmax_cast_us_per_step"]
                / impl_map["triton_fused_linear_ce_bf16_dw"][
                    "ce_softmax_cast_us_per_step"
                ],
                4,
            ),
            "checkpointed_gemm_inflation_over_triton": round(
                impl_map["checkpointed_chunked_compile_ce"]["gemm_us_per_step"]
                / impl_map["triton_fused_linear_ce_bf16_dw"]["gemm_us_per_step"],
                4,
            ),
        }
    return by_shape


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.root / args.input_dir
    smoke = load_json(input_dir / "smoke_fused_linear_cross_entropy.json")
    reps = [load_json(input_dir / name) for name in REPETITION_FILES]
    e7_original = load_json(
        args.root / "results/raw/full/transformer_training_full.json"
    )

    scaling = summarize_scaling(reps)
    chunk_frontier = summarize_chunk_frontier(reps)
    kernels = summarize_kernel_attribution(reps)

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
    env_parity = all(
        rep["environment"].get(k) == e7_env.get(k)
        for rep in reps
        for k in env_fields
    )

    all_full_records = [rec for rep in reps for rec in rep["records"]]
    fp32_ce_records = [
        rec
        for rec in all_full_records
        if rec["implementation"] != "eager_bf16_logits_ce"
    ]
    bf16_ce_records = [
        rec
        for rec in all_full_records
        if rec["implementation"] == "eager_bf16_logits_ce"
    ]

    max_fp32_ce_loss_err = max(
        float(r["loss_abs_error_vs_fp32"]) for r in fp32_ce_records
    )
    min_bf16_ce_loss_err = min(
        float(r["loss_abs_error_vs_fp32"]) for r in bf16_ce_records
    )
    max_dx_rel = max(float(r["dx_l2_rel_error_vs_fp32"]) for r in all_full_records)
    max_dw_rel = max(float(r["dw_l2_rel_error_vs_fp32"]) for r in all_full_records)
    max_chain_rel = max(
        float(r["scaled_backward_chain_rule_rel_error"]) for r in all_full_records
    )

    # Analytic memory closure: torch_compile_full_ce materializes [BT, V] BF16 within 2%,
    # and triton_fused_linear_ce_bf16_dw materializes [chunk_size, V] BF16 within 2%.
    compiler_boundary_verified = all(
        abs(
            next(
                i["peak_transient_workspace_mib_median"]
                for i in scaling[lbl]["implementations"]
                if i["implementation"] == "torch_compile_full_ce"
            )
            - scaling[lbl]["theoretical_full_bf16_logits_mib"]
        )
        / scaling[lbl]["theoretical_full_bf16_logits_mib"]
        <= 0.02
        for lbl in scaling
    )
    triton_tile_memory_verified = all(
        abs(
            next(
                i["peak_transient_workspace_mib_median"]
                for i in scaling[lbl]["implementations"]
                if i["implementation"] == "triton_fused_linear_ce_bf16_dw"
            )
            - scaling[lbl]["theoretical_chunk1024_bf16_logits_mib"]
        )
        / scaling[lbl]["theoretical_chunk1024_bf16_logits_mib"]
        <= 0.02
        for lbl in scaling
    )

    # Checkpointed chunking helps latency at H=1024 (>1.10x eager) but regresses below eager (<0.85x) at H=4096,
    # whereas triton_fused_linear_ce_bf16_dw beats eager at every tested shape!
    ckpt_8k_1k_speedup = next(
        i["speedup_over_eager_fp32_median"]
        for i in scaling["8192x1024x128256"]["implementations"]
        if i["implementation"] == "checkpointed_chunked_compile_ce"
    )
    ckpt_8k_4k_speedup = next(
        i["speedup_over_eager_fp32_median"]
        for i in scaling["8192x4096x128256"]["implementations"]
        if i["implementation"] == "checkpointed_chunked_compile_ce"
    )
    triton_all_faster_than_eager = all(
        scaling[lbl]["comparisons"]["triton_bf16_dw_speedup_over_eager_fp32"] > 1.10
        for lbl in scaling
    )
    recompute_crossover_verified = (
        ckpt_8k_1k_speedup > 1.10
        and ckpt_8k_4k_speedup < 0.85
        and triton_all_faster_than_eager
    )

    # FP32 out_dtype dW accumulation is chunk-count invariant (<1e-6 span across 1..64 chunks),
    # while BF16 addmm_ error increases monotonically with chunk count.
    fp32_dw_invariant = all(
        chunk_frontier[lbl]["modes"]["triton_fused_linear_ce_fp32_dw"][
            "dw_rel_error_span_c128_to_c8192"
        ]
        <= 1e-6
        and chunk_frontier[lbl]["modes"]["triton_fused_linear_ce_bf16_dw"][
            "dw_rel_error_span_c128_to_c8192"
        ]
        >= 1e-4
        for lbl in chunk_frontier
    )

    repeat_stability_ok = all(
        item["repeat_spread"] <= 0.03
        for lbl in scaling
        for item in scaling[lbl]["implementations"]
    )
    clocks_unlocked_between_reps = all(
        int(rep["environment"]["initial_telemetry"]["sm_clock_mhz"]) <= 1140
        for rep in reps
    )

    validation = {
        "smoke_present_and_valid": (
            len(smoke["records"]) == 12
            and len(smoke["kernel_attribution_records"]) == 4
            and max(
                float(r["loss_abs_error_vs_fp32"])
                for r in smoke["records"]
                if r["implementation"] != "eager_bf16_logits_ce"
            )
            <= 1e-4
        ),
        "three_full_process_repetitions": (
            len(reps) == 3
            and all(len(rep["records"]) == 58 for rep in reps)
            and all(len(rep["kernel_attribution_records"]) == 8 for rep in reps)
        ),
        "environment_parity_verified": env_parity,
        "correctness_and_chain_rule_gate_passed": (
            max_fp32_ce_loss_err <= 5e-5
            and min_bf16_ce_loss_err >= 1e-2
            and max_dx_rel <= 1e-2
            and max_dw_rel <= 5e-3
            and max_chain_rel <= 5e-3
        ),
        "compiler_mm_boundary_and_triton_tile_memory_verified": (
            compiler_boundary_verified and triton_tile_memory_verified
        ),
        "checkpoint_recompute_crossover_verified": recompute_crossover_verified,
        "fp32_out_dtype_chunk_invariance_verified": fp32_dw_invariant,
        "repeat_stability_and_clock_restore_verified": (
            repeat_stability_ok and clocks_unlocked_between_reps
        ),
    }
    validation["all_gates_passed"] = all(validation.values())
    if not validation["all_gates_passed"]:
        failed = [k for k, v in validation.items() if not v]
        raise RuntimeError(f"E112 validation gates failed: {failed}")

    s_16k_1k = scaling["16384x1024x128256"]
    s_8k_1k = scaling["8192x1024x128256"]
    s_8k_4k = scaling["8192x4096x128256"]

    return {
        "experiment": "E112_fused_linear_cross_entropy",
        "suite": "fused_linear_cross_entropy",
        "hardware": {
            "gpu_name": env0["gpu_name"],
            "compute_capability": env0["compute_capability"],
            "gpu_memory_bytes": env0["gpu_memory_bytes"],
            "cuda_driver": env0["cuda_driver"],
            "torch": env0["torch"],
            "torch_cuda": env0["torch_cuda"],
            "triton": env0["triton"],
            "nvml_max_clocks": env0["nvml_max_clocks"],
            "repetition_count": len(reps),
        },
        "correctness_summary": {
            "fp32_lse_paths_max_loss_abs_error": max_fp32_ce_loss_err,
            "eager_bf16_ce_min_loss_abs_error": min_bf16_ce_loss_err,
            "max_dx_l2_rel_error_vs_fp32": max_dx_rel,
            "max_dw_l2_rel_error_vs_fp32": max_dw_rel,
            "max_scaled_backward_chain_rule_rel_error": max_chain_rel,
        },
        "slm_llm_vocab_scaling": scaling,
        "chunk_size_frontier": chunk_frontier,
        "cuda_kernel_attribution": kernels,
        "key_findings": {
            "slm_16k_transient_memory_reduction_vs_eager_fp32": s_16k_1k[
                "comparisons"
            ]["triton_bf16_dw_transient_reduction_vs_eager_fp32"],
            "slm_16k_total_incremental_memory_reduction_vs_eager_fp32": next(
                i["incremental_memory_reduction_vs_eager_fp32"]
                for i in s_16k_1k["implementations"]
                if i["implementation"] == "triton_fused_linear_ce_bf16_dw"
            ),
            "slm_16k_transient_memory_reduction_vs_torch_compile": s_16k_1k[
                "comparisons"
            ]["triton_bf16_dw_transient_reduction_vs_unchunked_compile"],
            "slm_8k_speedup_over_eager_fp32": s_8k_1k["comparisons"][
                "triton_bf16_dw_speedup_over_eager_fp32"
            ],
            "llm_8k_speedup_over_eager_fp32": s_8k_4k["comparisons"][
                "triton_bf16_dw_speedup_over_eager_fp32"
            ],
            "checkpointed_chunk_speedup_slm_vs_llm": {
                "8192x1024x128256": ckpt_8k_1k_speedup,
                "8192x4096x128256": ckpt_8k_4k_speedup,
            },
            "c1024_throughput_retention_vs_unchunked": {
                "8192x1024x128256": chunk_frontier["8192x1024x128256"]["modes"][
                    "triton_fused_linear_ce_bf16_dw"
                ]["c1024_throughput_retention_vs_c8192"],
                "8192x4096x128256": chunk_frontier["8192x4096x128256"]["modes"][
                    "triton_fused_linear_ce_bf16_dw"
                ]["c1024_throughput_retention_vs_c8192"],
            },
            "c2048_throughput_retention_vs_unchunked": {
                "8192x1024x128256": chunk_frontier["8192x1024x128256"]["modes"][
                    "triton_fused_linear_ce_bf16_dw"
                ]["c2048_throughput_retention_vs_c8192"],
                "8192x4096x128256": chunk_frontier["8192x4096x128256"]["modes"][
                    "triton_fused_linear_ce_bf16_dw"
                ]["c2048_throughput_retention_vs_c8192"],
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
        default=Path("results/raw/fused_linear_cross_entropy"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/raw/fused_linear_cross_entropy/"
            "fused_linear_cross_entropy_full.json"
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
