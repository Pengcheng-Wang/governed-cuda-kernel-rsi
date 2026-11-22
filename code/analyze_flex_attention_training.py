from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


REPETITION_FILES = (
    "rep1_flex_attention_training.json",
    "rep2_flex_attention_training.json",
    "rep3_flex_attention_training.json",
)

DOC_REGIMES = (
    "single_doc_8192",
    "packed_4docs_long",
    "packed_16docs_mixed",
    "packed_64docs_sft",
)

DOC_IMPLEMENTATIONS = (
    "sdpa_gqa_bool_mask_auto",
    "sdpa_expanded_kv_efficient",
    "flex_score_mod_no_blockmask",
    "flex_blockmask_default",
    "flex_blockmask_max_autotune",
    "flash_attn_varlen_cu_seqlens",
)

SCORE_LAWS = (
    "causal_softcap30",
    "sw512_softcap30",
    "sw512_log_decay_bias",
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


def relative_spread(values: list[float]) -> float:
    med = statistics.median(values)
    if med == 0.0:
        return 0.0
    return (max(values) - min(values)) / med


def find_doc_record(
    payload: dict[str, Any],
    regime: str,
    implementation: str,
) -> dict[str, Any]:
    return next(
        rec
        for rec in payload["records"]
        if rec["experiment"] == "packed_document_gqa_training"
        and rec["regime"] == regime
        and rec["implementation"] == implementation
    )


def find_cbm_record(
    payload: dict[str, Any],
    regime: str,
) -> dict[str, Any]:
    return next(
        rec
        for rec in payload["blockmask_construction_records"]
        if rec["regime"] == regime
    )


def find_score_record(
    payload: dict[str, Any],
    score_law: str,
    implementation: str,
) -> dict[str, Any]:
    return next(
        rec
        for rec in payload["records"]
        if rec["experiment"] == "score_modified_gqa_training"
        and rec["score_law"] == score_law
        and rec["implementation"] == implementation
    )


def summarize_packed_document_gqa(reps: list[dict[str, Any]]) -> dict[str, Any]:
    by_regime: dict[str, Any] = {}
    for regime in DOC_REGIMES:
        cbm_rows = [find_cbm_record(rep, regime) for rep in reps]
        eager_cbm_ms = statistics.median(
            float(r["eager_create_block_mask_ms"]) for r in cbm_rows
        )
        comp_cbm_ms = statistics.median(
            float(r["compiled_create_block_mask_ms"]) for r in cbm_rows
        )

        auto_rows = [
            find_doc_record(rep, regime, "sdpa_gqa_bool_mask_auto") for rep in reps
        ]
        eff_rows = [
            find_doc_record(rep, regime, "sdpa_expanded_kv_efficient") for rep in reps
        ]
        auto_ms_reps = [float(r["median_ms"]) for r in auto_rows]
        eff_ms_reps = [float(r["median_ms"]) for r in eff_rows]
        auto_incr_mib = statistics.median(
            float(r["peak_incremental_memory_bytes"]) / (1024.0**2)
            for r in auto_rows
        )

        impl_rows: list[dict[str, Any]] = []
        lookup: dict[str, dict[str, Any]] = {}
        for impl in DOC_IMPLEMENTATIONS:
            matched = [find_doc_record(rep, regime, impl) for rep in reps]
            latencies = [float(r["median_ms"]) for r in matched]
            tflops = [float(r["effective_valid_attn_tflop_s"]) for r in matched]
            tok_s = [float(r["tokens_per_second"]) for r in matched]
            incr_mib = statistics.median(
                float(r["peak_incremental_memory_bytes"]) / (1024.0**2)
                for r in matched
            )
            trans_mib = statistics.median(
                float(r["peak_transient_workspace_bytes"]) / (1024.0**2)
                for r in matched
            )
            setup_s = statistics.median(float(r["setup_seconds"]) for r in matched)
            sp_vs_auto = [
                a / m for a, m in zip(auto_ms_reps, latencies, strict=True)
            ]
            sp_vs_eff = [
                e / m for e, m in zip(eff_ms_reps, latencies, strict=True)
            ]

            entry = {
                "implementation": impl,
                "median_ms": statistics.median(latencies),
                "repetition_medians_ms": latencies,
                "repeat_spread": relative_spread(latencies),
                "speedup_over_sdpa_auto_median": statistics.median(sp_vs_auto),
                "speedup_over_sdpa_efficient_expanded_median": statistics.median(
                    sp_vs_eff
                ),
                "effective_valid_attn_tflop_s_median": statistics.median(tflops),
                "tokens_per_second_median": statistics.median(tok_s),
                "peak_incremental_memory_mib_median": incr_mib,
                "peak_transient_workspace_mib_median": trans_mib,
                "incremental_memory_reduction_vs_sdpa_auto": auto_incr_mib
                / max(incr_mib, 1e-9),
                "setup_seconds_median": setup_s,
                "out_l2_rel_error_vs_fp32_max": max(
                    float(r["out_l2_rel_error_vs_fp32"]) for r in matched
                ),
                "dq_l2_rel_error_vs_fp32_max": max(
                    float(r["dq_l2_rel_error_vs_fp32"]) for r in matched
                ),
                "dk_l2_rel_error_vs_fp32_max": max(
                    float(r["dk_l2_rel_error_vs_fp32"]) for r in matched
                ),
                "dv_l2_rel_error_vs_fp32_max": max(
                    float(r["dv_l2_rel_error_vs_fp32"]) for r in matched
                ),
                "scaled_backward_chain_rule_rel_error_max": max(
                    float(r["scaled_backward_chain_rule_rel_error"]) for r in matched
                ),
            }
            impl_rows.append(entry)
            lookup[impl] = entry

        flex_def_ms = lookup["flex_blockmask_default"]["median_ms"]
        varlen_ms = lookup["flash_attn_varlen_cu_seqlens"]["median_ms"]
        nobm_ms = lookup["flex_score_mod_no_blockmask"]["median_ms"]

        by_regime[regime] = {
            "shape_b_s_hq_hkv_d": auto_rows[0]["shape_b_s_hq_hkv_d"],
            "num_documents": int(auto_rows[0]["num_documents"]),
            "mean_doc_len": float(auto_rows[0]["mean_doc_len"]),
            "max_doc_len": int(auto_rows[0]["max_doc_len"]),
            "active_pair_fraction": float(auto_rows[0]["active_pair_fraction"]),
            "blockmask_sparsity_percent": float(
                auto_rows[0]["blockmask_sparsity_percent"]
            ),
            "blockmask_active_tile_fraction": float(
                auto_rows[0]["blockmask_active_tile_fraction"]
            ),
            "subtile_valid_density": float(auto_rows[0]["subtile_valid_density"]),
            "blockmask_construction": {
                "eager_ms": eager_cbm_ms,
                "compiled_ms": comp_cbm_ms,
                "compile_speedup": eager_cbm_ms / comp_cbm_ms,
                "compiled_amortized_per_layer_32L_ms": comp_cbm_ms / 32.0,
                "compiled_32L_amortized_fraction_of_flex_step": (comp_cbm_ms / 32.0)
                / flex_def_ms,
            },
            "implementations": impl_rows,
            "comparisons": {
                "blockmask_speedup_over_no_blockmask": nobm_ms / flex_def_ms,
                "flex_default_speedup_over_sdpa_auto": lookup[
                    "flex_blockmask_default"
                ]["speedup_over_sdpa_auto_median"],
                "flex_default_speedup_over_sdpa_efficient": lookup[
                    "flex_blockmask_default"
                ]["speedup_over_sdpa_efficient_expanded_median"],
                "flash_varlen_speedup_over_flex_default": flex_def_ms / varlen_ms,
                "flex_autotune_over_flex_default_speedup": flex_def_ms
                / lookup["flex_blockmask_max_autotune"]["median_ms"],
            },
        }
    return by_regime


def summarize_score_modified_gqa(reps: list[dict[str, Any]]) -> dict[str, Any]:
    by_law: dict[str, Any] = {}
    for law in SCORE_LAWS:
        law_impls = [
            rec["implementation"]
            for rec in reps[0]["records"]
            if rec["experiment"] == "score_modified_gqa_training"
            and rec["score_law"] == law
        ]
        eager_rows = [
            find_score_record(rep, law, "eager_materialized_score_mod") for rep in reps
        ]
        eager_ms_reps = [float(r["median_ms"]) for r in eager_rows]
        eager_incr_mib = statistics.median(
            float(r["peak_incremental_memory_bytes"]) / (1024.0**2)
            for r in eager_rows
        )

        impl_rows: list[dict[str, Any]] = []
        lookup: dict[str, dict[str, Any]] = {}
        for impl in law_impls:
            matched = [find_score_record(rep, law, impl) for rep in reps]
            latencies = [float(r["median_ms"]) for r in matched]
            tflops = [float(r["effective_valid_attn_tflop_s"]) for r in matched]
            incr_mib = statistics.median(
                float(r["peak_incremental_memory_bytes"]) / (1024.0**2)
                for r in matched
            )
            trans_mib = statistics.median(
                float(r["peak_transient_workspace_bytes"]) / (1024.0**2)
                for r in matched
            )
            setup_s = statistics.median(float(r["setup_seconds"]) for r in matched)
            sp_vs_eager = [
                e / m for e, m in zip(eager_ms_reps, latencies, strict=True)
            ]

            entry = {
                "implementation": impl,
                "median_ms": statistics.median(latencies),
                "repetition_medians_ms": latencies,
                "repeat_spread": relative_spread(latencies),
                "speedup_over_eager_materialized_median": statistics.median(
                    sp_vs_eager
                ),
                "effective_valid_attn_tflop_s_median": statistics.median(tflops),
                "peak_incremental_memory_mib_median": incr_mib,
                "peak_transient_workspace_mib_median": trans_mib,
                "incremental_memory_reduction_vs_eager": eager_incr_mib
                / max(incr_mib, 1e-9),
                "setup_seconds_median": setup_s,
                "out_l2_rel_error_vs_fp32_max": max(
                    float(r["out_l2_rel_error_vs_fp32"]) for r in matched
                ),
                "dq_l2_rel_error_vs_fp32_max": max(
                    float(r["dq_l2_rel_error_vs_fp32"]) for r in matched
                ),
                "dk_l2_rel_error_vs_fp32_max": max(
                    float(r["dk_l2_rel_error_vs_fp32"]) for r in matched
                ),
                "dv_l2_rel_error_vs_fp32_max": max(
                    float(r["dv_l2_rel_error_vs_fp32"]) for r in matched
                ),
                "scaled_backward_chain_rule_rel_error_max": max(
                    float(r["scaled_backward_chain_rule_rel_error"]) for r in matched
                ),
            }
            impl_rows.append(entry)
            lookup[impl] = entry

        comp_mat_mib = lookup["compiled_materialized_score_mod"][
            "peak_incremental_memory_mib_median"
        ]
        flex_def_mib = lookup["flex_attention_default"][
            "peak_incremental_memory_mib_median"
        ]
        flex_def_ms = lookup["flex_attention_default"]["median_ms"]

        comp_mat_trans_mib = lookup["compiled_materialized_score_mod"][
            "peak_transient_workspace_mib_median"
        ]
        eager_trans_mib = lookup["eager_materialized_score_mod"][
            "peak_transient_workspace_mib_median"
        ]
        flex_def_trans_mib = lookup["flex_attention_default"][
            "peak_transient_workspace_mib_median"
        ]
        comparisons: dict[str, Any] = {
            "flex_default_speedup_over_eager_materialized": lookup[
                "flex_attention_default"
            ]["speedup_over_eager_materialized_median"],
            "flex_default_speedup_over_compiled_materialized": lookup[
                "compiled_materialized_score_mod"
            ]["median_ms"]
            / flex_def_ms,
            "flex_autotune_speedup_over_compiled_materialized": lookup[
                "compiled_materialized_score_mod"
            ]["median_ms"]
            / lookup["flex_attention_max_autotune"]["median_ms"],
            "flex_default_memory_reduction_vs_eager_materialized": eager_incr_mib
            / flex_def_mib,
            "flex_default_transient_reduction_vs_eager_materialized": eager_trans_mib
            / flex_def_trans_mib,
            "flex_default_memory_reduction_vs_compiled_materialized": comp_mat_mib
            / flex_def_mib,
            "flex_default_transient_reduction_vs_compiled_materialized": (
                comp_mat_trans_mib / flex_def_trans_mib
            ),
            "online_fp32_dk_accuracy_gain_vs_materialized_bf16": lookup[
                "eager_materialized_score_mod"
            ]["dk_l2_rel_error_vs_fp32_max"]
            / max(lookup["flex_attention_default"]["dk_l2_rel_error_vs_fp32_max"], 1e-12),
        }
        if "flash_attn_native_cuda" in lookup:
            fa_ms = lookup["flash_attn_native_cuda"]["median_ms"]
            comparisons["flash_native_speedup_over_flex_default"] = (
                flex_def_ms / fa_ms
            )
            comparisons["flash_native_speedup_over_flex_autotune"] = (
                lookup["flex_attention_max_autotune"]["median_ms"] / fa_ms
            )

        by_law[law] = {
            "shape_b_s_hq_hkv_d": eager_rows[0]["shape_b_s_hq_hkv_d"],
            "flash_attn_native_supported": bool(
                eager_rows[0]["flash_attn_native_supported"]
            ),
            "blockmask_sparsity_percent": float(
                eager_rows[0]["blockmask_sparsity_percent"]
            ),
            "active_pair_fraction": float(eager_rows[0]["active_pair_fraction"]),
            "implementations": impl_rows,
            "comparisons": comparisons,
        }
    return by_law


def summarize_kernels(reps: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = [
        (r["workload"], r["implementation"])
        for r in reps[0]["kernel_attribution_records"]
    ]
    for workload, impl in keys:
        matched = [
            next(
                r
                for r in rep["kernel_attribution_records"]
                if r["workload"] == workload and r["implementation"] == impl
            )
            for rep in reps
        ]
        tot_us = statistics.median(
            float(r["total_cuda_kernel_us_per_step"]) for r in matched
        )
        krn_cnt = statistics.median(
            float(r["cuda_kernels_per_step"]) for r in matched
        )
        cats = sorted(
            {cat for r in matched for cat in r["category_us_per_step"].keys()}
        )
        cat_med = {
            cat: round(
                statistics.median(
                    float(r["category_us_per_step"].get(cat, 0.0)) for r in matched
                ),
                3,
            )
            for cat in cats
        }
        out.setdefault(workload, {})[impl] = {
            "total_cuda_kernel_us_per_step": round(tot_us, 3),
            "cuda_kernels_per_step": round(krn_cnt, 2),
            "category_us_per_step": cat_med,
            "top_cuda_kernels_rep1": matched[0]["top_cuda_kernels"][:5],
        }
    return out


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.root / args.input_dir
    smoke = load_json(input_dir / "smoke_flex_attention_training.json")
    reps = [load_json(input_dir / name) for name in REPETITION_FILES]
    e7_original = load_json(
        args.root / "results/raw/full/transformer_training_full.json"
    )

    packed_doc = summarize_packed_document_gqa(reps)
    score_mod = summarize_score_modified_gqa(reps)
    kernels = summarize_kernels(reps)

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
    max_out_rel = max(float(r["out_l2_rel_error_vs_fp32"]) for r in all_full_records)
    max_dq_rel = max(float(r["dq_l2_rel_error_vs_fp32"]) for r in all_full_records)
    max_dk_rel = max(float(r["dk_l2_rel_error_vs_fp32"]) for r in all_full_records)
    max_dv_rel = max(float(r["dv_l2_rel_error_vs_fp32"]) for r in all_full_records)
    max_chain_rel = max(
        float(r["scaled_backward_chain_rule_rel_error"]) for r in all_full_records
    )

    # Gate 1: SDPA auto with enable_gqa=True + bool attn_mask silently falls back to >30 GiB math path
    sdpa_auto_math_fallback = all(
        next(
            i["peak_incremental_memory_mib_median"]
            for i in packed_doc[reg]["implementations"]
            if i["implementation"] == "sdpa_gqa_bool_mask_auto"
        )
        >= 30000.0
        for reg in DOC_REGIMES
    )

    # Gate 2: BlockMask accelerates FlexAttention dramatically as doc packing sparsity rises,
    # while dense SDPA does invariant O(S^2) work
    sdpa_eff_1doc = next(
        i["median_ms"]
        for i in packed_doc["single_doc_8192"]["implementations"]
        if i["implementation"] == "sdpa_expanded_kv_efficient"
    )
    sdpa_eff_64doc = next(
        i["median_ms"]
        for i in packed_doc["packed_64docs_sft"]["implementations"]
        if i["implementation"] == "sdpa_expanded_kv_efficient"
    )
    flex_1doc = next(
        i["median_ms"]
        for i in packed_doc["single_doc_8192"]["implementations"]
        if i["implementation"] == "flex_blockmask_default"
    )
    flex_16doc = next(
        i["median_ms"]
        for i in packed_doc["packed_16docs_mixed"]["implementations"]
        if i["implementation"] == "flex_blockmask_default"
    )
    blockmask_sparsity_verified = (
        abs(sdpa_eff_64doc - sdpa_eff_1doc) / sdpa_eff_1doc <= 0.08
        and (flex_1doc / flex_16doc) >= 3.0
        and packed_doc["packed_16docs_mixed"]["comparisons"][
            "blockmask_speedup_over_no_blockmask"
        ]
        >= 5.0
    )

    # Gate 3: torch.compile on materialized score_mod still holds >8 GiB O(S^2) memory vs <=800 MiB O(S) FlexAttention
    mat_s2_wall_verified = all(
        next(
            i["peak_incremental_memory_mib_median"]
            for i in score_mod[law]["implementations"]
            if i["implementation"] == "compiled_materialized_score_mod"
        )
        >= 8000.0
        and next(
            i["peak_incremental_memory_mib_median"]
            for i in score_mod[law]["implementations"]
            if i["implementation"] == "flex_attention_default"
        )
        <= 800.0
        for law in SCORE_LAWS
    )

    # Gate 4: Custom log-decay bias runs fused only in FlexAttention (>4.8x faster & >11x less memory than compiled materialized)
    log_law = score_mod["sw512_log_decay_bias"]
    custom_score_mod_verified = (
        log_law["flash_attn_native_supported"] is False
        and log_law["comparisons"][
            "flex_default_speedup_over_compiled_materialized"
        ]
        >= 4.8
        and log_law["comparisons"][
            "flex_default_memory_reduction_vs_compiled_materialized"
        ]
        >= 11.0
    )

    clocks_restored = all(
        int(rep["environment"]["initial_telemetry"]["sm_clock_mhz"]) <= 1140
        for rep in reps
    )

    validation = {
        "smoke_present_and_valid": (
            len(smoke["records"]) == 26
            and len(smoke["blockmask_construction_records"]) == 2
            and len(smoke["kernel_attribution_records"]) == 8
        ),
        "three_full_process_repetitions": (
            len(reps) == 3
            and all(len(rep["records"]) == 38 for rep in reps)
            and all(len(rep["blockmask_construction_records"]) == 4 for rep in reps)
            and all(len(rep["kernel_attribution_records"]) == 8 for rep in reps)
        ),
        "environment_parity_verified": env_parity,
        "correctness_and_chain_rule_gate_passed": (
            max_out_rel <= 0.015
            and max_dq_rel <= 0.015
            and max_dk_rel <= 0.015
            and max_dv_rel <= 0.015
            and max_chain_rel <= 0.01
        ),
        "sdpa_gqa_mask_silent_math_fallback_verified": sdpa_auto_math_fallback,
        "blockmask_sparsity_speedup_verified": blockmask_sparsity_verified,
        "compiled_materialized_s2_memory_wall_verified": mat_s2_wall_verified,
        "custom_log_decay_flex_only_fused_verified": custom_score_mod_verified,
        "clock_restore_verified": clocks_restored,
    }
    validation["all_gates_passed"] = all(validation.values())
    if not validation["all_gates_passed"]:
        failed = [k for k, v in validation.items() if not v]
        raise RuntimeError(f"E113 validation gates failed: {failed}")

    return {
        "experiment": "E113_flex_attention_training",
        "suite": "flex_attention_training",
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
            "max_out_l2_rel_error_vs_fp32": max_out_rel,
            "max_dq_l2_rel_error_vs_fp32": max_dq_rel,
            "max_dk_l2_rel_error_vs_fp32": max_dk_rel,
            "max_dv_l2_rel_error_vs_fp32": max_dv_rel,
            "max_scaled_backward_chain_rule_rel_error": max_chain_rel,
        },
        "packed_document_gqa": packed_doc,
        "score_modified_gqa": score_mod,
        "cuda_kernel_attribution": kernels,
        "key_findings": {
            "sdpa_gqa_bool_mask_silent_fallback_memory_mib": next(
                i["peak_incremental_memory_mib_median"]
                for i in packed_doc["packed_16docs_mixed"]["implementations"]
                if i["implementation"] == "sdpa_gqa_bool_mask_auto"
            ),
            "packed_16docs_flex_speedups": packed_doc["packed_16docs_mixed"][
                "comparisons"
            ],
            "packed_64docs_sft_subtile_crossover": packed_doc["packed_64docs_sft"][
                "comparisons"
            ],
            "compiled_create_block_mask_16docs_ms": packed_doc["packed_16docs_mixed"][
                "blockmask_construction"
            ]["compiled_ms"],
            "create_block_mask_compile_speedup_16docs": packed_doc[
                "packed_16docs_mixed"
            ]["blockmask_construction"]["compile_speedup"],
            "sw512_softcap30_comparisons": score_mod["sw512_softcap30"][
                "comparisons"
            ],
            "sw512_log_decay_flex_only_comparisons": score_mod[
                "sw512_log_decay_bias"
            ]["comparisons"],
        },
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw/flex_attention_training"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/raw/flex_attention_training/flex_attention_training_full.json"
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
