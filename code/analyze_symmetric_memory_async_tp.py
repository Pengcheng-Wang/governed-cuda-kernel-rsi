from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


REPETITION_FILES = (
    "rep1_symmetric_memory_async_tp.json",
    "rep2_symmetric_memory_async_tp.json",
    "rep3_symmetric_memory_async_tp.json",
)

TP_SIZES = (2, 4, 8)
ALLREDUCE_IMPLS = (
    "nccl_all_reduce",
    "symm_one_shot_staged",
    "symm_one_shot_with_copy",
    "symm_two_shot_staged",
)
BLOCK_IMPLS = (
    "megatron_tp_func_eager",
    "megatron_sp_func_eager",
    "dtensor_sp_naive_eager",
    "dtensor_sp_shared_eager",
    "dtensor_bsh_sp_compiled",
    "dtensor_bsh_async_tp_compiled",
    "dtensor_sbh_sp_compiled",
    "dtensor_sbh_async_tp_compiled",
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


def summarize_deprecation_gate(reps: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tp in TP_SIZES:
        matched = [
            next(r for r in rep["deprecation_gate_records"] if int(r["tp_size"]) == tp)
            for rep in reps
        ]
        noop_ms = [float(r["without_enable_call"]["median_ms"]) for r in matched]
        act_ms = [float(r["with_enable_and_fresh_cache"]["median_ms"]) for r in matched]
        speedups = [n / a for n, a in zip(noop_ms, act_ms, strict=True)]
        r0 = matched[0]
        out[f"tp{tp}"] = {
            "tp_size": tp,
            "global_tokens": int(r0["global_tokens"]),
            "default_is_symm_mem_enabled_all_false": all(
                r["default_is_symm_mem_enabled"] is False for r in matched
            ),
            "future_warning_emitted_all_true": all(
                r["future_warning_emitted"] is True for r in matched
            ),
            "future_warning_text": r0["future_warning_text"],
            "post_enable_is_symm_mem_enabled_all_true": all(
                r["post_enable_is_symm_mem_enabled"] is True for r in matched
            ),
            "fx_cache_poisoned_all_true": all(
                r["same_cache_after_enable_call"]["fx_cache_poisoned"] is True
                for r in matched
            ),
            "without_enable_call": {
                "median_ms": round(statistics.median(noop_ms), 6),
                "repetition_medians_ms": noop_ms,
                "repeat_spread": round(relative_spread(noop_ms), 6),
                "symm_fused_ag_mm_calls": int(r0["without_enable_call"]["symm_fused_ag_mm_calls"]),
                "symm_fused_mm_rs_calls": int(r0["without_enable_call"]["symm_fused_mm_rs_calls"]),
                "c10d_ag_calls": int(r0["without_enable_call"]["c10d_ag_calls"]),
                "c10d_rs_calls": int(r0["without_enable_call"]["c10d_rs_calls"]),
            },
            "same_cache_after_enable_call": r0["same_cache_after_enable_call"],
            "with_enable_and_fresh_cache": {
                "median_ms": round(statistics.median(act_ms), 6),
                "repetition_medians_ms": act_ms,
                "repeat_spread": round(relative_spread(act_ms), 6),
                "symm_fused_ag_mm_calls": int(r0["with_enable_and_fresh_cache"]["symm_fused_ag_mm_calls"]),
                "symm_fused_mm_rs_calls": int(r0["with_enable_and_fresh_cache"]["symm_fused_mm_rs_calls"]),
                "c10d_ag_calls": int(r0["with_enable_and_fresh_cache"]["c10d_ag_calls"]),
                "c10d_rs_calls": int(r0["with_enable_and_fresh_cache"]["c10d_rs_calls"]),
            },
            "speedup_from_enabling_symm_mem_median": round(statistics.median(speedups), 6),
        }
    return out


def summarize_allreduce(reps: list[dict[str, Any]]) -> dict[str, Any]:
    sizes_kib = sorted({int(r["size_kib"]) for r in reps[0]["allreduce_records"]})
    out: dict[str, Any] = {}
    for tp in TP_SIZES:
        tp_rows: dict[str, Any] = {}
        for sz in sizes_kib:
            nccl_reps = [
                float(
                    next(
                        r["median_ms"]
                        for r in rep["allreduce_records"]
                        if int(r["tp_size"]) == tp
                        and int(r["size_kib"]) == sz
                        and r["implementation"] == "nccl_all_reduce"
                    )
                )
                for rep in reps
            ]
            impl_map: dict[str, Any] = {}
            for impl in ALLREDUCE_IMPLS:
                matched = [
                    next(
                        r
                        for r in rep["allreduce_records"]
                        if int(r["tp_size"]) == tp
                        and int(r["size_kib"]) == sz
                        and r["implementation"] == impl
                    )
                    for rep in reps
                ]
                lats_ms = [float(r["median_ms"]) for r in matched]
                bus_bws = [float(r["bus_bw_gbps"]) for r in matched]
                alg_bws = [float(r["alg_bw_gbps"]) for r in matched]
                sp_reps = [n / m for n, m in zip(nccl_reps, lats_ms, strict=True)]
                impl_map[impl] = {
                    "median_us": round(statistics.median(lats_ms) * 1000.0, 3),
                    "median_ms": round(statistics.median(lats_ms), 6),
                    "repetition_medians_us": [round(x * 1000.0, 3) for x in lats_ms],
                    "repeat_spread": round(relative_spread(lats_ms), 6),
                    "speedup_vs_nccl_median": round(statistics.median(sp_reps), 6),
                    "alg_bw_gbps_median": round(statistics.median(alg_bws), 4),
                    "bus_bw_gbps_median": round(statistics.median(bus_bws), 4),
                    "nvlink_read_byte_multiplier": float(matched[0]["nvlink_read_byte_multiplier"]),
                    "max_abs_error_max": max(float(r["max_abs_error"]) for r in matched),
                    "rel_l2_error_max": max(float(r["rel_l2_error"]) for r in matched),
                }
            tp_rows[f"{sz}_kib"] = {
                "size_kib": sz,
                "bytes": sz * 1024,
                "implementations": impl_map,
                "two_shot_over_one_shot_speedup": round(
                    impl_map["symm_one_shot_staged"]["median_ms"]
                    / impl_map["symm_two_shot_staged"]["median_ms"],
                    6,
                ),
            }
        out[f"tp{tp}"] = tp_rows
    return out


def summarize_primitive_overlap(reps: list[dict[str, Any]]) -> dict[str, Any]:
    m_globals = sorted({int(r["m_global"]) for r in reps[0]["primitive_overlap_records"]})
    primitives = ("all_gather_dual_matmul", "matmul_reduce_scatter")
    out: dict[str, Any] = {}
    for tp in TP_SIZES:
        tp_map: dict[str, Any] = {}
        for prim in primitives:
            rows: list[dict[str, Any]] = []
            for m_glob in m_globals:
                matched = [
                    next(
                        r
                        for r in rep["primitive_overlap_records"]
                        if int(r["tp_size"]) == tp
                        and r["primitive"] == prim
                        and int(r["m_global"]) == m_glob
                    )
                    for rep in reps
                ]
                seq_ms = [float(r["nccl_seq_ms"]) for r in matched]
                sym_ms = [float(r["symm_fused_ms"]) for r in matched]
                comm_ms = [float(r["comm_only_ms"]) for r in matched]
                mm_ms = [float(r["mm_only_ms"]) for r in matched]
                sp_reps = [s / f for s, f in zip(seq_ms, sym_ms, strict=True)]
                hid_frac = [float(r["effective_hidden_comm_fraction"]) for r in matched]
                rows.append(
                    {
                        "m_global": m_glob,
                        "m_shard": int(matched[0]["m_shard"]),
                        "comm_only_ms_median": round(statistics.median(comm_ms), 6),
                        "mm_only_ms_median": round(statistics.median(mm_ms), 6),
                        "nccl_seq_ms_median": round(statistics.median(seq_ms), 6),
                        "symm_fused_ms_median": round(statistics.median(sym_ms), 6),
                        "seq_repeat_spread": round(relative_spread(seq_ms), 6),
                        "symm_repeat_spread": round(relative_spread(sym_ms), 6),
                        "speedup_vs_seq_median": round(statistics.median(sp_reps), 6),
                        "comm_fraction_of_seq_median": round(
                            statistics.median(float(r["comm_fraction_of_seq"]) for r in matched),
                            6,
                        ),
                        "effective_hidden_comm_fraction_median": round(
                            statistics.median(hid_frac),
                            6,
                        ),
                        "seq_tflops_per_gpu_median": round(
                            statistics.median(float(r["seq_tflops_per_gpu"]) for r in matched),
                            4,
                        ),
                        "symm_tflops_per_gpu_median": round(
                            statistics.median(float(r["symm_tflops_per_gpu"]) for r in matched),
                            4,
                        ),
                        "max_abs_error_max": max(float(r["max_abs_error"]) for r in matched),
                        "rel_l2_error_max": max(float(r["rel_l2_error"]) for r in matched),
                    }
                )
            tp_map[prim] = rows
        out[f"tp{tp}"] = tp_map
    return out


def summarize_transformer_blocks(reps: list[dict[str, Any]]) -> dict[str, Any]:
    shapes = sorted(
        {(int(r["batch_size"]), int(r["seq_len"])) for r in reps[0]["transformer_block_records"]}
    )
    out: dict[str, Any] = {}
    for tp in TP_SIZES:
        tp_map: dict[str, Any] = {}
        for bsz, seqlen in shapes:
            key = f"{bsz}x{seqlen}"
            tp_eager_reps = [
                float(
                    next(
                        r["median_ms"]
                        for r in rep["transformer_block_records"]
                        if int(r["tp_size"]) == tp
                        and int(r["batch_size"]) == bsz
                        and int(r["seq_len"]) == seqlen
                        and r["implementation"] == "megatron_tp_func_eager"
                    )
                )
                for rep in reps
            ]
            sbh_sp_comp_reps = [
                float(
                    next(
                        r["median_ms"]
                        for r in rep["transformer_block_records"]
                        if int(r["tp_size"]) == tp
                        and int(r["batch_size"]) == bsz
                        and int(r["seq_len"]) == seqlen
                        and r["implementation"] == "dtensor_sbh_sp_compiled"
                    )
                )
                for rep in reps
            ]
            bsh_sp_comp_reps = [
                float(
                    next(
                        r["median_ms"]
                        for r in rep["transformer_block_records"]
                        if int(r["tp_size"]) == tp
                        and int(r["batch_size"]) == bsz
                        and int(r["seq_len"]) == seqlen
                        and r["implementation"] == "dtensor_bsh_sp_compiled"
                    )
                )
                for rep in reps
            ]
            tp_eager_mib = statistics.median(
                float(
                    next(
                        r["incremental_peak_mib"]
                        for r in rep["transformer_block_records"]
                        if int(r["tp_size"]) == tp
                        and int(r["batch_size"]) == bsz
                        and int(r["seq_len"]) == seqlen
                        and r["implementation"] == "megatron_tp_func_eager"
                    )
                )
                for rep in reps
            )

            impl_rows: dict[str, Any] = {}
            for impl in BLOCK_IMPLS:
                matched = [
                    next(
                        r
                        for r in rep["transformer_block_records"]
                        if int(r["tp_size"]) == tp
                        and int(r["batch_size"]) == bsz
                        and int(r["seq_len"]) == seqlen
                        and r["implementation"] == impl
                    )
                    for rep in reps
                ]
                lats = [float(r["median_ms"]) for r in matched]
                toks = [float(r["tokens_per_s"]) for r in matched]
                tflops = [float(r["achieved_tflops_per_gpu"]) for r in matched]
                mibs = [float(r["incremental_peak_mib"]) for r in matched]
                comp_s = [float(r["compile_seconds"]) for r in matched]
                sp_tp = [e / m for e, m in zip(tp_eager_reps, lats, strict=True)]
                sp_sbh = [c / m for c, m in zip(sbh_sp_comp_reps, lats, strict=True)]
                sp_bsh = [c / m for c, m in zip(bsh_sp_comp_reps, lats, strict=True)]
                med_mib = statistics.median(mibs)
                r0 = matched[0]
                impl_rows[impl] = {
                    "sequence_layout": r0["sequence_layout"],
                    "median_ms": round(statistics.median(lats), 6),
                    "repetition_medians_ms": lats,
                    "repeat_spread": round(relative_spread(lats), 6),
                    "tokens_per_s_median": round(statistics.median(toks), 2),
                    "achieved_tflops_per_gpu_median": round(statistics.median(tflops), 4),
                    "incremental_peak_mib_median": round(med_mib, 2),
                    "memory_saving_fraction_vs_tp_eager": round(
                        (tp_eager_mib - med_mib) / max(tp_eager_mib, 1e-6),
                        6,
                    ),
                    "compile_seconds_median": round(statistics.median(comp_s), 4),
                    "symm_fused_ag_mm_calls": int(r0["symm_fused_ag_mm_calls"]),
                    "symm_fused_mm_rs_calls": int(r0["symm_fused_mm_rs_calls"]),
                    "c10d_ag_calls": int(r0["c10d_ag_calls"]),
                    "c10d_rs_calls": int(r0["c10d_rs_calls"]),
                    "cat_split_Copy_kernels": int(r0["cat_split_Copy_kernels"]),
                    "speedup_vs_megatron_tp_eager_median": round(statistics.median(sp_tp), 6),
                    "speedup_vs_sbh_sp_compiled_median": round(statistics.median(sp_sbh), 6),
                    "speedup_vs_bsh_sp_compiled_median": round(statistics.median(sp_bsh), 6),
                    "output_rel_l2_error_max": max(float(r["output_rel_l2_error"]) for r in matched),
                    "dx_rel_l2_error_max": max(float(r["dx_rel_l2_error"]) for r in matched),
                    "max_weight_grad_rel_l2_error_max": max(
                        float(r["max_weight_grad_rel_l2_error"]) for r in matched
                    ),
                    "chain_rule_1p75x_rel_error_max": max(
                        float(r["chain_rule_1p75x_rel_error"]) for r in matched
                    ),
                }

            tp_map[key] = {
                "batch_size": bsz,
                "seq_len": seqlen,
                "global_tokens": bsz * seqlen,
                "tokens_per_rank_sp": (bsz * seqlen) // tp,
                "implementations": impl_rows,
                "comparisons": {
                    "sp_func_vs_tp_func_speedup": round(
                        impl_rows["megatron_tp_func_eager"]["median_ms"]
                        / impl_rows["megatron_sp_func_eager"]["median_ms"],
                        6,
                    ),
                    "dtensor_dispatch_slowdown_shared_vs_func_sp": round(
                        impl_rows["dtensor_sp_shared_eager"]["median_ms"]
                        / impl_rows["megatron_sp_func_eager"]["median_ms"],
                        6,
                    ),
                    "naive_5ag_vs_shared_2ag_dtensor_slowdown": round(
                        impl_rows["dtensor_sp_naive_eager"]["median_ms"]
                        / impl_rows["dtensor_sp_shared_eager"]["median_ms"],
                        6,
                    ),
                    "bsh_compile_speedup_over_dtensor_shared_eager": round(
                        impl_rows["dtensor_sp_shared_eager"]["median_ms"]
                        / impl_rows["dtensor_bsh_sp_compiled"]["median_ms"],
                        6,
                    ),
                    "sbh_vs_bsh_sp_compiled_speedup": round(
                        impl_rows["dtensor_bsh_sp_compiled"]["median_ms"]
                        / impl_rows["dtensor_sbh_sp_compiled"]["median_ms"],
                        6,
                    ),
                    "sbh_vs_bsh_async_tp_speedup": round(
                        impl_rows["dtensor_bsh_async_tp_compiled"]["median_ms"]
                        / impl_rows["dtensor_sbh_async_tp_compiled"]["median_ms"],
                        6,
                    ),
                    "bsh_async_tp_speedup_over_bsh_sp_compiled": impl_rows[
                        "dtensor_bsh_async_tp_compiled"
                    ]["speedup_vs_bsh_sp_compiled_median"],
                    "sbh_async_tp_speedup_over_sbh_sp_compiled": impl_rows[
                        "dtensor_sbh_async_tp_compiled"
                    ]["speedup_vs_sbh_sp_compiled_median"],
                    "sbh_async_tp_speedup_over_megatron_tp_eager": impl_rows[
                        "dtensor_sbh_async_tp_compiled"
                    ]["speedup_vs_megatron_tp_eager_median"],
                    "sbh_sp_compiled_memory_reduction_ratio_vs_tp_eager": round(
                        impl_rows["megatron_tp_func_eager"]["incremental_peak_mib_median"]
                        / max(impl_rows["dtensor_sbh_sp_compiled"]["incremental_peak_mib_median"], 1e-6),
                        6,
                    ),
                },
            }
        out[f"tp{tp}"] = tp_map
    return out


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.root / args.input_dir
    smoke = load_json(input_dir / "smoke_symmetric_memory_async_tp.json")
    reps = [load_json(input_dir / name) for name in REPETITION_FILES]
    e7_original = load_json(args.root / "results/raw/full/transformer_training_full.json")

    dep_summary = summarize_deprecation_gate(reps)
    ar_summary = summarize_allreduce(reps)
    prim_summary = summarize_primitive_overlap(reps)
    block_summary = summarize_transformer_blocks(reps)

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
    pre_clocks = [
        int(entry["pre"]["sm_clock_mhz"])
        for rep in reps
        for tp_key in ("tp2", "tp4", "tp8")
        for entry in rep["telemetry_by_tp"][tp_key]
    ]
    post_clocks = [
        int(entry["post"]["sm_clock_mhz"])
        for rep in reps
        for tp_key in ("tp2", "tp4", "tp8")
        for entry in rep["telemetry_by_tp"][tp_key]
    ]
    post_powers = [
        float(entry["post"]["power_w"])
        for rep in reps
        for tp_key in ("tp2", "tp4", "tp8")
        for entry in rep["telemetry_by_tp"][tp_key]
    ]
    clocks_locked_1410 = (
        all(c == 1410 for c in pre_clocks)
        and min(post_clocks) >= 1300
        and statistics.median(post_clocks) >= 1395
    )

    all_ar = [r for rep in reps for r in rep["allreduce_records"]]
    all_prim = [r for rep in reps for r in rep["primitive_overlap_records"]]
    all_blk = [r for rep in reps for r in rep["transformer_block_records"]]

    max_ar_rel = max(float(r["rel_l2_error"]) for r in all_ar)
    max_prim_rel = max(float(r["rel_l2_error"]) for r in all_prim)
    max_blk_out_rel = max(float(r["output_rel_l2_error"]) for r in all_blk)
    max_blk_dx_rel = max(float(r["dx_rel_l2_error"]) for r in all_blk)
    max_blk_dw_rel = max(float(r["max_weight_grad_rel_l2_error"]) for r in all_blk)
    max_blk_chain_rel = max(float(r["chain_rule_1p75x_rel_error"]) for r in all_blk)

    deprecation_trap_verified = all(
        dep_summary[f"tp{tp}"]["default_is_symm_mem_enabled_all_false"]
        and dep_summary[f"tp{tp}"]["future_warning_emitted_all_true"]
        and dep_summary[f"tp{tp}"]["post_enable_is_symm_mem_enabled_all_true"]
        and dep_summary[f"tp{tp}"]["fx_cache_poisoned_all_true"]
        and dep_summary[f"tp{tp}"]["without_enable_call"]["symm_fused_ag_mm_calls"] == 0
        and dep_summary[f"tp{tp}"]["without_enable_call"]["symm_fused_mm_rs_calls"] == 0
        and dep_summary[f"tp{tp}"]["with_enable_and_fresh_cache"]["symm_fused_ag_mm_calls"] == 4
        and dep_summary[f"tp{tp}"]["with_enable_and_fresh_cache"]["symm_fused_mm_rs_calls"] == 2
        for tp in TP_SIZES
    )

    tp8_ar = ar_summary["tp8"]
    symm_ar_crossover_verified = (
        tp8_ar["64_kib"]["implementations"]["symm_two_shot_staged"]["speedup_vs_nccl_median"] >= 2.0
        and tp8_ar["1024_kib"]["implementations"]["symm_two_shot_staged"]["speedup_vs_nccl_median"] >= 1.8
        and tp8_ar["64_kib"]["implementations"]["symm_one_shot_staged"]["speedup_vs_nccl_median"] >= 1.5
        and tp8_ar["65536_kib"]["implementations"]["symm_one_shot_staged"]["speedup_vs_nccl_median"] <= 0.35
        and tp8_ar["65536_kib"]["two_shot_over_one_shot_speedup"] >= 3.0
    )

    tp8_ag = {r["m_global"]: r for r in prim_summary["tp8"]["all_gather_dual_matmul"]}
    tp8_rs = {r["m_global"]: r for r in prim_summary["tp8"]["matmul_reduce_scatter"]}
    primitive_crossover_verified = (
        tp8_ag[2048]["speedup_vs_seq_median"] < 0.90
        and tp8_rs[2048]["speedup_vs_seq_median"] < 0.90
        and tp8_ag[16384]["speedup_vs_seq_median"] > 1.05
        and tp8_ag[32768]["speedup_vs_seq_median"] > 1.05
        and tp8_rs[32768]["speedup_vs_seq_median"] > 1.05
    )

    tp8_b1 = block_summary["tp8"]["1x4096"]
    tp8_b4 = block_summary["tp8"]["4x4096"]
    tp8_b8 = block_summary["tp8"]["8x4096"]
    sp_mem_and_compile_verified = (
        tp8_b4["implementations"]["dtensor_sbh_sp_compiled"]["memory_saving_fraction_vs_tp_eager"] >= 0.25
        and tp8_b8["implementations"]["dtensor_sbh_sp_compiled"]["memory_saving_fraction_vs_tp_eager"] >= 0.25
        and tp8_b4["comparisons"]["bsh_compile_speedup_over_dtensor_shared_eager"] >= 1.03
        and tp8_b1["implementations"]["dtensor_bsh_async_tp_compiled"]["symm_fused_ag_mm_calls"] == 0
        and tp8_b1["implementations"]["dtensor_sbh_async_tp_compiled"]["symm_fused_ag_mm_calls"] == 4
        and tp8_b4["implementations"]["dtensor_sbh_async_tp_compiled"]["cat_split_Copy_kernels"] == 0
        and tp8_b4["implementations"]["dtensor_bsh_async_tp_compiled"]["cat_split_Copy_kernels"] > 0
    )

    async_tp_large_batch_verified = (
        tp8_b4["comparisons"]["sbh_async_tp_speedup_over_sbh_sp_compiled"] > 1.03
        and tp8_b8["comparisons"]["sbh_async_tp_speedup_over_sbh_sp_compiled"] > 1.03
        and tp8_b8["comparisons"]["sbh_async_tp_speedup_over_megatron_tp_eager"] > 1.15
    )

    validation = {
        "smoke_present_and_valid": (
            smoke["tp_sizes"] == [2, 8]
            and len(smoke["deprecation_gate_records"]) == 2
            and len(smoke["allreduce_records"]) == 16
            and len(smoke["primitive_overlap_records"]) == 8
            and len(smoke["transformer_block_records"]) == 32
        ),
        "three_full_process_repetitions": (
            len(reps) == 3
            and all(rep["tp_sizes"] == [2, 4, 8] for rep in reps)
            and all(len(rep["deprecation_gate_records"]) == 3 for rep in reps)
            and all(len(rep["allreduce_records"]) == 72 for rep in reps)
            and all(len(rep["primitive_overlap_records"]) == 30 for rep in reps)
            and all(len(rep["transformer_block_records"]) == 96 for rep in reps)
        ),
        "environment_and_locked_1410mhz_parity_verified": env_parity and clocks_locked_1410,
        "correctness_and_chain_rule_gate_passed": (
            max_ar_rel <= 0.01
            and max_prim_rel <= 0.01
            and max_blk_out_rel <= 0.015
            and max_blk_dx_rel <= 0.015
            and max_blk_dw_rel <= 0.015
            and max_blk_chain_rel <= 0.01
        ),
        "deprecation_noop_trap_verified": deprecation_trap_verified,
        "symm_allreduce_crossover_verified": symm_ar_crossover_verified,
        "primitive_wave_quantization_crossover_verified": primitive_crossover_verified,
        "sequence_parallel_layout_and_memory_verified": sp_mem_and_compile_verified,
        "async_tp_large_batch_speedup_verified": async_tp_large_batch_verified,
    }
    validation["all_gates_passed"] = all(validation.values())
    if not validation["all_gates_passed"]:
        failed = [k for k, v in validation.items() if not v]
        raise RuntimeError(f"E114 validation gates failed: {failed}")

    return {
        "experiment": "E114_symmetric_memory_async_tp",
        "suite": "symmetric_memory_async_tp",
        "hardware": {
            "gpu_name": env0["gpu_name"],
            "gpu_count": env0["gpu_count"],
            "compute_capability": env0["compute_capability"],
            "gpu_memory_bytes": env0["gpu_memory_bytes"],
            "cuda_driver": env0["cuda_driver"],
            "torch": env0["torch"],
            "torch_cuda": env0["torch_cuda"],
            "triton": env0["triton"],
            "flash_attn": env0["flash_attn"],
            "nvshmem_available": env0["nvshmem_available"],
            "locked_sm_clock_mhz": 1410,
            "all_pre_clocks_1410mhz": all(c == 1410 for c in pre_clocks),
            "post_sm_clock_min_mhz": min(post_clocks),
            "post_sm_clock_median_mhz": int(statistics.median(post_clocks)),
            "max_observed_power_w": round(max(post_powers), 2),
            "repetition_count": len(reps),
        },
        "model_config": reps[0]["model_config"],
        "correctness_summary": {
            "max_allreduce_rel_l2_error": max_ar_rel,
            "max_primitive_rel_l2_error": max_prim_rel,
            "max_block_output_rel_l2_error": max_blk_out_rel,
            "max_block_dx_rel_l2_error": max_blk_dx_rel,
            "max_block_weight_grad_rel_l2_error": max_blk_dw_rel,
            "max_block_chain_rule_1p75x_rel_error": max_blk_chain_rel,
        },
        "deprecation_gate_audit": dep_summary,
        "nvswitch_allreduce_frontier": ar_summary,
        "primitive_overlap_frontier": prim_summary,
        "transformer_block_training": block_summary,
        "key_findings": {
            "tp8_64kib_two_shot_speedup_vs_nccl": tp8_ar["64_kib"]["implementations"][
                "symm_two_shot_staged"
            ]["speedup_vs_nccl_median"],
            "tp8_1mib_two_shot_speedup_vs_nccl": tp8_ar["1024_kib"]["implementations"][
                "symm_two_shot_staged"
            ]["speedup_vs_nccl_median"],
            "tp8_64mib_two_shot_over_one_shot": tp8_ar["65536_kib"][
                "two_shot_over_one_shot_speedup"
            ],
            "tp8_primitive_ag_dual_mm_m2048_vs_m32768": {
                "m2048_speedup": tp8_ag[2048]["speedup_vs_seq_median"],
                "m16384_speedup": tp8_ag[16384]["speedup_vs_seq_median"],
                "m32768_speedup": tp8_ag[32768]["speedup_vs_seq_median"],
            },
            "tp8_primitive_mm_rs_m2048_vs_m32768": {
                "m2048_speedup": tp8_rs[2048]["speedup_vs_seq_median"],
                "m16384_speedup": tp8_rs[16384]["speedup_vs_seq_median"],
                "m32768_speedup": tp8_rs[32768]["speedup_vs_seq_median"],
            },
            "tp8_block_1x4096_comparisons": tp8_b1["comparisons"],
            "tp8_block_4x4096_comparisons": tp8_b4["comparisons"],
            "tp8_block_8x4096_comparisons": tp8_b8["comparisons"],
            "tp8_deprecation_gate_speedup_4x4096": dep_summary["tp8"][
                "speedup_from_enabling_symm_mem_median"
            ],
        },
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw/symmetric_memory_async_tp"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/raw/symmetric_memory_async_tp/symmetric_memory_async_tp_full.json"
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
