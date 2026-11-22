from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


REPETITION_FILES = (
    "rep1_context_parallel_attention.json",
    "rep2_context_parallel_attention.json",
    "rep3_context_parallel_attention.json",
)

CP_SIZES = (2, 4, 8)


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


def summarize_single_gpu_baselines(reps: list[dict[str, Any]]) -> dict[str, Any]:
    attn_seqs = sorted({int(r["seq_len"]) for r in reps[0]["runs"]["cp8"]["single_gpu_attention"]})
    attn_map: dict[str, Any] = {}
    for seq_len in attn_seqs:
        matched = [
            next(
                r
                for r in rep["runs"]["cp8"]["single_gpu_attention"]
                if int(r["seq_len"]) == seq_len
            )
            for rep in reps
        ]
        lats = [float(r["median_ms"]) for r in matched]
        tfs = [float(r["cluster_tflops"]) for r in matched]
        mibs = [float(r["peak_incremental_mib"]) for r in matched]
        attn_map[str(seq_len)] = {
            "seq_len": seq_len,
            "median_ms": round(statistics.median(lats), 4),
            "repetition_medians_ms": lats,
            "repeat_spread": round(relative_spread(lats), 6),
            "tflops_median": round(statistics.median(tfs), 2),
            "peak_incremental_mib_median": round(statistics.median(mibs), 2),
        }

    blk_seqs = sorted({int(r["seq_len"]) for r in reps[0]["runs"]["cp8"]["single_gpu_block"]})
    blk_map: dict[str, Any] = {}
    for seq_len in blk_seqs:
        matched = [
            next(
                r
                for r in rep["runs"]["cp8"]["single_gpu_block"]
                if int(r["seq_len"]) == seq_len
            )
            for rep in reps
        ]
        lats = [float(r["median_ms"]) for r in matched]
        tfs = [float(r["cluster_tflops"]) for r in matched]
        mibs = [float(r["peak_incremental_mib"]) for r in matched]
        gibs = [float(r["projected_32layer_activation_gib"]) for r in matched]
        blk_map[str(seq_len)] = {
            "seq_len": seq_len,
            "median_ms": round(statistics.median(lats), 4),
            "repetition_medians_ms": lats,
            "repeat_spread": round(relative_spread(lats), 6),
            "tflops_median": round(statistics.median(tfs), 2),
            "peak_incremental_mib_median": round(statistics.median(mibs), 2),
            "projected_32layer_activation_gib_median": round(statistics.median(gibs), 2),
            "attn_flop_fraction": float(matched[0]["attn_flop_fraction"]),
        }

    return {"attention": attn_map, "llama3_block": blk_map}


def summarize_suite_0_audit(reps: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cp in CP_SIZES:
        cp_key = f"cp{cp}"
        s0_reps = [rep["runs"][cp_key]["suite_0_pytorch_cp_audit"] for rep in reps]
        modes = [r["mode"] for r in s0_reps[0]["records"]]
        seq_len = int(s0_reps[0]["seq_len"])

        ag_lats = [
            float(next(r["median_ms"] for r in s0["records"] if r["mode"] == "pytorch_cp_allgather"))
            for s0 in s0_reps
        ]
        ag_mibs = [
            float(
                next(
                    r["peak_incremental_mib"]
                    for r in s0["records"]
                    if r["mode"] == "pytorch_cp_allgather"
                )
            )
            for s0 in s0_reps
        ]
        ag_mib_med = statistics.median(ag_mibs)

        mode_summary: dict[str, Any] = {}
        for mode in modes:
            matched = [
                next(r for r in s0["records"] if r["mode"] == mode) for s0 in s0_reps
            ]
            lats = [float(r["median_ms"]) for r in matched]
            tfs = [float(r["cluster_tflops"]) for r in matched]
            mibs = [float(r["peak_incremental_mib"]) for r in matched]
            sp_vs_ag = [a / m for a, m in zip(ag_lats, lats, strict=True)]
            r0 = matched[0]
            entry: dict[str, Any] = {
                "median_ms": round(statistics.median(lats), 4),
                "repetition_medians_ms": lats,
                "repeat_spread": round(relative_spread(lats), 6),
                "cluster_tflops_median": round(statistics.median(tfs), 2),
                "peak_incremental_mib_median": round(statistics.median(mibs), 2),
                "speedup_vs_pytorch_cp_allgather_median": round(statistics.median(sp_vs_ag), 6),
                "memory_ratio_vs_pytorch_cp_allgather": round(
                    ag_mib_med / max(statistics.median(mibs), 1e-6), 6
                ),
                "out_rel_l2_max": max(float(r["out_rel_l2"]) for r in matched),
                "dq_rel_l2_max": max(float(r["dq_rel_l2"]) for r in matched),
                "dk_rel_l2_max": max(float(r["dk_rel_l2"]) for r in matched),
                "dv_rel_l2_max": max(float(r["dv_rel_l2"]) for r in matched),
                "chain_rule_err_max": max(float(r["chain_rule_err"]) for r in matched),
            }
            if "allgather_calls" in r0:
                entry["allgather_calls"] = int(r0["allgather_calls"])
                entry["alltoall_calls"] = int(r0["alltoall_calls"])
                entry["comm_input_elems"] = int(r0["comm_input_elems"])
            mode_summary[mode] = entry

        out[cp_key] = {
            "cp_size": cp,
            "seq_len": seq_len,
            "default_rotate_method": s0_reps[0]["default_rotate_method"],
            "empty_buffers_raises_index_error_all_true": all(
                bool(s0["empty_buffers_raises_index_error"]) for s0 in s0_reps
            ),
            "empty_buffers_error": s0_reps[0]["empty_buffers_error"],
            "triton_over_eager_lse_speedup": round(
                mode_summary["ring_zigzag_eager_lse"]["median_ms"]
                / mode_summary["ring_zigzag_triton_lse"]["median_ms"],
                6,
            ),
            "modes": mode_summary,
        }
    return out


def summarize_suite_a_scaling(
    reps: list[dict[str, Any]], single_gpu_attn: dict[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cp in CP_SIZES:
        cp_key = f"cp{cp}"
        sa_reps = [rep["runs"][cp_key]["suite_a_sequence_scaling"] for rep in reps]
        seqs = sorted({int(r["seq_len"]) for r in sa_reps[0]})
        seq_map: dict[str, Any] = {}
        for seq_len in seqs:
            modes = [r["mode"] for r in sa_reps[0] if int(r["seq_len"]) == seq_len]
            one_gpu_ms = float(single_gpu_attn[str(seq_len)]["median_ms"])
            contig_lats = [
                float(
                    next(
                        r["median_ms"]
                        for r in sa
                        if int(r["seq_len"]) == seq_len and r["mode"] == "ring_contiguous_overlap"
                    )
                )
                for sa in sa_reps
            ]

            mode_rows: dict[str, Any] = {}
            for mode in modes:
                matched = [
                    next(
                        r
                        for r in sa
                        if int(r["seq_len"]) == seq_len and r["mode"] == mode
                    )
                    for sa in sa_reps
                ]
                lats = [float(r["median_ms"]) for r in matched]
                rmean = [float(r["rank_mean_median_ms"]) for r in matched]
                tfs = [float(r["cluster_tflops"]) for r in matched]
                ptfs = [float(r["per_gpu_tflops"]) for r in matched]
                mibs = [float(r["peak_incremental_mib"]) for r in matched]
                med_ms = statistics.median(lats)
                sp_vs_1gpu = one_gpu_ms / med_ms
                sp_vs_contig = [c / m for c, m in zip(contig_lats, lats, strict=True)]
                r0 = matched[0]
                mode_rows[mode] = {
                    "ulysses_degree": int(r0["ulysses_degree"]),
                    "ring_degree": int(r0["ring_degree"]),
                    "s_local": int(r0["s_local"]),
                    "median_ms": round(med_ms, 4),
                    "rank_mean_median_ms": round(statistics.median(rmean), 4),
                    "repetition_medians_ms": lats,
                    "repeat_spread": round(relative_spread(lats), 6),
                    "cluster_tflops_median": round(statistics.median(tfs), 2),
                    "per_gpu_tflops_median": round(statistics.median(ptfs), 2),
                    "peak_incremental_mib_median": round(statistics.median(mibs), 2),
                    "speedup_vs_single_gpu": round(sp_vs_1gpu, 4),
                    "strong_scaling_efficiency": round(sp_vs_1gpu / cp, 4),
                    "speedup_vs_contiguous_ring_median": round(statistics.median(sp_vs_contig), 4),
                    "out_rel_l2_max": max(float(r["out_rel_l2"]) for r in matched),
                    "dq_rel_l2_max": max(float(r["dq_rel_l2"]) for r in matched),
                    "dk_rel_l2_max": max(float(r["dk_rel_l2"]) for r in matched),
                    "dv_rel_l2_max": max(float(r["dv_rel_l2"]) for r in matched),
                    "chain_rule_err_max": max(float(r["chain_rule_err"]) for r in matched),
                    "ref_checked": bool(r0["ref_checked"]),
                }

            seq_map[str(seq_len)] = {
                "seq_len": seq_len,
                "single_gpu_ms": one_gpu_ms,
                "theoretical_contiguous_imbalance_ratio": round((2 * cp - 1) / cp, 4),
                "measured_zigzag_seq_vs_contiguous_speedup": round(
                    mode_rows["ring_contiguous_overlap"]["median_ms"]
                    / mode_rows["ring_zigzag_seq"]["median_ms"],
                    4,
                ),
                "measured_zigzag_overlap_vs_seq_speedup": round(
                    mode_rows["ring_zigzag_seq"]["median_ms"]
                    / mode_rows["ring_zigzag_overlap"]["median_ms"],
                    4,
                ),
                "measured_zigzag_overlap_vs_contiguous_speedup": round(
                    mode_rows["ring_contiguous_overlap"]["median_ms"]
                    / mode_rows["ring_zigzag_overlap"]["median_ms"],
                    4,
                ),
                "gqa_native_vs_kv_expanded_ulysses_speedup": round(
                    mode_rows["ulysses_kv_expanded"]["median_ms"]
                    / mode_rows["ulysses_gqa_native"]["median_ms"],
                    4,
                ),
                "ulysses_gqa_vs_zigzag_overlap_speedup": round(
                    mode_rows["ring_zigzag_overlap"]["median_ms"]
                    / mode_rows["ulysses_gqa_native"]["median_ms"],
                    4,
                ),
                "modes": mode_rows,
            }
        out[cp_key] = seq_map
    return out


def summarize_suite_b_sparse_gqa(reps: list[dict[str, Any]]) -> dict[str, Any]:
    sb_reps = [rep["runs"]["cp8"]["suite_b_sparse_gqa_wall"] for rep in reps]
    seqs = sorted({int(r["seq_len"]) for r in sb_reps[0]})
    kv_heads = sorted({int(r["num_kv_heads"]) for r in sb_reps[0]}, reverse=True)
    out: dict[str, Any] = {}
    for seq_len in seqs:
        kv_map: dict[str, Any] = {}
        for h_kv in kv_heads:
            modes = [
                r["mode"]
                for r in sb_reps[0]
                if int(r["seq_len"]) == seq_len and int(r["num_kv_heads"]) == h_kv
            ]
            m_rows: dict[str, Any] = {}
            for mode in modes:
                matched = [
                    next(
                        r
                        for r in sb
                        if int(r["seq_len"]) == seq_len
                        and int(r["num_kv_heads"]) == h_kv
                        and r["mode"] == mode
                    )
                    for sb in sb_reps
                ]
                lats = [float(r["median_ms"]) for r in matched]
                tfs = [float(r["cluster_tflops"]) for r in matched]
                mibs = [float(r["peak_incremental_mib"]) for r in matched]
                r0 = matched[0]
                m_rows[mode] = {
                    "ulysses_degree": int(r0["ulysses_degree"]),
                    "ring_degree": int(r0["ring_degree"]),
                    "kv_replication_factor": int(r0["kv_replication_factor"]),
                    "median_ms": round(statistics.median(lats), 4),
                    "repetition_medians_ms": lats,
                    "repeat_spread": round(relative_spread(lats), 6),
                    "cluster_tflops_median": round(statistics.median(tfs), 2),
                    "peak_incremental_mib_median": round(statistics.median(mibs), 2),
                    "out_rel_l2_max": max(float(r["out_rel_l2"]) for r in matched),
                    "dq_rel_l2_max": max(float(r["dq_rel_l2"]) for r in matched),
                    "dk_rel_l2_max": max(float(r["dk_rel_l2"]) for r in matched),
                    "dv_rel_l2_max": max(float(r["dv_rel_l2"]) for r in matched),
                }
            kv_map[f"hkv_{h_kv}"] = {
                "num_kv_heads": h_kv,
                "gqa_group_size": 32 // h_kv,
                "modes": m_rows,
            }
        out[str(seq_len)] = kv_map
    return out


def summarize_suite_c_block(
    reps: list[dict[str, Any]], single_gpu_blk: dict[str, Any]
) -> dict[str, Any]:
    sc_reps = [rep["runs"]["cp8"]["suite_c_llama3_block"] for rep in reps]
    seqs = sorted({int(r["seq_len"]) for r in sc_reps[0]})
    out: dict[str, Any] = {}
    for seq_len in seqs:
        modes = [r["mode"] for r in sc_reps[0] if int(r["seq_len"]) == seq_len]
        one_gpu_entry = single_gpu_blk.get(str(seq_len))
        one_gpu_ms = float(one_gpu_entry["median_ms"]) if one_gpu_entry else None
        one_gpu_mib = float(one_gpu_entry["peak_incremental_mib_median"]) if one_gpu_entry else None

        m_rows: dict[str, Any] = {}
        for mode in modes:
            matched = [
                next(
                    r for r in sc if int(r["seq_len"]) == seq_len and r["mode"] == mode
                )
                for sc in sc_reps
            ]
            pure_lats = [float(r["pure_cp_median_ms"]) for r in matched]
            sync_lats = [float(r["with_dw_sync_median_ms"]) for r in matched]
            ovh_lats = [float(r["dw_sync_overhead_ms"]) for r in matched]
            pure_tfs = [float(r["cluster_tflops_pure_cp"]) for r in matched]
            sync_tfs = [float(r["cluster_tflops_with_dw_sync"]) for r in matched]
            toks = [float(r["tokens_per_sec_with_dw_sync"]) for r in matched]
            mibs = [float(r["peak_incremental_mib"]) for r in matched]
            gibs = [float(r["projected_32layer_activation_gib"]) for r in matched]
            pure_med = statistics.median(pure_lats)
            sync_med = statistics.median(sync_lats)
            mib_med = statistics.median(mibs)
            r0 = matched[0]
            m_rows[mode] = {
                "ulysses_degree": int(r0["ulysses_degree"]),
                "ring_degree": int(r0["ring_degree"]),
                "s_local": int(r0["s_local"]),
                "pure_cp_median_ms": round(pure_med, 4),
                "with_dw_sync_median_ms": round(sync_med, 4),
                "dw_sync_overhead_ms_median": round(statistics.median(ovh_lats), 4),
                "pure_repeat_spread": round(relative_spread(pure_lats), 6),
                "sync_repeat_spread": round(relative_spread(sync_lats), 6),
                "cluster_tflops_pure_cp_median": round(statistics.median(pure_tfs), 2),
                "cluster_tflops_with_dw_sync_median": round(statistics.median(sync_tfs), 2),
                "tokens_per_sec_with_dw_sync_median": round(statistics.median(toks), 1),
                "peak_incremental_mib_median": round(mib_med, 2),
                "projected_32layer_activation_gib_median": round(statistics.median(gibs), 2),
                "speedup_vs_single_gpu_pure": (
                    round(one_gpu_ms / pure_med, 4) if one_gpu_ms is not None else None
                ),
                "speedup_vs_single_gpu_with_dw_sync": (
                    round(one_gpu_ms / sync_med, 4) if one_gpu_ms is not None else None
                ),
                "memory_reduction_vs_single_gpu": (
                    round(one_gpu_mib / max(mib_med, 1e-6), 4) if one_gpu_mib is not None else None
                ),
                "out_rel_l2_max": max(float(r["out_rel_l2"]) for r in matched),
                "dx_rel_l2_max": max(float(r["dx_rel_l2"]) for r in matched),
                "dw_max_rel_l2_max": max(float(r["dw_max_rel_l2"]) for r in matched),
                "chain_rule_err_max": max(float(r["chain_rule_err"]) for r in matched),
                "ref_checked": bool(r0["ref_checked"]),
            }

        out[str(seq_len)] = {
            "seq_len": seq_len,
            "attn_flop_fraction": float(sc_reps[0][0]["attn_flop_fraction"])
            if seq_len == int(sc_reps[0][0]["seq_len"])
            else float(
                next(r["attn_flop_fraction"] for r in sc_reps[0] if int(r["seq_len"]) == seq_len)
            ),
            "single_gpu_ms": one_gpu_ms,
            "single_gpu_projected_32layer_gib": (
                float(one_gpu_entry["projected_32layer_activation_gib_median"])
                if one_gpu_entry
                else None
            ),
            "zigzag_over_contiguous_block_speedup_pure": round(
                m_rows["ring_contiguous_block"]["pure_cp_median_ms"]
                / m_rows["ring_zigzag_block"]["pure_cp_median_ms"],
                4,
            ),
            "ulysses_over_pytorch_cp_block_speedup_pure": round(
                m_rows["pytorch_cp_allgather_block"]["pure_cp_median_ms"]
                / m_rows["ulysses_gqa_block"]["pure_cp_median_ms"],
                4,
            ),
            "modes": m_rows,
        }
    return out


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.root / args.input_dir
    smoke = load_json(input_dir / "smoke_context_parallel_attention.json")
    reps = [load_json(input_dir / name) for name in REPETITION_FILES]
    e7_original = load_json(args.root / "results/raw/full/transformer_training_full.json")

    single_gpu = summarize_single_gpu_baselines(reps)
    suite_0 = summarize_suite_0_audit(reps)
    suite_a = summarize_suite_a_scaling(reps, single_gpu["attention"])
    suite_b = summarize_suite_b_sparse_gqa(reps)
    suite_c = summarize_suite_c_block(reps, single_gpu["llama3_block"])

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
        int(entry["sm_clock_mhz"])
        for rep in reps
        for cp_key in ("cp2", "cp4", "cp8")
        for entry in rep["runs"][cp_key]["telemetry_pre"]
    ]
    post_clocks = [
        int(entry["sm_clock_mhz"])
        for rep in reps
        for cp_key in ("cp2", "cp4", "cp8")
        for entry in rep["runs"][cp_key]["telemetry_post"]
    ]
    post_powers = [
        float(entry["power_w"])
        for rep in reps
        for cp_key in ("cp2", "cp4", "cp8")
        for entry in rep["runs"][cp_key]["telemetry_post"]
    ]
    clocks_locked_1410 = (
        all(c == 1410 for c in pre_clocks)
        and min(post_clocks) >= 1275
        and statistics.median(post_clocks) >= 1380
    )

    s0_all = [
        r
        for rep in reps
        for cp_key in ("cp2", "cp4", "cp8")
        for r in rep["runs"][cp_key]["suite_0_pytorch_cp_audit"]["records"]
    ]
    sa_all = [
        r
        for rep in reps
        for cp_key in ("cp2", "cp4", "cp8")
        for r in rep["runs"][cp_key]["suite_a_sequence_scaling"]
    ]
    sb_all = [
        r
        for rep in reps
        for r in rep["runs"]["cp8"]["suite_b_sparse_gqa_wall"]
    ]
    sc_all = [
        r
        for rep in reps
        for r in rep["runs"]["cp8"]["suite_c_llama3_block"]
    ]

    max_attn_out_err = max(float(r["out_rel_l2"]) for r in s0_all + sa_all + sb_all)
    max_attn_dq_err = max(float(r["dq_rel_l2"]) for r in s0_all + sa_all + sb_all)
    max_attn_dk_err = max(float(r["dk_rel_l2"]) for r in s0_all + sa_all + sb_all)
    max_attn_dv_err = max(float(r["dv_rel_l2"]) for r in s0_all + sa_all + sb_all)
    max_attn_chain_err = max(float(r["chain_rule_err"]) for r in s0_all + sa_all)
    max_blk_out_err = max(float(r["out_rel_l2"]) for r in sc_all)
    max_blk_dx_err = max(float(r["dx_rel_l2"]) for r in sc_all)
    max_blk_dw_err = max(float(r["dw_max_rel_l2"]) for r in sc_all)
    max_blk_chain_err = max(float(r["chain_rule_err"]) for r in sc_all)

    s0_cp8 = suite_0["cp8"]
    pytorch_cp_audit_verified = (
        all(
            suite_0[f"cp{cp}"]["default_rotate_method"] == "ALL_GATHER"
            and suite_0[f"cp{cp}"]["empty_buffers_raises_index_error_all_true"]
            and suite_0[f"cp{cp}"]["triton_over_eager_lse_speedup"] > 1.015
            for cp in CP_SIZES
        )
        and (
            suite_0["cp2"]["triton_over_eager_lse_speedup"]
            < suite_0["cp4"]["triton_over_eager_lse_speedup"]
            < suite_0["cp8"]["triton_over_eager_lse_speedup"]
        )
        and suite_0["cp8"]["triton_over_eager_lse_speedup"] >= 1.07
        and s0_cp8["modes"]["ring_zigzag_triton_lse"]["speedup_vs_pytorch_cp_allgather_median"] >= 1.15
        and s0_cp8["modes"]["ulysses_gqa_native"]["speedup_vs_pytorch_cp_allgather_median"] >= 1.35
        and s0_cp8["modes"]["ulysses_gqa_native"]["memory_ratio_vs_pytorch_cp_allgather"] >= 1.50
    )

    sa_cp8 = suite_a["cp8"]
    zigzag_and_overlap_verified = (
        sa_cp8["65536"]["measured_zigzag_seq_vs_contiguous_speedup"] >= 1.50
        and sa_cp8["131072"]["measured_zigzag_seq_vs_contiguous_speedup"] >= 1.60
        and sa_cp8["16384"]["measured_zigzag_overlap_vs_seq_speedup"] >= 1.15
        and sa_cp8["32768"]["measured_zigzag_overlap_vs_seq_speedup"] >= 1.15
    )

    ulysses_and_usp_verified = (
        all(
            sa_cp8[str(s)]["gqa_native_vs_kv_expanded_ulysses_speedup"] > 1.01
            for s in (8192, 16384, 32768, 65536, 131072)
        )
        and (
            sa_cp8["8192"]["modes"]["ring_zigzag_overlap"]["median_ms"]
            / sa_cp8["8192"]["modes"]["usp_u4_r2"]["median_ms"]
        )
        >= 2.0
        and sa_cp8["131072"]["modes"]["usp_u4_r2"]["speedup_vs_single_gpu"] >= 7.30
        and sa_cp8["131072"]["modes"]["ulysses_gqa_native"]["speedup_vs_single_gpu"] >= 7.50
        and sa_cp8["131072"]["modes"]["ulysses_gqa_native"]["strong_scaling_efficiency"] >= 0.93
    )

    sb_64k = suite_b["65536"]
    ring_mibs_64k = [
        sb_64k[f"hkv_{h}"]["modes"]["ring_zigzag_u1_r8"]["peak_incremental_mib_median"]
        for h in (8, 4, 2, 1)
    ]
    ring_lats_64k = [
        sb_64k[f"hkv_{h}"]["modes"]["ring_zigzag_u1_r8"]["median_ms"]
        for h in (8, 4, 2, 1)
    ]
    sparse_gqa_verified = (
        ring_mibs_64k[0] > ring_mibs_64k[1] > ring_mibs_64k[2] > ring_mibs_64k[3]
        and ring_lats_64k[0] > ring_lats_64k[3]
        and ring_mibs_64k[3]
        < sb_64k["hkv_1"]["modes"]["ulysses_u8_kv_replicated"]["peak_incremental_mib_median"]
        and all(
            sb_64k[f"hkv_{h}"]["modes"]["usp_matched_norep"]["kv_replication_factor"] == 1
            for h in (8, 4, 2, 1)
        )
    )

    sc_64k = suite_c["65536"]
    sc_128k = suite_c["131072"]
    block_verified = (
        sc_64k["single_gpu_projected_32layer_gib"] >= 350.0
        and sc_64k["modes"]["ulysses_gqa_block"]["projected_32layer_activation_gib_median"] <= 45.0
        and sc_64k["modes"]["ulysses_gqa_block"]["speedup_vs_single_gpu_pure"] >= 7.0
        and sc_128k["zigzag_over_contiguous_block_speedup_pure"] >= 1.50
        and sc_128k["modes"]["ulysses_gqa_block"]["cluster_tflops_pure_cp_median"] >= 1550.0
        and sc_128k["modes"]["usp_u4_r2_block"]["cluster_tflops_pure_cp_median"] >= 1500.0
    )

    validation = {
        "smoke_present_and_valid": (
            set(smoke["runs"].keys()) == {"cp2", "cp8"}
            and len(smoke["runs"]["cp8"]["suite_a_sequence_scaling"]) == 14
            and len(smoke["runs"]["cp8"]["suite_b_sparse_gqa_wall"]) == 12
            and len(smoke["runs"]["cp8"]["suite_c_llama3_block"]) == 10
        ),
        "three_full_process_repetitions": (
            len(reps) == 3
            and all(set(rep["runs"].keys()) == {"cp2", "cp4", "cp8"} for rep in reps)
            and all(len(rep["runs"]["cp8"]["suite_a_sequence_scaling"]) == 35 for rep in reps)
            and all(len(rep["runs"]["cp8"]["suite_b_sparse_gqa_wall"]) == 24 for rep in reps)
            and all(len(rep["runs"]["cp8"]["suite_c_llama3_block"]) == 20 for rep in reps)
        ),
        "environment_and_locked_1410mhz_parity_verified": env_parity and clocks_locked_1410,
        "correctness_and_chain_rule_gate_passed": (
            max_attn_out_err <= 0.005
            and max_attn_dq_err <= 0.005
            and max_attn_dk_err <= 0.008
            and max_attn_dv_err <= 0.008
            and max_attn_chain_err <= 0.01
            and max_blk_out_err <= 0.006
            and max_blk_dx_err <= 0.006
            and max_blk_dw_err <= 0.012
            and max_blk_chain_err <= 0.012
        ),
        "pytorch_cp_audit_verified": pytorch_cp_audit_verified,
        "zigzag_load_balance_and_overlap_verified": zigzag_and_overlap_verified,
        "ulysses_and_usp_strong_scaling_verified": ulysses_and_usp_verified,
        "sparse_gqa_head_wall_verified": sparse_gqa_verified,
        "llama3_block_long_context_verified": block_verified,
    }
    validation["all_gates_passed"] = all(validation.values())
    if not validation["all_gates_passed"]:
        failed = [k for k, v in validation.items() if not v]
        raise RuntimeError(f"E115 validation gates failed: {failed}")

    return {
        "experiment": "E115_context_parallel_long_context_attention",
        "suite": "context_parallel_attention",
        "hardware": {
            "gpu_name": env0["gpu_name"],
            "gpu_count": env0["gpu_count"],
            "compute_capability": env0["compute_capability"],
            "cuda_driver": env0["cuda_driver"],
            "torch": env0["torch"],
            "torch_cuda": env0["torch_cuda"],
            "triton": env0["triton"],
            "flash_attn": env0["flash_attn"],
            "pre_sm_clock_all_mhz": sorted(set(pre_clocks)),
            "post_sm_clock_min_mhz": min(post_clocks),
            "post_sm_clock_median_mhz": int(statistics.median(post_clocks)),
            "post_power_max_w": round(max(post_powers), 2),
        },
        "model_geometry": reps[0]["model_geometry"],
        "max_errors": {
            "attn_out_rel_l2_max": round(max_attn_out_err, 6),
            "attn_dq_rel_l2_max": round(max_attn_dq_err, 6),
            "attn_dk_rel_l2_max": round(max_attn_dk_err, 6),
            "attn_dv_rel_l2_max": round(max_attn_dv_err, 6),
            "attn_chain_rule_max": round(max_attn_chain_err, 6),
            "block_out_rel_l2_max": round(max_blk_out_err, 6),
            "block_dx_rel_l2_max": round(max_blk_dx_err, 6),
            "block_dw_rel_l2_max": round(max_blk_dw_err, 6),
            "block_chain_rule_max": round(max_blk_chain_err, 6),
        },
        "single_gpu_baselines": single_gpu,
        "suite_0_pytorch_cp_audit": suite_0,
        "suite_a_sequence_scaling": suite_a,
        "suite_b_sparse_gqa_wall": suite_b,
        "suite_c_llama3_block": suite_c,
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and validate E115 Context Parallelism 3-repetition results."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw/context_parallel_attention"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raw/context_parallel_attention/context_parallel_attention_full.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    summary_payload = run_analysis(cli_args)
    out_path = cli_args.root / cli_args.output
    write_json(out_path, summary_payload)
    print(f"Wrote {out_path} (all_gates_passed={summary_payload['validation']['all_gates_passed']})")
