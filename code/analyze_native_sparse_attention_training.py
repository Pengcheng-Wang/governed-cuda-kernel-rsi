#!/usr/bin/env python3
"""Deterministic 3-repetition analyzer and 9-gate validator for E125:
Natively Trainable Dynamic Sparse Attention (DeepSeek NSA & Moonshot Kimi MoBA),
100% Lock-Free Split-Reduce CSC Backward, Zero-Selection-Recompute NSA-SAC, &
8-GPU NVSwitch Group-Preserving 2D USP-NSA Context Parallelism (CP = 8, S = 16K..131K).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _med(vals: list[float]) -> float:
    return float(statistics.median(vals))


def _iqr(vals: list[float]) -> float:
    ss = sorted(vals)
    q25 = ss[max(0, (len(ss) - 1) // 4)]
    q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
    return float(q75 - q25)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic 3-repetition analyzer and 9-gate validator for E125"
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.root / "results" / "raw" / "native_sparse_attention_training"
    smoke = json.loads(
        (raw_dir / "smoke_native_sparse_attention_training.json").read_text(encoding="utf-8")
    )
    reps = [
        json.loads(
            (raw_dir / f"rep{i}_native_sparse_attention_training.json").read_text(encoding="utf-8")
        )
        for i in (1, 2, 3)
    ]

    envs = [
        (
            r["environment"].get("torch"),
            r["environment"].get("triton"),
            r["environment"].get("flash_attn"),
            r["environment"].get("gpu_name"),
            r["world_size"],
        )
        for r in reps
    ]
    smoke_and_env_ok = (
        smoke.get("mode") == "smoke"
        and smoke.get("world_size") == 8
        and len(set(envs)) == 1
        and reps[0]["world_size"] == 8
    )

    init_clocks = [float(r.get("initial_sm_clock_mhz", 0.0)) for r in reps]
    all_clocks: list[float] = []
    all_powers: list[float] = []

    def _harvest_telemetry(d: dict[str, Any]) -> None:
        clk = float(d.get("sm_clock_mhz", -1.0))
        pwr = float(d.get("power_w", -1.0))
        if clk > 0.0:
            all_clocks.append(clk)
        if pwr > 0.0:
            all_powers.append(pwr)

    for r in reps:
        for mk, mv in r["suite_0_audits"]["sm80_gqa_tile_saturation_roofline"]["modes"].items():
            if isinstance(mv, dict):
                _harvest_telemetry(mv)
        for sv in r["suite_0_audits"]["sink_heavy_backward_atomic_vs_csc_ablation"].values():
            for mk in ("query_parallel_fp32_atomic", "unsplit_csc_sink_straggler", "compacted_worktile_csc_split256"):
                _harvest_telemetry(sv[mk])
        for sv in r["suite_a_1gpu_sparse_attention_scaling"].values():
            for mk in (
                "dense_fa2_gqa",
                "flex_attn_block128_dynamic",
                "moba_per_head_eager_scatter",
                "moba_gqa_group_triton_lse",
                "deepseek_nsa_3branch_tok1_csc",
                "deepseek_nsa_3branch_sub4_csc",
            ):
                if isinstance(sv.get(mk), dict) and "median_ms" in sv[mk]:
                    _harvest_telemetry(sv[mk])
        for sv in r["suite_b_4layer_nsa_sac"].values():
            for mk in ("dense_fa2_no_ac", "nsa_no_ac", "nsa_zero_sel_recompute_sac", "nsa_full_layer_ac"):
                _harvest_telemetry(sv[mk])
        for sv in r["suite_c_cp8_usp_nsa_8gpu"]["scaling_cp8"].values():
            for mk in (
                "cp8_ulysses8_replicated_kv_dense_fa2",
                "cp8_seq_sharded_kv_demand_allgather_nsa",
                "cp8_ulysses8_replicated_kv_nsa",
                "cp8_usp_4x2_group_preserving_nsa",
            ):
                _harvest_telemetry(sv[mk])

    med_clock = _med(all_clocks) if all_clocks else 0.0
    min_clock = min(all_clocks) if all_clocks else 0.0
    max_clock = max(all_clocks) if all_clocks else 0.0
    max_power = max(all_powers) if all_powers else 0.0
    clocks_locked_1410 = (
        all(c >= 1380.0 for c in init_clocks)
        and max_clock >= 1410.0
        and med_clock >= 1380.0
        and min_clock >= 1260.0
    )

    # Suite 0 Aggregation
    s0_list = [r["suite_0_audits"] for r in reps]
    flex_ref = s0_list[0]["pytorch_flex_attention_triple_defect_audit"]
    flex_agg = {
        "defect1_uncompiled_create_block_mask_hq64_s16k": {
            "oom_triggered_all_reps": all(
                bool(s["pytorch_flex_attention_triple_defect_audit"]["defect1_uncompiled_create_block_mask_hq64_s16k"]["oom_triggered"])
                for s in s0_list
            ),
            "requested_gib": float(flex_ref["defect1_uncompiled_create_block_mask_hq64_s16k"]["requested_gib"]),
            "error_excerpt": flex_ref["defect1_uncompiled_create_block_mask_hq64_s16k"]["error_excerpt"],
        },
        "defect2_from_kv_blocks_transpose_ordered": {
            "bug_triggered_all_reps": all(
                bool(s["pytorch_flex_attention_triple_defect_audit"]["defect2_from_kv_blocks_transpose_ordered"]["bug_triggered"])
                for s in s0_list
            ),
            "root_cause": flex_ref["defect2_from_kv_blocks_transpose_ordered"]["root_cause"],
            "error_excerpt": flex_ref["defect2_from_kv_blocks_transpose_ordered"]["error_excerpt"],
        },
        "defect3_block_size_64_inductor_lowering": {
            "rejected_all_reps": all(
                bool(s["pytorch_flex_attention_triple_defect_audit"]["defect3_block_size_64_inductor_lowering"]["rejected"])
                for s in s0_list
            ),
            "error_excerpt": flex_ref["defect3_block_size_64_inductor_lowering"]["error_excerpt"],
        },
        "defect4_s64k_gqa64_bwd_illegal_memory_access": {
            "illegal_address_triggered_all_reps": all(
                bool(s["pytorch_flex_attention_triple_defect_audit"]["defect4_s64k_gqa64_bwd_illegal_memory_access"]["illegal_address_triggered"])
                for s in s0_list
            ),
            "returncode": int(flex_ref["defect4_s64k_gqa64_bwd_illegal_memory_access"]["returncode"]),
            "error_excerpt": flex_ref["defect4_s64k_gqa64_bwd_illegal_memory_access"]["error_excerpt"],
        },
    }

    tile_ref = s0_list[0]["sm80_gqa_tile_saturation_roofline"]
    quest_ms = _med(
        [float(s["sm80_gqa_tile_saturation_roofline"]["modes"]["quest_per_head_m1_64q"]["median_ms"]) for s in s0_list]
    )
    tok1_32q_ms = _med(
        [float(s["sm80_gqa_tile_saturation_roofline"]["modes"]["gqa_h4_tok1_padded_32q8kv"]["median_ms"]) for s in s0_list]
    )
    tile_modes_agg: dict[str, Any] = {}
    for mk in (
        "quest_per_head_m1_64q",
        "gqa_h4_tok1_padded_32q8kv",
        "nsa_h16_tok1_64q4kv",
        "nsa_h16_sub4_64q4kv",
        "nsa_h4_sub16_stride16_32q8kv",
    ):
        ms_v = [float(s["sm80_gqa_tile_saturation_roofline"]["modes"][mk]["median_ms"]) for s in s0_list]
        tf_v = [float(s["sm80_gqa_tile_saturation_roofline"]["modes"][mk]["achieved_sparse_tflops"]) for s in s0_list]
        m_med = _med(ms_v)
        tile_modes_agg[mk] = {
            "ms_median": round(m_med, 3),
            "ms_iqr": round(_iqr(ms_v), 3),
            "achieved_sparse_tflops_median": round(_med(tf_v), 2),
            "mma_active_rows": int(tile_ref["modes"][mk]["mma_active_rows"]),
        }
    tile_agg = {
        "seqlen": int(tile_ref["seqlen"]),
        "modes": tile_modes_agg,
        "sub4_64q_speedup_over_quest_m1": round(
            quest_ms / max(tile_modes_agg["nsa_h16_sub4_64q4kv"]["ms_median"], 1e-9), 3
        ),
        "tok1_64q_speedup_over_quest_m1": round(
            quest_ms / max(tile_modes_agg["nsa_h16_tok1_64q4kv"]["ms_median"], 1e-9), 3
        ),
        "sub16_32q_speedup_over_tok1_padded_32q": round(
            tok1_32q_ms / max(tile_modes_agg["nsa_h4_sub16_stride16_32q8kv"]["ms_median"], 1e-9), 3
        ),
    }

    csc_agg: dict[str, Any] = {}
    for sk in s0_list[0]["sink_heavy_backward_atomic_vs_csc_ablation"]:
        at_ms = [
            float(s["sink_heavy_backward_atomic_vs_csc_ablation"][sk]["query_parallel_fp32_atomic"]["median_ms"])
            for s in s0_list
        ]
        u0_ms = [
            float(s["sink_heavy_backward_atomic_vs_csc_ablation"][sk]["unsplit_csc_sink_straggler"]["median_ms"])
            for s in s0_list
        ]
        cs_ms = [
            float(s["sink_heavy_backward_atomic_vs_csc_ablation"][sk]["compacted_worktile_csc_split256"]["median_ms"])
            for s in s0_list
        ]
        at_med, u0_med, cs_med = _med(at_ms), _med(u0_ms), _med(cs_ms)
        csc_agg[sk] = {
            "query_parallel_fp32_atomic_ms_median": round(at_med, 3),
            "query_parallel_fp32_atomic_ms_iqr": round(_iqr(at_ms), 3),
            "unsplit_csc_sink_straggler_ms_median": round(u0_med, 3),
            "unsplit_csc_sink_straggler_ms_iqr": round(_iqr(u0_ms), 3),
            "compacted_worktile_csc_split256_ms_median": round(cs_med, 3),
            "compacted_worktile_csc_split256_ms_iqr": round(_iqr(cs_ms), 3),
            "speedup_compacted_csc_over_atomic": round(at_med / max(cs_med, 1e-9), 3),
            "speedup_compacted_csc_over_unsplit_csc": round(u0_med / max(cs_med, 1e-9), 3),
            "dk_rel_l2_csc_vs_atomic_max": round(
                max(float(s["sink_heavy_backward_atomic_vs_csc_ablation"][sk]["dk_rel_l2_csc_vs_atomic"]) for s in s0_list), 6
            ),
            "dv_rel_l2_csc_vs_atomic_max": round(
                max(float(s["sink_heavy_backward_atomic_vs_csc_ablation"][sk]["dv_rel_l2_csc_vs_atomic"]) for s in s0_list), 6
            ),
            "dq_rel_l2_csc_vs_unsplit_max": round(
                max(float(s["sink_heavy_backward_atomic_vs_csc_ablation"][sk]["dq_rel_l2_csc_vs_unsplit"]) for s in s0_list), 6
            ),
        }

    ora_keys = list(s0_list[0]["bf16_vs_fp64_oracle_parity_s512"].keys())
    oracle_agg = {
        ok: round(max(float(s["bf16_vs_fp64_oracle_parity_s512"][ok]) for s in s0_list), 6)
        for ok in ora_keys
    }
    rich_keys = list(s0_list[0]["fp64_richardson_central_difference"]["ratios"].keys())
    rich_ratios = {
        rk: round(min(float(s["fp64_richardson_central_difference"]["ratios"][rk]) for s in s0_list), 4)
        for rk in rich_keys
    }
    rich_agg = {
        "ratios_min_across_reps": rich_ratios,
        "min_ratio": round(min(rich_ratios.values()), 4),
        "max_ratio": round(
            max(float(s["fp64_richardson_central_difference"]["max_ratio"]) for s in s0_list), 4
        ),
        "nsa_dv_win_linear_fd_abs_err_max": max(
            float(s["fp64_richardson_central_difference"]["nsa_dv_win_linear_fd_abs_err"]) for s in s0_list
        ),
    }

    suite_0_summary = {
        "pytorch_flex_attention_quadruple_defect_audit": flex_agg,
        "sm80_gqa_tile_saturation_roofline": tile_agg,
        "sink_heavy_backward_atomic_vs_csc_ablation": csc_agg,
        "bf16_vs_fp64_oracle_parity_s512_max": oracle_agg,
        "fp64_richardson_central_difference": rich_agg,
    }

    # Suite A Aggregation
    sa_list = [r["suite_a_1gpu_sparse_attention_scaling"] for r in reps]
    sa_agg: dict[str, Any] = {}
    for sk, sref in sa_list[0].items():
        fa2_med = _med([float(a[sk]["dense_fa2_gqa"]["median_ms"]) for a in sa_list])
        row_a: dict[str, Any] = {"seqlen": int(sref["seqlen"])}
        for mk in (
            "dense_fa2_gqa",
            "moba_per_head_eager_scatter",
            "moba_gqa_group_triton_lse",
            "deepseek_nsa_3branch_tok1_csc",
            "deepseek_nsa_3branch_sub4_csc",
        ):
            ms_v = [float(a[sk][mk]["median_ms"]) for a in sa_list]
            fwd_v = [float(a[sk][mk]["fwd_ms"]) for a in sa_list]
            bwd_v = [float(a[sk][mk]["bwd_ms"]) for a in sa_list]
            mem_v = [float(a[sk][mk]["inc_peak_mib"]) for a in sa_list]
            m_med = _med(ms_v)
            row_a[mk] = {
                "ms_median": round(m_med, 3),
                "ms_iqr": round(_iqr(ms_v), 3),
                "fwd_ms_median": round(_med(fwd_v), 3),
                "bwd_ms_median": round(_med(bwd_v), 3),
                "inc_peak_mib_median": round(_med(mem_v), 2),
                "speedup_over_dense_fa2": round(fa2_med / max(m_med, 1e-9), 3),
            }
        fx_ref = sref["flex_attn_block128_dynamic"]
        if fx_ref.get("status") == "ok":
            fx_ms = [float(a[sk]["flex_attn_block128_dynamic"]["median_ms"]) for a in sa_list]
            fx_mem = [float(a[sk]["flex_attn_block128_dynamic"]["inc_peak_mib"]) for a in sa_list]
            fx_med = _med(fx_ms)
            row_a["flex_attn_block128_dynamic"] = {
                "status": "ok",
                "ms_median": round(fx_med, 3),
                "ms_iqr": round(_iqr(fx_ms), 3),
                "inc_peak_mib_median": round(_med(fx_mem), 2),
                "speedup_over_dense_fa2": round(fa2_med / max(fx_med, 1e-9), 3),
            }
        else:
            row_a["flex_attn_block128_dynamic"] = dict(fx_ref)

        row_a["moba_gqa_speedup_over_per_head"] = round(
            row_a["moba_per_head_eager_scatter"]["ms_median"]
            / max(row_a["moba_gqa_group_triton_lse"]["ms_median"], 1e-9),
            3,
        )
        row_a["nsa_sub4_speedup_over_tok1"] = round(
            row_a["deepseek_nsa_3branch_tok1_csc"]["ms_median"]
            / max(row_a["deepseek_nsa_3branch_sub4_csc"]["ms_median"], 1e-9),
            3,
        )
        sa_agg[sk] = row_a

    # Suite B Aggregation
    sb_list = [r["suite_b_4layer_nsa_sac"] for r in reps]
    sb_agg: dict[str, Any] = {}
    for sk, sref in sb_list[0].items():
        row_b: dict[str, Any] = {
            "seqlen": int(sref["seqlen"]),
            "nsa_sac_dw_rel_l2_vs_no_ac_max": round(
                max(float(b[sk]["nsa_sac_dw_rel_l2_vs_no_ac"]) for b in sb_list), 8
            ),
            "nsa_sac_dx_rel_l2_vs_no_ac_max": round(
                max(float(b[sk]["nsa_sac_dx_rel_l2_vs_no_ac"]) for b in sb_list), 8
            ),
        }
        for mk in ("dense_fa2_no_ac", "nsa_no_ac", "nsa_zero_sel_recompute_sac", "nsa_full_layer_ac"):
            ms_v = [float(b[sk][mk]["median_ms"]) for b in sb_list]
            tps_v = [float(b[sk][mk]["tok_per_s"]) for b in sb_list]
            mem_v = [float(b[sk][mk]["inc_peak_mib"]) for b in sb_list]
            row_b[mk] = {
                "ms_median": round(_med(ms_v), 3),
                "ms_iqr": round(_iqr(ms_v), 3),
                "tok_per_s_median": round(_med(tps_v), 1),
                "inc_peak_mib_median": round(_med(mem_v), 2),
            }
        row_b["nsa_no_ac_speedup_over_dense_fa2"] = round(
            row_b["dense_fa2_no_ac"]["ms_median"] / max(row_b["nsa_no_ac"]["ms_median"], 1e-9), 3
        )
        row_b["nsa_sac_speedup_over_dense_fa2"] = round(
            row_b["dense_fa2_no_ac"]["ms_median"] / max(row_b["nsa_zero_sel_recompute_sac"]["ms_median"], 1e-9), 3
        )
        row_b["nsa_sac_speedup_over_full_ac"] = round(
            row_b["nsa_full_layer_ac"]["ms_median"] / max(row_b["nsa_zero_sel_recompute_sac"]["ms_median"], 1e-9), 3
        )
        row_b["nsa_sac_mem_saving_ratio"] = round(
            row_b["nsa_no_ac"]["inc_peak_mib_median"]
            / max(row_b["nsa_zero_sel_recompute_sac"]["inc_peak_mib_median"], 1e-9),
            3,
        )
        sb_agg[sk] = row_b

    # Suite C Aggregation
    sc_list = [r["suite_c_cp8_usp_nsa_8gpu"] for r in reps]
    c1_ref = sc_list[0]["parity_and_coupon_collector_audit_s16k"]
    c1_agg = {
        "seqlen": int(c1_ref["seqlen"]),
        "ulysses8_out_rel_l2_max": round(
            max(float(c["parity_and_coupon_collector_audit_s16k"]["ulysses8_out_rel_l2"]) for c in sc_list), 6
        ),
        "ulysses8_dq_rel_l2_max": round(
            max(float(c["parity_and_coupon_collector_audit_s16k"]["ulysses8_dq_rel_l2"]) for c in sc_list), 6
        ),
        "ulysses8_dk_slc_rel_l2_max": round(
            max(float(c["parity_and_coupon_collector_audit_s16k"]["ulysses8_dk_slc_rel_l2"]) for c in sc_list), 6
        ),
        "usp_4x2_out_rel_l2_max": round(
            max(float(c["parity_and_coupon_collector_audit_s16k"]["usp_4x2_out_rel_l2"]) for c in sc_list), 6
        ),
        "usp_4x2_dq_rel_l2_max": round(
            max(float(c["parity_and_coupon_collector_audit_s16k"]["usp_4x2_dq_rel_l2"]) for c in sc_list), 6
        ),
        "usp_4x2_dk_cmp_raw_rel_l2_max": round(
            max(float(c["parity_and_coupon_collector_audit_s16k"]["usp_4x2_dk_cmp_raw_rel_l2"]) for c in sc_list), 6
        ),
        "usp_4x2_dk_slc_rel_l2_max": round(
            max(float(c["parity_and_coupon_collector_audit_s16k"]["usp_4x2_dk_slc_rel_l2"]) for c in sc_list), 6
        ),
        "usp_4x2_dk_win_rel_l2_max": round(
            max(float(c["parity_and_coupon_collector_audit_s16k"]["usp_4x2_dk_win_rel_l2"]) for c in sc_list), 6
        ),
        "coupon_collector_tail_rank_unique_kv_block_union_fraction": round(
            _med([float(c["parity_and_coupon_collector_audit_s16k"]["coupon_collector_tail_rank_unique_kv_block_union_fraction"]) for c in sc_list]), 4
        ),
        "coupon_collector_tail_rank_kv_ranks_touched_fraction": round(
            _med([float(c["parity_and_coupon_collector_audit_s16k"]["coupon_collector_tail_rank_kv_ranks_touched_fraction"]) for c in sc_list]), 4
        ),
    }

    c2_agg: dict[str, Any] = {}
    for sk, sref in sc_list[0]["scaling_cp8"].items():
        row_c: dict[str, Any] = {"seqlen": int(sref["seqlen"])}
        for mk in (
            "cp8_ulysses8_replicated_kv_dense_fa2",
            "cp8_seq_sharded_kv_demand_allgather_nsa",
            "cp8_ulysses8_replicated_kv_nsa",
            "cp8_usp_4x2_group_preserving_nsa",
        ):
            ms_v = [float(c["scaling_cp8"][sk][mk]["median_ms"]) for c in sc_list]
            tps_v = [float(c["scaling_cp8"][sk][mk]["tok_per_s"]) for c in sc_list]
            mem_v = [float(c["scaling_cp8"][sk][mk]["max_rank_inc_peak_mib"]) for c in sc_list]
            d_mk: dict[str, Any] = {
                "ms_median": round(_med(ms_v), 3),
                "ms_iqr": round(_iqr(ms_v), 3),
                "tok_per_s_median": round(_med(tps_v), 1),
                "max_rank_inc_peak_mib_median": round(_med(mem_v), 2),
            }
            if "comm_mib_per_rank" in sref[mk]:
                d_mk["comm_mib_per_rank"] = float(sref[mk]["comm_mib_per_rank"])
            row_c[mk] = d_mk

        usp_ms = row_c["cp8_usp_4x2_group_preserving_nsa"]["ms_median"]
        row_c["usp_speedup_over_dense_fa2"] = round(
            row_c["cp8_ulysses8_replicated_kv_dense_fa2"]["ms_median"] / max(usp_ms, 1e-9), 3
        )
        row_c["usp_speedup_over_seq_sharded_kv_ag"] = round(
            row_c["cp8_seq_sharded_kv_demand_allgather_nsa"]["ms_median"] / max(usp_ms, 1e-9), 3
        )
        row_c["usp_speedup_over_ulysses8_replicated_nsa"] = round(
            row_c["cp8_ulysses8_replicated_kv_nsa"]["ms_median"] / max(usp_ms, 1e-9), 3
        )
        row_c["usp_comm_reduction_vs_ulysses8_pct"] = round(
            100.0
            * (
                1.0
                - row_c["cp8_usp_4x2_group_preserving_nsa"]["comm_mib_per_rank"]
                / max(row_c["cp8_ulysses8_replicated_kv_nsa"]["comm_mib_per_rank"], 1e-9)
            ),
            2,
        )
        row_c["usp_peak_mem_saving_vs_seq_ag_ratio"] = round(
            row_c["cp8_seq_sharded_kv_demand_allgather_nsa"]["max_rank_inc_peak_mib_median"]
            / max(row_c["cp8_usp_4x2_group_preserving_nsa"]["max_rank_inc_peak_mib_median"], 1e-9),
            3,
        )
        c2_agg[sk] = row_c

    suite_c_summary = {
        "parity_and_coupon_collector_audit_s16k": c1_agg,
        "scaling_cp8": c2_agg,
    }

    # 9 Strict Deterministic Validation Gates
    gates = {
        "smoke_present_and_env_parity": smoke_and_env_ok,
        "clocks_locked_1410": clocks_locked_1410,
        "pytorch_213_flex_attention_quadruple_defect_audit": (
            flex_agg["defect1_uncompiled_create_block_mask_hq64_s16k"]["oom_triggered_all_reps"]
            and flex_agg["defect2_from_kv_blocks_transpose_ordered"]["bug_triggered_all_reps"]
            and flex_agg["defect3_block_size_64_inductor_lowering"]["rejected_all_reps"]
            and flex_agg["defect4_s64k_gqa64_bwd_illegal_memory_access"]["illegal_address_triggered_all_reps"]
        ),
        "sm80_query_group_tile_saturation_roofline": (
            tile_agg["sub4_64q_speedup_over_quest_m1"] > 24.0
            and tile_agg["modes"]["nsa_h16_sub4_64q4kv"]["achieved_sparse_tflops_median"] > 100.0
            and tile_agg["sub16_32q_speedup_over_tok1_padded_32q"] > 5.8
        ),
        "lock_free_split256_csc_backward_parity_and_speedup": (
            csc_agg["S_16384"]["speedup_compacted_csc_over_atomic"] > 45.0
            and csc_agg["S_32768"]["speedup_compacted_csc_over_atomic"] > 50.0
            and csc_agg["S_32768"]["speedup_compacted_csc_over_unsplit_csc"] > 1.70
            and csc_agg["S_32768"]["dk_rel_l2_csc_vs_atomic_max"] < 0.001
            and csc_agg["S_32768"]["dv_rel_l2_csc_vs_atomic_max"] < 0.001
            and csc_agg["S_32768"]["dq_rel_l2_csc_vs_unsplit_max"] == 0.0
        ),
        "fp64_oracle_and_richardson_central_difference": (
            max(oracle_agg.values()) < 0.006
            and rich_agg["min_ratio"] > 3.80
            and rich_agg["nsa_dv_win_linear_fd_abs_err_max"] < 1e-12
        ),
        "suite_a_1gpu_nsa_and_moba_long_context_speedups": (
            sa_agg["S_16384"]["deepseek_nsa_3branch_sub4_csc"]["speedup_over_dense_fa2"] > 1.65
            and sa_agg["S_32768"]["deepseek_nsa_3branch_sub4_csc"]["speedup_over_dense_fa2"] > 2.90
            and sa_agg["S_65536"]["deepseek_nsa_3branch_sub4_csc"]["speedup_over_dense_fa2"] > 4.65
            and sa_agg["S_65536"]["moba_gqa_speedup_over_per_head"] > 2.70
            and sa_agg["S_65536"]["moba_gqa_group_triton_lse"]["speedup_over_dense_fa2"] > 8.0
        ),
        "suite_b_4layer_nsa_sac_memory_and_speedup": (
            sb_agg["S_32768"]["nsa_no_ac_speedup_over_dense_fa2"] > 1.90
            and sb_agg["S_32768"]["nsa_sac_speedup_over_full_ac"] > 1.15
            and sb_agg["S_32768"]["nsa_sac_mem_saving_ratio"] > 1.25
            and sb_agg["S_16384"]["nsa_sac_dw_rel_l2_vs_no_ac_max"] < 0.01
            and sb_agg["S_16384"]["nsa_sac_dx_rel_l2_vs_no_ac_max"] < 0.005
        ),
        "suite_c_8gpu_2d_usp_nsa_parity_and_128k_scaling": (
            c1_agg["usp_4x2_out_rel_l2_max"] < 1e-5
            and c1_agg["usp_4x2_dq_rel_l2_max"] < 1e-4
            and c1_agg["usp_4x2_dk_cmp_raw_rel_l2_max"] < 0.01
            and c1_agg["usp_4x2_dk_slc_rel_l2_max"] < 0.01
            and c1_agg["coupon_collector_tail_rank_unique_kv_block_union_fraction"] > 0.90
            and c1_agg["coupon_collector_tail_rank_kv_ranks_touched_fraction"] == 1.0
            and c2_agg["S_131072"]["usp_speedup_over_dense_fa2"] > 6.0
            and c2_agg["S_131072"]["usp_speedup_over_seq_sharded_kv_ag"] > 1.30
            and c2_agg["S_131072"]["usp_speedup_over_ulysses8_replicated_nsa"] > 1.05
            and c2_agg["S_131072"]["usp_peak_mem_saving_vs_seq_ag_ratio"] > 1.80
        ),
    }
    all_passed = all(gates.values())

    full_out = {
        "experiment": "E125_native_sparse_attention_training",
        "repetitions": 3,
        "world_size": 8,
        "environment": reps[0]["environment"],
        "clock_telemetry": {
            "initial_sm_clock_mhz_min": round(min(init_clocks), 1),
            "median_sm_clock_mhz": round(med_clock, 1),
            "min_sm_clock_mhz": round(min_clock, 1),
            "max_sm_clock_mhz": round(max_clock, 1),
            "max_power_w": round(max_power, 2),
        },
        "validation_gates": {
            "all_gates_passed": all_passed,
            "gates": gates,
        },
        "suite_0_audits": suite_0_summary,
        "suite_a_1gpu_sparse_attention_scaling": sa_agg,
        "suite_b_4layer_nsa_sac": sb_agg,
        "suite_c_cp8_usp_nsa_8gpu": suite_c_summary,
    }

    out_path = raw_dir / "native_sparse_attention_training_full.json"
    out_path.write_text(
        json.dumps(full_out, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    readme_lines = [
        "# E125 Raw Artifacts: Native Trainable Sparse Attention (`DeepSeek NSA` & `Kimi MoBA`), Lock-Free Split-Reduce `CSC` Backward, `NSA-SAC`, & 8-GPU `2D USP-NSA` (`CP = 8`)",
        "",
        "Generated by `code/bench_native_sparse_attention_training.py` across 8x NVIDIA A100-SXM4-80GB (`NV12` NVSwitch) with locked SM clocks (`1,410 MHz`), and aggregated deterministically by `code/analyze_native_sparse_attention_training.py`.",
        "",
        "## Files",
        "- `smoke_native_sparse_attention_training.json`: 8-GPU `--mode smoke` functional & numerical gate artifact.",
        "- `rep1_native_sparse_attention_training.json`, `rep2_native_sparse_attention_training.json`, `rep3_native_sparse_attention_training.json`: 3 independent OS-process `--mode full` 8-GPU benchmark artifacts.",
        "- `native_sparse_attention_training_full.json`: Median/IQR 3-repetition summary and 9-gate deterministic validation report (`all_gates_passed = true`).",
        "- `SHA256SUMS`: SHA-256 digests of all JSON artifacts in this directory.",
        "",
    ]
    (raw_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8", newline="\n")

    tracked_files = [
        "smoke_native_sparse_attention_training.json",
        "rep1_native_sparse_attention_training.json",
        "rep2_native_sparse_attention_training.json",
        "rep3_native_sparse_attention_training.json",
        "native_sparse_attention_training_full.json",
    ]
    sha_lines = [f"{_sha256_file(raw_dir / fn)}  {fn}" for fn in tracked_files]
    (raw_dir / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8", newline="\n")

    print(f"Wrote {out_path} (all_gates_passed={all_passed})")
    if not all_passed:
        raise SystemExit(f"Validation gates failed: {gates}")


if __name__ == "__main__":
    main()
