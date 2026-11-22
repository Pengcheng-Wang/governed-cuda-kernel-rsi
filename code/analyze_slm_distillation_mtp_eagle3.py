#!/usr/bin/env python3
"""Deterministic 3-repetition analyzer and 9-gate validator for E122:
SLM Knowledge Distillation Coupled-Monoid Kernels, DeepSeek-V3 MTP,
EAGLE-3 GQA-Packed Two-Region TTT Attention, & 8-GPU Vocab-Parallel Distillation.
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic 3-repetition analyzer and 9-gate validator for E122"
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.root / "results" / "raw" / "slm_distillation_mtp_eagle3"
    smoke = json.loads(
        (raw_dir / "smoke_slm_distillation_mtp_eagle3.json").read_text(encoding="utf-8")
    )
    reps = [
        json.loads(
            (raw_dir / f"rep{i}_slm_distillation_mtp_eagle3.json").read_text(
                encoding="utf-8"
            )
        )
        for i in (1, 2, 3)
    ]

    envs = [
        (
            r["environment"].get("torch"),
            r["environment"].get("triton"),
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
    for r in reps:
        for sc in r["suite_a_slm_distillation_and_mtp"][
            "slm_distillation_vocab_scaling"
        ].values():
            for m in sc["modes"].values():
                if m.get("sm_clock_after_mhz", 0.0) > 0:
                    all_clocks.append(float(m["sm_clock_after_mhz"]))
                if m.get("power_after_w", 0.0) > 0:
                    all_powers.append(float(m["power_after_w"]))
        for sc in r["suite_b_eagle3_ttt_layer"]["sequence_scaling"].values():
            for m in sc["modes"].values():
                if m.get("sm_clock_after_mhz", 0.0) > 0:
                    all_clocks.append(float(m["sm_clock_after_mhz"]))
                if m.get("power_after_w", 0.0) > 0:
                    all_powers.append(float(m["power_after_w"]))

    med_clock = _med(all_clocks) if all_clocks else 0.0
    min_clock = min(all_clocks) if all_clocks else 0.0
    max_clock = max(all_clocks) if all_clocks else 0.0
    max_power = max(all_powers) if all_powers else 0.0
    clocks_locked_1410 = (
        all(c >= 1380.0 for c in init_clocks)
        and max_clock >= 1380.0
        and med_clock >= 1275.0
        and min_clock >= 1200.0
    )

    # Suite 0 Aggregation
    s0_list = [r["suite_0_audits"] for r in reps]
    shift_audit = s0_list[0]["fp32_cross_distribution_shift_audit"]
    topk_audit = s0_list[0]["topk_truncation_gradient_bias_audit"]

    min_fd = min(
        float(s["fp64_chain_rule_refinement"]["min_refinement_ratio"]) for s in s0_list
    )
    fd_med = {
        k: round(
            _med(
                [
                    float(s["fp64_chain_rule_refinement"]["ratios_h_vs_half_h"][k])
                    for s in s0_list
                ]
            ),
            4,
        )
        for k in s0_list[0]["fp64_chain_rule_refinement"]["ratios_h_vs_half_h"]
    }

    kd_par_max = {
        mk: {
            k: round(
                max(float(s["bf16_kd_parity"][mk][k]) for s in s0_list),
                6,
            )
            for k in s0_list[0]["bf16_kd_parity"][mk]
        }
        for mk in s0_list[0]["bf16_kd_parity"]
    }

    ab_list = [s["eagle3_ttt_parity_and_gqa_packing_ablation"] for s in s0_list]
    ttt_par_max = {
        k: round(max(float(a["bf16_parity"][k]) for a in ab_list), 6)
        for k in ab_list[0]["bf16_parity"]
    }
    unp_ms = _med([float(a["bwd_dq_unpacked_4x_kv_reload_ms"]) for a in ab_list])
    pk_ms = _med([float(a["bwd_dq_gqa_packed_single_kv_load_ms"]) for a in ab_list])
    gqa_pack_sp = round(unp_ms / max(pk_ms, 1e-6), 4)

    rm_par_max = {
        k: round(max(float(s["dual_rmsnorm_bf16_parity"][k]) for s in s0_list), 6)
        for k in s0_list[0]["dual_rmsnorm_bf16_parity"]
    }

    suite_0_summary = {
        "fp32_cross_distribution_shift_audit": shift_audit,
        "topk_truncation_gradient_bias_audit": topk_audit,
        "bf16_kd_parity_max": kd_par_max,
        "fp64_chain_rule_refinement": {
            "ratios_h_vs_half_h_median": fd_med,
            "min_refinement_ratio": round(min_fd, 4),
        },
        "eagle3_ttt_parity_and_gqa_packing_ablation": {
            "seq_len": ab_list[0]["seq_len"],
            "bf16_parity_max": ttt_par_max,
            "bwd_dq_unpacked_4x_kv_reload_ms_median": round(unp_ms, 4),
            "bwd_dq_gqa_packed_single_kv_load_ms_median": round(pk_ms, 4),
            "gqa_packing_bwd_dq_speedup": gqa_pack_sp,
        },
        "dual_rmsnorm_bf16_parity_max": rm_par_max,
    }

    # Suite A Aggregation
    sa_list = [r["suite_a_slm_distillation_and_mtp"] for r in reps]
    kd_agg: dict[str, Any] = {}
    for nk, nref in sa_list[0]["slm_distillation_vocab_scaling"].items():
        tri_rkl_ms = _med(
            [
                float(
                    a["slm_distillation_vocab_scaling"][nk]["modes"][
                        "triton_coupled_monoid_rkl_bf16dw"
                    ]["median_ms"]
                )
                for a in sa_list
            ]
        )
        tri_rkl_ws = _med(
            [
                float(
                    a["slm_distillation_vocab_scaling"][nk]["modes"][
                        "triton_coupled_monoid_rkl_bf16dw"
                    ]["transient_workspace_mib"]
                )
                for a in sa_list
            ]
        )
        modes_out: dict[str, Any] = {}
        for mk in nref["modes"]:
            ms_v = [
                float(a["slm_distillation_vocab_scaling"][nk]["modes"][mk]["median_ms"])
                for a in sa_list
            ]
            mem_v = [
                float(
                    a["slm_distillation_vocab_scaling"][nk]["modes"][mk]["peak_inc_mib"]
                )
                for a in sa_list
            ]
            ws_v = [
                float(
                    a["slm_distillation_vocab_scaling"][nk]["modes"][mk][
                        "transient_workspace_mib"
                    ]
                )
                for a in sa_list
            ]
            tf_v = [
                float(
                    a["slm_distillation_vocab_scaling"][nk]["modes"][mk][
                        "effective_gemm_tflops"
                    ]
                )
                for a in sa_list
            ]
            m_med = _med(ms_v)
            ws_med = _med(ws_v)
            modes_out[mk] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "peak_inc_mib_median": round(_med(mem_v), 2),
                "transient_workspace_mib_median": round(ws_med, 2),
                "effective_gemm_tflops_median": round(_med(tf_v), 2),
                "triton_rkl_speedup_over_this": round(
                    m_med / max(tri_rkl_ms, 1e-6), 4
                ),
                "triton_rkl_workspace_reduction_over_this": round(
                    ws_med / max(tri_rkl_ws, 1e-6), 4
                ),
            }
        kd_agg[nk] = {
            "num_tokens": nref["num_tokens"],
            "teacher_hidden": nref["teacher_hidden"],
            "student_hidden": nref["student_hidden"],
            "vocab_size": nref["vocab_size"],
            "modes": modes_out,
        }

    mtp_ref = sa_list[0]["deepseek_v3_mtp_shared_head"]
    mtp_bat_ms = _med(
        [
            float(
                a["deepseek_v3_mtp_shared_head"]["modes"][
                    "mtp_fused_dual_rmsnorm_batched_shared_fp32dw"
                ]["median_ms"]
            )
            for a in sa_list
        ]
    )
    mtp_modes_out: dict[str, Any] = {}
    for mk in mtp_ref["modes"]:
        ms_v = [
            float(a["deepseek_v3_mtp_shared_head"]["modes"][mk]["median_ms"])
            for a in sa_list
        ]
        ws_v = [
            float(
                a["deepseek_v3_mtp_shared_head"]["modes"][mk][
                    "transient_workspace_mib"
                ]
            )
            for a in sa_list
        ]
        mem_v = [
            float(a["deepseek_v3_mtp_shared_head"]["modes"][mk]["peak_inc_mib"])
            for a in sa_list
        ]
        m_med = _med(ms_v)
        mtp_modes_out[mk] = {
            "ms_median": round(m_med, 4),
            "ms_iqr": round(_iqr(ms_v), 4),
            "peak_inc_mib_median": round(_med(mem_v), 2),
            "transient_workspace_mib_median": round(_med(ws_v), 2),
            "batched_shared_speedup_over_this": round(
                m_med / max(mtp_bat_ms, 1e-6), 4
            ),
        }

    suite_a_summary = {
        "slm_distillation_vocab_scaling": kd_agg,
        "deepseek_v3_mtp_shared_head": {
            "tokens_per_head": mtp_ref["tokens_per_head"],
            "num_heads_total": mtp_ref["num_heads_total"],
            "hidden": mtp_ref["hidden"],
            "vocab_size": mtp_ref["vocab_size"],
            "aux_dw_precision_audit": mtp_ref["aux_dw_precision_audit"],
            "modes": mtp_modes_out,
        },
    }

    # Suite B Aggregation
    sb_list = [r["suite_b_eagle3_ttt_layer"]["sequence_scaling"] for r in reps]
    sb_agg: dict[str, Any] = {}
    for sk, sref in sb_list[0].items():
        fl_ms = _med(
            [float(b[sk]["modes"]["eagle3_flex_blockmask_5s"]["median_ms"]) for b in sb_list]
        )
        modes_b: dict[str, Any] = {}
        for mk in sref["modes"]:
            ms_v = [float(b[sk]["modes"][mk]["median_ms"]) for b in sb_list]
            mem_v = [float(b[sk]["modes"][mk]["peak_inc_mib"]) for b in sb_list]
            ws_v = [
                float(b[sk]["modes"][mk]["transient_workspace_mib"]) for b in sb_list
            ]
            m_med = _med(ms_v)
            modes_b[mk] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "peak_inc_mib_median": round(_med(mem_v), 2),
                "transient_workspace_mib_median": round(_med(ws_v), 2),
                "speedup_vs_flex_blockmask": round(fl_ms / max(m_med, 1e-6), 4),
            }
        sb_agg[sk] = {
            "batch": sref["batch"],
            "context_len": sref["context_len"],
            "ttt_steps": sref["ttt_steps"],
            "total_draft_queries": sref["total_draft_queries"],
            "modes": modes_b,
        }

    # Suite C Aggregation
    sc_list = [r["suite_c_vocab_parallel_distillation"] for r in reps]
    c1_agg: dict[str, Any] = {}
    for mk in sc_list[0]["vocab_parallel_vs_1gpu_gradient_parity"]:
        c1_agg[mk] = {
            k: round(
                max(
                    float(
                        c["vocab_parallel_vs_1gpu_gradient_parity"][mk][k]
                    )
                    for c in sc_list
                ),
                6,
            )
            for k in sc_list[0]["vocab_parallel_vs_1gpu_gradient_parity"][mk]
        }

    c2_ref = sc_list[0]["online_8gpu_teacher_student_scaling"]
    mat_ms = _med(
        [
            float(
                c["online_8gpu_teacher_student_scaling"]["modes"][
                    "fsdp2_colocated_materialized_rkl"
                ]["step_ms_median"]
            )
            for c in sc_list
        ]
    )
    col_tri_ms = _med(
        [
            float(
                c["online_8gpu_teacher_student_scaling"]["modes"][
                    "fsdp2_colocated_triton_coupled_rkl"
                ]["step_ms_median"]
            )
            for c in sc_list
        ]
    )
    mat_mem = _med(
        [
            float(
                c["online_8gpu_teacher_student_scaling"]["modes"][
                    "fsdp2_colocated_materialized_rkl"
                ]["per_rank_peak_inc_mib"]
            )
            for c in sc_list
        ]
    )
    modes_c2: dict[str, Any] = {}
    for mk in c2_ref["modes"]:
        ms_v = [
            float(
                c["online_8gpu_teacher_student_scaling"]["modes"][mk][
                    "step_ms_median"
                ]
            )
            for c in sc_list
        ]
        mem_v = [
            float(
                c["online_8gpu_teacher_student_scaling"]["modes"][mk][
                    "per_rank_peak_inc_mib"
                ]
            )
            for c in sc_list
        ]
        tps_v = [
            float(
                c["online_8gpu_teacher_student_scaling"]["modes"][mk][
                    "global_tokens_per_s"
                ]
            )
            for c in sc_list
        ]
        m_med = _med(ms_v)
        mem_med = _med(mem_v)
        modes_c2[mk] = {
            "step_ms_median": round(m_med, 4),
            "step_ms_iqr": round(_iqr(ms_v), 4),
            "per_rank_peak_inc_mib_median": round(mem_med, 2),
            "global_tokens_per_s_median": round(_med(tps_v), 1),
            "speedup_vs_fsdp2_materialized": round(mat_ms / max(m_med, 1e-6), 4),
            "speedup_vs_fsdp2_triton_colocated": round(
                col_tri_ms / max(m_med, 1e-6), 4
            ),
            "memory_reduction_vs_fsdp2_materialized": round(
                mat_mem / max(mem_med, 1e-6), 4
            ),
        }

    suite_c_summary = {
        "vocab_parallel_vs_1gpu_gradient_parity_max": c1_agg,
        "online_8gpu_teacher_student_scaling": {
            "teacher_params_billions": c2_ref["teacher_params_billions"],
            "student_params_billions": c2_ref["student_params_billions"],
            "global_batch": c2_ref["global_batch"],
            "seq_len": c2_ref["seq_len"],
            "global_tokens": c2_ref["global_tokens"],
            "vocab_size": c2_ref["vocab_size"],
            "vocab_shard_per_gpu": c2_ref["vocab_shard_per_gpu"],
            "coupled_monoid_comm_mib": c2_ref["coupled_monoid_comm_mib"],
            "materialized_logit_comm_mib": c2_ref["materialized_logit_comm_mib"],
            "modes": modes_c2,
        },
    }

    # 9 Deterministic Validation Gates
    kd8 = kd_agg["8192"]["modes"]
    b2k = sb_agg["2048"]["modes"]
    c_hyb = modes_c2["hybrid_fsdp2_backbone_vocab_parallel_coupled_rkl"]
    c_vp = modes_c2["tp8_teacher_vocab_parallel_coupled_rkl"]

    gates = {
        "smoke_present_and_env_parity": smoke_and_env_ok,
        "clocks_locked_1410": clocks_locked_1410,
        "fp32_coupled_monoid_shift_invariance_and_topk_bias": (
            shift_audit["opposing_shift_4096"]["naive_uncentered_fp32_rel_l2"] > 1.0
            and shift_audit["opposing_shift_4096"]["naive_uncentered_negative_count"] > 0
            and shift_audit["opposing_shift_4096"]["coupled_monoid_fp32_rel_l2"] < 3e-3
            and shift_audit["opposing_shift_32768"]["coupled_monoid_fp32_rel_l2"] < 3e-3
            and shift_audit["opposing_shift_32768"]["coupled_monoid_negative_count"] == 0
            and topk_audit["top_128"]["dws_rel_l2_bias"] > 1.0
        ),
        "fp64_richardson_chain_rule": min_fd > 3.90,
        "bf16_kd_dual_rmsnorm_and_eagle3_parity": (
            max(v for d in kd_par_max.values() for v in d.values()) < 0.01
            and max(ttt_par_max.values()) < 0.01
            and max(rm_par_max.values()) < 0.01
            and gqa_pack_sp > 1.25
        ),
        "suite_a_slm_distillation_and_mtp_speed_and_workspace": (
            kd8["eager_fp32_rkl_materialized"]["triton_rkl_speedup_over_this"] > 1.80
            and kd8["eager_fp32_rkl_materialized"][
                "triton_rkl_workspace_reduction_over_this"
            ]
            > 50.0
            and mtp_modes_out["mtp_eager_4head_materialized"][
                "batched_shared_speedup_over_this"
            ]
            > 1.30
            and (
                mtp_modes_out["mtp_eager_4head_materialized"][
                    "transient_workspace_mib_median"
                ]
                / max(
                    mtp_modes_out[
                        "mtp_fused_dual_rmsnorm_batched_shared_fp32dw"
                    ]["transient_workspace_mib_median"],
                    1e-6,
                )
            )
            > 5.0
        ),
        "suite_b_eagle3_gqa_packed_two_region_speedup": (
            b2k["eagle3_triton_gqa_packed_two_region"]["speedup_vs_flex_blockmask"]
            > 1.06
            and (
                b2k["eagle3_sdpa_staircase_mask"]["ms_median"]
                / max(
                    b2k["eagle3_triton_gqa_packed_two_region"]["ms_median"], 1e-6
                )
            )
            > 5.0
        ),
        "suite_c_vocab_parallel_tp8_parity_vs_1gpu": (
            c1_agg["top128_truncated_rkl_bias_control"]["dw_lm_head_rel_l2"] > 1.0
            and c1_agg["vocab_parallel_tp8_fkl"]["dw_lm_head_rel_l2"] < 0.006
            and c1_agg["vocab_parallel_tp8_rkl"]["dw_lm_head_rel_l2"] < 0.006
            and c1_agg["vocab_parallel_tp8_jsd_0p5"]["dw_lm_head_rel_l2"] < 0.006
            and max(c1_agg["vocab_parallel_tp8_fkl"].values()) < 0.03
            and max(c1_agg["vocab_parallel_tp8_rkl"].values()) < 0.03
            and max(c1_agg["vocab_parallel_tp8_jsd_0p5"].values()) < 0.03
        ),
        "suite_c_hybrid_and_tp8_vocab_parallel_online_kd_speed_and_memory": (
            c_hyb["speedup_vs_fsdp2_materialized"] > 1.30
            and c_hyb["speedup_vs_fsdp2_triton_colocated"] > 1.08
            and (c_vp["step_ms_median"] / max(c_hyb["step_ms_median"], 1e-6)) > 1.10
            and c_hyb["memory_reduction_vs_fsdp2_materialized"] > 4.5
            and c_vp["speedup_vs_fsdp2_materialized"] > 1.12
            and c_vp["memory_reduction_vs_fsdp2_materialized"] > 4.5
        ),
    }
    all_passed = all(gates.values())

    full_out = {
        "experiment": "E122_slm_distillation_mtp_eagle3",
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
        "suite_a_slm_distillation_and_mtp": suite_a_summary,
        "suite_b_eagle3_ttt_layer": {"sequence_scaling": sb_agg},
        "suite_c_vocab_parallel_distillation": suite_c_summary,
    }

    out_path = raw_dir / "slm_distillation_mtp_eagle3_full.json"
    out_path.write_text(
        json.dumps(full_out, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Wrote {out_path} (all_gates_passed={all_passed})")
    if not all_passed:
        raise SystemExit(f"Validation gates failed: {gates}")


if __name__ == "__main__":
    main()
