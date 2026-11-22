from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _med(vals: list[float]) -> float:
    return float(statistics.median(vals))


def _iqr(vals: list[float]) -> float:
    s = sorted(vals)
    if len(s) < 2:
        return 0.0
    q25 = s[(len(s) - 1) // 4]
    q75 = s[(3 * (len(s) - 1)) // 4]
    return float(q75 - q25)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic 3-repetition analyzer and 9-gate validator for E119"
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.root / "results" / "raw" / "linear_attention_gdn_training"
    smoke = json.loads(
        (raw_dir / "smoke_linear_attention_gdn_training.json").read_text(
            encoding="utf-8"
        )
    )
    reps = [
        json.loads(
            (raw_dir / f"rep{i}_linear_attention_gdn_training.json").read_text(
                encoding="utf-8"
            )
        )
        for i in (1, 2, 3)
    ]

    # 1. Environment & Clock Telemetry
    envs = [
        (
            r["environment"].get("torch"),
            r["environment"].get("torch_cuda"),
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

    all_clocks: list[float] = []
    for r in reps:
        for sc in r["suite_a_single_gpu_scaling"].values():
            for m in sc["modes"].values():
                if m.get("sm_clock_after_mhz", 0.0) > 0:
                    all_clocks.append(float(m["sm_clock_after_mhz"]))
    med_clock = _med(all_clocks) if all_clocks else 0.0
    min_clock = min(all_clocks) if all_clocks else 0.0
    clocks_locked_1410 = med_clock >= 1380.0 and min_clock >= 1260.0

    # 2. Suite 0 Aggregation
    s0_list = [r["suite_0_audits"] for r in reps]
    ov0 = s0_list[0]["overflow_audit"]
    naive_overflow_all = all(
        bool(s["overflow_audit"]["naive_global_cumsum_has_nan_or_inf"])
        for s in s0_list
    )
    triton_finite_all = all(
        bool(s["overflow_audit"]["triton_chunkwise_ssd_all_finite"])
        for s in s0_list
    )
    wy_fp64_max = max(
        float(
            s["fp64_chain_rule_refinement"].get(
                "wy_vs_recurrent_fp64_max_rel_l2", 0.0
            )
        )
        for s in s0_list
    )
    min_fd_ratio = min(
        float(s["fp64_chain_rule_refinement"]["min_refinement_ratio"])
        for s in s0_list
    )
    fd_ratios_med = {
        k: round(
            _med(
                [
                    float(
                        s["fp64_chain_rule_refinement"]["ratios_h_vs_half_h"][k]
                    )
                    for s in s0_list
                ]
            ),
            4,
        )
        for k in s0_list[0]["fp64_chain_rule_refinement"]["ratios_h_vs_half_h"]
    }

    gdn_par_max = {
        k: round(
            max(float(s["triton_bf16_gdn_parity"][k]) for s in s0_list), 6
        )
        for k in s0_list[0]["triton_bf16_gdn_parity"]
    }
    ssd_par_max = {
        k: round(
            max(float(s["triton_bf16_ssd_parity"][k]) for s in s0_list), 6
        )
        for k in s0_list[0]["triton_bf16_ssd_parity"]
    }

    wy_pt_ms = _med(
        [
            float(
                s["wy_prep_kernel_audit"]["pytorch_solve_triangular_wy"][
                    "median_ms"
                ]
            )
            for s in s0_list
        ]
    )
    wy_ieee_ms = _med(
        [
            float(
                s["wy_prep_kernel_audit"]["triton_nilpotent_ieee_cuda_core"][
                    "median_ms"
                ]
            )
            for s in s0_list
        ]
    )
    wy_tc_ms = _med(
        [
            float(
                s["wy_prep_kernel_audit"]["triton_nilpotent_bf16_tensor_core"][
                    "median_ms"
                ]
            )
            for s in s0_list
        ]
    )
    wy_pt_mem = _med(
        [
            float(
                s["wy_prep_kernel_audit"]["pytorch_solve_triangular_wy"][
                    "peak_inc_mib"
                ]
            )
            for s in s0_list
        ]
    )
    wy_tc_mem = _med(
        [
            float(
                s["wy_prep_kernel_audit"]["triton_nilpotent_bf16_tensor_core"][
                    "peak_inc_mib"
                ]
            )
            for s in s0_list
        ]
    )

    suite_0_summary = {
        "overflow_audit": {
            "seq_len": ov0["seq_len"],
            "min_global_cumulative_log_gate": ov0[
                "min_global_cumulative_log_gate"
            ],
            "naive_global_cumsum_has_nan_or_inf_all": naive_overflow_all,
            "triton_chunkwise_ssd_all_finite_all": triton_finite_all,
        },
        "fp64_chain_rule_refinement": {
            "wy_vs_recurrent_fp64_max_rel_l2": float(f"{wy_fp64_max:.3e}"),
            "ratios_h_vs_half_h_median": fd_ratios_med,
            "min_refinement_ratio": round(min_fd_ratio, 4),
        },
        "triton_bf16_gdn_parity_max_rel_l2": gdn_par_max,
        "triton_bf16_ssd_parity_max_rel_l2": ssd_par_max,
        "wy_prep_kernel_audit": {
            "num_chunks": s0_list[0]["wy_prep_kernel_audit"]["num_chunks"],
            "pytorch_solve_triangular_ms_median": round(wy_pt_ms, 4),
            "pytorch_solve_triangular_peak_inc_mib": round(wy_pt_mem, 2),
            "triton_nilpotent_ieee_ms_median": round(wy_ieee_ms, 4),
            "triton_nilpotent_bf16_tc_ms_median": round(wy_tc_ms, 4),
            "triton_nilpotent_bf16_tc_peak_inc_mib": round(wy_tc_mem, 2),
            "tc_speedup_vs_pytorch_solve_triangular": round(
                wy_pt_ms / max(wy_tc_ms, 1e-6), 4
            ),
            "tc_speedup_vs_triton_ieee": round(
                wy_ieee_ms / max(wy_tc_ms, 1e-6), 4
            ),
        },
    }

    # 3. Suite A Aggregation
    suite_a_summary: dict[str, Any] = {}
    for sk, sref in reps[0]["suite_a_single_gpu_scaling"].items():
        fa2_ms = _med(
            [
                float(
                    r["suite_a_single_gpu_scaling"][sk]["modes"][
                        "flash_attn2_gqa_16q4kv"
                    ]["median_ms"]
                )
                for r in reps
            ]
        )
        modes_agg: dict[str, Any] = {}
        for mk, mref in sref["modes"].items():
            ms_v = [
                float(
                    r["suite_a_single_gpu_scaling"][sk]["modes"][mk][
                        "median_ms"
                    ]
                )
                for r in reps
            ]
            mem_v = [
                float(
                    r["suite_a_single_gpu_scaling"][sk]["modes"][mk][
                        "peak_inc_mib"
                    ]
                )
                for r in reps
            ]
            m_med = _med(ms_v)
            entry: dict[str, Any] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "peak_inc_mib_median": round(_med(mem_v), 2),
                "ktok_per_s_median": round(
                    (float(sref["batch"] * sref["seq_len"]) / (m_med * 1e-3))
                    / 1e3,
                    2,
                ),
                "speedup_vs_flash_attn2_gqa": round(
                    fa2_ms / max(m_med, 1e-6), 4
                ),
            }
            if "cuda_launches" in mref:
                entry["cuda_launches"] = int(mref["cuda_launches"])
            if "first_call_compile_s" in mref:
                entry["first_call_compile_s_median"] = round(
                    _med(
                        [
                            float(
                                r["suite_a_single_gpu_scaling"][sk]["modes"][
                                    mk
                                ]["first_call_compile_s"]
                            )
                            for r in reps
                        ]
                    ),
                    2,
                )
            modes_agg[mk] = entry
        suite_a_summary[sk] = {
            "batch": sref["batch"],
            "heads": sref["heads"],
            "head_dim": sref["head_dim"],
            "seq_len": sref["seq_len"],
            "modes": modes_agg,
        }

    # 4. Suite B Aggregation
    sb_list = [r["suite_b_mqar_and_hybrid_stack"] for r in reps]
    mqar_agg: dict[str, Any] = {}
    for mk in sb_list[0]["mqar_associative_overwrite"]:
        mqar_agg[mk] = {
            "steps": sb_list[0]["mqar_associative_overwrite"][mk]["steps"],
            "initial_ce_loss_median": round(
                _med(
                    [
                        float(
                            b["mqar_associative_overwrite"][mk][
                                "initial_ce_loss"
                            ]
                        )
                        for b in sb_list
                    ]
                ),
                4,
            ),
            "eval_ce_loss_median": round(
                _med(
                    [
                        float(
                            b["mqar_associative_overwrite"][mk]["eval_ce_loss"]
                        )
                        for b in sb_list
                    ]
                ),
                4,
            ),
            "eval_accuracy_all_median": round(
                _med(
                    [
                        float(
                            b["mqar_associative_overwrite"][mk][
                                "eval_accuracy_all"
                            ]
                        )
                        for b in sb_list
                    ]
                ),
                4,
            ),
            "eval_accuracy_overwritten_keys_median": round(
                _med(
                    [
                        float(
                            b["mqar_associative_overwrite"][mk][
                                "eval_accuracy_overwritten_keys"
                            ]
                        )
                        for b in sb_list
                    ]
                ),
                4,
            ),
            "eval_accuracy_untouched_keys_median": round(
                _med(
                    [
                        float(
                            b["mqar_associative_overwrite"][mk][
                                "eval_accuracy_untouched_keys"
                            ]
                        )
                        for b in sb_list
                    ]
                ),
                4,
            ),
        }

    hyb_agg: dict[str, Any] = {}
    for sk, sref in sb_list[0]["four_layer_hybrid_stack_scaling"].items():
        fa_stack_ms = _med(
            [
                float(
                    b["four_layer_hybrid_stack_scaling"][sk]["pure_fa2_4layer"][
                        "median_ms"
                    ]
                )
                for b in sb_list
            ]
        )
        sk_out: dict[str, Any] = {}
        for mk in sref:
            ms_v = [
                float(
                    b["four_layer_hybrid_stack_scaling"][sk][mk]["median_ms"]
                )
                for b in sb_list
            ]
            mem_v = [
                float(
                    b["four_layer_hybrid_stack_scaling"][sk][mk]["peak_inc_mib"]
                )
                for b in sb_list
            ]
            m_med = _med(ms_v)
            sk_out[mk] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "peak_inc_mib_median": round(_med(mem_v), 2),
                "speedup_vs_pure_fa2_4layer": round(
                    fa_stack_ms / max(m_med, 1e-6), 4
                ),
            }
        hyb_agg[sk] = sk_out

    mqar_ctrl_agg: dict[str, Any] = {}
    if "mqar_uncrowded_positive_control" in sb_list[0]:
        for mk in sb_list[0]["mqar_uncrowded_positive_control"]:
            mqar_ctrl_agg[mk] = {
                "steps": sb_list[0]["mqar_uncrowded_positive_control"][mk][
                    "steps"
                ],
                "eval_ce_loss_median": round(
                    _med(
                        [
                            float(
                                b["mqar_uncrowded_positive_control"][mk][
                                    "eval_ce_loss"
                                ]
                            )
                            for b in sb_list
                        ]
                    ),
                    4,
                ),
                "eval_accuracy_all_median": round(
                    _med(
                        [
                            float(
                                b["mqar_uncrowded_positive_control"][mk][
                                    "eval_accuracy_all"
                                ]
                            )
                            for b in sb_list
                        ]
                    ),
                    4,
                ),
            }

    suite_b_summary = {
        "mqar_uncrowded_positive_control": mqar_ctrl_agg,
        "mqar_associative_overwrite": mqar_agg,
        "four_layer_hybrid_stack_scaling": hyb_agg,
    }

    # 5. Suite C Aggregation
    sc_list = [r["suite_c_cp8_long_context"] for r in reps]
    cp_par_max = {
        k: round(
            max(float(c["cp8_vs_1gpu_parity"][k]) for c in sc_list), 6
        )
        for k in sc_list[0]["cp8_vs_1gpu_parity"]
    }

    cp_scale_agg: dict[str, Any] = {}
    for sk, sref in sc_list[0]["cp8_long_context_scaling"].items():
        uly_ms = _med(
            [
                float(
                    c["cp8_long_context_scaling"][sk]["modes"][
                        "cp8_ulysses_flash_attn2_gqa"
                    ]["step_ms_median"]
                )
                for c in sc_list
            ]
        )
        modes_c: dict[str, Any] = {}
        for mk, mref in sref["modes"].items():
            ms_v = [
                float(
                    c["cp8_long_context_scaling"][sk]["modes"][mk][
                        "step_ms_median"
                    ]
                )
                for c in sc_list
            ]
            kt_v = [
                float(
                    c["cp8_long_context_scaling"][sk]["modes"][mk][
                        "cluster_ktok_per_s"
                    ]
                )
                for c in sc_list
            ]
            mem_v = [
                float(
                    c["cp8_long_context_scaling"][sk]["modes"][mk][
                        "per_rank_peak_inc_mib"
                    ]
                )
                for c in sc_list
            ]
            m_med = _med(ms_v)
            ent_c: dict[str, Any] = {
                "step_ms_median": round(m_med, 4),
                "step_ms_iqr": round(_iqr(ms_v), 4),
                "cluster_ktok_per_s_median": round(_med(kt_v), 2),
                "per_rank_peak_inc_mib_median": round(_med(mem_v), 2),
                "speedup_vs_ulysses_fa2": round(uly_ms / max(m_med, 1e-6), 4),
            }
            if "comm_mib_per_rank" in mref:
                ent_c["comm_mib_per_rank"] = mref["comm_mib_per_rank"]
            modes_c[mk] = ent_c
        if "cp8_4mixer_pure_4fa2" in modes_c:
            fa4_ms = modes_c["cp8_4mixer_pure_4fa2"]["step_ms_median"]
            hyb4_ms = modes_c["cp8_4mixer_hybrid_3gdn_1fa2"]["step_ms_median"]
            modes_c["cp8_4mixer_hybrid_3gdn_1fa2"][
                "speedup_vs_pure_4fa2"
            ] = round(fa4_ms / max(hyb4_ms, 1e-6), 4)
        cp_scale_agg[sk] = {
            "global_seq_len": sref["global_seq_len"],
            "local_seq_len": sref["local_seq_len"],
            "heads": sref["heads"],
            "head_dim": sref["head_dim"],
            "modes": modes_c,
        }

    suite_c_summary = {
        "cp8_vs_1gpu_parity": cp_par_max,
        "cp8_long_context_scaling": cp_scale_agg,
    }

    # 6. Validation Gates (9 Deterministic Gates)
    s2k = suite_a_summary["2048"]["modes"]
    s8k = suite_a_summary["8192"]["modes"]
    s64k = suite_a_summary["65536"]["modes"]
    c128k = cp_scale_agg["131072"]["modes"]
    c256k = cp_scale_agg["262144"]["modes"]

    gates = {
        "smoke_present_and_env_parity": smoke_and_env_ok,
        "clocks_locked_1410": clocks_locked_1410,
        "overflow_and_fp64_chain_rule": (
            naive_overflow_all
            and triton_finite_all
            and min_fd_ratio > 3.90
        ),
        "triton_bf16_ssd_and_gdn_parity": (
            max(gdn_par_max.values()) < 0.02
            and max(ssd_par_max.values()) < 0.02
        ),
        "wy_tensor_core_nilpotent_speedup": (
            suite_0_summary["wy_prep_kernel_audit"][
                "tc_speedup_vs_pytorch_solve_triangular"
            ]
            > 2.5
            and suite_0_summary["wy_prep_kernel_audit"][
                "tc_speedup_vs_triton_ieee"
            ]
            > 2.5
        ),
        "suite_a_subquadratic_crossover_and_launch_reduction": (
            (
                s2k["pytorch_recurrent_step_eager"]["cuda_launches"]
                / max(s2k["triton_chunkwise_gdn_wy"]["cuda_launches"], 1)
            )
            > 100.0
            and (
                s2k["pytorch_recurrent_step_eager"]["ms_median"]
                / max(s2k["triton_chunkwise_gdn_wy"]["ms_median"], 1e-6)
            )
            > 20.0
            and (
                s8k["pytorch_chunkwise_wy_eager"]["ms_median"]
                / max(s8k["triton_chunkwise_gdn_wy"]["ms_median"], 1e-6)
            )
            > 2.0
            and s64k["triton_chunkwise_ssd_gla"]["speedup_vs_flash_attn2_gqa"]
            > 2.0
            and s64k["triton_chunkwise_gdn_wy"]["speedup_vs_flash_attn2_gqa"]
            > 2.0
        ),
        "suite_b_mqar_overwrite_and_hybrid_stack": (
            mqar_ctrl_agg["ssd_gla"]["eval_accuracy_all_median"] > 0.90
            and mqar_agg["gdn_wy"]["eval_accuracy_all_median"] > 0.90
            and (
                mqar_agg["gdn_wy"]["eval_accuracy_overwritten_keys_median"]
                - mqar_agg["ssd_gla"]["eval_accuracy_overwritten_keys_median"]
            )
            > 0.50
            and hyb_agg["65536"]["hybrid_3gdn_1fa2_4layer"][
                "speedup_vs_pure_fa2_4layer"
            ]
            > 1.35
        ),
        "cp8_distributed_parity": max(cp_par_max.values()) < 0.02,
        "cp8_256k_scaling_and_hybrid_speedup": (
            c256k["cp8_ssd_parallel_monoid"]["speedup_vs_ulysses_fa2"] > 5.0
            and c256k["cp8_gdn_parallel_monoid"]["speedup_vs_ulysses_fa2"] > 5.0
            and (
                c256k["cp8_ulysses_flash_attn2_gqa"]["comm_mib_per_rank"]
                / max(c256k["cp8_gdn_parallel_monoid"]["comm_mib_per_rank"], 1e-6)
            )
            > 20.0
            and c128k["cp8_4mixer_hybrid_3gdn_1fa2"]["speedup_vs_pure_4fa2"]
            > 2.0
        ),
    }
    all_passed = all(gates.values())

    full_out = {
        "experiment": "E119_linear_attention_gdn_training",
        "repetitions": 3,
        "world_size": 8,
        "environment": reps[0]["environment"],
        "clock_telemetry": {
            "median_sm_clock_mhz": round(med_clock, 1),
            "min_sm_clock_mhz": round(min_clock, 1),
        },
        "validation_gates": {
            "all_gates_passed": all_passed,
            "gates": gates,
        },
        "suite_0_audits": suite_0_summary,
        "suite_a_single_gpu_scaling": suite_a_summary,
        "suite_b_mqar_and_hybrid_stack": suite_b_summary,
        "suite_c_cp8_long_context": suite_c_summary,
        "rep_elapsed_seconds": [r["elapsed_s"] for r in reps],
    }

    out_path = raw_dir / "linear_attention_gdn_training_full.json"
    out_path.write_text(
        json.dumps(full_out, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Wrote {out_path} (all_gates_passed={all_passed})")
    if not all_passed:
        failed = [k for k, v in gates.items() if not v]
        raise SystemExit(f"Validation gates failed: {failed}")


if __name__ == "__main__":
    main()
