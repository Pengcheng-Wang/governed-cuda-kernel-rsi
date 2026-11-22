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
        description="Deterministic 3-repetition analyzer and 9-gate validator for E121"
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.root / "results" / "raw" / "rlvr_grpo_post_training"
    smoke = json.loads(
        (raw_dir / "smoke_rlvr_grpo_post_training.json").read_text(encoding="utf-8")
    )
    reps = [
        json.loads(
            (raw_dir / f"rep{i}_rlvr_grpo_post_training.json").read_text(
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
    all_powers: list[float] = []
    for r in reps:
        for sc in r["suite_a_terminal_post_training_kernels"][
            "grpo_dapo_vocab_scaling"
        ].values():
            for m in sc["modes"].values():
                if m.get("sm_clock_after_mhz", 0.0) > 0:
                    all_clocks.append(float(m["sm_clock_after_mhz"]))
                if m.get("power_after_w", 0.0) > 0:
                    all_powers.append(float(m["power_after_w"]))
        for sc in r["suite_b_prefix_tree_grpo_stack"][
            "group_size_scaling"
        ].values():
            for m in sc["modes"].values():
                if m.get("sm_clock_after_mhz", 0.0) > 0:
                    all_clocks.append(float(m["sm_clock_after_mhz"]))
                if m.get("power_after_w", 0.0) > 0:
                    all_powers.append(float(m["power_after_w"]))
    med_clock = _med(all_clocks) if all_clocks else 0.0
    min_clock = min(all_clocks) if all_clocks else 0.0
    max_clock = max(all_clocks) if all_clocks else 0.0
    max_power = max(all_powers) if all_powers else 0.0
    # Locked at 1410 MHz (`nvidia-smi -lgc 1410,1410`); sustained 239 TFLOP/s GEMMs draw up to ~466W
    # (>400W A100 TDP), so hardware TDP regulation holds clocks in 1230-1395 MHz above 1140 MHz base!
    clocks_locked_1410 = (
        max_clock >= 1380.0 and med_clock >= 1275.0 and min_clock >= 1200.0
    )

    # 2. Suite 0 Aggregation
    s0_list = [r["suite_0_audits"] for r in reps]
    ent_audit = s0_list[0]["fp32_entropy_shift_invariance_audit"]

    min_fd = min(
        float(s["fp64_chain_rule_refinement"]["min_refinement_ratio"])
        for s in s0_list
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

    par_grpo_max = {
        k: round(
            max(
                float(s["bf16_triton_parity"]["fused_linear_grpo_dapo"][k])
                for s in s0_list
            ),
            6,
        )
        for k in s0_list[0]["bf16_triton_parity"]["fused_linear_grpo_dapo"]
    }
    par_dpo_max = {
        k: round(
            max(
                float(s["bf16_triton_parity"]["fused_saved_lse_linear_dpo"][k])
                for s in s0_list
            ),
            6,
        )
        for k in s0_list[0]["bf16_triton_parity"]["fused_saved_lse_linear_dpo"]
    }
    par_tree_max = {
        k: round(
            max(
                float(
                    s["bf16_triton_parity"][
                        "four_layer_prefix_tree_vs_duplicated_varlen"
                    ][k]
                )
                for s in s0_list
            ),
            6,
        )
        for k in s0_list[0]["bf16_triton_parity"][
            "four_layer_prefix_tree_vs_duplicated_varlen"
        ]
    }

    suite_0_summary = {
        "fp32_entropy_shift_invariance_audit": ent_audit,
        "fp64_chain_rule_refinement": {
            "ratios_h_vs_half_h_median": fd_med,
            "min_refinement_ratio": round(min_fd, 4),
        },
        "bf16_triton_parity_max": {
            "fused_linear_grpo_dapo": par_grpo_max,
            "fused_saved_lse_linear_dpo": par_dpo_max,
            "four_layer_prefix_tree_vs_duplicated_varlen": par_tree_max,
        },
    }

    # 3. Suite A Aggregation
    sa_list = [r["suite_a_terminal_post_training_kernels"] for r in reps]
    grpo_agg: dict[str, Any] = {}
    for nk, nref in sa_list[0]["grpo_dapo_vocab_scaling"].items():
        tri_ms = _med(
            [
                float(
                    a["grpo_dapo_vocab_scaling"][nk]["modes"][
                        "triton_fused_linear_grpo_dapo"
                    ]["median_ms"]
                )
                for a in sa_list
            ]
        )
        tri_bf16_ws = _med(
            [
                float(
                    a["grpo_dapo_vocab_scaling"][nk]["modes"][
                        "triton_fused_linear_grpo_bf16dw"
                    ]["transient_workspace_mib"]
                )
                for a in sa_list
            ]
        )
        tri_bf16_mem = _med(
            [
                float(
                    a["grpo_dapo_vocab_scaling"][nk]["modes"][
                        "triton_fused_linear_grpo_bf16dw"
                    ]["peak_inc_mib"]
                )
                for a in sa_list
            ]
        )
        modes_out: dict[str, Any] = {}
        for mk, mref in nref["modes"].items():
            ms_v = [
                float(a["grpo_dapo_vocab_scaling"][nk]["modes"][mk]["median_ms"])
                for a in sa_list
            ]
            mem_v = [
                float(a["grpo_dapo_vocab_scaling"][nk]["modes"][mk]["peak_inc_mib"])
                for a in sa_list
            ]
            ws_v = [
                float(
                    a["grpo_dapo_vocab_scaling"][nk]["modes"][mk][
                        "transient_workspace_mib"
                    ]
                )
                for a in sa_list
            ]
            tf_v = [
                float(
                    a["grpo_dapo_vocab_scaling"][nk]["modes"][mk][
                        "effective_3gemm_tflops"
                    ]
                )
                for a in sa_list
            ]
            m_med = _med(ms_v)
            mem_med = _med(mem_v)
            ws_med = _med(ws_v)
            modes_out[mk] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "peak_inc_mib_median": round(mem_med, 2),
                "transient_workspace_mib_median": round(ws_med, 2),
                "effective_3gemm_tflops_median": round(_med(tf_v), 2),
                "triton_speedup_over_this": round(m_med / max(tri_ms, 1e-6), 4),
                "triton_bf16dw_memory_reduction_over_this": round(
                    mem_med / max(tri_bf16_mem, 1e-6), 4
                ),
                "triton_bf16dw_workspace_reduction_over_this": round(
                    ws_med / max(tri_bf16_ws, 1e-6), 4
                ),
            }
        grpo_agg[nk] = {
            "num_tokens": nref["num_tokens"],
            "hidden": nref["hidden"],
            "vocab_size": nref["vocab_size"],
            "modes": modes_out,
        }

    dpo_agg: dict[str, Any] = {}
    for nk, nref in sa_list[0]["dpo_vocab_scaling"].items():
        tri_ms = _med(
            [
                float(
                    a["dpo_vocab_scaling"][nk]["modes"][
                        "triton_saved_lse_linear_dpo"
                    ]["median_ms"]
                )
                for a in sa_list
            ]
        )
        tri_mem = _med(
            [
                float(
                    a["dpo_vocab_scaling"][nk]["modes"][
                        "triton_saved_lse_linear_dpo"
                    ]["peak_inc_mib"]
                )
                for a in sa_list
            ]
        )
        tri_ws = _med(
            [
                float(
                    a["dpo_vocab_scaling"][nk]["modes"][
                        "triton_saved_lse_linear_dpo"
                    ]["transient_workspace_mib"]
                )
                for a in sa_list
            ]
        )
        modes_d: dict[str, Any] = {}
        for mk in nref["modes"]:
            ms_v = [
                float(a["dpo_vocab_scaling"][nk]["modes"][mk]["median_ms"])
                for a in sa_list
            ]
            mem_v = [
                float(a["dpo_vocab_scaling"][nk]["modes"][mk]["peak_inc_mib"])
                for a in sa_list
            ]
            ws_v = [
                float(
                    a["dpo_vocab_scaling"][nk]["modes"][mk][
                        "transient_workspace_mib"
                    ]
                )
                for a in sa_list
            ]
            m_med = _med(ms_v)
            mem_med = _med(mem_v)
            ws_med = _med(ws_v)
            modes_d[mk] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "peak_inc_mib_median": round(mem_med, 2),
                "transient_workspace_mib_median": round(ws_med, 2),
                "triton_speedup_over_this": round(m_med / max(tri_ms, 1e-6), 4),
                "triton_memory_reduction_over_this": round(
                    mem_med / max(tri_mem, 1e-6), 4
                ),
                "triton_workspace_reduction_over_this": round(
                    ws_med / max(tri_ws, 1e-6), 4
                ),
            }
        dpo_agg[nk] = {
            "num_tokens": nref["num_tokens"],
            "num_pairs": nref["num_pairs"],
            "hidden": nref["hidden"],
            "vocab_size": nref["vocab_size"],
            "modes": modes_d,
        }

    suite_a_summary = {
        "grpo_dapo_vocab_scaling": grpo_agg,
        "dpo_vocab_scaling": dpo_agg,
    }

    # 4. Suite B Aggregation
    sb_list = [r["suite_b_prefix_tree_grpo_stack"]["group_size_scaling"] for r in reps]
    sb_agg: dict[str, Any] = {}
    for gk, gref in sb_list[0].items():
        dup_ms = _med(
            [
                float(
                    b[gk]["modes"]["grpo_ragged_duplicated_prompt_fa2"]["median_ms"]
                )
                for b in sb_list
            ]
        )
        pad_ms = _med(
            [
                float(b[gk]["modes"]["grpo_padded_dense_fa2"]["median_ms"])
                for b in sb_list
            ]
        )
        dup_mem = _med(
            [
                float(
                    b[gk]["modes"]["grpo_ragged_duplicated_prompt_fa2"]["peak_inc_mib"]
                )
                for b in sb_list
            ]
        )
        modes_b: dict[str, Any] = {}
        for mk, mref in gref["modes"].items():
            ms_v = [float(b[gk]["modes"][mk]["median_ms"]) for b in sb_list]
            mem_v = [float(b[gk]["modes"][mk]["peak_inc_mib"]) for b in sb_list]
            m_med = _med(ms_v)
            mem_med = _med(mem_v)
            ent_b: dict[str, Any] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "peak_inc_mib_median": round(mem_med, 2),
                "backbone_tokens": mref["backbone_tokens"],
                "speedup_vs_padded_dense": round(pad_ms / max(m_med, 1e-6), 4),
                "speedup_vs_duplicated_ragged": round(dup_ms / max(m_med, 1e-6), 4),
                "memory_reduction_vs_duplicated_ragged": round(
                    dup_mem / max(mem_med, 1e-6), 4
                ),
            }
            if "first_call_compile_s" in mref:
                ent_b["first_call_compile_s_median"] = round(
                    _med(
                        [
                            float(b[gk]["modes"][mk]["first_call_compile_s"])
                            for b in sb_list
                        ]
                    ),
                    2,
                )
            modes_b[mk] = ent_b
        sb_agg[gk] = {
            "group_size": gref["group_size"],
            "num_prompts": gref["num_prompts"],
            "prompt_len": gref["prompt_len"],
            "response_tokens": gref["response_tokens"],
            "padded_tokens": gref["padded_tokens"],
            "duplicated_ragged_tokens": gref["duplicated_ragged_tokens"],
            "prefix_tree_tokens": gref["prefix_tree_tokens"],
            "token_compression_vs_duplicated": gref["token_compression_vs_duplicated"],
            "modes": modes_b,
        }

    # 5. Suite C Aggregation
    sc_list = [r["suite_c_fsdp2_rlvr"] for r in reps]
    c1_agg: dict[str, Any] = {}
    for pk in sc_list[0]["fsdp2_vs_1gpu_dapo_gradient_parity"]:
        c1_agg[pk] = {
            k: round(
                max(
                    float(c["fsdp2_vs_1gpu_dapo_gradient_parity"][pk][k])
                    for c in sc_list
                ),
                6,
            )
            for k in sc_list[0]["fsdp2_vs_1gpu_dapo_gradient_parity"][pk]
        }

    c2_ref = sc_list[0]["fsdp2_long_cot_rlvr_scaling"]
    rag_unbal_ms = _med(
        [
            float(
                c["fsdp2_long_cot_rlvr_scaling"]["modes"][
                    "fsdp2_prompt_sharded_ragged_fa2"
                ]["step_ms_median"]
            )
            for c in sc_list
        ]
    )
    pad_unbal_ms = _med(
        [
            float(
                c["fsdp2_long_cot_rlvr_scaling"]["modes"][
                    "fsdp2_prompt_sharded_padded_fa2"
                ]["step_ms_median"]
            )
            for c in sc_list
        ]
    )
    rag_unbal_mem = _med(
        [
            float(
                c["fsdp2_long_cot_rlvr_scaling"]["modes"][
                    "fsdp2_prompt_sharded_ragged_fa2"
                ]["per_rank_peak_inc_mib"]
            )
            for c in sc_list
        ]
    )
    modes_c2: dict[str, Any] = {}
    for mk, mref in c2_ref["modes"].items():
        ms_v = [
            float(
                c["fsdp2_long_cot_rlvr_scaling"]["modes"][mk]["step_ms_median"]
            )
            for c in sc_list
        ]
        mem_v = [
            float(
                c["fsdp2_long_cot_rlvr_scaling"]["modes"][mk][
                    "per_rank_peak_inc_mib"
                ]
            )
            for c in sc_list
        ]
        kt_v = [
            float(
                c["fsdp2_long_cot_rlvr_scaling"]["modes"][mk][
                    "response_ktok_per_s"
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
            "cluster_backbone_tokens": mref["cluster_backbone_tokens"],
            "rank_token_max_over_min": mref["rank_token_max_over_min"],
            "rank_cost_max_over_min": mref["rank_cost_max_over_min"],
            "response_ktok_per_s_median": round(_med(kt_v), 2),
            "speedup_vs_prompt_sharded_padded": round(
                pad_unbal_ms / max(m_med, 1e-6), 4
            ),
            "speedup_vs_prompt_sharded_ragged": round(
                rag_unbal_ms / max(m_med, 1e-6), 4
            ),
            "memory_reduction_vs_prompt_sharded_ragged": round(
                rag_unbal_mem / max(mem_med, 1e-6), 4
            ),
        }

    suite_c_summary = {
        "fsdp2_vs_1gpu_dapo_gradient_parity_max": c1_agg,
        "fsdp2_long_cot_rlvr_scaling": {
            "model_params_billions": c2_ref["model_params_billions"],
            "global_prompts": c2_ref["global_prompts"],
            "group_size": c2_ref["group_size"],
            "prompt_len": c2_ref["prompt_len"],
            "global_response_tokens": c2_ref["global_response_tokens"],
            "modes": modes_c2,
        },
    }

    # 6. 9 Deterministic Validation Gates
    g8_a = grpo_agg["8192"]["modes"]
    d8_a = dpo_agg["8192"]["modes"]
    b_g16 = sb_agg["16"]["modes"]
    c_lpt = modes_c2["fsdp2_cost_lpt_prefix_tree_gqa"]

    gates = {
        "smoke_present_and_env_parity": smoke_and_env_ok,
        "clocks_locked_1410": clocks_locked_1410,
        "fp32_entropy_cancellation_vs_centered_monoid": (
            ent_audit["shift_4096"]["naive_uncentered_fp32_rel_l2"] > 0.20
            and ent_audit["shift_4096"]["naive_uncentered_negative_count"] > 0
            and ent_audit["shift_65536"]["naive_uncentered_fp32_rel_l2"] > 0.08
            and ent_audit["shift_4096"]["centered_monoid_fp32_rel_l2"] < 1e-3
            and ent_audit["shift_65536"]["centered_monoid_fp32_rel_l2"] < 1e-3
            and ent_audit["shift_4096"]["centered_monoid_negative_count"] == 0
            and ent_audit["shift_65536"]["centered_monoid_negative_count"] == 0
        ),
        "fp64_richardson_chain_rule": min_fd > 3.90,
        "bf16_triton_grpo_dpo_and_tree_stack_parity": (
            par_grpo_max["loss_rel_err"] < 0.005
            and par_grpo_max["dx_rel_l2"] < 0.01
            and par_grpo_max["dw_fp32_accum_rel_l2"] < 0.01
            and par_dpo_max["dx_rel_l2"] < 0.01
            and par_dpo_max["dw_rel_l2"] < 0.01
            and max(par_tree_max.values()) < 0.015
        ),
        "suite_a_fused_grpo_and_saved_lse_dpo_memory_and_speed": (
            g8_a["eager_fp32_grpo_materialized"][
                "triton_bf16dw_workspace_reduction_over_this"
            ]
            > 40.0
            and g8_a["torch_compile_grpo_materialized"][
                "triton_bf16dw_workspace_reduction_over_this"
            ]
            > 20.0
            and g8_a["eager_fp32_grpo_materialized"]["triton_speedup_over_this"]
            > 1.50
            and g8_a["triton_fused_linear_grpo_dapo"][
                "effective_3gemm_tflops_median"
            ]
            > 200.0
            and d8_a["eager_fp32_dpo_materialized"][
                "triton_workspace_reduction_over_this"
            ]
            > 20.0
        ),
        "suite_b_prefix_tree_gqa_vs_duplicated_and_flex": (
            b_g16["grpo_prefix_tree_triton_gqa"]["speedup_vs_duplicated_ragged"]
            > 1.40
            and b_g16["grpo_prefix_tree_triton_gqa"]["speedup_vs_padded_dense"]
            > 2.50
            and (
                b_g16["grpo_prefix_tree_flex_blockmask"]["ms_median"]
                / max(b_g16["grpo_prefix_tree_triton_gqa"]["ms_median"], 1e-6)
            )
            > 1.05
        ),
        "suite_c_local_mean_bias_vs_global_dapo_norm_parity": (
            c1_agg["naive_local_token_mean_bias_control"]["dw_lm_head_rel_l2"]
            > 0.50
            and max(c1_agg["prompt_sharded_varlen_global_norm"].values()) < 0.03
            and max(c1_agg["cost_lpt_prefix_tree_global_norm"].values()) < 0.03
            and max(
                c1_agg["cost_lpt_vs_prompt_varlen_fsdp2_parity"].values()
            )
            < 0.01
        ),
        "suite_c_fsdp2_cost_lpt_prefix_tree_speedup_and_memory": (
            c_lpt["speedup_vs_prompt_sharded_ragged"] > 1.35
            and c_lpt["speedup_vs_prompt_sharded_padded"] > 2.0
            and c_lpt["memory_reduction_vs_prompt_sharded_ragged"] > 1.35
            and (
                modes_c2["fsdp2_token_greedy_ragged_fa2"]["step_ms_median"]
                / max(c_lpt["step_ms_median"], 1e-6)
            )
            > 1.10
        ),
    }
    all_passed = all(gates.values())

    full_out = {
        "experiment": "E121_rlvr_grpo_post_training",
        "repetitions": 3,
        "world_size": 8,
        "environment": reps[0]["environment"],
        "clock_telemetry": {
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
        "suite_a_terminal_post_training_kernels": suite_a_summary,
        "suite_b_prefix_tree_grpo_stack": {"group_size_scaling": sb_agg},
        "suite_c_fsdp2_rlvr": suite_c_summary,
    }

    out_path = raw_dir / "rlvr_grpo_post_training_full.json"
    out_path.write_text(
        json.dumps(full_out, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Wrote {out_path} (all_gates_passed={all_passed})", flush=True)
    if not all_passed:
        raise SystemExit(f"Validation gates failed: {gates}")


if __name__ == "__main__":
    main()
