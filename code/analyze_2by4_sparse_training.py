#!/usr/bin/env python3
"""Deterministic 3-repetition analyzer and 9-gate validator for E124:
SM80 2:4 Semi-Structured Sparse Tensor-Core Training (cuSPARSELt / Exact-90 Bipartite DP /
KKT-Exact Madow MVUE / Grad-SR-STE), Layout-Aware Sparse-SwiGLU, & 8-GPU NVSwitch
9 bits/param Compressed-AllGather FSDP2 Training.
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
        description="Deterministic 3-repetition analyzer and 9-gate validator for E124"
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.root / "results" / "raw" / "2by4_sparse_training"
    smoke = json.loads(
        (raw_dir / "smoke_2by4_sparse_training.json").read_text(encoding="utf-8")
    )
    reps = [
        json.loads(
            (raw_dir / f"rep{i}_2by4_sparse_training.json").read_text(encoding="utf-8")
        )
        for i in (1, 2, 3)
    ]

    envs = [
        (
            r["environment"].get("torch"),
            r["environment"].get("triton"),
            r["environment"].get("cusparselt_version"),
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
        for m in r["suite_0_audits"]["sm80_2by4_gemm_roofline"].values():
            if m.get("sm_clock_mhz", 0.0) > 0:
                all_clocks.append(float(m["sm_clock_mhz"]))
            if m.get("power_w", 0.0) > 0:
                all_powers.append(float(m["power_w"]))
        for sc in r["suite_a_linear_3gemm_scaling"]["token_scaling"].values():
            for m in sc["modes"].values():
                if m.get("sm_clock_mhz", 0.0) > 0:
                    all_clocks.append(float(m["sm_clock_mhz"]))
                if m.get("power_w", 0.0) > 0:
                    all_powers.append(float(m["power_w"]))
        for sc in r["suite_b_sparse_swiglu_stack"]["token_scaling"].values():
            for m in sc["modes"].values():
                if m.get("sm_clock_mhz", 0.0) > 0:
                    all_clocks.append(float(m["sm_clock_mhz"]))
                if m.get("power_w", 0.0) > 0:
                    all_powers.append(float(m["power_w"]))
        for m in r["suite_c_9bit_2by4_fsdp2_8gpu"]["fsdp2_8gpu_training_scaling"][
            "modes"
        ].values():
            if m.get("sm_clock_mhz", 0.0) > 0:
                all_clocks.append(float(m["sm_clock_mhz"]))
            if m.get("power_w", 0.0) > 0:
                all_powers.append(float(m["power_w"]))

    med_clock = _med(all_clocks) if all_clocks else 0.0
    min_clock = min(all_clocks) if all_clocks else 0.0
    max_clock = max(all_clocks) if all_clocks else 0.0
    max_power = max(all_powers) if all_powers else 0.0
    clocks_locked_1410 = (
        all(c >= 1380.0 for c in init_clocks)
        and max_clock >= 1380.0
        and med_clock >= 1260.0
        and min_clock >= 1140.0
    )

    # Suite 0 Aggregation
    s0_list = [r["suite_0_audits"] for r in reps]
    rf_ref = s0_list[0]["sm80_2by4_gemm_roofline"]
    dense_rf_ms = _med(
        [float(s["sm80_2by4_gemm_roofline"]["bf16_cublas_dense"]["median_ms"]) for s in s0_list]
    )
    rf_agg: dict[str, Any] = {}
    for kk in rf_ref:
        ms_v = [float(s["sm80_2by4_gemm_roofline"][kk]["median_ms"]) for s in s0_list]
        tf_v = [float(s["sm80_2by4_gemm_roofline"][kk]["effective_tflops"]) for s in s0_list]
        m_med = _med(ms_v)
        rf_agg[kk] = {
            "ms_median": round(m_med, 4),
            "ms_iqr": round(_iqr(ms_v), 4),
            "effective_tflops_median": round(_med(tf_v), 2),
            "speedup_vs_dense_bf16": round(dense_rf_ms / max(m_med, 1e-9), 4),
        }

    comp_ref = s0_list[0]["compression_kernel_ablation"]
    pyt_tile_ms = _med(
        [
            float(s["compression_kernel_ablation"]["pytorch_tile_4x4_greedy_abs"]["median_ms"])
            for s in s0_list
        ]
    )
    comp_agg: dict[str, Any] = {}
    for ck in comp_ref:
        ms_v = [float(s["compression_kernel_ablation"][ck]["median_ms"]) for s in s0_list]
        ws_v = [
            float(s["compression_kernel_ablation"][ck]["transient_workspace_mib"])
            for s in s0_list
        ]
        m_med = _med(ms_v)
        comp_agg[ck] = {
            "ms_median": round(m_med, 4),
            "ms_iqr": round(_iqr(ms_v), 4),
            "transient_workspace_mib_median": round(_med(ws_v), 2),
            "speedup_vs_pytorch_tile_greedy": round(pyt_tile_ms / max(m_med, 1e-9), 4),
        }

    bug_audit = s0_list[0]["pytorch_213_tile_bug_audit"]
    mvue_audit = s0_list[0]["kkt_mvue_monte_carlo_unbiasedness"]
    sr_traj = s0_list[0]["sr_ste_flip_rate_trajectories"]
    fd_ref = s0_list[0]["fp64_richardson_chain_rule"]
    nonquad_ratios = [
        min(float(s["fp64_richardson_chain_rule"][tk]["richardson_ratio"]) for s in s0_list)
        for tk in ("dx", "dgamma", "dw_gate_up_masked")
    ]
    min_nonquad_fd = min(nonquad_ratios)
    dw_down_exact_quad_err = max(
        float(s["fp64_richardson_chain_rule"]["dw_down_masked"]["err_h"]) for s in s0_list
    )
    max_man_vs_auto = max(
        float(s["fp64_richardson_chain_rule"]["manual_vs_autograd_max_rel_l2"]) for s in s0_list
    )
    fd_agg = {
        "manual_vs_autograd_max_rel_l2": max_man_vs_auto,
        "min_nonquadratic_richardson_ratio": round(min_nonquad_fd, 4),
        "dw_down_quadratic_exact_err_h_max": dw_down_exact_quad_err,
        "per_tensor": {
            tk: {
                "err_h": fd_ref[tk]["err_h"],
                "err_h_half": fd_ref[tk]["err_h_half"],
                "richardson_ratio": fd_ref[tk]["richardson_ratio"],
            }
            for tk in ("dx", "dgamma", "dw_gate_up_masked", "dw_down_masked")
        },
    }

    suite_0_summary = {
        "sm80_2by4_gemm_roofline": rf_agg,
        "compression_kernel_ablation": comp_agg,
        "pytorch_213_tile_bug_audit": bug_audit,
        "cslt_splitk_fallbacks_observed": reps[0].get("cslt_splitk_fallbacks", []),
        "kkt_mvue_monte_carlo_unbiasedness": mvue_audit,
        "sr_ste_flip_rate_trajectories": sr_traj,
        "fp64_richardson_chain_rule": fd_agg,
    }

    # Suite A Aggregation
    sa_list = [r["suite_a_linear_3gemm_scaling"]["token_scaling"] for r in reps]
    sa_agg: dict[str, Any] = {}
    for mk_tok, mref in sa_list[0].items():
        d_ms = _med(
            [float(a[mk_tok]["modes"]["bf16_cublas_dense"]["median_ms"]) for a in sa_list]
        )
        modes_a: dict[str, Any] = {}
        for mode_k in mref["modes"]:
            ms_v = [float(a[mk_tok]["modes"][mode_k]["median_ms"]) for a in sa_list]
            tf_v = [float(a[mk_tok]["modes"][mode_k]["effective_tflops"]) for a in sa_list]
            pk_v = [float(a[mk_tok]["modes"][mode_k]["peak_incremental_mib"]) for a in sa_list]
            m_med = _med(ms_v)
            modes_a[mode_k] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "effective_tflops_median": round(_med(tf_v), 2),
                "peak_incremental_mib_median": round(_med(pk_v), 2),
                "speedup_vs_dense_bf16": round(d_ms / max(m_med, 1e-9), 4),
            }
        sa_agg[mk_tok] = {
            "num_tokens": mref["num_tokens"],
            "in_features": mref["in_features"],
            "out_features": mref["out_features"],
            "dense_equiv_gflops": mref["dense_equiv_gflops"],
            "modes": modes_a,
        }

    # Suite B Aggregation
    sb_list = [r["suite_b_sparse_swiglu_stack"] for r in reps]
    ga_ref = sb_list[0]["gated_activation_layout_ablation"]
    strided_ms = _med(
        [
            float(
                b["gated_activation_layout_ablation"]["modes"][
                    "col_major_strided_eager_swiglu"
                ]["median_ms"]
            )
            for b in sb_list
        ]
    )
    ga_modes: dict[str, Any] = {}
    for gk in ga_ref["modes"]:
        ms_v = [
            float(b["gated_activation_layout_ablation"]["modes"][gk]["median_ms"])
            for b in sb_list
        ]
        gb_v = [
            float(b["gated_activation_layout_ablation"]["modes"][gk]["effective_gbps"])
            for b in sb_list
        ]
        m_med = _med(ms_v)
        ga_modes[gk] = {
            "ms_median": round(m_med, 4),
            "ms_iqr": round(_iqr(ms_v), 4),
            "effective_gbps_median": round(_med(gb_v), 2),
            "speedup_vs_col_major_strided_eager": round(strided_ms / max(m_med, 1e-9), 4),
        }
    gated_ablation_agg = {
        "shape_m_2i": ga_ref["shape_m_2i"],
        "col_coalesced_out_rel_l2_max": round(
            max(float(b["gated_activation_layout_ablation"]["col_coalesced_out_rel_l2"]) for b in sb_list),
            6,
        ),
        "col_coalesced_dgu_rel_l2_max": round(
            max(float(b["gated_activation_layout_ablation"]["col_coalesced_dgu_rel_l2"]) for b in sb_list),
            6,
        ),
        "row_fused_mvue_dgu_rel_l2_max": round(
            max(float(b["gated_activation_layout_ablation"]["row_fused_mvue_dgu_rel_l2"]) for b in sb_list),
            6,
        ),
        "modes": ga_modes,
    }

    stack_par_max = {
        k: round(max(float(b["stack_masked_bf16_parity"][k]) for b in sb_list), 6)
        for k in sb_list[0]["stack_masked_bf16_parity"]
    }

    sb_agg: dict[str, Any] = {}
    for mk_tok, mref in sb_list[0]["token_scaling"].items():
        eag_ms = _med(
            [
                float(b["token_scaling"][mk_tok]["modes"]["mlp_bf16_dense_eager"]["median_ms"])
                for b in sb_list
            ]
        )
        cmp_ms = _med(
            [
                float(b["token_scaling"][mk_tok]["modes"]["mlp_bf16_dense_compile"]["median_ms"])
                for b in sb_list
            ]
        )
        modes_b: dict[str, Any] = {}
        for mode_k in mref["modes"]:
            ms_v = [
                float(b["token_scaling"][mk_tok]["modes"][mode_k]["median_ms"]) for b in sb_list
            ]
            tf_v = [
                float(b["token_scaling"][mk_tok]["modes"][mode_k]["effective_tflops"])
                for b in sb_list
            ]
            pk_v = [
                float(b["token_scaling"][mk_tok]["modes"][mode_k]["peak_incremental_mib"])
                for b in sb_list
            ]
            m_med = _med(ms_v)
            modes_b[mode_k] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "effective_tflops_median": round(_med(tf_v), 2),
                "peak_incremental_mib_median": round(_med(pk_v), 2),
                "speedup_vs_dense_eager": round(eag_ms / max(m_med, 1e-9), 4),
                "speedup_vs_dense_compile": round(cmp_ms / max(m_med, 1e-9), 4),
            }
        sb_agg[mk_tok] = {
            "num_tokens": mref["num_tokens"],
            "num_layers": mref["num_layers"],
            "hidden": mref["hidden"],
            "intermediate": mref["intermediate"],
            "modes": modes_b,
        }

    # Suite C Aggregation
    sc_list = [r["suite_c_9bit_2by4_fsdp2_8gpu"] for r in reps]
    c1_ref = sc_list[0]["distributed_9bit_allgather_parity"]
    c1_agg = {
        "fwd_9bit_ag_rel_l2_vs_1gpu_exact90_max": round(
            max(
                float(c["distributed_9bit_allgather_parity"]["fwd_9bit_ag_rel_l2_vs_1gpu_exact90"])
                for c in sc_list
            ),
            6,
        ),
        "dx_9bit_ag_rel_l2_vs_1gpu_exact90_max": round(
            max(
                float(c["distributed_9bit_allgather_parity"]["dx_9bit_ag_rel_l2_vs_1gpu_exact90"])
                for c in sc_list
            ),
            6,
        ),
        "dw_gate_up_2gemm_9bit_ag_rel_l2_vs_1gpu_max": round(
            max(
                float(
                    c["distributed_9bit_allgather_parity"][
                        "dw_gate_up_2gemm_9bit_ag_rel_l2_vs_1gpu"
                    ]
                )
                for c in sc_list
            ),
            6,
        ),
        "dw_gate_up_3gemm_fst_mc_avg_rel_l2_vs_1gpu_max": round(
            max(
                float(
                    c["distributed_9bit_allgather_parity"][
                        "dw_gate_up_3gemm_fst_mc_avg_rel_l2_vs_1gpu"
                    ]
                )
                for c in sc_list
            ),
            6,
        ),
        "mc_dist_steps": c1_ref["mc_dist_steps"],
    }

    c2_ref = sc_list[0]["fsdp2_8gpu_training_scaling"]
    eag_c_ms = _med(
        [
            float(
                c["fsdp2_8gpu_training_scaling"]["modes"]["fsdp2_bf16_dense_eager"][
                    "median_ms"
                ]
            )
            for c in sc_list
        ]
    )
    cmp_c_ms = _med(
        [
            float(
                c["fsdp2_8gpu_training_scaling"]["modes"]["fsdp2_bf16_dense_compile"][
                    "median_ms"
                ]
            )
            for c in sc_list
        ]
    )
    post_c_ms = _med(
        [
            float(
                c["fsdp2_8gpu_training_scaling"]["modes"][
                    "fsdp2_2by4_post_ag_redundant_fst"
                ]["median_ms"]
            )
            for c in sc_list
        ]
    )
    modes_c2: dict[str, Any] = {}
    for mk in c2_ref["modes"]:
        ms_v = [float(c["fsdp2_8gpu_training_scaling"]["modes"][mk]["median_ms"]) for c in sc_list]
        tps_v = [
            float(c["fsdp2_8gpu_training_scaling"]["modes"][mk]["global_tok_per_s"])
            for c in sc_list
        ]
        mem_v = [
            float(
                c["fsdp2_8gpu_training_scaling"]["modes"][mk][
                    "peak_allocated_mib_max_rank"
                ]
            )
            for c in sc_list
        ]
        m_med = _med(ms_v)
        modes_c2[mk] = {
            "step_ms_median": round(m_med, 3),
            "step_ms_iqr": round(_iqr(ms_v), 3),
            "global_tok_per_s_median": round(_med(tps_v), 1),
            "peak_allocated_mib_max_rank_median": round(_med(mem_v), 2),
            "speedup_vs_fsdp2_dense_eager": round(eag_c_ms / max(m_med, 1e-9), 4),
            "speedup_vs_fsdp2_dense_compile": round(cmp_c_ms / max(m_med, 1e-9), 4),
            "speedup_vs_post_ag_redundant_fst": round(post_c_ms / max(m_med, 1e-9), 4),
        }

    suite_c_summary = {
        "distributed_9bit_allgather_parity": c1_agg,
        "fsdp2_8gpu_training_scaling": {
            "num_layers": c2_ref["num_layers"],
            "params_billions": c2_ref["params_billions"],
            "ffn_linear_param_share": c2_ref["ffn_linear_param_share"],
            "local_tokens_per_gpu": c2_ref["local_tokens_per_gpu"],
            "global_tokens_per_step": c2_ref["global_tokens_per_step"],
            "seq_len": c2_ref["seq_len"],
            "bf16_allgather_mib_per_step": c2_ref["bf16_allgather_mib_per_step"],
            "ffn_hybrid_9bit_allgather_mib_per_step": c2_ref[
                "ffn_hybrid_9bit_allgather_mib_per_step"
            ],
            "full_9bit_allgather_mib_per_step": c2_ref["full_9bit_allgather_mib_per_step"],
            "allgather_reduction_ratio_full_9bit": round(
                c2_ref["bf16_allgather_mib_per_step"]
                / max(c2_ref["full_9bit_allgather_mib_per_step"], 1e-9),
                4,
            ),
            "modes": modes_c2,
        },
    }

    # 9 Deterministic Validation Gates
    a8k = sa_agg["8192"]["modes"]
    a16k = sa_agg["16384"]["modes"]
    b8k = sb_agg["8192"]["modes"]
    c_9b_l1 = modes_c2["fsdp2_2by4_9bit_ag_fst_refresh1"]
    c_9b_l10 = modes_c2["fsdp2_2by4_9bit_ag_fst_refresh10"]
    c_hyb = modes_c2["fsdp2_2by4_ffn_hybrid_9bit_ag_refresh1"]

    gates = {
        "smoke_present_and_env_parity": smoke_and_env_ok,
        "clocks_locked_1410": clocks_locked_1410,
        "sm80_2by4_roofline_and_pytorch_213_bug_audit": (
            rf_agg["cusparselt_searched_tuned"]["effective_tflops_median"] > 420.0
            and rf_agg["cusparselt_searched_tuned"]["speedup_vs_dense_bf16"] > 1.55
            and rf_agg["pytorch_cutlass_2by4"]["speedup_vs_dense_bf16"] < 0.90
            and comp_agg["triton_exact90_dp_dual_pack"]["speedup_vs_pytorch_tile_greedy"] > 1.50
            and bug_audit["pytorch_default_empty_alg"]["kept_positive_fraction"] > 0.70
            and bug_audit["pytorch_largest_abs_greedy"]["deadlocked_4x4_tile_pct"] > 14.0
            and bug_audit["triton_exact90_bipartite_dp"]["deadlocked_4x4_tile_pct"] < 0.01
            and bug_audit["triton_exact90_bipartite_dp"]["retained_l1_pct_of_90mask_oracle"]
            >= 99.999
            and bug_audit["exact90_cslt_fwd_rel_l2_vs_sparse_dense"] < 0.001
            and bug_audit["exact90_cslt_dx_rel_l2_vs_sparse_dense"] < 0.001
        ),
        "kkt_mvue_unbiasedness_and_grad_sr_ste_stability": (
            mvue_audit["greedy_top2_mse_biased"]["steps_1024"] > 0.15
            and mvue_audit["kkt_exact_madow_mvue_2by4"]["steps_1"]
            < 0.55 * mvue_audit["uniform_unbiased_2x"]["steps_1"]
            and mvue_audit["kkt_exact_madow_mvue_2by4"]["steps_1024"] < 0.020
            and sr_traj["sparse_2by4_grad_sr_ste"]["tail_flip_rate_mean"]
            < 0.05 * sr_traj["sparse_2by4_naive_ste"]["tail_flip_rate_mean"]
            and sr_traj["sparse_2by4_weight_sr_ste"]["tail_flip_rate_mean"]
            > 0.80 * sr_traj["sparse_2by4_naive_ste"]["tail_flip_rate_mean"]
        ),
        "fp64_richardson_chain_rule": (
            min_nonquad_fd > 3.95
            and dw_down_exact_quad_err < 1e-13
            and max_man_vs_auto < 1e-13
        ),
        "gated_layout_coalescing_and_stack_masked_parity": (
            ga_modes["col_major_coalesced_triton_swiglu"][
                "speedup_vs_col_major_strided_eager"
            ]
            > 6.0
            and ga_modes["row_major_fused_swiglu_plus_mvue24_pack"][
                "speedup_vs_col_major_strided_eager"
            ]
            > 4.5
            and max(stack_par_max.values()) < 0.015
        ),
        "suite_a_3gemm_fst_breaks_sm80_312tflops_bf16_ceiling": (
            a8k["cslt_exact90_3gemm_fst_refresh1"]["effective_tflops_median"] > 380.0
            and a16k["cslt_exact90_3gemm_fst_refresh1"]["effective_tflops_median"] > 380.0
            and a8k["cslt_exact90_3gemm_fst_refresh1"]["speedup_vs_dense_bf16"] > 1.45
            and a8k["pytorch_cutlass_2by4_2gemm"]["speedup_vs_dense_bf16"] < 0.98
        ),
        "suite_b_4layer_sparse_swiglu_beats_eager_and_compile": (
            b8k["mlp_2by4_fst_cslt_refresh1"]["speedup_vs_dense_eager"] > 1.40
            and b8k["mlp_2by4_fst_cslt_refresh1"]["speedup_vs_dense_compile"] > 1.25
            and b8k["mlp_2by4_pytorch_cutlass_2gemm"]["speedup_vs_dense_compile"] < 0.90
        ),
        "suite_c_8gpu_9bit_allgather_fsdp2_parity_and_speedup": (
            c1_agg["fwd_9bit_ag_rel_l2_vs_1gpu_exact90_max"] < 0.010
            and c1_agg["dx_9bit_ag_rel_l2_vs_1gpu_exact90_max"] < 0.010
            and c1_agg["dw_gate_up_2gemm_9bit_ag_rel_l2_vs_1gpu_max"] < 0.010
            and c1_agg["dw_gate_up_3gemm_fst_mc_avg_rel_l2_vs_1gpu_max"] < 0.100
            and c_9b_l1["speedup_vs_fsdp2_dense_eager"] > 1.35
            and c_9b_l1["speedup_vs_fsdp2_dense_compile"] > 1.25
            and c_9b_l10["speedup_vs_post_ag_redundant_fst"] > 1.07
            and c_hyb["speedup_vs_fsdp2_dense_compile"] > 1.18
        ),
    }
    all_passed = all(gates.values())

    full_out = {
        "experiment": "E124_2by4_sparse_training",
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
        "suite_a_linear_3gemm_scaling": {
            "token_scaling": sa_agg,
        },
        "suite_b_sparse_swiglu_stack": {
            "gated_activation_layout_ablation": gated_ablation_agg,
            "stack_masked_bf16_parity_max": stack_par_max,
            "token_scaling": sb_agg,
        },
        "suite_c_9bit_2by4_fsdp2_8gpu": suite_c_summary,
    }

    out_path = raw_dir / "2by4_sparse_training_full.json"
    out_path.write_text(
        json.dumps(full_out, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Wrote {out_path} (all_gates_passed={all_passed})")
    if not all_passed:
        print(f"Validation gates failed: {gates}")


if __name__ == "__main__":
    main()
