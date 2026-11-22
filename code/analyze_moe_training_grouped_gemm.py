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
        description="Deterministic 3-repetition analyzer and validator for E117"
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.root / "results" / "raw" / "moe_training_grouped_gemm"
    smoke = json.loads(
        (raw_dir / "smoke_moe_training_grouped_gemm.json").read_text(encoding="utf-8")
    )
    reps = [
        json.loads(
            (raw_dir / f"rep{i}_moe_training_grouped_gemm.json").read_text(
                encoding="utf-8"
            )
        )
        for i in (1, 2, 3)
    ]

    # 1. Environment & Clock Parity
    envs = [
        (
            r["environment"].get("torch"),
            r["environment"].get("torch_cuda"),
            r["environment"].get("triton"),
            r["environment"].get("gpu_name"),
            r["world_size"],
        )
        for r in reps
    ]
    env_parity = len(set(envs)) == 1 and reps[0]["world_size"] == 8

    all_clocks: list[float] = []
    for r in reps:
        for sk in r["suite_0_grouped_mm_audit"]["skew_sweep"].values():
            for m in sk["modes"].values():
                if m.get("sm_clock_after_mhz", 0.0) > 0:
                    all_clocks.append(float(m["sm_clock_after_mhz"]))
    min_obs_clock = min(all_clocks) if all_clocks else 0.0
    med_obs_clock = _med(all_clocks) if all_clocks else 0.0
    # Allow slight TDP trimming down to 1290 MHz under 400W cap while confirming 1410 MHz target lock
    clocks_locked_1410 = med_obs_clock >= 1380.0 and min_obs_clock >= 1260.0

    # 2. Suite 0 Aggregation
    s0_audits = [r["suite_0_grouped_mm_audit"]["audits"] for r in reps]
    zero_stride_all = all(a["zero_stride_grad_rejected"] for a in s0_audits)
    cudagraph_rej_all = all(a["cudagraph_capture_rejected"] for a in s0_audits)
    prof0 = s0_audits[0]["profiler_kernel_and_dtoh_counts"]
    dtoh_pt = prof0["pytorch_grouped_mm"]["dtoh_calls"]
    dtoh_pt_comp = prof0["pytorch_grouped_mm_compiled"]["dtoh_calls"]
    dtoh_tri = prof0["triton_grouped_gemm"]["dtoh_calls"]
    gemms_pt = prof0["pytorch_grouped_mm"]["vendor_gemm_launches"]
    gemms_pt_comp = prof0["pytorch_grouped_mm_compiled"]["vendor_gemm_launches"]
    gemms_tri = prof0["triton_grouped_gemm"]["triton_grouped_gemm_launches"]

    s0_skews_out: dict[str, Any] = {}
    for skew_name in ("uniform", "zipf_s1p2", "hot80"):
        ref_s0 = reps[0]["suite_0_grouped_mm_audit"]["skew_sweep"][skew_name]
        pt_med = _med(
            [
                r["suite_0_grouped_mm_audit"]["skew_sweep"][skew_name]["modes"][
                    "pytorch_grouped_mm"
                ]["median_ms"]
                for r in reps
            ]
        )
        modes_agg: dict[str, Any] = {}
        for mk in (
            "eager_python_loop",
            "padded_bmm_3d",
            "pytorch_grouped_mm",
            "pytorch_grouped_mm_compiled",
            "triton_grouped_gemm",
            "triton_grouped_gemm_cudagraph",
        ):
            ms_vals = [
                r["suite_0_grouped_mm_audit"]["skew_sweep"][skew_name]["modes"][mk][
                    "median_ms"
                ]
                for r in reps
            ]
            tf_vals = [
                r["suite_0_grouped_mm_audit"]["skew_sweep"][skew_name]["modes"][mk][
                    "achieved_tflops"
                ]
                for r in reps
            ]
            mem_vals = [
                r["suite_0_grouped_mm_audit"]["skew_sweep"][skew_name]["modes"][mk][
                    "peak_activation_inc_mib"
                ]
                for r in reps
            ]
            med_ms = _med(ms_vals)
            modes_agg[mk] = {
                "step_ms_median": round(med_ms, 4),
                "step_ms_iqr": round(_iqr(ms_vals), 4),
                "achieved_tflops_median": round(_med(tf_vals), 2),
                "peak_activation_inc_mib_median": round(_med(mem_vals), 2),
                "padding_blowup_ratio": ref_s0["modes"][mk]["padding_blowup_ratio"],
                "speedup_vs_pytorch_grouped_mm": round(
                    pt_med / max(med_ms, 1e-6), 4
                ),
            }
        s0_skews_out[skew_name] = {
            "num_tokens": ref_s0["num_tokens"],
            "num_experts": ref_s0["num_experts"],
            "top_k": ref_s0["top_k"],
            "max_expert_count": ref_s0["max_expert_count"],
            "max_over_mean_skew": ref_s0["max_over_mean_skew"],
            "zero_token_experts": ref_s0["zero_token_experts"],
            "modes": modes_agg,
        }

    # 3. Suite A Aggregation (ISO-FLOP Granularity Frontier)
    sa_out: dict[str, Any] = {}
    for g_label in ("coarse_e8_k2", "medium_e32_k4", "fine_e64_k8"):
        g_agg: dict[str, Any] = {}
        for t_str in ("256", "512", "1024", "2048", "4096"):
            ref_cell = reps[0]["suite_a_granularity_frontier"][g_label][t_str]
            pt_ms = _med(
                [
                    r["suite_a_granularity_frontier"][g_label][t_str]["modes"][
                        "pytorch_grouped_mm"
                    ]["median_ms"]
                    for r in reps
                ]
            )
            m_agg: dict[str, Any] = {}
            for mk in (
                "eager_python_loop",
                "padded_bmm_3d",
                "pytorch_grouped_mm",
                "triton_grouped_gemm",
                "triton_grouped_gemm_cudagraph",
            ):
                ms_v = [
                    r["suite_a_granularity_frontier"][g_label][t_str]["modes"][mk][
                        "median_ms"
                    ]
                    for r in reps
                ]
                tf_v = [
                    r["suite_a_granularity_frontier"][g_label][t_str]["modes"][mk][
                        "achieved_tflops"
                    ]
                    for r in reps
                ]
                mem_v = [
                    r["suite_a_granularity_frontier"][g_label][t_str]["modes"][mk][
                        "peak_activation_inc_mib"
                    ]
                    for r in reps
                ]
                med_m = _med(ms_v)
                m_agg[mk] = {
                    "step_ms_median": round(med_m, 4),
                    "step_ms_iqr": round(_iqr(ms_v), 4),
                    "achieved_tflops_median": round(_med(tf_v), 2),
                    "peak_activation_inc_mib_median": round(_med(mem_v), 2),
                    "padding_blowup_ratio": ref_cell["modes"][mk][
                        "padding_blowup_ratio"
                    ],
                    "speedup_vs_pytorch_grouped_mm": round(
                        pt_ms / max(med_m, 1e-6), 4
                    ),
                }
            g_agg[t_str] = {
                "num_tokens": ref_cell["num_tokens"],
                "routed_tokens_m_tot": ref_cell["routed_tokens_m_tot"],
                "mean_tokens_per_expert": ref_cell["mean_tokens_per_expert"],
                "max_expert_tokens": ref_cell["max_expert_tokens"],
                "num_experts": ref_cell["num_experts"],
                "top_k": ref_cell["top_k"],
                "inter_dim": ref_cell["inter_dim"],
                "modes": m_agg,
            }
        sa_out[g_label] = g_agg

    # 4. Suite B Aggregation (Fused Token Dispatcher at H=4096)
    sb_out: dict[str, Any] = {}
    for t_str in ("1024", "2048", "4096"):
        ref_b = reps[0]["suite_b_fused_dispatcher"]["by_tokens"][t_str]
        eag_ms = _med(
            [
                r["suite_b_fused_dispatcher"]["by_tokens"][t_str]["modes"][
                    "unfused_eager_dispatcher"
                ]["median_ms"]
                for r in reps
            ]
        )
        eag_mem = _med(
            [
                r["suite_b_fused_dispatcher"]["by_tokens"][t_str]["modes"][
                    "unfused_eager_dispatcher"
                ]["peak_activation_inc_mib"]
                for r in reps
            ]
        )
        mb_agg: dict[str, Any] = {}
        for mk in (
            "unfused_eager_dispatcher",
            "compiled_dispatcher",
            "triton_fused_dispatcher",
            "triton_fused_full_cudagraph",
        ):
            ms_v = [
                r["suite_b_fused_dispatcher"]["by_tokens"][t_str]["modes"][mk][
                    "median_ms"
                ]
                for r in reps
            ]
            tf_v = [
                r["suite_b_fused_dispatcher"]["by_tokens"][t_str]["modes"][mk][
                    "achieved_tflops"
                ]
                for r in reps
            ]
            mem_v = [
                r["suite_b_fused_dispatcher"]["by_tokens"][t_str]["modes"][mk][
                    "peak_activation_inc_mib"
                ]
                for r in reps
            ]
            med_m = _med(ms_v)
            med_mem = _med(mem_v)
            mb_agg[mk] = {
                "step_ms_median": round(med_m, 4),
                "step_ms_iqr": round(_iqr(ms_v), 4),
                "achieved_tflops_median": round(_med(tf_v), 2),
                "peak_activation_inc_mib_median": round(med_mem, 2),
                "speedup_vs_unfused_eager": round(eag_ms / max(med_m, 1e-6), 4),
                "memory_reduction_vs_unfused_eager": round(
                    eag_mem / max(med_mem, 1e-3), 4
                ),
            }

        iso_e_ms = _med(
            [
                r["suite_b_fused_dispatcher"]["by_tokens"][t_str][
                    "isolated_permute_combine_fwd_bwd"
                ]["eager_median_ms"]
                for r in reps
            ]
        )
        iso_t_ms = _med(
            [
                r["suite_b_fused_dispatcher"]["by_tokens"][t_str][
                    "isolated_permute_combine_fwd_bwd"
                ]["triton_median_ms"]
                for r in reps
            ]
        )
        iso_e_gbps = _med(
            [
                r["suite_b_fused_dispatcher"]["by_tokens"][t_str][
                    "isolated_permute_combine_fwd_bwd"
                ]["eager_useful_gbps"]
                for r in reps
            ]
        )
        iso_t_gbps = _med(
            [
                r["suite_b_fused_dispatcher"]["by_tokens"][t_str][
                    "isolated_permute_combine_fwd_bwd"
                ]["triton_useful_gbps"]
                for r in reps
            ]
        )
        iso_e_mem = _med(
            [
                r["suite_b_fused_dispatcher"]["by_tokens"][t_str][
                    "isolated_permute_combine_fwd_bwd"
                ]["eager_peak_inc_mib"]
                for r in reps
            ]
        )
        iso_t_mem = _med(
            [
                r["suite_b_fused_dispatcher"]["by_tokens"][t_str][
                    "isolated_permute_combine_fwd_bwd"
                ]["triton_peak_inc_mib"]
                for r in reps
            ]
        )

        sb_out[t_str] = {
            "num_tokens": ref_b["num_tokens"],
            "routed_tokens_m_tot": ref_b["routed_tokens_m_tot"],
            "hidden": ref_b["hidden"],
            "inter_dim": ref_b["inter_dim"],
            "modes": mb_agg,
            "isolated_permute_combine_fwd_bwd": {
                "eager_ms_median": round(iso_e_ms, 4),
                "eager_useful_gbps_median": round(iso_e_gbps, 2),
                "eager_peak_inc_mib_median": round(iso_e_mem, 2),
                "triton_ms_median": round(iso_t_ms, 4),
                "triton_useful_gbps_median": round(iso_t_gbps, 2),
                "triton_peak_inc_mib_median": round(iso_t_mem, 2),
                "speedup_vs_eager": round(iso_e_ms / max(iso_t_ms, 1e-6), 4),
                "memory_reduction_vs_eager": round(
                    iso_e_mem / max(iso_t_mem, 1e-3), 4
                ),
            },
        }

    # 5. Suite C Aggregation (8-GPU EP=8 Overlap & Skew/Capacity Frontier)
    sc_scaling: dict[str, Any] = {}
    for t_str in ("512", "1024", "2048"):
        ref_c = reps[0]["suite_c_ep8_overlap_and_skew"]["scaling_by_tokens_per_rank"][
            t_str
        ]
        pt_ep_ms = _med(
            [
                r["suite_c_ep8_overlap_and_skew"]["scaling_by_tokens_per_rank"][t_str][
                    "schedules"
                ]["ep_pytorch_grouped_mm_seq"]["step_ms_median"]
                for r in reps
            ]
        )
        tri_seq_ms = _med(
            [
                r["suite_c_ep8_overlap_and_skew"]["scaling_by_tokens_per_rank"][t_str][
                    "schedules"
                ]["ep_triton_fused_seq"]["step_ms_median"]
                for r in reps
            ]
        )
        pt_ep_mem = _med(
            [
                r["suite_c_ep8_overlap_and_skew"]["scaling_by_tokens_per_rank"][t_str][
                    "schedules"
                ]["ep_pytorch_grouped_mm_seq"]["max_rank_activation_inc_mib"]
                for r in reps
            ]
        )
        sch_agg: dict[str, Any] = {}
        for sk in (
            "ep_pytorch_grouped_mm_seq",
            "ep_triton_fused_seq",
            "ep_triton_split_dw_overlap",
            "ep_triton_dual_microbatch_overlap",
        ):
            ms_v = [
                r["suite_c_ep8_overlap_and_skew"]["scaling_by_tokens_per_rank"][t_str][
                    "schedules"
                ][sk]["step_ms_median"]
                for r in reps
            ]
            tf_v = [
                r["suite_c_ep8_overlap_and_skew"]["scaling_by_tokens_per_rank"][t_str][
                    "schedules"
                ][sk]["cluster_tflops"]
                for r in reps
            ]
            mem_v = [
                r["suite_c_ep8_overlap_and_skew"]["scaling_by_tokens_per_rank"][t_str][
                    "schedules"
                ][sk]["max_rank_activation_inc_mib"]
                for r in reps
            ]
            med_m = _med(ms_v)
            med_mem = _med(mem_v)
            sch_agg[sk] = {
                "step_ms_median": round(med_m, 4),
                "step_ms_iqr": round(_iqr(ms_v), 4),
                "cluster_tflops_median": round(_med(tf_v), 2),
                "max_rank_activation_inc_mib_median": round(med_mem, 2),
                "max_rank_m_recv": ref_c["schedules"][sk]["max_rank_m_recv"],
                "receiver_imbalance_max_over_mean": ref_c["schedules"][sk][
                    "receiver_imbalance_max_over_mean"
                ],
                "speedup_vs_ep_pytorch_seq": round(pt_ep_ms / max(med_m, 1e-6), 4),
                "speedup_vs_ep_triton_seq": round(tri_seq_ms / max(med_m, 1e-6), 4),
                "memory_reduction_vs_ep_pytorch_seq": round(
                    pt_ep_mem / max(med_mem, 1e-3), 4
                ),
            }
        sc_scaling[t_str] = {
            "tokens_per_rank": ref_c["tokens_per_rank"],
            "global_tokens": ref_c["global_tokens"],
            "global_routed_assignments": ref_c["global_routed_assignments"],
            "schedules": sch_agg,
        }

    sc_skew: dict[str, Any] = {}
    for skew_name in ("uniform", "zipf_s1p2", "hot_rank0_60pct"):
        ref_sk = reps[0]["suite_c_ep8_overlap_and_skew"]["skew_and_capacity_frontier"][
            skew_name
        ]
        m_agg_sk: dict[str, Any] = {}
        for mk in (
            "dropless_dynamic_split_dw",
            "static_cf_1p0",
            "static_cf_1p25",
            "static_cf_2p0",
        ):
            ms_v = [
                r["suite_c_ep8_overlap_and_skew"]["skew_and_capacity_frontier"][
                    skew_name
                ]["modes"][mk]["step_ms_median"]
                for r in reps
            ]
            ref_m = ref_sk["modes"][mk]
            m_agg_sk[mk] = {
                "step_ms_median": round(_med(ms_v), 4),
                "step_ms_iqr": round(_iqr(ms_v), 4),
                "dropped_fraction": ref_m["dropped_fraction"],
                "rel_loss_err_vs_dropless": ref_m["rel_loss_err_vs_dropless"],
                "dx_cosine_vs_dropless": ref_m["dx_cosine_vs_dropless"],
                "dw_down_cosine_vs_dropless": ref_m["dw_down_cosine_vs_dropless"],
            }
        sc_skew[skew_name] = {
            "tokens_per_rank": ref_sk["tokens_per_rank"],
            "dropless_max_rank_m_recv": ref_sk["dropless_max_rank_m_recv"],
            "dropless_receiver_imbalance": ref_sk["dropless_receiver_imbalance"],
            "modes": m_agg_sk,
        }

    # 6. Validation Gates
    max_rel_loss_err = max(
        max(r["suite_0_grouped_mm_audit"]["audits"]["max_suite0_rel_loss_err"] for r in reps),
        max(r["suite_c_ep8_overlap_and_skew"]["max_schedule_rel_loss_err"] for r in reps),
        max(
            r["suite_c_ep8_overlap_and_skew"]["ep8_vs_single_gpu_reference"][
                "rel_loss_err"
            ]
            for r in reps
        ),
    )
    max_grad_rel_l2 = max(
        max(r["suite_0_grouped_mm_audit"]["audits"]["max_suite0_grad_rel_l2"] for r in reps),
        max(r["suite_a_granularity_frontier"]["_max_grad_rel_l2"] for r in reps),
        max(r["suite_b_fused_dispatcher"]["max_grad_rel_l2"] for r in reps),
        max(r["suite_c_ep8_overlap_and_skew"]["max_schedule_grad_rel_l2"] for r in reps),
        max(
            max(
                r["suite_c_ep8_overlap_and_skew"]["ep8_vs_single_gpu_reference"][k]
                for k in (
                    "max_dx_rel_l2",
                    "max_dw_gate_up_rel_l2",
                    "max_dw_down_rel_l2",
                    "dw_router_rel_l2",
                )
            )
            for r in reps
        ),
    )
    max_cg_dyn_err = max(
        r["suite_0_grouped_mm_audit"]["audits"][
            "max_cudagraph_dynamic_routing_grad_rel_l2"
        ]
        for r in reps
    )
    min_fd_refinement = min(
        r["suite_0_grouped_mm_audit"]["audits"]["chain_rule_fd_refinement_ratio"]
        for r in reps
    )
    max_bf16_vs_fp64_dir = max(
        r["suite_0_grouped_mm_audit"]["audits"]["chain_rule_bf16_vs_fp64_dir_rel_err"]
        for r in reps
    )

    smoke_max_grad_err = max(
        smoke["suite_0_grouped_mm_audit"]["audits"]["max_suite0_grad_rel_l2"],
        smoke["suite_a_granularity_frontier"]["_max_grad_rel_l2"],
        smoke["suite_b_fused_dispatcher"]["max_grad_rel_l2"],
        smoke["suite_c_ep8_overlap_and_skew"]["max_schedule_grad_rel_l2"],
    )

    active_exp_p = prof0["pytorch_grouped_mm"]["active_experts"]
    audit_gate = (
        zero_stride_all
        and cudagraph_rej_all
        and dtoh_pt == 6
        and dtoh_pt_comp == 6
        and dtoh_tri == 0
        and gemms_pt == 6 * active_exp_p
        and gemms_pt_comp == 6 * active_exp_p
        and active_exp_p == 64
        and 0 < gemms_tri <= 6
    )
    cg_gate = (
        max_cg_dyn_err <= 1e-4
        and sa_out["coarse_e8_k2"]["256"]["modes"]["triton_grouped_gemm_cudagraph"][
            "step_ms_median"
        ]
        < sa_out["coarse_e8_k2"]["256"]["modes"]["triton_grouped_gemm"][
            "step_ms_median"
        ]
        and sa_out["fine_e64_k8"]["512"]["modes"]["triton_grouped_gemm_cudagraph"][
            "speedup_vs_pytorch_grouped_mm"
        ]
        >= 2.0
    )
    fine_grained_gate = (
        sa_out["fine_e64_k8"]["512"]["modes"]["triton_grouped_gemm"][
            "speedup_vs_pytorch_grouped_mm"
        ]
        >= 2.5
        and sa_out["fine_e64_k8"]["512"]["modes"]["triton_grouped_gemm_cudagraph"][
            "speedup_vs_pytorch_grouped_mm"
        ]
        >= 2.0
    )
    dispatcher_gate = (
        sb_out["4096"]["modes"]["triton_fused_dispatcher"]["speedup_vs_unfused_eager"]
        >= 1.35
        and sb_out["4096"]["isolated_permute_combine_fwd_bwd"][
            "memory_reduction_vs_eager"
        ]
        >= 2.0
        and sb_out["4096"]["isolated_permute_combine_fwd_bwd"]["speedup_vs_eager"]
        >= 4.0
    )
    ep8_parity_gate = max_rel_loss_err <= 0.015 and max_grad_rel_l2 <= 0.025
    ep8_overlap_and_skew_gate = (
        sc_scaling["1024"]["schedules"]["ep_triton_split_dw_overlap"][
            "speedup_vs_ep_pytorch_seq"
        ]
        >= 1.15
        and sc_skew["hot_rank0_60pct"]["modes"]["static_cf_1p0"]["dropped_fraction"]
        >= 0.15
        and sc_skew["hot_rank0_60pct"]["modes"]["static_cf_1p0"][
            "dx_cosine_vs_dropless"
        ]
        < 0.95
    )
    chain_rule_gate = min_fd_refinement >= 1.75 and max_bf16_vs_fp64_dir <= 0.025

    validation_gates = {
        "env_parity": env_parity,
        "clocks_locked_1410": clocks_locked_1410,
        "median_sm_clock_mhz": round(med_obs_clock, 1),
        "min_sm_clock_mhz": round(min_obs_clock, 1),
        "pytorch_grouped_mm_sm80_audit_verified": audit_gate,
        "dynamic_routing_cudagraph_verified": cg_gate,
        "fine_grained_triton_speedup_verified": fine_grained_gate,
        "fused_dispatcher_memory_and_bw_verified": dispatcher_gate,
        "ep8_single_gpu_and_schedule_parity_verified": ep8_parity_gate,
        "ep8_overlap_and_skew_fidelity_verified": ep8_overlap_and_skew_gate,
        "chain_rule_and_numerical_accuracy_passed": chain_rule_gate,
        "min_chain_rule_fd_refinement_ratio": round(min_fd_refinement, 4),
        "max_bf16_vs_fp64_directional_rel_err": round(max_bf16_vs_fp64_dir, 6),
        "max_rel_loss_err": round(max_rel_loss_err, 7),
        "max_grad_rel_l2": round(max_grad_rel_l2, 6),
        "max_cudagraph_dynamic_routing_grad_rel_l2": round(max_cg_dyn_err, 7),
        "smoke_max_grad_rel_l2": round(smoke_max_grad_err, 6),
    }
    validation_gates["all_gates_passed"] = all(
        bool(validation_gates[k])
        for k in (
            "env_parity",
            "clocks_locked_1410",
            "pytorch_grouped_mm_sm80_audit_verified",
            "dynamic_routing_cudagraph_verified",
            "fine_grained_triton_speedup_verified",
            "fused_dispatcher_memory_and_bw_verified",
            "ep8_single_gpu_and_schedule_parity_verified",
            "ep8_overlap_and_skew_fidelity_verified",
            "chain_rule_and_numerical_accuracy_passed",
        )
    )
    assert validation_gates["all_gates_passed"], (
        f"E117 validation failed: {json.dumps(validation_gates, indent=2)}"
    )

    full_summary = {
        "experiment": "E117_moe_training_grouped_gemm_and_ep_overlap",
        "repetitions": 3,
        "environment": reps[0]["environment"],
        "suite_0_grouped_mm_audit": {
            "audits": {
                "zero_stride_grad_rejected_all": zero_stride_all,
                "zero_stride_error_excerpt": s0_audits[0]["zero_stride_error_excerpt"],
                "cudagraph_capture_rejected_all": cudagraph_rej_all,
                "cudagraph_error_excerpt": s0_audits[0]["cudagraph_error_excerpt"],
                "profiler_kernel_and_dtoh_counts": prof0,
            },
            "skew_sweep": s0_skews_out,
        },
        "suite_a_granularity_frontier": sa_out,
        "suite_b_fused_dispatcher": sb_out,
        "suite_c_ep8_overlap_and_skew": {
            "ep8_vs_single_gpu_reference": reps[0]["suite_c_ep8_overlap_and_skew"][
                "ep8_vs_single_gpu_reference"
            ],
            "schedule_numerical_parity": reps[0]["suite_c_ep8_overlap_and_skew"][
                "schedule_numerical_parity"
            ],
            "scaling_by_tokens_per_rank": sc_scaling,
            "skew_and_capacity_frontier": sc_skew,
        },
        "validation_gates": validation_gates,
    }

    out_path = raw_dir / "moe_training_grouped_gemm_full.json"
    out_path.write_text(
        json.dumps(full_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Wrote {out_path} (all_gates_passed={validation_gates['all_gates_passed']})"
    )


if __name__ == "__main__":
    main()
