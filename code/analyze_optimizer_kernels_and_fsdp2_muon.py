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
        description="Deterministic 3-repetition analyzer and 9-gate validator for E118"
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.root / "results" / "raw" / "optimizer_kernels_and_fsdp2_muon"
    smoke = json.loads(
        (raw_dir / "smoke_optimizer_kernels_and_fsdp2_muon.json").read_text(
            encoding="utf-8"
        )
    )
    reps = [
        json.loads(
            (raw_dir / f"rep{i}_optimizer_kernels_and_fsdp2_muon.json").read_text(
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
    env_parity_and_8gpu = len(set(envs)) == 1 and reps[0]["world_size"] == 8

    all_clocks: list[float] = []
    for r in reps:
        for sc in r["suite_a_adamw_roofline"].values():
            for m in sc["modes"].values():
                if m.get("sm_clock_after_mhz", 0.0) > 0:
                    all_clocks.append(float(m["sm_clock_after_mhz"]))
    min_obs_clock = min(all_clocks) if all_clocks else 0.0
    med_obs_clock = _med(all_clocks) if all_clocks else 0.0
    clocks_locked_1410 = med_obs_clock >= 1380.0 and min_obs_clock >= 1260.0

    # 2. Suite 0 Aggregation
    s0_list = [r["suite_0_audits"] for r in reps]
    gd0 = s0_list[0]["grad_dtype_audit"]
    all_3_grad_dtype_rejected = all(
        s["grad_dtype_audit"]["backends"]["single_tensor"][
            "rejected_fp32w_bf16_grad_dtype"
        ]
        and s["grad_dtype_audit"]["backends"]["foreach"][
            "rejected_fp32w_bf16_grad_dtype"
        ]
        and s["grad_dtype_audit"]["backends"]["fused"][
            "rejected_fp32w_bf16_grad_dtype"
        ]
        for s in s0_list
    )
    max_tri_gd_rel_l2 = max(
        float(s["grad_dtype_audit"]["triton_fp32w_bf16g_vs_fp32_ref_rel_l2"])
        for s in s0_list
    )
    min_int8_cos = min(
        float(s["grad_dtype_audit"]["triton_int8mv_step1_delta_cosine"])
        for s in s0_list
    )

    sr0 = s0_list[0]["sub_ulp_stochastic_rounding_proof"]
    max_sr_delta_rel_err = max(
        float(
            s["sub_ulp_stochastic_rounding_proof"]["sr_accumulated_delta_rel_err"]
        )
        for s in s0_list
    )
    min_rtn_stagnant = min(
        float(s["sub_ulp_stochastic_rounding_proof"]["rtn_stagnant_fraction"])
        for s in s0_list
    )

    m1g_pt_ms = _med(
        [
            float(s["single_gpu_muon_launch_audit"]["pytorch_muon_median_ms"])
            for s in s0_list
        ]
    )
    m1g_bt_ms = _med(
        [
            float(
                s["single_gpu_muon_launch_audit"]["batched_triton_muon_median_ms"]
            )
            for s in s0_list
        ]
    )
    m1g_speedup = m1g_pt_ms / max(m1g_bt_ms, 1e-6)
    m1g_max_err = max(
        float(
            s["single_gpu_muon_launch_audit"]["max_param_rel_l2_vs_pytorch_muon"]
        )
        for s in s0_list
    )
    m1g_pt_launches = int(
        s0_list[0]["single_gpu_muon_launch_audit"]["pytorch_muon_cuda_launches"]
    )
    m1g_bt_launches = int(
        s0_list[0]["single_gpu_muon_launch_audit"][
            "batched_triton_muon_cuda_launches"
        ]
    )
    m1g_foreach_rej = all(
        bool(s["single_gpu_muon_launch_audit"]["foreach_rejected"]) for s in s0_list
    )

    suite_0_summary = {
        "grad_dtype_audit": {
            "all_3_pytorch_adamw_backends_rejected": all_3_grad_dtype_rejected,
            "backends": gd0["backends"],
            "triton_fp32w_bf16g_max_rel_l2": round(max_tri_gd_rel_l2, 7),
            "triton_int8mv_min_step1_delta_cosine": round(min_int8_cos, 6),
        },
        "sub_ulp_stochastic_rounding_proof": {
            **sr0,
            "max_sr_accumulated_delta_rel_err_across_reps": round(
                max_sr_delta_rel_err, 5
            ),
        },
        "single_gpu_muon_launch_audit": {
            "num_2d_matrices": s0_list[0]["single_gpu_muon_launch_audit"][
                "num_2d_matrices"
            ],
            "foreach_rejected": m1g_foreach_rej,
            "pytorch_muon_cuda_launches": m1g_pt_launches,
            "batched_triton_muon_cuda_launches": m1g_bt_launches,
            "launch_reduction_factor": round(
                float(m1g_pt_launches) / max(float(m1g_bt_launches), 1.0), 2
            ),
            "max_param_rel_l2_vs_pytorch_muon": round(m1g_max_err, 7),
            "pytorch_muon_ms_median": round(m1g_pt_ms, 4),
            "batched_triton_muon_ms_median": round(m1g_bt_ms, 4),
            "speedup_vs_pytorch_muon": round(m1g_speedup, 4),
        },
    }

    # 3. Suite A Aggregation
    suite_a_summary: dict[str, Any] = {}
    for scale_k, scale_ref in reps[0]["suite_a_adamw_roofline"].items():
        fused_med_ms = _med(
            [
                float(
                    r["suite_a_adamw_roofline"][scale_k]["modes"][
                        "pytorch_fused_fp32_clip"
                    ]["median_ms"]
                )
                for r in reps
            ]
        )
        single_med_ms = _med(
            [
                float(
                    r["suite_a_adamw_roofline"][scale_k]["modes"][
                        "pytorch_single_tensor_fp32_clip"
                    ]["median_ms"]
                )
                for r in reps
            ]
        )
        modes_out: dict[str, Any] = {}
        for m_k, m_ref in scale_ref["modes"].items():
            ms_vals = [
                float(r["suite_a_adamw_roofline"][scale_k]["modes"][m_k]["median_ms"])
                for r in reps
            ]
            gp_vals = [
                float(
                    r["suite_a_adamw_roofline"][scale_k]["modes"][m_k]["gparams_per_s"]
                )
                for r in reps
            ]
            bw_vals = [
                float(
                    r["suite_a_adamw_roofline"][scale_k]["modes"][m_k][
                        "achieved_hbm_gbps"
                    ]
                )
                for r in reps
            ]
            m_med = _med(ms_vals)
            modes_out[m_k] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_vals), 4),
                "gparams_per_s_median": round(_med(gp_vals), 3),
                "achieved_hbm_gbps_median": round(_med(bw_vals), 2),
                "physical_hbm_bytes_per_param": m_ref[
                    "physical_hbm_bytes_per_param"
                ],
                "resident_bytes_per_param": m_ref["resident_bytes_per_param"],
                "optimizer_state_bytes_per_param": m_ref[
                    "optimizer_state_bytes_per_param"
                ],
                "speedup_vs_pytorch_fused_fp32_clip": round(
                    fused_med_ms / max(m_med, 1e-6), 4
                ),
                "speedup_vs_pytorch_single_tensor": round(
                    single_med_ms / max(m_med, 1e-6), 4
                ),
            }
        suite_a_summary[scale_k] = {
            "num_tensors": scale_ref["num_tensors"],
            "tensor_shape": scale_ref["tensor_shape"],
            "total_params": scale_ref["total_params"],
            "modes": modes_out,
        }

    # 4. Suite B Aggregation
    sb_ref = reps[0]["suite_b_convergence_and_stagnation"]
    sb_opts: dict[str, Any] = {}
    for opt_k in sb_ref["optimizers"]:
        init_l = _med(
            [
                float(
                    r["suite_b_convergence_and_stagnation"]["optimizers"][opt_k][
                        "initial_loss"
                    ]
                )
                for r in reps
            ]
        )
        mid_l = _med(
            [
                float(
                    r["suite_b_convergence_and_stagnation"]["optimizers"][opt_k][
                        "mid_loss"
                    ]
                )
                for r in reps
            ]
        )
        fin_l = _med(
            [
                float(
                    r["suite_b_convergence_and_stagnation"]["optimizers"][opt_k][
                        "final_loss"
                    ]
                )
                for r in reps
            ]
        )
        red_r = _med(
            [
                float(
                    r["suite_b_convergence_and_stagnation"]["optimizers"][opt_k][
                        "loss_reduction_ratio"
                    ]
                )
                for r in reps
            ]
        )
        stag_f = _med(
            [
                float(
                    r["suite_b_convergence_and_stagnation"]["optimizers"][opt_k][
                        "mean_step_stagnant_param_fraction"
                    ]
                )
                for r in reps
            ]
        )
        tail_stag_f = _med(
            [
                float(
                    r["suite_b_convergence_and_stagnation"]["optimizers"][opt_k][
                        "tail_step_stagnant_param_fraction"
                    ]
                )
                for r in reps
            ]
        )
        sb_opts[opt_k] = {
            "initial_loss_median": round(init_l, 6),
            "mid_loss_median": round(mid_l, 6),
            "final_loss_median": round(fin_l, 6),
            "loss_reduction_ratio_median": round(red_r, 4),
            "mean_step_stagnant_param_fraction_median": round(stag_f, 5),
            "tail_step_stagnant_param_fraction_median": round(tail_stag_f, 5),
        }
    suite_b_summary = {
        "steps": sb_ref["steps"],
        "lr_peak": sb_ref["lr_peak"],
        "lr_min": sb_ref["lr_min"],
        "optimizers": sb_opts,
    }

    # 5. Suite C Aggregation
    sc_list = [r["suite_c_fsdp2_muon_and_adamw"] for r in reps]
    ver_c = {
        "max_pytorch_dtensor_muon_rel_l2": round(
            max(
                float(
                    c["muon_8gpu_vs_1gpu_verification"]["pytorch_dtensor_muon_rel_l2"]
                )
                for c in sc_list
            ),
            7,
        ),
        "max_owner_seq_muon_rel_l2": round(
            max(
                float(c["muon_8gpu_vs_1gpu_verification"]["owner_seq_muon_rel_l2"])
                for c in sc_list
            ),
            7,
        ),
        "max_owner_pipelined_muon_rel_l2": round(
            max(
                float(
                    c["muon_8gpu_vs_1gpu_verification"]["owner_pipelined_muon_rel_l2"]
                )
                for c in sc_list
            ),
            7,
        ),
        "max_owner_pipelined_update_delta_rel_l2": round(
            max(
                float(
                    c["muon_8gpu_vs_1gpu_verification"][
                        "owner_pipelined_update_delta_rel_l2"
                    ]
                )
                for c in sc_list
            ),
            7,
        ),
        "min_shard_local_wrong_param_rel_l2": round(
            min(
                float(
                    c["muon_8gpu_vs_1gpu_verification"][
                        "shard_local_wrong_param_rel_l2"
                    ]
                )
                for c in sc_list
            ),
            6,
        ),
        "min_shard_local_wrong_update_delta_rel_l2": round(
            min(
                float(
                    c["muon_8gpu_vs_1gpu_verification"][
                        "shard_local_wrong_update_delta_rel_l2"
                    ]
                )
                for c in sc_list
            ),
            5,
        ),
    }

    c1_ref = sc_list[0]["isolated_3p49b_fsdp2_optimizer_step"]
    dt_muon_c1_ms = _med(
        [
            float(
                c["isolated_3p49b_fsdp2_optimizer_step"]["modes"][
                    "fsdp2_pytorch_dtensor_muon"
                ]["step_ms_median"]
            )
            for c in sc_list
        ]
    )
    fused_c1_ms = _med(
        [
            float(
                c["isolated_3p49b_fsdp2_optimizer_step"]["modes"][
                    "fsdp2_adamw_fused_clip"
                ]["step_ms_median"]
            )
            for c in sc_list
        ]
    )
    c1_modes_out: dict[str, Any] = {}
    for mk, mref in c1_ref["modes"].items():
        ms_vals = [
            float(
                c["isolated_3p49b_fsdp2_optimizer_step"]["modes"][mk]["step_ms_median"]
            )
            for c in sc_list
        ]
        gp_vals = [
            float(
                c["isolated_3p49b_fsdp2_optimizer_step"]["modes"][mk][
                    "cluster_gparams_per_s"
                ]
            )
            for c in sc_list
        ]
        m_med = _med(ms_vals)
        c1_modes_out[mk] = {
            "step_ms_median": round(m_med, 4),
            "step_ms_iqr": round(_iqr(ms_vals), 4),
            "cluster_gparams_per_s_median": round(_med(gp_vals), 3),
            "nccl_collective_launches": int(mref["nccl_collective_launches"]),
            "per_rank_state_mib": mref["per_rank_state_mib"],
            "per_rank_param_grad_state_mib": mref["per_rank_param_grad_state_mib"],
            "speedup_vs_pytorch_dtensor_muon": round(
                dt_muon_c1_ms / max(m_med, 1e-6), 4
            ),
            "speedup_vs_fsdp2_adamw_fused": round(fused_c1_ms / max(m_med, 1e-6), 4),
        }

    e2e_ref = sc_list[0]["end_to_end_1p01b_fsdp2_training_step"]
    dt_e2e_ms = _med(
        [
            float(
                c["end_to_end_1p01b_fsdp2_training_step"]["modes"][
                    "e2e_fsdp2_pytorch_dtensor_muon"
                ]["total_step_ms_median"]
            )
            for c in sc_list
        ]
    )
    foreach_e2e_ms = _med(
        [
            float(
                c["end_to_end_1p01b_fsdp2_training_step"]["modes"][
                    "e2e_fsdp2_adamw_foreach"
                ]["total_step_ms_median"]
            )
            for c in sc_list
        ]
    )
    e2e_modes_out: dict[str, Any] = {}
    for mk, mref in e2e_ref["modes"].items():
        tot_vals = [
            float(
                c["end_to_end_1p01b_fsdp2_training_step"]["modes"][mk][
                    "total_step_ms_median"
                ]
            )
            for c in sc_list
        ]
        opt_vals = [
            float(
                c["end_to_end_1p01b_fsdp2_training_step"]["modes"][mk][
                    "opt_clip_ms_median"
                ]
            )
            for c in sc_list
        ]
        tps_vals = [
            float(
                c["end_to_end_1p01b_fsdp2_training_step"]["modes"][mk][
                    "cluster_tokens_per_s"
                ]
            )
            for c in sc_list
        ]
        pk_vals = [
            float(
                c["end_to_end_1p01b_fsdp2_training_step"]["modes"][mk][
                    "peak_allocated_mib"
                ]
            )
            for c in sc_list
        ]
        st_vals = [
            float(
                c["end_to_end_1p01b_fsdp2_training_step"]["modes"][mk][
                    "steady_post_step_mib"
                ]
            )
            for c in sc_list
        ]
        t_med = _med(tot_vals)
        o_med = _med(opt_vals)
        e2e_modes_out[mk] = {
            "total_step_ms_median": round(t_med, 4),
            "total_step_ms_iqr": round(_iqr(tot_vals), 4),
            "opt_clip_ms_median": round(o_med, 4),
            "opt_fraction_of_step": round(o_med / max(t_med, 1e-6), 4),
            "cluster_tokens_per_s_median": round(_med(tps_vals), 1),
            "peak_allocated_mib_median": round(_med(pk_vals), 2),
            "steady_post_step_mib_median": round(_med(st_vals), 2),
            "speedup_vs_pytorch_dtensor_muon": round(dt_e2e_ms / max(t_med, 1e-6), 4),
            "speedup_vs_adamw_foreach": round(foreach_e2e_ms / max(t_med, 1e-6), 4),
        }

    suite_c_summary = {
        "muon_8gpu_vs_1gpu_verification": ver_c,
        "isolated_3p49b_fsdp2_optimizer_step": {
            "num_layers": c1_ref["num_layers"],
            "hidden": c1_ref["hidden"],
            "inter_dim": c1_ref["inter_dim"],
            "total_params": c1_ref["total_params"],
            "modes": c1_modes_out,
        },
        "end_to_end_1p01b_fsdp2_training_step": {
            "num_layers": e2e_ref["num_layers"],
            "hidden": e2e_ref["hidden"],
            "inter_dim": e2e_ref["inter_dim"],
            "total_params": e2e_ref["total_params"],
            "global_tokens_per_step": e2e_ref["global_tokens_per_step"],
            "modes": e2e_modes_out,
        },
    }

    # 6. Validation Gates (9 Deterministic Gates)
    llm_modes = suite_a_summary["llm_1p07b"]["modes"]
    dt_nccl = c1_modes_out["fsdp2_pytorch_dtensor_muon"]["nccl_collective_launches"]
    own_nccl = c1_modes_out["fsdp2_owner_pipelined_muon"]["nccl_collective_launches"]

    gates = {
        "smoke_present": smoke.get("mode") == "smoke" and smoke.get("world_size") == 8,
        "env_parity_and_8gpu": env_parity_and_8gpu,
        "clocks_locked_1410": clocks_locked_1410,
        "pytorch_grad_dtype_adamw_audit": (
            all_3_grad_dtype_rejected
            and max_tri_gd_rel_l2 < 1e-3
            and min_int8_cos > 0.999
        ),
        "sub_ulp_sr_bf16_unbiased_proof": (
            min_rtn_stagnant == 1.0 and max_sr_delta_rel_err < 0.025
        ),
        "single_gpu_batched_muon_parity_and_speedup": (
            m1g_foreach_rej
            and m1g_pt_launches > 2000
            and (m1g_pt_launches / max(m1g_bt_launches, 1)) > 15.0
            and m1g_max_err < 5e-4
            and m1g_speedup > 1.15
        ),
        "suite_a_lowbit_triton_beats_fused_adamw": (
            llm_modes["triton_sr_bf16_inclip"]["speedup_vs_pytorch_fused_fp32_clip"]
            > 1.50
            and llm_modes["triton_fp32w_bf16g_bf16mv_inclip"][
                "speedup_vs_pytorch_fused_fp32_clip"
            ]
            > 1.50
            and llm_modes["triton_fp32w_bf16g_int8mv_inclip"][
                "speedup_vs_pytorch_fused_fp32_clip"
            ]
            > 1.15
            and llm_modes["triton_fp32w_bf16g_fp32mv_inclip"][
                "speedup_vs_pytorch_fused_fp32_clip"
            ]
            > 1.15
        ),
        "suite_b_sr_and_int8_converge_while_rtn_stagnates": (
            sb_opts["triton_bf16_rtn_stagnant"][
                "tail_step_stagnant_param_fraction_median"
            ]
            > 0.70
            and sb_opts["triton_sr_bf16"]["final_loss_median"]
            < 0.65 * sb_opts["triton_bf16_rtn_stagnant"]["final_loss_median"]
            and sb_opts["triton_sr_bf16"]["loss_reduction_ratio_median"] > 2.5
            and sb_opts["triton_fp32w_int8mv"]["loss_reduction_ratio_median"] > 2.5
            and sb_opts["batched_muon_plus_adamw"]["loss_reduction_ratio_median"] > 2.5
        ),
        "fsdp2_muon_parity_and_e2e_wins": (
            ver_c["max_owner_pipelined_muon_rel_l2"] < 5e-4
            and ver_c["min_shard_local_wrong_update_delta_rel_l2"] > 0.25
            and (dt_nccl / max(own_nccl, 1)) > 100.0
            and c1_modes_out["fsdp2_owner_pipelined_muon"][
                "speedup_vs_pytorch_dtensor_muon"
            ]
            > 3.5
            and e2e_modes_out["e2e_fsdp2_owner_pipelined_muon"][
                "speedup_vs_pytorch_dtensor_muon"
            ]
            > 1.5
            and e2e_modes_out["e2e_fsdp2_triton_sr_bf16"][
                "steady_post_step_mib_median"
            ]
            < e2e_modes_out["e2e_fsdp2_adamw_fused"]["steady_post_step_mib_median"]
        ),
    }
    all_passed = all(gates.values())

    full_out = {
        "experiment": "E118_optimizer_kernels_and_fsdp2_muon",
        "repetitions": 3,
        "world_size": 8,
        "environment": reps[0]["environment"],
        "clock_telemetry": {
            "median_sm_clock_mhz": round(med_obs_clock, 1),
            "min_sm_clock_mhz": round(min_obs_clock, 1),
        },
        "validation_gates": {
            "all_gates_passed": all_passed,
            "gates": gates,
        },
        "suite_0_audits": suite_0_summary,
        "suite_a_adamw_roofline": suite_a_summary,
        "suite_b_convergence_and_stagnation": suite_b_summary,
        "suite_c_fsdp2_muon_and_adamw": suite_c_summary,
        "rep_elapsed_seconds": [r["elapsed_s"] for r in reps],
    }

    out_path = raw_dir / "optimizer_kernels_and_fsdp2_muon_full.json"
    out_path.write_text(
        json.dumps(full_out, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Wrote {out_path} (all_gates_passed={all_passed})")
    if not all_passed:
        failed = [k for k, v in gates.items() if not v]
        raise SystemExit(f"Validation gates failed: {failed}")


if __name__ == "__main__":
    main()
