#!/usr/bin/env python3
"""Deterministic 3-repetition analyzer and 9-gate validator for E123:
Block-Hadamard INT8 Tensor-Core Training (SwitchBack / Jetfire / COAT / Halo),
COAT Compressed INT8 Activations, & 8-GPU Quantized-AllGather FSDP2 Training.
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
        description="Deterministic 3-repetition analyzer and 9-gate validator for E123"
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.root / "results" / "raw" / "int8_quantized_training"
    smoke = json.loads(
        (raw_dir / "smoke_int8_quantized_training.json").read_text(encoding="utf-8")
    )
    reps = [
        json.loads(
            (raw_dir / f"rep{i}_int8_quantized_training.json").read_text(
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
        for sc in r["suite_a_linear_3gemm_scaling"][
            "linear_3gemm_token_scaling"
        ].values():
            for m in sc["modes"].values():
                if m.get("sm_clock_after_mhz", 0.0) > 0:
                    all_clocks.append(float(m["sm_clock_after_mhz"]))
                if m.get("power_after_w", 0.0) > 0:
                    all_powers.append(float(m["power_after_w"]))
        for sc in r["suite_b_coat_swiglu_mlp_stack"]["token_scaling"].values():
            for m in sc["modes"].values():
                if m.get("sm_clock_after_mhz", 0.0) > 0:
                    all_clocks.append(float(m["sm_clock_after_mhz"]))
                if m.get("power_after_w", 0.0) > 0:
                    all_powers.append(float(m["power_after_w"]))
        for m in r["suite_c_quantized_fsdp2_8gpu"]["fsdp2_8gpu_training_scaling"][
            "modes"
        ].values():
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
    rf_ref = s0_list[0]["sm80_int8_tensor_core_roofline"]
    bf16_rf_ms = _med(
        [
            float(s["sm80_int8_tensor_core_roofline"]["bf16_cublas"]["median_ms"])
            for s in s0_list
        ]
    )
    rf_agg: dict[str, Any] = {"shape": rf_ref["shape"], "kernels": {}}
    for kk in rf_ref:
        if kk == "shape":
            continue
        ms_v = [
            float(s["sm80_int8_tensor_core_roofline"][kk]["median_ms"])
            for s in s0_list
        ]
        tf_v = [
            float(s["sm80_int8_tensor_core_roofline"][kk]["tops_or_tflops"])
            for s in s0_list
        ]
        m_med = _med(ms_v)
        rf_agg["kernels"][kk] = {
            "ms_median": round(m_med, 4),
            "ms_iqr": round(_iqr(ms_v), 4),
            "tops_or_tflops_median": round(_med(tf_v), 2),
            "speedup_vs_bf16_cublas": round(bf16_rf_ms / max(m_med, 1e-6), 4),
        }

    qa_naive = _med(
        [
            float(
                s["sm80_ptx_quantizer_ablation"][
                    "naive_fp32_atomic_philox10_spilling_ms"
                ]
            )
            for s in s0_list
        ]
    )
    qa_fast = _med(
        [
            float(
                s["sm80_ptx_quantizer_ablation"][
                    "split_s32_bitcast_splitmix_zero_spill_ms"
                ]
            )
            for s in s0_list
        ]
    )
    quantizer_ablation = {
        "shape": s0_list[0]["sm80_ptx_quantizer_ablation"]["shape"],
        "naive_fp32_atomic_philox10_spilling_ms_median": round(qa_naive, 4),
        "split_s32_bitcast_splitmix_zero_spill_ms_median": round(qa_fast, 4),
        "speedup": round(qa_naive / max(qa_fast, 1e-6), 4),
    }

    outlier_audit = s0_list[0]["outlier_and_sr_accumulation_audit"]
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
    coat_par_max = {
        k: round(max(float(s["coat_fused_kernel_parity"][k]) for s in s0_list), 6)
        for k in s0_list[0]["coat_fused_kernel_parity"]
    }

    suite_0_summary = {
        "sm80_int8_tensor_core_roofline": rf_agg,
        "sm80_ptx_quantizer_ablation": quantizer_ablation,
        "outlier_and_sr_accumulation_audit": outlier_audit,
        "fp64_chain_rule_refinement": {
            "ratios_h_vs_half_h_median": fd_med,
            "min_refinement_ratio": round(min_fd, 4),
        },
        "coat_fused_kernel_parity_max": coat_par_max,
    }

    # Suite A Aggregation
    sa_list = [
        r["suite_a_linear_3gemm_scaling"]["linear_3gemm_token_scaling"] for r in reps
    ]
    sa_agg: dict[str, Any] = {}
    for mk_tok, mref in sa_list[0].items():
        bf_ms = _med(
            [float(a[mk_tok]["modes"]["bf16_cublas_eager"]["median_ms"]) for a in sa_list]
        )
        modes_a: dict[str, Any] = {}
        for mode_k in mref["modes"]:
            ms_v = [float(a[mk_tok]["modes"][mode_k]["median_ms"]) for a in sa_list]
            tf_v = [
                float(a[mk_tok]["modes"][mode_k]["effective_tflops"]) for a in sa_list
            ]
            sav_v = [
                float(a[mk_tok]["modes"][mode_k]["saved_activation_mib"])
                for a in sa_list
            ]
            m_med = _med(ms_v)
            modes_a[mode_k] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "effective_tflops_median": round(_med(tf_v), 2),
                "saved_activation_mib_median": round(_med(sav_v), 2),
                "speedup_vs_bf16_cublas": round(bf_ms / max(m_med, 1e-6), 4),
            }
        sa_agg[mk_tok] = {
            "num_tokens": mref["num_tokens"],
            "in_features": mref["in_features"],
            "out_features": mref["out_features"],
            "modes": modes_a,
        }

    # Suite B Aggregation
    sb_list = [r["suite_b_coat_swiglu_mlp_stack"] for r in reps]
    stack_par_max = {
        k: round(max(float(b["stack_bf16_parity"][k]) for b in sb_list), 6)
        for k in sb_list[0]["stack_bf16_parity"]
    }
    sb_agg: dict[str, Any] = {}
    for mk_tok, mref in sb_list[0]["token_scaling"].items():
        eag_ms = _med(
            [
                float(b["token_scaling"][mk_tok]["modes"]["mlp_bf16_eager"]["median_ms"])
                for b in sb_list
            ]
        )
        cmp_ms = _med(
            [
                float(
                    b["token_scaling"][mk_tok]["modes"]["mlp_bf16_torch_compile"][
                        "median_ms"
                    ]
                )
                for b in sb_list
            ]
        )
        eag_sav = _med(
            [
                float(
                    b["token_scaling"][mk_tok]["modes"]["mlp_bf16_eager"][
                        "saved_activations_mib"
                    ]
                )
                for b in sb_list
            ]
        )
        modes_b: dict[str, Any] = {}
        for mode_k in mref["modes"]:
            ms_v = [
                float(b["token_scaling"][mk_tok]["modes"][mode_k]["median_ms"])
                for b in sb_list
            ]
            tf_v = [
                float(b["token_scaling"][mk_tok]["modes"][mode_k]["effective_tflops"])
                for b in sb_list
            ]
            pk_v = [
                float(b["token_scaling"][mk_tok]["modes"][mode_k]["peak_inc_mib"])
                for b in sb_list
            ]
            sav_v = [
                float(
                    b["token_scaling"][mk_tok]["modes"][mode_k]["saved_activations_mib"]
                )
                for b in sb_list
            ]
            m_med = _med(ms_v)
            sav_med = _med(sav_v)
            modes_b[mode_k] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "effective_tflops_median": round(_med(tf_v), 2),
                "peak_inc_mib_median": round(_med(pk_v), 2),
                "saved_activations_mib_median": round(sav_med, 2),
                "speedup_vs_bf16_eager": round(eag_ms / max(m_med, 1e-6), 4),
                "speedup_vs_bf16_compile": round(cmp_ms / max(m_med, 1e-6), 4),
                "saved_activation_reduction_vs_eager": round(
                    eag_sav / max(sav_med, 1e-6), 4
                ),
            }
        sb_agg[mk_tok] = {
            "num_tokens": mref["num_tokens"],
            "num_layers": mref["num_layers"],
            "hidden": mref["hidden"],
            "intermediate": mref["intermediate"],
            "modes": modes_b,
        }

    # Suite C Aggregation
    sc_list = [r["suite_c_quantized_fsdp2_8gpu"] for r in reps]
    c1_ref = sc_list[0]["distributed_gradient_parity"]
    c1_agg = {
        "fsdp2_bf16_vs_1gpu_max": {
            k: round(
                max(
                    float(c["distributed_gradient_parity"]["fsdp2_bf16_vs_1gpu"][k])
                    for c in sc_list
                ),
                6,
            )
            for k in c1_ref["fsdp2_bf16_vs_1gpu"]
        },
        "halo_coat_quant_fsdp2_vs_1gpu_max": {
            k: round(
                max(
                    float(
                        c["distributed_gradient_parity"][
                            "halo_coat_quant_fsdp2_vs_1gpu"
                        ][k]
                    )
                    for c in sc_list
                ),
                6,
            )
            for k in c1_ref["halo_coat_quant_fsdp2_vs_1gpu"]
        },
        "naive_per_tensor_rtn_int8_grad_comm_bias_rel_l2_max": round(
            max(
                float(
                    c["distributed_gradient_parity"][
                        "naive_per_tensor_rtn_int8_grad_comm_bias_rel_l2"
                    ]
                )
                for c in sc_list
            ),
            6,
        ),
    }

    c2_ref = sc_list[0]["fsdp2_8gpu_training_scaling"]
    eag_c_ms = _med(
        [
            float(
                c["fsdp2_8gpu_training_scaling"]["modes"]["fsdp2_bf16_eager_reshard"][
                    "step_ms_median"
                ]
            )
            for c in sc_list
        ]
    )
    cmp_c_ms = _med(
        [
            float(
                c["fsdp2_8gpu_training_scaling"]["modes"][
                    "fsdp2_bf16_compile_reshard"
                ]["step_ms_median"]
            )
            for c in sc_list
        ]
    )
    eag_c_mem = _med(
        [
            float(
                c["fsdp2_8gpu_training_scaling"]["modes"]["fsdp2_bf16_eager_reshard"][
                    "per_rank_peak_inc_mib"
                ]
            )
            for c in sc_list
        ]
    )
    modes_c2: dict[str, Any] = {}
    for mk in c2_ref["modes"]:
        ms_v = [
            float(c["fsdp2_8gpu_training_scaling"]["modes"][mk]["step_ms_median"])
            for c in sc_list
        ]
        mem_v = [
            float(
                c["fsdp2_8gpu_training_scaling"]["modes"][mk]["per_rank_peak_inc_mib"]
            )
            for c in sc_list
        ]
        tps_v = [
            float(
                c["fsdp2_8gpu_training_scaling"]["modes"][mk]["global_tokens_per_s"]
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
            "speedup_vs_fsdp2_bf16_eager": round(eag_c_ms / max(m_med, 1e-6), 4),
            "speedup_vs_fsdp2_bf16_compile": round(cmp_c_ms / max(m_med, 1e-6), 4),
            "memory_reduction_vs_fsdp2_bf16_eager": round(
                eag_c_mem / max(mem_med, 1e-6), 4
            ),
        }

    suite_c_summary = {
        "distributed_gradient_parity": c1_agg,
        "fsdp2_8gpu_training_scaling": {
            "num_layers": c2_ref["num_layers"],
            "params_billions": c2_ref["params_billions"],
            "local_tokens_per_gpu": c2_ref["local_tokens_per_gpu"],
            "global_tokens_per_step": c2_ref["global_tokens_per_step"],
            "seq_len": c2_ref["seq_len"],
            "bf16_allgather_mib_per_step": c2_ref["bf16_allgather_mib_per_step"],
            "int8_allgather_mib_per_step": c2_ref["int8_allgather_mib_per_step"],
            "modes": modes_c2,
        },
    }

    # 9 Deterministic Validation Gates
    rfk = rf_agg["kernels"]
    oa = outlier_audit
    a8k = sa_agg["8192"]["modes"]
    a16k = sa_agg["16384"]["modes"]
    b8k = sb_agg["8192"]["modes"]
    c_qf = modes_c2["halo_coat_quant_fsdp2"]

    gates = {
        "smoke_present_and_env_parity": smoke_and_env_ok,
        "clocks_locked_1410": clocks_locked_1410,
        "sm80_int8_roofline_and_ptx_quantizer_ablation": (
            rfk["cutlass_sm80_scaled_mm_int8_to_bf16"]["tops_or_tflops_median"] > 440.0
            and rfk["triton_epilogue_only_coalesced_int8"]["tops_or_tflops_median"]
            > 380.0
            and rfk["triton_jetfire_inloop_promote_1x128x128"]["tops_or_tflops_median"]
            < rfk["bf16_cublas"]["tops_or_tflops_median"]
            and rfk["torch_int_mm_e6_contiguous_copy_and_dequant"][
                "speedup_vs_bf16_cublas"
            ]
            < 1.05
            and quantizer_ablation["speedup"] > 10.0
        ),
        "outlier_hadamard_and_sr_accumulation_convergence": (
            oa["unrotated_switchback_3gemm_int8"]["dw_rel_l2"]
            > 3.0 * oa["halo_block128_hadamard_sr_3gemm_int8"]["dw_rel_l2"]
            and oa["kronecker_64x64_hadamard_fwd_y_rel_l2"]
            < 0.5 * oa["jetfire_1x128x128_fwd_y_rel_l2"]
            and oa["sub_lsb_256step_accumulation_audit"][
                "rtn_int8_sub_lsb_rel_l2_bias"
            ]
            > 0.90
            and oa["sub_lsb_256step_accumulation_audit"]["sr_int8_sub_lsb_rel_l2_err"]
            < 0.15
        ),
        "fp64_richardson_chain_rule": min_fd > 3.90,
        "coat_fused_kernels_and_4layer_stack_parity": (
            max(coat_par_max.values()) < 0.020 and max(stack_par_max.values()) < 0.075
        ),
        "suite_a_3gemm_breaks_sm80_312tflops_bf16_ceiling": (
            a8k["halo_cutlass_3gemm_sr_int8_gacc1"]["effective_tflops_median"] > 312.0
            and a16k["halo_cutlass_3gemm_sr_int8_gacc4"]["effective_tflops_median"]
            > 340.0
            and a8k["halo_cutlass_3gemm_sr_int8_gacc1"]["speedup_vs_bf16_cublas"] > 1.15
            and a8k["triton_jetfire_block128_3gemm_int8"]["speedup_vs_bf16_cublas"]
            < 0.90
        ),
        "suite_b_coat_swiglu_speed_and_saved_activation_compression": (
            b8k["mlp_halo_coat_int8_fused"]["speedup_vs_bf16_eager"] > 1.12
            and b8k["mlp_halo_coat_int8_fused"]["speedup_vs_bf16_compile"] > 1.03
            and b8k["mlp_halo_coat_int8_fused"][
                "saved_activation_reduction_vs_eager"
            ]
            > 1.80
        ),
        "suite_c_8gpu_quantized_fsdp2_parity_speed_and_memory": (
            max(c1_agg["halo_coat_quant_fsdp2_vs_1gpu_max"].values()) < 0.045
            and c1_agg["naive_per_tensor_rtn_int8_grad_comm_bias_rel_l2_max"] > 0.25
            and c_qf["speedup_vs_fsdp2_bf16_eager"] > 1.10
            and c_qf["speedup_vs_fsdp2_bf16_compile"] > 1.03
            and c_qf["memory_reduction_vs_fsdp2_bf16_eager"] > 1.30
        ),
    }
    all_passed = all(gates.values())

    full_out = {
        "experiment": "E123_int8_quantized_training",
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
            "linear_3gemm_token_scaling": sa_agg,
        },
        "suite_b_coat_swiglu_mlp_stack": {
            "stack_bf16_parity_max": stack_par_max,
            "token_scaling": sb_agg,
        },
        "suite_c_quantized_fsdp2_8gpu": suite_c_summary,
    }

    out_path = raw_dir / "int8_quantized_training_full.json"
    out_path.write_text(json.dumps(full_out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} (all_gates_passed={all_passed})")
    if not all_passed:
        print(f"Validation gates failed: {gates}")


if __name__ == "__main__":
    main()
