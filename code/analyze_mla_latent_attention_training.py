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
        description="Deterministic 3-repetition analyzer and 9-gate validator for E120"
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.root / "results" / "raw" / "mla_latent_attention_training"
    smoke = json.loads(
        (raw_dir / "smoke_mla_latent_attention_training.json").read_text(
            encoding="utf-8"
        )
    )
    reps = [
        json.loads(
            (raw_dir / f"rep{i}_mla_latent_attention_training.json").read_text(
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
            r["environment"].get("flashinfer"),
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
        for sc in r["suite_a_mla_layer_scaling"].values():
            for m in sc["modes"].values():
                if m.get("sm_clock_after_mhz", 0.0) > 0:
                    all_clocks.append(float(m["sm_clock_after_mhz"]))
    med_clock = _med(all_clocks) if all_clocks else 0.0
    min_clock = min(all_clocks) if all_clocks else 0.0
    clocks_locked_1410 = med_clock >= 1380.0 and min_clock >= 1260.0

    # 2. Suite 0 Aggregation
    s0_list = [r["suite_0_audits"] for r in reps]
    eco = s0_list[0]["ecosystem_compatibility_audit"]
    dec_eco = eco["decompressed_192_to_128"]
    abs_eco = eco["weight_absorbed_576_to_512_mqa"]

    eco_audit_ok = (
        not dec_eco["flash_attn_func_direct_192_128"]["supported"]
        and dec_eco["flash_attn_func_vpadded_192_192"]["supported"]
        and not dec_eco["sdpa_flash_192_128"]["supported"]
        and dec_eco["sdpa_efficient_192_128"]["supported"]
        and not abs_eco["flash_attn_func_576_576"]["supported"]
        and not abs_eco["sdpa_flash_576_512"]["supported"]
        and not abs_eco["sdpa_efficient_576_512"]["supported"]
        and not abs_eco["flex_attention_576_512_sm80"]["supported"]
        and abs_eco["flashinfer_mla_paged_fa2"]["supported"]
        and not abs_eco["flashinfer_mla_paged_fa3"]["supported"]
    )

    min_fd = min(
        float(s["fp64_chain_rule_refinement"]["min_refinement_ratio"])
        for s in s0_list
    )
    split_fd_med = {
        k: round(
            _med(
                [
                    float(
                        s["fp64_chain_rule_refinement"][
                            "split_mla_ratios_h_vs_half_h"
                        ][k]
                    )
                    for s in s0_list
                ]
            ),
            4,
        )
        for k in s0_list[0]["fp64_chain_rule_refinement"][
            "split_mla_ratios_h_vs_half_h"
        ]
    }
    abs_fd_med = {
        k: round(
            _med(
                [
                    float(
                        s["fp64_chain_rule_refinement"][
                            "absorbed_mla_ratios_h_vs_half_h"
                        ][k]
                    )
                    for s in s0_list
                ]
            ),
            4,
        )
        for k in s0_list[0]["fp64_chain_rule_refinement"][
            "absorbed_mla_ratios_h_vs_half_h"
        ]
    }

    par_split_max = {
        k: round(
            max(
                float(s["bf16_triton_parity"]["split_mla_128_64_to_128"][k])
                for s in s0_list
            ),
            6,
        )
        for k in s0_list[0]["bf16_triton_parity"]["split_mla_128_64_to_128"]
    }
    par_abs_max = {
        k: round(
            max(
                float(s["bf16_triton_parity"]["absorbed_mla_512_64_to_512"][k])
                for s in s0_list
            ),
            6,
        )
        for k in s0_list[0]["bf16_triton_parity"]["absorbed_mla_512_64_to_512"]
    }
    par_nr_max = {
        k: round(
            max(
                float(s["bf16_triton_parity"]["fused_latent_rmsnorm_rope"][k])
                for s in s0_list
            ),
            6,
        )
        for k in s0_list[0]["bf16_triton_parity"]["fused_latent_rmsnorm_rope"]
    }

    ab_list = [s["sm80_split_mla_kernel_ablation"] for s in s0_list]
    atom_ms = _med([float(a["bwd_only_1pass_global_atomic_dq"]["median_ms"]) for a in ab_list])
    stg1_ms = _med([float(a["bwd_only_2pass_sync_stages1"]["median_ms"]) for a in ab_list])
    stg2_ms = _med([float(a["bwd_only_2pass_cp_async_stages2"]["median_ms"]) for a in ab_list])
    tri_ag_ms = _med([float(a["autograd_fwd_bwd_triton_split_mla"]["median_ms"]) for a in ab_list])
    tri_ag_mem = _med([float(a["autograd_fwd_bwd_triton_split_mla"]["peak_inc_mib"]) for a in ab_list])
    fa2_ag_ms = _med([float(a["autograd_fwd_bwd_fa2_vpadded_192"]["median_ms"]) for a in ab_list])
    fa2_ag_mem = _med([float(a["autograd_fwd_bwd_fa2_vpadded_192"]["peak_inc_mib"]) for a in ab_list])
    eff_ag_ms = _med([float(a["autograd_fwd_bwd_sdpa_cutlass_efficient"]["median_ms"]) for a in ab_list])
    eff_ag_mem = _med([float(a["autograd_fwd_bwd_sdpa_cutlass_efficient"]["peak_inc_mib"]) for a in ab_list])

    suite_0_summary = {
        "ecosystem_compatibility_audit": eco,
        "fp64_chain_rule_refinement": {
            "split_mla_ratios_h_vs_half_h_median": split_fd_med,
            "absorbed_mla_ratios_h_vs_half_h_median": abs_fd_med,
            "min_refinement_ratio": round(min_fd, 4),
        },
        "bf16_triton_parity_max_rel_l2": {
            "split_mla_128_64_to_128": par_split_max,
            "absorbed_mla_512_64_to_512": par_abs_max,
            "fused_latent_rmsnorm_rope": par_nr_max,
        },
        "sm80_split_mla_kernel_ablation": {
            "seq_len": ab_list[0]["seq_len"],
            "heads": ab_list[0]["heads"],
            "bwd_1pass_global_atomic_dq_ms_median": round(atom_ms, 4),
            "bwd_2pass_sync_stages1_ms_median": round(stg1_ms, 4),
            "bwd_2pass_cp_async_stages2_ms_median": round(stg2_ms, 4),
            "speedup_2pass_vs_global_atomic_dq": round(atom_ms / max(stg2_ms, 1e-6), 4),
            "speedup_stages2_vs_stages1": round(stg1_ms / max(stg2_ms, 1e-6), 4),
            "autograd_triton_split_mla_ms_median": round(tri_ag_ms, 4),
            "autograd_triton_split_mla_peak_mib": round(tri_ag_mem, 2),
            "autograd_fa2_vpadded_192_ms_median": round(fa2_ag_ms, 4),
            "autograd_fa2_vpadded_192_peak_mib": round(fa2_ag_mem, 2),
            "autograd_sdpa_cutlass_efficient_ms_median": round(eff_ag_ms, 4),
            "autograd_sdpa_cutlass_efficient_peak_mib": round(eff_ag_mem, 2),
            "memory_reduction_vs_fa2_vpadded": round(fa2_ag_mem / max(tri_ag_mem, 1e-6), 4),
            "speedup_vs_sdpa_cutlass_efficient": round(eff_ag_ms / max(tri_ag_ms, 1e-6), 4),
        },
    }

    # 3. Suite A Aggregation
    suite_a_summary: dict[str, Any] = {}
    for sk, sref in reps[0]["suite_a_mla_layer_scaling"].items():
        fa2_ms = _med(
            [
                float(
                    r["suite_a_mla_layer_scaling"][sk]["modes"][
                        "mla_fa2_vpadded_eager"
                    ]["median_ms"]
                )
                for r in reps
            ]
        )
        fa2_mem = _med(
            [
                float(
                    r["suite_a_mla_layer_scaling"][sk]["modes"][
                        "mla_fa2_vpadded_eager"
                    ]["peak_inc_mib"]
                )
                for r in reps
            ]
        )
        modes_a: dict[str, Any] = {}
        for mk, mref in sref["modes"].items():
            ms_v = [
                float(r["suite_a_mla_layer_scaling"][sk]["modes"][mk]["median_ms"])
                for r in reps
            ]
            mem_v = [
                float(
                    r["suite_a_mla_layer_scaling"][sk]["modes"][mk]["peak_inc_mib"]
                )
                for r in reps
            ]
            m_med = _med(ms_v)
            mem_med = _med(mem_v)
            ent: dict[str, Any] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "peak_inc_mib_median": round(mem_med, 2),
                "ktok_per_s_median": round(
                    (float(sref["seq_len"]) / (m_med * 1e-3)) / 1e3, 2
                ),
                "speedup_vs_fa2_vpadded": round(fa2_ms / max(m_med, 1e-6), 4),
                "memory_reduction_vs_fa2_vpadded": round(
                    fa2_mem / max(mem_med, 1e-6), 4
                ),
            }
            if "first_call_compile_s" in mref:
                ent["first_call_compile_s_median"] = round(
                    _med(
                        [
                            float(
                                r["suite_a_mla_layer_scaling"][sk]["modes"][mk][
                                    "first_call_compile_s"
                                ]
                            )
                            for r in reps
                        ]
                    ),
                    2,
                )
            modes_a[mk] = ent
        suite_a_summary[sk] = {
            "hidden": sref["hidden"],
            "heads": sref["heads"],
            "seq_len": sref["seq_len"],
            "modes": modes_a,
        }

    # 4. Suite B Aggregation
    sb_list = [r["suite_b_crossover_and_latent_sac"] for r in reps]
    cross_agg: dict[str, Any] = {}
    for sqk, sqref in sb_list[0]["prefill_to_decode_absorption_crossover"].items():
        dec_ms = _med(
            [
                float(
                    b["prefill_to_decode_absorption_crossover"][sqk][
                        "decompressed_kv_fa2"
                    ]["median_ms"]
                )
                for b in sb_list
            ]
        )
        tri_ms = _med(
            [
                float(
                    b["prefill_to_decode_absorption_crossover"][sqk][
                        "weight_absorbed_triton_mla"
                    ]["median_ms"]
                )
                for b in sb_list
            ]
        )
        c_ent: dict[str, Any] = {
            "batch": sqref["batch"],
            "heads": sqref["heads"],
            "kv_context_len": sqref["kv_context_len"],
            "query_len": sqref["query_len"],
            "latent_kv_cache_mib": sqref["latent_kv_cache_mib"],
            "decompressed_kv_cache_mib": sqref["decompressed_kv_cache_mib"],
            "kv_cache_compression_ratio": sqref["kv_cache_compression_ratio"],
            "decompressed_kv_fa2_ms_median": round(dec_ms, 4),
            "weight_absorbed_triton_mla_ms_median": round(tri_ms, 4),
            "absorbed_triton_speedup_vs_decompressed": round(
                dec_ms / max(tri_ms, 1e-6), 4
            ),
        }
        if "weight_absorbed_flashinfer_fa2_paged" in sqref:
            fi_ms = _med(
                [
                    float(
                        b["prefill_to_decode_absorption_crossover"][sqk][
                            "weight_absorbed_flashinfer_fa2_paged"
                        ]["median_ms"]
                    )
                    for b in sb_list
                ]
            )
            c_ent["weight_absorbed_flashinfer_fa2_ms_median"] = round(fi_ms, 4)
            c_ent["absorbed_flashinfer_speedup_vs_decompressed"] = round(
                dec_ms / max(fi_ms, 1e-6), 4
            )
        cross_agg[sqk] = c_ent

    sac_par_max = {
        k: round(
            max(float(b["latent_sac_parity_vs_no_ac"][k]) for b in sb_list), 6
        )
        for k in sb_list[0]["latent_sac_parity_vs_no_ac"]
    }

    sac_agg: dict[str, Any] = {}
    for sk, sref in sb_list[0]["four_layer_latent_sac_scaling"].items():
        no_ac_mem = _med(
            [
                float(b["four_layer_latent_sac_scaling"][sk]["no_ac"]["peak_inc_mib"])
                for b in sb_list
            ]
        )
        full_ac_ms = _med(
            [
                float(
                    b["four_layer_latent_sac_scaling"][sk]["full_layer_ac"][
                        "median_ms"
                    ]
                )
                for b in sb_list
            ]
        )
        s_out: dict[str, Any] = {}
        for pk in sref:
            ms_v = [
                float(b["four_layer_latent_sac_scaling"][sk][pk]["median_ms"])
                for b in sb_list
            ]
            mem_v = [
                float(b["four_layer_latent_sac_scaling"][sk][pk]["peak_inc_mib"])
                for b in sb_list
            ]
            m_med = _med(ms_v)
            mem_med = _med(mem_v)
            s_out[pk] = {
                "ms_median": round(m_med, 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "peak_inc_mib_median": round(mem_med, 2),
                "memory_reduction_vs_no_ac": round(no_ac_mem / max(mem_med, 1e-6), 4),
                "speedup_vs_full_layer_ac": round(full_ac_ms / max(m_med, 1e-6), 4),
            }
        sac_agg[sk] = s_out

    suite_b_summary = {
        "prefill_to_decode_absorption_crossover": cross_agg,
        "latent_sac_parity_vs_no_ac_max_rel_l2": sac_par_max,
        "four_layer_latent_sac_scaling": sac_agg,
    }

    # 5. Suite C Aggregation
    sc_list = [r["suite_c_cp8_latent_ring"] for r in reps]
    cp_par_lat = {
        k: round(
            max(
                float(c["cp8_vs_1gpu_parity"]["compressed_latent_ring"][k])
                for c in sc_list
            ),
            6,
        )
        for k in sc_list[0]["cp8_vs_1gpu_parity"]["compressed_latent_ring"]
    }
    cp_par_dec = {
        k: round(
            max(
                float(c["cp8_vs_1gpu_parity"]["decompressed_kv_ring"][k])
                for c in sc_list
            ),
            6,
        )
        for k in sc_list[0]["cp8_vs_1gpu_parity"]["decompressed_kv_ring"]
    }
    cp_par_uly = {
        k: round(
            max(
                float(c["cp8_vs_1gpu_parity"]["ulysses_triton_split"][k])
                for c in sc_list
            ),
            6,
        )
        for k in sc_list[0]["cp8_vs_1gpu_parity"]["ulysses_triton_split"]
    }

    cp_scale_agg: dict[str, Any] = {}
    for sk, sref in sc_list[0]["cp8_long_context_scaling"].items():
        dec_ring_ms = _med(
            [
                float(
                    c["cp8_long_context_scaling"][sk]["modes"][
                        "cp8_mla_decompressed_zigzag_ring"
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
            modes_c[mk] = {
                "step_ms_median": round(m_med, 4),
                "step_ms_iqr": round(_iqr(ms_v), 4),
                "cluster_ktok_per_s_median": round(_med(kt_v), 2),
                "per_rank_peak_inc_mib_median": round(_med(mem_v), 2),
                "comm_mib_per_rank": mref["comm_mib_per_rank"],
                "speedup_vs_decompressed_ring": round(
                    dec_ring_ms / max(m_med, 1e-6), 4
                ),
            }
        cp_scale_agg[sk] = {
            "global_seq_len": sref["global_seq_len"],
            "local_seq_len": sref["local_seq_len"],
            "heads": sref["heads"],
            "hidden": sref["hidden"],
            "modes": modes_c,
        }

    suite_c_summary = {
        "cp8_vs_1gpu_parity_max_rel_l2": {
            "compressed_latent_ring": cp_par_lat,
            "decompressed_kv_ring": cp_par_dec,
            "ulysses_triton_split": cp_par_uly,
        },
        "cp8_long_context_scaling": cp_scale_agg,
    }

    # 6. 9 Deterministic Validation Gates
    s0_ab = suite_0_summary["sm80_split_mla_kernel_ablation"]
    sa_8k = suite_a_summary["8192"]["modes"]
    sa_16k = suite_a_summary["16384"]["modes"]
    c_16k = cp_scale_agg["16384"]["modes"]
    c_131k = cp_scale_agg["131072"]["modes"]

    gates = {
        "smoke_present_and_env_parity": smoke_and_env_ok,
        "clocks_locked_1410": clocks_locked_1410,
        "ecosystem_compatibility_and_sm80_sram_audit": eco_audit_ok,
        "fp64_richardson_chain_rule": min_fd > 3.90,
        "bf16_triton_split_absorbed_normrope_parity": (
            max(par_split_max.values()) < 0.01
            and max(par_abs_max.values()) < 0.01
            and max(par_nr_max.values()) < 0.01
        ),
        "sm80_kernel_ablation_and_memory_savings": (
            s0_ab["speedup_2pass_vs_global_atomic_dq"] > 10.0
            and s0_ab["speedup_stages2_vs_stages1"] > 10.0
            and s0_ab["memory_reduction_vs_fa2_vpadded"] > 1.30
            and s0_ab["speedup_vs_sdpa_cutlass_efficient"] > 2.0
        ),
        "suite_a_decompressed_vs_absorbed_training_and_cutlass": (
            (
                sa_16k["mla_sdpa_cutlass_efficient"]["ms_median"]
                / max(sa_16k["mla_triton_split_fused"]["ms_median"], 1e-6)
            )
            > 1.5
            and (
                sa_8k["mla_triton_weight_absorbed"]["ms_median"]
                / max(sa_8k["mla_triton_split_fused"]["ms_median"], 1e-6)
            )
            > 10.0
            and sa_16k["mla_triton_split_fused"][
                "memory_reduction_vs_fa2_vpadded"
            ]
            > 1.15
        ),
        "suite_b_decode_to_prefill_crossover_and_latent_sac": (
            cross_agg["1"]["absorbed_triton_speedup_vs_decompressed"] > 4.0
            and cross_agg["1"]["absorbed_flashinfer_speedup_vs_decompressed"]
            > 5.0
            and cross_agg["4096"]["absorbed_triton_speedup_vs_decompressed"]
            < 0.50
            and max(sac_par_max.values()) < 0.01
            and sac_agg["16384"]["compressed_latent_sac"][
                "memory_reduction_vs_no_ac"
            ]
            > 1.50
            and sac_agg["16384"]["compressed_latent_sac"][
                "speedup_vs_full_layer_ac"
            ]
            > 1.08
        ),
        "cp8_latent_ring_parity_and_131k_comm_speedup": (
            max(cp_par_lat.values()) < 0.01
            and max(cp_par_dec.values()) < 0.01
            and max(cp_par_uly.values()) < 0.01
            and (
                c_131k["cp8_mla_decompressed_zigzag_ring"]["comm_mib_per_rank"]
                / max(
                    c_131k["cp8_mla_compressed_latent_zigzag_ring"][
                        "comm_mib_per_rank"
                    ],
                    1e-6,
                )
            )
            > 14.0
            and (
                c_131k["cp8_mla_decompressed_zigzag_ring"][
                    "per_rank_peak_inc_mib_median"
                ]
                / max(
                    c_131k["cp8_mla_compressed_latent_zigzag_ring"][
                        "per_rank_peak_inc_mib_median"
                    ],
                    1e-6,
                )
            )
            > 1.80
            and c_16k["cp8_mla_compressed_latent_zigzag_ring"][
                "speedup_vs_decompressed_ring"
            ]
            > 1.10
            and c_131k["cp8_mla_compressed_latent_zigzag_ring"][
                "speedup_vs_decompressed_ring"
            ]
            > 1.02
            and (
                c_131k["cp8_mla_ulysses_fa2_vpadded"][
                    "per_rank_peak_inc_mib_median"
                ]
                / max(
                    c_131k["cp8_mla_ulysses_triton_split"][
                        "per_rank_peak_inc_mib_median"
                    ],
                    1e-6,
                )
            )
            > 1.25
        ),
    }
    all_passed = all(gates.values())

    full_out = {
        "experiment": "E120_mla_latent_attention_training",
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
        "suite_a_mla_layer_scaling": suite_a_summary,
        "suite_b_crossover_and_latent_sac": suite_b_summary,
        "suite_c_cp8_latent_ring": suite_c_summary,
        "rep_elapsed_seconds": [r["elapsed_s"] for r in reps],
    }

    out_path = raw_dir / "mla_latent_attention_training_full.json"
    out_path.write_text(
        json.dumps(full_out, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Wrote {out_path} (all_gates_passed={all_passed})")
    if not all_passed:
        failed = [k for k, v in gates.items() if not v]
        raise SystemExit(f"Validation gates failed: {failed}")


if __name__ == "__main__":
    main()
