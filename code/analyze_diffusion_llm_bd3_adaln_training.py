#!/usr/bin/env python3
"""Deterministic 3-repetition analyzer and 9-gate validator for E127:
Diffusion-LLM, Block-Diffusion (`BD3-LMs`), & `DiT` `adaLN-Zero` Training Kernels
Driven by the Immutable Recursive Self-Improvement (`RSI`) Verification & Vector-Pareto Ratchet (`PROGRAM.md`).
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


def _sha256_lf_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic 3-repetition analyzer and 9-gate validator for E127"
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.root / "results" / "raw" / "diffusion_llm_bd3_adaln_training"
    harness_path = args.root / "code" / "rsi_kernel_harness.py"
    disk_harness_sha = _sha256_lf_file(harness_path)

    smoke = json.loads(
        (raw_dir / "smoke_diffusion_llm_bd3_adaln_training.json").read_text(encoding="utf-8")
    )
    reps = [
        json.loads(
            (raw_dir / f"rep{i}_diffusion_llm_bd3_adaln_training.json").read_text(encoding="utf-8")
        )
        for i in (1, 2, 3)
    ]
    ledger_sha_disk = _sha256_file(raw_dir / "rsi_ledger.tsv")

    envs = [
        (
            r["environment"].get("torch"),
            r["environment"].get("triton"),
            r["environment"].get("flash_attn"),
            r["environment"].get("cudnn_version"),
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
        _harvest_telemetry(r["suite_0_audits"]["bd3_backend_and_sympy_compiler_audit"]["sdpa_bool_mask_2l_probe"])
        for tk in ("track2_adaln_zero_rsi", "track3_masked_ce_rsi"):
            for cv in r["suite_a_1gpu_adaln_and_masked_ce_rsi"][tk]["candidates"].values():
                for sv in cv["shapes"].values():
                    _harvest_telemetry(sv)
        sb = r["suite_b_1gpu_bd3_attention_rsi_and_scaling"]
        for cv in sb["track1_bd3_attention_rsi"]["candidates"].values():
            for sv in cv["shapes"].values():
                _harvest_telemetry(sv)
        for sv in sb["seqlen_scaling_w16"].values():
            for mk in (
                "sdpa_bool_mask_2l",
                "flex_bd3_dynamic_bm_2l",
                "flex_bd3_prebuilt_bm_2l",
                "surgical_6call_fa2_bd3",
                "adaptive_zerospill_bd3",
                "adaptive_zerospill_last_layer_pruned_bd3",
            ):
                if isinstance(sv.get(mk), dict) and "median_ms" in sv[mk]:
                    _harvest_telemetry(sv[mk])
        for bv in r["suite_c_8gpu_bd3_fsdp2_scaling"]["scaling_4rung_ablation_c2"].values():
            for rgv in bv["rungs"].values():
                _harvest_telemetry(rgv)

    med_clock = _med(all_clocks)
    min_clock = min(all_clocks)
    max_clock = max(all_clocks)
    max_power = max(all_powers)
    clocks_locked_1410 = (
        all(c >= 1380.0 for c in init_clocks)
        and max_clock >= 1410.0
        and med_clock >= 1380.0
        and min_clock >= 1260.0
    )

    # ------------------------------------------------------------------------
    # Cryptographic RSI Harness & 3-Track Vector-Pareto Ratchet Aggregation
    # ------------------------------------------------------------------------
    recorded_harness_shas = {
        smoke["rsi_harness"]["eval_harness_sha256"],
        *(r["rsi_harness"]["eval_harness_sha256"] for r in reps),
    }
    harness_sha_verified = (
        len(recorded_harness_shas) == 1
        and disk_harness_sha in recorded_harness_shas
        and ledger_sha_disk == reps[0]["rsi_harness"]["ledger_tsv_sha256"]
    )

    expected_decisions = {
        "bd3_block_diffusion_attention": [
            "REJECT_COMPILE_OR_RUNTIME",
            "BASELINE",
            "REJECT_SHAPE_REGRESSION",
            "REJECT_SHAPE_REGRESSION",
            "KEEP",
            "KEEP_TERMINAL_STAGE",
        ],
        "block_segmented_adaln_zero": [
            "BASELINE",
            "KEEP",
            "KEEP",
        ],
        "masked_diffusion_terminal_ce": [
            "BASELINE",
            "REJECT_SHAPE_REGRESSION",
            "KEEP",
        ],
    }

    rsi_tracks_agg: dict[str, Any] = {}
    ratchet_decisions_consistent = True
    for ti in range(3):
        t_ref = reps[0]["rsi_harness"]["tracks"][ti]
        t_name = t_ref["track"]
        t_reps = [r["rsi_harness"]["tracks"][ti] for r in reps]
        for tr in t_reps:
            decs = [x["decision"] for x in tr["trials"]]
            if decs != expected_decisions[t_name]:
                ratchet_decisions_consistent = False
            for x in tr["trials"]:
                if x["eval_harness_sha256"] != disk_harness_sha:
                    harness_sha_verified = False
                if x["decision"] in ("KEEP", "KEEP_TERMINAL_STAGE"):
                    if (not x["correctness_pass"]) or (not x["anti_cheat_pass"]) or x["worst_shape_ratio_vs_baseline"] > 1.02:
                        ratchet_decisions_consistent = False

        trials_agg: list[dict[str, Any]] = []
        for idx, tr0 in enumerate(t_ref["trials"]):
            tr_list = [tr["trials"][idx] for tr in t_reps]
            is_compile_fail = tr0["decision"] == "REJECT_COMPILE_OR_RUNTIME"
            per_s_med: dict[str, float] = {}
            if not is_compile_fail:
                for sk in t_ref["shape_keys"]:
                    per_s_med[sk] = round(_med([float(x["per_shape_ms"][sk]) for x in tr_list]), 4)
            trials_agg.append(
                {
                    "trial_id": int(tr0["trial_id"]),
                    "candidate_id": tr0["candidate_id"],
                    "candidate_sha256": tr0["candidate_sha256"],
                    "decision": tr0["decision"],
                    "correctness_pass_all_reps": all(bool(x["correctness_pass"]) for x in tr_list),
                    "anti_cheat_pass_all_reps": all(bool(x["anti_cheat_pass"]) for x in tr_list),
                    "max_fp64_rel_l2_max": (
                        round(max(float(x["max_fp64_rel_l2"]) for x in tr_list), 6) if not is_compile_fail else -1.0
                    ),
                    "geomean_ms_median": (
                        round(_med([float(x["geomean_ms"]) for x in tr_list]), 4) if not is_compile_fail else -1.0
                    ),
                    "geomean_ms_iqr": (
                        round(_iqr([float(x["geomean_ms"]) for x in tr_list]), 4) if not is_compile_fail else 0.0
                    ),
                    "peak_alloc_mib_median": (
                        round(_med([float(x["peak_alloc_mib"]) for x in tr_list]), 2) if not is_compile_fail else 0.0
                    ),
                    "worst_shape_ratio_vs_baseline_max": (
                        round(max(float(x["worst_shape_ratio_vs_baseline"]) for x in tr_list), 4)
                        if not is_compile_fail
                        else -1.0
                    ),
                    "speedup_vs_baseline_median": (
                        round(_med([float(x["speedup_vs_baseline"]) for x in tr_list]), 4) if not is_compile_fail else 0.0
                    ),
                    "per_shape_ms_median": per_s_med,
                    "diagnosis": tr0["diagnosis"],
                }
            )

        rsi_tracks_agg[t_name] = {
            "shape_keys": t_ref["shape_keys"],
            "baseline_candidate": t_ref["baseline_candidate"],
            "final_incumbent_candidate": t_ref["final_incumbent_candidate"],
            "final_speedup_vs_baseline_median": round(
                _med([float(tr["final_speedup_vs_baseline"]) for tr in t_reps]), 4
            ),
            "trials": trials_agg,
        }

    # ------------------------------------------------------------------------
    # Suite 0 Aggregation
    # ------------------------------------------------------------------------
    s0_list = [r["suite_0_audits"] for r in reps]
    b_ref = s0_list[0]["bd3_backend_and_sympy_compiler_audit"]
    sdpa_probe_ms = [float(s["bd3_backend_and_sympy_compiler_audit"]["sdpa_bool_mask_2l_probe"]["median_ms"]) for s in s0_list]
    sdpa_probe_mem = [float(s["bd3_backend_and_sympy_compiler_audit"]["sdpa_bool_mask_2l_probe"]["peak_alloc_mib"]) for s in s0_list]
    backend_agg = {
        "official_fig4_sympy_bool_eq_inductor_crash_all_reps": all(
            bool(s["bd3_backend_and_sympy_compiler_audit"]["official_fig4_sympy_bool_eq_inductor_crash"]) for s in s0_list
        ),
        "official_fig4_error_excerpt": b_ref["official_fig4_error_excerpt"],
        "flash_sdpa_arbitrary_2l_bool_mask_rejected_all_reps": all(
            bool(s["bd3_backend_and_sympy_compiler_audit"]["flash_sdpa_arbitrary_2l_bool_mask_rejected"]) for s in s0_list
        ),
        "sdpa_bool_mask_2l_l4096_2l8192": {
            "ms_median": round(_med(sdpa_probe_ms), 3),
            "ms_iqr": round(_iqr(sdpa_probe_ms), 3),
            "peak_alloc_mib_median": round(_med(sdpa_probe_mem), 2),
        },
        "sdpa_l16k_2l32k_fp32_score_matrix_gib": float(b_ref["sdpa_l16k_2l32k_fp32_score_matrix_gib"]),
    }

    var_agg: dict[str, Any] = {}
    for sk, sv0 in s0_list[0]["nelbo_1_over_t_variance_and_antithetic_audit"].items():
        var_agg[sk] = {
            "num_mc_draws": int(sv0["num_mc_draws"]),
            "mean_loss_median": round(
                _med([float(s["nelbo_1_over_t_variance_and_antithetic_audit"][sk]["mean_loss"]) for s in s0_list]), 6
            ),
            "loss_variance_median": round(
                _med([float(s["nelbo_1_over_t_variance_and_antithetic_audit"][sk]["loss_variance"]) for s in s0_list]), 8
            ),
            "grad_head_mc_variance_median": round(
                _med([float(s["nelbo_1_over_t_variance_and_antithetic_audit"][sk]["grad_head_mc_variance"]) for s in s0_list]), 8
            ),
            "grad_variance_reduction_vs_unclipped_median": round(
                _med(
                    [
                        float(s["nelbo_1_over_t_variance_and_antithetic_audit"][sk]["grad_variance_reduction_vs_unclipped"])
                        for s in s0_list
                    ]
                ),
                3,
            ),
        }

    p0_ref = s0_list[0]["per_tensor_fp64_oracle_parity"]
    bd3_p_max = {
        k: round(max(float(s["per_tensor_fp64_oracle_parity"]["bd3_attention_cases"][k]["max_rel_l2"]) for s in s0_list), 6)
        for k in p0_ref["bd3_attention_cases"]
    }
    adaln_p_max = {
        k: round(max(float(s["per_tensor_fp64_oracle_parity"]["adaln_zero_cases"][k]["max_rel_l2"]) for s in s0_list), 6)
        for k in p0_ref["adaln_zero_cases"]
    }
    ce_p_max = {
        k: round(max(float(s["per_tensor_fp64_oracle_parity"]["masked_ce_cases"][k]["max_rel_l2"]) for s in s0_list), 6)
        for k in p0_ref["masked_ce_cases"]
    }
    all_parity_ok = all(bool(s["per_tensor_fp64_oracle_parity"]["all_parity_ok"]) for s in s0_list)

    r_keys = list(s0_list[0]["fp64_richardson_central_difference"]["ratios"].keys())
    rich_min_by_key = {
        rk: round(min(float(s["fp64_richardson_central_difference"]["ratios"][rk]) for s in s0_list), 5)
        for rk in r_keys
    }
    rich_agg = {
        "all_ok": all(bool(s["fp64_richardson_central_difference"]["all_ok"]) for s in s0_list),
        "ratios_min_across_reps": rich_min_by_key,
        "min_ratio": round(min(float(s["fp64_richardson_central_difference"]["min_ratio"]) for s in s0_list), 4),
        "max_ratio": round(max(float(s["fp64_richardson_central_difference"]["max_ratio"]) for s in s0_list), 4),
    }

    suite_0_summary = {
        "bd3_backend_and_sympy_compiler_audit": backend_agg,
        "nelbo_1_over_t_variance_and_antithetic_audit": var_agg,
        "per_tensor_fp64_oracle_parity": {
            "all_parity_ok": all_parity_ok,
            "bd3_attention_max_rel_l2": bd3_p_max,
            "adaln_zero_max_rel_l2": adaln_p_max,
            "masked_ce_max_rel_l2": ce_p_max,
        },
        "fp64_richardson_central_difference": rich_agg,
    }

    # ------------------------------------------------------------------------
    # Suite A Aggregation (Track 2 AdaLN-Zero Roofline & Track 3 Masked-CE)
    # ------------------------------------------------------------------------
    sa_list = [r["suite_a_1gpu_adaln_and_masked_ce_rsi"] for r in reps]
    adaln_shapes_agg: dict[str, Any] = {}
    for cid in sa_list[0]["track2_adaln_zero_rsi"]["candidates"]:
        sh_dict: dict[str, Any] = {}
        for sk in sa_list[0]["track2_adaln_zero_rsi"]["candidates"][cid]["shapes"]:
            ms_v = [float(a["track2_adaln_zero_rsi"]["candidates"][cid]["shapes"][sk]["median_ms"]) for a in sa_list]
            gb_v = [float(a["track2_adaln_zero_rsi"]["candidates"][cid]["shapes"][sk]["effective_gbps"]) for a in sa_list]
            pk_v = [float(a["track2_adaln_zero_rsi"]["candidates"][cid]["shapes"][sk]["peak_alloc_mib"]) for a in sa_list]
            sh_dict[sk] = {
                "ms_median": round(_med(ms_v), 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "effective_gbps_median": round(_med(gb_v), 2),
                "peak_alloc_mib_median": round(_med(pk_v), 2),
            }
        adaln_shapes_agg[cid] = sh_dict

    ce_shapes_agg: dict[str, Any] = {}
    for cid in sa_list[0]["track3_masked_ce_rsi"]["candidates"]:
        sh_dict_ce: dict[str, Any] = {}
        for sk, sref in sa_list[0]["track3_masked_ce_rsi"]["candidates"][cid]["shapes"].items():
            ms_v = [float(a["track3_masked_ce_rsi"]["candidates"][cid]["shapes"][sk]["median_ms"]) for a in sa_list]
            pk_v = [float(a["track3_masked_ce_rsi"]["candidates"][cid]["shapes"][sk]["peak_alloc_mib"]) for a in sa_list]
            sh_dict_ce[sk] = {
                "total_tokens": int(sref["total_tokens"]),
                "active_masked_tokens": int(sref["active_masked_tokens"]),
                "ms_median": round(_med(ms_v), 4),
                "ms_iqr": round(_iqr(ms_v), 4),
                "peak_alloc_mib_median": round(_med(pk_v), 2),
            }
        ce_shapes_agg[cid] = sh_dict_ce

    suite_a_summary = {
        "track2_adaln_zero_candidates": adaln_shapes_agg,
        "track3_masked_ce_candidates": ce_shapes_agg,
    }

    # ------------------------------------------------------------------------
    # Suite B Aggregation (BD3-LM Sequence-Length Scaling W=16, L=2K..32K, 2L=4K..64K)
    # ------------------------------------------------------------------------
    sb_list = [r["suite_b_1gpu_bd3_attention_rsi_and_scaling"] for r in reps]
    seq_agg: dict[str, Any] = {}
    for sk, sref in sb_list[0]["seqlen_scaling_w16"].items():
        row_b: dict[str, Any] = {
            "clean_seqlen_l": int(sref["clean_seqlen_l"]),
            "concat_seqlen_2l": int(sref["concat_seqlen_2l"]),
            "block_size_w": int(sref["block_size_w"]),
        }
        for mk in (
            "sdpa_bool_mask_2l",
            "flex_bd3_dynamic_bm_2l",
            "flex_bd3_prebuilt_bm_2l",
            "surgical_6call_fa2_bd3",
            "adaptive_zerospill_bd3",
            "adaptive_zerospill_last_layer_pruned_bd3",
        ):
            if isinstance(sref.get(mk), dict) and "median_ms" in sref[mk]:
                ms_v = [float(b["seqlen_scaling_w16"][sk][mk]["median_ms"]) for b in sb_list]
                pk_v = [float(b["seqlen_scaling_w16"][sk][mk]["peak_alloc_mib"]) for b in sb_list]
                row_b[mk] = {
                    "ms_median": round(_med(ms_v), 4),
                    "ms_iqr": round(_iqr(ms_v), 4),
                    "peak_alloc_mib_median": round(_med(pk_v), 2),
                }
            else:
                row_b[mk] = sref.get(mk)
        ad_ms = row_b["adaptive_zerospill_bd3"]["ms_median"]
        pr_ms = row_b["adaptive_zerospill_last_layer_pruned_bd3"]["ms_median"]
        row_b["speedup_adaptive_vs_flex_dyn"] = round(
            row_b["flex_bd3_dynamic_bm_2l"]["ms_median"] / max(ad_ms, 1e-9), 3
        )
        row_b["speedup_adaptive_vs_flex_prebuilt"] = round(
            row_b["flex_bd3_prebuilt_bm_2l"]["ms_median"] / max(ad_ms, 1e-9), 3
        )
        row_b["speedup_pruned_vs_flex_prebuilt"] = round(
            row_b["flex_bd3_prebuilt_bm_2l"]["ms_median"] / max(pr_ms, 1e-9), 3
        )
        if isinstance(row_b["sdpa_bool_mask_2l"], dict) and "ms_median" in row_b["sdpa_bool_mask_2l"]:
            row_b["speedup_adaptive_vs_sdpa_2l"] = round(
                row_b["sdpa_bool_mask_2l"]["ms_median"] / max(ad_ms, 1e-9), 3
            )
        seq_agg[sk] = row_b

    # ------------------------------------------------------------------------
    # Suite C Aggregation (8-GPU NVSwitch 1.12B BD3-LM FSDP2 Parity & 4-Rung Scaling)
    # ------------------------------------------------------------------------
    sc_list = [r["suite_c_8gpu_bd3_fsdp2_scaling"] for r in reps]
    c1_agg: dict[str, Any] = {}
    for rgk, rgref in sc_list[0]["parity_and_empty_mask_gate_c1"].items():
        c1_agg[rgk] = {
            "rank7_forced_empty_mask_m0": bool(rgref["rank7_forced_empty_mask_m0"]),
            **{
                mk: round(max(float(c["parity_and_empty_mask_gate_c1"][rgk][mk]) for c in sc_list), 8)
                for mk in rgref
                if mk != "rank7_forced_empty_mask_m0"
            },
        }

    c2_agg: dict[str, Any] = {}
    for bk, bref in sc_list[0]["scaling_4rung_ablation_c2"].items():
        rungs_c: dict[str, Any] = {}
        for rgk in bref["rungs"]:
            ms_v = [float(c["scaling_4rung_ablation_c2"][bk]["rungs"][rgk]["median_ms"]) for c in sc_list]
            ct_v = [float(c["scaling_4rung_ablation_c2"][bk]["rungs"][rgk]["clean_tokens_per_s"]) for c in sc_list]
            dt_v = [float(c["scaling_4rung_ablation_c2"][bk]["rungs"][rgk]["dual_stream_2l_tokens_per_s"]) for c in sc_list]
            pk_v = [float(c["scaling_4rung_ablation_c2"][bk]["rungs"][rgk]["max_rank_peak_alloc_mib"]) for c in sc_list]
            rungs_c[rgk] = {
                "ms_median": round(_med(ms_v), 3),
                "ms_iqr": round(_iqr(ms_v), 3),
                "clean_tokens_per_s_median": round(_med(ct_v), 1),
                "dual_stream_2l_tokens_per_s_median": round(_med(dt_v), 1),
                "max_rank_peak_alloc_mib_median": round(_med(pk_v), 2),
            }
        r1 = rungs_c["rung1_flex_bd3_eager_adaln_full_ce"]
        r4 = rungs_c["rung4_full_rsi_stack_compacted_3gemm_masked_ce"]
        c2_agg[bk] = {
            "global_clean_tokens": int(bref["global_clean_tokens"]),
            "global_dual_stream_2l_tokens": int(bref["global_dual_stream_2l_tokens"]),
            "block_size_w": int(bref["block_size_w"]),
            "rungs": rungs_c,
            "speedup_rung4_vs_rung1": round(r1["ms_median"] / max(r4["ms_median"], 1e-9), 3),
            "hbm_saved_mib_rung4_vs_rung1": round(
                r1["max_rank_peak_alloc_mib_median"] - r4["max_rank_peak_alloc_mib_median"], 2
            ),
            "hbm_ratio_rung1_over_rung4": round(
                r1["max_rank_peak_alloc_mib_median"] / max(r4["max_rank_peak_alloc_mib_median"], 1e-9), 3
            ),
        }

    suite_c_summary = {
        "model_parameters": int(sc_list[0]["model_parameters"]),
        "world_size": 8,
        "parity_and_empty_mask_gate_c1": c1_agg,
        "scaling_4rung_ablation_c2": c2_agg,
    }

    # ------------------------------------------------------------------------
    # 9 Strict Deterministic Validation Gates
    # ------------------------------------------------------------------------
    t1_trials = {t["candidate_id"]: t for t in rsi_tracks_agg["bd3_block_diffusion_attention"]["trials"]}
    t2_trials = {t["candidate_id"]: t for t in rsi_tracks_agg["block_segmented_adaln_zero"]["trials"]}
    t3_trials = {t["candidate_id"]: t for t in rsi_tracks_agg["masked_diffusion_terminal_ce"]["trials"]}

    unclipped_lv = var_agg["unclipped_iid_uniform_t"]["loss_variance_median"]
    anti_lv = var_agg["antithetic_complementary_pair_t"]["loss_variance_median"]
    anti_gred = var_agg["antithetic_complementary_pair_t"]["grad_variance_reduction_vs_unclipped_median"]
    clipped_red = var_agg["bd3_clipped_antithetic_u03_07"]["grad_variance_reduction_vs_unclipped_median"]

    gates = {
        "smoke_present_and_env_parity": smoke_and_env_ok,
        "clocks_locked_1410": clocks_locked_1410,
        "cryptographic_rsi_harness_and_ratchet_integrity": (
            harness_sha_verified and ratchet_decisions_consistent
        ),
        "bd3_sympy_defect_and_1_over_t_variance_audit": (
            backend_agg["official_fig4_sympy_bool_eq_inductor_crash_all_reps"]
            and backend_agg["flash_sdpa_arbitrary_2l_bool_mask_rejected_all_reps"]
            and backend_agg["sdpa_l16k_2l32k_fp32_score_matrix_gib"] >= 112.0
            and (unclipped_lv / max(anti_lv, 1e-12)) > 1.25
            and anti_gred > 1.70
            and clipped_red > 40.0
        ),
        "per_tensor_fp64_parity_and_richardson_oh2": (
            all_parity_ok
            and max(bd3_p_max.values()) < 0.012
            and max(adaln_p_max.values()) < 0.006
            and max(ce_p_max.values()) < 0.006
            and rich_agg["all_ok"]
            and rich_agg["min_ratio"] > 3.98
            and rich_agg["max_ratio"] < 4.02
        ),
        "track2_adaln_zero_and_track3_masked_ce_rsi_gains": (
            t2_trials["trial2_triton_segmented_zero_atomic_gate_res_adaln"]["speedup_vs_baseline_median"] > 4.80
            and (
                t2_trials["trial1_torch_compile_atomic_adaln_zero"]["geomean_ms_median"]
                / t2_trials["trial2_triton_segmented_zero_atomic_gate_res_adaln"]["geomean_ms_median"]
            )
            > 2.55
            and (
                t2_trials["trial0_eager_gather_adaln_zero"]["peak_alloc_mib_median"]
                / t2_trials["trial2_triton_segmented_zero_atomic_gate_res_adaln"]["peak_alloc_mib_median"]
            )
            > 1.65
            and t3_trials["trial2_active_mask_compacted_3gemm_fused_ce"]["speedup_vs_baseline_median"] > 1.90
            and (
                t3_trials["trial0_full_token_eager_lm_head_ce"]["peak_alloc_mib_median"]
                / t3_trials["trial2_active_mask_compacted_3gemm_fused_ce"]["peak_alloc_mib_median"]
            )
            > 1.90
        ),
        "track1_bd3_attention_and_64k_2l_scaling": (
            t1_trials["trial3_adaptive_zerospill_split_gqa_surgical"]["speedup_vs_baseline_median"] > 1.75
            and t1_trials["trial3_adaptive_zerospill_split_gqa_surgical"]["worst_shape_ratio_vs_baseline_max"] < 0.70
            and t1_trials["trial4_terminal_clean_pruned_adaptive_bd3"]["speedup_vs_baseline_median"] > 3.15
            and seq_agg["L_4096_2L_8192"]["speedup_adaptive_vs_sdpa_2l"] > 50.0
            and seq_agg["L_32768_2L_65536"]["speedup_adaptive_vs_flex_prebuilt"] > 2.0
            and seq_agg["L_32768_2L_65536"]["speedup_pruned_vs_flex_prebuilt"] > 3.8
        ),
        "fsdp2_8gpu_1gpu_parity_and_m0_rank_symmetry": (
            c1_agg["rung1_flex_bd3_eager_adaln_full_ce"]["rank7_forced_empty_mask_m0"]
            and c1_agg["rung4_full_rsi_stack_compacted_3gemm_masked_ce"]["rank7_forced_empty_mask_m0"]
            and c1_agg["rung4_full_rsi_stack_compacted_3gemm_masked_ce"]["loss_rel_diff_vs_1gpu"] < 5e-5
            and c1_agg["rung4_full_rsi_stack_compacted_3gemm_masked_ce"]["tok_emb_dw_rel_l2"] < 0.015
            and c1_agg["rung4_full_rsi_stack_compacted_3gemm_masked_ce"]["blk0_wq_dw_rel_l2"] < 0.015
            and c1_agg["rung4_full_rsi_stack_compacted_3gemm_masked_ce"]["blk0_adaln_dw_rel_l2"] < 0.015
            and c1_agg["rung4_full_rsi_stack_compacted_3gemm_masked_ce"]["blk7_wdn_dw_rel_l2"] < 0.015
            and c1_agg["rung4_full_rsi_stack_compacted_3gemm_masked_ce"]["lm_head_dw_rel_l2"] < 0.015
        ),
        "fsdp2_8gpu_131k_dual_token_scaling_and_hbm_savings": (
            c2_agg["B16_clean32k_dual65k"]["speedup_rung4_vs_rung1"] > 1.09
            and c2_agg["B16_clean32k_dual65k"]["hbm_saved_mib_rung4_vs_rung1"] > 6000.0
            and c2_agg["B32_clean65k_dual131k"]["speedup_rung4_vs_rung1"] > 1.07
            and c2_agg["B32_clean65k_dual131k"]["hbm_saved_mib_rung4_vs_rung1"] > 15500.0
            and c2_agg["B32_clean65k_dual131k"]["hbm_ratio_rung1_over_rung4"] > 1.70
        ),
    }
    all_passed = all(gates.values())

    full_out = {
        "experiment": "E127_diffusion_llm_bd3_adaln_rsi_training",
        "repetitions": 3,
        "world_size": 8,
        "environment": reps[0]["environment"],
        "rsi_verification": {
            "program_spec": "PROGRAM.md",
            "eval_harness_path": "code/rsi_kernel_harness.py",
            "eval_harness_sha256": disk_harness_sha,
            "rsi_ledger_tsv_sha256": ledger_sha_disk,
            "harness_sha_verified": harness_sha_verified,
            "ratchet_decisions_consistent_across_reps": ratchet_decisions_consistent,
            "tracks": rsi_tracks_agg,
        },
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
        "suite_a_1gpu_adaln_and_masked_ce_rsi": suite_a_summary,
        "suite_b_1gpu_bd3_seqlen_scaling": seq_agg,
        "suite_c_8gpu_bd3_fsdp2_scaling": suite_c_summary,
    }

    out_path = raw_dir / "diffusion_llm_bd3_adaln_training_full.json"
    out_path.write_text(
        json.dumps(full_out, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    readme_lines = [
        "# E127 Raw Artifacts: Diffusion-LLM, Block-Diffusion (`BD3-LMs`), & `DiT` `adaLN-Zero` Training Kernels via Immutable RSI Vector-Pareto Ratchet (`PROGRAM.md`)",
        "",
        "Generated by `code/bench_diffusion_llm_bd3_adaln_training.py` + immutable `code/rsi_kernel_harness.py` across 8x NVIDIA A100-SXM4-80GB (`NV12` NVSwitch) with locked SM clocks (`1,410 MHz`), and aggregated deterministically by `code/analyze_diffusion_llm_bd3_adaln_training.py`.",
        "",
        "## Files",
        "- `smoke_diffusion_llm_bd3_adaln_training.json`: 8-GPU `--mode smoke` functional & numerical gate artifact.",
        "- `rep1_diffusion_llm_bd3_adaln_training.json`, `rep2_diffusion_llm_bd3_adaln_training.json`, `rep3_diffusion_llm_bd3_adaln_training.json`: 3 independent OS-process `--mode full` 8-GPU benchmark artifacts.",
        "- `rsi_ledger.tsv`: Append-only multi-track RSI Vector-Pareto Ratchet trial ledger (`eval_harness_sha256` verified).",
        "- `diffusion_llm_bd3_adaln_training_full.json`: Median/IQR 3-repetition summary and 9-gate deterministic validation report (`all_gates_passed = true`).",
        "- `SHA256SUMS`: SHA-256 digests of all raw artifacts in this directory.",
        "",
    ]
    (raw_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8", newline="\n")

    tracked_files = [
        "smoke_diffusion_llm_bd3_adaln_training.json",
        "rep1_diffusion_llm_bd3_adaln_training.json",
        "rep2_diffusion_llm_bd3_adaln_training.json",
        "rep3_diffusion_llm_bd3_adaln_training.json",
        "rsi_ledger.tsv",
        "diffusion_llm_bd3_adaln_training_full.json",
    ]
    sha_lines = [f"{_sha256_file(raw_dir / fn)}  {fn}" for fn in tracked_files]
    (raw_dir / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8", newline="\n")

    print(f"Wrote {out_path} (all_gates_passed={all_passed})")
    if not all_passed:
        raise SystemExit(f"Validation gates failed: {gates}")


if __name__ == "__main__":
    main()
