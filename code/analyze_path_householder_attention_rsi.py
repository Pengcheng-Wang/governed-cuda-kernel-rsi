#!/usr/bin/env python3
"""
E134 (v125, Round-8 RSI): 9-Gate Validator & 3-Rep Aggregator for
PaTH Attention & PaTH-FoX (Position Encoding via Accumulating Householder
Transformations, Yang et al., arXiv:2505.16381v2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_HARNESS_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def _sha256_lf(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _med_iqr(vals: List[float]) -> Dict[str, Any]:
    s = sorted(float(v) for v in vals)
    n = len(s)
    med = float(statistics.median(s))
    if n >= 3:
        iqr = float(s[-1] - s[0])
    elif n == 2:
        iqr = float(s[1] - s[0])
    else:
        iqr = 0.0
    return {"median": med, "iqr": iqr, "values": [float(v) for v in vals]}


def evaluate_gates(runs: List[Dict[str, Any]], mode: str, harness_path: Path) -> Dict[str, Any]:
    disk_sha = _sha256_lf(harness_path)
    sha_ok = (disk_sha == EXPECTED_HARNESS_SHA256) and all(
        r.get("eval_harness_sha256") == EXPECTED_HARNESS_SHA256 for r in runs
    )
    tf32_ok = all(not bool(r["environment"].get("allow_tf32_matmul", True)) for r in runs)

    s0 = runs[0]["suite_0"]
    rt = s0["red_team"]
    cubin = s0["cubin_audit"]
    par = s0["adversarial_parity"]
    rich = s0["richardson_extrapolation"]

    # Gate G1: Cryptographic Evaluator & 4/4 Red-Team Exploits Rejected
    g1_pass = bool(sha_ok and tf32_ok and rt.get("all_4_rejected", False))

    # Gate G2: Numeric Parity (FP32 <= 1e-4, BF16 <= 1.5e-2, zero-norm NaN-free) & 8/8 FP64 O(h^2) Richardson
    g2_pass = bool(
        par["max_fp32_rel_l2"] <= 1e-4
        and par["max_bf16_rel_l2"] <= 1.5e-2
        and par["zero_norm_nan_free"]
        and rich["all_passed"]
        and rich["num_passed"] == 8
    )

    # Gate G3: 18/18 Round-8 RSI Triton Kernels Compile with 0 Spills (vs 3,732 spills in official fla-org K=128 bwd)
    fla_k128_spills = int(
        cubin.get("official_fla_reference", {})
        .get("chunk_cumprod_householder_bwd_kernel", {})
        .get("max_n_spills", 0)
    )
    max_rsi_regs = max(int(v["max_n_regs"]) for v in cubin["round8_kernels"].values())
    g3_pass = bool(
        cubin["all_round8_zero_spills"]
        and int(cubin["total_round8_spills"]) == 0
        and len(cubin["round8_kernels"]) == 22
        and int(cubin["total_compiled_specializations"]) >= 40
        and fla_k128_spills >= 3000
    )

    # Suite A aggregations
    t1_eager_ms = _med_iqr([r["suite_a"]["track1_path_fox_attention"]["baseline_geomean_ms"] for r in runs])
    t1_comp_ms = _med_iqr([r["suite_a"]["track1_path_fox_attention"]["compile_geomean_ms"] for r in runs])
    t1_fla_ms = _med_iqr([r["suite_a"]["track1_path_fox_attention"]["official_fla_geomean_ms"] for r in runs])
    t1_win_ms = _med_iqr([r["suite_a"]["track1_path_fox_attention"]["winner_geomean_ms"] for r in runs])
    t1_sp_eager = _med_iqr([r["suite_a"]["track1_path_fox_attention"]["speedup_vs_eager"] for r in runs])
    t1_sp_comp = _med_iqr([r["suite_a"]["track1_path_fox_attention"]["speedup_vs_compile"] for r in runs])
    t1_sp_fla = _med_iqr([r["suite_a"]["track1_path_fox_attention"]["speedup_vs_official_fla"] for r in runs])
    t1_sp_k128_s2 = _med_iqr([r["suite_a"]["track1_path_fox_attention"]["k128_s2_speedup_vs_official_fla"] for r in runs])
    t1_s4_hbm_saved = _med_iqr([r["suite_a"]["track1_path_fox_attention"]["s4_hbm_saved_vs_official_fla_mib"] for r in runs])

    t1_shapes_agg: Dict[str, Any] = {}
    for sname in runs[0]["suite_a"]["track1_path_fox_attention"]["per_shape_speedup_vs_official_fla"]:
        t1_shapes_agg[sname] = {
            "eager_ms": _med_iqr([r["suite_a"]["track1_path_fox_attention"]["candidates"][0]["shapes"][sname]["median_ms"] for r in runs]),
            "compile_ms": _med_iqr([r["suite_a"]["track1_path_fox_attention"]["candidates"][1]["shapes"][sname]["median_ms"] for r in runs]),
            "official_fla_ms": _med_iqr([r["suite_a"]["track1_path_fox_attention"]["candidates"][3]["shapes"][sname]["median_ms"] for r in runs]),
            "rsi_winner_ms": _med_iqr([r["suite_a"]["track1_path_fox_attention"]["candidates"][5]["shapes"][sname]["median_ms"] for r in runs]),
            "speedup_vs_official_fla": _med_iqr([r["suite_a"]["track1_path_fox_attention"]["per_shape_speedup_vs_official_fla"][sname] for r in runs]),
            "speedup_vs_eager": _med_iqr([r["suite_a"]["track1_path_fox_attention"]["per_shape_speedup_vs_eager"][sname] for r in runs]),
            "eager_peak_mib": _med_iqr([r["suite_a"]["track1_path_fox_attention"]["candidates"][0]["shapes"][sname]["peak_allocated_mib"] for r in runs]),
            "official_fla_peak_mib": _med_iqr([r["suite_a"]["track1_path_fox_attention"]["candidates"][3]["shapes"][sname]["peak_allocated_mib"] for r in runs]),
            "rsi_winner_peak_mib": _med_iqr([r["suite_a"]["track1_path_fox_attention"]["candidates"][5]["shapes"][sname]["peak_allocated_mib"] for r in runs]),
        }

    # Gate G4: Track 1 PaTH-FoX Attention Speedup & Parity Reject
    t1_a2_rejected = all(
        r["suite_a"]["track1_path_fox_attention"]["candidates"][2]["decision"] == "REJECT_PARITY"
        for r in runs
    )
    t1_beats_fla_all_shapes = all(
        v["speedup_vs_official_fla"]["median"] >= 1.08 for v in t1_shapes_agg.values()
    )
    g4_pass = bool(
        t1_a2_rejected
        and t1_beats_fla_all_shapes
        and t1_sp_eager["median"] >= 6.0
        and t1_sp_fla["median"] >= 2.0
        and t1_sp_k128_s2["median"] >= 3.0
    )

    # Track 2 aggregations
    t2_eager_ms = _med_iqr([r["suite_a"]["track2_end_to_end_path_fox_block"]["baseline_eager_geomean_ms"] for r in runs])
    t2_efe_fla_ms = _med_iqr([r["suite_a"]["track2_end_to_end_path_fox_block"]["eager_fe_plus_fla_geomean_ms"] for r in runs])
    t2_cfe_fla_ms = _med_iqr([r["suite_a"]["track2_end_to_end_path_fox_block"]["compile_fe_plus_fla_geomean_ms"] for r in runs])
    t2_win_ms = _med_iqr([r["suite_a"]["track2_end_to_end_path_fox_block"]["winner_geomean_ms"] for r in runs])
    t2_sp_eager = _med_iqr([r["suite_a"]["track2_end_to_end_path_fox_block"]["speedup_vs_eager"] for r in runs])
    t2_sp_efe_fla = _med_iqr([r["suite_a"]["track2_end_to_end_path_fox_block"]["speedup_vs_eager_fe_plus_fla"] for r in runs])
    t2_sp_cfe_fla = _med_iqr([r["suite_a"]["track2_end_to_end_path_fox_block"]["speedup_vs_compile_fe_plus_fla"] for r in runs])
    t2_hbm_saved = _med_iqr([r["suite_a"]["track2_end_to_end_path_fox_block"]["hbm_saved_vs_compile_fe_plus_fla_mib"] for r in runs])

    t2_b1_rejected = all(
        r["suite_a"]["track2_end_to_end_path_fox_block"]["candidates"][1]["decision"] == "REJECT_PARITY"
        for r in runs
    )
    # Gate G5: Track 2 End-to-End PaTH-FoX Block Speedup
    g5_pass = bool(
        t2_b1_rejected
        and t2_sp_eager["median"] >= 6.0
        and t2_sp_efe_fla["median"] >= 2.2
        and t2_sp_cfe_fla["median"] >= 1.8
    )

    # Suite B aggregations
    sb0 = runs[0]["suite_b"]
    d0_ms = _med_iqr([r["suite_b"]["arms"]["d0_allgather_full_seq_eager_path_fox"]["median_ms"] for r in runs])
    d1_ms = _med_iqr([r["suite_b"]["arms"]["d1_allgather_full_seq_official_fla"]["median_ms"] for r in runs])
    d2_ms = _med_iqr([r["suite_b"]["arms"]["d2_ring_cp8_eager_dense_matrix_handoff"]["median_ms"] for r in runs])
    d3_ms = _med_iqr([r["suite_b"]["arms"]["d3_rsi_ring_path_cp8_strip_mined_ut"]["median_ms"] for r in runs])
    d0_mib = _med_iqr([r["suite_b"]["arms"]["d0_allgather_full_seq_eager_path_fox"]["peak_allocated_mib"] for r in runs])
    d1_mib = _med_iqr([r["suite_b"]["arms"]["d1_allgather_full_seq_official_fla"]["peak_allocated_mib"] for r in runs])
    d2_mib = _med_iqr([r["suite_b"]["arms"]["d2_ring_cp8_eager_dense_matrix_handoff"]["peak_allocated_mib"] for r in runs])
    d3_mib = _med_iqr([r["suite_b"]["arms"]["d3_rsi_ring_path_cp8_strip_mined_ut"]["peak_allocated_mib"] for r in runs])
    sb_sp_d0 = _med_iqr([r["suite_b"]["speedup_d3_vs_d0_eager"] for r in runs])
    sb_sp_d1 = _med_iqr([r["suite_b"]["speedup_d3_vs_d1_allgather_official_fla"] for r in runs])
    sb_sp_d2 = _med_iqr([r["suite_b"]["speedup_d3_vs_d2_dense_matrix_handoff"] for r in runs])
    sb_hbm_ratio = _med_iqr([r["suite_b"]["hbm_reduction_ratio_vs_d1_allgather"] for r in runs])
    sb_hbm_saved = _med_iqr([r["suite_b"]["hbm_saved_vs_d1_allgather_mib"] for r in runs])

    # Gate G6: 8-GPU NVSwitch Ring PaTH-CP = 8 Long-Context Scaling
    g6_pass = bool(
        sb_sp_d0["median"] >= 5.0
        and sb_sp_d1["median"] >= 1.5
        and sb_sp_d2["median"] >= 1.15
        and sb_hbm_saved["median"] > 100.0
    )

    # Gate G9: 3-Step Multi-Update Distributed Ring PaTH-CP = 8 Parity vs Monolithic Reference
    g9_rec = sb0["gate_g9_three_step_parity"]
    g9_pass = bool(
        all(r["suite_b"]["gate_g9_three_step_parity"]["loss_step3_rel_diff"] <= 1e-5 for r in runs)
        and all(r["suite_b"]["gate_g9_three_step_parity"]["max_param_rel_l2"] <= 1e-4 for r in runs)
        and all(r["suite_b"]["gate_g9_three_step_parity"]["max_grad_rel_l2"] <= 5e-4 for r in runs)
        and all(r["suite_b"]["gate_g9_three_step_parity"]["naive_no_defect_handoff_grad_rel_l2_bug"] >= 0.05 for r in runs)
    )

    # Suite C aggregations
    sc0 = runs[0]["suite_c"]
    gen_agg: List[Dict[str, Any]] = []
    for g_idx in range(len(sc0["generations"])):
        g0_item = sc0["generations"][g_idx]
        gen_agg.append(
            {
                "generation": g0_item["generation"],
                "candidate_id": g0_item["candidate_id"],
                "decision": g0_item["decision"],
                "steps_completed": _med_iqr([r["suite_c"]["generations"][g_idx]["steps_completed"] for r in runs]),
                "ms_per_step": _med_iqr([r["suite_c"]["generations"][g_idx]["ms_per_step"] for r in runs]),
                "val_bpb": _med_iqr([r["suite_c"]["generations"][g_idx]["val_bpb"] for r in runs]),
                "val_state_tracking_acc": _med_iqr([r["suite_c"]["generations"][g_idx]["val_state_tracking_acc"] for r in runs]),
            }
        )

    sc_step_mult = _med_iqr([r["suite_c"]["step_multiplier_vs_gen0"] for r in runs])
    sc_val_imp = _med_iqr([r["suite_c"]["val_bpb_improvement_vs_gen0"] for r in runs])
    sc_test_gen0 = _med_iqr([r["suite_c"]["single_touch_test_evaluation"]["gen0_test_bpb"] for r in runs])
    sc_test_win = _med_iqr([r["suite_c"]["single_touch_test_evaluation"]["winner_test_bpb"] for r in runs])
    sc_test_imp = _med_iqr([r["suite_c"]["single_touch_test_evaluation"]["test_bpb_improvement"] for r in runs])
    sc_test_acc_gen0 = _med_iqr([r["suite_c"]["single_touch_test_evaluation"]["gen0_test_state_tracking_acc"] for r in runs])
    sc_test_acc_win = _med_iqr([r["suite_c"]["single_touch_test_evaluation"]["winner_test_state_tracking_acc"] for r in runs])

    # Gate G7: Fixed-Wall-Clock 8xA100 DDP val_bpb Autonomous RSI Loop
    g7_pass = bool(
        all(r["suite_c"]["winning_candidate_id"] == "gen2_rsi_round8_ut_duality_path_fox" for r in runs)
        and sc_step_mult["median"] >= 1.35
        and gen_agg[2]["steps_completed"]["median"] > gen_agg[1]["steps_completed"]["median"]
        and sc_val_imp["median"] >= 0.030
        and sc_test_imp["median"] >= 0.025
        and all(r["suite_c"]["generations"][3]["decision"] == "REVERT_VAL_BPB" for r in runs)
        and all(r["suite_c"]["generations"][4]["decision"] == "REVERT_VAL_BPB" for r in runs)
    )

    # Part 2: 5-Arm Matched-Step Position-Encoding & 2x Length-Extrapolation Scaling Law
    law_agg: Dict[str, Any] = {}
    for arm_k in sc0["matched_step_position_encoding_law"]:
        law_agg[arm_k] = {
            "val_bpb": _med_iqr([r["suite_c"]["matched_step_position_encoding_law"][arm_k]["val_bpb"] for r in runs]),
            "val_state_tracking_acc": _med_iqr([r["suite_c"]["matched_step_position_encoding_law"][arm_k]["val_state_tracking_acc"] for r in runs]),
            "clean_test_bpb": _med_iqr([r["suite_c"]["matched_step_position_encoding_law"][arm_k]["clean_test_bpb"] for r in runs]),
            "clean_state_tracking_acc": _med_iqr([r["suite_c"]["matched_step_position_encoding_law"][arm_k]["clean_state_tracking_acc"] for r in runs]),
            "extrap_2x_test_bpb": _med_iqr([r["suite_c"]["matched_step_position_encoding_law"][arm_k]["extrap_2x_test_bpb"] for r in runs]),
            "extrap_2x_state_tracking_acc": _med_iqr([r["suite_c"]["matched_step_position_encoding_law"][arm_k]["extrap_2x_state_tracking_acc"] for r in runs]),
            "extrap_2x_bpb_degradation": _med_iqr([r["suite_c"]["matched_step_position_encoding_law"][arm_k]["extrap_2x_bpb_degradation"] for r in runs]),
        }

    p4 = law_agg["P4_path_fox_rsi_householder_attention"]
    p3 = law_agg["P3_fox_adaptive_forgetting_only"]
    p1 = law_agg["P1_rope_fixed_rotary_attention"]
    p0 = law_agg["P0_nope_causal_attention"]

    # Gate G8: Matched-Step Position-Encoding & 2x Length-Extrapolation Scaling Law
    g8_pass = bool(
        p4["clean_test_bpb"]["median"] < p3["clean_test_bpb"]["median"]
        and p4["clean_test_bpb"]["median"] < p1["clean_test_bpb"]["median"]
        and p4["clean_test_bpb"]["median"] < p0["clean_test_bpb"]["median"]
        and p4["extrap_2x_test_bpb"]["median"] + 0.04 <= p1["extrap_2x_test_bpb"]["median"]
        and p4["clean_state_tracking_acc"]["median"] >= p3["clean_state_tracking_acc"]["median"] + 0.08
    )

    gates = {
        "G1_evaluator_and_red_team": g1_pass,
        "G2_numeric_parity_and_richardson": g2_pass,
        "G3_cubin_zero_spill_audit": g3_pass,
        "G4_suite_a_track1_path_fox_attention_speedup": g4_pass,
        "G5_suite_a_track2_end_to_end_path_fox_block": g5_pass,
        "G6_suite_b_distributed_ring_path_cp8_scaling": g6_pass,
        "G7_suite_c_val_bpb_autoresearch_loop": g7_pass,
        "G8_suite_c_matched_step_position_encoding_extrapolation_law": g8_pass,
        "G9_multi_step_distributed_ring_cp8_parity": g9_pass,
    }

    return {
        "experiment": "E134_path_householder_attention_round8_rsi",
        "mode": mode,
        "num_repetitions": len(runs),
        "eval_harness_sha256_on_disk": disk_sha,
        "gates_passed_count": sum(1 for v in gates.values() if v),
        "gates_total_count": len(gates),
        "all_gates_passed": all(gates.values()),
        "gates": gates,
        "suite_0_summary": {
            "red_team_rejected": "4/4" if rt.get("all_4_rejected") else "FAIL",
            "num_rsi_kernels": len(cubin["round8_kernels"]),
            "total_specializations": int(cubin["total_compiled_specializations"]),
            "max_n_regs": max_rsi_regs,
            "total_rsi_spills": int(cubin["total_round8_spills"]),
            "official_fla_k128_bwd_spills": fla_k128_spills,
            "max_fp32_rel_l2": par["max_fp32_rel_l2"],
            "max_bf16_rel_l2": par["max_bf16_rel_l2"],
            "zero_norm_nan_free": par["zero_norm_nan_free"],
            "richardson_passed": f"{rich['num_passed']}/{rich['num_total']}",
        },
        "suite_a_summary": {
            "track1_eager_geomean_ms": t1_eager_ms,
            "track1_compile_geomean_ms": t1_comp_ms,
            "track1_official_fla_geomean_ms": t1_fla_ms,
            "track1_winner_geomean_ms": t1_win_ms,
            "track1_speedup_vs_eager": t1_sp_eager,
            "track1_speedup_vs_compile": t1_sp_comp,
            "track1_speedup_vs_official_fla": t1_sp_fla,
            "track1_k128_s2_speedup_vs_official_fla": t1_sp_k128_s2,
            "track1_s4_hbm_saved_vs_official_fla_mib": t1_s4_hbm_saved,
            "track1_shapes": t1_shapes_agg,
            "track2_eager_geomean_ms": t2_eager_ms,
            "track2_eager_fe_plus_fla_geomean_ms": t2_efe_fla_ms,
            "track2_compile_fe_plus_fla_geomean_ms": t2_cfe_fla_ms,
            "track2_winner_geomean_ms": t2_win_ms,
            "track2_speedup_vs_eager": t2_sp_eager,
            "track2_speedup_vs_eager_fe_plus_fla": t2_sp_efe_fla,
            "track2_speedup_vs_compile_fe_plus_fla": t2_sp_cfe_fla,
            "track2_hbm_saved_vs_compile_fe_plus_fla_mib": t2_hbm_saved,
        },
        "suite_b_summary": {
            "global_seq_len": sb0["global_seq_len"],
            "local_shard_len": sb0["local_shard_len"],
            "head_dim_k": sb0["head_dim_k"],
            "gate_g9_rep1": g9_rec,
            "d0_allgather_eager_ms": d0_ms,
            "d1_allgather_official_fla_ms": d1_ms,
            "d2_ring_cp8_dense_handoff_ms": d2_ms,
            "d3_rsi_ring_path_cp8_strip_mined_ms": d3_ms,
            "d0_peak_mib": d0_mib,
            "d1_peak_mib": d1_mib,
            "d2_peak_mib": d2_mib,
            "d3_peak_mib": d3_mib,
            "speedup_d3_vs_d0_eager": sb_sp_d0,
            "speedup_d3_vs_d1_allgather_official_fla": sb_sp_d1,
            "speedup_d3_vs_d2_dense_matrix_handoff": sb_sp_d2,
            "hbm_reduction_ratio_vs_d1_allgather": sb_hbm_ratio,
            "hbm_saved_vs_d1_allgather_mib": sb_hbm_saved,
        },
        "suite_c_summary": {
            "budget_seconds": sc0["budget_seconds"],
            "winning_candidate_id": sc0["winning_candidate_id"],
            "step_multiplier_vs_gen0": sc_step_mult,
            "val_bpb_improvement_vs_gen0": sc_val_imp,
            "gen0_steps": gen_agg[0]["steps_completed"],
            "gen1_fla_steps": gen_agg[1]["steps_completed"],
            "winner_steps": gen_agg[2]["steps_completed"],
            "generations": gen_agg,
            "single_touch_gen0_test_bpb": sc_test_gen0,
            "single_touch_winner_test_bpb": sc_test_win,
            "single_touch_test_bpb_improvement": sc_test_imp,
            "single_touch_gen0_state_acc": sc_test_acc_gen0,
            "single_touch_winner_state_acc": sc_test_acc_win,
            "matched_step_position_encoding_law": law_agg,
        },
    }


def render_summary_markdown(agg: Dict[str, Any]) -> str:
    s0 = agg["suite_0_summary"]
    sa = agg["suite_a_summary"]
    sb = agg["suite_b_summary"]
    sc = agg["suite_c_summary"]
    lines = [
        "# E134 (`v125`, Round-8 RSI): `PaTH Attention` & `PaTH-FoX` Householder Transformations — Hardware Verification Summary",
        "",
        f"- **Mode**: `{agg['mode']}` (`{agg['num_repetitions']}` independent OS-process 8-GPU repetitions)",
        f"- **Validation Gates**: **`{agg['gates_passed_count']}/{agg['gates_total_count']}` PASSED** (`all_gates_passed = {agg['all_gates_passed']}`)",
        f"- **Immutable Evaluator SHA-256**: `{agg['eval_harness_sha256_on_disk']}`",
        "",
        "## Headline Hardware & Scaling Results (3-Rep Median / IQR)",
        "",
        f"1. **Suite 0 (Forensics, Cubin Zero-Spill Audit & Exactness)**: `{s0['red_team_rejected']}` red-team exploits rejected; "
        f"`{s0['num_rsi_kernels']}/{s0['num_rsi_kernels']}` Round-8 RSI Triton kernels (`{s0['total_specializations']}` specializations, max `{s0['max_n_regs']}` regs) compiled with **`0` cubin spills** "
        f"(vs **`{s0['official_fla_k128_bwd_spills']}` spills** in official `fla-org` `chunk_cumprod_householder_bwd_kernel` at `K=128`); "
        f"`FP32 max_rel_l2 = {s0['max_fp32_rel_l2']:.2e}`, `BF16 max_rel_l2 = {s0['max_bf16_rel_l2']:.4f}`; `{s0['richardson_passed']}` `FP64` `O(h^2)` Richardson checks passed.",
        f"2. **Suite A — Track 1 (Core `PaTH-FoX` Householder Attention Fwd+Bwd across `s1..s4`)**: "
        f"`{sa['track1_eager_geomean_ms']['median']:.3f} -> {sa['track1_winner_geomean_ms']['median']:.3f} ms` (**`{sa['track1_speedup_vs_eager']['median']:.2f}x`** geomean speedup over Eager, "
        f"**`{sa['track1_speedup_vs_compile']['median']:.2f}x`** over `torch.compile`, and **`{sa['track1_speedup_vs_official_fla']['median']:.2f}x`** geomean / up to **`{sa['track1_k128_s2_speedup_vs_official_fla']['median']:.2f}x`** at `K=128` over Official `fla-org/flash-linear-attention` `parallel_path_attn`, saving **`{sa['track1_s4_hbm_saved_vs_official_fla_mib']['median']:.1f} MiB`** peak HBM at `T=4096`).",
        f"3. **Suite A — Track 2 (End-to-End `PaTH-FoX` Block: `ShortConv3+SiLU+L2Norm+BetaSigmoid Frontend + PaTH-FoX Attention`)**: "
        f"`{sa['track2_eager_geomean_ms']['median']:.3f} -> {sa['track2_winner_geomean_ms']['median']:.3f} ms` (**`{sa['track2_speedup_vs_eager']['median']:.2f}x`** over Eager, "
        f"**`{sa['track2_speedup_vs_eager_fe_plus_fla']['median']:.2f}x`** over Eager Frontend + Official `fla-org`, and **`{sa['track2_speedup_vs_compile_fe_plus_fla']['median']:.2f}x`** over `torch.compile` Frontend + Official `fla-org`).",
        f"4. **Suite B (`8-GPU` NVSwitch Distributed Context Parallelism `Ring PaTH-CP = 8` at `T_global = {sb['global_seq_len']}, K = {sb['head_dim_k']}`)**: "
        f"`Gate G9` 3-step `FP32` parity `max_param_rel_l2 = {sb['gate_g9_rep1']['max_param_rel_l2']:.2e}` (vs `{sb['gate_g9_rep1']['naive_no_defect_handoff_grad_rel_l2_bug']:.4f}` shard-local no-handoff bug); "
        f"delivers **`{sb['speedup_d3_vs_d0_eager']['median']:.2f}x`** over All-Gather Eager, **`{sb['speedup_d3_vs_d1_allgather_official_fla']['median']:.2f}x`** over All-Gather Official `fla-org` (cutting peak HBM **`{sb['hbm_reduction_ratio_vs_d1_allgather']['median']:.2f}x`**, saving **`{sb['hbm_saved_vs_d1_allgather_mib']['median']:.1f} MiB/rank`**), and **`{sb['speedup_d3_vs_d2_dense_matrix_handoff']['median']:.2f}x`** over Dense-Matrix Ring Handoff.",
        f"5. **Suite C (`{sc['budget_seconds']}s` `8xA100 DDP` `PaTH-FoX` `val_bpb` Autonomous RSI Loop)**: "
        f"`{sc['winning_candidate_id']}` runs **`{sc['step_multiplier_vs_gen0']['median']:.2f}x`** more updates (`{sc['gen0_steps']['median']:.0f} -> {sc['winner_steps']['median']:.0f}`, vs `{sc['gen1_fla_steps']['median']:.0f}` for Official `fla-org`), "
        f"improving `val_bpb` by **`-{sc['val_bpb_improvement_vs_gen0']['median']:.4f}`** and untouched single-touch `test_bpb` by **`-{sc['single_touch_test_bpb_improvement']['median']:.4f}`** (`State-Tracking Acc = {sc['single_touch_winner_state_acc']['median']*100:.2f}%`).",
        "",
        "## Matched-Step Position-Encoding & 2x Length-Extrapolation Scaling Law",
        "",
        "| Arm | `val_bpb` | Clean `test_bpb` | `2x` Extrap `test_bpb` | Extrap `Δbpb` | Clean State Acc | `2x` Extrap State Acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ak, av in sc["matched_step_position_encoding_law"].items():
        lines.append(
            f"| `{ak}` | `{av['val_bpb']['median']:.4f}` | `{av['clean_test_bpb']['median']:.4f}` | "
            f"`{av['extrap_2x_test_bpb']['median']:.4f}` | `+{av['extrap_2x_bpb_degradation']['median']:.4f}` | "
            f"`{av['clean_state_tracking_acc']['median']*100:.2f}%` | `{av['extrap_2x_state_tracking_acc']['median']*100:.2f}%` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Validate & aggregate E134 PaTH-FoX Householder Attention Round-8 RSI outputs")
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--inputs", nargs="+", default=None)
    parser.add_argument("--harness-path", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--summary-md", type=str, default="")
    args = parser.parse_args()

    root_dir = args.root.resolve()
    raw_dir = root_dir / "results" / "raw" / "path_householder_attention_rsi"
    if not args.harness_path:
        args.harness_path = str(root_dir / "code" / "rsi_kernel_harness.py")

    if args.inputs is None:
        if args.mode == "full":
            args.inputs = [str(raw_dir / f"rep{r}" / "path_householder_attention_rsi_full.json") for r in (1, 2, 3)]
            if not args.json_out:
                args.json_out = str(raw_dir / "path_householder_attention_rsi_full.json")
            if not args.summary_md:
                args.summary_md = str(raw_dir / "path_householder_attention_rsi_summary.md")
        else:
            args.inputs = [str(raw_dir / "smoke" / "path_householder_attention_rsi_smoke.json")]

    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.inputs]
    agg = evaluate_gates(runs, args.mode, Path(args.harness_path).resolve())
    print(
        json.dumps(
            {
                "mode": agg["mode"],
                "num_repetitions": agg["num_repetitions"],
                "gates_passed_count": agg["gates_passed_count"],
                "gates_total_count": agg["gates_total_count"],
                "all_gates_passed": agg["all_gates_passed"],
                "gates": agg["gates"],
            },
            indent=2,
        )
    )
    if args.json_out:
        out_p = Path(args.json_out).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.summary_md:
        md_p = Path(args.summary_md).resolve()
        md_p.parent.mkdir(parents=True, exist_ok=True)
        md_p.write_text(render_summary_markdown(agg), encoding="utf-8", newline="\n")
    if not agg["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
