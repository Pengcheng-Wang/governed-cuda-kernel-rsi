#!/usr/bin/env python3
"""
E135 (v126, Round-9 RSI): 9-Gate Validator & 3-Rep Aggregator for
Mixture-of-Recursions (MoR: Learning Dynamic Recursive Depths for Adaptive
Token-Level Computation, Bae et al., arXiv:2507.10524v2, NeurIPS 2025;
official repository `raymin0223/mixture_of_recursions` commit `82c97f29cd140efa01f5cc4fe0de73a4c3b6ffd6`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
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
    forensics = s0["upstream_forensics"]

    # Gate G1: Cryptographic Evaluator & 4/4 Red-Team Exploits Rejected
    g1_pass = bool(sha_ok and tf32_ok and rt.get("all_4_rejected", False))

    # Gate G2: Numeric Parity (FP32 <= 1e-4, BF16 <= 1.5e-2, zero-consumer key exact 0.0) & 7/7 FP64 O(h^2) Richardson
    g2_pass = bool(
        par["max_fp32_rel_l2"] <= 1e-4
        and par["max_bf16_rel_l2"] <= 1.5e-2
        and par["zero_consumer_key_exact_zero_grad"]
        and rich["all_passed"]
        and int(rich["num_passed"]) == 7
    )

    # Gate G3: 16/16 Round-9 RSI Triton Kernels Compile with 0 Spills (vs monolithic [64x128] dKV control)
    naive_k128_spills = int(
        cubin.get("naive_monolithic_64x128_dkv_control", {}).get("max_n_spills", 0)
    )
    max_rsi_regs = max(int(v["max_n_regs"]) for v in cubin["round9_kernels"].values())
    g3_pass = bool(
        cubin["all_round9_zero_spills"]
        and int(cubin["total_round9_spills"]) == 0
        and int(cubin["num_round9_kernels"]) == 16
        and int(cubin["total_compiled_specializations"]) >= 24
    )

    # Suite A aggregations — Track 1
    t1_off_ms = _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["baseline_official_kvshare_geomean_ms"] for r in runs])
    t1_sdpa_ms = _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["sdpa_4d_mask_geomean_ms"] for r in runs])
    t1_spill_ms = _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["monolithic_spill_geomean_ms"] for r in runs])
    t1_win_ms = _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["winner_geomean_ms"] for r in runs])
    t1_sp_off = _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["speedup_vs_official_kvshare"] for r in runs])
    t1_sp_sdpa = _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["speedup_vs_sdpa_4d_mask"] for r in runs])
    t1_sp_spill = _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["speedup_vs_monolithic_spill"] for r in runs])
    t1_s4_hbm_saved = _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["s4_hbm_saved_vs_official_mib"] for r in runs])

    t1_shapes_agg: Dict[str, Any] = {}
    for sname in runs[0]["suite_a"]["track1_rect_flash_mor_attention"]["per_shape_speedup_vs_official"]:
        t1_shapes_agg[sname] = {
            "official_kvshare_ms": _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["candidates"][0]["shapes"][sname]["median_ms"] for r in runs]),
            "sdpa_4d_mask_ms": _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["candidates"][1]["shapes"][sname]["median_ms"] for r in runs]),
            "monolithic_spill_ms": _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["candidates"][4]["shapes"][sname]["median_ms"] for r in runs]),
            "rsi_winner_ms": _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["candidates"][5]["shapes"][sname]["median_ms"] for r in runs]),
            "speedup_vs_official": _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["per_shape_speedup_vs_official"][sname] for r in runs]),
            "speedup_vs_sdpa": _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["per_shape_speedup_vs_sdpa"][sname] for r in runs]),
            "official_peak_mib": _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["candidates"][0]["shapes"][sname]["peak_allocated_mib"] for r in runs]),
            "sdpa_peak_mib": _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["candidates"][1]["shapes"][sname]["peak_allocated_mib"] for r in runs]),
            "rsi_winner_peak_mib": _med_iqr([r["suite_a"]["track1_rect_flash_mor_attention"]["candidates"][5]["shapes"][sname]["peak_allocated_mib"] for r in runs]),
        }

    # Gate G4: Track 1 Rectangular Flash-MoR Speedup & Rejections
    t1_a2_rejected = all(
        r["suite_a"]["track1_rect_flash_mor_attention"]["candidates"][2]["decision"] == "REJECT_PARITY"
        for r in runs
    )
    t1_a3_rejected = all(
        r["suite_a"]["track1_rect_flash_mor_attention"]["candidates"][3]["decision"] == "REJECT_PARITY"
        for r in runs
    )
    t1_beats_off_all = all(
        v["speedup_vs_official"]["median"] >= 1.15 for v in t1_shapes_agg.values()
    )
    g4_pass = bool(
        t1_a2_rejected
        and t1_a3_rejected
        and t1_beats_off_all
        and t1_sp_off["median"] >= 1.30
        and t1_sp_sdpa["median"] >= 2.0
    )

    # Suite A aggregations — Track 2
    t2_pad4d_ms = _med_iqr([r["suite_a"]["track2_end_to_end_mor_stack"]["baseline_pad4d_geomean_ms"] for r in runs])
    t2_kvclone_ms = _med_iqr([r["suite_a"]["track2_end_to_end_mor_stack"]["official_kvshare_clone_geomean_ms"] for r in runs])
    t2_win_ms = _med_iqr([r["suite_a"]["track2_end_to_end_mor_stack"]["winner_geomean_ms"] for r in runs])
    t2_sp_pad4d = _med_iqr([r["suite_a"]["track2_end_to_end_mor_stack"]["speedup_vs_pad4d"] for r in runs])
    t2_sp_kvclone = _med_iqr([r["suite_a"]["track2_end_to_end_mor_stack"]["speedup_vs_official_kvshare_clone"] for r in runs])
    t2_hbm_saved_pad4d = _med_iqr([r["suite_a"]["track2_end_to_end_mor_stack"]["hbm_saved_vs_pad4d_mib"] for r in runs])
    t2_hbm_saved_kvclone = _med_iqr([r["suite_a"]["track2_end_to_end_mor_stack"]["hbm_saved_vs_official_kvshare_clone_mib"] for r in runs])

    t2_b2_rejected = all(
        r["suite_a"]["track2_end_to_end_mor_stack"]["candidates"][2]["decision"] == "REJECT_PARITY"
        for r in runs
    )
    # Gate G5: Track 2 End-to-End Middle-Cycle N_r=3 MoR Stack Speedup
    g5_pass = bool(
        t2_b2_rejected
        and t2_sp_pad4d["median"] >= 1.50
        and t2_sp_kvclone["median"] >= 1.25
        and t2_hbm_saved_pad4d["median"] > 50.0
    )

    # Suite B aggregations
    sb0 = runs[0]["suite_b"]
    d0_ms = _med_iqr([r["suite_b"]["arms"]["d0_naive_fsdp2_6gather_official_pad4d"]["median_ms"] for r in runs])
    d1_ms = _med_iqr([r["suite_b"]["arms"]["d1_naive_fsdp2_6gather_official_kvshare_fullseq"]["median_ms"] for r in runs])
    d2_ms = _med_iqr([r["suite_b"]["arms"]["d2_pinned_fsdp2_1gather_local_rsi_unbalanced"]["median_ms"] for r in runs])
    d3_ms = _med_iqr([r["suite_b"]["arms"]["d3_pinned_fsdp2_1gather_mor_lpt_alltoallv_rsi"]["median_ms"] for r in runs])
    d0_mib = _med_iqr([r["suite_b"]["arms"]["d0_naive_fsdp2_6gather_official_pad4d"]["peak_allocated_mib"] for r in runs])
    d1_mib = _med_iqr([r["suite_b"]["arms"]["d1_naive_fsdp2_6gather_official_kvshare_fullseq"]["peak_allocated_mib"] for r in runs])
    d2_mib = _med_iqr([r["suite_b"]["arms"]["d2_pinned_fsdp2_1gather_local_rsi_unbalanced"]["peak_allocated_mib"] for r in runs])
    d3_mib = _med_iqr([r["suite_b"]["arms"]["d3_pinned_fsdp2_1gather_mor_lpt_alltoallv_rsi"]["peak_allocated_mib"] for r in runs])
    sb_sp_d0 = _med_iqr([r["suite_b"]["speedup_d3_vs_d0_pad4d"] for r in runs])
    sb_sp_d1 = _med_iqr([r["suite_b"]["speedup_d3_vs_d1_official_kvshare"] for r in runs])
    sb_sp_d2 = _med_iqr([r["suite_b"]["speedup_d3_vs_d2_unbalanced_rsi"] for r in runs])
    sb_hbm_saved_d0 = _med_iqr([r["suite_b"]["hbm_saved_vs_d0_pad4d_mib"] for r in runs])
    sb_hbm_saved_d1 = _med_iqr([r["suite_b"]["hbm_saved_vs_d1_official_kvshare_mib"] for r in runs])

    g9_rec = sb0["gate_g9_three_step_parity"]
    # Gate G6: 8-GPU NVSwitch Recursion-Pinned FSDP2 + MoR-LPT AllToAllv Scaling
    g6_pass = bool(
        int(g9_rec["fsdp2_default_reshard_collectives"]["all_gather_calls"]) == 6
        and int(g9_rec["fsdp2_pinned_recursion_collectives"]["all_gather_calls"]) == 1
        and sb_sp_d0["median"] >= 1.50
        and sb_sp_d1["median"] >= 1.25
        and sb_sp_d2["median"] >= 1.10
    )

    # Gate G9: 3-Step Multi-Update Distributed MoR-LPT AllToAllv + Global Balancing Parity
    g9_pass = bool(
        all(r["suite_b"]["gate_g9_three_step_parity"]["loss_step3_rel_diff"] <= 1e-5 for r in runs)
        and all(r["suite_b"]["gate_g9_three_step_parity"]["max_param_rel_l2"] <= 1e-4 for r in runs)
        and all(r["suite_b"]["gate_g9_three_step_parity"]["max_grad_rel_l2"] <= 5e-4 for r in runs)
        and all(r["suite_b"]["gate_g9_three_step_parity"]["local_balancing_router_grad_rel_l2_bug"] >= 0.10 for r in runs)
    )

    # Suite C aggregations
    sc0 = runs[0]["suite_c"]
    pref_audit = sc0["prefix_invariance_audit"]
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
                "val_hop1_acc": _med_iqr([r["suite_c"]["generations"][g_idx]["val_hop1_acc"] for r in runs]),
                "val_hop3_acc": _med_iqr([r["suite_c"]["generations"][g_idx]["val_hop3_acc"] for r in runs]),
                "mean_depth_hop1": _med_iqr([r["suite_c"]["generations"][g_idx]["mean_depth_hop1"] for r in runs]),
                "mean_depth_hop3": _med_iqr([r["suite_c"]["generations"][g_idx]["mean_depth_hop3"] for r in runs]),
            }
        )

    sc_step_mult = _med_iqr([r["suite_c"]["step_multiplier_vs_gen0"] for r in runs])
    sc_val_imp = _med_iqr([r["suite_c"]["val_bpb_improvement_vs_gen0"] for r in runs])
    sc_test_gen0 = _med_iqr([r["suite_c"]["single_touch_test_evaluation"]["gen0_test_bpb"] for r in runs])
    sc_test_win = _med_iqr([r["suite_c"]["single_touch_test_evaluation"]["winner_test_bpb"] for r in runs])
    sc_test_imp = _med_iqr([r["suite_c"]["single_touch_test_evaluation"]["test_bpb_improvement"] for r in runs])
    sc_test_hop3_gen0 = _med_iqr([r["suite_c"]["single_touch_test_evaluation"]["gen0_test_hop3_acc"] for r in runs])
    sc_test_hop3_win = _med_iqr([r["suite_c"]["single_touch_test_evaluation"]["winner_test_hop3_acc"] for r in runs])

    # Gate G7: Prefix-Invariant Fixed-Wall-Clock 8xA100 DDP val_bpb Autonomous RSI Loop
    g7_pass = bool(
        all(r["suite_c"]["prefix_invariance_audit"]["strict_causal_verified"] for r in runs)
        and all(r["suite_c"]["winning_candidate_id"] == "gen2_rsi_round9_rect_flash_mor_kv_share" for r in runs)
        and sc_step_mult["median"] >= 1.15
        and sc_val_imp["median"] >= 0.020
        and sc_test_imp["median"] >= 0.015
        and all(r["suite_c"]["generations"][3]["decision"] == "REVERT_VAL_BPB" for r in runs)
        and all(r["suite_c"]["generations"][4]["decision"] == "REVERT_SLOWER" for r in runs)
    )

    # Part 2: 5-Arm Matched-Step Parameter-Sharing & Multi-Hop Latent-Depth Reasoning Scaling Law
    law_agg: Dict[str, Any] = {}
    for arm_k in sc0["matched_step_latent_depth_law"]:
        law_agg[arm_k] = {
            "wall_clock_s": _med_iqr([r["suite_c"]["matched_step_latent_depth_law"][arm_k]["wall_clock_s"] for r in runs]),
            "val_bpb": _med_iqr([r["suite_c"]["matched_step_latent_depth_law"][arm_k]["val_bpb"] for r in runs]),
            "clean_test_bpb": _med_iqr([r["suite_c"]["matched_step_latent_depth_law"][arm_k]["clean_test_bpb"] for r in runs]),
            "test_hop1_acc": _med_iqr([r["suite_c"]["matched_step_latent_depth_law"][arm_k]["test_hop1_acc"] for r in runs]),
            "test_hop3_acc": _med_iqr([r["suite_c"]["matched_step_latent_depth_law"][arm_k]["test_hop3_acc"] for r in runs]),
            "mean_depth_hop1": _med_iqr([r["suite_c"]["matched_step_latent_depth_law"][arm_k]["mean_depth_hop1"] for r in runs]),
            "mean_depth_hop3": _med_iqr([r["suite_c"]["matched_step_latent_depth_law"][arm_k]["mean_depth_hop3"] for r in runs]),
            "adaptive_depth_gap_hop3_minus_hop1": _med_iqr([r["suite_c"]["matched_step_latent_depth_law"][arm_k]["adaptive_depth_gap_hop3_minus_hop1"] for r in runs]),
        }

    m0 = law_agg["M0_shallow_vanilla_iso_param_nr1"]
    m1 = law_agg["M1_fixed_recursive_middle_cycle_nr3"]
    m2 = law_agg["M2_mor_token_choice_rec_cache_nr3"]
    m3 = law_agg["M3_mor_expert_aux_bce_rec_cache_nr3"]
    m4 = law_agg["M4_mor_causal_kv_share_flash_mor_rsi_nr3"]

    # Gate G8: Matched-Step Parameter-Sharing & Multi-Hop Latent-Depth Reasoning Law
    g8_pass = bool(
        m4["clean_test_bpb"]["median"] < m0["clean_test_bpb"]["median"]
        and m4["clean_test_bpb"]["median"] < m2["clean_test_bpb"]["median"]
        and m4["clean_test_bpb"]["median"] < m3["clean_test_bpb"]["median"]
        and m4["test_hop3_acc"]["median"] >= m0["test_hop3_acc"]["median"] + 0.15
        and m4["test_hop3_acc"]["median"] >= m3["test_hop3_acc"]["median"] + 0.05
        and m4["adaptive_depth_gap_hop3_minus_hop1"]["median"] >= 0.50
    )

    gates = {
        "G1_evaluator_and_red_team": g1_pass,
        "G2_numeric_parity_and_richardson": g2_pass,
        "G3_cubin_zero_spill_audit": g3_pass,
        "G4_suite_a_track1_rect_flash_mor_speedup": g4_pass,
        "G5_suite_a_track2_end_to_end_mor_stack": g5_pass,
        "G6_suite_b_pinned_fsdp2_mor_lpt_alltoallv": g6_pass,
        "G7_suite_c_val_bpb_autoresearch_loop": g7_pass,
        "G8_suite_c_matched_step_latent_depth_law": g8_pass,
        "G9_multi_step_distributed_mor_lpt_parity": g9_pass,
    }

    return {
        "experiment": "E135_mor_recursive_depth_round9_rsi",
        "mode": mode,
        "num_repetitions": len(runs),
        "eval_harness_sha256_on_disk": disk_sha,
        "gates_passed_count": sum(1 for v in gates.values() if v),
        "gates_total_count": len(gates),
        "all_gates_passed": all(gates.values()),
        "gates": gates,
        "suite_0_summary": {
            "red_team_rejected": "4/4" if rt.get("all_4_rejected") else "FAIL",
            "num_rsi_kernels": int(cubin["num_round9_kernels"]),
            "total_specializations": int(cubin["total_compiled_specializations"]),
            "max_n_regs": max_rsi_regs,
            "total_rsi_spills": int(cubin["total_round9_spills"]),
            "naive_monolithic_64x128_dkv_spills": naive_k128_spills,
            "max_fp32_rel_l2": par["max_fp32_rel_l2"],
            "max_bf16_rel_l2": par["max_bf16_rel_l2"],
            "zero_consumer_key_exact_zero_grad": par["zero_consumer_key_exact_zero_grad"],
            "richardson_passed": f"{rich['num_passed']}/{rich['num_total']}",
            "upstream_forensics": forensics,
        },
        "suite_a_summary": {
            "track1_official_kvshare_geomean_ms": t1_off_ms,
            "track1_sdpa_4d_mask_geomean_ms": t1_sdpa_ms,
            "track1_monolithic_spill_geomean_ms": t1_spill_ms,
            "track1_winner_geomean_ms": t1_win_ms,
            "track1_speedup_vs_official_kvshare": t1_sp_off,
            "track1_speedup_vs_sdpa_4d_mask": t1_sp_sdpa,
            "track1_speedup_vs_monolithic_spill": t1_sp_spill,
            "track1_s4_hbm_saved_vs_official_mib": t1_s4_hbm_saved,
            "track1_shapes": t1_shapes_agg,
            "track2_pad4d_geomean_ms": t2_pad4d_ms,
            "track2_official_kvshare_clone_geomean_ms": t2_kvclone_ms,
            "track2_winner_geomean_ms": t2_win_ms,
            "track2_speedup_vs_pad4d": t2_sp_pad4d,
            "track2_speedup_vs_official_kvshare_clone": t2_sp_kvclone,
            "track2_hbm_saved_vs_pad4d_mib": t2_hbm_saved_pad4d,
            "track2_hbm_saved_vs_official_kvshare_clone_mib": t2_hbm_saved_kvclone,
        },
        "suite_b_summary": {
            "docs_per_rank": sb0["docs_per_rank"],
            "global_docs": sb0["global_docs"],
            "doc_seq_len": sb0["doc_seq_len"],
            "gate_g9_rep1": g9_rec,
            "d0_naive_fsdp2_pad4d_ms": d0_ms,
            "d1_naive_fsdp2_kvshare_fullseq_ms": d1_ms,
            "d2_pinned_fsdp2_unbalanced_rsi_ms": d2_ms,
            "d3_pinned_fsdp2_mor_lpt_alltoallv_rsi_ms": d3_ms,
            "d0_peak_mib": d0_mib,
            "d1_peak_mib": d1_mib,
            "d2_peak_mib": d2_mib,
            "d3_peak_mib": d3_mib,
            "speedup_d3_vs_d0_pad4d": sb_sp_d0,
            "speedup_d3_vs_d1_official_kvshare": sb_sp_d1,
            "speedup_d3_vs_d2_unbalanced_rsi": sb_sp_d2,
            "hbm_saved_vs_d0_pad4d_mib": sb_hbm_saved_d0,
            "hbm_saved_vs_d1_official_kvshare_mib": sb_hbm_saved_d1,
        },
        "suite_c_summary": {
            "prefix_invariance_audit": pref_audit,
            "budget_seconds": sc0["budget_seconds"],
            "winning_candidate_id": sc0["winning_candidate_id"],
            "step_multiplier_vs_gen0": sc_step_mult,
            "val_bpb_improvement_vs_gen0": sc_val_imp,
            "gen0_steps": gen_agg[0]["steps_completed"],
            "gen1_steps": gen_agg[1]["steps_completed"],
            "winner_steps": gen_agg[2]["steps_completed"],
            "generations": gen_agg,
            "single_touch_gen0_test_bpb": sc_test_gen0,
            "single_touch_winner_test_bpb": sc_test_win,
            "single_touch_test_bpb_improvement": sc_test_imp,
            "single_touch_gen0_hop3_acc": sc_test_hop3_gen0,
            "single_touch_winner_hop3_acc": sc_test_hop3_win,
            "matched_step_latent_depth_law": law_agg,
        },
    }


def render_summary_markdown(agg: Dict[str, Any]) -> str:
    s0 = agg["suite_0_summary"]
    sa = agg["suite_a_summary"]
    sb = agg["suite_b_summary"]
    sc = agg["suite_c_summary"]
    g9 = sb["gate_g9_rep1"]
    lines = [
        "# E135 (`v126`, Round-9 RSI): `Mixture-of-Recursions (MoR)` Dynamic Latent-Depth Training Kernels — Hardware Verification Summary",
        "",
        f"- **Mode**: `{agg['mode']}` (`{agg['num_repetitions']}` independent OS-process 8-GPU repetitions)",
        f"- **Validation Gates**: **`{agg['gates_passed_count']}/{agg['gates_total_count']}` PASSED** (`all_gates_passed = {agg['all_gates_passed']}`)",
        f"- **Immutable Evaluator SHA-256**: `{agg['eval_harness_sha256_on_disk']}`",
        "",
        "## Headline Hardware & Scaling Results (3-Rep Median / IQR)",
        "",
        f"1. **Suite 0 (Upstream `raymin0223/mixture_of_recursions` Forensics, Cubin Zero-Spill Audit & Exactness)**: "
        f"`{s0['red_team_rejected']}` red-team exploits rejected; "
        f"`{s0['num_rsi_kernels']}/{s0['num_rsi_kernels']}` Round-9 RSI Triton kernels (`{s0['total_specializations']}` specializations, max `{s0['max_n_regs']}` regs) compiled with **`0` cubin spills** "
        f"(vs `{s0['naive_monolithic_64x128_dkv_spills']}` spills in monolithic `[64x128]` `dKV`); "
        f"`FP32 max_rel_l2 = {s0['max_fp32_rel_l2']:.2e}`, `BF16 max_rel_l2 = {s0['max_bf16_rel_l2']:.4f}`; `{s0['richardson_passed']}` `FP64` `O(h^2)` Richardson checks passed.",
        f"2. **Suite A — Track 1 (Position-Indexed Rectangular Causal-Gather `Flash-MoR` `Q_act[K_r] x KV_shared[T]` Fwd+Bwd across `s1..s4`)**: "
        f"`{sa['track1_official_kvshare_geomean_ms']['median']:.3f} -> {sa['track1_winner_geomean_ms']['median']:.3f} ms` (**`{sa['track1_speedup_vs_official_kvshare']['median']:.2f}x`** geomean speedup over Official `kv_sharing` full-sequence `FA2`+gather and **`{sa['track1_speedup_vs_sdpa_4d_mask']['median']:.2f}x`** over 4D-masked SDPA, saving **`{sa['track1_s4_hbm_saved_vs_official_mib']['median']:.1f} MiB`** peak HBM at `T=8192`).",
        f"3. **Suite A — Track 2 (End-to-End `Middle-Cycle` `N_r = 3` Recursive `MoR` Stack Fwd+Bwd across `m1..m3`)**: "
        f"`{sa['track2_pad4d_geomean_ms']['median']:.3f} -> {sa['track2_winner_geomean_ms']['median']:.3f} ms` (**`{sa['track2_speedup_vs_pad4d']['median']:.2f}x`** over Official `pad_sequence`+4D-mask loop, saving **`{sa['track2_hbm_saved_vs_pad4d_mib']['median']:.1f} MiB`** peak HBM, and **`{sa['track2_speedup_vs_official_kvshare_clone']['median']:.2f}x`** over Official `kv_sharing` full-sequence clone).",
        f"4. **Suite B (`8-GPU` NVSwitch Recursion-Pinned Tied-Weight `FSDP2` + Whole-Document `MoR-LPT AllToAllv` at `B_global = {sb['global_docs']}, T = {sb['doc_seq_len']}`)**: "
        f"Cut tied-weight `FSDP2` `AllGather` collectives **`{g9['fsdp2_default_reshard_collectives']['all_gather_calls']} -> {g9['fsdp2_pinned_recursion_collectives']['all_gather_calls']}`** (**`{g9['fsdp2_allgather_call_reduction_ratio']:.1f}x`** reduction), collapsed rank active-token FLOP skew **`{g9['unbalanced_rank_flop_skew_ratio']:.2f}x -> {g9['lpt_balanced_rank_flop_skew_ratio']:.2f}x`**, verified 3-step `Gate G9` parity (`max_param_rel_l2 = {g9['max_param_rel_l2']:.2e}` vs `{g9['local_balancing_router_grad_rel_l2_bug']:.4f}` rank-local balancing router gradient bias), and achieved **`{sb['speedup_d3_vs_d0_pad4d']['median']:.2f}x`** over Naive `FSDP2`+Pad4D, **`{sb['speedup_d3_vs_d1_official_kvshare']['median']:.2f}x`** over Naive `FSDP2`+Official `KV-Share`, and **`{sb['speedup_d3_vs_d2_unbalanced_rsi']['median']:.2f}x`** pure `MoR-LPT` NVSwitch gain.",
        f"5. **Suite C (`{sc['budget_seconds']}s` Strict-Causal `8xA100 DDP` `val_bpb` Autonomous RSI Loop)**: "
        f"Verified `strict_causal_prefix_max_abs_diff = {sc['prefix_invariance_audit']['strict_causal_prefix_max_abs_diff']:.1e}` (vs `{sc['prefix_invariance_audit']['leaky_batch_topk_prefix_max_abs_diff']:.4f}` batch top-k causal leakage); "
        f"`{sc['winning_candidate_id']}` ran **`{sc['step_multiplier_vs_gen0']['median']:.2f}x`** more updates (`{sc['gen0_steps']['median']:.0f} -> {sc['winner_steps']['median']:.0f}`), improving `val_bpb` by **`-{sc['val_bpb_improvement_vs_gen0']['median']:.4f}`** and untouched single-touch `test_bpb` by **`-{sc['single_touch_test_bpb_improvement']['median']:.4f}`** (`3-Hop Acc = {sc['single_touch_winner_hop3_acc']['median']*100:.2f}%`).",
        "",
        "## Matched-Step Parameter-Sharing & Multi-Hop Latent-Depth Reasoning Scaling Law",
        "",
        "| Arm | Wall (s) | `val_bpb` | Clean `test_bpb` | `1-Hop` Acc | `3-Hop` Acc | `E[r | 1-Hop]` | `E[r | 3-Hop]` | Depth Gap `Δr` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ak, av in sc["matched_step_latent_depth_law"].items():
        lines.append(
            f"| `{ak}` | `{av['wall_clock_s']['median']:.2f}s` | `{av['val_bpb']['median']:.4f}` | "
            f"`{av['clean_test_bpb']['median']:.4f}` | `{av['test_hop1_acc']['median']*100:.2f}%` | "
            f"`{av['test_hop3_acc']['median']*100:.2f}%` | `{av['mean_depth_hop1']['median']:.2f}` | "
            f"`{av['mean_depth_hop3']['median']:.2f}` | `+{av['adaptive_depth_gap_hop3_minus_hop1']['median']:.2f}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Validate & aggregate E135 Mixture-of-Recursions Round-9 RSI outputs")
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--inputs", nargs="+", default=None)
    parser.add_argument("--harness-path", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--summary-md", type=str, default="")
    args = parser.parse_args()

    root_dir = args.root.resolve()
    raw_dir = root_dir / "results" / "raw" / "mor_recursive_depth_rsi"
    if not args.harness_path:
        args.harness_path = str(root_dir / "code" / "rsi_kernel_harness.py")

    if args.inputs is None:
        if args.mode == "full":
            args.inputs = [str(raw_dir / f"rep{r}" / "mor_recursive_depth_rsi_full.json") for r in (1, 2, 3)]
            if not args.json_out:
                args.json_out = str(raw_dir / "mor_recursive_depth_rsi_full.json")
            if not args.summary_md:
                args.summary_md = str(raw_dir / "mor_recursive_depth_rsi_summary.md")
        else:
            args.inputs = [str(raw_dir / "smoke" / "mor_recursive_depth_rsi_smoke.json")]

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
        sys.exit(2)


if __name__ == "__main__":
    main()
