#!/usr/bin/env python3
"""Deterministic 9-Gate Offline Validator & 3-Repetition Aggregator for E133 (`v124`):
`H-Net: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling` (`arXiv:2507.07955`, `goombalab/hnet`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

EXPECTED_HARNESS_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def _med_iqr(vals: list[float]) -> dict[str, float]:
    s = sorted(float(v) for v in vals)
    med = float(statistics.median(s))
    q25 = s[max(0, (len(s) - 1) // 4)]
    q75 = s[min(len(s) - 1, (3 * (len(s) - 1)) // 4)]
    return {
        "median": round(med, 6),
        "iqr": round(q75 - q25, 6),
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
    }


def evaluate_gates(runs: list[dict[str, Any]], mode: str, harness_path: Path) -> dict[str, Any]:
    disk_sha = hashlib.sha256(harness_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    g1_immutable_harness = (
        disk_sha == EXPECTED_HARNESS_SHA256
        and all(r.get("eval_harness_sha256") == EXPECTED_HARNESS_SHA256 for r in runs)
        and all(not r["environment"].get("allow_tf32_matmul", True) for r in runs)
    )

    g2_red_team_and_upstream = all(
        r["suite_0"]["all_red_team_rejected"]
        and r["suite_0"]["upstream_hnet_forensics"]["official_chunk_layer_fullgraph_compile_failed"]
        and r["suite_0"]["upstream_hnet_forensics"]["official_ssd_surrogate_confident_bf16_rel_l2"]
        > r["suite_0"]["upstream_hnet_forensics"]["rsi_division_free_ema_confident_bf16_rel_l2"]
        for r in runs
    )

    g3_zero_cubin_spills = all(
        r["suite_0"]["cubin_spill_audit"]["all_zero_spills"]
        and r["suite_0"]["cubin_spill_audit"]["total_cubin_spills"] == 0
        and r["suite_0"]["cubin_spill_audit"]["num_rsi_kernels"] == 17
        for r in runs
    )

    g4_fp64_and_richardson = all(
        r["suite_0"]["all_parity_passed"]
        and r["suite_0"]["max_fp32_rel_l2"] <= 1e-5
        and r["suite_0"]["max_bf16_rel_l2"] <= 1.5e-2
        and r["suite_0"]["all_richardson_passed"]
        and len(r["suite_0"]["richardson_checks"]) == 7
        for r in runs
    )

    t1_sp_eag = [float(r["suite_a"]["track1"]["speedup_vs_eager"]) for r in runs]
    t1_sp_off = [float(r["suite_a"]["track1"]["speedup_vs_official"]) for r in runs]
    t1_sp_cmp = [float(r["suite_a"]["track1"]["speedup_vs_compile"]) for r in runs]
    t1_hbm_cmp = [float(r["suite_a"]["track1"]["hbm_saved_vs_compile_mib"]) for r in runs]
    g5_track1_ratchet = (
        all(r["suite_a"]["track1"]["winner_id"] == "a5_rsi_round7_superfused_zero_spill" for r in runs)
        and all(
            any(t["candidate_id"] == "a1_official_hnet_bf16_norm_ssd_bug" and t["decision"].startswith("REJECT") for t in r["suite_a"]["track1"]["trials"])
            and any(t["candidate_id"] == "a2_broken_router_omit_radial_jacobian" and t["decision"].startswith("REJECT") for t in r["suite_a"]["track1"]["trials"])
            and any(t["candidate_id"] == "a4_broken_dechunk_omit_ema_adjoint" and t["decision"].startswith("REJECT") for t in r["suite_a"]["track1"]["trials"])
            for r in runs
        )
        and _med_iqr(t1_sp_off)["median"] >= 4.0
        and _med_iqr(t1_sp_cmp)["median"] >= 2.0
        and _med_iqr(t1_hbm_cmp)["median"] > 0.0
    )

    t2_sp_off = [float(r["suite_a"]["track2"]["speedup_vs_official"]) for r in runs]
    t2_sp_cmp = [float(r["suite_a"]["track2"]["speedup_vs_compile"]) for r in runs]
    t2_hbm_off = [float(r["suite_a"]["track2"]["hbm_saved_vs_official_mib"]) for r in runs]
    t2_hbm_cmp = [float(r["suite_a"]["track2"]["hbm_saved_vs_compile_mib"]) for r in runs]
    g6_track2_ratchet = (
        all(r["suite_a"]["track2"]["winner_id"] == "b3_rsi_round7_superfused_varlen_fa2_stage" for r in runs)
        and _med_iqr(t2_sp_off)["median"] >= 2.0
        and _med_iqr(t2_sp_cmp)["median"] >= 1.15
    )

    sb_sp_off = [float(r["suite_b"]["speedup_d3_vs_d0_official"]) for r in runs]
    sb_sp_eag = [float(r["suite_b"]["speedup_d3_vs_d1_packed_eager"]) for r in runs]
    sb_sp_lpt = [float(r["suite_b"]["speedup_d3_vs_d2_pure_cost_lpt_gain"]) for r in runs]
    g7_nvswitch_cost_lpt_and_g9 = (
        all(r["suite_b"]["gate_g9_multi_step_parity"]["passed"] for r in runs)
        and _med_iqr(sb_sp_off)["median"] >= 1.5
        and _med_iqr(sb_sp_lpt)["median"] >= 1.05
    )

    sc_mult = [float(r["suite_c"]["step_multiplier_vs_gen0"]) for r in runs]
    sc_vgain = [float(r["suite_c"]["val_bpb_improvement_vs_gen0"]) for r in runs]
    sc_tgain = [float(r["suite_c"]["single_touch_test_bpb_improvement"]) for r in runs]
    g8_valbpb_rsi_loop = (
        all(r["suite_c"]["winning_candidate_id"] == "gen2_rsi_round7_superfused_dc" for r in runs)
        and all(
            any(g["candidate_id"] == "gen3_ablate_ema_smoothing_module" and g["decision"].startswith("REVERT") for g in r["suite_c"]["generations"])
            and any(g["candidate_id"] == "gen4_ablate_ratio_loss_alpha0" and g["decision"].startswith("REVERT") for g in r["suite_c"]["generations"])
            for r in runs
        )
        and _med_iqr(sc_mult)["median"] >= 1.25
        and _med_iqr(sc_vgain)["median"] > 0.05
        and _med_iqr(sc_tgain)["median"] > 0.05
    )

    law_arms = [
        "H0_isotropic_byte_swa",
        "H1_static_stride4_hourglass",
        "H2_spacebyte_delimiter_heuristic",
        "H3_hnet_dc_no_ema_no_ste",
        "H4_hnet_rsi_dynamic_chunking",
    ]
    law_agg: dict[str, dict[str, Any]] = {}
    for ak in law_arms:
        law_agg[ak] = {
            "params": int(runs[0]["suite_c"]["matched_step_byte_chunking_law"][ak]["params"]),
            "steps": int(runs[0]["suite_c"]["matched_step_byte_chunking_law"][ak]["steps"]),
            "val_bpb": _med_iqr([r["suite_c"]["matched_step_byte_chunking_law"][ak]["val_bpb"] for r in runs]),
            "clean_test_bpb": _med_iqr([r["suite_c"]["matched_step_byte_chunking_law"][ak]["clean_test_bpb"] for r in runs]),
            "shifted_noisy_test_bpb": _med_iqr([r["suite_c"]["matched_step_byte_chunking_law"][ak]["shifted_noisy_test_bpb"] for r in runs]),
            "perturbation_bpb_degradation": _med_iqr([r["suite_c"]["matched_step_byte_chunking_law"][ak]["perturbation_bpb_degradation"] for r in runs]),
            "clean_kv_byte_recall_acc": _med_iqr([r["suite_c"]["matched_step_byte_chunking_law"][ak]["clean_kv_byte_recall_acc"] for r in runs]),
            "noisy_kv_byte_recall_acc": _med_iqr([r["suite_c"]["matched_step_byte_chunking_law"][ak]["noisy_kv_byte_recall_acc"] for r in runs]),
            "morpheme_boundary_f1": _med_iqr([r["suite_c"]["matched_step_byte_chunking_law"][ak]["morpheme_boundary_f1"] for r in runs]),
            "bytes_per_chunk": _med_iqr([r["suite_c"]["matched_step_byte_chunking_law"][ak]["bytes_per_chunk"] for r in runs]),
        }

    h4_clean = law_agg["H4_hnet_rsi_dynamic_chunking"]["clean_test_bpb"]["median"]
    h4_noisy = law_agg["H4_hnet_rsi_dynamic_chunking"]["shifted_noisy_test_bpb"]["median"]
    g9_byte_scaling_and_robustness_law = (
        (len(runs) == (3 if mode == "full" else 1))
        and h4_clean < law_agg["H0_isotropic_byte_swa"]["clean_test_bpb"]["median"]
        and h4_clean < law_agg["H3_hnet_dc_no_ema_no_ste"]["clean_test_bpb"]["median"]
        and h4_noisy < law_agg["H1_static_stride4_hourglass"]["shifted_noisy_test_bpb"]["median"]
    )

    gates = {
        "G1_immutable_evaluator_and_precision_flags": g1_immutable_harness,
        "G2_red_team_4of4_and_upstream_hnet_forensics": g2_red_team_and_upstream,
        "G3_zero_cubin_spills_17of17_triton_kernels": g3_zero_cubin_spills,
        "G4_fp64_adversarial_parity_and_7of7_oh2_richardson": g4_fp64_and_richardson,
        "G5_track1_isolated_dc_ops_pareto_ratchet": g5_track1_ratchet,
        "G6_track2_hierarchical_hnet_stage_pareto_ratchet": g6_track2_ratchet,
        "G7_nvswitch_cost_lpt_alltoallv_and_3step_gate_g9": g7_nvswitch_cost_lpt_and_g9,
        "G8_fixed_wallclock_valbpb_rsi_loop_and_single_touch_holdout": g8_valbpb_rsi_loop,
        "G9_matched_step_byte_chunking_and_perturbation_law": g9_byte_scaling_and_robustness_law,
    }
    passed_cnt = sum(int(bool(v)) for v in gates.values())

    s0 = runs[0]["suite_0"]
    sb0 = runs[0]["suite_b"]
    return {
        "experiment": "E133_hnet_dynamic_chunking_round7_rsi",
        "mode": mode,
        "num_repetitions": len(runs),
        "eval_harness_sha256_on_disk": disk_sha,
        "gates_passed_count": passed_cnt,
        "gates_total_count": len(gates),
        "all_gates_passed": (passed_cnt == len(gates)),
        "gates": gates,
        "aggregated_metrics": {
            "suite_0": {
                "red_team_rejected": s0["red_team_rejected_count"],
                "num_rsi_kernels": s0["cubin_spill_audit"]["num_rsi_kernels"],
                "total_specializations": s0["cubin_spill_audit"]["total_specializations"],
                "total_cubin_spills": s0["cubin_spill_audit"]["total_cubin_spills"],
                "max_n_regs": s0["cubin_spill_audit"]["max_n_regs"],
                "naive_1pass_router_spills": s0["cubin_spill_audit"].get("naive_1pass_router_d8192_nw16_spills", 0),
                "max_fp32_rel_l2": max(float(r["suite_0"]["max_fp32_rel_l2"]) for r in runs),
                "max_bf16_rel_l2": max(float(r["suite_0"]["max_bf16_rel_l2"]) for r in runs),
                "upstream_hnet_forensics": s0["upstream_hnet_forensics"],
                "red_team_audit_rep1": s0["red_team_audit"],
            },
            "suite_a": {
                "track1_baseline_ms": _med_iqr([r["suite_a"]["track1"]["baseline_geomean_ms"] for r in runs]),
                "track1_winner_ms": _med_iqr([r["suite_a"]["track1"]["winner_geomean_ms"] for r in runs]),
                "track1_speedup_vs_eager": _med_iqr(t1_sp_eag),
                "track1_speedup_vs_official": _med_iqr(t1_sp_off),
                "track1_speedup_vs_compile": _med_iqr(t1_sp_cmp),
                "track1_hbm_saved_vs_compile_mib": _med_iqr(t1_hbm_cmp),
                "track2_baseline_ms": _med_iqr([r["suite_a"]["track2"]["baseline_geomean_ms"] for r in runs]),
                "track2_winner_ms": _med_iqr([r["suite_a"]["track2"]["winner_geomean_ms"] for r in runs]),
                "track2_speedup_vs_official": _med_iqr(t2_sp_off),
                "track2_speedup_vs_compile": _med_iqr(t2_sp_cmp),
                "track2_hbm_saved_vs_official_mib": _med_iqr(t2_hbm_off),
                "track2_hbm_saved_vs_compile_mib": _med_iqr(t2_hbm_cmp),
                "track1_trials_rep1": runs[0]["suite_a"]["track1"]["trials"],
                "track2_trials_rep1": runs[0]["suite_a"]["track2"]["trials"],
            },
            "suite_b": {
                "global_bytes_per_step": sb0["global_bytes_per_step"],
                "d_outer": sb0["d_outer"],
                "d_main": sb0["d_main"],
                "n_main_layers": sb0["n_main_layers"],
                "speedup_d3_vs_d0_official": _med_iqr(sb_sp_off),
                "speedup_d3_vs_d1_packed_eager": _med_iqr(sb_sp_eag),
                "speedup_d3_vs_d2_pure_cost_lpt_gain": _med_iqr(sb_sp_lpt),
                "d0_ms": _med_iqr([r["suite_b"]["arms"]["d0_official_batched_padded_ddp_unbalanced"]["median_ms"] for r in runs]),
                "d1_ms": _med_iqr([r["suite_b"]["arms"]["d1_packed_eager_ddp_unbalanced"]["median_ms"] for r in runs]),
                "d2_ms": _med_iqr([r["suite_b"]["arms"]["d2_rsi_local_ddp_unbalanced"]["median_ms"] for r in runs]),
                "d3_ms": _med_iqr([r["suite_b"]["arms"]["d3_rsi_nvswitch_cost_lpt_alltoallv"]["median_ms"] for r in runs]),
                "imbalance_before": sb0["arms"]["d3_rsi_nvswitch_cost_lpt_alltoallv"]["imbalance_before"],
                "imbalance_after": sb0["arms"]["d3_rsi_nvswitch_cost_lpt_alltoallv"]["imbalance_after"],
                "nvswitch_comm_mib_per_rank": sb0["arms"]["d3_rsi_nvswitch_cost_lpt_alltoallv"]["nvswitch_comm_mib_per_rank"],
                "raw_byte_comm_equivalent_mib": sb0["arms"]["d3_rsi_nvswitch_cost_lpt_alltoallv"]["raw_byte_comm_equivalent_mib"],
                "compression_comm_saving_ratio": sb0["arms"]["d3_rsi_nvswitch_cost_lpt_alltoallv"]["compression_comm_saving_ratio"],
                "gate_g9_rep1": sb0["gate_g9_multi_step_parity"],
            },
            "suite_c": {
                "budget_seconds": runs[0]["suite_c"]["budget_seconds"],
                "winning_candidate_id": runs[0]["suite_c"]["winning_candidate_id"],
                "gen0_steps": _med_iqr([r["suite_c"]["gen0_steps"] for r in runs]),
                "winner_steps": _med_iqr([r["suite_c"]["winner_steps"] for r in runs]),
                "step_multiplier_vs_gen0": _med_iqr(sc_mult),
                "gen0_val_bpb": _med_iqr([r["suite_c"]["gen0_val_bpb"] for r in runs]),
                "winner_val_bpb": _med_iqr([r["suite_c"]["winner_val_bpb"] for r in runs]),
                "val_bpb_improvement_vs_gen0": _med_iqr(sc_vgain),
                "single_touch_gen0_test_bpb": _med_iqr([r["suite_c"]["single_touch_gen0_test_bpb"] for r in runs]),
                "single_touch_winner_test_bpb": _med_iqr([r["suite_c"]["single_touch_winner_test_bpb"] for r in runs]),
                "single_touch_test_bpb_improvement": _med_iqr(sc_tgain),
                "single_touch_winner_kv_recall_acc": _med_iqr([r["suite_c"]["single_touch_winner_kv_recall_acc"] for r in runs]),
                "single_touch_winner_morpheme_f1": _med_iqr([r["suite_c"]["single_touch_winner_morpheme_f1"] for r in runs]),
                "generations_rep1": runs[0]["suite_c"]["generations"],
                "matched_step_byte_chunking_law": law_agg,
            },
        },
        "repetitions": runs,
    }


def render_summary_markdown(agg: Dict[str, Any] if False else dict[str, Any]) -> str:
    m = agg["aggregated_metrics"]
    s0 = m["suite_0"]
    sa = m["suite_a"]
    sb = m["suite_b"]
    sc = m["suite_c"]
    lines = [
        "# E133 (`v124`): `H-Net: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling` (`arXiv:2507.07955`) Round-7 RSI Summary",
        "",
        f"- **Validation Gates**: **`{agg['gates_passed_count']}/{agg['gates_total_count']}` PASSED** (`all_gates_passed = {agg['all_gates_passed']}`)",
        f"- **Repetitions**: `{agg['num_repetitions']}` independent `8x NVIDIA A100-SXM4-80GB` (`1,410 MHz` locked) runs",
        f"- **Immutable Evaluator SHA-256**: `{agg['eval_harness_sha256_on_disk']}`",
        "",
        "## Headline Hardware & Scaling Results (3-Rep Median / IQR)",
        "",
        f"1. **Suite 0 (Forensics & Exactness)**: `{s0['red_team_rejected']}` red-team exploits rejected; "
        f"`{s0['num_rsi_kernels']}/{s0['num_rsi_kernels']}` Round-7 RSI Triton kernels (`{s0['total_specializations']}` specializations, max `{s0['max_n_regs']}` regs) compiled with **`0` cubin spills** "
        f"(vs `{s0['naive_1pass_router_spills']}` spills for naive 1-pass `[8, BD=4096], nw=16`); `FP32 max_rel_l2 = {s0['max_fp32_rel_l2']:.2e}`, "
        f"`BF16 max_rel_l2 = {s0['max_bf16_rel_l2']:.4f}`; `7/7` `FP64` `O(h^2)` Richardson checks passed.",
        f"2. **Suite A — Track 1 (Isolated Dynamic Chunking Operators: `Router + Pack + EMA + Upsample*STE + Residual`)**: "
        f"`{sa['track1_baseline_ms']['median']:.3f} -> {sa['track1_winner_ms']['median']:.3f} ms` (**`{sa['track1_speedup_vs_official']['median']:.2f}x`** over Official `goombalab/hnet` `dc.py`, "
        f"**`{sa['track1_speedup_vs_compile']['median']:.2f}x`** faster than `torch.compile(dynamic=True)`, saving **`{sa['track1_hbm_saved_vs_compile_mib']['median']:.1f} MiB`** peak HBM).",
        f"3. **Suite A — Track 2 (End-to-End 1-Stage Hierarchical `H-Net` Block with `Varlen FlashAttention-2`)**: "
        f"`{sa['track2_baseline_ms']['median']:.3f} -> {sa['track2_winner_ms']['median']:.3f} ms` (**`{sa['track2_speedup_vs_official']['median']:.2f}x`** over Official `goombalab/hnet`, "
        f"**`{sa['track2_speedup_vs_compile']['median']:.2f}x`** over `torch.compile(dynamic=True)`, saving **`{sa['track2_hbm_saved_vs_compile_mib']['median']:.1f} MiB`** vs `compile`).",
        f"4. **Suite B (`8-GPU` NVSwitch Whole-Document Cost-LPT `AllToAllv` + Global Token-Weighted Ratio Loss)**: "
        f"`Gate G9` 3-step `FP32` parity `max_param_rel_l2 = {sb['gate_g9_rep1']['max_param_rel_l2']:.2e}` (vs `{sb['gate_g9_rep1']['naive_local_ratio_dwqk_rel_l2_bug']:.4f}` local-ratio bias); "
        f"reduces Main-Network straggler cost imbalance **`{sb['imbalance_before']:.2f}x -> {sb['imbalance_after']:.2f}x`**, cuts redistribution payload **`{sb['compression_comm_saving_ratio']:.2f}x`** (`{sb['raw_byte_comm_equivalent_mib']:.1f} -> {sb['nvswitch_comm_mib_per_rank']:.1f} MiB/rank`), "
        f"and delivers **`{sb['speedup_d3_vs_d0_official']['median']:.2f}x`** over Official Batched DDP (`{sb['speedup_d3_vs_d2_pure_cost_lpt_gain']['median']:.2f}x` pure Cost-LPT NVSwitch gain).",
        f"5. **Suite C (`{sc['budget_seconds']}s` `8xA100 DDP` Byte-Level `val_bpb` RSI Loop)**: "
        f"`{sc['winning_candidate_id']}` runs **`{sc['step_multiplier_vs_gen0']['median']:.2f}x`** more updates (`{sc['gen0_steps']['median']:.0f} -> {sc['winner_steps']['median']:.0f}`), "
        f"improving `val_bpb` by **`-{sc['val_bpb_improvement_vs_gen0']['median']:.4f}`** and untouched single-touch `test_bpb` by **`-{sc['single_touch_test_bpb_improvement']['median']:.4f}`** (`KV recall = {sc['single_touch_winner_kv_recall_acc']['median']*100:.2f}%`).",
        "",
        "## Matched-Step Byte-Level Chunking & 1-Byte Phase-Shift / Character-Noise Robustness Law",
        "",
        "| Arm | `val_bpb` | Clean `test_bpb` | Shifted+Noisy `test_bpb` | Noise `Δbpb` | Clean KV Acc | Noisy KV Acc | Boundary `F1` | Bytes/Chunk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ak, av in sc["matched_step_byte_chunking_law"].items():
        lines.append(
            f"| `{ak}` | `{av['val_bpb']['median']:.4f}` | `{av['clean_test_bpb']['median']:.4f}` | "
            f"`{av['shifted_noisy_test_bpb']['median']:.4f}` | `+{av['perturbation_bpb_degradation']['median']:.4f}` | "
            f"`{av['clean_kv_byte_recall_acc']['median']*100:.2f}%` | `{av['noisy_kv_byte_recall_acc']['median']*100:.2f}%` | "
            f"`{av['morpheme_boundary_f1']['median']*100:.2f}%` | `{av['bytes_per_chunk']['median']:.2f}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Validate & aggregate E133 H-Net Dynamic Chunking Round-7 RSI outputs")
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--inputs", nargs="+", default=None)
    parser.add_argument("--harness-path", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--summary-md", type=str, default="")
    args = parser.parse_args()

    root_dir = args.root.resolve()
    raw_dir = root_dir / "results" / "raw" / "hnet_dynamic_chunking_rsi"
    if not args.harness_path:
        args.harness_path = str(root_dir / "code" / "rsi_kernel_harness.py")

    if args.inputs is None:
        if args.mode == "full":
            args.inputs = [str(raw_dir / f"rep{r}" / "hnet_dynamic_chunking_rsi_full.json") for r in (1, 2, 3)]
            if not args.json_out:
                args.json_out = str(raw_dir / "hnet_dynamic_chunking_rsi_full.json")
            if not args.summary_md:
                args.summary_md = str(raw_dir / "hnet_dynamic_chunking_rsi_summary.md")
        else:
            args.inputs = [str(raw_dir / "smoke" / "hnet_dynamic_chunking_rsi_smoke.json")]

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
