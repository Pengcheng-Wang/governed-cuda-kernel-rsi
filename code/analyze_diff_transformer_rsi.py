#!/usr/bin/env python3
"""
E136 (v127, Round-10 RSI): 3-Repetition Aggregator & 9-Gate Falsification Verifier
==================================================================================

Reads `diff_transformer_rsi_smoke.json` and `diff_transformer_rsi_rep{1,2,3}.json`,
aggregates medians and means across the 3 independent OS-process full repetitions,
verifies that all 9/9 E136 falsification gates pass on every run, and writes
`diff_transformer_rsi_summary.json` plus a standardized 7-entry `SHA256SUMS`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def median_vals(vals: List[float]) -> float:
    return float(statistics.median(vals))


def mean_vals(vals: List[float]) -> float:
    return float(statistics.mean(vals))


def main():
    parser = argparse.ArgumentParser(description="Aggregate E136 3-rep Differential Transformer RSI results")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "results" / "raw" / "diff_transformer_rsi"),
    )
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)

    smoke_path = raw_dir / "diff_transformer_rsi_smoke.json"
    rep_paths = [raw_dir / f"diff_transformer_rsi_rep{i}.json" for i in (1, 2, 3)]

    with open(smoke_path, "r", encoding="utf-8") as f:
        smoke = json.load(f)
    reps = []
    for p in rep_paths:
        with open(p, "r", encoding="utf-8") as f:
            reps.append(json.load(f))

    # Aggregate Suite A Track 1 & Track 2 medians
    t1_shapes = list(reps[0]["suite_a"]["track1_candidates"]["a0_official_multihead_flashdiffv2"].keys())
    t1_cands = list(reps[0]["suite_a"]["track1_candidates"].keys())
    track1_median_ms: Dict[str, Dict[str, float]] = {c: {} for c in t1_cands}
    for c in t1_cands:
        for s in t1_shapes:
            track1_median_ms[c][s] = median_vals(
                [r["suite_a"]["track1_candidates"][c][s]["latency_ms"] for r in reps]
            )

    track1_speedups_vs_a0 = {
        s: track1_median_ms["a0_official_multihead_flashdiffv2"][s]
        / track1_median_ms["a5_rsi_paired_flash_diffv2_zero_spill"][s]
        for s in t1_shapes
    }
    track1_speedups_vs_a2 = {
        s: track1_median_ms["a2_compiled_multihead_flashdiffv2"][s]
        / track1_median_ms["a5_rsi_paired_flash_diffv2_zero_spill"][s]
        for s in t1_shapes
    }

    t2_shapes = list(reps[0]["suite_a"]["track2_candidates"]["b0_official_multihead_flashdiff_2_4x_fa2"].keys())
    t2_cands = list(reps[0]["suite_a"]["track2_candidates"].keys())
    track2_median_ms: Dict[str, Dict[str, float]] = {c: {} for c in t2_cands}
    for c in t2_cands:
        for m in t2_shapes:
            track2_median_ms[c][m] = median_vals(
                [r["suite_a"]["track2_candidates"][c][m]["latency_ms"] for r in reps]
            )

    track2_speedups_vs_b0 = {
        m: track2_median_ms["b0_official_multihead_flashdiff_2_4x_fa2"][m]
        / track2_median_ms["b3_rsi_paired_flash_diffv1_fused_subln"][m]
        for m in t2_shapes
    }
    track2_speedups_vs_b1 = {
        m: track2_median_ms["b1_official_multihead_flashdiff_1_2x_fa2_unfused_subln"][m]
        / track2_median_ms["b3_rsi_paired_flash_diffv1_fused_subln"][m]
        for m in t2_shapes
    }

    # Aggregate Suite B medians
    sb_allgather_ms = median_vals([r["suite_b"]["latency_ms"]["allgather_diffv2_ms"] for r in reps])
    sb_2x_ring_ms = median_vals([r["suite_b"]["latency_ms"]["separate_2x_ring_diffv2_ms"] for r in reps])
    sb_rsi_ring_ms = median_vals([r["suite_b"]["latency_ms"]["dual_lse_ring_diffv2_rsi_ms"] for r in reps])

    # Aggregate Suite C medians
    c_exps = list(reps[0]["suite_c"]["loop_results"].keys())
    suite_c_loop_medians: Dict[str, Dict[str, Any]] = {}
    for exp_id in c_exps:
        suite_c_loop_medians[exp_id] = {
            "variant": reps[0]["suite_c"]["loop_results"][exp_id]["variant"],
            "decision": reps[0]["suite_c"]["loop_results"][exp_id]["decision"],
            "median_steps_completed": int(median_vals([r["suite_c"]["loop_results"][exp_id]["steps_completed"] for r in reps])),
            "median_val_bpb": median_vals([r["suite_c"]["loop_results"][exp_id]["val_bpb"] for r in reps]),
            "median_val_needle_acc": median_vals([r["suite_c"]["loop_results"][exp_id]["val_needle_acc"] for r in reps]),
        }

    arms = list(reps[0]["suite_c"]["five_arm_scaling_law"].keys())
    five_arm_medians: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        five_arm_medians[arm] = {
            "variant": reps[0]["suite_c"]["five_arm_scaling_law"][arm]["variant"],
            "matched_steps": reps[0]["suite_c"]["five_arm_scaling_law"][arm]["matched_steps"],
            "median_train_wall_ms": median_vals([r["suite_c"]["five_arm_scaling_law"][arm]["train_wall_ms"] for r in reps]),
            "median_val_bpb": median_vals([r["suite_c"]["five_arm_scaling_law"][arm]["val_bpb"] for r in reps]),
            "median_clean_test_bpb": median_vals([r["suite_c"]["five_arm_scaling_law"][arm]["clean_test_bpb"] for r in reps]),
            "median_clean_needle_acc": median_vals([r["suite_c"]["five_arm_scaling_law"][arm]["clean_needle_acc"] for r in reps]),
            "median_distractor_test_bpb": median_vals([r["suite_c"]["five_arm_scaling_law"][arm]["distractor_test_bpb"] for r in reps]),
            "median_distractor_needle_acc": median_vals([r["suite_c"]["five_arm_scaling_law"][arm]["distractor_needle_acc"] for r in reps]),
            "median_distractor_attn_mass": median_vals([r["suite_c"]["five_arm_scaling_law"][arm]["distractor_attn_mass"] for r in reps]),
        }

    gate_names = list(reps[0]["gates"].keys())
    gates_3rep_summary: Dict[str, Any] = {}
    all_passed = True
    for g in gate_names:
        smoke_ok = bool(smoke["gates"][g]["pass"])
        reps_ok = [bool(r["gates"][g]["pass"]) for r in reps]
        g_ok = smoke_ok and all(reps_ok)
        all_passed = all_passed and g_ok
        gates_3rep_summary[g] = {
            "smoke_pass": smoke_ok,
            "rep_passes": reps_ok,
            "all_pass": g_ok,
        }

    summary = {
        "experiment": "E136_differential_transformer_v1_v2_rsi",
        "version": "v127",
        "paper": {
            "title": "Differential Transformer (V1 & V2)",
            "authors": "Tianchi Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei",
            "arxiv": "2410.05258",
            "venue": "ICLR 2025 + Jan 2026 DIFF V2",
            "official_repo": "https://github.com/microsoft/unilm/tree/master/Diff-Transformer",
        },
        "frozen_harness_sha256": reps[0]["suite_0"]["frozen_harness_sha256"],
        "smoke_wall_clock_s": smoke["total_wall_clock_seconds"],
        "rep_wall_clock_s": [r["total_wall_clock_seconds"] for r in reps],
        "suite_0_summary": {
            "fp32_max_rel_l2": max(r["suite_0"]["fp32_max_rel_l2"] for r in reps),
            "bf16_max_rel_l2": max(r["suite_0"]["bf16_max_rel_l2"] for r in reps),
            "v1_upstream_gqa_repeat_kv_bug_rel_l2": reps[0]["suite_0"]["v1_upstream_gqa_repeat_kv_bug_rel_l2"],
            "richardson_all_in_3p7_4p3": all(r["suite_0"]["richardson_all_in_3p7_4p3"] for r in reps),
            "redteam_all_rejected": all(r["suite_0"]["redteam_all_rejected"] for r in reps),
            "rsi_kernels_compiled_count": reps[0]["suite_0"]["rsi_kernels_compiled_count"],
            "rsi_kernels_max_spills": max(r["suite_0"]["rsi_kernels_max_spills"] for r in reps),
            "monolithic_d128_spill_kernel": reps[0]["suite_0"]["cubin_spill_report"]["_flash_diffv2_bwd_monolithic_spill_kernel"],
        },
        "suite_a_summary": {
            "track1_median_latency_ms": track1_median_ms,
            "track1_speedups_a5_vs_a0": track1_speedups_vs_a0,
            "track1_speedups_a5_vs_a2": track1_speedups_vs_a2,
            "track1_peak_speedup_vs_a0": max(track1_speedups_vs_a0.values()),
            "track1_min_speedup_vs_a0": min(track1_speedups_vs_a0.values()),
            "track2_median_latency_ms": track2_median_ms,
            "track2_speedups_b3_vs_b0": track2_speedups_vs_b0,
            "track2_speedups_b3_vs_b1": track2_speedups_vs_b1,
            "track2_peak_speedup_vs_b0": max(track2_speedups_vs_b0.values()),
            "track2_min_speedup_vs_b0": min(track2_speedups_vs_b0.values()),
        },
        "suite_b_summary": {
            "world_size": 8,
            "t_global_bench": reps[0]["suite_b"]["t_global_bench"],
            "analytical_2shard_local_sub_rel_err": reps[0]["suite_b"]["analytical_2shard_local_sub_rel_err"],
            "empirical_8shard_local_sub_rel_l2": reps[0]["suite_b"]["empirical_8shard_local_sub_rel_l2"],
            "gate_g9_max_loss_rel_diff": max(r["suite_b"]["gate_g9_max_loss_rel_diff"] for r in reps),
            "gate_g9_max_param_rel_l2": max(r["suite_b"]["gate_g9_max_param_rel_l2"] for r in reps),
            "gate_g9_max_grad_rel_l2": max(r["suite_b"]["gate_g9_max_grad_rel_l2"] for r in reps),
            "median_allgather_diffv2_ms": sb_allgather_ms,
            "median_2x_separate_ring_diffv2_ms": sb_2x_ring_ms,
            "median_dual_lse_ring_diffv2_rsi_ms": sb_rsi_ring_ms,
            "speedup_vs_allgather": sb_allgather_ms / sb_rsi_ring_ms,
            "speedup_vs_2x_separate_ring": sb_2x_ring_ms / sb_rsi_ring_ms,
        },
        "suite_c_summary": {
            "loop_medians": suite_c_loop_medians,
            "final_incumbent_clean_test_bpb": median_vals([r["suite_c"]["final_incumbent_clean_test_bpb"] for r in reps]),
            "final_incumbent_distractor_test_bpb": median_vals([r["suite_c"]["final_incumbent_distractor_test_bpb"] for r in reps]),
            "five_arm_scaling_law_medians": five_arm_medians,
        },
        "gates_3rep_summary": gates_3rep_summary,
        "gates_passed_count": sum(1 for v in gates_3rep_summary.values() if v["all_pass"]),
        "gates_total_count": len(gates_3rep_summary),
        "all_nine_gates_passed": all_passed,
    }

    summary_path = raw_dir / "diff_transformer_rsi_summary.json"
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2)

    # Write standardized 7-entry SHA256SUMS manifest
    manifest_files = [
        "diff_transformer_rsi_rep1.json",
        "diff_transformer_rsi_rep2.json",
        "diff_transformer_rsi_rep3.json",
        "diff_transformer_rsi_smoke.json",
        "diff_transformer_rsi_summary.json",
        "suite0_redteam_ledger.jsonl",
        "suite_a_ledger_rep1.jsonl",
    ]
    sha_lines = []
    for fname in manifest_files:
        fpath = raw_dir / fname
        sha_lines.append(f"{sha256_file(fpath)}  {fname}\n")
    with open(raw_dir / "SHA256SUMS", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(sha_lines)

    print(
        f"[E136 Analyzer] Summary written to {summary_path} | "
        f"Gates passed: {summary['gates_passed_count']}/{summary['gates_total_count']} (all_pass={all_passed})"
    )


if __name__ == "__main__":
    main()
