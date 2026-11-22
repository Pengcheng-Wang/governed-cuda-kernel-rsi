#!/usr/bin/env python3
"""
E138 (v129, Round-12 RSI): 3-Repetition Aggregator & 9-Gate Falsification Verifier
==================================================================================

Reads `log_linear_attention_rsi_smoke.json` and `log_linear_attention_rsi_rep{1,2,3}.json`,
verifies that all 9/9 E138 falsification gates pass on every run, aggregates 3-repetition
medians across Suite 0, Suite A, Suite B, and Suite C, and writes
`log_linear_attention_rsi_summary.json`, `log_linear_attention_rsi_summary.md`, and a
standardized 7-entry `SHA256SUMS`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _med(vals: List[float]) -> float:
    return float(statistics.median(vals))


def main():
    parser = argparse.ArgumentParser(description="Aggregate E138 3-rep Log-Linear Attention RSI results")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "results" / "raw" / "log_linear_attention_rsi"),
    )
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)

    smoke = json.loads((raw_dir / "log_linear_attention_rsi_smoke.json").read_text(encoding="utf-8"))
    reps = [json.loads((raw_dir / f"log_linear_attention_rsi_rep{i}.json").read_text(encoding="utf-8")) for i in (1, 2, 3)]

    all_runs = [smoke] + reps
    all_passed = all(r["all_nine_gates_passed"] for r in all_runs)
    gates_agg = {k: all(r["gates"][k] for r in all_runs) for k in reps[0]["gates"]}

    s0 = reps[0]["suite_0"]
    t1_up_med = round(_med([r["suite_a"]["track1_speedup_vs_upstream"] for r in reps]), 4)
    t1_comp_med = round(_med([r["suite_a"]["track1_speedup_vs_compile"] for r in reps]), 4)
    t2_up_med = round(_med([r["suite_a"]["track2_speedup_vs_upstream"] for r in reps]), 4)
    t2_comp_med = round(_med([r["suite_a"]["track2_speedup_vs_compile"] for r in reps]), 4)

    sb0 = reps[0]["suite_b"]
    cp_d64_med = round(_med([r["suite_b"]["cp8_benchmarks"]["cp8_T8192_H16_D64"]["speedup"] for r in reps]), 4)
    cp_d128_med = round(_med([r["suite_b"]["cp8_benchmarks"]["cp8_T8192_H16_D128"]["speedup"] for r in reps]), 4)

    base_val_med = round(_med([r["suite_c"]["baseline_val_bpb"] for r in reps]), 4)
    win_val_med = round(_med([r["suite_c"]["winner_val_bpb"] for r in reps]), 4)
    val_imp_med = round(_med([r["suite_c"]["val_bpb_improvement"] for r in reps]), 4)
    base_test_med = round(_med([r["suite_c"]["baseline_test_bpb"] for r in reps]), 4)
    win_test_med = round(_med([r["suite_c"]["winner_test_bpb"] for r in reps]), 4)
    test_imp_med = round(_med([r["suite_c"]["test_bpb_improvement"] for r in reps]), 4)
    win_rec_acc_med = round(_med([r["suite_c"]["winner_test_recall_acc"] for r in reps]), 4)

    summary = {
        "experiment_id": "E138",
        "version": "v129",
        "round": "Round-12 RSI",
        "title": "Log-Linear Attention (Log-Linear Mamba-2 & Log-Linear Gated DeltaNet, arXiv:2506.04761v3, Mar 2026)",
        "eval_harness_sha256": reps[0]["eval_harness_sha256"],
        "num_repetitions": 3,
        "smoke_passed": bool(smoke["all_nine_gates_passed"]),
        "all_nine_gates_passed": all_passed,
        "gates": gates_agg,
        "suite0_summary": {
            "xor_level_mismatches_512x512": s0["xor_level_mismatches_512x512"],
            "max_fp32_rel_l2": max(r["suite_0"]["max_fp32_rel_l2"] for r in reps),
            "max_bf16_rel_l2": max(r["suite_0"]["max_bf16_rel_l2"] for r in reps),
            "collapsed_lambda_rel_l2": s0["collapsed_lambda_rel_l2"],
            "richardson_checks": s0["richardson_checks"],
            "redteam_caught": s0["redteam_caught"],
            "redteam_total": s0["redteam_total"],
            "num_rsi_kernels": s0["num_rsi_kernels"],
            "total_rsi_spills": s0["total_rsi_spills"],
            "monolithic_8level_negative_control_spills": s0["monolithic_8level_negative_control_spills"],
            "kernel_cubin_audits": s0["kernel_cubin_audits"],
        },
        "suite_a_summary": {
            "track1_speedup_vs_upstream_median": t1_up_med,
            "track1_speedup_vs_compile_median": t1_comp_med,
            "track2_speedup_vs_upstream_median": t2_up_med,
            "track2_speedup_vs_compile_median": t2_comp_med,
            "track1_rep1_shapes": reps[0]["suite_a"]["track1_candidates"],
            "track2_rep1_shapes": reps[0]["suite_a"]["track2_candidates"],
        },
        "suite_b_summary": {
            "world_size": sb0["world_size"],
            "steps_verified": sb0["steps_verified"],
            "max_loss_rel_diff": max(r["suite_b"]["max_loss_rel_diff"] for r in reps),
            "max_param_rel_l2": max(r["suite_b"]["max_param_rel_l2"] for r in reps),
            "max_grad_rel_l2": max(r["suite_b"]["max_grad_rel_l2"] for r in reps),
            "shard_local_no_coarse_handoff_rel_l2_bug": sb0["shard_local_no_coarse_handoff_rel_l2_bug"],
            "cp8_T8192_D64_speedup_median": cp_d64_med,
            "cp8_T8192_D128_speedup_median": cp_d128_med,
            "cp8_rep1_benchmarks": sb0["cp8_benchmarks"],
        },
        "suite_c_summary": {
            "incumbent_winner_id": reps[0]["suite_c"]["incumbent_winner_id"],
            "baseline_val_bpb_median": base_val_med,
            "winner_val_bpb_median": win_val_med,
            "val_bpb_improvement_median": val_imp_med,
            "baseline_test_bpb_median": base_test_med,
            "winner_test_bpb_median": win_test_med,
            "test_bpb_improvement_median": test_imp_med,
            "winner_test_recall_acc_median": win_rec_acc_med,
            "test_split_touches": reps[0]["suite_c"]["test_split_touches"],
            "rsi_loop_history_rep1": reps[0]["suite_c"]["rsi_loop_history"],
            "matched_step_5arm_scaling_law_rep1": reps[0]["suite_c"]["matched_step_5arm_scaling_law"],
        },
    }

    summary_json_path = raw_dir / "log_linear_attention_rsi_summary.json"
    summary_md_path = raw_dir / "log_linear_attention_rsi_summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")

    md_lines = [
        "# E138 (`v129`, Round-12 RSI): `Log-Linear Attention` (`H-Mamba-2` & `H-GDN`) 3-Repetition Summary",
        "",
        f"- **Frozen Evaluator `code/rsi_kernel_harness.py` SHA-256**: `{summary['eval_harness_sha256']}`",
        f"- **All 9 Falsification Gates Passed**: `{all_passed}` (`1` smoke + `3` full 8-GPU repetitions)",
        f"- **Bitwise XOR `clz` Fenwick Level Identity (`512x512` pairs)**: `{s0['xor_level_mismatches_512x512']}` mismatches",
        f"- **Zero-Spill Triton Kernels**: `{s0['num_rsi_kernels']}/{s0['num_rsi_kernels']}` kernels at `0` cubin spills (vs `{s0['monolithic_8level_negative_control_spills']}` spills for monolithic 8-level negative control)",
        f"- **Track 1 `Log-Linear Mamba-2` Geomean Speedup**: `{t1_up_med:.3f}x` vs Upstream `HanGuo97/log-linear-attention`, `{t1_comp_med:.3f}x` vs `torch.compile`",
        f"- **Track 2 `Log-Linear Gated DeltaNet` Geomean Speedup**: `{t2_up_med:.3f}x` vs Upstream `HanGuo97/log-linear-attention`, `{t2_comp_med:.3f}x` vs `torch.compile`",
        f"- **8-GPU `Fenwick-CP=8` Parity (`Gate G9`)**: `max_loss_rel_diff = {summary['suite_b_summary']['max_loss_rel_diff']:.2e}`, `max_param_rel_l2 = {summary['suite_b_summary']['max_param_rel_l2']:.2e}`, `D64 speedup = {cp_d64_med:.3f}x`, `D128 speedup = {cp_d128_med:.3f}x`",
        f"- **Fixed-Wall-Clock 8xA100 DDP `val_bpb` RSI Improvement**: `{base_val_med:.4f} -> {win_val_med:.4f}` (`-{val_imp_med:.4f} bpb`), Single-Touch `test_bpb`: `{base_test_med:.4f} -> {win_test_med:.4f}` (`recall_acc = {win_rec_acc_med * 100.0:.2f}%`)",
        "",
    ]
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8", newline="\n")

    # Write standardized 7-entry SHA256SUMS for raw directory
    tracked_files = [
        "log_linear_attention_rsi_rep1.json",
        "log_linear_attention_rsi_rep2.json",
        "log_linear_attention_rsi_rep3.json",
        "log_linear_attention_rsi_smoke.json",
        "log_linear_attention_rsi_summary.json",
        "suite0_redteam_ledger.jsonl",
        "suite_a_ledger_rep1.jsonl",
    ]
    sha_lines = [f"{_sha256_file(raw_dir / fname)}  {fname}" for fname in tracked_files]
    (raw_dir / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8", newline="\n")
    print(f"[E138] Wrote {summary_json_path}, {summary_md_path}, and SHA256SUMS ({len(tracked_files)} files).")
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
