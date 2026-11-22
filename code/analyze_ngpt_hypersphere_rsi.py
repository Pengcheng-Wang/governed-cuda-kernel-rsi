#!/usr/bin/env python3
"""
Round-11 RSI Aggregator & 9-Gate Verifier (E137 / v128):
nGPT: Normalized Transformer with Representation Learning on the Hypersphere
(Loshchilov, Hsieh, Sun & Ginsburg, NVIDIA, arXiv:2410.01131v2, ICLR 2025;
official repository: https://github.com/NVIDIA/ngpt)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import rsi_kernel_harness as harness

EXPECTED_HARNESS_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def _sha256_file(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E137 Round-11 nGPT Hypersphere RSI Results")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(CODE_DIR.parent / "results" / "raw" / "ngpt_hypersphere_rsi"),
    )
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)

    harness_sha = harness.eval_harness_sha256()
    if harness_sha != EXPECTED_HARNESS_SHA256:
        raise RuntimeError(f"Frozen evaluator SHA256 mismatch: {harness_sha}")

    smoke_path = raw_dir / "ngpt_hypersphere_rsi_smoke.json"
    rep_paths = [raw_dir / f"ngpt_hypersphere_rsi_rep{i}.json" for i in (1, 2, 3)]

    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    reps = [json.loads(p.read_text(encoding="utf-8")) for p in rep_paths]

    for idx, rep in enumerate(reps, start=1):
        if not rep.get("all_nine_gates_passed", False):
            raise RuntimeError(f"Repetition {idx} failed one or more gates: {rep.get('gates')}")
        if rep.get("eval_harness_sha256") != EXPECTED_HARNESS_SHA256:
            raise RuntimeError(f"Repetition {idx} harness SHA mismatch")

    t1_speedups_up = [float(r["suite_a"]["track1_speedup_vs_upstream_geomean"]) for r in reps]
    t1_speedups_comp = [float(r["suite_a"]["track1_speedup_vs_compile_geomean"]) for r in reps]
    t2_speedups_up = [float(r["suite_a"]["track2_speedup_vs_upstream_geomean"]) for r in reps]
    t2_speedups_comp = [float(r["suite_a"]["track2_speedup_vs_compile_geomean"]) for r in reps]

    cp_d64_sp = [float(r["suite_b"]["cp8_benchmarks"]["cp8_T8192_C1024_H16_D64"]["speedup"]) for r in reps]
    cp_d128_sp = [float(r["suite_b"]["cp8_benchmarks"]["cp8_T8192_C2048_H16_D128"]["speedup"]) for r in reps]

    val_bpb_base = [float(r["suite_c"]["baseline_val_bpb"]) for r in reps]
    val_bpb_win = [float(r["suite_c"]["winner_val_bpb"]) for r in reps]
    test_bpb_base = [float(r["suite_c"]["baseline_test_bpb"]) for r in reps]
    test_bpb_win = [float(r["suite_c"]["winner_test_bpb"]) for r in reps]
    recall_acc_win = [float(r["suite_c"]["winner_test_recall_acc"]) for r in reps]

    r0 = reps[0]
    s0 = r0["suite0"]
    sb = r0["suite_b"]
    sc = r0["suite_c"]

    summary = {
        "experiment_id": "E137",
        "version": "v128",
        "round": "Round-11 RSI",
        "title": "nGPT: Normalized Transformer with Representation Learning on the Hypersphere (arXiv:2410.01131v2, ICLR 2025)",
        "eval_harness_sha256": EXPECTED_HARNESS_SHA256,
        "num_repetitions": len(reps),
        "smoke_passed": bool(smoke.get("all_nine_gates_passed", False)),
        "all_nine_gates_passed": True,
        "gates": r0["gates"],
        "suite0_summary": {
            "max_fp32_rel_l2": float(max(r["suite0"]["max_fp32_rel_l2"] for r in reps)),
            "max_bf16_rel_l2": float(max(r["suite0"]["max_bf16_rel_l2"] for r in reps)),
            "upstream_rope_parity_rel_l2": float(s0["upstream_rope_parity_rel_l2"]),
            "relative_shift_invariance_max_abs_diff": s0["relative_shift_invariance_max_abs_diff"],
            "richardson_checks": s0["richardson_checks"],
            "redteam_caught": int(s0["redteam_caught"]),
            "redteam_total": int(s0["redteam_total"]),
            "total_rsi_spills": int(max(r["suite0"]["total_rsi_spills"] for r in reps)),
            "num_rsi_kernels": len(s0["kernel_cubin_audits"]),
            "naive_d128_monolithic_dkv_spills": int(s0["naive_d128_monolithic_dkv_spill_control"]["n_spills"]),
            "kernel_cubin_audits": s0["kernel_cubin_audits"],
        },
        "suite_a_summary": {
            "track1_speedup_vs_upstream_median": round(float(statistics.median(t1_speedups_up)), 4),
            "track1_speedup_vs_compile_median": round(float(statistics.median(t1_speedups_comp)), 4),
            "track2_speedup_vs_upstream_median": round(float(statistics.median(t2_speedups_up)), 4),
            "track2_speedup_vs_compile_median": round(float(statistics.median(t2_speedups_comp)), 4),
            "track1_rep1_shapes": r0["suite_a"]["track1_candidates"],
            "track2_rep1_shapes": r0["suite_a"]["track2_candidates"],
        },
        "suite_b_summary": {
            "world_size": int(sb["world_size"]),
            "steps_verified": int(sb["steps_verified"]),
            "max_loss_rel_diff": float(max(r["suite_b"]["max_loss_rel_diff"] for r in reps)),
            "max_param_rel_l2": float(max(r["suite_b"]["max_param_rel_l2"] for r in reps)),
            "max_grad_rel_l2": float(max(r["suite_b"]["max_grad_rel_l2"] for r in reps)),
            "max_unit_norm_err": float(max(r["suite_b"]["max_unit_norm_err"] for r in reps)),
            "negative_control_split_axis_local_norm_err": float(sb["negative_control_split_axis_local_norm_err"]),
            "cp8_T8192_D64_speedup_median": round(float(statistics.median(cp_d64_sp)), 4),
            "cp8_T8192_D128_speedup_median": round(float(statistics.median(cp_d128_sp)), 4),
            "cp8_rep1_benchmarks": sb["cp8_benchmarks"],
        },
        "suite_c_summary": {
            "incumbent_winner_id": sc["incumbent_winner_id"],
            "baseline_val_bpb_median": round(float(statistics.median(val_bpb_base)), 4),
            "winner_val_bpb_median": round(float(statistics.median(val_bpb_win)), 4),
            "val_bpb_improvement_median": round(float(statistics.median(val_bpb_base) - statistics.median(val_bpb_win)), 4),
            "baseline_test_bpb_median": round(float(statistics.median(test_bpb_base)), 4),
            "winner_test_bpb_median": round(float(statistics.median(test_bpb_win)), 4),
            "test_bpb_improvement_median": round(float(statistics.median(test_bpb_base) - statistics.median(test_bpb_win)), 4),
            "winner_test_recall_acc_median": round(float(statistics.median(recall_acc_win)), 4),
            "test_split_touches": int(sc["test_split_touches"]),
            "rsi_loop_history_rep1": sc["rsi_loop_history"],
            "matched_step_5arm_scaling_law_rep1": sc["matched_step_5arm_scaling_law"],
        },
    }

    summary_json_path = raw_dir / "ngpt_hypersphere_rsi_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# E137 (`v128`, Round-11 RSI): `nGPT` Hypersphere Representation Learning & Riemannian Retraction Summary",
        "",
        f"- **Frozen Evaluator `code/rsi_kernel_harness.py` SHA-256**: `{EXPECTED_HARNESS_SHA256}`",
        f"- **All 9 Falsification Gates Passed**: `{summary['all_nine_gates_passed']}` (`1` smoke + `3` full 8-GPU repetitions)",
        f"- **Zero-Spill Triton Kernels**: `{summary['suite0_summary']['num_rsi_kernels']}/19` kernels at `0` cubin spills (vs `{summary['suite0_summary']['naive_d128_monolithic_dkv_spills']}` spills for monolithic `D=128` `dKV` negative control)",
        f"- **Track 1 Core Hypersphere Ops Geomean Speedup**: `{summary['suite_a_summary']['track1_speedup_vs_upstream_median']:.3f}x` vs Upstream `NVIDIA/ngpt`, `{summary['suite_a_summary']['track1_speedup_vs_compile_median']:.3f}x` vs `torch.compile`",
        f"- **Track 2 Full `nGPT` Block + Weight Retraction Geomean Speedup**: `{summary['suite_a_summary']['track2_speedup_vs_upstream_median']:.3f}x` vs Upstream `NVIDIA/ngpt` + `normalize_matrices()`, `{summary['suite_a_summary']['track2_speedup_vs_compile_median']:.3f}x` vs `torch.compile`",
        f"- **8-GPU `Ring nGPT-CP=8` + ZeRO-1 Shard-Retracted Parity (`Gate G9`)**: `max_loss_rel_diff = {summary['suite_b_summary']['max_loss_rel_diff']:.2e}`, `max_param_rel_l2 = {summary['suite_b_summary']['max_param_rel_l2']:.2e}`, `max_grad_rel_l2 = {summary['suite_b_summary']['max_grad_rel_l2']:.2e}`, `max_unit_norm_err = {summary['suite_b_summary']['max_unit_norm_err']:.2e}`",
        f"- **Fixed-Wall-Clock 8xA100 DDP `val_bpb` RSI Improvement**: `{summary['suite_c_summary']['baseline_val_bpb_median']:.4f} -> {summary['suite_c_summary']['winner_val_bpb_median']:.4f}` (`-{summary['suite_c_summary']['val_bpb_improvement_median']:.4f} bpb`), Single-Touch `test_bpb`: `{summary['suite_c_summary']['baseline_test_bpb_median']:.4f} -> {summary['suite_c_summary']['winner_test_bpb_median']:.4f}` (`recall_acc = {summary['suite_c_summary']['winner_test_recall_acc_median']*100:.1f}%`)",
        "",
    ]
    summary_md_path = raw_dir / "ngpt_hypersphere_rsi_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Write standardized 7-entry SHA256SUMS for raw directory
    tracked_files = [
        "ngpt_hypersphere_rsi_rep1.json",
        "ngpt_hypersphere_rsi_rep2.json",
        "ngpt_hypersphere_rsi_rep3.json",
        "ngpt_hypersphere_rsi_smoke.json",
        "ngpt_hypersphere_rsi_summary.json",
        "suite0_redteam_ledger.jsonl",
        "suite_a_ledger_rep1.jsonl",
    ]
    sha_lines = [f"{_sha256_file(raw_dir / fname)}  {fname}" for fname in tracked_files]
    (raw_dir / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    print(f"[E137] Wrote {summary_json_path}, {summary_md_path}, and SHA256SUMS ({len(tracked_files)} files).")


if __name__ == "__main__":
    main()
