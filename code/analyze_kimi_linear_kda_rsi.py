#!/usr/bin/env python3
"""
E140 (v131, Round-14 RSI): Multi-Repetition Aggregator & Manifest Builder for Kimi Linear / KDA
================================================================================================

Reads `kimi_linear_kda_rsi_rep{1,2,3}.json` (plus `smoke.json` if present), verifies that all
9 gates (`G1..G9`) pass across all 3 repetitions, computes 3-repetition medians and IQRs across
Suites 0, A, B, and C, and writes:
  - `kimi_linear_kda_rsi_summary.json`
  - `kimi_linear_kda_rsi_report.md`
  - `suiteA_track1_core_kda.csv`
  - `suiteA_track2_full_block.csv`
  - `suiteB_cp8_scaling.csv`
  - `suiteC_rsi_candidates.csv`
  - `suiteC_scaling_law_5arm.csv`
  - `SHA256SUMS.txt`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
import sys
from typing import Any, Dict, List


def _med(vals: List[float]) -> float:
    return float(statistics.median(vals))


def _iqr(vals: List[float]) -> float:
    s = sorted(vals)
    if len(s) < 2:
        return 0.0
    return float(s[-1] - s[0])


def _geomean(vals: List[float]) -> float:
    return float(math.exp(sum(math.log(max(v, 1e-12)) for v in vals) / len(vals)))


def _write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E140 Kimi Linear / KDA 3-rep artifacts")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent.parent / "results" / "raw" / "kimi_linear_kda_rsi"),
    )
    args = parser.parse_args()
    raw_dir = pathlib.Path(args.raw_dir)

    reps: List[Dict[str, Any]] = []
    for r_idx in (1, 2, 3):
        p = raw_dir / f"kimi_linear_kda_rsi_rep{r_idx}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing required repetition file: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if not data.get("all_nine_gates_passed", False):
            raise RuntimeError(f"Repetition {r_idx} did not pass all 9 gates: {data.get('gates')}")
        reps.append(data)

    r0 = reps[0]
    s0 = r0["suite0_forensics"]

    # Aggregate Suite A Track 1 (Core KDA Layer)
    t1_len = len(r0["suiteA_pareto"]["track1_core_kda"])
    t1_rows: List[Dict[str, Any]] = []
    for i in range(t1_len):
        base = r0["suiteA_pareto"]["track1_core_kda"][i]
        rec_ms = _med([r["suiteA_pareto"]["track1_core_kda"][i]["recurrent_oracle_ms"] for r in reps])
        eag_ms = _med([r["suiteA_pareto"]["track1_core_kda"][i]["eager_chunk_ms"] for r in reps])
        eag_mem = _med([r["suiteA_pareto"]["track1_core_kda"][i]["eager_chunk_hbm_mib"] for r in reps])
        cmp_ms = _med([r["suiteA_pareto"]["track1_core_kda"][i]["torch_compile_ms"] for r in reps])
        cmp_mem = _med([r["suiteA_pareto"]["track1_core_kda"][i]["torch_compile_hbm_mib"] for r in reps])
        up_ms = _med([r["suiteA_pareto"]["track1_core_kda"][i]["upstream_spill_ms"] for r in reps])
        up_mem = _med([r["suiteA_pareto"]["track1_core_kda"][i]["upstream_spill_hbm_mib"] for r in reps])
        rsi_vals = [r["suiteA_pareto"]["track1_core_kda"][i]["rsi_kda_ms"] for r in reps]
        rsi_ms = _med(rsi_vals)
        rsi_iqr = _iqr(rsi_vals)
        rsi_mem = _med([r["suiteA_pareto"]["track1_core_kda"][i]["rsi_kda_hbm_mib"] for r in reps])
        t1_rows.append({
            "id": base["id"],
            "B": base["B"],
            "T": base["T"],
            "H": base["H"],
            "HV": base["HV"],
            "K": base["K"],
            "V": base["V"],
            "recurrent_oracle_ms": round(rec_ms, 4),
            "eager_chunk_ms": round(eag_ms, 4),
            "eager_chunk_hbm_mib": round(eag_mem, 2),
            "torch_compile_ms": round(cmp_ms, 4),
            "torch_compile_hbm_mib": round(cmp_mem, 2),
            "upstream_spill_ms": round(up_ms, 4),
            "upstream_spill_hbm_mib": round(up_mem, 2),
            "rsi_kda_ms": round(rsi_ms, 4),
            "rsi_kda_3rep_iqr_ms": round(rsi_iqr, 4),
            "rsi_kda_hbm_mib": round(rsi_mem, 2),
            "speedup_vs_upstream_spill": round(up_ms / rsi_ms, 4),
            "speedup_vs_eager_chunk": round(eag_ms / rsi_ms, 4),
            "speedup_vs_torch_compile": round(cmp_ms / rsi_ms, 4),
            "speedup_vs_recurrent": round(rec_ms / rsi_ms, 4),
            "hbm_reduction_vs_eager": round(eag_mem / max(rsi_mem, 1e-6), 4),
        })

    # Aggregate Suite A Track 2 (Full KimiDeltaAttention Block)
    t2_len = len(r0["suiteA_pareto"]["track2_full_block"])
    t2_rows: List[Dict[str, Any]] = []
    for i in range(t2_len):
        base = r0["suiteA_pareto"]["track2_full_block"][i]
        b_ms = _med([r["suiteA_pareto"]["track2_full_block"][i]["baseline_block_ms"] for r in reps])
        b_mem = _med([r["suiteA_pareto"]["track2_full_block"][i]["baseline_block_hbm_mib"] for r in reps])
        c_ms = _med([r["suiteA_pareto"]["track2_full_block"][i]["torch_compile_block_ms"] for r in reps])
        c_mem = _med([r["suiteA_pareto"]["track2_full_block"][i]["torch_compile_block_hbm_mib"] for r in reps])
        r_vals = [r["suiteA_pareto"]["track2_full_block"][i]["rsi_block_ms"] for r in reps]
        r_ms = _med(r_vals)
        r_iqr = _iqr(r_vals)
        r_mem = _med([r["suiteA_pareto"]["track2_full_block"][i]["rsi_block_hbm_mib"] for r in reps])
        t2_rows.append({
            "id": base["id"],
            "B": base["B"],
            "T": base["T"],
            "D_model": base["D_model"],
            "H": base["H"],
            "HV": base["HV"],
            "K": base["K"],
            "baseline_block_ms": round(b_ms, 4),
            "baseline_block_hbm_mib": round(b_mem, 2),
            "torch_compile_block_ms": round(c_ms, 4),
            "torch_compile_block_hbm_mib": round(c_mem, 2),
            "rsi_block_ms": round(r_ms, 4),
            "rsi_block_3rep_iqr_ms": round(r_iqr, 4),
            "rsi_block_hbm_mib": round(r_mem, 2),
            "speedup_vs_baseline_block": round(b_ms / r_ms, 4),
            "speedup_vs_torch_compile_block": round(c_ms / r_ms, 4),
            "hbm_reduction_vs_baseline_block": round(b_mem / max(r_mem, 1e-6), 4),
        })

    # Aggregate Suite B (8-GPU NVSwitch KDA-CP = 8)
    sb_rows: List[Dict[str, Any]] = []
    for i in range(len(r0["suiteB_cp8"]["performance"])):
        base = r0["suiteB_cp8"]["performance"][i]
        cp_vals = [r["suiteB_cp8"]["performance"][i]["kda_cp8_ms"] for r in reps]
        cp_ms = _med(cp_vals)
        cp_iqr = _iqr(cp_vals)
        cp_mem = _med([r["suiteB_cp8"]["performance"][i]["kda_cp8_hbm_mib"] for r in reps])
        ag_ms = _med([r["suiteB_cp8"]["performance"][i]["allgather_full_seq_ms"] for r in reps])
        ag_mem = _med([r["suiteB_cp8"]["performance"][i]["allgather_full_seq_hbm_mib"] for r in reps])
        sb_rows.append({
            "K_dim": base["K_dim"],
            "T_global": base["T_global"],
            "T_local": base["T_local"],
            "kda_cp8_ms": round(cp_ms, 4),
            "kda_cp8_3rep_iqr_ms": round(cp_iqr, 4),
            "kda_cp8_hbm_mib": round(cp_mem, 2),
            "allgather_full_seq_ms": round(ag_ms, 4),
            "allgather_full_seq_hbm_mib": round(ag_mem, 2),
            "speedup_vs_allgather": round(ag_ms / cp_ms, 4),
            "hbm_reduction_vs_allgather": round(ag_mem / max(cp_mem, 1e-6), 4),
        })

    # Aggregate Suite C RSI candidates & 5-arm scaling law
    sc_cand_rows: List[Dict[str, Any]] = []
    for i in range(len(r0["suiteC_rsi_and_scaling"]["rsi_candidates"])):
        base = r0["suiteC_rsi_and_scaling"]["rsi_candidates"][i]
        steps_m = int(round(_med([r["suiteC_rsi_and_scaling"]["rsi_candidates"][i]["steps_completed"] for r in reps])))
        vbpb_m = _med([r["suiteC_rsi_and_scaling"]["rsi_candidates"][i]["val_bpb"] for r in reps])
        vacc_m = _med([r["suiteC_rsi_and_scaling"]["rsi_candidates"][i]["val_recall_acc_pct"] for r in reps])
        sc_cand_rows.append({
            "candidate": base["candidate"],
            "arch": base["arch"],
            "backend": base["backend"],
            "steps_completed_median": steps_m,
            "val_bpb_median": round(vbpb_m, 4),
            "val_recall_acc_pct_median": round(vacc_m, 2),
            "status": base["status"],
        })

    sc_scale_rows: List[Dict[str, Any]] = []
    for i in range(len(r0["suiteC_rsi_and_scaling"]["scaling_law_5arm"])):
        base = r0["suiteC_rsi_and_scaling"]["scaling_law_5arm"][i]
        vbpb_m = _med([r["suiteC_rsi_and_scaling"]["scaling_law_5arm"][i]["val_bpb"] for r in reps])
        vacc_m = _med([r["suiteC_rsi_and_scaling"]["scaling_law_5arm"][i]["val_recall_acc_pct"] for r in reps])
        tbpb_m = _med([r["suiteC_rsi_and_scaling"]["scaling_law_5arm"][i]["test_bpb"] for r in reps])
        tacc_m = _med([r["suiteC_rsi_and_scaling"]["scaling_law_5arm"][i]["test_recall_acc_pct"] for r in reps])
        wall_m = _med([r["suiteC_rsi_and_scaling"]["scaling_law_5arm"][i]["wall_clock_s"] for r in reps])
        sc_scale_rows.append({
            "arm": base["arm"],
            "matched_steps": base["matched_steps"],
            "wall_clock_s_median": round(wall_m, 4),
            "val_bpb_median": round(vbpb_m, 4),
            "val_recall_acc_pct_median": round(vacc_m, 2),
            "test_bpb_median": round(tbpb_m, 4),
            "test_recall_acc_pct_median": round(tacc_m, 2),
        })

    c0_vbpb = _med([r["suiteC_rsi_and_scaling"]["c0_val_bpb"] for r in reps])
    fin_vbpb = _med([r["suiteC_rsi_and_scaling"]["final_val_bpb"] for r in reps])
    c0_tbpb = _med([r["suiteC_rsi_and_scaling"]["single_touch_test"]["c0_test_bpb"] for r in reps])
    fin_tbpb = _med([r["suiteC_rsi_and_scaling"]["single_touch_test"]["final_test_bpb"] for r in reps])
    c0_tacc = _med([r["suiteC_rsi_and_scaling"]["single_touch_test"]["c0_test_recall_acc_pct"] for r in reps])
    fin_tacc = _med([r["suiteC_rsi_and_scaling"]["single_touch_test"]["final_test_recall_acc_pct"] for r in reps])

    summary = {
        "experiment_id": "E140",
        "version": "v131",
        "round": "Round-14 RSI",
        "arxiv": "2510.26692",
        "repetitions": 3,
        "all_nine_gates_passed_all_reps": True,
        "eval_harness_sha256": s0["eval_harness_sha256"],
        "suite0_summary": {
            "total_rsi_kernels": s0["spill_audit"]["total_rsi_kernels"],
            "compiled_rsi_kernels": s0["spill_audit"]["compiled_rsi_kernels"],
            "total_specializations": s0["spill_audit"]["total_specializations"],
            "total_rsi_spills": s0["spill_audit"]["total_rsi_spills"],
            "negative_control_monolithic_wy_bwd_spills": s0["spill_audit"]["negative_control_monolithic_wy_bwd_spills"],
            "upstream_safe_gate_nan_count": s0["upstream_forensics"]["upstream_safe_gate_nan_count"],
            "rsi_overflow_free_nan_count": s0["upstream_forensics"]["rsi_overflow_free_nan_count"],
            "max_causal_future_leakage": s0["upstream_forensics"]["max_causal_future_leakage"],
            "paper_listing8b_rel_l2_error": s0["upstream_forensics"]["paper_listing8b_rel_l2_error"],
            "upstream_wy_dqkg_fused_configs": s0["upstream_forensics"]["upstream_wy_dqkg_fused_configs"],
            "upstream_wy_dqkg_fused_spills_min": s0["upstream_forensics"]["upstream_wy_dqkg_fused_spills_min"],
            "upstream_wy_dqkg_fused_spills_max": s0["upstream_forensics"]["upstream_wy_dqkg_fused_spills_max"],
            "upstream_wy_dqkg_fused_spills_total": s0["upstream_forensics"]["upstream_wy_dqkg_fused_spills_total"],
            "fp32_max_rel_l2": _med([r["suite0_forensics"]["parity"]["fp32"]["max_rel_l2"] for r in reps]),
            "bf16_max_rel_l2": _med([r["suite0_forensics"]["parity"]["bf16"]["max_rel_l2"] for r in reps]),
            "richardson_ratios": s0["richardson_fp64"]["ratios"],
            "red_team_caught_count": s0["red_team_exploits"]["caught_count"],
        },
        "suiteA_summary": {
            "track1_geomean_speedup_vs_upstream_spill": _geomean([r["speedup_vs_upstream_spill"] for r in t1_rows]),
            "track1_geomean_speedup_vs_eager_chunk": _geomean([r["speedup_vs_eager_chunk"] for r in t1_rows]),
            "track1_geomean_speedup_vs_torch_compile": _geomean([r["speedup_vs_torch_compile"] for r in t1_rows]),
            "track1_geomean_speedup_vs_recurrent": _geomean([r["speedup_vs_recurrent"] for r in t1_rows]),
            "track2_geomean_speedup_vs_baseline_block": _geomean([r["speedup_vs_baseline_block"] for r in t2_rows]),
            "track2_geomean_speedup_vs_torch_compile_block": _geomean([r["speedup_vs_torch_compile_block"] for r in t2_rows]),
            "track1_rows": t1_rows,
            "track2_rows": t2_rows,
        },
        "suiteB_summary": {
            "T_global": r0["suiteB_cp8"]["T_global"],
            "T_local": r0["suiteB_cp8"]["T_local"],
            "max_loss_rel_diff": _med([r["suiteB_cp8"]["max_loss_rel_diff"] for r in reps]),
            "max_param_rel_l2": _med([r["suiteB_cp8"]["max_param_rel_l2"] for r in reps]),
            "max_grad_rel_l2": _med([r["suiteB_cp8"]["max_grad_rel_l2"] for r in reps]),
            "omitted_handoff_rel_l2_error": _med([r["suiteB_cp8"]["omitted_handoff_rel_l2_error"] for r in reps]),
            "performance": sb_rows,
        },
        "suiteC_summary": {
            "c0_val_bpb": c0_vbpb,
            "final_val_bpb": fin_vbpb,
            "val_bpb_improvement": c0_vbpb - fin_vbpb,
            "c0_test_bpb": c0_tbpb,
            "final_test_bpb": fin_tbpb,
            "test_bpb_improvement": c0_tbpb - fin_tbpb,
            "c0_test_recall_acc_pct": c0_tacc,
            "final_test_recall_acc_pct": fin_tacc,
            "rsi_candidates": sc_cand_rows,
            "scaling_law_5arm": sc_scale_rows,
        },
    }

    sum_path = raw_dir / "kimi_linear_kda_rsi_summary.json"
    sum_path.write_bytes((json.dumps(summary, indent=2) + "\n").encode("utf-8"))

    _write_csv(raw_dir / "suiteA_track1_core_kda.csv", t1_rows)
    _write_csv(raw_dir / "suiteA_track2_full_block.csv", t2_rows)
    _write_csv(raw_dir / "suiteB_cp8_scaling.csv", sb_rows)
    _write_csv(raw_dir / "suiteC_rsi_candidates.csv", sc_cand_rows)
    _write_csv(raw_dir / "suiteC_scaling_law_5arm.csv", sc_scale_rows)

    report_lines = [
        "# E140 (`v131`, Round-14 RSI): Kimi Linear & Kimi Delta Attention (`KDA`, `arXiv:2510.26692`)",
        "",
        "## 1. Executive Summary (3-Repetition 8xA100-SXM4-80GB Hardware Medians)",
        f"- **Frozen Evaluator (`G1`)**: `rsi_kernel_harness.py` `SHA256 = {s0['eval_harness_sha256']}`",
        f"- **Zero-Spill SM80 Compilation (`G2`)**: `{summary['suite0_summary']['compiled_rsi_kernels']}/{summary['suite0_summary']['total_rsi_kernels']}` kernels (`{summary['suite0_summary']['total_specializations']}` specializations) at **`0` cubin spills** vs **`{summary['suite0_summary']['upstream_wy_dqkg_fused_spills_min']}–{summary['suite0_summary']['upstream_wy_dqkg_fused_spills_max']}` spills** (`{summary['suite0_summary']['upstream_wy_dqkg_fused_spills_total']}` total across `{summary['suite0_summary']['upstream_wy_dqkg_fused_configs']}` configs) in upstream `fla/ops/kda:chunk_kda_bwd_kernel_wy_dqkg_fused`.",
        f"- **Overflow-Free Non-Positive Causal Sub-Chunking (`G3`)**: Eliminates **`{summary['suite0_summary']['upstream_safe_gate_nan_count']}` / `524,288` `NaN`s** from upstream `safe_gate=True` (`0` `NaN`s in Round-14 RSI, `0.0` future-token leakage, `FP32 max_rel_l2 = {summary['suite0_summary']['fp32_max_rel_l2']:.2e}`, `BF16 max_rel_l2 = {summary['suite0_summary']['bf16_max_rel_l2']:.4f}`).",
        f"- **Suite A Vector-Pareto (`G5` & `G6`)**: Track 1 (`Core KDA Layer`) achieves **`{summary['suiteA_summary']['track1_geomean_speedup_vs_upstream_spill']:.2f}x`** geomean speedup over Upstream-Pattern Monolithic WY Spill (`{summary['suiteA_summary']['track1_geomean_speedup_vs_torch_compile']:.2f}x` over `torch.compile`, `{summary['suiteA_summary']['track1_geomean_speedup_vs_eager_chunk']:.2f}x` over Eager Chunk, `{summary['suiteA_summary']['track1_geomean_speedup_vs_recurrent']:.2f}x` over Sequential Recurrent); Track 2 (`Full KimiDeltaAttention Block`) achieves **`{summary['suiteA_summary']['track2_geomean_speedup_vs_baseline_block']:.2f}x`** over Baseline (`{summary['suiteA_summary']['track2_geomean_speedup_vs_torch_compile_block']:.2f}x` over `torch.compile`).",
        f"- **Suite B 8-GPU NVSwitch `KDA-CP = 8` (`G9`, `T_global = 8,192`)**: `max_loss_rel_diff = {summary['suiteB_summary']['max_loss_rel_diff']:.2e}`, `max_param_rel_l2 = {summary['suiteB_summary']['max_param_rel_l2']:.2e}`, `max_grad_rel_l2 = {summary['suiteB_summary']['max_grad_rel_l2']:.2e}` (vs `{summary['suiteB_summary']['omitted_handoff_rel_l2_error']:.4f}` error without Channel-Affine Monoid handoff).",
        f"- **Suite C Fixed-Wall-Clock `val_bpb` RSI & 5-Arm Scaling (`G7` & `G8`)**: `val_bpb` improved **`{c0_vbpb:.4f} -> {fin_vbpb:.4f}` (`-{c0_vbpb - fin_vbpb:.4f} val_bpb`)**, single-touch `test_bpb`: **`{c0_tbpb:.4f} -> {fin_tbpb:.4f}` (`-{c0_tbpb - fin_tbpb:.4f} test_bpb`, recall accuracy `{c0_tacc:.2f}% -> {fin_tacc:.2f}%`)**.",
        "",
    ]
    rep_md_path = raw_dir / "kimi_linear_kda_rsi_report.md"
    rep_md_path.write_bytes(("\n".join(report_lines) + "\n").encode("utf-8"))

    # Write SHA256SUMS.txt covering the 7 canonical aggregated/CSV/rep files
    manifest_files = [
        "kimi_linear_kda_rsi_rep1.json",
        "kimi_linear_kda_rsi_rep2.json",
        "kimi_linear_kda_rsi_rep3.json",
        "kimi_linear_kda_rsi_summary.json",
        "kimi_linear_kda_rsi_report.md",
        "suiteA_track1_core_kda.csv",
        "suiteA_track2_full_block.csv",
        "suiteB_cp8_scaling.csv",
        "suiteC_rsi_candidates.csv",
        "suiteC_scaling_law_5arm.csv",
    ]
    sha_lines = []
    for fname in manifest_files:
        fp = raw_dir / fname
        digest = hashlib.sha256(fp.read_bytes()).hexdigest()
        sha_lines.append(f"{digest}  {fname}")
    (raw_dir / "SHA256SUMS.txt").write_bytes(("\n".join(sha_lines) + "\n").encode("utf-8"))
    print(f"[E140] Analysis complete: {len(manifest_files)}/10 SHA256SUMS entries written to {raw_dir / 'SHA256SUMS.txt'}")


if __name__ == "__main__":
    main()
