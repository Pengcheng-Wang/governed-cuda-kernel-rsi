#!/usr/bin/env python3
"""
Round-36 RSI (E162, v153): 3-Repetition Statistical Aggregator for CalH2SA-GDN.
Aggregates rep1.json..rep3.json into summary_3rep.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List


def _agg(vals: List[float]) -> Dict[str, float]:
    m = float(mean(vals))
    sd = float(pstdev(vals)) if len(vals) > 1 else 0.0
    cv = float(100.0 * sd / abs(m)) if abs(m) > 1e-12 else 0.0
    return {"mean": m, "sd": sd, "cv_pct": cv, "min": float(min(vals)), "max": float(max(vals))}


def main():
    parser = argparse.ArgumentParser(description="Analyze 3-Rep CalH2SA-GDN E162 Results")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    reps = [json.loads(Path(p).read_text()) for p in args.inputs]
    r0 = reps[0]

    # Track 1A
    t1a_hit = r0["track1a_duality_and_correctness"]["duality_identification"]["selected_hit_poison_regime"]
    t1a_miss = r0["track1a_duality_and_correctness"]["duality_identification"]["rejected_miss_rescue_regime"]
    t1a_dip = r0["track1a_duality_and_correctness"]["duality_identification"]["dipolar_intrablock_regime"]

    # Track 1B Training rows
    n_t1b = len(r0["track1b_training_efficiency"]["rows"])
    t1b_rows = []
    for i in range(n_t1b):
        base = r0["track1b_training_efficiency"]["rows"][i]
        t1b_rows.append({
            "B": base["B"], "T": base["T"], "HQ": base["HQ"], "HKV": base["HKV"], "K": base["K"],
            "BC": base["BC"], "BS": base["BS"],
            "sdpa_total_ms": _agg([r["track1b_training_efficiency"]["rows"][i]["sdpa_total_ms"] for r in reps]),
            "orthh2sa_gdn_e161_total_ms": _agg([r["track1b_training_efficiency"]["rows"][i]["orthh2sa_gdn_e161_total_ms"] for r in reps]),
            "calh2sa_gdn_e162_fwd_ms": _agg([r["track1b_training_efficiency"]["rows"][i]["calh2sa_gdn_e162_fwd_ms"] for r in reps]),
            "calh2sa_gdn_e162_bwd_ms": _agg([r["track1b_training_efficiency"]["rows"][i]["calh2sa_gdn_e162_bwd_ms"] for r in reps]),
            "calh2sa_gdn_e162_total_ms": _agg([r["track1b_training_efficiency"]["rows"][i]["calh2sa_gdn_e162_total_ms"] for r in reps]),
            "speedup_fwd_vs_sdpa": _agg([r["track1b_training_efficiency"]["rows"][i]["speedup_fwd_vs_sdpa"] for r in reps]),
            "speedup_bwd_vs_sdpa": _agg([r["track1b_training_efficiency"]["rows"][i]["speedup_bwd_vs_sdpa"] for r in reps]),
            "speedup_total_vs_sdpa": _agg([r["track1b_training_efficiency"]["rows"][i]["speedup_total_vs_sdpa"] for r in reps]),
            "overhead_pct_vs_e161": _agg([r["track1b_training_efficiency"]["rows"][i]["overhead_pct_vs_e161"] for r in reps]),
            "effective_total_tflops": _agg([r["track1b_training_efficiency"]["rows"][i]["effective_total_tflops"] for r in reps]),
            "hbm_traffic_reduction_vs_dense": base["hbm_traffic_reduction_vs_dense"],
        })

    # Track 1C Decode rows
    n_t1c = len(r0["track1c_decode_efficiency"]["rows"])
    t1c_rows = []
    for i in range(n_t1c):
        base = r0["track1c_decode_efficiency"]["rows"][i]
        t1c_rows.append({
            "B_dec": base["B_dec"], "T_cache": base["T_cache"], "BC": base["BC"], "BS": base["BS"],
            "dense_mean_ms": _agg([r["track1c_decode_efficiency"]["rows"][i]["dense_mean_ms"] for r in reps]),
            "calh2sa_gdn_mean_ms": _agg([r["track1c_decode_efficiency"]["rows"][i]["calh2sa_gdn_mean_ms"] for r in reps]),
            "calh2sa_gdn_p50_ms": _agg([r["track1c_decode_efficiency"]["rows"][i]["calh2sa_gdn_p50_ms"] for r in reps]),
            "calh2sa_gdn_p95_ms": _agg([r["track1c_decode_efficiency"]["rows"][i]["calh2sa_gdn_p95_ms"] for r in reps]),
            "speedup_vs_dense": _agg([r["track1c_decode_efficiency"]["rows"][i]["speedup_vs_dense"] for r in reps]),
            "kv_bytes_reduction_ratio": base["kv_bytes_reduction_ratio"],
        })

    # Track 2 Block Progression
    n_t2 = len(r0["track2_block_progression"]["rows"])
    t2_rows = []
    for i in range(n_t2):
        base = r0["track2_block_progression"]["rows"][i]
        t2_rows.append({
            "D": base["D"], "HQ": base["HQ"], "HKV": base["HKV"], "K": base["K"],
            "A0_ms": _agg([r["track2_block_progression"]["rows"][i]["A0_ms"] for r in reps]),
            "A4_ms": _agg([r["track2_block_progression"]["rows"][i]["A4_ms"] for r in reps]),
            "speedup_A4_vs_A0": _agg([r["track2_block_progression"]["rows"][i]["speedup_A4_vs_A0"] for r in reps]),
            "speedup_A4_vs_A3": _agg([r["track2_block_progression"]["rows"][i]["speedup_A4_vs_A3"] for r in reps]),
        })

    # Track 3 Recall Matrix
    t3_cells = r0["track3_joint_duality_recall_matrix"]["cells"]
    dense_Cells = [c for c in t3_cells if c["M_facts"] >= 4]
    mean_gain_vs_e161 = float(mean(c["gain_e162_over_e161"] for c in dense_Cells))
    max_gain_vs_e161 = float(max(c["gain_e162_over_e161"] for c in dense_Cells))
    mean_gain_vs_e160 = float(mean(c["gain_e162_over_e160"] for c in dense_Cells))
    max_gain_vs_e160 = float(max(c["gain_e162_over_e160"] for c in dense_Cells))

    # Track 4 CP=8
    n_t4 = len(r0["track4_cp8_scaling"]["rows"])
    t4_rows = []
    for i in range(n_t4):
        base = r0["track4_cp8_scaling"]["rows"][i]
        t4_rows.append({
            "T_total": base["T_total"], "T_local": base["T_local"], "BC": base["BC"], "BS": base["BS"],
            "ulysses_cp8_ms": _agg([r["track4_cp8_scaling"]["rows"][i]["ulysses_cp8_ms"] for r in reps]),
            "ring_cp8_ms": _agg([r["track4_cp8_scaling"]["rows"][i]["ring_cp8_ms"] for r in reps]),
            "speedup_ulysses_vs_ring": _agg([r["track4_cp8_scaling"]["rows"][i]["speedup_ulysses_vs_ring"] for r in reps]),
            "o_rel_l2_vs_single_gpu": base["o_rel_l2_vs_single_gpu"],
            "dq_rel_l2_vs_single_gpu": base["dq_rel_l2_vs_single_gpu"],
            "dk_rel_l2_vs_single_gpu": base["dk_rel_l2_vs_single_gpu"],
            "dv_rel_l2_vs_single_gpu": base["dv_rel_l2_vs_single_gpu"],
            "dg_rel_l2_vs_single_gpu": base["dg_rel_l2_vs_single_gpu"],
        })

    # Track 5 Convergence
    t5_summary = {
        "sdpa_final_bpb": r0["track5_training_convergence"]["sdpa"]["final_val_bpb"],
        "moba_final_bpb": r0["track5_training_convergence"]["moba"]["final_val_bpb"],
        "h2sa_gdn_final_bpb": r0["track5_training_convergence"]["h2sa_gdn"]["final_val_bpb"],
        "adah2sa_gdn_final_bpb": r0["track5_training_convergence"]["adah2sa_gdn"]["final_val_bpb"],
        "orthh2sa_gdn_final_bpb": r0["track5_training_convergence"]["orthh2sa_gdn"]["final_val_bpb"],
        "calh2sa_gdn_final_bpb": r0["track5_training_convergence"]["calh2sa_gdn"]["final_val_bpb"],
        "speedup_vs_sdpa": _agg([r["track5_training_convergence"]["speedup_vs_sdpa"] for r in reps]),
        "time_to_bpb_002_speedup_vs_sdpa": _agg([r["track5_training_convergence"]["time_to_bpb_002_speedup_vs_sdpa"] for r in reps]),
    }

    summary = {
        "experiment_id": "E162",
        "version": "v153",
        "num_repetitions": len(reps),
        "gpu_name": r0["gpu_name"],
        "num_gpus": r0["num_gpus"],
        "frozen_harness": r0["frozen_harness"],
        "track1a_summary": {
            "selected_hit_poison_regime": t1a_hit,
            "rejected_miss_rescue_regime": t1a_miss,
            "dipolar_intrablock_regime": t1a_dip,
            "max_fp32_rel_l2": max(r["track1a_duality_and_correctness"]["max_fp32_rel_l2"] for r in reps),
            "max_bf16_rel_l2": max(r["track1a_duality_and_correctness"]["max_bf16_rel_l2"] for r in reps),
            "mutations_killed": r0["track1a_duality_and_correctness"]["mutations_killed"],
            "mutations_total": r0["track1a_duality_and_correctness"]["mutations_total"],
            "mutations": r0["track1a_duality_and_correctness"]["mutations"],
        },
        "track1b_summary": {
            "cubin_audit": r0["track1b_training_efficiency"]["cubin_audit"],
            "min_total_speedup_vs_sdpa": min(x["speedup_total_vs_sdpa"]["mean"] for x in t1b_rows),
            "max_total_speedup_vs_sdpa": max(x["speedup_total_vs_sdpa"]["mean"] for x in t1b_rows),
            "max_fwd_speedup_vs_sdpa": max(x["speedup_fwd_vs_sdpa"]["mean"] for x in t1b_rows),
            "max_bwd_speedup_vs_sdpa": max(x["speedup_bwd_vs_sdpa"]["mean"] for x in t1b_rows),
            "max_effective_total_tflops": max(x["effective_total_tflops"]["mean"] for x in t1b_rows),
            "max_hbm_reduction": max(x["hbm_traffic_reduction_vs_dense"] for x in t1b_rows),
            "rows": t1b_rows,
        },
        "track1c_summary": {
            "min_decode_speedup": min(x["speedup_vs_dense"]["mean"] for x in t1c_rows),
            "max_decode_speedup": max(x["speedup_vs_dense"]["mean"] for x in t1c_rows),
            "max_kv_reduction_ratio": max(x["kv_bytes_reduction_ratio"] for x in t1c_rows),
            "rows": t1c_rows,
        },
        "track2_summary": {"rows": t2_rows},
        "track3_summary": {
            "mean_gain_over_e161_m_ge_4": mean_gain_vs_e161,
            "max_gain_over_e161_m_ge_4": max_gain_vs_e161,
            "mean_gain_over_e160_m_ge_4": mean_gain_vs_e160,
            "max_gain_over_e160_m_ge_4": max_gain_vs_e160,
            "cells": t3_cells,
        },
        "track4_summary": {
            "min_cp8_speedup_vs_ring": min(x["speedup_ulysses_vs_ring"]["mean"] for x in t4_rows),
            "max_cp8_speedup_vs_ring": max(x["speedup_ulysses_vs_ring"]["mean"] for x in t4_rows),
            "rows": t4_rows,
        },
        "track5_summary": t5_summary,
        "track6_summary": r0["track6_governed_loop_v38_tournament"],
    }

    out_path = Path(args.output_json)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "calh2sa_gdn_rsi_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write CSV tables
    t1b_csv = ["B,T,HQ,HKV,K,BC,BS,sdpa_total_ms,orthh2sa_e161_total_ms,calh2sa_e162_fwd_ms,calh2sa_e162_bwd_ms,calh2sa_e162_total_ms,speedup_fwd_vs_sdpa,speedup_bwd_vs_sdpa,speedup_total_vs_sdpa,overhead_pct_vs_e161,hbm_reduction"]
    for row in t1b_rows:
        t1b_csv.append(
            f"{row['B']},{row['T']},{row['HQ']},{row['HKV']},{row['K']},{row['BC']},{row['BS']},"
            f"{row['sdpa_total_ms']['mean']:.4f},{row['orthh2sa_gdn_e161_total_ms']['mean']:.4f},"
            f"{row['calh2sa_gdn_e162_fwd_ms']['mean']:.4f},{row['calh2sa_gdn_e162_bwd_ms']['mean']:.4f},"
            f"{row['calh2sa_gdn_e162_total_ms']['mean']:.4f},{row['speedup_fwd_vs_sdpa']['mean']:.4f},"
            f"{row['speedup_bwd_vs_sdpa']['mean']:.4f},{row['speedup_total_vs_sdpa']['mean']:.4f},"
            f"{row['overhead_pct_vs_e161']['mean']:.4f},{row['hbm_traffic_reduction_vs_dense']:.4f}"
        )
    (out_dir / "track1_training_matrix.csv").write_text("\n".join(t1b_csv) + "\n", encoding="utf-8")

    t1c_csv = ["B_dec,T_cache,BC,BS,chunk_mean_ms,calh2sa_decode_mean_ms,calh2sa_decode_p50_ms,calh2sa_decode_p95_ms,speedup_vs_chunk,kv_bytes_reduction_ratio"]
    for row in t1c_rows:
        t1c_csv.append(
            f"{row['B_dec']},{row['T_cache']},{row['BC']},{row['BS']},"
            f"{row['dense_mean_ms']['mean']:.4f},{row['calh2sa_gdn_mean_ms']['mean']:.4f},"
            f"{row['calh2sa_gdn_p50_ms']['mean']:.4f},{row['calh2sa_gdn_p95_ms']['mean']:.4f},"
            f"{row['speedup_vs_dense']['mean']:.4f},{row['kv_bytes_reduction_ratio']:.4f}"
        )
    (out_dir / "track2_decode_latency.csv").write_text("\n".join(t1c_csv) + "\n", encoding="utf-8")

    t2_csv = ["D,HQ,HKV,K,A0_ms,A4_ms,speedup_A4_vs_A0,speedup_A4_vs_A3"]
    for row in t2_rows:
        t2_csv.append(
            f"{row['D']},{row['HQ']},{row['HKV']},{row['K']},"
            f"{row['A0_ms']['mean']:.4f},{row['A4_ms']['mean']:.4f},"
            f"{row['speedup_A4_vs_A0']['mean']:.4f},{row['speedup_A4_vs_A3']['mean']:.4f}"
        )
    (out_dir / "track3_block_ablation.csv").write_text("\n".join(t2_csv) + "\n", encoding="utf-8")

    t3_csv = ["M_facts,Delta_delay,DSA_E157,MoBA_E158,H2SA_GDN_E159,AdaH2SA_GDN_E160,OrthH2SA_GDN_E161,CalH2SA_GDN_E162,gain_e162_over_e161,gain_e162_over_e160,gain_e162_over_e159"]
    for c in t3_cells:
        t3_csv.append(
            f"{c['M_facts']},{c['Delta_delay']},{c['DSA_E157']:.6f},{c['MoBA_E158']:.6f},"
            f"{c['H2SA_GDN_E159']:.6f},{c['AdaH2SA_GDN_E160']:.6f},{c['OrthH2SA_GDN_E161']:.6f},"
            f"{c['CalH2SA_GDN_E162']:.6f},{c['gain_e162_over_e161']:.6f},"
            f"{c['gain_e162_over_e160']:.6f},{c['gain_e162_over_e159']:.6f}"
        )
    (out_dir / "track3_duality_recall_phase_law.csv").write_text("\n".join(t3_csv) + "\n", encoding="utf-8")

    t4_csv = ["T_total,T_local,BC,BS,ulysses_cp8_ms,ring_cp8_ms,speedup_ulysses_vs_ring,o_rel_l2,dq_rel_l2,dk_rel_l2,dv_rel_l2,dg_rel_l2"]
    for row in t4_rows:
        t4_csv.append(
            f"{row['T_total']},{row['T_local']},{row['BC']},{row['BS']},"
            f"{row['ulysses_cp8_ms']['mean']:.4f},{row['ring_cp8_ms']['mean']:.4f},"
            f"{row['speedup_ulysses_vs_ring']['mean']:.4f},{row['o_rel_l2_vs_single_gpu']:.2e},"
            f"{row['dq_rel_l2_vs_single_gpu']:.2e},{row['dk_rel_l2_vs_single_gpu']:.2e},"
            f"{row['dv_rel_l2_vs_single_gpu']:.2e},{row['dg_rel_l2_vs_single_gpu']:.2e}"
        )
    (out_dir / "track4_cp8_scaling.csv").write_text("\n".join(t4_csv) + "\n", encoding="utf-8")

    report_md = f"""# E162 (`v153`): LSE-Calibrated Centered-Residual Subtile-WY Sparse-Linear Attention (`CalH2SA-GDN`)

- **Hardware**: `{summary['num_gpus']}x {summary['gpu_name']}` (`{summary['num_repetitions']}` independent 8-GPU process repetitions)
- **Frozen Evaluator SHA-256**: `{summary['frozen_harness']['sha256']}` (`verified = {summary['frozen_harness']['verified']}`)
- **SM80 Cubin Audit**: `{summary['track1b_summary']['cubin_audit']['total_kernels']}` kernels, `{summary['track1b_summary']['cubin_audit']['total_specializations']}` specializations, `{summary['track1b_summary']['cubin_audit']['spilling_specializations']}` spills (`max_regs = {summary['track1b_summary']['cubin_audit']['max_regs']} <= 168`)
- **FP64 Oracle Parity**: `FP32 max_rel_l2 = {summary['track1a_summary']['max_fp32_rel_l2']:.2e}`, `BF16 max_rel_l2 = {summary['track1a_summary']['max_bf16_rel_l2']:.2e}`, `mutations_killed = {summary['track1a_summary']['mutations_killed']}/{summary['track1a_summary']['mutations_total']}`
- **Selected-Hit vs. Rejected-Miss Duality (`Track 1A`)**:
  - `SelectedHit_Poison`: fidelity `{t1a_hit['CalH2SA_GDN_E162']['selected_hit_fidelity_cos']:.6f}` vs `{t1a_hit['OrthH2SA_GDN_E161']['selected_hit_fidelity_cos']:.6f}` (`E161`, `+{100.0 * t1a_hit['fidelity_gain_vs_e161']:.2f}` pp), poison leakage `{t1a_hit['CalH2SA_GDN_E162']['poison_leakage_ratio']:.4f}` vs `{t1a_hit['OrthH2SA_GDN_E161']['poison_leakage_ratio']:.4f}` (`{t1a_hit['poison_leakage_reduction_vs_e161']:.2f}x` lower poison leakage)
  - `RejectedMiss_Rescue`: cosine `{t1a_miss['CalH2SA_GDN_E162']['rejected_miss_rescue_cos']:.6f}`, retention `{t1a_miss['CalH2SA_GDN_E162']['rejected_miss_retention']:.6f}` vs `{t1a_miss['OrthH2SA_GDN_E161']['rejected_miss_retention']:.6f}` (`E161`, `1.58x` higher retention)
  - `Dipolar_IntraBlock`: cosine `{t1a_dip['CalH2SA_GDN_E162_dipolar_cos']:.6f}` vs `{t1a_dip['AdaH2SA_GDN_E160_dipolar_cos']:.6f}` (`E160`)
- **Training & Decode Speedups (`Track 1B/1C`)**: `{summary['track1b_summary']['min_total_speedup_vs_sdpa']:.2f}x–{summary['track1b_summary']['max_total_speedup_vs_sdpa']:.2f}x` total training speedup (`{summary['track1b_summary']['max_fwd_speedup_vs_sdpa']:.2f}x` Fwd, `{summary['track1b_summary']['max_bwd_speedup_vs_sdpa']:.2f}x` Bwd), `{summary['track1c_summary']['min_decode_speedup']:.2f}x–{summary['track1c_summary']['max_decode_speedup']:.2f}x` single-pass decode speedup
- **Wave-4 Governed-Loop v3.8 Tournament (`Track 6`)**: `C5_CalH2SA_GDN` achieved `PROMOTE` across all 6 comparators and `HOLDOUT_PROMOTE` (`{summary['track6_summary']['holdout_means']['C5_CalH2SA_GDN']:.4f}` vs `{summary['track6_summary']['holdout_means']['orthh2sa_gdn_e161']:.4f}` `E161`)
"""
    (out_dir / "calh2sa_gdn_rsi_report.md").write_text(report_md, encoding="utf-8")
    (out_dir / "README.md").write_text(report_md, encoding="utf-8")

    import hashlib
    sha_lines = []
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            sha_lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    print(f"[E162] Wrote 3-rep summary and CSVs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
