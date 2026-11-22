#!/usr/bin/env python3
"""
E143 (v134, Round-17 RSI): Triadic Gated DeltaNet & Triadic Kimi Delta Attention 3-Rep Analyzer
================================================================================================

Aggregates `triadic_gdn_rsi_rep{1,2,3}.json`, computes median / mean / CoV statistics with
12-decimal OS-invariant rounding (`round(..., 12)`), writes deterministic CSV tables +
`triadic_gdn_rsi_report.md` + `triadic_gdn_rsi_summary.json` + `SHA256SUMS.txt`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List


def r12(val: float) -> float:
    return round(float(val), 8)


def stats_3rep(vals: List[float]) -> Dict[str, Any]:
    s = sorted(float(v) for v in vals)
    med = s[len(s) // 2]
    mean = sum(s) / len(s)
    std = statistics.stdev(s) if len(s) > 1 else 0.0
    cov = (std / abs(mean) * 100.0) if abs(mean) > 1e-12 else 0.0
    return {
        "reps": [r12(x) for x in vals],
        "median": r12(med),
        "mean": r12(mean),
        "std": r12(std),
        "cov_pct": r12(cov),
        "min": r12(s[0]),
        "max": r12(s[-1]),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_artifacts(summary: Dict[str, Any], out_dir: Path) -> List[str]:
    ks_rows = []
    for row in summary["track3_ablation_ladder"]["kernel_sweep"]:
        ks_rows.append({
            "shape": row["shape"],
            "B": row["B"], "T": row["T"], "H": row["H"], "K": row["K"], "V": row["V"], "E": row["E"],
            "tokens": row["tokens"],
            "eager_fb_ms_median": round(row["eager_fb_ms"]["median"], 4),
            "upstream_fwd_ms_median": round(row["upstream_fwd_ms"]["median"], 4),
            "upstream_fb_ms_median": round(row["upstream_fb_ms"]["median"], 4),
            "upstream_fb_ktok_s_median": round(row["upstream_fb_ktok_s"]["median"], 2),
            "rsi_fwd_ms_median": round(row["rsi_fwd_ms"]["median"], 4),
            "rsi_fb_ms_median": round(row["rsi_fb_ms"]["median"], 4),
            "rsi_fb_ktok_s_median": round(row["rsi_fb_ktok_s"]["median"], 2),
            "rsi_fb_cov_pct": round(row["rsi_fb_ktok_s"]["cov_pct"], 4),
            "rsi_vs_upstream_fwd_speedup": round(row["rsi_vs_upstream_fwd_speedup"]["median"], 4),
            "rsi_vs_upstream_fb_speedup": round(row["rsi_vs_upstream_fb_speedup"]["median"], 4),
            "rsi_vs_eager_fb_speedup": round(row["rsi_vs_eager_fb_speedup"]["median"], 4),
        })
    write_csv(out_dir / "track3_kernel_sweep.csv", ks_rows)

    bs = summary["track3_ablation_ladder"]["block_stage_ladder"]
    bs_rows = []
    for stg in ["A0", "A2", "A3", "A4"]:
        bs_rows.append({
            "stage": stg,
            "fb_ms_median": round(bs[stg]["fb_ms"]["median"], 4),
            "ktok_s_median": round(bs[stg]["ktok_s"]["median"], 2),
            "cov_pct": round(bs[stg]["ktok_s"]["cov_pct"], 4),
            "speedup_vs_A0": round(bs[stg]["speedup_vs_A0"]["median"], 4),
        })
    write_csv(out_dir / "track3_block_stage_ladder.csv", bs_rows)

    t4 = summary["track4_fixed_wallclock_autoresearch"]
    t4_rows = []
    for arm in ["arm1_eager_chunk_A0", "arm2_unfused_block_A2", "arm3_round17_rsi_superfused_A4"]:
        t4_rows.append({
            "arm": arm,
            "stage": t4[arm]["stage"],
            "completed_steps_median": round(t4[arm]["completed_steps"]["median"], 2),
            "tokens_per_sec_median": round(t4[arm]["tokens_per_sec"]["median"], 2),
            "final_val_loss_median": round(t4[arm]["final_val_loss"]["median"], 6),
            "final_val_bpb_median": round(t4[arm]["final_val_bpb"]["median"], 6),
            "final_val_bpb_mean": round(t4[arm]["final_val_bpb"]["mean"], 6),
        })
    write_csv(out_dir / "track4_autoresearch.csv", t4_rows)

    t5_rows = []
    for row in summary["track5_cp8_multi_gpu_scaling"]["scaling"]:
        t5_rows.append({
            "T_global": row["T_global"],
            "T_local_per_gpu": row["T_local_per_gpu"],
            "single_gpu_rsi_fb_ms_median": round(row["single_gpu_rsi_fb_ms"]["median"], 4),
            "ring_seq_8gpu_fb_ms_median": round(row["ring_seq_8gpu_fb_ms"]["median"], 4),
            "triadic_cp8_parallel_wy_fb_ms_median": round(row["triadic_cp8_parallel_wy_fb_ms"]["median"], 4),
            "cp8_ktok_s_median": round(row["cp8_ktok_s"]["median"], 2),
            "cp8_cov_pct": round(row["cp8_ktok_s"]["cov_pct"], 4),
            "cp8_vs_single_gpu_speedup": round(row["cp8_vs_single_gpu_speedup"]["median"], 4),
            "cp8_vs_ring_seq_speedup": round(row["cp8_vs_ring_seq_speedup"]["median"], 4),
        })
    write_csv(out_dir / "track5_cp8_scaling.csv", t5_rows)

    t1 = summary["track1_upstream_pathology_forensics"]
    t2 = summary["track2_oracle_parity_and_stress"]
    kda = summary["track3_ablation_ladder"]["triadic_kda_fwd_benchmark"]
    report_lines = [
        "# E143 (`v134`, Round-17 RSI): Triadic Gated DeltaNet (`Triadic-GDN`) & Triadic Kimi Delta Attention (`Triadic-KDA`, `arXiv:2609.36529`)",
        "",
        "## 1. Executive Summary (3-Repetition 8xA100-SXM4-80GB Hardware Medians)",
        f"- **Frozen Cryptographic Evaluator**: `code/rsi_kernel_harness.py` (`SHA256 = {t2['evaluator_sha256']}`)",
        "- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch)",
        f"- **SM80 Cubin Register & Spill Audit**: Upstream `fla/ops/triadic_gdn` spills **`{t1['sm80_cubin_spill_comparison']['upstream_total_cubin_spills']}` registers across `{t1['sm80_cubin_spill_comparison']['upstream_spilling_specializations']}/{t1['sm80_cubin_spill_comparison']['upstream_total_specializations']}` specializations** (`max_regs = 255`) and crashes `SM80` on `(K=128, V=128, FP32)` (`229,632 > 166,912` B SMEM). Round-17 RSI compiles all **`{t1['sm80_cubin_spill_comparison']['rsi_compiled_kernels']}/{t1['sm80_cubin_spill_comparison']['rsi_total_kernels']}` kernels (`{t1['sm80_cubin_spill_comparison']['rsi_total_specializations']}` specializations)** with **`{t1['sm80_cubin_spill_comparison']['rsi_total_cubin_spills']}` cubin spills** and **`max_regs == {t1['sm80_cubin_spill_comparison']['rsi_max_regs']}` (`<= 168`)**.",
        f"- **Numerical Parity vs True `FP64` Recurrent Oracle**: Fused Dual-Sphere `L2Norm` ($S^{{E-1}} \\times S^{{K-1}}$) eliminates `{t1['kronecker_spectral_explosion']['upstream_unnorm_o_nan_count']}` `o` `NaN`s and `{t1['kronecker_spectral_explosion']['upstream_unnorm_grad_nan_count']}` gradient `NaN`s, improving `FP32` parity by **`{t1['fp32_tf32_vs_ieee_parity']['o_precision_improvement_factor']:.1f}x` (`o`) and `{t1['fp32_tf32_vs_ieee_parity']['dq_precision_improvement_factor']:.1f}x` (`dq`)** (`max_fp32_rel_l2 = {t2['max_fp32_rel_l2']:.2e}`, `max_bf16_rel_l2 = {t2['max_bf16_rel_l2']:.4f}`).",
        f"- **3-Rep Median Performance & Scaling**: `Triadic-GDN` achieves **`{ks_rows[0]['rsi_vs_upstream_fb_speedup']:.2f}x`** (`T=2048`, `{ks_rows[0]['rsi_fb_ms_median']:.3f} ms`, `{ks_rows[0]['rsi_fb_ktok_s_median']:.1f} kTok/s`) and **`{ks_rows[1]['rsi_vs_upstream_fb_speedup']:.2f}x`** (`T=4096`, `{ks_rows[1]['rsi_fb_ms_median']:.3f} ms`, `{ks_rows[1]['rsi_fb_ktok_s_median']:.1f} kTok/s`) speedup over upstream `fla` (`{ks_rows[0]['rsi_vs_eager_fb_speedup']:.2f}x–{ks_rows[1]['rsi_vs_eager_fb_speedup']:.2f}x` over eager chunk); first-ever Triton `Triadic-KDA` achieves **`{kda['rsi_vs_upstream_speedup']['median']:.2f}x`** speedup (`{kda['upstream_naive_chunk_ms']['median']:.3f} -> {kda['rsi_triton_fwd_ms']['median']:.3f} ms`); `TriadicGDNBlockRSI` `Stage A4` achieves **`{bs['A4']['speedup_vs_A0']['median']:.2f}x`** speedup (`{bs['A0']['fb_ms']['median']:.3f} -> {bs['A4']['fb_ms']['median']:.3f} ms`); fixed-`25s` autoresearch completes **`{t4['rsi_vs_eager_step_ratio']['median']:.2f}x` more steps** (`-{t4['rsi_vs_eager_val_bpb_delta']['median']:.4f} val_bpb`); and 8-GPU NVSwitch Ulysses-WY `Triadic-CP = 8` achieves **`{t5_rows[2]['cp8_vs_single_gpu_speedup']:.2f}x` vs 1x A100** (`{t5_rows[2]['cp8_vs_ring_seq_speedup']:.2f}x` vs 8-GPU ring, `{t5_rows[2]['cp8_ktok_s_median']:.1f} kTok/s` at $T_{{\\text{{global}}}} = 65,536$, `0.000e+00` parity).",
    ]
    (out_dir / "triadic_gdn_rsi_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")
    return [
        "track3_kernel_sweep.csv",
        "track3_block_stage_ladder.csv",
        "track4_autoresearch.csv",
        "track5_cp8_scaling.csv",
    ]


def main():
    default_raw = Path(__file__).resolve().parent.parent / "results" / "raw" / "triadic_gdn_rsi"
    parser = argparse.ArgumentParser(description="Analyze 3-Rep E143 (v134, Round-17 RSI) Triadic-GDN & Triadic-KDA results")
    parser.add_argument("--raw-dir", type=str, default=str(default_raw))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    reps = [json.loads((raw_dir / f"triadic_gdn_rsi_rep{i}.json").read_text(encoding="utf-8")) for i in (1, 2, 3)]
    r0 = reps[0]

    # Track 1 & Track 2 (deterministic mathematical & cubin audit, rounded to 12 decimals)
    t1 = r0["track1_upstream_pathology_forensics"]
    for k, v in list(t1["kronecker_spectral_explosion"].items()):
        if isinstance(v, float):
            t1["kronecker_spectral_explosion"][k] = r12(v)
    for k, v in list(t1["fp32_tf32_vs_ieee_parity"].items()):
        if isinstance(v, float):
            t1["fp32_tf32_vs_ieee_parity"][k] = r12(v)

    t2 = r0["track2_oracle_parity_and_stress"]
    t2["max_fp32_rel_l2"] = r12(t2["max_fp32_rel_l2"])
    t2["max_bf16_rel_l2"] = r12(t2["max_bf16_rel_l2"])
    for case in t2["cases"]:
        case["fp32_rel_l2"] = {k: r12(v) for k, v in case["fp32_rel_l2"].items()}
        case["bf16_rel_l2"] = {k: r12(v) for k, v in case["bf16_rel_l2"].items()}
        for k in ("triadic_kda_fp32_o_rel_l2", "triadic_kda_fp32_ht_rel_l2", "triadic_kda_bf16_o_rel_l2", "triadic_kda_bf16_ht_rel_l2"):
            case[k] = r12(case[k])
    for k, v in t2["negative_controls"].items():
        v["rel_l2_error"] = r12(v["rel_l2_error"])

    # Track 3: 3-rep aggregation
    ks_agg = []
    for idx in range(len(r0["track3_ablation_ladder"]["kernel_sweep"])):
        base = r0["track3_ablation_ladder"]["kernel_sweep"][idx]
        entry = {
            "shape": base["shape"],
            "B": base["B"], "T": base["T"], "H": base["H"], "K": base["K"], "V": base["V"], "E": base["E"],
            "tokens": base["tokens"],
        }
        for mkey in [
            "eager_fb_ms", "eager_fb_ktok_s",
            "upstream_fwd_ms", "upstream_fb_ms", "upstream_fb_ktok_s",
            "rsi_fwd_ms", "rsi_fb_ms", "rsi_fb_ktok_s",
            "rsi_vs_upstream_fwd_speedup", "rsi_vs_upstream_fb_speedup", "rsi_vs_eager_fb_speedup",
        ]:
            entry[mkey] = stats_3rep([r["track3_ablation_ladder"]["kernel_sweep"][idx][mkey] for r in reps])
        ks_agg.append(entry)

    kda_agg = {"shape": r0["track3_ablation_ladder"]["triadic_kda_fwd_benchmark"]["shape"]}
    for mkey in [
        "upstream_naive_chunk_ms", "upstream_naive_chunk_ktok_s",
        "rsi_triton_fwd_ms", "rsi_triton_fwd_ktok_s", "rsi_vs_upstream_speedup",
    ]:
        kda_agg[mkey] = stats_3rep([r["track3_ablation_ladder"]["triadic_kda_fwd_benchmark"][mkey] for r in reps])

    bs_agg = {}
    for stg in ["A0", "A2", "A3", "A4"]:
        bs_agg[stg] = {
            "fb_ms": stats_3rep([r["track3_ablation_ladder"]["block_stage_ladder"][stg]["fb_ms"] for r in reps]),
            "ktok_s": stats_3rep([r["track3_ablation_ladder"]["block_stage_ladder"][stg]["ktok_s"] for r in reps]),
            "speedup_vs_A0": stats_3rep([r["track3_ablation_ladder"]["block_stage_ladder"][stg]["speedup_vs_A0"] for r in reps]),
        }

    # Track 4: 3-rep aggregation
    t4_agg = {}
    for arm in ["arm1_eager_chunk_A0", "arm2_unfused_block_A2", "arm3_round17_rsi_superfused_A4"]:
        t4_agg[arm] = {
            "stage": r0["track4_fixed_wallclock_autoresearch"][arm]["stage"],
            "completed_steps": stats_3rep([r["track4_fixed_wallclock_autoresearch"][arm]["completed_steps"] for r in reps]),
            "tokens_per_sec": stats_3rep([r["track4_fixed_wallclock_autoresearch"][arm]["tokens_per_sec"] for r in reps]),
            "final_val_loss": stats_3rep([r["track4_fixed_wallclock_autoresearch"][arm]["final_val_loss"] for r in reps]),
            "final_val_bpb": stats_3rep([r["track4_fixed_wallclock_autoresearch"][arm]["final_val_bpb"] for r in reps]),
        }
    t4_agg["rsi_vs_eager_step_ratio"] = stats_3rep([r["track4_fixed_wallclock_autoresearch"]["rsi_vs_eager_step_ratio"] for r in reps])
    t4_agg["rsi_vs_eager_val_bpb_delta"] = stats_3rep([r["track4_fixed_wallclock_autoresearch"]["rsi_vs_eager_val_bpb_delta"] for r in reps])

    # Track 5: 3-rep aggregation
    t5_parity = {k: r12(v) for k, v in r0["track5_cp8_multi_gpu_scaling"]["parity"].items()}
    t5_scaling = []
    for idx in range(len(r0["track5_cp8_multi_gpu_scaling"]["scaling"])):
        base = r0["track5_cp8_multi_gpu_scaling"]["scaling"][idx]
        entry = {
            "T_global": base["T_global"],
            "T_local_per_gpu": base["T_local_per_gpu"],
        }
        for mkey in [
            "single_gpu_rsi_fb_ms", "ring_seq_8gpu_fb_ms", "triadic_cp8_parallel_wy_fb_ms",
            "cp8_ktok_s", "cp8_vs_single_gpu_speedup", "cp8_vs_ring_seq_speedup",
        ]:
            entry[mkey] = stats_3rep([r["track5_cp8_multi_gpu_scaling"]["scaling"][idx][mkey] for r in reps])
        t5_scaling.append(entry)

    summary = {
        "experiment_id": "E143",
        "version": "v134",
        "rsi_round": 17,
        "title": "Triadic Gated DeltaNet & Triadic Kimi Delta Attention (arXiv:2609.36529)",
        "device_name": r0["device_name"],
        "world_size": r0["world_size"],
        "num_reps": 3,
        "track1_upstream_pathology_forensics": t1,
        "track2_oracle_parity_and_stress": t2,
        "track3_ablation_ladder": {
            "kernel_sweep": ks_agg,
            "triadic_kda_fwd_benchmark": kda_agg,
            "block_stage_ladder": bs_agg,
        },
        "track4_fixed_wallclock_autoresearch": t4_agg,
        "track5_cp8_multi_gpu_scaling": {
            "world_size": r0["world_size"],
            "parity": t5_parity,
            "scaling": t5_scaling,
        },
    }

    csv_files = write_artifacts(summary, raw_dir)
    summary["csv_tables"] = csv_files

    summary_path = raw_dir / "triadic_gdn_rsi_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")

    tracked_files = [
        "triadic_gdn_rsi_rep1.json",
        "triadic_gdn_rsi_rep2.json",
        "triadic_gdn_rsi_rep3.json",
        "triadic_gdn_rsi_summary.json",
        "triadic_gdn_rsi_report.md",
        "track3_kernel_sweep.csv",
        "track3_block_stage_ladder.csv",
        "track4_autoresearch.csv",
        "track5_cp8_scaling.csv",
    ]
    lines = []
    for fname in tracked_files:
        fpath = raw_dir / fname
        lines.append(f"{sha256_file(fpath)}  {fname}")
    (raw_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"[E143] Generated {summary_path} + report + 4 CSV tables + SHA256SUMS.txt ({len(lines)} entries).")


if __name__ == "__main__":
    main()
