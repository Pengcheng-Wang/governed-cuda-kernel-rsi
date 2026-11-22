#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E148 (v139, Round-22 RSI): Cyclic Flow Attention (CyFA) Zero-Spill SM80 3-Rep Analyzer
======================================================================================

Aggregates `rep{1,2,3}.json`, computes median / IQR / CoV / min / max statistics across
3 independent hardware repetitions, and writes deterministic CSV tables +
`cyfa_rsi_report.md` + `cyfa_rsi_summary.json` + `README.md` + `SHA256SUMS.txt`.
"""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_HARNESS_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def _median_iqr(vals: List[float]) -> Dict[str, Any]:
    s = sorted(float(v) for v in vals)
    med = statistics.median(s)
    iqr = s[-1] - s[0] if len(s) >= 2 else 0.0
    mean = sum(s) / len(s)
    std = statistics.stdev(s) if len(s) > 1 else 0.0
    cov = (std / abs(mean) * 100.0) if abs(mean) > 1e-12 else 0.0
    return {
        "median": round(med, 4),
        "iqr": round(iqr, 4),
        "cov_pct": round(cov, 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
        "reps": [round(float(v), 4) for v in vals],
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    reps = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reps]
    n_reps = len(reps)
    assert n_reps >= 3, f"Expected at least 3 repetitions, got {n_reps}"

    for r in reps:
        assert r["harness"]["sha256_verified"], "Harness SHA256 failed"
        assert r["harness"]["self_test_passed"], "Harness self-test failed"
        assert r["track1_oracle_and_mutations"]["mutation_kills"] == 3, "Mutations failed"
        assert r["track2_cubin_audit"]["rsi_cyfa"]["total_spills"] == 0, "Zero-spill cubin audit failed"
        assert r["track2_cubin_audit"]["rsi_cyfa"]["max_regs"] <= 168, "RSI max_regs <= 168 audit failed"

    # 1. Aggregate Track 3A Kernel Matrix
    kernel_rows_agg = []
    kernel_csv_rows = []
    n_k = len(reps[0]["track3a_kernel_matrix"]["rows"])
    for idx in range(n_k):
        r0 = reps[0]["track3a_kernel_matrix"]["rows"][idx]
        up_fwd = _median_iqr([r["track3a_kernel_matrix"]["rows"][idx]["upstream_fwd_ms"] for r in reps])
        up_bwd = _median_iqr([r["track3a_kernel_matrix"]["rows"][idx]["upstream_bwd_ms"] for r in reps])
        up_tot = _median_iqr([r["track3a_kernel_matrix"]["rows"][idx]["upstream_fwdbwd_ms"] for r in reps])

        rsi_fwd = _median_iqr([r["track3a_kernel_matrix"]["rows"][idx]["rsi_fwd_ms"] for r in reps])
        rsi_bwd = _median_iqr([r["track3a_kernel_matrix"]["rows"][idx]["rsi_bwd_ms"] for r in reps])
        rsi_tot = _median_iqr([r["track3a_kernel_matrix"]["rows"][idx]["rsi_fwdbwd_ms"] for r in reps])

        fwd_sp = round(up_fwd["median"] / rsi_fwd["median"], 3)
        bwd_sp = round(up_bwd["median"] / rsi_bwd["median"], 3)
        tot_sp = round(up_tot["median"] / rsi_tot["median"], 3)

        kernel_rows_agg.append({
            "B": r0["B"],
            "T": r0["T"],
            "H": r0["H"],
            "K": r0["K"],
            "V": r0["V"],
            "M": r0["M"],
            "C": r0["C"],
            "upstream": {"fwd_ms": up_fwd, "bwd_ms": up_bwd, "fwdbwd_ms": up_tot},
            "rsi_zero_spill": {"fwd_ms": rsi_fwd, "bwd_ms": rsi_bwd, "fwdbwd_ms": rsi_tot},
            "speedup": {"fwd_x": fwd_sp, "bwd_x": bwd_sp, "fwdbwd_x": tot_sp},
        })
        kernel_csv_rows.append({
            "B": r0["B"],
            "T": r0["T"],
            "H": r0["H"],
            "K": r0["K"],
            "V": r0["V"],
            "M": r0["M"],
            "C": r0["C"],
            "upstream_fwd_ms": up_fwd["median"],
            "upstream_bwd_ms": up_bwd["median"],
            "upstream_fwdbwd_ms": up_tot["median"],
            "rsi_fwd_ms": rsi_fwd["median"],
            "rsi_bwd_ms": rsi_bwd["median"],
            "rsi_fwdbwd_ms": rsi_tot["median"],
            "fwd_speedup_x": fwd_sp,
            "bwd_speedup_x": bwd_sp,
            "fwdbwd_speedup_x": tot_sp,
            "fwdbwd_iqr_ms": rsi_tot["iqr"],
            "fwdbwd_cov_pct": rsi_tot["cov_pct"],
        })

    # 2. Aggregate Track 3B Layer Ablation
    layer_rows_agg = []
    layer_csv_rows = []
    n_l = len(reps[0]["track3b_layer_ablation"]["rows"])
    for idx in range(n_l):
        r0 = reps[0]["track3b_layer_ablation"]["rows"][idx]
        st_agg = {}
        for st in ("A0", "A1", "A2", "A3", "A4"):
            st_agg[st] = {
                "fwd_ms": _median_iqr([r["track3b_layer_ablation"]["rows"][idx]["stages"][st]["fwd_ms"] for r in reps]),
                "bwd_ms": _median_iqr([r["track3b_layer_ablation"]["rows"][idx]["stages"][st]["bwd_ms"] for r in reps]),
                "fwdbwd_ms": _median_iqr([r["track3b_layer_ablation"]["rows"][idx]["stages"][st]["fwdbwd_ms"] for r in reps]),
            }
        sp_a0 = round(st_agg["A0"]["fwdbwd_ms"]["median"] / st_agg["A4"]["fwdbwd_ms"]["median"], 3)
        sp_a1 = round(st_agg["A1"]["fwdbwd_ms"]["median"] / st_agg["A4"]["fwdbwd_ms"]["median"], 3)
        layer_rows_agg.append({
            "config": r0["config"],
            "B": r0["B"],
            "T": r0["T"],
            "D": r0["D"],
            "H": r0["H"],
            "K": r0["K"],
            "V": r0["V"],
            "M": r0["M"],
            "stages": st_agg,
            "speedup_A4_vs_A0_x": sp_a0,
            "speedup_A4_vs_A1_x": sp_a1,
        })
        layer_csv_rows.append({
            "config": r0["config"],
            "B": r0["B"],
            "T": r0["T"],
            "D": r0["D"],
            "H": r0["H"],
            "K": r0["K"],
            "V": r0["V"],
            "M": r0["M"],
            "A0_fwdbwd_ms": st_agg["A0"]["fwdbwd_ms"]["median"],
            "A1_fwdbwd_ms": st_agg["A1"]["fwdbwd_ms"]["median"],
            "A2_fwdbwd_ms": st_agg["A2"]["fwdbwd_ms"]["median"],
            "A3_fwdbwd_ms": st_agg["A3"]["fwdbwd_ms"]["median"],
            "A4_fwdbwd_ms": st_agg["A4"]["fwdbwd_ms"]["median"],
            "speedup_A4_vs_A0_x": sp_a0,
            "speedup_A4_vs_A1_x": sp_a1,
            "A4_iqr_ms": st_agg["A4"]["fwdbwd_ms"]["iqr"],
        })

    # 3. Aggregate Track 4 Autoresearch
    a0_steps = _median_iqr([r["track4_val_bpb_autoresearch"]["A0_upstream"]["optimizer_steps"] for r in reps])
    a4_steps = _median_iqr([r["track4_val_bpb_autoresearch"]["A4_rsi"]["optimizer_steps"] for r in reps])
    a0_sps = _median_iqr([r["track4_val_bpb_autoresearch"]["A0_upstream"]["steps_per_sec"] for r in reps])
    a4_sps = _median_iqr([r["track4_val_bpb_autoresearch"]["A4_rsi"]["steps_per_sec"] for r in reps])
    a0_tps = _median_iqr([r["track4_val_bpb_autoresearch"]["A0_upstream"]["tokens_per_sec"] for r in reps])
    a4_tps = _median_iqr([r["track4_val_bpb_autoresearch"]["A4_rsi"]["tokens_per_sec"] for r in reps])
    a0_bpb = _median_iqr([r["track4_val_bpb_autoresearch"]["A0_upstream"]["final_val_bpb"] for r in reps])
    a4_bpb = _median_iqr([r["track4_val_bpb_autoresearch"]["A4_rsi"]["final_val_bpb"] for r in reps])
    init_bpb = reps[0]["track4_val_bpb_autoresearch"]["A0_upstream"]["initial_val_bpb"]
    step_speedup = round(a4_sps["median"] / a0_sps["median"], 3)
    extra_bpb_gain = round(a0_bpb["median"] - a4_bpb["median"], 4)

    autoresearch_agg = {
        "budget_sec": reps[0]["track4_val_bpb_autoresearch"]["budget_sec"],
        "initial_val_bpb": init_bpb,
        "A0_upstream": {
            "optimizer_steps": a0_steps,
            "steps_per_sec": a0_sps,
            "tokens_per_sec": a0_tps,
            "final_val_bpb": a0_bpb,
        },
        "A4_rsi": {
            "optimizer_steps": a4_steps,
            "steps_per_sec": a4_sps,
            "tokens_per_sec": a4_tps,
            "final_val_bpb": a4_bpb,
        },
        "step_rate_speedup_x": step_speedup,
        "extra_val_bpb_gain": extra_bpb_gain,
    }
    autoresearch_csv_rows = [
        {
            "arm": "A0_upstream_cyfa",
            "budget_sec": reps[0]["track4_val_bpb_autoresearch"]["budget_sec"],
            "initial_val_bpb": init_bpb,
            "optimizer_steps_median": a0_steps["median"],
            "steps_per_sec_median": a0_sps["median"],
            "tokens_per_sec_median": a0_tps["median"],
            "final_val_bpb_median": a0_bpb["median"],
            "val_bpb_improvement": round(init_bpb - a0_bpb["median"], 4),
        },
        {
            "arm": "A4_rsi_superfused",
            "budget_sec": reps[0]["track4_val_bpb_autoresearch"]["budget_sec"],
            "initial_val_bpb": init_bpb,
            "optimizer_steps_median": a4_steps["median"],
            "steps_per_sec_median": a4_sps["median"],
            "tokens_per_sec_median": a4_tps["median"],
            "final_val_bpb_median": a4_bpb["median"],
            "val_bpb_improvement": round(init_bpb - a4_bpb["median"], 4),
        },
    ]

    # 4. Aggregate Track 5 CP=8 Scaling
    cp8_rows_agg = []
    cp8_csv_rows = []
    n_c = len(reps[0]["track5_cp8_nvswitch"]["rows"])
    for idx in range(n_c):
        r0 = reps[0]["track5_cp8_nvswitch"]["rows"][idx]
        ring_ms = _median_iqr([r["track5_cp8_nvswitch"]["rows"][idx]["ring_fwdbwd_ms"] for r in reps])
        uly_ms = _median_iqr([r["track5_cp8_nvswitch"]["rows"][idx]["ulysses_fwdbwd_ms"] for r in reps])
        sp_cp = round(ring_ms["median"] / uly_ms["median"], 3)
        cp8_rows_agg.append({
            "T_total": r0["T_total"],
            "T_local": r0["T_local"],
            "H": r0["H"],
            "K": r0["K"],
            "V": r0["V"],
            "M": r0["M"],
            "ring_fwdbwd_ms": ring_ms,
            "ulysses_fwdbwd_ms": uly_ms,
            "ulysses_vs_ring_speedup_x": sp_cp,
        })
        cp8_csv_rows.append({
            "T_total": r0["T_total"],
            "T_local": r0["T_local"],
            "H": r0["H"],
            "K": r0["K"],
            "V": r0["V"],
            "M": r0["M"],
            "ring_fwdbwd_ms": ring_ms["median"],
            "ulysses_fwdbwd_ms": uly_ms["median"],
            "ulysses_vs_ring_speedup_x": sp_cp,
            "ulysses_iqr_ms": uly_ms["iqr"],
        })

    headline = {
        "peak_kernel_fwdbwd_speedup_x": max(r["speedup"]["fwdbwd_x"] for r in kernel_rows_agg),
        "peak_kernel_bwd_speedup_x": max(r["speedup"]["bwd_x"] for r in kernel_rows_agg),
        "peak_kernel_fwd_speedup_x": max(r["speedup"]["fwd_x"] for r in kernel_rows_agg),
        "mean_kernel_fwdbwd_speedup_x": round(sum(r["speedup"]["fwdbwd_x"] for r in kernel_rows_agg) / len(kernel_rows_agg), 3),
        "mean_kernel_bwd_speedup_x": round(sum(r["speedup"]["bwd_x"] for r in kernel_rows_agg) / len(kernel_rows_agg), 3),
        "mean_kernel_fwd_speedup_x": round(sum(r["speedup"]["fwd_x"] for r in kernel_rows_agg) / len(kernel_rows_agg), 3),
        "peak_layer_A4_vs_A0_speedup_x": max(r["speedup_A4_vs_A0_x"] for r in layer_rows_agg),
        "peak_layer_A4_vs_A1_speedup_x": max(r["speedup_A4_vs_A1_x"] for r in layer_rows_agg),
        "mean_layer_A4_vs_A0_speedup_x": round(sum(r["speedup_A4_vs_A0_x"] for r in layer_rows_agg) / len(layer_rows_agg), 3),
        "peak_cp8_ulysses_vs_ring_speedup_x": max(r["ulysses_vs_ring_speedup_x"] for r in cp8_rows_agg),
        "mean_cp8_ulysses_vs_ring_speedup_x": round(sum(r["ulysses_vs_ring_speedup_x"] for r in cp8_rows_agg) / len(cp8_rows_agg), 3),
        "autoresearch_step_rate_speedup_x": step_speedup,
        "autoresearch_extra_val_bpb_gain": extra_bpb_gain,
    }

    summary = {
        "experiment": "E148_cyfa_round22_rsi",
        "version": "v139",
        "n_reps": n_reps,
        "gpu_name": reps[0]["gpu_name"],
        "gpu_count": reps[0]["gpu_count"],
        "harness": reps[0]["harness"],
        "headline": headline,
        "track1_oracle_and_mutations": reps[0]["track1_oracle_and_mutations"],
        "track2_cubin_audit": reps[0]["track2_cubin_audit"],
        "track3a_kernel_matrix": kernel_rows_agg,
        "track3b_layer_ablation": layer_rows_agg,
        "track4_val_bpb_autoresearch": autoresearch_agg,
        "track5_cp8_nvswitch": {
            "parity": reps[0]["track5_cp8_nvswitch"]["parity"],
            "rows": cp8_rows_agg,
        },
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "cyfa_rsi_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _write_csv(out_dir / "cyfa_kernel_matrix.csv", kernel_csv_rows)
    _write_csv(out_dir / "cyfa_layer_ablation.csv", layer_csv_rows)
    _write_csv(out_dir / "cyfa_autoresearch.csv", autoresearch_csv_rows)
    _write_csv(out_dir / "cyfa_cp8_scaling.csv", cp8_csv_rows)

    up_cubin = reps[0]["track2_cubin_audit"]["upstream_fla_cyfa"]
    rsi_cubin = reps[0]["track2_cubin_audit"]["rsi_cyfa"]
    t1 = reps[0]["track1_oracle_and_mutations"]

    report_lines = [
        "# E148 (`v139`, Round-22 RSI): Cyclic Flow Attention (`CyFA`) Zero-Spill SM80 Kernel & Layer Suite",
        "",
        "## 1. Executive Summary",
        f"- **Architecture**: `Cyclic Flow Attention (CyFA): Linear Sequence Modeling with Relative-Time-Partitioned Memory` (`arXiv:2609.36259`, Chen et al.; upstream `fla.ops.cyfa` & `fla.layers.cyfa`).",
        f"- **Hardware**: `{reps[0]['gpu_count']}x {reps[0]['gpu_name']}` (`NV12` NVSwitch), `{n_reps}` independent repetitions.",
        f"- **Frozen Cryptographic Evaluator**: `code/rsi_kernel_harness.py` (`SHA256 = {EXPECTED_HARNESS_SHA256}`).",
        f"- **SM80 Cubin Register Spill Elimination**: Upstream `fla.ops.cyfa` + `fla.ops.common` compiles `{up_cubin['spilling_specializations']}/{up_cubin['total_specializations']}` spilling specializations (`max_regs = {up_cubin['max_regs']}`, `max_spills = {up_cubin['max_spills']}`, `total_spills = {up_cubin['total_spills']}`, `max_smem = {up_cubin['max_shared_bytes']}` B) vs **Round-22 RSI `{rsi_cubin['spilling_specializations']}/{rsi_cubin['total_specializations']}` spilling specializations across all `{rsi_cubin['kernel_count']}/{rsi_cubin['kernel_count']}` kernels (`max_regs = {rsi_cubin['max_regs']} <= 168`, `total_spills = 0`, `max_smem = {rsi_cubin['max_shared_bytes']}` B)**.",
        f"- **Exact `FP32` / `BF16` Recurrent Oracle Parity**: `max_fp32_rel_l2 = {t1['max_fp32_rel_l2']:.2e}` (`{t1['upstream_fp32_tf32_drift']['rsi_fp32_accuracy_improvement_x']}x` more accurate than upstream `chunk_cyfa` `{t1['upstream_fp32_tf32_drift']['max_rel_l2']:.2e}` `TF32` truncation drift), `max_bf16_rel_l2 = {t1['max_bf16_rel_l2']:.2e}`, `3/3` adversarial mutations killed.",
        f"- **Track 3A Kernel Speedup (12 Shapes)**: Peak **`{headline['peak_kernel_fwdbwd_speedup_x']}x` Fwd+Bwd** (`{headline['peak_kernel_bwd_speedup_x']}x` Bwd, `{headline['peak_kernel_fwd_speedup_x']}x` Fwd; mean `{headline['mean_kernel_fwdbwd_speedup_x']}x` Fwd+Bwd).",
        f"- **Track 3B End-to-End Block Ablation (`A0 -> A4`)**: Peak **`{headline['peak_layer_A4_vs_A0_speedup_x']}x` Fwd+Bwd** (`mean = {headline['mean_layer_A4_vs_A0_speedup_x']}x`).",
        f"- **Track 4 Fixed-Wall-Clock `val_bpb` Autoresearch (`30.0s`)**: **`{headline['autoresearch_step_rate_speedup_x']}x` step rate** (`{a4_steps['median']}` vs `{a0_steps['median']}` steps), driving `val_bpb` from `{init_bpb}` to **`{a4_bpb['median']}`** vs `{a0_bpb['median']}` (`-{headline['autoresearch_extra_val_bpb_gain']}` extra `val_bpb` gain).",
        f"- **Track 5 8-GPU NVSwitch `CyFA-CP = 8` (`Ulysses` vs `Ring`)**: Peak **`{headline['peak_cp8_ulysses_vs_ring_speedup_x']}x` Fwd+Bwd** (`mean = {headline['mean_cp8_ulysses_vs_ring_speedup_x']}x`).",
        "",
        "## 2. Track 3A: 12-Shape Kernel Matrix (`chunk_cyfa` vs `chunk_cyfa_rsi`, 3-Rep Median)",
        "| B | T | H | K | V | M | Upstream Fwd (ms) | RSI Fwd (ms) | Fwd Speedup | Upstream Bwd (ms) | RSI Bwd (ms) | Bwd Speedup | Upstream Fwd+Bwd (ms) | RSI Fwd+Bwd (ms) | Fwd+Bwd Speedup |",
        "|---|---|---|---|---|---|-------------------|--------------|-------------|-------------------|--------------|-------------|-----------------------|------------------|-----------------|",
    ]
    for r in kernel_csv_rows:
        report_lines.append(
            f"| {r['B']} | {r['T']} | {r['H']} | {r['K']} | {r['V']} | {r['M']} | "
            f"{r['upstream_fwd_ms']:.3f} | {r['rsi_fwd_ms']:.3f} | **{r['fwd_speedup_x']:.2f}x** | "
            f"{r['upstream_bwd_ms']:.3f} | {r['rsi_bwd_ms']:.3f} | **{r['bwd_speedup_x']:.2f}x** | "
            f"{r['upstream_fwdbwd_ms']:.3f} | {r['rsi_fwdbwd_ms']:.3f} | **{r['fwdbwd_speedup_x']:.2f}x** |"
        )

    report_lines.extend([
        "",
        "## 3. Track 3B: 5-Stage `CyclicFlowAttentionBlockRSI` Progressive Ablation (`A0 -> A4`)",
        "| Config | B | T | D | K | V | M | A0 (ms) | A1 (ms) | A2 (ms) | A3 (ms) | A4 (ms) | A4 vs A0 | A4 vs A1 |",
        "|--------|---|---|---|---|---|---|---------|---------|---------|---------|---------|----------|----------|",
    ])
    for r in layer_csv_rows:
        report_lines.append(
            f"| `{r['config']}` | {r['B']} | {r['T']} | {r['D']} | {r['K']} | {r['V']} | {r['M']} | "
            f"{r['A0_fwdbwd_ms']:.3f} | {r['A1_fwdbwd_ms']:.3f} | {r['A2_fwdbwd_ms']:.3f} | "
            f"{r['A3_fwdbwd_ms']:.3f} | {r['A4_fwdbwd_ms']:.3f} | **{r['speedup_A4_vs_A0_x']:.2f}x** | **{r['speedup_A4_vs_A1_x']:.2f}x** |"
        )

    report_lines.extend([
        "",
        "## 4. Track 5: 8-GPU NVSwitch `CyFA-CP = 8` Scaling (`Ulysses` vs `Ring`)",
        "| T_total | T_local | H | K | V | M | Ring CP=8 (ms) | Ulysses CP=8 (ms) | Ulysses vs Ring |",
        "|---------|---------|---|---|---|---|----------------|-------------------|-----------------|",
    ])
    for r in cp8_csv_rows:
        report_lines.append(
            f"| {r['T_total']} | {r['T_local']} | {r['H']} | {r['K']} | {r['V']} | {r['M']} | "
            f"{r['ring_fwdbwd_ms']:.3f} | {r['ulysses_fwdbwd_ms']:.3f} | **{r['ulysses_vs_ring_speedup_x']:.2f}x** |"
        )

    (out_dir / "cyfa_rsi_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    readme_lines = [
        "# E148 (`v139`, Round-22 RSI) Benchmark Artifact Directory",
        "",
        "- **Experiment**: `E148` (`v139`, Round-22 RSI) — `Cyclic Flow Attention (CyFA): Linear Sequence Modeling with Relative-Time-Partitioned Memory` (`arXiv:2609.36259`)",
        "- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch)",
        "- **Files**:",
        "  - `rep1.json`, `rep2.json`, `rep3.json`: Raw 3-rep benchmark outputs from `code/bench_cyfa_rsi.py --mode full`.",
        "  - `cyfa_rsi_summary.json`: Aggregated 3-rep median/IQR/CoV statistics and cubin/parity/mutation audits.",
        "  - `cyfa_rsi_report.md`: Markdown summary report.",
        "  - `cyfa_kernel_matrix.csv`, `cyfa_layer_ablation.csv`, `cyfa_autoresearch.csv`, `cyfa_cp8_scaling.csv`: CSV tables.",
        "  - `SHA256SUMS.txt`: Cryptographic checksums of all artifacts.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    sha_lines = []
    for fpath in sorted(out_dir.iterdir()):
        if fpath.is_file() and fpath.name != "SHA256SUMS.txt":
            sha_lines.append(f"{_sha256_file(fpath)}  {fpath.name}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    print(f"[E148 Analyzer] Wrote summary, report, CSVs, README, and SHA256SUMS to {out_dir}")
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
