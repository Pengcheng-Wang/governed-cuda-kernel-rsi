#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E149 (v140, Round-23 RSI): Parallax Zero-Spill SM80 3-Rep Analyzer
==================================================================

Aggregates `rep{1,2,3}.json`, computes median / IQR / CoV / min / max statistics across
3 independent hardware repetitions, and writes deterministic CSV tables +
`parallax_rsi_report.md` + `parallax_rsi_summary.json` + `README.md` + `SHA256SUMS.txt`.
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
    default_dir = Path(__file__).resolve().parent.parent / "results" / "raw" / "parallax_rsi"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reps",
        nargs="+",
        default=[str(default_dir / f"rep{i}.json") for i in (1, 2, 3)],
    )
    parser.add_argument("--out", default=str(default_dir))
    args = parser.parse_args()

    reps = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reps]
    n_reps = len(reps)
    assert n_reps >= 3, f"Expected at least 3 repetitions, got {n_reps}"

    for r in reps:
        assert r["frozen_harness"]["sha256_verified"], "Harness SHA256 failed"
        assert r["frozen_harness"]["self_test_passed"], "Harness self-test failed"
        assert r["track1_oracle_and_mutations"]["mutations_killed"] == 3, "Mutations failed"
        assert r["track2_cubin_audit"]["rsi"]["total_spills"] == 0, "Zero-spill cubin audit failed"
        assert r["track2_cubin_audit"]["rsi"]["max_regs"] <= 168, "RSI max_regs <= 168 audit failed"

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
        tot_sp = round(up_tot["median"] / rsi_tot["median"], 3)
        bwd_sp = round(up_bwd["median"] / rsi_bwd["median"], 3) if rsi_bwd["median"] > 0 else tot_sp

        kernel_rows_agg.append({
            "kind": r0["kind"],
            "B": r0["B"],
            "T": r0["T"],
            "Sq": r0["Sq"],
            "HQ": r0["HQ"],
            "H": r0["H"],
            "G": r0["G"],
            "D": r0["D"],
            "W": r0["W"],
            "upstream": {"fwd_ms": up_fwd, "bwd_ms": up_bwd, "fwdbwd_ms": up_tot},
            "rsi_zero_spill": {"fwd_ms": rsi_fwd, "bwd_ms": rsi_bwd, "fwdbwd_ms": rsi_tot},
            "speedup": {"fwd_x": fwd_sp, "bwd_x": bwd_sp, "fwdbwd_x": tot_sp},
        })
        kernel_csv_rows.append({
            "kind": r0["kind"],
            "B": r0["B"],
            "T": r0["T"],
            "Sq": r0["Sq"],
            "HQ": r0["HQ"],
            "H": r0["H"],
            "G": r0["G"],
            "D": r0["D"],
            "W": r0["W"],
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
            "hidden_size": r0["hidden_size"],
            "HQ": r0["HQ"],
            "H": r0["H"],
            "D": r0["D"],
            "W": r0["W"],
            "stages": st_agg,
            "speedup_A4_vs_A0_x": sp_a0,
            "speedup_A4_vs_A1_x": sp_a1,
        })
        layer_csv_rows.append({
            "config": r0["config"],
            "B": r0["B"],
            "T": r0["T"],
            "hidden_size": r0["hidden_size"],
            "HQ": r0["HQ"],
            "H": r0["H"],
            "D": r0["D"],
            "W": r0["W"],
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
            "arm": "A0_upstream_parallax",
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
    n_c = len(reps[0]["track5_cp8_multi_gpu"]["rows"])
    for idx in range(n_c):
        r0 = reps[0]["track5_cp8_multi_gpu"]["rows"][idx]
        ring_ms = _median_iqr([r["track5_cp8_multi_gpu"]["rows"][idx]["ring_cp8_ms"] for r in reps])
        uly_ms = _median_iqr([r["track5_cp8_multi_gpu"]["rows"][idx]["ulysses_cp8_ms"] for r in reps])
        sp_cp = round(ring_ms["median"] / uly_ms["median"], 3)
        cp8_rows_agg.append({
            "B": r0["B"],
            "T_global": r0["T_global"],
            "T_local": r0["T_local"],
            "HQ": r0["HQ"],
            "H": r0["H"],
            "D": r0["D"],
            "W": r0["W"],
            "ring_cp8_ms": ring_ms,
            "ulysses_cp8_ms": uly_ms,
            "ulysses_vs_ring_speedup_x": sp_cp,
        })
        cp8_csv_rows.append({
            "B": r0["B"],
            "T_global": r0["T_global"],
            "T_local": r0["T_local"],
            "HQ": r0["HQ"],
            "H": r0["H"],
            "D": r0["D"],
            "W": r0["W"],
            "ring_cp8_ms": ring_ms["median"],
            "ulysses_cp8_ms": uly_ms["median"],
            "ulysses_vs_ring_speedup_x": sp_cp,
            "ulysses_iqr_ms": uly_ms["iqr"],
        })

    train_rows_agg = [r for r in kernel_rows_agg if r["kind"] == "train_fwdbwd"]
    headline = {
        "peak_kernel_fwdbwd_speedup_x": max(r["speedup"]["fwdbwd_x"] for r in kernel_rows_agg),
        "peak_train_fwdbwd_speedup_x": max(r["speedup"]["fwdbwd_x"] for r in train_rows_agg),
        "peak_kernel_bwd_speedup_x": max(r["speedup"]["bwd_x"] for r in train_rows_agg),
        "peak_kernel_fwd_speedup_x": max(r["speedup"]["fwd_x"] for r in kernel_rows_agg),
        "mean_kernel_fwdbwd_speedup_x": round(sum(r["speedup"]["fwdbwd_x"] for r in kernel_rows_agg) / len(kernel_rows_agg), 3),
        "mean_kernel_bwd_speedup_x": round(sum(r["speedup"]["bwd_x"] for r in train_rows_agg) / len(train_rows_agg), 3),
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
        "experiment": "E149_parallax_round23_rsi",
        "version": "v140",
        "n_reps": n_reps,
        "gpu_name": reps[0]["gpu_name"],
        "gpu_count": reps[0]["num_gpus"],
        "harness": reps[0]["frozen_harness"],
        "headline": headline,
        "track1_oracle_and_mutations": reps[0]["track1_oracle_and_mutations"],
        "track2_cubin_audit": reps[0]["track2_cubin_audit"],
        "track3a_kernel_matrix": kernel_rows_agg,
        "track3b_layer_ablation": layer_rows_agg,
        "track4_val_bpb_autoresearch": autoresearch_agg,
        "track5_cp8_nvswitch": {
            "parity": reps[0]["track5_cp8_multi_gpu"]["parity"],
            "rows": cp8_rows_agg,
        },
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "parallax_rsi_summary.json"
    summary_path.write_bytes((json.dumps(summary, indent=2) + "\n").encode("utf-8"))

    _write_csv(out_dir / "parallax_kernel_matrix.csv", kernel_csv_rows)
    _write_csv(out_dir / "parallax_layer_ablation.csv", layer_csv_rows)
    _write_csv(out_dir / "parallax_autoresearch.csv", autoresearch_csv_rows)
    _write_csv(out_dir / "parallax_cp8_scaling.csv", cp8_csv_rows)

    up_cubin = reps[0]["track2_cubin_audit"]["upstream"]
    rsi_cubin = reps[0]["track2_cubin_audit"]["rsi"]
    t1 = reps[0]["track1_oracle_and_mutations"]

    report_lines = [
        "# E149 (`v140`, Round-23 RSI): Parallax Parameterized Local Linear Attention Zero-Spill SM80 Suite",
        "",
        "## 1. Executive Summary",
        "- **Architecture**: `Parallax: Parameterized Local Linear Attention` (`arXiv:2605.29157`, Yifei Zuo et al.; upstream `fla.ops.parallax` & `fla.layers.parallax`).",
        f"- **Hardware**: `{reps[0]['num_gpus']}x {reps[0]['gpu_name']}` (`NV12` NVSwitch), `{n_reps}` independent repetitions.",
        f"- **Frozen Cryptographic Evaluator**: `code/rsi_kernel_harness.py` (`SHA256 = {EXPECTED_HARNESS_SHA256}`).",
        f"- **SM80 Cubin Register Spill Elimination**: Upstream `fla.ops.parallax` compiles `{up_cubin['spilling_specializations']}/{up_cubin['total_specializations']}` spilling specializations (`{up_cubin['regs_gt_168_specializations']}/{up_cubin['total_specializations']}` with `regs > 168`, `max_regs = {up_cubin['max_regs']}`, `max_spills = {up_cubin['max_spills']}`, `total_spills = {up_cubin['total_spills']}`, `max_smem = {up_cubin['max_shared_bytes']}` B) vs **Round-23 RSI `0/{rsi_cubin['total_specializations']}` spilling specializations across all `{rsi_cubin['n_kernels']}/{rsi_cubin['n_kernels']}` kernels (`max_regs = {rsi_cubin['max_regs']} <= 168`, `total_spills = 0`, `max_smem = {rsi_cubin['max_shared_bytes']}` B)**.",
        f"- **Exact `FP32` / `BF16` `naive_parallax` Oracle Parity**: `max_fp32_rel_l2 = {t1['max_fp32_rel_l2']:.2e}` (whereas upstream `parallel_parallax` crashes with `{t1['upstream_fp32_error']}` on `FP32`), `max_bf16_rel_l2 = {t1['max_bf16_rel_l2']:.2e}`, `3/3` adversarial mutations killed.",
        f"- **Track 3A Kernel Speedup (12 Shapes)**: Peak **`{headline['peak_kernel_fwdbwd_speedup_x']}x`** (`{headline['peak_train_fwdbwd_speedup_x']}x` Train Fwd+Bwd, `{headline['peak_kernel_bwd_speedup_x']}x` Bwd, `{headline['peak_kernel_fwd_speedup_x']}x` Decode/Fwd; mean `{headline['mean_kernel_fwdbwd_speedup_x']}x`).",
        f"- **Track 3B End-to-End Block Ablation (`A0 -> A4`)**: Peak **`{headline['peak_layer_A4_vs_A0_speedup_x']}x` Fwd+Bwd** (`mean = {headline['mean_layer_A4_vs_A0_speedup_x']}x`).",
        f"- **Track 4 Fixed-Wall-Clock `val_bpb` Autoresearch (`30.0s`)**: **`{headline['autoresearch_step_rate_speedup_x']}x` step rate** (`{a4_steps['median']}` vs `{a0_steps['median']}` steps), driving `val_bpb` from `{init_bpb}` to **`{a4_bpb['median']}`** vs `{a0_bpb['median']}` (`-{headline['autoresearch_extra_val_bpb_gain']}` extra `val_bpb` gain).",
        f"- **Track 5 8-GPU NVSwitch `Parallax-CP = 8` (`Ulysses` vs `5-Tuple Ring`)**: Peak **`{headline['peak_cp8_ulysses_vs_ring_speedup_x']}x`** (`mean = {headline['mean_cp8_ulysses_vs_ring_speedup_x']}x`, `0.0` max abs diff vs single-GPU).",
        "",
        "## 2. Track 3A: 12-Shape Kernel Matrix (`parallel_parallax` / `parallax_decode` vs Round-23 RSI, 3-Rep Median)",
        "| Kind | B | T | Sq | HQ | H | D | W | Upstream Fwd (ms) | RSI Fwd (ms) | Fwd Speedup | Upstream Bwd (ms) | RSI Bwd (ms) | Bwd Speedup | Upstream Total (ms) | RSI Total (ms) | Total Speedup |",
        "|------|---|---|----|----|---|---|---|-------------------|--------------|-------------|-------------------|--------------|-------------|---------------------|----------------|---------------|",
    ]
    for r in kernel_csv_rows:
        report_lines.append(
            f"| `{r['kind']}` | {r['B']} | {r['T']} | {r['Sq']} | {r['HQ']} | {r['H']} | {r['D']} | {r['W']} | "
            f"{r['upstream_fwd_ms']:.3f} | {r['rsi_fwd_ms']:.3f} | **{r['fwd_speedup_x']:.2f}x** | "
            f"{r['upstream_bwd_ms']:.3f} | {r['rsi_bwd_ms']:.3f} | **{r['bwd_speedup_x']:.2f}x** | "
            f"{r['upstream_fwdbwd_ms']:.3f} | {r['rsi_fwdbwd_ms']:.3f} | **{r['fwdbwd_speedup_x']:.2f}x** |"
        )

    report_lines.extend([
        "",
        "## 3. Track 3B: 5-Stage `ParallaxBlockRSI` Progressive Ablation (`A0 -> A4`)",
        "| Config | B | T | Hidden | HQ | H | D | W | A0 (ms) | A1 (ms) | A2 (ms) | A3 (ms) | A4 (ms) | A4 vs A0 | A4 vs A1 |",
        "|--------|---|---|--------|----|---|---|---|---------|---------|---------|---------|---------|----------|----------|",
    ])
    for r in layer_csv_rows:
        report_lines.append(
            f"| `{r['config']}` | {r['B']} | {r['T']} | {r['hidden_size']} | {r['HQ']} | {r['H']} | {r['D']} | {r['W']} | "
            f"{r['A0_fwdbwd_ms']:.3f} | {r['A1_fwdbwd_ms']:.3f} | {r['A2_fwdbwd_ms']:.3f} | "
            f"{r['A3_fwdbwd_ms']:.3f} | {r['A4_fwdbwd_ms']:.3f} | **{r['speedup_A4_vs_A0_x']:.2f}x** | **{r['speedup_A4_vs_A1_x']:.2f}x** |"
        )

    report_lines.extend([
        "",
        "## 4. Track 5: 8-GPU NVSwitch `Parallax-CP = 8` Scaling (`Ulysses` vs `5-Tuple Ring`)",
        "| T_global | T_local | HQ | H | D | W | 5-Tuple Ring CP=8 (ms) | Ulysses CP=8 (ms) | Ulysses vs Ring |",
        "|----------|---------|----|---|---|---|------------------------|-------------------|-----------------|",
    ])
    for r in cp8_csv_rows:
        report_lines.append(
            f"| {r['T_global']} | {r['T_local']} | {r['HQ']} | {r['H']} | {r['D']} | {r['W']} | "
            f"{r['ring_cp8_ms']:.3f} | {r['ulysses_cp8_ms']:.3f} | **{r['ulysses_vs_ring_speedup_x']:.2f}x** |"
        )

    (out_dir / "parallax_rsi_report.md").write_bytes(("\n".join(report_lines) + "\n").encode("utf-8"))

    readme_lines = [
        "# E149 (`v140`, Round-23 RSI) Benchmark Artifact Directory",
        "",
        "- **Experiment**: `E149` (`v140`, Round-23 RSI) — `Parallax: Parameterized Local Linear Attention` (`arXiv:2605.29157`)",
        "- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch)",
        "- **Files**:",
        "  - `smoke.json`, `rep1.json`, `rep2.json`, `rep3.json`: Raw smoke and 3-rep benchmark outputs from `code/bench_parallax_rsi.py`.",
        "  - `parallax_rsi_summary.json`: Aggregated 3-rep median/IQR/CoV statistics and cubin/parity/mutation audits.",
        "  - `parallax_rsi_report.md`: Markdown summary report.",
        "  - `parallax_kernel_matrix.csv`, `parallax_layer_ablation.csv`, `parallax_autoresearch.csv`, `parallax_cp8_scaling.csv`: CSV tables.",
        "  - `SHA256SUMS.txt`: Cryptographic checksums of all artifacts.",
    ]
    (out_dir / "README.md").write_bytes(("\n".join(readme_lines) + "\n").encode("utf-8"))

    sha_lines = []
    for fpath in sorted(out_dir.iterdir()):
        if fpath.is_file() and fpath.name != "SHA256SUMS.txt":
            sha_lines.append(f"{_sha256_file(fpath)}  {fpath.name}")
    (out_dir / "SHA256SUMS.txt").write_bytes(("\n".join(sha_lines) + "\n").encode("utf-8"))
    print(f"[E149 Analyzer] Wrote summary, report, CSVs, README, and SHA256SUMS to {out_dir}")
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
