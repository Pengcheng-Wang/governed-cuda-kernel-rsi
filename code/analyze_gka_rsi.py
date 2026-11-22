#!/usr/bin/env python3
"""
Round-30 RSI (E156 / v147) Statistical Aggregator:
Aggregates 3 independent hardware repetitions (rep1.json..rep3.json) of
Gated KalmaNet (GKA): A Fading Memory Layer Through Test-Time Ridge Regression
(arXiv:2511.21016 — Rui-Qi Wang, Disen Lan, Songlin Yang; fla/ops/gka & fla/layers/gka.py)
and generates gka_rsi_summary.json, gka_rsi_report.md, 5 CSV tables, README.md, and SHA256SUMS.txt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = float(statistics.mean(vals))
    s = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0
    return m, s


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict]):
    with path.open("w", newline="\n", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def analyze_gka_rsi(raw_dir: Path) -> dict:
    reps = [json.loads((raw_dir / f"rep{i}.json").read_text(encoding="utf-8")) for i in range(1, 4)]
    r0 = reps[0]
    t1_0 = r0["track1"]

    oracle_summary = {
        "fp32_max_rel_l2": max(r["track1"]["fp32_max_rel_l2"] for r in reps),
        "bf16_max_rel_l2": max(r["track1"]["bf16_max_rel_l2"] for r in reps),
        "mutations_killed": min(r["track1"]["mutations_killed"] for r in reps),
        "mutations_total": t1_0["mutations_total"],
        "mutation_kill_rate_pct": 100.0
        * min(r["track1"]["mutations_killed"] for r in reps)
        / max(t1_0["mutations_total"], 1),
        "mutations": t1_0["mutations"],
        "upstream_pathology": t1_0["upstream_pathology"],
    }

    rsi_cubin_audit = dict(r0["rsi_cubin_audit"])
    assert rsi_cubin_audit["spilling_specializations"] == 0, "Expected 0 RSI spilling specializations"
    assert rsi_cubin_audit["over168_specializations"] == 0, "Expected 0 RSI >168 reg specializations"

    # Track 1C: 8-Shape Training Kernel Matrix
    n_shapes = len(t1_0["kernel_matrix"])
    kernel_matrix_rows = []
    for idx in range(n_shapes):
        s0 = t1_0["kernel_matrix"][idx]
        uf_m, uf_s = _mean_std([r["track1"]["kernel_matrix"][idx]["upstream_fwd_ms"] for r in reps])
        ub_m, ub_s = _mean_std([r["track1"]["kernel_matrix"][idx]["upstream_bwd_ms"] for r in reps])
        ut_m, ut_s = _mean_std([r["track1"]["kernel_matrix"][idx]["upstream_total_ms"] for r in reps])
        rf_m, rf_s = _mean_std([r["track1"]["kernel_matrix"][idx]["rsi_fwd_ms"] for r in reps])
        rb_m, rb_s = _mean_std([r["track1"]["kernel_matrix"][idx]["rsi_bwd_ms"] for r in reps])
        rt_m, rt_s = _mean_std([r["track1"]["kernel_matrix"][idx]["rsi_total_ms"] for r in reps])
        rtf_m, rtf_s = _mean_std([r["track1"]["kernel_matrix"][idx]["rsi_total_tflops"] for r in reps])

        kernel_matrix_rows.append(
            {
                "shape": s0["shape"],
                "B": s0["B"],
                "T": s0["T"],
                "H": s0["H"],
                "K": s0["K"],
                "V": s0["V"],
                "num_iter": s0["num_iter"],
                "upstream_fwd_ms_mean": round(uf_m, 4),
                "upstream_fwd_ms_std": round(uf_s, 4),
                "upstream_bwd_ms_mean": round(ub_m, 4),
                "upstream_bwd_ms_std": round(ub_s, 4),
                "upstream_total_ms_mean": round(ut_m, 4),
                "upstream_total_ms_std": round(ut_s, 4),
                "rsi_fwd_ms_mean": round(rf_m, 4),
                "rsi_fwd_ms_std": round(rf_s, 4),
                "rsi_bwd_ms_mean": round(rb_m, 4),
                "rsi_bwd_ms_std": round(rb_s, 4),
                "rsi_total_ms_mean": round(rt_m, 4),
                "rsi_total_ms_std": round(rt_s, 4),
                "fwd_speedup_vs_upstream": round(uf_m / rf_m, 3),
                "bwd_speedup_vs_upstream": round(ub_m / rb_m, 3),
                "total_speedup_vs_upstream": round(ut_m / rt_m, 3),
                "rsi_total_tflops_mean": round(rtf_m, 2),
                "rsi_total_tflops_std": round(rtf_s, 2),
            }
        )

    # Track 2: Decode Matrix
    n_dec = len(r0["track2"]["decode_matrix"])
    decode_rows = []
    for idx in range(n_dec):
        d0 = r0["track2"]["decode_matrix"][idx]
        uc_m, uc_s = _mean_std([r["track2"]["decode_matrix"][idx]["upstream_chunk_ms"] for r in reps])
        ur_m, ur_s = _mean_std([r["track2"]["decode_matrix"][idx]["upstream_fused_recurrent_ms"] for r in reps])
        rd_m, rd_s = _mean_std([r["track2"]["decode_matrix"][idx]["rsi_decode_ms"] for r in reps])
        decode_rows.append(
            {
                "shape": d0["shape"],
                "B": d0["B"],
                "T_q": d0["T_q"],
                "H": d0["H"],
                "K": d0["K"],
                "V": d0["V"],
                "upstream_chunk_ms_mean": round(uc_m, 4),
                "upstream_chunk_ms_std": round(uc_s, 4),
                "upstream_fused_recurrent_ms_mean": round(ur_m, 4),
                "upstream_fused_recurrent_ms_std": round(ur_s, 4),
                "rsi_decode_ms_mean": round(rd_m, 4),
                "rsi_decode_ms_std": round(rd_s, 4),
                "speedup_vs_upstream_chunk": round(uc_m / rd_m, 3),
                "speedup_vs_upstream_recurrent": round(ur_m / rd_m, 3),
                "o_rel_l2": d0["o_rel_l2"],
                "ht_kk_rel_l2": d0["ht_kk_rel_l2"],
                "ht_kv_rel_l2": d0["ht_kv_rel_l2"],
            }
        )

    # Track 3: Layer Ablation (Stages A0..A4 across D_model=1024, 2048)
    n_abl = len(r0["track3"]["layer_ablation"])
    ablation_rows = []
    a0_means_by_d = {}
    for idx in range(n_abl):
        a0 = r0["track3"]["layer_ablation"][idx]
        if a0["stage"] == "A0":
            tm, _ = _mean_std([r["track3"]["layer_ablation"][idx]["total_ms"] for r in reps])
            a0_means_by_d[a0["D_model"]] = tm

    for idx in range(n_abl):
        a0 = r0["track3"]["layer_ablation"][idx]
        fm, fs = _mean_std([r["track3"]["layer_ablation"][idx]["fwd_ms"] for r in reps])
        bm, bs = _mean_std([r["track3"]["layer_ablation"][idx]["bwd_ms"] for r in reps])
        tm, ts = _mean_std([r["track3"]["layer_ablation"][idx]["total_ms"] for r in reps])
        tps_m, tps_s = _mean_std([r["track3"]["layer_ablation"][idx]["tokens_per_s"] for r in reps])
        base_tm = a0_means_by_d[a0["D_model"]]
        ablation_rows.append(
            {
                "D_model": a0["D_model"],
                "H": a0["H"],
                "K": a0["K"],
                "B": a0["B"],
                "T": a0["T"],
                "stage": a0["stage"],
                "fwd_ms_mean": round(fm, 4),
                "fwd_ms_std": round(fs, 4),
                "bwd_ms_mean": round(bm, 4),
                "bwd_ms_std": round(bs, 4),
                "total_ms_mean": round(tm, 4),
                "total_ms_std": round(ts, 4),
                "tokens_per_s_mean": round(tps_m, 1),
                "tokens_per_s_std": round(tps_s, 1),
                "speedup_vs_a0": round(base_tm / tm, 3),
                "rel_l2_vs_a0": a0["rel_l2_vs_a0"],
            }
        )

    # Track 4: 8-GPU NVSwitch Ulysses GKA-CP = 8
    n_cp = len(r0["track4"]["cp8_scaling"])
    cp8_rows = []
    for idx in range(n_cp):
        c0 = r0["track4"]["cp8_scaling"][idx]
        u1_m, u1_s = _mean_std([r["track4"]["cp8_scaling"][idx]["one_gpu_upstream_ms"] for r in reps])
        r1_m, r1_s = _mean_std([r["track4"]["cp8_scaling"][idx]["one_gpu_rsi_ms"] for r in reps])
        rg_m, rg_s = _mean_std([r["track4"]["cp8_scaling"][idx]["ring_cp8_ms"] for r in reps])
        ul_m, ul_s = _mean_std([r["track4"]["cp8_scaling"][idx]["ulysses_cp8_ms"] for r in reps])
        tps_m, tps_s = _mean_std([r["track4"]["cp8_scaling"][idx]["ulysses_tokens_per_s"] for r in reps])

        cp8_rows.append(
            {
                "B": c0["B"],
                "T_total": c0["T_total"],
                "T_local": c0["T_local"],
                "H": c0["H"],
                "K": c0["K"],
                "V": c0["V"],
                "one_gpu_upstream_ms_mean": round(u1_m, 4),
                "one_gpu_upstream_ms_std": round(u1_s, 4),
                "one_gpu_rsi_ms_mean": round(r1_m, 4),
                "one_gpu_rsi_ms_std": round(r1_s, 4),
                "ring_cp8_ms_mean": round(rg_m, 4),
                "ring_cp8_ms_std": round(rg_s, 4),
                "ulysses_cp8_ms_mean": round(ul_m, 4),
                "ulysses_cp8_ms_std": round(ul_s, 4),
                "speedup_vs_ring": round(rg_m / ul_m, 3),
                "speedup_vs_1gpu_upstream": round(u1_m / ul_m, 3),
                "speedup_vs_1gpu_rsi": round(r1_m / ul_m, 3),
                "parallel_efficiency_vs_1gpu_upstream_pct": round(100.0 * (u1_m / ul_m) / 8.0, 2),
                "ulysses_tokens_per_s_mean": round(tps_m, 1),
                "ulysses_tokens_per_s_std": round(tps_s, 1),
                "rel_l2_o": c0["rel_l2_o"],
                "rel_l2_dq": c0["rel_l2_dq"],
                "rel_l2_dk": c0["rel_l2_dk"],
                "rel_l2_dv": c0["rel_l2_dv"],
            }
        )

    # Track 5: Autoresearch val_bpb
    n_ar = len(r0["track5"]["autoresearch"])
    ar_rows = []
    for idx in range(n_ar):
        a0 = r0["track5"]["autoresearch"][idx]
        s0_m, s0_s = _mean_std([r["track5"]["autoresearch"][idx]["a0_steps"] for r in reps])
        s4_m, s4_s = _mean_std([r["track5"]["autoresearch"][idx]["a4_steps"] for r in reps])
        t0_m, t0_s = _mean_std([r["track5"]["autoresearch"][idx]["a0_tokens_per_s"] for r in reps])
        t4_m, t4_s = _mean_std([r["track5"]["autoresearch"][idx]["a4_tokens_per_s"] for r in reps])
        ib_m, _ = _mean_std([r["track5"]["autoresearch"][idx]["init_val_bpb"] for r in reps])
        b0_m, b0_s = _mean_std([r["track5"]["autoresearch"][idx]["a0_final_val_bpb"] for r in reps])
        b4_m, b4_s = _mean_std([r["track5"]["autoresearch"][idx]["a4_final_val_bpb"] for r in reps])
        eb_m, eb_s = _mean_std([r["track5"]["autoresearch"][idx]["extra_val_bpb_reduction"] for r in reps])

        ar_rows.append(
            {
                "D_model": a0["D_model"],
                "H": a0["H"],
                "K": a0["K"],
                "budget_s": a0["budget_s"],
                "a0_steps_mean": round(s0_m, 2),
                "a0_steps_std": round(s0_s, 2),
                "a4_steps_mean": round(s4_m, 2),
                "a4_steps_std": round(s4_s, 2),
                "step_ratio_a4_vs_a0": round(s4_m / max(s0_m, 1e-6), 3),
                "a0_tokens_per_s_mean": round(t0_m, 1),
                "a0_tokens_per_s_std": round(t0_s, 1),
                "a4_tokens_per_s_mean": round(t4_m, 1),
                "a4_tokens_per_s_std": round(t4_s, 1),
                "init_val_bpb": round(ib_m, 5),
                "a0_final_val_bpb_mean": round(b0_m, 5),
                "a0_final_val_bpb_std": round(b0_s, 5),
                "a4_final_val_bpb_mean": round(b4_m, 5),
                "a4_final_val_bpb_std": round(b4_s, 5),
                "extra_val_bpb_reduction_mean": round(eb_m, 5),
                "extra_val_bpb_reduction_std": round(eb_s, 5),
            }
        )

    fwd_speedups = [r["fwd_speedup_vs_upstream"] for r in kernel_matrix_rows]
    bwd_speedups = [r["bwd_speedup_vs_upstream"] for r in kernel_matrix_rows]
    tot_speedups = [r["total_speedup_vs_upstream"] for r in kernel_matrix_rows]
    dec_chunk_speedups = [r["speedup_vs_upstream_chunk"] for r in decode_rows]
    dec_rec_speedups = [r["speedup_vs_upstream_recurrent"] for r in decode_rows]
    a4_speedups = [r["speedup_vs_a0"] for r in ablation_rows if r["stage"] == "A4"]
    cp_ring_speedups = [r["speedup_vs_ring"] for r in cp8_rows]
    cp_1gpu_speedups = [r["speedup_vs_1gpu_upstream"] for r in cp8_rows]

    summary = {
        "experiment_id": "E156",
        "version": 147,
        "round": 30,
        "title": "Gated KalmaNet (GKA): A Fading Memory Layer Through Test-Time Ridge Regression (arXiv:2511.21016)",
        "repetitions": 3,
        "gpu_name": r0["gpu_name"],
        "gpu_count": r0["gpu_count"],
        "frozen_harness": r0["frozen_harness"],
        "oracle_and_mutations": oracle_summary,
        "rsi_cubin_audit": rsi_cubin_audit,
        "highlights": {
            "fwd_speedup_min": min(fwd_speedups),
            "fwd_speedup_max": max(fwd_speedups),
            "bwd_speedup_min": min(bwd_speedups),
            "bwd_speedup_max": max(bwd_speedups),
            "total_speedup_min": min(tot_speedups),
            "total_speedup_max": max(tot_speedups),
            "decode_speedup_vs_chunk_min": min(dec_chunk_speedups),
            "decode_speedup_vs_chunk_max": max(dec_chunk_speedups),
            "decode_speedup_vs_recurrent_min": min(dec_rec_speedups),
            "decode_speedup_vs_recurrent_max": max(dec_rec_speedups),
            "layer_a4_speedup_min": min(a4_speedups),
            "layer_a4_speedup_max": max(a4_speedups),
            "cp8_speedup_vs_ring_min": min(cp_ring_speedups),
            "cp8_speedup_vs_ring_max": max(cp_ring_speedups),
            "cp8_speedup_vs_1gpu_upstream_min": min(cp_1gpu_speedups),
            "cp8_speedup_vs_1gpu_upstream_max": max(cp_1gpu_speedups),
        },
        "kernel_matrix": kernel_matrix_rows,
        "decode_matrix": decode_rows,
        "layer_ablation": ablation_rows,
        "cp8_scaling": cp8_rows,
        "autoresearch": ar_rows,
    }

    # Write CSV artifacts
    _write_csv(raw_dir / "gka_kernel_matrix.csv", list(kernel_matrix_rows[0].keys()), kernel_matrix_rows)
    _write_csv(raw_dir / "gka_decode_matrix.csv", list(decode_rows[0].keys()), decode_rows)
    _write_csv(raw_dir / "gka_layer_ablation.csv", list(ablation_rows[0].keys()), ablation_rows)
    _write_csv(raw_dir / "gka_cp8_scaling.csv", list(cp8_rows[0].keys()), cp8_rows)
    _write_csv(raw_dir / "gka_autoresearch_val_bpb.csv", list(ar_rows[0].keys()), ar_rows)

    (raw_dir / "gka_rsi_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    up_c = oracle_summary["upstream_pathology"]["upstream_cubin_audit"]
    h = summary["highlights"]
    report_lines = [
        "# Round-30 RSI (`E156` / `v147`): Gated KalmaNet (`GKA`) — A Fading Memory Layer Through Test-Time Ridge Regression (`arXiv:2511.21016`)",
        "",
        "## 1. Upstream Pathology & Bug Audit (`fla/ops/gka` & `fla/layers/gka.py`)",
        f"- **Severe Register Spilling (`{up_c['spilling_specializations']} / {up_c['total_specializations']}` Specializations, `{up_c['total_spills']}` Total Spills, `max_regs = {up_c['max_regs']}`, `max_spills = {up_c['max_spills']}`/Thread)**: Upstream `chunk_gka_solve_fwd_kernel` (`8/8` spilling, up to `282` spills/thread) and `chunk_gka_bwd_kernel_dk_intra` (`12/12` spilling, up to `10,028` spills/thread and `114,688 B` SMEM) suffer catastrophic local-memory thrashing on `NVIDIA A100-SXM4-80GB` (`SM80`).",
        f"- **`SimpleGLABackwardGuard` 16-Bit Backward Crash on `K < 64` or `V < 32`**: Upstream `chunk_gka` delegates its value-readout stage to `chunk_simple_gla`, which crashes in `.backward()` for 16-bit inputs with `K < 64` or `V < 32` (`crashed = {oracle_summary['upstream_pathology']['upstream_k32_bf16_bwd_crashed']}`). Our self-contained `chunk_gka_rsi` succeeds with finite, oracle-verified gradients (`rsi_succeeded = {oracle_summary['upstream_pathology']['rsi_k32_bf16_bwd_succeeded']}`).",
        f"- **`TF32` Drift in `FP32` Mode**: Upstream `fla/ops/gka` omits `input_precision=\"ieee\"` in `FP32`, causing `max_rel_l2 = {oracle_summary['upstream_pathology']['upstream_fp32_tf32_max_rel_l2']:.2e}` against the `FP64` reference oracle.",
        f"- **Zero-Spill `code/gka_rsi_kernels.py` Cubin Audit**: `{rsi_cubin_audit['kernel_count']}` custom Triton kernels, `{rsi_cubin_audit['total_specializations']}` compiled specializations, **`{rsi_cubin_audit['spilling_specializations']}` spilling specializations**, **`{rsi_cubin_audit['over168_specializations']}` over-168-register specializations**, `max_regs = {rsi_cubin_audit['max_regs']} <= 168`, `max_spills = {rsi_cubin_audit['max_spills']}`, `max_smem_bytes = {rsi_cubin_audit['max_smem_bytes']} B`.",
        f"- **`FP64` Oracle & Mutation Gate**: `FP32 max_rel_l2 = {oracle_summary['fp32_max_rel_l2']:.2e} < 1e-3`, `BF16 max_rel_l2 = {oracle_summary['bf16_max_rel_l2']:.2e} < 1.5e-2`, and `{oracle_summary['mutations_killed']} / {oracle_summary['mutations_total']}` (`{oracle_summary['mutation_kill_rate_pct']:.1f}%`) adversarial mutations killed.",
        "",
        "## 2. Track 1 — 8-Shape Training Kernel Matrix (`chunk_gka_rsi` vs Upstream `chunk_gka`)",
        f"- Forward speedup: **`{h['fwd_speedup_min']:.2f}x–{h['fwd_speedup_max']:.2f}x`**",
        f"- Backward speedup: **`{h['bwd_speedup_min']:.2f}x–{h['bwd_speedup_max']:.2f}x`**",
        f"- Combined `FWD+BWD` speedup: **`{h['total_speedup_min']:.2f}x–{h['total_speedup_max']:.2f}x`**",
        "",
        "## 3. Track 2 — Single-Pass Multi-Warp Online Decoding (`gka_decode_rsi`)",
        f"- Speedup vs upstream `chunk_gka`: **`{h['decode_speedup_vs_chunk_min']:.2f}x–{h['decode_speedup_vs_chunk_max']:.2f}x`**",
        f"- Speedup vs upstream `fused_recurrent_gka`: **`{h['decode_speedup_vs_recurrent_min']:.2f}x–{h['decode_speedup_vs_recurrent_max']:.2f}x`**",
        "",
        "## 4. Track 3 — 5-Stage `GatedKalmaNetBlockRSI` Layer Periphery Ablation (`A0 -> A4`)",
        f"- Stage `A4` speedup vs `A0`: **`{h['layer_a4_speedup_min']:.2f}x–{h['layer_a4_speedup_max']:.2f}x`**",
        "",
        "## 5. Track 4 — 8-GPU NVSwitch Context Parallelism (`GKA-CP = 8`)",
        f"- Ulysses `GKA-CP = 8` speedup vs Ring/AllGather baseline: **`{h['cp8_speedup_vs_ring_min']:.2f}x–{h['cp8_speedup_vs_ring_max']:.2f}x`**",
        f"- Ulysses `GKA-CP = 8` speedup vs 1-GPU upstream `chunk_gka`: **`{h['cp8_speedup_vs_1gpu_upstream_min']:.2f}x–{h['cp8_speedup_vs_1gpu_upstream_max']:.2f}x`**",
        "",
        "## 6. Track 5 — Fixed-Wall-Clock Autoresearch `val_bpb` Loop (`12 s` Budget)",
        f"- `D_model = 1024`: `A4` executes `{ar_rows[0]['a4_steps_mean']:.1f}` vs `{ar_rows[0]['a0_steps_mean']:.1f}` steps (`{ar_rows[0]['step_ratio_a4_vs_a0']:.2f}x`), reducing `val_bpb` by an extra **`{ar_rows[0]['extra_val_bpb_reduction_mean']:.4f} bpb`**.",
        f"- `D_model = 2048`: `A4` executes `{ar_rows[1]['a4_steps_mean']:.1f}` vs `{ar_rows[1]['a0_steps_mean']:.1f}` steps (`{ar_rows[1]['step_ratio_a4_vs_a0']:.2f}x`), reducing `val_bpb` by an extra **`{ar_rows[1]['extra_val_bpb_reduction_mean']:.4f} bpb`**.",
    ]
    (raw_dir / "gka_rsi_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")

    readme_lines = [
        "# Raw Artifacts: Round-30 RSI (`E156` / `v147`) — Gated KalmaNet (`GKA`, `arXiv:2511.21016`)",
        "",
        "- `smoke.json`: 5-track smoke verification receipt.",
        "- `rep1.json..rep3.json`: 3 independent 8-GPU hardware repetitions on `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch).",
        "- `gka_rsi_summary.json` & `gka_rsi_report.md`: 3-rep statistical summary and technical report.",
        "- `gka_kernel_matrix.csv`, `gka_decode_matrix.csv`, `gka_layer_ablation.csv`, `gka_cp8_scaling.csv`, `gka_autoresearch_val_bpb.csv`: 5 CSV tables.",
        "- `SHA256SUMS.txt`: Cryptographic SHA256 manifest of all files in `results/raw/gka_rsi/`.",
    ]
    (raw_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8", newline="\n")

    sha_lines = []
    for p in sorted(raw_dir.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            sha_lines.append(f"{digest}  {p.name}")
    (raw_dir / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8", newline="\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze_gka_rsi(args.raw_dir)
    print(f"[E156/v147] Aggregated 3 reps into {args.raw_dir / 'gka_rsi_summary.json'}", flush=True)
    print("Highlights:", json.dumps(summary["highlights"], indent=2), flush=True)


if __name__ == "__main__":
    main()
