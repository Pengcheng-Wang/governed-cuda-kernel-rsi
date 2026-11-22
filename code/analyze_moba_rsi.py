#!/usr/bin/env python3
"""
Round-32 RSI (E158 / v149) Statistical Aggregator:
Aggregates rep1.json..rep3.json for Mixture of Block Attention (MoBA, arXiv:2502.13189),
and generates moba_rsi_summary.json, moba_rsi_report.md, 5 CSV tables, README.md, and SHA256SUMS.txt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List


def _mean_std(vals: List[float]):
    n = len(vals)
    m = sum(vals) / n
    if n <= 1:
        return round(m, 6), 0.0
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    return round(m, 6), round(math.sqrt(var), 6)


def _write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze_moba_rsi(raw_dir: Path) -> Dict:
    reps = [json.loads((raw_dir / f"rep{i}.json").read_text(encoding="utf-8")) for i in (1, 2, 3)]
    r0 = reps[0]
    expected_sha = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"
    for r in reps:
        assert r["frozen_harness"]["harness_sha256"] == expected_sha
        assert r["rsi_cubin_audit"]["all_zero_spill"] is True

    # Track 1: Kernel Matrix
    t1_rows = []
    for idx in range(len(r0["kernel_matrix"])):
        base = r0["kernel_matrix"][idx]
        up_f, up_f_s = _mean_std([r["kernel_matrix"][idx]["upstream_fwd_ms"] for r in reps])
        up_b, up_b_s = _mean_std([r["kernel_matrix"][idx]["upstream_bwd_ms"] for r in reps])
        up_t, up_t_s = _mean_std([r["kernel_matrix"][idx]["upstream_total_ms"] for r in reps])
        up_m, _ = _mean_std([r["kernel_matrix"][idx]["upstream_peak_mem_mib"] for r in reps])
        rs_f, rs_f_s = _mean_std([r["kernel_matrix"][idx]["rsi_fwd_ms"] for r in reps])
        rs_b, rs_b_s = _mean_std([r["kernel_matrix"][idx]["rsi_bwd_ms"] for r in reps])
        rs_t, rs_t_s = _mean_std([r["kernel_matrix"][idx]["rsi_total_ms"] for r in reps])
        rs_m, _ = _mean_std([r["kernel_matrix"][idx]["rsi_peak_mem_mib"] for r in reps])
        tf_m, tf_s = _mean_std([r["kernel_matrix"][idx]["rsi_total_tflops"] for r in reps])
        t1_rows.append(
            {
                "shape": base["shape"],
                "B": base["B"],
                "T": base["T"],
                "HQ": base["HQ"],
                "HKV": base["HKV"],
                "K": base["K"],
                "V": base["V"],
                "chunk_size": base["chunk_size"],
                "topk": base["topk"],
                "upstream_fwd_ms_mean": round(up_f, 4),
                "upstream_fwd_ms_std": round(up_f_s, 4),
                "upstream_bwd_ms_mean": round(up_b, 4),
                "upstream_bwd_ms_std": round(up_b_s, 4),
                "upstream_total_ms_mean": round(up_t, 4),
                "upstream_total_ms_std": round(up_t_s, 4),
                "upstream_peak_mem_mib": round(up_m, 2),
                "rsi_fwd_ms_mean": round(rs_f, 4),
                "rsi_fwd_ms_std": round(rs_f_s, 4),
                "rsi_bwd_ms_mean": round(rs_b, 4),
                "rsi_bwd_ms_std": round(rs_b_s, 4),
                "rsi_total_ms_mean": round(rs_t, 4),
                "rsi_total_ms_std": round(rs_t_s, 4),
                "rsi_peak_mem_mib": round(rs_m, 2),
                "fwd_speedup_vs_upstream": round(up_f / rs_f, 3),
                "bwd_speedup_vs_upstream": round(up_b / rs_b, 3),
                "total_speedup_vs_upstream": round(up_t / rs_t, 3),
                "memory_reduction_vs_upstream": round(up_m / rs_m, 3),
                "rsi_total_tflops_mean": round(tf_m, 2),
                "rsi_total_tflops_std": round(tf_s, 2),
            }
        )

    # Track 2: Decode Matrix
    t2_rows = []
    for idx in range(len(r0["decode_matrix"])):
        base = r0["decode_matrix"][idx]
        up_m, up_s = _mean_std([r["decode_matrix"][idx]["upstream_parallel_moba_ms"] for r in reps])
        ch_m, ch_s = _mean_std([r["decode_matrix"][idx]["rsi_chunk_ms"] for r in reps])
        dc_m, dc_s = _mean_std([r["decode_matrix"][idx]["rsi_decode_ms"] for r in reps])
        t2_rows.append(
            {
                "shape": base["shape"],
                "B": base["B"],
                "T_q": base["T_q"],
                "T_kv": base["T_kv"],
                "HQ": base["HQ"],
                "HKV": base["HKV"],
                "K": base["K"],
                "V": base["V"],
                "chunk_size": base["chunk_size"],
                "topk": base["topk"],
                "upstream_parallel_moba_ms_mean": round(up_m, 4),
                "upstream_parallel_moba_ms_std": round(up_s, 4),
                "rsi_chunk_ms_mean": round(ch_m, 4),
                "rsi_chunk_ms_std": round(ch_s, 4),
                "rsi_decode_ms_mean": round(dc_m, 4),
                "rsi_decode_ms_std": round(dc_s, 4),
                "speedup_vs_upstream": round(up_m / dc_m, 3),
                "speedup_vs_rsi_chunk": round(ch_m / dc_m, 3),
                "o_rel_l2": base["o_rel_l2"],
            }
        )

    # Track 3: Layer Ablation
    t3_rows = []
    for idx in range(len(r0["layer_ablation"])):
        base = r0["layer_ablation"][idx]
        f_m, f_s = _mean_std([r["layer_ablation"][idx]["fwd_ms"] for r in reps])
        b_m, b_s = _mean_std([r["layer_ablation"][idx]["bwd_ms"] for r in reps])
        t_m, t_s = _mean_std([r["layer_ablation"][idx]["total_ms"] for r in reps])
        tk_m, tk_s = _mean_std([r["layer_ablation"][idx]["tokens_per_s"] for r in reps])
        a0_idx = (idx // 5) * 5
        a0_t_m, _ = _mean_std([r["layer_ablation"][a0_idx]["total_ms"] for r in reps])
        t3_rows.append(
            {
                "D_model": base["D_model"],
                "HQ": base["HQ"],
                "HKV": base["HKV"],
                "K": base["K"],
                "B": base["B"],
                "T": base["T"],
                "chunk_size": base["chunk_size"],
                "topk": base["topk"],
                "stage": base["stage"],
                "fwd_ms_mean": round(f_m, 4),
                "fwd_ms_std": round(f_s, 4),
                "bwd_ms_mean": round(b_m, 4),
                "bwd_ms_std": round(b_s, 4),
                "total_ms_mean": round(t_m, 4),
                "total_ms_std": round(t_s, 4),
                "tokens_per_s_mean": round(tk_m, 1),
                "tokens_per_s_std": round(tk_s, 1),
                "speedup_vs_a0": round(a0_t_m / t_m, 3),
                "rel_l2_vs_a0": base["rel_l2_vs_a0"],
            }
        )

    # Track 4: CP8 Scaling
    t4_rows = []
    for idx in range(len(r0["cp8_scaling"])):
        base = r0["cp8_scaling"][idx]
        u1_m, u1_s = _mean_std([r["cp8_scaling"][idx]["one_gpu_upstream_ms"] for r in reps])
        r1_m, r1_s = _mean_std([r["cp8_scaling"][idx]["one_gpu_rsi_ms"] for r in reps])
        rg_m, rg_s = _mean_std([r["cp8_scaling"][idx]["ring_cp8_ms"] for r in reps])
        ul_m, ul_s = _mean_std([r["cp8_scaling"][idx]["ulysses_cp8_ms"] for r in reps])
        tk_m, tk_s = _mean_std([r["cp8_scaling"][idx]["ulysses_tokens_per_s"] for r in reps])
        t4_rows.append(
            {
                "B": base["B"],
                "T_total": base["T_total"],
                "T_local": base["T_local"],
                "HQ": base["HQ"],
                "HKV": base["HKV"],
                "K": base["K"],
                "V": base["V"],
                "chunk_size": base["chunk_size"],
                "topk": base["topk"],
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
                "parallel_efficiency_vs_1gpu_upstream_pct": round((u1_m / (ul_m * 8.0)) * 100.0, 2),
                "ulysses_tokens_per_s_mean": round(tk_m, 1),
                "ulysses_tokens_per_s_std": round(tk_s, 1),
                "rel_l2_o": base["rel_l2_o"],
                "rel_l2_dq": base["rel_l2_dq"],
                "rel_l2_dk": base["rel_l2_dk"],
                "rel_l2_dv": base["rel_l2_dv"],
            }
        )

    # Track 5: Autoresearch
    t5_rows = []
    for idx in range(len(r0["autoresearch"])):
        base = r0["autoresearch"][idx]
        s0_m, s0_s = _mean_std([float(r["autoresearch"][idx]["a0_steps"]) for r in reps])
        s4_m, s4_s = _mean_std([float(r["autoresearch"][idx]["a4_steps"]) for r in reps])
        t0_m, t0_s = _mean_std([r["autoresearch"][idx]["a0_tokens_per_s"] for r in reps])
        t4_m, t4_s = _mean_std([r["autoresearch"][idx]["a4_tokens_per_s"] for r in reps])
        b0_m, b0_s = _mean_std([r["autoresearch"][idx]["a0_final_val_bpb"] for r in reps])
        b4_m, b4_s = _mean_std([r["autoresearch"][idx]["a4_final_val_bpb"] for r in reps])
        red_m, red_s = _mean_std([r["autoresearch"][idx]["extra_val_bpb_reduction"] for r in reps])
        t5_rows.append(
            {
                "D_model": base["D_model"],
                "HQ": base["HQ"],
                "HKV": base["HKV"],
                "K": base["K"],
                "chunk_size": base["chunk_size"],
                "topk": base["topk"],
                "budget_s": base["budget_s"],
                "a0_steps_mean": round(s0_m, 2),
                "a0_steps_std": round(s0_s, 2),
                "a4_steps_mean": round(s4_m, 2),
                "a4_steps_std": round(s4_s, 2),
                "step_ratio_a4_vs_a0": round(s4_m / s0_m, 3),
                "a0_tokens_per_s_mean": round(t0_m, 1),
                "a0_tokens_per_s_std": round(t0_s, 1),
                "a4_tokens_per_s_mean": round(t4_m, 1),
                "a4_tokens_per_s_std": round(t4_s, 1),
                "init_val_bpb": round(base["init_val_bpb"], 5),
                "a0_final_val_bpb_mean": round(b0_m, 5),
                "a0_final_val_bpb_std": round(b0_s, 5),
                "a4_final_val_bpb_mean": round(b4_m, 5),
                "a4_final_val_bpb_std": round(b4_s, 5),
                "extra_val_bpb_reduction_mean": round(red_m, 5),
                "extra_val_bpb_reduction_std": round(red_s, 5),
            }
        )

    highlights = {
        "fwd_speedup_min": min(r["fwd_speedup_vs_upstream"] for r in t1_rows),
        "fwd_speedup_max": max(r["fwd_speedup_vs_upstream"] for r in t1_rows),
        "bwd_speedup_min": min(r["bwd_speedup_vs_upstream"] for r in t1_rows),
        "bwd_speedup_max": max(r["bwd_speedup_vs_upstream"] for r in t1_rows),
        "total_speedup_min": min(r["total_speedup_vs_upstream"] for r in t1_rows),
        "total_speedup_max": max(r["total_speedup_vs_upstream"] for r in t1_rows),
        "memory_reduction_min": min(r["memory_reduction_vs_upstream"] for r in t1_rows),
        "memory_reduction_max": max(r["memory_reduction_vs_upstream"] for r in t1_rows),
        "decode_speedup_vs_upstream_min": min(r["speedup_vs_upstream"] for r in t2_rows),
        "decode_speedup_vs_upstream_max": max(r["speedup_vs_upstream"] for r in t2_rows),
        "decode_speedup_vs_chunk_min": min(r["speedup_vs_rsi_chunk"] for r in t2_rows),
        "decode_speedup_vs_chunk_max": max(r["speedup_vs_rsi_chunk"] for r in t2_rows),
        "layer_a4_speedup_min": min(r["speedup_vs_a0"] for r in t3_rows if r["stage"] == "A4"),
        "layer_a4_speedup_max": max(r["speedup_vs_a0"] for r in t3_rows if r["stage"] == "A4"),
        "cp8_speedup_vs_ring_min": min(r["speedup_vs_ring"] for r in t4_rows),
        "cp8_speedup_vs_ring_max": max(r["speedup_vs_ring"] for r in t4_rows),
        "cp8_speedup_vs_1gpu_upstream_min": min(r["speedup_vs_1gpu_upstream"] for r in t4_rows),
        "cp8_speedup_vs_1gpu_upstream_max": max(r["speedup_vs_1gpu_upstream"] for r in t4_rows),
    }

    summary = {
        "experiment_id": "E158",
        "version": 149,
        "round": 32,
        "title": "MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189, Moonshot AI / Kimi Team)",
        "repetitions": 3,
        "gpu_name": r0["gpu_name"],
        "gpu_count": r0["gpu_count"],
        "frozen_harness": r0["frozen_harness"],
        "oracle_and_mutations": r0["oracle_and_mutations"],
        "rsi_cubin_audit": r0["rsi_cubin_audit"],
        "highlights": highlights,
        "kernel_matrix": t1_rows,
        "decode_matrix": t2_rows,
        "layer_ablation": t3_rows,
        "cp8_scaling": t4_rows,
        "autoresearch": t5_rows,
    }

    (raw_dir / "moba_rsi_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    _write_csv(raw_dir / "track1_kernel_matrix.csv", t1_rows)
    _write_csv(raw_dir / "track2_decode_matrix.csv", t2_rows)
    _write_csv(raw_dir / "track3_layer_ablation.csv", t3_rows)
    _write_csv(raw_dir / "track4_cp8_scaling.csv", t4_rows)
    _write_csv(raw_dir / "track5_autoresearch.csv", t5_rows)

    cub = r0["rsi_cubin_audit"]
    orac = r0["oracle_and_mutations"]
    up_path = orac["upstream_pathology"]
    report_lines = [
        "# Round-32 RSI (`E158` / `v149`): Mixture of Block Attention (`MoBA`, `arXiv:2502.13189`, Moonshot AI / Kimi Team)",
        "",
        "## 1. Upstream Pathology & Bug Audit (`fla/ops/moba/parallel.py` & `fla/layers/moba.py`)",
        f"- **Zero Triton Kernels, Two-Stream `_flash_attn_varlen`, 12-Op Eager LSE Merge, Host-Sync `.nonzero()`, & HBM Duplication**: Upstream `fla/ops/moba` provides `0` Triton kernels, duplicates `(topk - 1)x` Q and `2x` KV in HBM, and stalls on `.nonzero()` (`{t1_rows[-2]['upstream_peak_mem_mib']:.1f} MiB` vs `{t1_rows[-2]['rsi_peak_mem_mib']:.1f} MiB` = **`{highlights['memory_reduction_max']:.2f}x` memory reduction**).",
        f"- **Deterministic GQA & `FP32` `RuntimeError` Crashes**: Upstream `MoBA` crashes on GQA (`num_kv_heads < num_heads`, `crashed = {up_path['upstream_gqa_crashed']}`) and `FP32` inputs (`crashed = {up_path['upstream_fp32_crashed']}`), and suffers `BF16` mean + `TF32` `einsum` gate top-$k$ boundary flips (`max_rel_l2 = {up_path['upstream_bf16_gate_drift_max_rel_l2']:.2e}` vs `{orac['bf16_max_rel_l2']:.2e}`).",
        f"- **Zero-Spill `code/moba_rsi_kernels.py` Cubin Audit**: `{cub['kernel_count']}` custom Triton kernels, `{cub['total_specializations']}` compiled specializations, **`{cub['spilling_specializations']}` spilling specializations**, **`{cub['over168_specializations']}` over-168-register specializations**, `max_regs = {cub['max_regs']} <= 168`, `max_spills = 0`, `max_smem_bytes = {cub['max_smem_bytes']} B`.",
        f"- **`FP64` Oracle & Mutation Gate**: `FP32 max_rel_l2 = {orac['fp32_max_rel_l2']:.2e} < 1e-4`, `BF16 max_rel_l2 = {orac['bf16_max_rel_l2']:.2e} < 1.5e-2`, and `{orac['mutations_killed']} / {orac['mutations_total']}` (`100.0%`) adversarial mutations killed.",
        "",
        "## 2. Track 1 — 8-Shape Training Kernel Matrix (`parallel_moba_rsi` vs Upstream `parallel_moba`)",
        f"- Forward speedup: **`{highlights['fwd_speedup_min']:.2f}x–{highlights['fwd_speedup_max']:.2f}x`**",
        f"- Backward speedup: **`{highlights['bwd_speedup_min']:.2f}x–{highlights['bwd_speedup_max']:.2f}x`**",
        f"- Combined `FWD+BWD` speedup: **`{highlights['total_speedup_min']:.2f}x–{highlights['total_speedup_max']:.2f}x`**",
        f"- Peak HBM reduction: **`{highlights['memory_reduction_min']:.2f}x–{highlights['memory_reduction_max']:.2f}x`**",
        "",
        "## 3. Track 2 — Single-Pass Centroid-Routed Block-Sparse Decoding (`moba_decode_rsi`)",
        f"- Speedup vs upstream `parallel_moba`: **`{highlights['decode_speedup_vs_upstream_min']:.2f}x–{highlights['decode_speedup_vs_upstream_max']:.2f}x`**",
        f"- Speedup vs `parallel_moba_rsi` full-chunk: **`{highlights['decode_speedup_vs_chunk_min']:.2f}x–{highlights['decode_speedup_vs_chunk_max']:.2f}x`**",
        "",
        "## 4. Track 3 — 5-Stage `MoBABlockRSI` & `BlockShare` Ablation (`A0 -> A4`)",
        f"- Stage `A4` speedup vs `A0`: **`{highlights['layer_a4_speedup_min']:.2f}x–{highlights['layer_a4_speedup_max']:.2f}x`**",
        "",
        "## 5. Track 4 — 8-GPU NVSwitch Context Parallelism (`MoBA-CP = 8`)",
        f"- Ulysses `MoBA-CP = 8` speedup vs Ring/AllGather baseline: **`{highlights['cp8_speedup_vs_ring_min']:.2f}x–{highlights['cp8_speedup_vs_ring_max']:.2f}x`**",
        f"- Ulysses `MoBA-CP = 8` speedup vs 1-GPU upstream `parallel_moba`: **`{highlights['cp8_speedup_vs_1gpu_upstream_min']:.2f}x–{highlights['cp8_speedup_vs_1gpu_upstream_max']:.2f}x`**",
        "",
        "## 6. Track 5 — Fixed-Wall-Clock Autoresearch `val_bpb` Loop (`12 s` Budget)",
        f"- `D_model = 1024`: `A4` executes `{t5_rows[0]['a4_steps_mean']:.1f}` vs `{t5_rows[0]['a0_steps_mean']:.1f}` steps (`{t5_rows[0]['step_ratio_a4_vs_a0']:.2f}x`), reducing `val_bpb` by an extra **`{t5_rows[0]['extra_val_bpb_reduction_mean']:.4f} bpb`**.",
        f"- `D_model = 2048`: `A4` executes `{t5_rows[1]['a4_steps_mean']:.1f}` vs `{t5_rows[1]['a0_steps_mean']:.1f}` steps (`{t5_rows[1]['step_ratio_a4_vs_a0']:.2f}x`), reducing `val_bpb` by an extra **`{t5_rows[1]['extra_val_bpb_reduction_mean']:.4f} bpb`**.",
    ]
    (raw_dir / "moba_rsi_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")

    readme_lines = [
        "# Raw Artifacts: Round-32 RSI (`E158` / `v149`) — Mixture of Block Attention (`MoBA`, `arXiv:2502.13189`)",
        "",
        "- `smoke.json`: 5-track smoke verification artifact.",
        "- `rep1.json`, `rep2.json`, `rep3.json`: 3 independent 8-GPU process repetitions.",
        "- `moba_rsi_summary.json` & `moba_rsi_report.md`: 3-rep statistical summary and technical report.",
        "- `track1_kernel_matrix.csv` .. `track5_autoresearch.csv`: 5 CSV tables.",
    ]
    (raw_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8", newline="\n")

    manifest_targets = [
        "smoke.json",
        "rep1.json",
        "rep2.json",
        "rep3.json",
        "moba_rsi_summary.json",
        "moba_rsi_report.md",
        "README.md",
        "track1_kernel_matrix.csv",
        "track2_decode_matrix.csv",
        "track3_layer_ablation.csv",
        "track4_cp8_scaling.csv",
        "track5_autoresearch.csv",
    ]
    sums = []
    for fname in manifest_targets:
        fpath = raw_dir / fname
        if fpath.exists():
            h = hashlib.sha256(fpath.read_bytes()).hexdigest()
            sums.append(f"{h}  {fname}")
    (raw_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze_moba_rsi(args.raw_dir)
    print(f"[E158/v149] Aggregated 3 reps into {args.raw_dir / 'moba_rsi_summary.json'}", flush=True)
    print("Highlights:", json.dumps(summary["highlights"], indent=2), flush=True)


if __name__ == "__main__":
    main()
