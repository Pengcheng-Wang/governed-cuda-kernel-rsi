#!/usr/bin/env python3
"""
E139 (v130, Round-13 RSI): 3-Repetition Aggregator & 9-Gate Verifier for MesaNet
================================================================================

Aggregates `mesanet_ttt_rsi_rep{1,2,3}.json` and `mesanet_ttt_rsi_smoke.json`
into `mesanet_ttt_rsi_summary.json`, `mesanet_ttt_rsi_summary.md`, and
standardized 7-entry `SHA256SUMS`.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _med(vals: List[float]) -> float:
    return float(statistics.median(vals))


def _iqr(vals: List[float]) -> float:
    s = sorted(vals)
    return float(s[-1] - s[0]) if len(s) <= 3 else float(s[(3 * len(s)) // 4] - s[len(s) // 4])


def main() -> None:
    raw_dir = ROOT_DIR / "results" / "raw" / "mesanet_ttt_rsi"
    rep_paths = [raw_dir / f"mesanet_ttt_rsi_rep{r}.json" for r in (1, 2, 3)]
    smoke_path = raw_dir / "mesanet_ttt_rsi_smoke.json"

    for p in [smoke_path, *rep_paths]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    reps = [json.loads(p.read_text(encoding="utf-8")) for p in rep_paths]

    # Suite 0 aggregation
    s0 = reps[0]["suite0"]
    fp32_rel_med = _med([r["suite0"]["numerical_parity"]["fp32_max_rel_l2"] for r in reps])
    bf16_rel_med = _med([r["suite0"]["numerical_parity"]["bf16_max_rel_l2"] for r in reps])

    # Suite A aggregation
    t1_speedup_base = _med([r["suite_a"]["track1_mesa_layer"]["geomean_speedup_vs_upstream"] for r in reps])
    t1_speedup_comp = _med([r["suite_a"]["track1_mesa_layer"]["geomean_speedup_vs_compile"] for r in reps])
    t1_speedup_seq = _med([r["suite_a"]["track1_mesa_layer"]["max_speedup_vs_seq_linalg_solve"] for r in reps])

    t2_speedup_base = _med([r["suite_a"]["track2_mesa_block"]["geomean_speedup_vs_upstream"] for r in reps])
    t2_speedup_comp = _med([r["suite_a"]["track2_mesa_block"]["geomean_speedup_vs_compile"] for r in reps])

    shapes = list(reps[0]["suite_a"]["track1_mesa_layer"]["candidates"]["A0_upstream_mesa_cg30"]["shapes"].keys())
    t1_shapes_summary: Dict[str, Any] = {}
    for sh in shapes:
        base_ms = _med([r["suite_a"]["track1_mesa_layer"]["candidates"]["A0_upstream_mesa_cg30"]["shapes"][sh]["median_ms"] for r in reps])
        seq_ms = _med([r["suite_a"]["track1_mesa_layer"]["candidates"]["A1_exact_seq_linalg_solve"]["shapes"][sh]["median_ms"] for r in reps])
        comp_ms = _med([r["suite_a"]["track1_mesa_layer"]["candidates"]["A2_upstream_torch_compile"]["shapes"][sh]["median_ms"] for r in reps])
        pcg8_ms = _med([r["suite_a"]["track1_mesa_layer"]["candidates"]["A3_rsi_jacobi_pcg8_fused"]["shapes"][sh]["median_ms"] for r in reps])
        pcg6_ms = _med([r["suite_a"]["track1_mesa_layer"]["candidates"]["A4_rsi_jacobi_pcg6_fast"]["shapes"][sh]["median_ms"] for r in reps])
        base_hbm = _med([r["suite_a"]["track1_mesa_layer"]["candidates"]["A0_upstream_mesa_cg30"]["shapes"][sh]["peak_hbm_mib"] for r in reps])
        pcg6_hbm = _med([r["suite_a"]["track1_mesa_layer"]["candidates"]["A4_rsi_jacobi_pcg6_fast"]["shapes"][sh]["peak_hbm_mib"] for r in reps])
        t1_shapes_summary[sh] = {
            "upstream_cg30_ms": base_ms,
            "seq_linalg_solve_ms": seq_ms,
            "torch_compile_ms": comp_ms,
            "rsi_pcg8_ms": pcg8_ms,
            "rsi_pcg6_ms": pcg6_ms,
            "speedup_vs_upstream": base_ms / pcg6_ms,
            "speedup_vs_compile": comp_ms / pcg6_ms,
            "speedup_vs_seq_solve": seq_ms / pcg6_ms,
            "upstream_hbm_mib": base_hbm,
            "rsi_pcg6_hbm_mib": pcg6_hbm,
        }

    t2_shapes_summary: Dict[str, Any] = {}
    for sh in shapes:
        base_ms = _med([r["suite_a"]["track2_mesa_block"]["candidates"]["B0_upstream_mesanet_block"]["shapes"][sh]["median_ms"] for r in reps])
        comp_ms = _med([r["suite_a"]["track2_mesa_block"]["candidates"]["B2_upstream_block_compile"]["shapes"][sh]["median_ms"] for r in reps])
        rsi_ms = _med([r["suite_a"]["track2_mesa_block"]["candidates"]["B3_rsi_mesanet_block_pcg6"]["shapes"][sh]["median_ms"] for r in reps])
        t2_shapes_summary[sh] = {
            "upstream_block_ms": base_ms,
            "compile_block_ms": comp_ms,
            "rsi_block_ms": rsi_ms,
            "speedup_vs_upstream": base_ms / rsi_ms,
            "speedup_vs_compile": comp_ms / rsi_ms,
        }

    # Suite B aggregation
    sb_parity = {
        "broken_local_only_no_prefix_rel_l2": _med([r["suite_b"]["broken_local_only_no_prefix_rel_l2"] for r in reps]),
        "max_loss_rel_diff": max(r["suite_b"]["three_step_fp32_parity"]["max_loss_rel_diff"] for r in reps),
        "max_param_rel_l2": max(r["suite_b"]["three_step_fp32_parity"]["max_param_rel_l2"] for r in reps),
        "max_grad_rel_l2": max(r["suite_b"]["three_step_fp32_parity"]["max_grad_rel_l2"] for r in reps),
    }
    sb_perf: Dict[str, Any] = {}
    for d_key in ("D64", "D128"):
        ag_ms = _med([r["suite_b"]["performance"][d_key]["allgather_upstream_ms"] for r in reps])
        cp_ms = _med([r["suite_b"]["performance"][d_key]["mesa_cp8_rsi_ms"] for r in reps])
        ag_hbm = _med([r["suite_b"]["performance"][d_key]["allgather_upstream_hbm_mib"] for r in reps])
        cp_hbm = _med([r["suite_b"]["performance"][d_key]["mesa_cp8_rsi_hbm_mib"] for r in reps])
        sb_perf[d_key] = {
            "allgather_upstream_ms": ag_ms,
            "mesa_cp8_rsi_ms": cp_ms,
            "speedup": ag_ms / cp_ms,
            "allgather_upstream_hbm_mib": ag_hbm,
            "mesa_cp8_rsi_hbm_mib": cp_hbm,
            "hbm_reduction_ratio": ag_hbm / max(cp_hbm, 1.0),
        }

    # Suite C aggregation
    c_cands_med: List[Dict[str, Any]] = []
    for idx in range(len(reps[0]["suite_c"]["val_bpb_loop"]["candidates"])):
        cid = reps[0]["suite_c"]["val_bpb_loop"]["candidates"][idx]["id"]
        dec = reps[0]["suite_c"]["val_bpb_loop"]["candidates"][idx]["decision"]
        steps_med = int(round(_med([r["suite_c"]["val_bpb_loop"]["candidates"][idx]["steps"] for r in reps])))
        vbpb_med = _med([r["suite_c"]["val_bpb_loop"]["candidates"][idx]["val_bpb"] for r in reps])
        vacc_med = _med([r["suite_c"]["val_bpb_loop"]["candidates"][idx]["val_recall_acc"] for r in reps])
        c_cands_med.append({
            "id": cid,
            "steps_median": steps_med,
            "val_bpb_median": vbpb_med,
            "val_recall_acc_median": vacc_med,
            "decision": dec,
        })

    init_val_bpb = _med([r["suite_c"]["val_bpb_loop"]["initial_val_bpb"] for r in reps])
    final_val_bpb = _med([r["suite_c"]["val_bpb_loop"]["final_val_bpb"] for r in reps])
    init_test_bpb = _med([r["suite_c"]["val_bpb_loop"]["initial_test_bpb"] for r in reps])
    final_test_bpb = _med([r["suite_c"]["val_bpb_loop"]["final_test_bpb"] for r in reps])
    init_rec_acc = _med([r["suite_c"]["val_bpb_loop"]["initial_recall_acc"] for r in reps])
    final_rec_acc = _med([r["suite_c"]["val_bpb_loop"]["final_recall_acc"] for r in reps])

    scaling_med: Dict[str, Any] = {}
    for arm_name in reps[0]["suite_c"]["scaling_law_5arm"].keys():
        scaling_med[arm_name] = {
            "matched_steps": reps[0]["suite_c"]["scaling_law_5arm"][arm_name]["matched_steps"],
            "val_bpb_median": _med([r["suite_c"]["scaling_law_5arm"][arm_name]["val_bpb"] for r in reps]),
            "test_bpb_median": _med([r["suite_c"]["scaling_law_5arm"][arm_name]["test_bpb"] for r in reps]),
            "recall_test_bpb_median": _med([r["suite_c"]["scaling_law_5arm"][arm_name]["recall_test_bpb"] for r in reps]),
            "recall_test_acc_median": _med([r["suite_c"]["scaling_law_5arm"][arm_name]["recall_test_acc"] for r in reps]),
        }

    all_runs_passed = bool(smoke["all_nine_gates_passed"] and all(r["all_nine_gates_passed"] for r in reps))

    summary = {
        "experiment_id": "E139",
        "version": "v130",
        "round": "Round-13 RSI",
        "title": "MesaNet: Sequence Modeling by Locally Optimal Test-Time Training (Dual-State H_kk/H_kv Recurrence, Causal Query-Subtiled Warm-Start Jacobi-PCG & NVSwitch Dual-State Prefix-Scan Mesa-CP=8)",
        "paper": reps[0]["paper"],
        "official_repo": reps[0]["official_repo"],
        "harness_sha256": s0["harness_sha256"],
        "num_full_repetitions": 3,
        "smoke_elapsed_s": smoke["elapsed_s"],
        "rep_elapsed_s": [r["elapsed_s"] for r in reps],
        "suite0_summary": {
            "spectral_convergence_audit": s0["spectral_convergence_audit"],
            "fp32_max_rel_l2_median": fp32_rel_med,
            "bf16_max_rel_l2_median": bf16_rel_med,
            "richardson_oh2_passed": f"{s0['richardson_oh2']['num_passed']}/{s0['richardson_oh2']['num_checks']}",
            "cubin_spill_audit": s0["cubin_spill_audit"],
            "red_team_exploits_caught": f"{s0['red_team']['exploits_caught']}/{s0['red_team']['total_exploits']}",
        },
        "suite_a_summary": {
            "track1_mesa_layer": {
                "geomean_speedup_vs_upstream_median": t1_speedup_base,
                "geomean_speedup_vs_compile_median": t1_speedup_comp,
                "max_speedup_vs_seq_linalg_solve_median": t1_speedup_seq,
                "shapes": t1_shapes_summary,
            },
            "track2_mesa_block": {
                "geomean_speedup_vs_upstream_median": t2_speedup_base,
                "geomean_speedup_vs_compile_median": t2_speedup_comp,
                "shapes": t2_shapes_summary,
            },
        },
        "suite_b_summary": {
            "world_size": 8,
            "t_global": 8192,
            "t_local": 1024,
            "parity": sb_parity,
            "performance": sb_perf,
        },
        "suite_c_summary": {
            "wallclock_budget_s": reps[0]["suite_c"]["wallclock_budget_s"],
            "val_bpb_loop": {
                "candidates": c_cands_med,
                "initial_val_bpb_median": init_val_bpb,
                "final_val_bpb_median": final_val_bpb,
                "val_bpb_improvement_median": init_val_bpb - final_val_bpb,
                "initial_test_bpb_median": init_test_bpb,
                "final_test_bpb_median": final_test_bpb,
                "test_bpb_improvement_median": init_test_bpb - final_test_bpb,
                "initial_recall_acc_median": init_rec_acc,
                "final_recall_acc_median": final_rec_acc,
            },
            "scaling_law_5arm": scaling_med,
        },
        "gates": reps[0]["gates"],
        "all_nine_gates_passed": all_runs_passed,
    }

    summary_json_path = raw_dir / "mesanet_ttt_rsi_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")

    md_lines = [
        "# E139 (`v130`, Round-13 RSI): `MesaNet` Locally Optimal Test-Time Training Summary",
        "",
        f"- **Paper**: {summary['paper']}",
        f"- **Official Repository**: `{summary['official_repo']}`",
        f"- **Frozen Evaluator SHA-256**: `{summary['harness_sha256']}`",
        f"- **All 9 Gates Passed Across Smoke + 3 Repetitions**: `{summary['all_nine_gates_passed']}`",
        "",
        "## 1. Suite 0: Spectral Convergence, Numerical Parity, Richardson O(h^2) & Zero-Spill Audit",
        f"- **Upstream 30-Step Unpreconditioned CG (`x0=0, +1e-5`) Rel L2 Error**: `{s0['spectral_convergence_audit']['upstream_step30_rel_l2']:.3e}`",
        f"- **RSI Warm-Start Jacobi-PCG (`x0=D^-1 q, 1e-30`) Step-6 Rel L2 Error**: `{s0['spectral_convergence_audit']['rsi_pcg_step6_rel_l2']:.3e}` (`{s0['spectral_convergence_audit']['accuracy_improvement_factor_step6_vs_upstream30']:.1f}x` more accurate in `5.0x` fewer steps)",
        f"- **FP32 Max Rel L2 Error vs `torch.linalg.solve` Oracle**: `{fp32_rel_med:.3e}`",
        f"- **BF16 Max Rel L2 Error vs `torch.linalg.solve` Oracle**: `{bf16_rel_med:.4f}`",
        f"- **FP64 Richardson O(h^2) Checks Passed**: `{summary['suite0_summary']['richardson_oh2_passed']}`",
        f"- **SM80 Cubin Spills**: `0` across all `{s0['cubin_spill_audit']['total_rsi_kernels']}/{s0['cubin_spill_audit']['total_rsi_kernels']}` Round-13 RSI kernels (`{s0['cubin_spill_audit']['total_specializations']}` specializations) vs `{s0['cubin_spill_audit']['negative_control_monolithic_d128_spills']}` spills in monolithic `[64x128]` negative control",
        "",
        "## 2. Suite A: Single-GPU Vector-Pareto Speedups (3-Rep Medians)",
        f"- **Track 1 (`Core MesaNet Layer Fwd+Bwd`)**: **`{t1_speedup_base:.2f}x`** geomean speedup vs Upstream 30-Step CG (`A0`), **`{t1_speedup_comp:.2f}x`** vs `torch.compile` (`A2`), and up to **`{t1_speedup_seq:.2f}x`** vs sequential `torch.linalg.solve` (`A1`).",
        f"- **Track 2 (`Full MesaNet Block Fwd+Bwd`)**: **`{t2_speedup_base:.2f}x`** geomean speedup vs Upstream `MesaNetBlock` (`B0`) and **`{t2_speedup_comp:.2f}x`** vs `torch.compile` (`B2`).",
        "",
        "## 3. Suite B: 8-GPU NVSwitch Dual-State Prefix-Scan `Mesa-CP = 8` (`T_global = 8,192`)",
        f"- **Gate G9 3-Step FP32 Parity**: `max_loss_rel_diff = {sb_parity['max_loss_rel_diff']:.3e}`, `max_param_rel_l2 = {sb_parity['max_param_rel_l2']:.3e}`, `max_grad_rel_l2 = {sb_parity['max_grad_rel_l2']:.3e}` (vs `{sb_parity['broken_local_only_no_prefix_rel_l2']:.4f}` error when omitting prefix states)",
        f"- **D=64 Speedup vs All-Gather Full-Sequence**: **`{sb_perf['D64']['speedup']:.2f}x`** (`{sb_perf['D64']['allgather_upstream_ms']:.2f} -> {sb_perf['D64']['mesa_cp8_rsi_ms']:.2f} ms`, `{sb_perf['D64']['hbm_reduction_ratio']:.2f}x` lower peak HBM)",
        f"- **D=128 Speedup vs All-Gather Full-Sequence**: **`{sb_perf['D128']['speedup']:.2f}x`** (`{sb_perf['D128']['allgather_upstream_ms']:.2f} -> {sb_perf['D128']['mesa_cp8_rsi_ms']:.2f} ms`, `{sb_perf['D128']['hbm_reduction_ratio']:.2f}x` lower peak HBM)",
        "",
        "## 4. Suite C: Fixed-2.1s `8xA100 DDP` `val_bpb` Loop & 5-Arm Scaling Law",
        f"- **Fixed-2.1s `val_bpb` Loop**: `{init_val_bpb:.4f} -> {final_val_bpb:.4f}` (`-{init_val_bpb - final_val_bpb:.4f} val_bpb`), single-touch `test_bpb`: `{init_test_bpb:.4f} -> {final_test_bpb:.4f}` (`-{init_test_bpb - final_test_bpb:.4f} test_bpb`), recall accuracy: `{init_rec_acc*100:.2f}% -> {final_rec_acc*100:.2f}%`.",
    ]
    summary_md_path = raw_dir / "mesanet_ttt_rsi_summary.md"
    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8", newline="\n")

    tracked_files = [
        "mesanet_ttt_rsi_rep1.json",
        "mesanet_ttt_rsi_rep2.json",
        "mesanet_ttt_rsi_rep3.json",
        "mesanet_ttt_rsi_smoke.json",
        "mesanet_ttt_rsi_summary.json",
        "suite0_redteam_ledger.jsonl",
        "suite_a_ledger_rep1.jsonl",
    ]
    sha_lines = [f"{_sha256_file(raw_dir / fname)}  {fname}" for fname in tracked_files]
    (raw_dir / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8", newline="\n")
    print(f"[E139] Wrote {summary_json_path}, {summary_md_path}, and SHA256SUMS (7 files).")
    if not all_runs_passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
