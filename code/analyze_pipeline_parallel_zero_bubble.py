from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


REPETITION_FILES = (
    "rep1_pipeline_parallel_zero_bubble.json",
    "rep2_pipeline_parallel_zero_bubble.json",
    "rep3_pipeline_parallel_zero_bubble.json",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def relative_spread(values: list[float]) -> float:
    med = statistics.median(values)
    if med == 0.0:
        return 0.0
    return (max(values) - min(values)) / med


def summarize_pp1_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    lats = [float(e["median_ms"]) for e in entries]
    toks = [float(e["tokens_per_s"]) for e in entries]
    tfs = [float(e["achieved_tflops"]) for e in entries]
    incs = [float(e["activation_inc_mib"]) for e in entries]
    tots = [float(e["total_peak_allocated_mib"]) for e in entries]
    return {
        "median_ms": round(statistics.median(lats), 4),
        "repetition_medians_ms": lats,
        "repeat_spread": round(relative_spread(lats), 6),
        "tokens_per_s_median": round(statistics.median(toks), 2),
        "achieved_tflops_median": round(statistics.median(tfs), 2),
        "activation_inc_mib_median": round(statistics.median(incs), 2),
        "total_peak_allocated_mib_median": round(statistics.median(tots), 2),
        "loss": float(entries[0]["loss"]),
        "flops_breakdown": entries[0]["flops_breakdown"],
    }


def summarize_schedule_entries(
    entries: list[dict[str, Any]],
    *,
    pp1_median_ms: float | None = None,
) -> dict[str, Any]:
    e0 = entries[0]
    pp_size = int(e0["pp_size"])
    lats = [float(e["median_ms"]) for e in entries]
    rmeans = [float(e["rank_mean_median_ms"]) for e in entries]
    toks = [float(e["tokens_per_s"]) for e in entries]
    tfs = [float(e["achieved_tflops"]) for e in entries]
    ptfs = [float(e["per_GPU_tflops"]) for e in entries]
    max_incs = [float(e["max_rank_activation_inc_mib"]) for e in entries]
    min_incs = [float(e["min_rank_activation_inc_mib"]) for e in entries]
    imbs = [float(e["rank_mem_imbalance_ratio"]) for e in entries]
    max_tots = [float(e["max_rank_total_peak_mib"]) for e in entries]
    bases = [float(e["static_base_mib_rank0"]) for e in entries]

    per_rank_inc_med = [
        round(statistics.median([float(e["per_rank_activation_inc_mib"][r]) for e in entries]), 2)
        for r in range(pp_size)
    ]
    per_rank_tot_med = [
        round(statistics.median([float(e["per_rank_total_peak_mib"][r]) for e in entries]), 2)
        for r in range(pp_size)
    ]

    med_ms = statistics.median(lats)
    out: dict[str, Any] = {
        "schedule": e0["schedule"],
        "dw_backend": e0["dw_backend"],
        "ac_mode": e0["ac_mode"],
        "pp_size": pp_size,
        "num_stages": int(e0["num_stages"]),
        "virtual_stages_per_rank": int(e0["virtual_stages_per_rank"]),
        "n_microbatches": int(e0["n_microbatches"]),
        "microbatch_size": int(e0["microbatch_size"]),
        "seq_len": int(e0["seq_len"]),
        "global_tokens": int(e0["global_tokens"]),
        "median_ms": round(med_ms, 4),
        "rank_mean_median_ms": round(statistics.median(rmeans), 4),
        "repetition_medians_ms": lats,
        "repeat_spread": round(relative_spread(lats), 6),
        "tokens_per_s_median": round(statistics.median(toks), 2),
        "achieved_tflops_median": round(statistics.median(tfs), 2),
        "per_gpu_tflops_median": round(statistics.median(ptfs), 2),
        "static_base_mib_rank0_median": round(statistics.median(bases), 2),
        "per_rank_activation_inc_mib_median": per_rank_inc_med,
        "per_rank_total_peak_mib_median": per_rank_tot_med,
        "max_rank_activation_inc_mib_median": round(statistics.median(max_incs), 2),
        "min_rank_activation_inc_mib_median": round(statistics.median(min_incs), 2),
        "rank0_activation_inc_mib_median": per_rank_inc_med[0],
        "last_rank_activation_inc_mib_median": per_rank_inc_med[-1],
        "rank_mem_imbalance_ratio_median": round(statistics.median(imbs), 3),
        "max_rank_total_peak_mib_median": round(statistics.median(max_tots), 2),
        "cross_rank_stage_transitions": int(e0["cross_rank_stage_transitions"]),
        "total_p2p_payload_mib": float(e0["total_p2p_payload_mib"]),
        "rel_loss_err_max": round(max(float(e["rel_loss_err"]) for e in entries), 8),
        "max_dw_rel_l2_max": round(max(float(e["max_dw_rel_l2"]) for e in entries), 6),
        "chain_rule_rel_l2_max": round(max(float(e["chain_rule_rel_l2"]) for e in entries), 6),
    }
    if pp1_median_ms is not None and pp1_median_ms > 0.0:
        sp = pp1_median_ms / med_ms
        eff = sp / pp_size
        out["speedup_vs_pp1"] = round(sp, 4)
        out["strong_scaling_efficiency"] = round(eff, 4)
        out["measured_bubble_fraction"] = round(1.0 - eff, 4)
        out["theoretical_1f1b_bubble_fraction"] = round(
            (pp_size - 1.0) / (int(e0["n_microbatches"]) + pp_size - 1.0), 4
        )
    return out


def summarize_suite_0(reps: list[dict[str, Any]]) -> dict[str, Any]:
    s0_reps = [r["suite_0_audit_and_dw_backends"] for r in reps]
    aud_reps = [s["audits"] for s in s0_reps]
    seq_keys = sorted(aud_reps[0]["isolated_stage_timings"].keys(), key=lambda x: int(x))

    iso_summary: dict[str, Any] = {}
    for sk in seq_keys:
        matched = [a["isolated_stage_timings"][sk] for a in aud_reps]
        tf = statistics.median([float(m["t_F_ms"]) for m in matched])
        tbx = statistics.median([float(m["t_Bx_ms"]) for m in matched])
        tbw = statistics.median([float(m["t_Bw_ms"]) for m in matched])
        tbf = statistics.median([float(m["t_B_full_ms"]) for m in matched])
        iso_summary[sk] = {
            "seq_len": int(sk),
            "stage_layers": int(matched[0]["stage_layers"]),
            "t_F_ms_median": round(tf, 4),
            "t_Bx_ms_median": round(tbx, 4),
            "t_Bw_ms_median": round(tbw, 4),
            "t_Bx_plus_Bw_ms_median": round(tbx + tbw, 4),
            "t_B_full_ms_median": round(tbf, 4),
            "bx_to_f_ratio": round(tbx / max(tf, 1e-6), 4),
            "bw_to_f_ratio": round(tbw / max(tf, 1e-6), 4),
            "bx_to_bw_time_ratio": round(tbx / max(tbw, 1e-6), 4),
            "split_overhead_vs_full_bwd_ratio": round((tbx + tbw) / max(tbf, 1e-6), 4),
        }

    pp1_ref = summarize_pp1_entries([s["pp1_reference"] for s in s0_reps])
    b_keys = list(s0_reps[0]["backends"].keys())
    backends_summary: dict[str, Any] = {}
    for bk in b_keys:
        backends_summary[bk] = summarize_schedule_entries(
            [s["backends"][bk] for s in s0_reps],
            pp1_median_ms=pp1_ref["median_ms"],
        )

    comparisons = {
        "1f1b_module_compile_vs_eager_speedup": round(
            backends_summary["1f1b_eager"]["median_ms"]
            / backends_summary["1f1b_module_compile"]["median_ms"],
            4,
        ),
        "1f1b_compiled_ops_vs_eager_speedup": round(
            backends_summary["1f1b_eager"]["median_ms"]
            / backends_summary["1f1b_compiled_ops"]["median_ms"],
            4,
        ),
        "zb1p_custom_dw_eager_vs_autograd_split_speedup": round(
            backends_summary["zb1p_autograd_split"]["median_ms"]
            / backends_summary["zb1p_custom_dw_eager"]["median_ms"],
            4,
        ),
        "zb1p_custom_dw_compiled_vs_autograd_split_speedup": round(
            backends_summary["zb1p_autograd_split"]["median_ms"]
            / backends_summary["zb1p_custom_dw_compiled_ops"]["median_ms"],
            4,
        ),
        "zbv_custom_dw_eager_vs_autograd_split_speedup": round(
            backends_summary["zbv_autograd_split"]["median_ms"]
            / backends_summary["zbv_custom_dw_eager"]["median_ms"],
            4,
        ),
        "zbv_custom_dw_compiled_vs_autograd_split_speedup": round(
            backends_summary["zbv_autograd_split"]["median_ms"]
            / backends_summary["zbv_custom_dw_compiled_ops"]["median_ms"],
            4,
        ),
        "zbv_custom_dw_compiled_memory_saving_vs_autograd_split": round(
            backends_summary["zbv_autograd_split"]["max_rank_activation_inc_mib_median"]
            / max(backends_summary["zbv_custom_dw_compiled_ops"]["max_rank_activation_inc_mib_median"], 1e-6),
            4,
        ),
        "dualpipe_v_custom_dw_eager_vs_autograd_split_speedup": round(
            backends_summary["dualpipe_v_autograd_split"]["median_ms"]
            / backends_summary["dualpipe_v_custom_dw_eager"]["median_ms"],
            4,
        ),
        "dualpipe_v_custom_dw_compiled_vs_autograd_split_speedup": round(
            backends_summary["dualpipe_v_autograd_split"]["median_ms"]
            / backends_summary["dualpipe_v_custom_dw_compiled_ops"]["median_ms"],
            4,
        ),
    }

    return {
        "seq_len": int(s0_reps[0]["seq_len"]),
        "n_microbatches": int(s0_reps[0]["n_microbatches"]),
        "audits": {
            "optimized_module_rejected_in_zbv_all": all(
                bool(a["optimized_module_rejected_in_zbv"]) for a in aud_reps
            ),
            "optimized_module_rejected_in_dualpipe_v_all": all(
                bool(a["optimized_module_rejected_in_dualpipe_v"]) for a in aud_reps
            ),
            "optimized_module_error_excerpt": aud_reps[0]["optimized_module_error_excerpt"],
            "dualpipe_v_requires_m_ge_2pp_all": all(
                bool(a["dualpipe_v_requires_m_ge_2pp"]) for a in aud_reps
            ),
            "dynamic_shape_inference_phantom_dw_entries_all": [
                int(a["dynamic_shape_inference_phantom_dw_entries"]) for a in aud_reps
            ],
            "dynamic_shape_inference_dw_rel_l2_min": min(
                float(a["dynamic_shape_inference_dw_rel_l2"]) for a in aud_reps
            ),
            "static_shape_args_extra_dw_entries_max": max(
                int(a["static_shape_args_extra_dw_entries"]) for a in aud_reps
            ),
            "static_shape_args_dw_rel_l2_max": max(
                float(a["static_shape_args_dw_rel_l2"]) for a in aud_reps
            ),
            "isolated_stage_timings": iso_summary,
        },
        "pp1_reference": pp1_ref,
        "backends": backends_summary,
        "comparisons": comparisons,
    }


def summarize_suite_a(reps: list[dict[str, Any]]) -> dict[str, Any]:
    sa_reps = [r["suite_a_bubble_and_memory_sweep"] for r in reps]
    mb_keys = sorted(sa_reps[0]["pp1"].keys(), key=lambda x: int(x))

    pp1_map: dict[str, Any] = {}
    for mbk in mb_keys:
        pp1_map[mbk] = summarize_pp1_entries([sa["pp1"][mbk] for sa in sa_reps])

    out: dict[str, Any] = {"pp1": pp1_map, "pp4": {}, "pp8": {}}
    for ppk in ("pp4", "pp8"):
        for mbk in mb_keys:
            pp1_ms = pp1_map[mbk]["median_ms"]
            sched_keys = list(sa_reps[0][ppk][mbk]["modes"].keys())
            modes_map: dict[str, Any] = {}
            for sk in sched_keys:
                modes_map[sk] = summarize_schedule_entries(
                    [sa[ppk][mbk]["modes"][sk] for sa in sa_reps],
                    pp1_median_ms=pp1_ms,
                )
            f1b_ms = modes_map["1f1b"]["median_ms"]
            for sk, mentry in modes_map.items():
                mentry["speedup_vs_1f1b"] = round(f1b_ms / mentry["median_ms"], 4)
            out[ppk][mbk] = {
                "n_microbatches": int(mbk),
                "pp1_median_ms": pp1_ms,
                "modes": modes_map,
            }
    return out


def summarize_suite_b(reps: list[dict[str, Any]]) -> dict[str, Any]:
    sb_reps = [r["suite_b_seq_skew_frontier"] for r in reps]
    seq_keys = sorted(sb_reps[0].keys(), key=lambda x: int(x))
    out: dict[str, Any] = {}
    for sk in seq_keys:
        pp1_entry = summarize_pp1_entries([sb[sk]["pp1"] for sb in sb_reps])
        pp1_ms = pp1_entry["median_ms"]
        sched_keys = list(sb_reps[0][sk]["modes"].keys())
        modes_map: dict[str, Any] = {}
        for mkey in sched_keys:
            modes_map[mkey] = summarize_schedule_entries(
                [sb[sk]["modes"][mkey] for sb in sb_reps],
                pp1_median_ms=pp1_ms,
            )
        f1b_ms = modes_map["1f1b"]["median_ms"]
        for mentry in modes_map.values():
            mentry["speedup_vs_1f1b"] = round(f1b_ms / mentry["median_ms"], 4)
        out[sk] = {
            "seq_len": int(sk),
            "pp1": pp1_entry,
            "zbv_over_1f1b_speedup": modes_map["zbv"]["speedup_vs_1f1b"],
            "dualpipe_v_over_1f1b_speedup": modes_map["dualpipe_v"]["speedup_vs_1f1b"],
            "zb1p_over_1f1b_speedup": modes_map["zb1p"]["speedup_vs_1f1b"],
            "interleaved_1f1b_over_1f1b_speedup": modes_map["interleaved_1f1b"]["speedup_vs_1f1b"],
            "modes": modes_map,
        }
    return out


def summarize_suite_c(reps: list[dict[str, Any]]) -> dict[str, Any]:
    sc_reps = [r["suite_c_selective_checkpointing"] for r in reps]
    pp1_entry = summarize_pp1_entries([sc["pp1_reference"] for sc in sc_reps])
    pp1_ms = pp1_entry["median_ms"]
    sched_keys = list(sc_reps[0]["schedules"].keys())
    ac_keys = ("no_ac", "sac_nonlin_only", "sac_mlp_recomp", "full_ac")

    sched_out: dict[str, Any] = {}
    for sk in sched_keys:
        ac_map: dict[str, Any] = {}
        for ack in ac_keys:
            ac_map[ack] = summarize_schedule_entries(
                [sc["schedules"][sk][ack] for sc in sc_reps],
                pp1_median_ms=pp1_ms,
            )
        no_ac_ms = ac_map["no_ac"]["median_ms"]
        no_ac_mib = ac_map["no_ac"]["max_rank_activation_inc_mib_median"]
        for ack, entry in ac_map.items():
            entry["overhead_ratio_vs_no_ac"] = round(entry["median_ms"] / no_ac_ms, 4)
            entry["activation_memory_reduction_vs_no_ac"] = round(
                no_ac_mib / max(entry["max_rank_activation_inc_mib_median"], 1e-6), 4
            )
        sched_out[sk] = ac_map

    return {
        "n_microbatches": int(sc_reps[0]["n_microbatches"]),
        "seq_len": int(sc_reps[0]["seq_len"]),
        "pp1_reference": pp1_entry,
        "schedules": sched_out,
    }


def collect_all_schedule_records(rep: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(rep["suite_0_audit_and_dw_backends"]["backends"].values())
    for ppk in ("pp4", "pp8"):
        for mbv in rep["suite_a_bubble_and_memory_sweep"][ppk].values():
            records.extend(mbv["modes"].values())
    for slv in rep["suite_b_seq_skew_frontier"].values():
        records.extend(slv["modes"].values())
    for acd in rep["suite_c_selective_checkpointing"]["schedules"].values():
        records.extend(acd.values())
    return records


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = args.root / args.input_dir
    smoke = load_json(input_dir / "smoke_pipeline_parallel_zero_bubble.json")
    reps = [load_json(input_dir / name) for name in REPETITION_FILES]
    e7_original = load_json(args.root / "results/raw/full/transformer_training_full.json")

    suite_0 = summarize_suite_0(reps)
    suite_a = summarize_suite_a(reps)
    suite_b = summarize_suite_b(reps)
    suite_c = summarize_suite_c(reps)

    e7_env = e7_original["environment"]
    env_fields = (
        "gpu_name",
        "gpu_memory_bytes",
        "compute_capability",
        "torch",
        "torch_cuda",
        "triton",
        "flash_attn",
        "cuda_driver",
    )
    env_parity = all(
        rep["environment"].get(k) == e7_env.get(k)
        for rep in reps
        for k in env_fields
    )

    pre_clocks = [
        int(entry["sm_clock_mhz"])
        for rep in reps
        for entry in rep["telemetry"]["pre"]
    ]
    post_clocks = [
        int(entry["sm_clock_mhz"])
        for rep in reps
        for entry in rep["telemetry"]["post"]
    ]
    post_powers = [
        float(entry["power_w"])
        for rep in reps
        for entry in rep["telemetry"]["post"]
    ]
    clocks_locked_1410 = (
        all(c == 1410 for c in pre_clocks)
        and min(post_clocks) >= 1245
        and statistics.median(post_clocks) >= 1350
    )

    all_records = [r for rep in reps for r in collect_all_schedule_records(rep)]
    smoke_records = collect_all_schedule_records(smoke)

    max_loss_err = max(float(r["rel_loss_err"]) for r in all_records)
    max_dw_err = max(float(r["max_dw_rel_l2"]) for r in all_records)
    max_cr_err = max(float(r["chain_rule_rel_l2"]) for r in all_records)
    smoke_max_loss_err = max(float(r["rel_loss_err"]) for r in smoke_records)
    smoke_max_dw_err = max(float(r["max_dw_rel_l2"]) for r in smoke_records)

    aud = suite_0["audits"]
    comp = suite_0["comparisons"]
    pytorch_pipelining_audit_verified = (
        aud["optimized_module_rejected_in_zbv_all"]
        and aud["optimized_module_rejected_in_dualpipe_v_all"]
        and aud["dualpipe_v_requires_m_ge_2pp_all"]
        and aud["dynamic_shape_inference_phantom_dw_entries_all"] == [1, 1, 1]
        and aud["dynamic_shape_inference_dw_rel_l2_min"] > 1.0
        and aud["static_shape_args_extra_dw_entries_max"] == 0
        and aud["static_shape_args_dw_rel_l2_max"] < 1e-5
        and comp["zb1p_custom_dw_compiled_vs_autograd_split_speedup"] > 1.03
        and comp["zbv_custom_dw_compiled_vs_autograd_split_speedup"] > 1.03
        and comp["dualpipe_v_custom_dw_compiled_vs_autograd_split_speedup"] > 1.03
        and comp["zbv_custom_dw_compiled_memory_saving_vs_autograd_split"] > 1.10
    )

    sa_pp8_32 = suite_a["pp8"]["32"]["modes"]
    gpipe_vs_1f1b_vs_v_memory_law_verified = (
        sa_pp8_32["gpipe"]["max_rank_activation_inc_mib_median"]
        > 2.5 * sa_pp8_32["1f1b"]["max_rank_activation_inc_mib_median"]
        and sa_pp8_32["1f1b"]["rank_mem_imbalance_ratio_median"] > 3.5
        and sa_pp8_32["zbv"]["rank_mem_imbalance_ratio_median"] < 1.50
        and sa_pp8_32["dualpipe_v"]["rank_mem_imbalance_ratio_median"] < 1.50
    )

    sa_pp8_16 = suite_a["pp8"]["16"]["modes"]
    sa_pp4_8 = suite_a["pp4"]["8"]["modes"]
    zero_bubble_and_dualpipe_speedup_verified = (
        sa_pp8_16["zb1p"]["speedup_vs_1f1b"] > 1.05
        and sa_pp8_16["zbv"]["speedup_vs_1f1b"] > 1.05
        and sa_pp8_16["dualpipe_v"]["speedup_vs_1f1b"] > 1.05
        and sa_pp4_8["zb1p"]["speedup_vs_1f1b"] > 1.05
        and sa_pp4_8["zbv"]["speedup_vs_1f1b"] > 1.05
        and sa_pp4_8["dualpipe_v"]["speedup_vs_1f1b"] > 1.05
    )

    iso = aud["isolated_stage_timings"]
    bx_bw_ratios = [iso[str(s)]["bx_to_bw_time_ratio"] for s in (1024, 2048, 4096, 8192)]
    sequence_length_bx_skew_verified = (
        bx_bw_ratios[0] < bx_bw_ratios[1] < bx_bw_ratios[2] < bx_bw_ratios[3]
    )

    sc_scheds = suite_c["schedules"]
    selective_checkpointing_frontier_verified = all(
        sc_scheds[sk]["no_ac"]["max_rank_activation_inc_mib_median"]
        * 1.03
        >= sc_scheds[sk]["sac_nonlin_only"]["max_rank_activation_inc_mib_median"]
        > sc_scheds[sk]["sac_mlp_recomp"]["max_rank_activation_inc_mib_median"]
        > sc_scheds[sk]["full_ac"]["max_rank_activation_inc_mib_median"]
        for sk in ("gpipe", "1f1b", "dualpipe_v")
    ) and (
        sc_scheds["1f1b"]["sac_mlp_recomp"]["activation_memory_reduction_vs_no_ac"] >= 1.8
        and sc_scheds["gpipe"]["sac_mlp_recomp"]["activation_memory_reduction_vs_no_ac"] >= 1.8
    )

    numerical_loss_and_dw_accuracy_passed = (
        max_loss_err <= 0.002
        and max_dw_err <= 0.025
        and smoke_max_loss_err <= 0.002
        and smoke_max_dw_err <= 0.025
    )
    chain_rule_passed = max_cr_err <= 0.020

    gates = {
        "env_parity": env_parity,
        "clocks_locked_1410": clocks_locked_1410,
        "pytorch_pipelining_audit_verified": pytorch_pipelining_audit_verified,
        "gpipe_vs_1f1b_vs_v_memory_law_verified": gpipe_vs_1f1b_vs_v_memory_law_verified,
        "zero_bubble_and_dualpipe_speedup_verified": zero_bubble_and_dualpipe_speedup_verified,
        "sequence_length_bx_skew_verified": sequence_length_bx_skew_verified,
        "selective_checkpointing_frontier_verified": selective_checkpointing_frontier_verified,
        "numerical_loss_and_dw_accuracy_passed": numerical_loss_and_dw_accuracy_passed,
        "chain_rule_passed": chain_rule_passed,
    }
    all_gates_passed = all(gates.values())

    return {
        "experiment": "E116_pipeline_parallel_zero_bubble_sac",
        "repetitions": len(reps),
        "environment": reps[0]["environment"],
        "model_spec": reps[0]["model_spec"],
        "telemetry_summary": {
            "pre_sm_clock_mhz_min": min(pre_clocks),
            "pre_sm_clock_mhz_max": max(pre_clocks),
            "post_sm_clock_mhz_min": min(post_clocks),
            "post_sm_clock_mhz_median": statistics.median(post_clocks),
            "post_sm_clock_mhz_max": max(post_clocks),
            "post_power_w_max": round(max(post_powers), 2),
            "repetition_wall_seconds": [float(r["wall_seconds"]) for r in reps],
        },
        "validation_gates": {
            **gates,
            "all_gates_passed": all_gates_passed,
            "max_rel_loss_err": round(max_loss_err, 8),
            "max_dw_rel_l2": round(max_dw_err, 6),
            "max_chain_rule_rel_l2": round(max_cr_err, 6),
            "smoke_max_rel_loss_err": round(smoke_max_loss_err, 8),
            "smoke_max_dw_rel_l2": round(smoke_max_dw_err, 6),
        },
        "suite_0_audit_and_dw_backends": suite_0,
        "suite_a_bubble_and_memory_sweep": suite_a,
        "suite_b_seq_skew_frontier": suite_b,
        "suite_c_selective_checkpointing": suite_c,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze E116 3-repetition Pipeline Parallelism, Zero-Bubble, DualPipeV, and SAC results"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw/pipeline_parallel_zero_bubble"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/raw/pipeline_parallel_zero_bubble/pipeline_parallel_zero_bubble_full.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_analysis(args)
    out_path = args.root / args.output
    write_json(out_path, payload)
    all_ok = payload["validation_gates"]["all_gates_passed"]
    print(f"Wrote {out_path} (all_gates_passed={all_ok})")
    if not all_ok:
        failed = [k for k, v in payload["validation_gates"].items() if v is False]
        raise SystemExit(f"Validation gates failed: {failed}")


if __name__ == "__main__":
    main()
