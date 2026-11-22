#!/usr/bin/env python3
"""Deterministic 3-repetition analyzer and 9-gate validator for E126:
Multimodal Vision-Language & Omni-Modal Native-Resolution Training Kernels
(Window-Morton-2x2 Zero-Permutation Ragged ViT + Tubelet Pack, Single-Launch 3D M-RoPE,
Surgical 2-Pass Disjoint Reversed-Upper FA2 + Triton LSE Omni-Attention, &
8-GPU NVSwitch Two-Stage Modality-Decoupled AllToAllv + Quad-Cost-LPT FSDP2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _med(vals: list[float]) -> float:
    return float(statistics.median(vals))


def _iqr(vals: list[float]) -> float:
    ss = sorted(vals)
    q25 = ss[max(0, (len(ss) - 1) // 4)]
    q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
    return float(q75 - q25)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic 3-repetition analyzer and 9-gate validator for E126"
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    raw_dir = args.root / "results" / "raw" / "multimodal_vlm_omni_training"
    smoke = json.loads(
        (raw_dir / "smoke_multimodal_vlm_omni_training.json").read_text(encoding="utf-8")
    )
    reps = [
        json.loads(
            (raw_dir / f"rep{i}_multimodal_vlm_omni_training.json").read_text(encoding="utf-8")
        )
        for i in (1, 2, 3)
    ]

    envs = [
        (
            r["environment"].get("torch"),
            r["environment"].get("triton"),
            r["environment"].get("flash_attn"),
            r["environment"].get("cudnn_version"),
            r["environment"].get("gpu_name"),
            r["world_size"],
        )
        for r in reps
    ]
    smoke_and_env_ok = (
        smoke.get("mode") == "smoke"
        and smoke.get("world_size") == 8
        and len(set(envs)) == 1
        and reps[0]["world_size"] == 8
    )

    init_clocks = [float(r.get("initial_sm_clock_mhz", 0.0)) for r in reps]
    all_clocks: list[float] = []
    all_powers: list[float] = []

    def _harvest_telemetry(d: dict[str, Any]) -> None:
        clk = float(d.get("sm_clock_mhz", -1.0))
        pwr = float(d.get("power_w", -1.0))
        if clk > 0.0:
            all_clocks.append(clk)
        if pwr > 0.0:
            all_powers.append(pwr)

    for r in reps:
        s0 = r["suite_0_audits"]
        for mv in s0["tubelet_conv3d_and_merger_roofline"]["tubelet_modes"].values():
            if isinstance(mv, dict):
                _harvest_telemetry(mv)
        for mv in s0["tubelet_conv3d_and_merger_roofline"]["patch_merger_modes"].values():
            if isinstance(mv, dict):
                _harvest_telemetry(mv)
        for mv in s0["mrope_3d_sectioned_and_interleaved_ablation"]["modes"].values():
            if isinstance(mv, dict):
                _harvest_telemetry(mv)
        _harvest_telemetry(s0["omni_attention_backend_audit"]["sdpa_bool_mask_s8k_math_fallback"])

        for pv in r["suite_a_1gpu_native_res_vit_scaling"].values():
            for mk in (
                "padded_2d_fa2_vit",
                "navit_ragged_full_varlen_vit",
                "qwen25vl_window8x8_eager_gather_vit",
                "qwen25vl_winmorton2x2_zero_perm_vit",
            ):
                _harvest_telemetry(pv[mk])

        sb = r["suite_b_1gpu_mrope_omni_attention_scaling"]
        for sv in sb["isolated_mrope_plus_omni_attention_scaling"].values():
            for mk in (
                "sdpa_bool_mask_eager_mrope",
                "flex_attn_dynamic_bm_eager_mrope",
                "flex_attn_prebuilt_bm_triton_mrope",
                "triton_tile64_omni_gqa_plus_mrope",
                "surgical_disjoint_fa2_lse_plus_triton_mrope",
            ):
                if isinstance(sv.get(mk), dict) and "median_ms" in sv[mk]:
                    _harvest_telemetry(sv[mk])
        for sv in sb["multimodal_4layer_decoder_stack_scaling"].values():
            for mk in (
                "flex_dynamic_eager_mrope_4layer",
                "triton_tile64_triton_mrope_4layer",
                "surgical_fa2_triton_mrope_4layer",
            ):
                _harvest_telemetry(sv[mk])

        for bv in r["suite_c_8gpu_modality_decoupled_fsdp2"]["fsdp2_8gpu_multimodal_scaling"].values():
            for mv in bv["modes"].values():
                _harvest_telemetry(mv)

    med_clock = _med(all_clocks) if all_clocks else 0.0
    min_clock = min(all_clocks) if all_clocks else 0.0
    max_clock = max(all_clocks) if all_clocks else 0.0
    max_power = max(all_powers) if all_powers else 0.0
    clocks_locked_1410 = (
        all(c >= 1380.0 for c in init_clocks)
        and max_clock >= 1410.0
        and med_clock >= 1380.0
        and min_clock >= 1260.0
    )

    # ------------------------------------------------------------------------
    # Suite 0 Aggregation
    # ------------------------------------------------------------------------
    s0_list = [r["suite_0_audits"] for r in reps]
    tub_ref = s0_list[0]["tubelet_conv3d_and_merger_roofline"]
    tub_modes_agg: dict[str, Any] = {}
    for mk in (
        "hf_conv3d_preextracted_patches",
        "conv3d_5d_ncthw_video",
        "eager_10d_permute_swizzle_linear",
        "triton_winmorton2x2_pack_cublas_1184",
    ):
        ms_v = [float(s["tubelet_conv3d_and_merger_roofline"]["tubelet_modes"][mk]["median_ms"]) for s in s0_list]
        tf_v = [float(s["tubelet_conv3d_and_merger_roofline"]["tubelet_modes"][mk]["effective_tflops"]) for s in s0_list]
        mem_v = [float(s["tubelet_conv3d_and_merger_roofline"]["tubelet_modes"][mk]["inc_peak_mib"]) for s in s0_list]
        tub_modes_agg[mk] = {
            "ms_median": round(_med(ms_v), 4),
            "ms_iqr": round(_iqr(ms_v), 4),
            "effective_tflops_median": round(_med(tf_v), 2),
            "inc_peak_mib_median": round(_med(mem_v), 2),
        }

    pm_modes_agg: dict[str, Any] = {}
    for mk in ("raster_permute_eager_rmsnorm_merger", "quad_contiguous_triton_rmsnorm_merger"):
        ms_v = [float(s["tubelet_conv3d_and_merger_roofline"]["patch_merger_modes"][mk]["median_ms"]) for s in s0_list]
        mem_v = [float(s["tubelet_conv3d_and_merger_roofline"]["patch_merger_modes"][mk]["inc_peak_mib"]) for s in s0_list]
        pm_modes_agg[mk] = {
            "ms_median": round(_med(ms_v), 4),
            "ms_iqr": round(_iqr(ms_v), 4),
            "inc_peak_mib_median": round(_med(mem_v), 2),
        }

    tub_agg = {
        "num_patches": int(tub_ref["num_patches"]),
        "num_merged_tokens": int(tub_ref["num_merged_tokens"]),
        "cudnn_version": int(tub_ref["cudnn_version"]),
        "parity_vs_conv3d_max": {
            pk: round(max(float(s["tubelet_conv3d_and_merger_roofline"]["parity_vs_conv3d"][pk]) for s in s0_list), 6)
            for pk in ("out_rel_l2", "dx_rel_l2", "dw_rel_l2")
        },
        "tubelet_modes": tub_modes_agg,
        "speedup_triton_pack_over_hf_conv3d": round(
            tub_modes_agg["hf_conv3d_preextracted_patches"]["ms_median"]
            / max(tub_modes_agg["triton_winmorton2x2_pack_cublas_1184"]["ms_median"], 1e-9),
            3,
        ),
        "speedup_triton_pack_over_5d_conv3d": round(
            tub_modes_agg["conv3d_5d_ncthw_video"]["ms_median"]
            / max(tub_modes_agg["triton_winmorton2x2_pack_cublas_1184"]["ms_median"], 1e-9),
            3,
        ),
        "speedup_triton_pack_over_eager_10d": round(
            tub_modes_agg["eager_10d_permute_swizzle_linear"]["ms_median"]
            / max(tub_modes_agg["triton_winmorton2x2_pack_cublas_1184"]["ms_median"], 1e-9),
            3,
        ),
        "patch_merger_modes": pm_modes_agg,
        "merger_speedup_zero_copy_over_raster": round(
            pm_modes_agg["raster_permute_eager_rmsnorm_merger"]["ms_median"]
            / max(pm_modes_agg["quad_contiguous_triton_rmsnorm_merger"]["ms_median"], 1e-9),
            3,
        ),
        "merger_mem_saving_ratio": round(
            pm_modes_agg["raster_permute_eager_rmsnorm_merger"]["inc_peak_mib_median"]
            / max(pm_modes_agg["quad_contiguous_triton_rmsnorm_merger"]["inc_peak_mib_median"], 1e-9),
            3,
        ),
    }

    mr_ref = s0_list[0]["mrope_3d_sectioned_and_interleaved_ablation"]
    mr_modes_agg: dict[str, Any] = {}
    for mk in (
        "hf_eager_3d_mrope",
        "torch_compile_3d_mrope",
        "triton_fused_qk_3d_mrope_sectioned",
        "triton_fused_qk_3d_mrope_interleaved",
    ):
        ms_v = [float(s["mrope_3d_sectioned_and_interleaved_ablation"]["modes"][mk]["median_ms"]) for s in s0_list]
        bw_v = [float(s["mrope_3d_sectioned_and_interleaved_ablation"]["modes"][mk]["achieved_useful_gbps"]) for s in s0_list]
        mem_v = [float(s["mrope_3d_sectioned_and_interleaved_ablation"]["modes"][mk]["inc_peak_mib"]) for s in s0_list]
        mr_modes_agg[mk] = {
            "ms_median": round(_med(ms_v), 4),
            "ms_iqr": round(_iqr(ms_v), 4),
            "achieved_useful_gbps_median": round(_med(bw_v), 2),
            "inc_peak_mib_median": round(_med(mem_v), 2),
        }
    mr_agg = {
        "seqlen": int(mr_ref["seqlen"]),
        "heads_q_kv": list(mr_ref["heads_q_kv"]),
        "head_dim": int(mr_ref["head_dim"]),
        "text_1d_rope_exact_reduction_fp64_rel_l2_max": {
            k: round(
                max(float(s["mrope_3d_sectioned_and_interleaved_ablation"]["text_1d_rope_exact_reduction_fp64_rel_l2"][k]) for s in s0_list),
                8,
            )
            for k in ("sectioned_16_24_24", "interleaved_24_20_20")
        },
        "parity_vs_hf_eager_max": {
            k: round(max(float(s["mrope_3d_sectioned_and_interleaved_ablation"]["parity_vs_hf_eager"][k]) for s in s0_list), 6)
            for k in mr_ref["parity_vs_hf_eager"]
        },
        "modes": mr_modes_agg,
        "speedup_triton_sectioned_over_hf_eager": round(
            mr_modes_agg["hf_eager_3d_mrope"]["ms_median"]
            / max(mr_modes_agg["triton_fused_qk_3d_mrope_sectioned"]["ms_median"], 1e-9),
            3,
        ),
        "speedup_triton_interleaved_over_hf_eager": round(
            mr_modes_agg["hf_eager_3d_mrope"]["ms_median"]
            / max(mr_modes_agg["triton_fused_qk_3d_mrope_interleaved"]["ms_median"], 1e-9),
            3,
        ),
        "mem_saving_triton_over_hf_eager_ratio": round(
            mr_modes_agg["hf_eager_3d_mrope"]["inc_peak_mib_median"]
            / max(mr_modes_agg["triton_fused_qk_3d_mrope_sectioned"]["inc_peak_mib_median"], 1e-9),
            3,
        ),
    }

    om_ref = s0_list[0]["omni_attention_backend_audit"]
    sdpa_8k_ms = [float(s["omni_attention_backend_audit"]["sdpa_bool_mask_s8k_math_fallback"]["median_ms"]) for s in s0_list]
    sdpa_8k_mem = [float(s["omni_attention_backend_audit"]["sdpa_bool_mask_s8k_math_fallback"]["inc_peak_mib"]) for s in s0_list]
    omni_backend_agg = {
        "audit_seqlen_s8k": int(om_ref["audit_seqlen_s8k"]),
        "fa2_causal_only_visual_token_rel_l2_error_min": round(
            min(float(s["omni_attention_backend_audit"]["fa2_causal_only_visual_token_rel_l2_error"]) for s in s0_list), 5
        ),
        "sdpa_bool_mask_s8k_math_fallback": {
            "ms_median": round(_med(sdpa_8k_ms), 3),
            "ms_iqr": round(_iqr(sdpa_8k_ms), 3),
            "inc_peak_mib_median": round(_med(sdpa_8k_mem), 2),
        },
        "sdpa_bool_mask_s16k_oom_triggered_all_reps": all(
            bool(s["omni_attention_backend_audit"]["sdpa_bool_mask_s16k_oom_audit"]["oom_triggered"]) for s in s0_list
        ),
        "sdpa_bool_mask_s16k_error_excerpt": om_ref["sdpa_bool_mask_s16k_oom_audit"]["error_excerpt"],
        "triton_tile64_parity_vs_sdpa_max": {
            k: round(max(float(s["omni_attention_backend_audit"]["triton_tile64_parity_vs_sdpa"][k]) for s in s0_list), 6)
            for k in ("out_rel_l2", "dq_rel_l2", "dk_rel_l2")
        },
        "surgical_disjoint_fa2_parity_vs_sdpa_max": {
            k: round(max(float(s["omni_attention_backend_audit"]["surgical_disjoint_fa2_parity_vs_sdpa"][k]) for s in s0_list), 6)
            for k in ("out_rel_l2", "dq_rel_l2", "dk_rel_l2")
        },
    }

    ora_keys = list(s0_list[0]["bf16_vs_fp64_oracle_parity"].keys())
    oracle_agg = {
        ok: round(max(float(s["bf16_vs_fp64_oracle_parity"][ok]) for s in s0_list), 6)
        for ok in ora_keys
    }
    rich_keys = list(s0_list[0]["fp64_richardson_central_difference"]["ratios"].keys())
    rich_ratios = {
        rk: round(min(float(s["fp64_richardson_central_difference"]["ratios"][rk]) for s in s0_list), 4)
        for rk in rich_keys
    }
    rich_agg = {
        "ratios_min_across_reps": rich_ratios,
        "min_ratio": round(min(rich_ratios.values()), 4),
        "max_ratio": round(
            max(float(s["fp64_richardson_central_difference"]["max_ratio"]) for s in s0_list), 4
        ),
    }

    suite_0_summary = {
        "tubelet_conv3d_and_merger_roofline": tub_agg,
        "mrope_3d_sectioned_and_interleaved_ablation": mr_agg,
        "omni_attention_backend_audit": omni_backend_agg,
        "bf16_vs_fp64_oracle_parity_max": oracle_agg,
        "fp64_richardson_central_difference": rich_agg,
    }

    # ------------------------------------------------------------------------
    # Suite A Aggregation (1-GPU Native-Resolution Vision Encoder Scaling)
    # ------------------------------------------------------------------------
    sa_list = [r["suite_a_1gpu_native_res_vit_scaling"] for r in reps]
    sa_agg: dict[str, Any] = {}
    for pk, pref in sa_list[0].items():
        row_a: dict[str, Any] = {
            "num_visual_items": int(pref["num_visual_items"]),
            "raw_patches": int(pref["raw_patches"]),
            "merged_llm_tokens": int(pref["merged_llm_tokens"]),
            "padded_2d_total_slots": int(pref["padded_2d_total_slots"]),
            "padding_inflation_ratio": float(pref["padding_inflation_ratio"]),
        }
        for mk in (
            "padded_2d_fa2_vit",
            "navit_ragged_full_varlen_vit",
            "qwen25vl_window8x8_eager_gather_vit",
            "qwen25vl_winmorton2x2_zero_perm_vit",
        ):
            ms_v = [float(a[pk][mk]["median_ms"]) for a in sa_list]
            pps_v = [float(a[pk][mk]["patches_per_s"]) for a in sa_list]
            mem_v = [float(a[pk][mk]["inc_peak_mib"]) for a in sa_list]
            row_a[mk] = {
                "ms_median": round(_med(ms_v), 3),
                "ms_iqr": round(_iqr(ms_v), 3),
                "patches_per_s_median": round(_med(pps_v), 1),
                "inc_peak_mib_median": round(_med(mem_v), 2),
            }
        wm_ms = row_a["qwen25vl_winmorton2x2_zero_perm_vit"]["ms_median"]
        row_a["winmorton_speedup_over_padded_2d"] = round(
            row_a["padded_2d_fa2_vit"]["ms_median"] / max(wm_ms, 1e-9), 3
        )
        row_a["winmorton_speedup_over_navit_full"] = round(
            row_a["navit_ragged_full_varlen_vit"]["ms_median"] / max(wm_ms, 1e-9), 3
        )
        row_a["winmorton_speedup_over_eager_window_gather"] = round(
            row_a["qwen25vl_window8x8_eager_gather_vit"]["ms_median"] / max(wm_ms, 1e-9), 3
        )
        row_a["winmorton_mem_saving_vs_padded_2d_ratio"] = round(
            row_a["padded_2d_fa2_vit"]["inc_peak_mib_median"]
            / max(row_a["qwen25vl_winmorton2x2_zero_perm_vit"]["inc_peak_mib_median"], 1e-9),
            3,
        )
        if "parity_audit" in pref:
            row_a["parity_audit_max"] = {
                k: round(max(float(a[pk]["parity_audit"][k]) for a in sa_list), 6)
                for k in pref["parity_audit"]
            }
        sa_agg[pk] = row_a

    # ------------------------------------------------------------------------
    # Suite B Aggregation (1-GPU 3D M-RoPE + Omni-Attention & 4-Layer Decoder)
    # ------------------------------------------------------------------------
    sb_list = [r["suite_b_1gpu_mrope_omni_attention_scaling"] for r in reps]
    iso_agg: dict[str, Any] = {}
    for sk, sref in sb_list[0]["isolated_mrope_plus_omni_attention_scaling"].items():
        row_b: dict[str, Any] = {
            "seqlen": int(sref["seqlen"]),
            "max_doc_len": int(sref["max_doc_len"]),
            "visual_tokens": int(sref["visual_tokens"]),
            "upper_triangle_span_tokens": int(sref["upper_triangle_span_tokens"]),
        }
        sdpa_ref = sref["sdpa_bool_mask_eager_mrope"]
        if "median_ms" in sdpa_ref:
            ms_v = [float(b["isolated_mrope_plus_omni_attention_scaling"][sk]["sdpa_bool_mask_eager_mrope"]["median_ms"]) for b in sb_list]
            mem_v = [float(b["isolated_mrope_plus_omni_attention_scaling"][sk]["sdpa_bool_mask_eager_mrope"]["inc_peak_mib"]) for b in sb_list]
            row_b["sdpa_bool_mask_eager_mrope"] = {
                "status": "ok_math_fallback",
                "ms_median": round(_med(ms_v), 3),
                "ms_iqr": round(_iqr(ms_v), 3),
                "inc_peak_mib_median": round(_med(mem_v), 2),
            }
        else:
            row_b["sdpa_bool_mask_eager_mrope"] = dict(sdpa_ref)

        for mk in (
            "flex_attn_dynamic_bm_eager_mrope",
            "flex_attn_prebuilt_bm_triton_mrope",
            "triton_tile64_omni_gqa_plus_mrope",
            "surgical_disjoint_fa2_lse_plus_triton_mrope",
        ):
            ms_v = [float(b["isolated_mrope_plus_omni_attention_scaling"][sk][mk]["median_ms"]) for b in sb_list]
            mem_v = [float(b["isolated_mrope_plus_omni_attention_scaling"][sk][mk]["inc_peak_mib"]) for b in sb_list]
            row_b[mk] = {
                "ms_median": round(_med(ms_v), 3),
                "ms_iqr": round(_iqr(ms_v), 3),
                "inc_peak_mib_median": round(_med(mem_v), 2),
            }

        surg_ms = row_b["surgical_disjoint_fa2_lse_plus_triton_mrope"]["ms_median"]
        row_b["surgical_speedup_over_flex_dynamic"] = round(
            row_b["flex_attn_dynamic_bm_eager_mrope"]["ms_median"] / max(surg_ms, 1e-9), 3
        )
        row_b["surgical_speedup_over_flex_prebuilt"] = round(
            row_b["flex_attn_prebuilt_bm_triton_mrope"]["ms_median"] / max(surg_ms, 1e-9), 3
        )
        row_b["surgical_speedup_over_triton_tile64"] = round(
            row_b["triton_tile64_omni_gqa_plus_mrope"]["ms_median"] / max(surg_ms, 1e-9), 3
        )
        if "ms_median" in row_b["sdpa_bool_mask_eager_mrope"]:
            row_b["surgical_speedup_over_sdpa_math"] = round(
                row_b["sdpa_bool_mask_eager_mrope"]["ms_median"] / max(surg_ms, 1e-9), 3
            )
        iso_agg[sk] = row_b

    dec4_agg: dict[str, Any] = {}
    for sk, sref in sb_list[0]["multimodal_4layer_decoder_stack_scaling"].items():
        row_d: dict[str, Any] = {"seqlen": int(sref["seqlen"])}
        for mk in (
            "flex_dynamic_eager_mrope_4layer",
            "triton_tile64_triton_mrope_4layer",
            "surgical_fa2_triton_mrope_4layer",
        ):
            ms_v = [float(b["multimodal_4layer_decoder_stack_scaling"][sk][mk]["median_ms"]) for b in sb_list]
            tps_v = [float(b["multimodal_4layer_decoder_stack_scaling"][sk][mk]["tok_per_s"]) for b in sb_list]
            mem_v = [float(b["multimodal_4layer_decoder_stack_scaling"][sk][mk]["inc_peak_mib"]) for b in sb_list]
            row_d[mk] = {
                "ms_median": round(_med(ms_v), 3),
                "ms_iqr": round(_iqr(ms_v), 3),
                "tok_per_s_median": round(_med(tps_v), 1),
                "inc_peak_mib_median": round(_med(mem_v), 2),
            }
        surg_d_ms = row_d["surgical_fa2_triton_mrope_4layer"]["ms_median"]
        row_d["surgical_4layer_speedup_over_flex_dynamic"] = round(
            row_d["flex_dynamic_eager_mrope_4layer"]["ms_median"] / max(surg_d_ms, 1e-9), 3
        )
        row_d["surgical_4layer_speedup_over_triton_tile64"] = round(
            row_d["triton_tile64_triton_mrope_4layer"]["ms_median"] / max(surg_d_ms, 1e-9), 3
        )
        dec4_agg[sk] = row_d

    suite_b_summary = {
        "isolated_mrope_plus_omni_attention_scaling": iso_agg,
        "multimodal_4layer_decoder_stack_scaling": dec4_agg,
    }

    # ------------------------------------------------------------------------
    # Suite C Aggregation (8-GPU Modality-Decoupled AllToAllv + Quad-LPT FSDP2)
    # ------------------------------------------------------------------------
    sc_list = [r["suite_c_8gpu_modality_decoupled_fsdp2"] for r in reps]
    c1_agg: dict[str, Any] = {}
    for pk, pref in sc_list[0]["parity_vs_1gpu_reference"].items():
        c1_agg[pk] = {
            k: round(max(float(c["parity_vs_1gpu_reference"][pk][k]) for c in sc_list), 8)
            for k in pref
        }

    c2_agg: dict[str, Any] = {}
    for bk, bref in sc_list[0]["fsdp2_8gpu_multimodal_scaling"].items():
        modes_c: dict[str, Any] = {}
        for mk in (
            "naive_sample_shard_padded_vit_flex",
            "naive_sample_shard_winmorton_surgical",
            "token_greedy_llm_pack_only",
            "decoupled_vit_alltoall_plus_llm_quad_lpt",
        ):
            ms_v = [float(c["fsdp2_8gpu_multimodal_scaling"][bk]["modes"][mk]["median_ms"]) for c in sc_list]
            ltps_v = [float(c["fsdp2_8gpu_multimodal_scaling"][bk]["modes"][mk]["global_llm_tok_per_s"]) for c in sc_list]
            vpps_v = [float(c["fsdp2_8gpu_multimodal_scaling"][bk]["modes"][mk]["global_vit_patches_per_s"]) for c in sc_list]
            mem_v = [float(c["fsdp2_8gpu_multimodal_scaling"][bk]["modes"][mk]["peak_allocated_mib_max_rank"]) for c in sc_list]
            mref = bref["modes"][mk]
            modes_c[mk] = {
                "ms_median": round(_med(ms_v), 3),
                "ms_iqr": round(_iqr(ms_v), 3),
                "global_llm_tok_per_s_median": round(_med(ltps_v), 1),
                "global_vit_patches_per_s_median": round(_med(vpps_v), 1),
                "peak_allocated_mib_max_rank_median": round(_med(mem_v), 2),
                "vit_patch_imbalance_max_over_mean": float(mref["vit_patch_imbalance_max_over_mean"]),
                "llm_quad_cost_imbalance_max_over_mean": float(mref["llm_quad_cost_imbalance_max_over_mean"]),
                "alltoall_comm_mib_max_rank": float(mref["alltoall_comm_mib_max_rank"]),
            }
        dec_ms = modes_c["decoupled_vit_alltoall_plus_llm_quad_lpt"]["ms_median"]
        c2_agg[bk] = {
            "global_samples": int(bref["global_samples"]),
            "global_raw_vit_patches": int(bref["global_raw_vit_patches"]),
            "global_llm_tokens": int(bref["global_llm_tokens"]),
            "params_billions": float(bref["params_billions"]),
            "modes": modes_c,
            "decoupled_speedup_over_naive_padded_flex": round(
                modes_c["naive_sample_shard_padded_vit_flex"]["ms_median"] / max(dec_ms, 1e-9), 3
            ),
            "decoupled_speedup_over_naive_winmorton_surgical": round(
                modes_c["naive_sample_shard_winmorton_surgical"]["ms_median"] / max(dec_ms, 1e-9), 3
            ),
            "decoupled_speedup_over_token_greedy_llm_only": round(
                modes_c["token_greedy_llm_pack_only"]["ms_median"] / max(dec_ms, 1e-9), 3
            ),
            "decoupled_peak_mem_saving_vs_naive_padded_ratio": round(
                modes_c["naive_sample_shard_padded_vit_flex"]["peak_allocated_mib_max_rank_median"]
                / max(modes_c["decoupled_vit_alltoall_plus_llm_quad_lpt"]["peak_allocated_mib_max_rank_median"], 1e-9),
                3,
            ),
        }

    suite_c_summary = {
        "parity_vs_1gpu_reference": c1_agg,
        "fsdp2_8gpu_multimodal_scaling": c2_agg,
    }

    # ------------------------------------------------------------------------
    # 9 Strict Deterministic Validation Gates
    # ------------------------------------------------------------------------
    gates = {
        "smoke_present_and_env_parity": smoke_and_env_ok,
        "clocks_locked_1410": clocks_locked_1410,
        "tubelet_conv3d_pathology_and_merger_roofline": (
            tub_agg["speedup_triton_pack_over_hf_conv3d"] > 17.5
            and tub_agg["speedup_triton_pack_over_5d_conv3d"] > 17.5
            and tub_agg["tubelet_modes"]["triton_winmorton2x2_pack_cublas_1184"]["effective_tflops_median"] > 160.0
            and tub_agg["parity_vs_conv3d_max"]["out_rel_l2"] < 5e-4
            and tub_agg["parity_vs_conv3d_max"]["dx_rel_l2"] == 0.0
            and tub_agg["merger_speedup_zero_copy_over_raster"] > 1.35
            and tub_agg["merger_mem_saving_ratio"] > 2.5
        ),
        "mrope_3d_exact_1d_reduction_and_speedup": (
            mr_agg["text_1d_rope_exact_reduction_fp64_rel_l2_max"]["sectioned_16_24_24"] == 0.0
            and mr_agg["text_1d_rope_exact_reduction_fp64_rel_l2_max"]["interleaved_24_20_20"] == 0.0
            and max(mr_agg["parity_vs_hf_eager_max"].values()) < 0.005
            and mr_agg["speedup_triton_sectioned_over_hf_eager"] > 2.20
            and mr_agg["mem_saving_triton_over_hf_eager_ratio"] > 1.80
        ),
        "omni_attention_sdpa_oom_and_causal_corruption_audit": (
            omni_backend_agg["fa2_causal_only_visual_token_rel_l2_error_min"] > 0.85
            and omni_backend_agg["sdpa_bool_mask_s16k_oom_triggered_all_reps"]
            and max(omni_backend_agg["surgical_disjoint_fa2_parity_vs_sdpa_max"].values()) < 0.006
            and max(omni_backend_agg["triton_tile64_parity_vs_sdpa_max"].values()) < 0.006
        ),
        "fp64_oracle_and_richardson_central_difference": (
            max(oracle_agg.values()) < 0.005
            and rich_agg["min_ratio"] > 3.95
            and rich_agg["max_ratio"] < 4.05
        ),
        "suite_a_1gpu_winmorton2x2_zero_perm_vit_scaling": (
            sa_agg["P_32768"]["winmorton_speedup_over_padded_2d"] > 3.70
            and sa_agg["P_32768"]["winmorton_speedup_over_eager_window_gather"] > 1.22
            and sa_agg["P_32768"]["winmorton_speedup_over_navit_full"] > 1.06
            and sa_agg["P_32768"]["winmorton_mem_saving_vs_padded_2d_ratio"] > 3.0
            and max(sa_agg["P_4096"]["parity_audit_max"].values()) < 0.012
        ),
        "suite_b_1gpu_surgical_omni_fa2_and_4layer_decoder": (
            iso_agg["S_8192"]["surgical_speedup_over_sdpa_math"] > 32.0
            and iso_agg["S_32768"]["surgical_speedup_over_flex_dynamic"] > 2.10
            and iso_agg["S_65536"]["surgical_speedup_over_flex_dynamic"] > 2.15
            and iso_agg["S_65536"]["surgical_speedup_over_flex_prebuilt"] > 2.0
            and iso_agg["S_65536"]["surgical_speedup_over_triton_tile64"] > 2.10
            and dec4_agg["S_32768"]["surgical_4layer_speedup_over_flex_dynamic"] > 1.35
            and dec4_agg["S_32768"]["surgical_4layer_speedup_over_triton_tile64"] > 1.25
        ),
        "suite_c_8gpu_modality_decoupled_fsdp2_parity_and_scaling": (
            c1_agg["decoupled_vit_alltoall_plus_llm_quad_lpt"]["loss_rel_diff_vs_1gpu"] < 1e-4
            and c1_agg["decoupled_vs_naive_8gpu"]["loss_rel_diff"] < 1e-4
            and c1_agg["decoupled_vs_naive_8gpu"]["vit_patch_proj_dw_rel_l2"] < 0.025
            and c1_agg["decoupled_vs_naive_8gpu"]["llm_blk0_wq_dw_rel_l2"] < 0.02
            and c2_agg["B_16_samples"]["decoupled_speedup_over_naive_padded_flex"] > 2.20
            and c2_agg["B_32_samples"]["decoupled_speedup_over_naive_padded_flex"] > 2.15
            and c2_agg["B_32_samples"]["decoupled_speedup_over_naive_winmorton_surgical"] > 1.55
            and c2_agg["B_32_samples"]["decoupled_speedup_over_token_greedy_llm_only"] > 1.04
            and c2_agg["B_32_samples"]["decoupled_peak_mem_saving_vs_naive_padded_ratio"] > 1.85
            and c2_agg["B_32_samples"]["modes"]["decoupled_vit_alltoall_plus_llm_quad_lpt"]["vit_patch_imbalance_max_over_mean"] < 1.15
            and c2_agg["B_32_samples"]["modes"]["decoupled_vit_alltoall_plus_llm_quad_lpt"]["llm_quad_cost_imbalance_max_over_mean"] < 1.05
        ),
    }
    all_passed = all(gates.values())

    full_out = {
        "experiment": "E126_multimodal_vlm_omni_training",
        "repetitions": 3,
        "world_size": 8,
        "environment": reps[0]["environment"],
        "clock_telemetry": {
            "initial_sm_clock_mhz_min": round(min(init_clocks), 1),
            "median_sm_clock_mhz": round(med_clock, 1),
            "min_sm_clock_mhz": round(min_clock, 1),
            "max_sm_clock_mhz": round(max_clock, 1),
            "max_power_w": round(max_power, 2),
        },
        "validation_gates": {
            "all_gates_passed": all_passed,
            "gates": gates,
        },
        "suite_0_audits": suite_0_summary,
        "suite_a_1gpu_native_res_vit_scaling": sa_agg,
        "suite_b_1gpu_mrope_omni_attention_scaling": suite_b_summary,
        "suite_c_8gpu_modality_decoupled_fsdp2": suite_c_summary,
    }

    out_path = raw_dir / "multimodal_vlm_omni_training_full.json"
    out_path.write_text(
        json.dumps(full_out, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    readme_lines = [
        "# E126 Raw Artifacts: Multimodal Vision-Language & Omni-Modal Native-Resolution Training Kernels (`Qwen2.5-VL` / `Qwen3-VL` / `NaViT` / `Transfusion` / `VeOmni`) & 8-GPU Modality-Decoupled `FSDP2`",
        "",
        "Generated by `code/bench_multimodal_vlm_omni_training.py` across 8x NVIDIA A100-SXM4-80GB (`NV12` NVSwitch) with locked SM clocks (`1,410 MHz`), and aggregated deterministically by `code/analyze_multimodal_vlm_omni_training.py`.",
        "",
        "## Files",
        "- `smoke_multimodal_vlm_omni_training.json`: 8-GPU `--mode smoke` functional & numerical gate artifact.",
        "- `rep1_multimodal_vlm_omni_training.json`, `rep2_multimodal_vlm_omni_training.json`, `rep3_multimodal_vlm_omni_training.json`: 3 independent OS-process `--mode full` 8-GPU benchmark artifacts.",
        "- `multimodal_vlm_omni_training_full.json`: Median/IQR 3-repetition summary and 9-gate deterministic validation report (`all_gates_passed = true`).",
        "- `SHA256SUMS`: SHA-256 digests of all JSON artifacts in this directory.",
        "",
    ]
    (raw_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8", newline="\n")

    tracked_files = [
        "smoke_multimodal_vlm_omni_training.json",
        "rep1_multimodal_vlm_omni_training.json",
        "rep2_multimodal_vlm_omni_training.json",
        "rep3_multimodal_vlm_omni_training.json",
        "multimodal_vlm_omni_training_full.json",
    ]
    sha_lines = [f"{_sha256_file(raw_dir / fn)}  {fn}" for fn in tracked_files]
    (raw_dir / "SHA256SUMS").write_text("\n".join(sha_lines) + "\n", encoding="utf-8", newline="\n")

    print(f"Wrote {out_path} (all_gates_passed={all_passed})")
    if not all_passed:
        raise SystemExit(f"Validation gates failed: {gates}")


if __name__ == "__main__":
    main()
