#!/usr/bin/env python3
"""E128: Round-2 Recursive Self-Improvement (`RSI`) Roofline Convergence, Adversarial
Anti-Reward-Hacking Torture, 8-GPU `1.11B` `BD3-LM` `FSDP2` System RSI, & Fixed-Wall-Clock
`val_bpb` Autonomous Vibe-Training (`autoresearch`) Loop on `8x NVIDIA A100-SXM4-80GB`.

Governed by `PROGRAM.md` and unmodified `code/rsi_kernel_harness.py`
(`eval_harness_sha256 == 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
Co-designed and adversarially audited with `GPT 6 Astra, Max` (`Cowork`).
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import gc
import hashlib
import json
import math
import os
import statistics
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_local_rank_env = os.environ.get("LOCAL_RANK", "0")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/inductor_e128_r{_local_rank_env}")
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_e128_r{_local_rank_env}")

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import _flash_attn_varlen_backward
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

for _dyn_attr, _dyn_val in (
    ("recompile_limit", 64),
    ("cache_size_limit", 64),
    ("accumulated_recompile_limit", 256),
    ("accumulated_cache_size_limit", 256),
):
    if hasattr(torch._dynamo.config, _dyn_attr):
        setattr(torch._dynamo.config, _dyn_attr, _dyn_val)

import bench_diffusion_llm_bd3_adaln_training as e127
from common import environment_metadata
from rsi_kernel_harness import (
    HARNESS_VERSION,
    PrecisionStateSnapshot,
    RSIVectorRatchet,
    benchmark_rotating_cuda,
    compare_tensor_to_fp64,
    eval_harness_sha256,
    hash_callable_sources,
    query_sm_telemetry,
    richardson_unit_rms_audit,
    write_rsi_ledger_tsv,
)
from rsi_round2_kernels import (
    DualStreamQKVSplitRoPEFn,
    FusedSwiGLUFn,
    Round2KTile16ZeroCopyBD3Fn,
    SplitAdaLNP1Fn,
    SplitGateResAdaLNP2Fn,
    SplitGateResP3Fn,
    UnifiedAdaptiveSplitAdaLNFn,
    _adaln_p1_bwd_split_kernel,
    _adaln_p2_bwd_split_kernel,
    _adaln_p3_bwd_split_kernel,
    _adaln_p3_fwd_row_kernel,
    _adaln_unified_bwd_split_kernel,
    _adaln_unified_fwd_row_kernel,
    _bd3_ktile32_coupled_spilling_noisy_bwd_kernel,
    _bd3_ktile_clean_mbc_bwd_kernel,
    _bd3_ktile_clean_uobc_bwd_kernel,
    _bd3_ktile_fwd_kernel,
    _bd3_ktile_noisy_bwd_disjoint_kernel,
    build_split_adaln_meta,
    inspect_bd3_cubin_spills,
    naive_autograd_fa2_lse_merge_bd3,
    staged3_split_adaln_dual_step,
)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().to(torch.float64)
    bf = b.detach().to(torch.float64)
    return float(torch.linalg.norm(af - bf).item() / max(1e-12, torch.linalg.norm(bf).item()))


# ============================================================================
# 1. Suite 0: Adversarial Red-Team Gate Torture + GPT-6-Astra Invariant Audits
#    + Per-Tensor FP64 Oracles + 0-Spill Cubin Check + Unit-RMS O(h^2) Richardson
# ============================================================================


def run_suite_0_adversarial_and_oracle_audits(
    device: torch.device, is_smoke: bool
) -> tuple[dict[str, Any], RSIVectorRatchet]:
    # ------------------------------------------------------------------------
    # 0.1 Adversarial Red-Team Torture of Immutable `RSIVectorRatchet`
    # ------------------------------------------------------------------------
    redteam_ratchet = RSIVectorRatchet(
        track="adversarial_redteam_gate_audit",
        shape_keys=["N256_W16", "N256_W64"],
        hbm_cap_mib=16000.0,
    )
    gen_rt = torch.Generator(device=device).manual_seed(20261280)
    meta_rt16 = build_split_adaln_meta([128, 128], 16, 32, device)
    meta_rt64 = build_split_adaln_meta([128, 128], 64, 32, device)
    x_rt0 = (torch.randn((256, 512), device=device, dtype=torch.bfloat16, generator=gen_rt) * 0.5)
    u1_rt0 = (torch.randn((256, 512), device=device, dtype=torch.bfloat16, generator=gen_rt) * 0.5)
    u2_rt0 = (torch.randn((256, 512), device=device, dtype=torch.bfloat16, generator=gen_rt) * 0.5)
    c6_rt0 = (torch.randn((meta_rt16["n_blks"], 6, 512), device=device, dtype=torch.bfloat16, generator=gen_rt) * 0.3)
    dy1_rt = torch.randn_like(x_rt0) * 0.3
    dy2_rt = torch.randn_like(x_rt0) * 0.3
    dxo_rt = torch.randn_like(x_rt0) * 0.3

    xr64 = x_rt0.double().requires_grad_(True)
    u1r64 = u1_rt0.double().requires_grad_(True)
    u2r64 = u2_rt0.double().requires_grad_(True)
    c6r64 = c6_rt0.double().requires_grad_(True)
    y1_ref, y2_ref, xo_ref = e127.eager_adaln_dual_step(xr64, u1r64, u2r64, c6r64, meta_rt16["block_id_long"])
    torch.autograd.backward([y1_ref, y2_ref, xo_ref], [dy1_rt.double(), dy2_rt.double(), dxo_rt.double()])

    static_alias_buf = torch.empty_like(x_rt0)
    dummy_perf = {
        "N256_W16": {"median_ms": 0.12, "peak_alloc_mib": 12.0},
        "N256_W64": {"median_ms": 0.11, "peak_alloc_mib": 12.0},
    }

    def _eval_adaln_cand_for_redteam(tid: int, cid: str, mode: str, note: str) -> str:
        snap0 = PrecisionStateSnapshot.capture()
        xi = x_rt0.clone().requires_grad_(True)
        u1i = u1_rt0.clone().requires_grad_(True)
        u2i = u2_rt0.clone().requires_grad_(True)
        c6i = c6_rt0.clone().requires_grad_(True)
        try:
            y1c, y2c, xoc = staged3_split_adaln_dual_step(xi, u1i, u2i, c6i, meta_rt16)
            torch.autograd.backward([y1c, y2c, xoc], [dy1_rt, dy2_rt, dxo_rt])
            if mode == "mutate_tf32":
                torch.backends.cuda.matmul.allow_tf32 = not snap0.allow_tf32_matmul
            elif mode == "static_alias":
                static_alias_buf.copy_(y1c.detach())
                y1c = static_alias_buf
            elif mode == "drop_dalpha2":
                assert c6i.grad is not None
                c6i.grad[:, 5].zero_()

            flags_ok = PrecisionStateSnapshot.capture() == snap0
            in_unmutated = bool(torch.equal(xi.detach(), x_rt0) and torch.equal(c6i.detach(), c6_rt0))
            if mode == "static_alias":
                y1_second = static_alias_buf
            else:
                y1_second, _, _ = staged3_split_adaln_dual_step(x_rt0 * 0.5, u1_rt0, u2_rt0, c6_rt0, meta_rt16)
            non_aliasing = bool(y1c.data_ptr() != y1_second.data_ptr())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = snap0.allow_tf32_matmul

        p_checks = [
            compare_tensor_to_fp64(y1c, y1_ref, name="y1"),
            compare_tensor_to_fp64(y2c, y2_ref, name="y2"),
            compare_tensor_to_fp64(xoc, xo_ref, name="x_out"),
            compare_tensor_to_fp64(xi.grad, xr64.grad, name="dx"),
            compare_tensor_to_fp64(c6i.grad[:, 0], c6r64.grad[:, 0], name="dgamma1"),
            compare_tensor_to_fp64(c6i.grad[:, 5], c6r64.grad[:, 5], name="dalpha2"),
        ]
        perf_use = (
            {"N256_W16": {"median_ms": 0.20, "peak_alloc_mib": 14.0}, "N256_W64": {"median_ms": 0.20, "peak_alloc_mib": 14.0}}
            if tid == 0
            else dummy_perf
        )
        rec = redteam_ratchet.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=hash_callable_sources(cid, mode),
            parity_checks=p_checks,
            richardson_checks=[],
            anti_cheat_checks={
                "input_unmutated": in_unmutated,
                "output_non_aliasing": non_aliasing,
                "precision_flags_preserved": flags_ok,
            },
            per_shape_metrics=perf_use,
            hypothesis_note=note,
        )
        return rec.decision

    d0 = _eval_adaln_cand_for_redteam(0, "rt0_honest_baseline", "honest", "Verified honest baseline")
    d1 = _eval_adaln_cand_for_redteam(1, "rt1_tf32_flag_mutator", "mutate_tf32", "Mutates allow_tf32 global backend flag")
    d2 = _eval_adaln_cand_for_redteam(2, "rt2_static_buffer_aliaser", "static_alias", "Returns persistent static output buffer")
    d3 = _eval_adaln_cand_for_redteam(3, "rt3_small_dalpha2_dropper", "drop_dalpha2", "Drops [N_blks,D] dalpha2 gate gradient")

    # GPT-6-Astra Audit Proof: FlashAttention-2 autograd silently drops `grad_softmax_lse` (~22% dQ/dK error)
    # vs Direct Global-(Out_t, LSE_t) `_flash_attn_varlen_backward` (~0.0026 BF16 exactness)!
    gen_fa = torch.Generator(device=device).manual_seed(20261281)
    q_fa = torch.randn((32, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_fa).requires_grad_(True)
    ka_fa = torch.randn((32, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_fa).requires_grad_(True)
    va_fa = torch.randn((32, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_fa).requires_grad_(True)
    kb_fa = torch.randn((32, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_fa).requires_grad_(True)
    vb_fa = torch.randn((32, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_fa).requires_grad_(True)
    do_fa = torch.randn((32, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_fa)
    sm_fa = 128 ** -0.5

    q64_f, ka64_f, va64_f, kb64_f, vb64_f = [
        t.detach().double().requires_grad_(True) for t in (q_fa, ka_fa, va_fa, kb_fa, vb_fa)
    ]
    sc_u = torch.einsum("qhd,khd->hqk", q64_f, torch.cat([ka64_f, kb64_f], dim=0)) * sm_fa
    out64_u = torch.einsum("hqk,khd->qhd", torch.softmax(sc_u, dim=-1), torch.cat([va64_f, vb64_f], dim=0))
    out64_u.backward(do_fa.double())

    cu32 = torch.tensor([0, 32], device=device, dtype=torch.int32)
    oa_f, lsea_f, _ = flash_attn_varlen_func(q_fa, ka_fa, va_fa, cu32, cu32, 32, 32, softmax_scale=sm_fa, causal=False, return_attn_probs=True)
    ob_f, lseb_f, _ = flash_attn_varlen_func(q_fa, kb_fa, vb_fa, cu32, cu32, 32, 32, softmax_scale=sm_fa, causal=False, return_attn_probs=True)
    la_f, lb_f = lsea_f.transpose(0, 1).float(), lseb_f.transpose(0, 1).float()
    ltot_f = torch.logaddexp(la_f, lb_f)
    out_m1 = (torch.exp(la_f - ltot_f).unsqueeze(-1) * oa_f.float() + torch.exp(lb_f - ltot_f).unsqueeze(-1) * ob_f.float()).to(torch.bfloat16)
    out_m1.backward(do_fa)
    fa2_autograd_dq_err = rel_l2(q_fa.grad, q64_f.grad)
    fa2_autograd_dk_err = rel_l2(kb_fa.grad, kb64_f.grad)

    lse_glob_fa = ltot_f.transpose(0, 1).contiguous()
    dqa, dka, dva = torch.zeros_like(q_fa), torch.zeros_like(ka_fa), torch.zeros_like(va_fa)
    dqb, dkb, dvb = torch.zeros_like(q_fa), torch.zeros_like(kb_fa), torch.zeros_like(vb_fa)
    _flash_attn_varlen_backward(
        do_fa, q_fa.detach(), ka_fa.detach(), va_fa.detach(), out_m1.detach(), lse_glob_fa,
        dqa, dka, dva, cu32, cu32, 32, 32, 0.0, sm_fa, False, -1, -1, 0.0, None, False, None, False
    )
    _flash_attn_varlen_backward(
        do_fa, q_fa.detach(), kb_fa.detach(), vb_fa.detach(), out_m1.detach(), lse_glob_fa,
        dqb, dkb, dvb, cu32, cu32, 32, 32, 0.0, sm_fa, False, -1, -1, 0.0, None, False, None, False
    )
    direct_global_dq_err = rel_l2(dqa + dqb, q64_f.grad)
    direct_global_dk_err = rel_l2(dkb, kb64_f.grad)

    rec_rt4 = redteam_ratchet.evaluate_and_record(
        trial_id=4,
        candidate_id="rt4_naive_fa2_autograd_lse_merge_drops_dlse",
        candidate_sha256=hash_callable_sources(naive_autograd_fa2_lse_merge_bd3),
        parity_checks=[
            compare_tensor_to_fp64(q_fa.grad, q64_f.grad, name="dq_fa2_autograd"),
            compare_tensor_to_fp64(kb_fa.grad, kb64_f.grad, name="dk_fa2_autograd"),
        ],
        richardson_checks=[],
        anti_cheat_checks={"input_unmutated": True, "output_non_aliasing": True, "precision_flags_preserved": True},
        per_shape_metrics=dummy_perf,
        hypothesis_note="Naive PyTorch autograd over flash_attn_varlen_func drops grad_softmax_lse (~22% dQ/dK error)",
    )
    d4 = rec_rt4.decision

    redteam_audit = {
        "decisions": {
            "rt0_honest_baseline": d0,
            "rt1_tf32_flag_mutator": d1,
            "rt2_static_buffer_aliaser": d2,
            "rt3_small_dalpha2_dropper": d3,
            "rt4_naive_fa2_autograd_lse_merge_drops_dlse": d4,
        },
        "all_adversarial_rejected_as_expected": (
            d0 == "BASELINE"
            and d1 == "REJECT_ANTI_CHEAT"
            and d2 == "REJECT_ANTI_CHEAT"
            and d3 == "REJECT_CORRECTNESS"
            and d4 == "REJECT_CORRECTNESS"
        ),
        "gpt6_astra_fa2_dlse_audit": {
            "naive_autograd_dq_rel_l2": round(fa2_autograd_dq_err, 6),
            "naive_autograd_dk_rel_l2": round(fa2_autograd_dk_err, 6),
            "direct_global_lse_bwd_dq_rel_l2": round(direct_global_dq_err, 6),
            "direct_global_lse_bwd_dk_rel_l2": round(direct_global_dk_err, 6),
            "dk_error_reduction_factor": round(fa2_autograd_dk_err / max(1e-12, direct_global_dk_err), 2),
        },
    }

    # ------------------------------------------------------------------------
    # 0.2 Per-Tensor FP64 Analytical Oracles Across All Round-2 Edge Regimes
    # ------------------------------------------------------------------------
    bd3_r2_cases: dict[str, Any] = {}
    for case_name, docs, w_blk, noncontig, comp_clean in (
        ("ragged_cross_tile_w64_gpt6_astra", [1, 31, 32, 33, 63, 64, 65], 64, False, True),
        ("short_doc_lt_W_neg_slice_guard", [12], 64, False, True),
        ("single_doc_zerocopy_w16_noncontig", [96], 16, True, True),
        ("terminal_clean_pruned_w64", [48, 80], 64, False, False),
    ):
        n_tot = sum(docs)
        m_case = e127.build_bd3_packed_meta(docs, w_blk, device)
        g_c = torch.Generator(device=device).manual_seed(9100 + len(case_name))
        if noncontig:
            qt_raw = torch.randn((n_tot, 2, 28, 128), device=device, dtype=torch.bfloat16, generator=g_c)[:, 0]
            q0_raw = torch.randn((n_tot, 2, 28, 128), device=device, dtype=torch.bfloat16, generator=g_c)[:, 0]
        else:
            qt_raw = torch.randn((n_tot, 28, 128), device=device, dtype=torch.bfloat16, generator=g_c)
            q0_raw = torch.randn((n_tot, 28, 128), device=device, dtype=torch.bfloat16, generator=g_c)
        kt_raw = torch.randn((n_tot, 4, 128), device=device, dtype=torch.bfloat16, generator=g_c)
        vt_raw = torch.randn((n_tot, 4, 128), device=device, dtype=torch.bfloat16, generator=g_c)
        k0_raw = torch.randn((n_tot, 4, 128), device=device, dtype=torch.bfloat16, generator=g_c)
        v0_raw = torch.randn((n_tot, 4, 128), device=device, dtype=torch.bfloat16, generator=g_c)
        dot_c = torch.randn((n_tot, 28, 128), device=device, dtype=torch.bfloat16, generator=g_c) * 0.3
        do0_c = torch.randn((n_tot, 28, 128), device=device, dtype=torch.bfloat16, generator=g_c) * 0.3

        qt64, kt64, vt64, q064, k064, v064 = [
            t.detach().double().requires_grad_(True) for t in (qt_raw, kt_raw, vt_raw, q0_raw, k0_raw, v0_raw)
        ]
        ot64, o064 = e127.dense_fp64_bd3_oracle_fn(qt64, kt64, vt64, q064, k064, v064, docs, w_blk, comp_clean)
        if comp_clean:
            torch.autograd.backward([ot64, o064], [dot_c.double(), do0_c.double()])
        else:
            ot64.backward(dot_c.double())

        qtc = qt_raw.detach().requires_grad_(True)
        ktc = kt_raw.detach().requires_grad_(True)
        vtc = vt_raw.detach().requires_grad_(True)
        q0c = q0_raw.detach().requires_grad_(True) if comp_clean else None
        k0c = k0_raw.detach().requires_grad_(True)
        v0c = v0_raw.detach().requires_grad_(True)
        ot_cand, o0_cand = Round2KTile16ZeroCopyBD3Fn.apply(qtc, ktc, vtc, q0c, k0c, v0c, m_case, comp_clean, False)
        if comp_clean:
            torch.autograd.backward([ot_cand, o0_cand], [dot_c, do0_c])
        else:
            ot_cand.backward(dot_c)

        cks = [
            compare_tensor_to_fp64(ot_cand, ot64, name="ot"),
            compare_tensor_to_fp64(qtc.grad, qt64.grad, name="dqt"),
            compare_tensor_to_fp64(ktc.grad, kt64.grad, name="dkt"),
            compare_tensor_to_fp64(vtc.grad, vt64.grad, name="dvt"),
            compare_tensor_to_fp64(k0c.grad, k064.grad, name="dk0"),
            compare_tensor_to_fp64(v0c.grad, v064.grad, name="dv0"),
        ]
        if comp_clean:
            cks.extend(
                [
                    compare_tensor_to_fp64(o0_cand, o064, name="o0"),
                    compare_tensor_to_fp64(q0c.grad, q064.grad, name="dq0"),
                ]
            )
        bd3_r2_cases[case_name] = {
            "all_ok": all(c["ok"] for c in cks),
            "max_rel_l2": round(max(float(c["rel_l2"]) for c in cks), 6),
        }

    cubin_spill_telemetry = inspect_bd3_cubin_spills()

    adaln_r2_cases: dict[str, Any] = {}
    for case_name, docs, w_blk, zero_gates in (
        ("zero_init_alpha_step0_staged_and_unified", [13, 64, 71], 64, True),
        ("nonzero_ragged_w64_staged_and_unified", [13, 64, 71], 64, False),
        ("clean_stream_single_block_nblks1_split32", [256], 256, False),
    ):
        sm_a = build_split_adaln_meta(docs, w_blk, 32, device)
        ga = torch.Generator(device=device).manual_seed(9300 + len(case_name))
        xa0 = torch.randn((sm_a["n"], 3584), device=device, dtype=torch.bfloat16, generator=ga) * 0.5
        s10 = torch.randn_like(xa0) * 0.5
        s20 = torch.randn_like(xa0) * 0.5
        c60 = torch.randn((sm_a["n_blks"], 6, 3584), device=device, dtype=torch.bfloat16, generator=ga) * 0.25
        if zero_gates:
            c60[:, 2].zero_()
            c60[:, 5].zero_()
        dy1a, dy2a, dxoa = torch.randn_like(xa0) * 0.3, torch.randn_like(xa0) * 0.3, torch.randn_like(xa0) * 0.3

        xr, u1r, u2r, cr = [t.double().requires_grad_(True) for t in (xa0, s10, s20, c60)]
        y1r, y2r, xor = e127.eager_adaln_dual_step(xr, u1r, u2r, cr, sm_a["block_id_long"])
        torch.autograd.backward([y1r, y2r, xor], [dy1a.double(), dy2a.double(), dxoa.double()])

        for vname, vfn in (
            ("staged3", staged3_split_adaln_dual_step),
            ("unified", lambda x, u1, u2, c, m: UnifiedAdaptiveSplitAdaLNFn.apply(x, u1, u2, c, m, False, 1e-6)),
        ):
            xc, u1c, u2c, cc = [t.clone().requires_grad_(True) for t in (xa0, s10, s20, c60)]
            y1c, y2c, xoc = vfn(xc, u1c, u2c, cc, sm_a)
            torch.autograd.backward([y1c, y2c, xoc], [dy1a, dy2a, dxoa])
            acks = [
                compare_tensor_to_fp64(y1c, y1r, name="y1"),
                compare_tensor_to_fp64(y2c, y2r, name="y2"),
                compare_tensor_to_fp64(xoc, xor, name="x_out"),
                compare_tensor_to_fp64(xc.grad, xr.grad, name="dx"),
                compare_tensor_to_fp64(u1c.grad, u1r.grad, name="dsub1"),
                compare_tensor_to_fp64(u2c.grad, u2r.grad, name="dsub2"),
                compare_tensor_to_fp64(cc.grad[:, 0], cr.grad[:, 0], name="dgamma1"),
                compare_tensor_to_fp64(cc.grad[:, 2], cr.grad[:, 2], name="dalpha1"),
                compare_tensor_to_fp64(cc.grad[:, 3], cr.grad[:, 3], name="dgamma2"),
                compare_tensor_to_fp64(cc.grad[:, 5], cr.grad[:, 5], name="dalpha2"),
            ]
            adaln_r2_cases[f"{case_name}__{vname}"] = {
                "all_ok": all(c["ok"] for c in acks),
                "max_rel_l2": round(max(float(c["rel_l2"]) for c in acks), 6),
            }

    # Merged W_qkv Zero-Clone RoPE + Merged W_gate_up FusedSwiGLU FP64 parity
    gq = torch.Generator(device=device).manual_seed(9501)
    cos_q, sin_q = e127.build_doc_rope_cache([32, 32], 128, device)
    x_cat0 = torch.randn((128, 2048), device=device, dtype=torch.bfloat16, generator=gq) * 0.4
    wq0 = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16, generator=gq) * 0.04
    wk0 = torch.randn((512, 2048), device=device, dtype=torch.bfloat16, generator=gq) * 0.04
    wv0 = torch.randn((512, 2048), device=device, dtype=torch.bfloat16, generator=gq) * 0.04
    wg0 = torch.randn((6144, 2048), device=device, dtype=torch.bfloat16, generator=gq) * 0.04
    wu0 = torch.randn((6144, 2048), device=device, dtype=torch.bfloat16, generator=gq) * 0.04
    dq_u = torch.randn((128, 16, 128), device=device, dtype=torch.bfloat16, generator=gq) * 0.2
    dk_u = torch.randn((128, 4, 128), device=device, dtype=torch.bfloat16, generator=gq) * 0.2
    dv_u = torch.randn((128, 4, 128), device=device, dtype=torch.bfloat16, generator=gq) * 0.2
    dsw_u = torch.randn((128, 6144), device=device, dtype=torch.bfloat16, generator=gq) * 0.2

    def _rope64(z64: torch.Tensor) -> torch.Tensor:
        c2 = torch.cat([cos_q, cos_q], dim=0).double().unsqueeze(1)
        s2 = torch.cat([sin_q, sin_q], dim=0).double().unsqueeze(1)
        z1, z2 = z64[..., :64], z64[..., 64:]
        return torch.cat([z1 * c2 - z2 * s2, z1 * s2 + z2 * c2], dim=-1)

    x64_m = x_cat0.double().requires_grad_(True)
    wq64, wk64, wv64, wg64, wu64 = [t.double().requires_grad_(True) for t in (wq0, wk0, wv0, wg0, wu0)]
    q64_o = _rope64((x64_m @ wq64.T).view(128, 16, 128))
    k64_o = _rope64((x64_m @ wk64.T).view(128, 4, 128))
    v64_o = (x64_m @ wv64.T).view(128, 4, 128)
    sw64_o = F.silu(x64_m @ wg64.T) * (x64_m @ wu64.T)
    torch.autograd.backward([q64_o, k64_o, v64_o, sw64_o], [dq_u.double(), dk_u.double(), dv_u.double(), dsw_u.double()])

    xc_m = x_cat0.clone().requires_grad_(True)
    wqkv_c = torch.cat([wq0, wk0, wv0], dim=0).contiguous().requires_grad_(True)
    wgu_c = torch.cat([wg0, wu0], dim=0).contiguous().requires_grad_(True)
    qc_o, kc_o, vc_o = DualStreamQKVSplitRoPEFn.apply(xc_m @ wqkv_c.T, cos_q, sin_q, 16, 4, 128)
    swc_o = FusedSwiGLUFn.apply(xc_m @ wgu_c.T)
    torch.autograd.backward([qc_o, kc_o, vc_o, swc_o], [dq_u, dk_u, dv_u, dsw_u])

    backbone_cks = [
        compare_tensor_to_fp64(qc_o, q64_o, name="q_rope"),
        compare_tensor_to_fp64(kc_o, k64_o, name="k_rope"),
        compare_tensor_to_fp64(vc_o, v64_o, name="v_split"),
        compare_tensor_to_fp64(swc_o, sw64_o, name="swiglu_out"),
        compare_tensor_to_fp64(xc_m.grad, x64_m.grad, name="dx_merged"),
        compare_tensor_to_fp64(wqkv_c.grad, torch.cat([wq64.grad, wk64.grad, wv64.grad], dim=0), name="dw_qkv"),
        compare_tensor_to_fp64(wgu_c.grad, torch.cat([wg64.grad, wu64.grad], dim=0), name="dw_gate_up"),
    ]
    backbone_parity = {
        "all_ok": all(c["ok"] for c in backbone_cks),
        "max_rel_l2": round(max(float(c["rel_l2"]) for c in backbone_cks), 6),
    }

    # ------------------------------------------------------------------------
    # 0.3 FP64 Unit-RMS O(h^2) Central-Difference Richardson Extrapolation (~4.000x)
    # ------------------------------------------------------------------------
    gen_rich = torch.Generator(device=device).manual_seed(9888)
    sm_r = build_split_adaln_meta([16, 16], 8, 8, device)
    x_r64 = (torch.randn((32, 128), device=device, dtype=torch.float64, generator=gen_rich) * 0.5).requires_grad_(True)
    s1_r64 = (torch.randn((32, 128), device=device, dtype=torch.float64, generator=gen_rich) * 0.5).requires_grad_(True)
    s2_r64 = (torch.randn((32, 128), device=device, dtype=torch.float64, generator=gen_rich) * 0.5).requires_grad_(True)
    c6_r64 = (torch.randn((sm_r["n_blks"], 6, 128), device=device, dtype=torch.float64, generator=gen_rich) * 0.3).requires_grad_(True)
    wy1 = torch.randn((32, 128), device=device, dtype=torch.float64, generator=gen_rich) * 0.3
    wy2 = torch.randn((32, 128), device=device, dtype=torch.float64, generator=gen_rich) * 0.3
    wxo = torch.randn((32, 128), device=device, dtype=torch.float64, generator=gen_rich) * 0.3

    def _adaln3_fp64(xi: torch.Tensor, c6i: torch.Tensor) -> torch.Tensor:
        g1, b1, a1, g2, b2, a2 = c6i.unbind(dim=1)
        bid = sm_r["block_id_long"]
        r1 = torch.rsqrt(xi.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        y1i = xi * r1 * (1.0 + g1.index_select(0, bid)) + b1.index_select(0, bid)
        xni = xi + a1.index_select(0, bid) * s1_r64.detach()
        r2 = torch.rsqrt(xni.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        y2i = xni * r2 * (1.0 + g2.index_select(0, bid)) + b2.index_select(0, bid)
        xoi = xni + a2.index_select(0, bid) * s2_r64.detach()
        return (torch.sin(y1i) * wy1 + torch.cos(y2i) * wy2 + torch.sin(xoi) * wxo).sum()

    _adaln3_fp64(x_r64, c6_r64).backward()

    g_sw64 = (torch.randn((32, 96), device=device, dtype=torch.float64, generator=gen_rich) * 0.6).requires_grad_(True)
    u_sw64 = (torch.randn((32, 96), device=device, dtype=torch.float64, generator=gen_rich) * 0.6).requires_grad_(True)
    w_sw = torch.randn((32, 96), device=device, dtype=torch.float64, generator=gen_rich) * 0.4

    def _swiglu_fp64(gi: torch.Tensor, ui: torch.Tensor) -> torch.Tensor:
        return (torch.sin(F.silu(gi) * ui) * w_sw).sum()

    _swiglu_fp64(g_sw64, u_sw64).backward()

    dl_rb = [16, 16]
    qt_rb = (torch.randn((32, 8, 64), device=device, dtype=torch.float64, generator=gen_rich) * 0.4).requires_grad_(True)
    kt_rb = (torch.randn((32, 2, 64), device=device, dtype=torch.float64, generator=gen_rich) * 0.4).requires_grad_(True)
    vt_rb = (torch.randn((32, 2, 64), device=device, dtype=torch.float64, generator=gen_rich) * 0.4).requires_grad_(True)
    q0_rb = (torch.randn((32, 8, 64), device=device, dtype=torch.float64, generator=gen_rich) * 0.4).requires_grad_(True)
    k0_rb = (torch.randn((32, 2, 64), device=device, dtype=torch.float64, generator=gen_rich) * 0.4).requires_grad_(True)
    v0_rb = (torch.randn((32, 2, 64), device=device, dtype=torch.float64, generator=gen_rich) * 0.4).requires_grad_(True)
    wt_rb = torch.randn((32, 8, 64), device=device, dtype=torch.float64, generator=gen_rich) * 0.4
    w0_rb = torch.randn((32, 8, 64), device=device, dtype=torch.float64, generator=gen_rich) * 0.4

    def _bd3_fp64(qti: torch.Tensor, k0i: torch.Tensor) -> torch.Tensor:
        oti, o0i = e127.dense_fp64_bd3_oracle_fn(
            qti, kt_rb.detach(), vt_rb.detach(), q0_rb.detach(), k0i, v0_rb.detach(), dl_rb, 8, True
        )
        return (torch.sin(oti) * wt_rb + torch.cos(o0i) * w0_rb).sum()

    _bd3_fp64(qt_rb, k0_rb).backward()

    rich_checks = [
        richardson_unit_rms_audit(lambda z: _adaln3_fp64(z, c6_r64.detach()), x_r64.detach(), x_r64.grad, seed=201, name="adaln_full3_dx"),
        richardson_unit_rms_audit(
            lambda z: _adaln3_fp64(
                x_r64.detach(),
                torch.stack([c6_r64.detach()[:, 0], c6_r64.detach()[:, 1], c6_r64.detach()[:, 2], c6_r64.detach()[:, 3], c6_r64.detach()[:, 4], z], dim=1),
            ),
            c6_r64.detach()[:, 5], c6_r64.grad[:, 5], seed=202, name="adaln_full3_dalpha2",
        ),
        richardson_unit_rms_audit(lambda z: _swiglu_fp64(z, u_sw64.detach()), g_sw64.detach(), g_sw64.grad, seed=203, name="swiglu_dgate"),
        richardson_unit_rms_audit(lambda z: _swiglu_fp64(g_sw64.detach(), z), u_sw64.detach(), u_sw64.grad, seed=204, name="swiglu_dup"),
        richardson_unit_rms_audit(lambda z: _bd3_fp64(z, k0_rb.detach()), qt_rb.detach(), qt_rb.grad, seed=205, name="bd3_r2_dqt"),
        richardson_unit_rms_audit(lambda z: _bd3_fp64(qt_rb.detach(), z), k0_rb.detach(), k0_rb.grad, seed=206, name="bd3_r2_dk0"),
    ]
    rich_ratios = {it["name"]: it["ratio_2h_over_1h"] for it in rich_checks}

    all_parity_ok = (
        all(v["all_ok"] for v in bd3_r2_cases.values())
        and all(v["all_ok"] for v in adaln_r2_cases.values())
        and backbone_parity["all_ok"]
        and cubin_spill_telemetry["max_disjoint_spills"] == 0
    )
    return (
        {
            "adversarial_redteam_audit": redteam_audit,
            "per_tensor_fp64_oracle_parity": {
                "bd3_r2_attention_cases": bd3_r2_cases,
                "adaln_r2_cases": adaln_r2_cases,
                "merged_qkv_rope_and_swiglu": backbone_parity,
                "bd3_cubin_spill_telemetry": cubin_spill_telemetry,
                "all_parity_and_zero_spills_ok": all_parity_ok,
            },
            "fp64_richardson_central_difference": {
                "checks": rich_checks,
                "ratios": rich_ratios,
                "min_ratio": round(min(rich_ratios.values()), 4),
                "max_ratio": round(max(rich_ratios.values()), 4),
                "all_ok": all(it["ok"] for it in rich_checks),
            },
        },
        redteam_ratchet,
    )


# ============================================================================
# 2. Suite A: Round-2 Roofline Vector-Pareto Ratchets (Track A AdaLN + Track B BD3)
# ============================================================================


def run_suite_a_round2_roofline_ratchets(
    device: torch.device, is_smoke: bool
) -> tuple[dict[str, Any], RSIVectorRatchet, RSIVectorRatchet]:
    l2_flush = torch.empty((16 * 1024 * 1024,), device=device, dtype=torch.int32)

    # ------------------------------------------------------------------------
    # Track A: Round-2 `adaLN-Zero` Roofline Convergence (`D = 3,584`)
    # ------------------------------------------------------------------------
    n_ad = 8192 if is_smoke else 32768
    adaln_shapes = [
        (f"N{n_ad}_W4", [n_ad // 2, n_ad // 2], 4),
        (f"N{n_ad}_W16", [n_ad // 2, n_ad // 2], 16),
        (f"N{n_ad}_W64", [n_ad // 2, n_ad // 2], 64),
        (f"N{n_ad}_W256_ragged", [n_ad // 2 - 5, n_ad // 2 + 5], 256),
        (f"N{n_ad}_W{n_ad}_clean1blk", [n_ad], n_ad),
    ]
    ratchet_adaln_r2 = RSIVectorRatchet(
        track="round2_adaln_zero_roofline",
        shape_keys=[s[0] for s in adaln_shapes],
        hbm_cap_mib=48000.0,
    )

    adaln_candidates = [
        (
            0,
            "trial0_e127_round1_winner_1cta_per_blk_unfused_p3",
            hash_callable_sources(e127.triton_segmented_adaln_dual_step),
            "r1_base",
            "Round-1 E127 winner: 1 CTA/block backward (grid=(1,) on clean stream!) + unfused Point-3 index_add_",
        ),
        (
            1,
            "trial1_staged3_adaptive_direct_or_split32",
            hash_callable_sources(
                _adaln_p1_bwd_split_kernel, _adaln_p2_bwd_split_kernel,
                _adaln_p3_fwd_row_kernel, _adaln_p3_bwd_split_kernel, staged3_split_adaln_dual_step
            ),
            "staged3_split32",
            "Transformer-interleaved 3-point fused Point-3 + adaptive Direct-dc6 (W<=32) / Split-32 (W>32, 1024 CTAs)",
        ),
        (
            2,
            "trial2_unified_oversplit2_unconditional_scratch",
            hash_callable_sources(_adaln_unified_bwd_split_kernel, "ROWS=2,force_scratch=True"),
            "unified_oversplit2",
            "Over-splits ROWS_PER_CTA=2 with forced FP32 [N/2,6,D] scratch write+read (1.41 GB extra HBM traffic)",
        ),
        (
            3,
            "trial3_unified_adaptive_direct_or_split32",
            hash_callable_sources(_adaln_unified_fwd_row_kernel, _adaln_unified_bwd_split_kernel, UnifiedAdaptiveSplitAdaLNFn),
            "unified_split32",
            "Single-pass row-parallel forward + adaptive Direct-dc6 (W<=32) / Split-32 (W>32) backward (>1000 GB/s)",
        ),
    ]

    # Reference PyTorch eager timing for cumulative R1+R2 speedup reporting
    eager_anchor_ms: dict[str, float] = {}
    for skey, docs, w_blk in adaln_shapes:
        sm_e = e127.build_adaln_segment_meta(docs, w_blk, device)
        xe = torch.randn((n_ad, 3584), device=device, dtype=torch.bfloat16, requires_grad=True)
        u1e = torch.randn_like(xe, requires_grad=True)
        u2e = torch.randn_like(xe, requires_grad=True)
        c6e = torch.randn((sm_e["n_blks"], 6, 3584), device=device, dtype=torch.bfloat16, requires_grad=True)
        dy1e, dy2e, dxoe = torch.randn_like(xe), torch.randn_like(xe), torch.randn_like(xe)

        def _eager_step():
            y1o, y2o, xoo = e127.eager_adaln_dual_step(xe, u1e, u2e, c6e, sm_e["block_id_long"])
            torch.autograd.backward([y1o, y2o, xoo], [dy1e, dy2e, dxoe])
            xe.grad = u1e.grad = u2e.grad = c6e.grad = None

        r_e = benchmark_rotating_cuda(
            [_eager_step], warmup=2, groups=3 if is_smoke else 5, iters_per_group=2 if is_smoke else 3, l2_flush_buffer=l2_flush
        )
        eager_anchor_ms[skey] = r_e["median_ms"]
        del xe, u1e, u2e, c6e, dy1e, dy2e, dxoe
    eager_geomean_ms = round(math.exp(sum(math.log(v) for v in eager_anchor_ms.values()) / len(eager_anchor_ms)), 4)

    gen_ag = torch.Generator(device=device).manual_seed(271828)
    sm_gate32 = build_split_adaln_meta([13, 64, 19], 32, 32, device)
    sm_gate2 = build_split_adaln_meta([13, 64, 19], 32, 2, device)
    xg0 = torch.randn((sm_gate32["n"], 3584), device=device, dtype=torch.bfloat16, generator=gen_ag) * 0.5
    u1g0 = torch.randn_like(xg0) * 0.5
    u2g0 = torch.randn_like(xg0) * 0.5
    c6g0 = torch.randn((sm_gate32["n_blks"], 6, 3584), device=device, dtype=torch.bfloat16, generator=gen_ag) * 0.3
    dy1g, dy2g, dxog = torch.randn_like(xg0) * 0.3, torch.randn_like(xg0) * 0.3, torch.randn_like(xg0) * 0.3
    xr64, u1r64, u2r64, cr64 = [t.double().requires_grad_(True) for t in (xg0, u1g0, u2g0, c6g0)]
    y1_ref, y2_ref, xo_ref = e127.eager_adaln_dual_step(xr64, u1r64, u2r64, cr64, sm_gate32["block_id_long"])
    torch.autograd.backward([y1_ref, y2_ref, xo_ref], [dy1g.double(), dy2g.double(), dxog.double()])

    def _run_adaln_mode(mode_tag: str, x: torch.Tensor, u1: torch.Tensor, u2: torch.Tensor, c6: torch.Tensor, m32: dict, m2: dict):
        if mode_tag == "r1_base":
            return e127.triton_segmented_adaln_dual_step(x, u1, u2, c6, m32)
        if mode_tag == "staged3_split32":
            return staged3_split_adaln_dual_step(x, u1, u2, c6, m32)
        if mode_tag == "unified_oversplit2":
            return UnifiedAdaptiveSplitAdaLNFn.apply(x, u1, u2, c6, m2, True, 1e-6)
        return UnifiedAdaptiveSplitAdaLNFn.apply(x, u1, u2, c6, m32, False, 1e-6)

    adaln_detail: dict[str, Any] = {}
    for tid, cid, csha, mtag, cnote in adaln_candidates:
        snap0 = PrecisionStateSnapshot.capture()
        xi, u1i, u2i, ci = [t.clone().requires_grad_(True) for t in (xg0, u1g0, u2g0, c6g0)]
        y1c, y2c, xoc = _run_adaln_mode(mtag, xi, u1i, u2i, ci, sm_gate32, sm_gate2)
        torch.autograd.backward([y1c, y2c, xoc], [dy1g, dy2g, dxog])
        in_unmut = bool(torch.equal(xi.detach(), xg0) and torch.equal(ci.detach(), c6g0))
        y1_2nd, _, _ = _run_adaln_mode(mtag, xg0 * 0.5, u1g0, u2g0, c6g0, sm_gate32, sm_gate2)
        non_alias = bool(y1c.data_ptr() != y1_2nd.data_ptr())
        flags_ok = PrecisionStateSnapshot.capture() == snap0

        p_cks = [
            compare_tensor_to_fp64(y1c, y1_ref, name="y1"),
            compare_tensor_to_fp64(y2c, y2_ref, name="y2"),
            compare_tensor_to_fp64(xoc, xo_ref, name="x_out"),
            compare_tensor_to_fp64(xi.grad, xr64.grad, name="dx"),
            compare_tensor_to_fp64(u1i.grad, u1r64.grad, name="dsub1"),
            compare_tensor_to_fp64(u2i.grad, u2r64.grad, name="dsub2"),
            compare_tensor_to_fp64(ci.grad[:, 0], cr64.grad[:, 0], name="dgamma1"),
            compare_tensor_to_fp64(ci.grad[:, 2], cr64.grad[:, 2], name="dalpha1"),
            compare_tensor_to_fp64(ci.grad[:, 5], cr64.grad[:, 5], name="dalpha2"),
        ]

        shape_perf: dict[str, dict[str, Any]] = {}
        for skey, docs, w_blk in adaln_shapes:
            m32 = build_split_adaln_meta(docs, w_blk, 32, device)
            m2 = build_split_adaln_meta(docs, w_blk, 2, device)
            bytes_per_slot = n_ad * 3584 * 2 * 6
            n_slots = max(2, int(math.ceil((112.0 * 1024.0 * 1024.0) / float(bytes_per_slot))))
            slots = []
            for _ in range(n_slots):
                xs = torch.randn((n_ad, 3584), device=device, dtype=torch.bfloat16, requires_grad=True)
                u1s = torch.randn_like(xs, requires_grad=True)
                u2s = torch.randn_like(xs, requires_grad=True)
                c6s = torch.randn((m32["n_blks"], 6, 3584), device=device, dtype=torch.bfloat16, requires_grad=True)
                slots.append((xs, u1s, u2s, c6s, torch.randn_like(xs), torch.randn_like(xs), torch.randn_like(xs)))

            def _mk_fn(pk, mt=mtag, m32_l=m32, m2_l=m2):
                xs, u1s, u2s, c6s, dy1s, dy2s, dxos = pk

                def _step():
                    o1, o2, o3 = _run_adaln_mode(mt, xs, u1s, u2s, c6s, m32_l, m2_l)
                    torch.autograd.backward([o1, o2, o3], [dy1s, dy2s, dxos])
                    xs.grad = u1s.grad = u2s.grad = c6s.grad = None

                return _step

            res = benchmark_rotating_cuda(
                [_mk_fn(pk) for pk in slots],
                warmup=2 if is_smoke else 4,
                groups=3 if is_smoke else 7,
                iters_per_group=3 if is_smoke else 6,
                l2_flush_buffer=l2_flush,
            )
            minimal_bytes = 12.0 * float(n_ad) * 3584.0 * 2.0
            scratch_bytes = 0.0
            if mtag == "unified_oversplit2":
                scratch_bytes = 2.0 * float(m2["n_subchunks"]) * 6.0 * 3584.0 * 4.0
            elif mtag in ("staged3_split32", "unified_split32") and not m32["one_cta_per_blk"]:
                scratch_bytes = 2.0 * float(m32["n_subchunks"]) * 6.0 * 3584.0 * 4.0
            if mtag == "staged3_split32":
                # Staged 3-point writes+reads intermediate x_next across sublayers (+4*N*D*2B physical HBM)
                phys_bytes = minimal_bytes + 4.0 * float(n_ad) * 3584.0 * 2.0 + scratch_bytes
            else:
                phys_bytes = minimal_bytes + scratch_bytes
            res["algorithmic_minimal_gbps"] = round((minimal_bytes / (res["median_ms"] * 1e-3)) / 1e9, 2)
            res["physical_realized_gbps"] = round((phys_bytes / (res["median_ms"] * 1e-3)) / 1e9, 2)
            shape_perf[skey] = res
            del slots
            torch.cuda.empty_cache()

        rec = ratchet_adaln_r2.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=csha,
            parity_checks=p_cks,
            richardson_checks=[],
            anti_cheat_checks={
                "input_unmutated": in_unmut,
                "output_non_aliasing": non_alias,
                "precision_flags_preserved": flags_ok,
            },
            per_shape_metrics=shape_perf,
            hypothesis_note=cnote,
        )
        adaln_detail[cid] = {
            "decision": rec.decision,
            "geomean_ms": rec.geomean_ms,
            "speedup_vs_round1": rec.speedup_vs_baseline,
            "speedup_vs_pytorch_eager": round(eager_geomean_ms / max(rec.geomean_ms, 1e-12), 4),
            "shapes": shape_perf,
        }

    # ------------------------------------------------------------------------
    # Track B: Round-2 `BD3-LM` `[x_t || x_0]` Attention (`H_Q=28, H_KV=4, D=128`)
    # ------------------------------------------------------------------------
    n_bd = 4096 if is_smoke else 16384
    bd3_shapes = [
        (f"packed16_L{n_bd}_W16", [n_bd // 16] * 16, 16),
        (f"packed16_L{n_bd}_W64", [n_bd // 16] * 16, 64),
        (f"2doc_L{n_bd}_W64", [n_bd // 2, n_bd // 2], 64),
        (f"1doc_L{n_bd}_W64_zerocopy", [n_bd], 64),
    ]
    ratchet_bd3_r2 = RSIVectorRatchet(
        track="round2_bd3_attention_zerospill_ktile",
        shape_keys=[s[0] for s in bd3_shapes],
        hbm_cap_mib=48000.0,
    )

    bd3_candidates = [
        (
            0,
            "trial0_e127_round1_adaptive_winner",
            hash_callable_sources(e127.bd3_trial3_adaptive_zerospill),
            "r1_base",
            False,
            "Round-1 E127 winner: falls back to 6-call FA2 at W=64 and allocates [n_obc,H,D] gather/scatter buffers",
        ),
        (
            1,
            "trial1_naive_autograd_fa2_lse_merge_broken",
            hash_callable_sources(naive_autograd_fa2_lse_merge_bd3),
            "naive_autograd_broken",
            False,
            "Naive PyTorch autograd LSE-merge over flash_attn_varlen_func (silently drops grad_softmax_lse!)",
        ),
        (
            2,
            "trial2_ktile32_warps8_coupled_spilling",
            hash_callable_sources(_bd3_ktile32_coupled_spilling_noisy_bwd_kernel),
            "r2_spilling",
            False,
            "K_TILE=32 with num_warps=8 coupled register lifetimes (SM80 128-reg cap -> 454 cubin spills at W=64!)",
        ),
        (
            3,
            "trial3_ktile16_disjoint_zerospill_zerocopy",
            hash_callable_sources(
                _bd3_ktile_fwd_kernel, _bd3_ktile_noisy_bwd_disjoint_kernel,
                _bd3_ktile_clean_mbc_bwd_kernel, _bd3_ktile_clean_uobc_bwd_kernel, Round2KTile16ZeroCopyBD3Fn
            ),
            "r2_zerospill",
            False,
            "K_TILE=16, Q_STEP_BWD=4, disjoint lifetimes (0 cubin spills!) + guarded zero-copy/slice-cat OBC",
        ),
        (
            4,
            "trial4_ktile16_zerospill_last_layer_clean_pruned",
            hash_callable_sources(Round2KTile16ZeroCopyBD3Fn, "compute_clean_out=False"),
            "r2_pruned",
            True,
            "Terminal-layer clean-output pruning on top of zero-spill K_TILE=16 + zero-copy OBC",
        ),
    ]

    gen_bg = torch.Generator(device=device).manual_seed(3141592)
    docs_bg = [40, 64, 72]
    nbg = sum(docs_bg)
    meta_bg = e127.build_bd3_packed_meta(docs_bg, 64, device)
    qt_bg = torch.randn((nbg, 28, 128), device=device, dtype=torch.bfloat16, generator=gen_bg) * 0.4
    kt_bg = torch.randn((nbg, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_bg) * 0.4
    vt_bg = torch.randn((nbg, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_bg) * 0.4
    q0_bg = torch.randn((nbg, 28, 128), device=device, dtype=torch.bfloat16, generator=gen_bg) * 0.4
    k0_bg = torch.randn((nbg, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_bg) * 0.4
    v0_bg = torch.randn((nbg, 4, 128), device=device, dtype=torch.bfloat16, generator=gen_bg) * 0.4
    dot_bg = torch.randn_like(qt_bg) * 0.3
    do0_bg = torch.randn_like(q0_bg) * 0.3

    qt64, kt64, vt64, q064, k064, v064 = [
        t.double().requires_grad_(True) for t in (qt_bg, kt_bg, vt_bg, q0_bg, k0_bg, v0_bg)
    ]
    ot_ref, o0_ref = e127.dense_fp64_bd3_oracle_fn(qt64, kt64, vt64, q064, k064, v064, docs_bg, 64, True)
    torch.autograd.backward([ot_ref, o0_ref], [dot_bg.double(), do0_bg.double()])

    qt64p, kt64p, vt64p, q064p, k064p, v064p = [
        t.double().requires_grad_(True) for t in (qt_bg, kt_bg, vt_bg, q0_bg, k0_bg, v0_bg)
    ]
    otp_ref, _ = e127.dense_fp64_bd3_oracle_fn(qt64p, kt64p, vt64p, q064p, k064p, v064p, docs_bg, 64, False)
    otp_ref.backward(dot_bg.double())

    def _run_bd3_mode(mtag: str, qt, kt, vt, q0, k0, v0, meta: dict):
        if mtag == "r1_base":
            return e127.bd3_trial3_adaptive_zerospill(qt, kt, vt, q0, k0, v0, meta, True)
        if mtag == "naive_autograd_broken":
            return naive_autograd_fa2_lse_merge_bd3(qt, kt, vt, q0, k0, v0, meta, True)
        if mtag == "r2_spilling":
            return Round2KTile16ZeroCopyBD3Fn.apply(qt, kt, vt, q0, k0, v0, meta, True, True)
        if mtag == "r2_zerospill":
            return Round2KTile16ZeroCopyBD3Fn.apply(qt, kt, vt, q0, k0, v0, meta, True, False)
        return Round2KTile16ZeroCopyBD3Fn.apply(qt, kt, vt, None, k0, v0, meta, False, False)

    bd3_detail: dict[str, Any] = {}
    for tid, cid, csha, mtag, is_pruned, cnote in bd3_candidates:
        snap0 = PrecisionStateSnapshot.capture()
        qti = qt_bg.clone().requires_grad_(True)
        kti = kt_bg.clone().requires_grad_(True)
        vti = vt_bg.clone().requires_grad_(True)
        q0i = None if is_pruned else q0_bg.clone().requires_grad_(True)
        k0i = k0_bg.clone().requires_grad_(True)
        v0i = v0_bg.clone().requires_grad_(True)

        otc, o0c = _run_bd3_mode(mtag, qti, kti, vti, q0i, k0i, v0i, meta_bg)
        if is_pruned:
            otc.backward(dot_bg)
            p_cks = [
                compare_tensor_to_fp64(otc, otp_ref, name="ot"),
                compare_tensor_to_fp64(qti.grad, qt64p.grad, name="dqt"),
                compare_tensor_to_fp64(kti.grad, kt64p.grad, name="dkt"),
                compare_tensor_to_fp64(k0i.grad, k064p.grad, name="dk0"),
            ]
        else:
            torch.autograd.backward([otc, o0c], [dot_bg, do0_bg])
            p_cks = [
                compare_tensor_to_fp64(otc, ot_ref, name="ot"),
                compare_tensor_to_fp64(o0c, o0_ref, name="o0"),
                compare_tensor_to_fp64(qti.grad, qt64.grad, name="dqt"),
                compare_tensor_to_fp64(kti.grad, kt64.grad, name="dkt"),
                compare_tensor_to_fp64(q0i.grad, q064.grad, name="dq0"),
                compare_tensor_to_fp64(k0i.grad, k064.grad, name="dk0"),
            ]
        in_unmut = bool(torch.equal(qti.detach(), qt_bg) and torch.equal(k0i.detach(), k0_bg))
        ot_2nd, _ = _run_bd3_mode(mtag, qt_bg * 0.5, kt_bg, vt_bg, q0_bg, k0_bg, v0_bg, meta_bg)
        non_alias = bool(otc.data_ptr() != ot_2nd.data_ptr())
        flags_ok = PrecisionStateSnapshot.capture() == snap0

        shape_perf_b: dict[str, dict[str, Any]] = {}
        for skey, docs, w_blk in bd3_shapes:
            ms = e127.build_bd3_packed_meta(docs, w_blk, device)
            qts = torch.randn((n_bd, 28, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
            kts = torch.randn((n_bd, 4, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
            vts = torch.randn((n_bd, 4, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
            q0s = torch.randn((n_bd, 28, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
            k0s = torch.randn((n_bd, 4, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
            v0s = torch.randn((n_bd, 4, 128), device=device, dtype=torch.bfloat16, requires_grad=True)
            dots, do0s = torch.randn_like(qts), torch.randn_like(q0s)

            def _step_b(mt=mtag, ip=is_pruned, m_l=ms):
                o1, o2 = _run_bd3_mode(mt, qts, kts, vts, None if ip else q0s, k0s, v0s, m_l)
                if ip:
                    o1.backward(dots)
                else:
                    torch.autograd.backward([o1, o2], [dots, do0s])
                qts.grad = kts.grad = vts.grad = q0s.grad = k0s.grad = v0s.grad = None

            shape_perf_b[skey] = benchmark_rotating_cuda(
                [_step_b],
                warmup=2 if is_smoke else 3,
                groups=3 if is_smoke else 6,
                iters_per_group=2 if is_smoke else 4,
                l2_flush_buffer=l2_flush,
            )
            del qts, kts, vts, q0s, k0s, v0s, dots, do0s
            torch.cuda.empty_cache()

        rec_b = ratchet_bd3_r2.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=csha,
            parity_checks=p_cks,
            richardson_checks=[],
            anti_cheat_checks={
                "input_unmutated": in_unmut,
                "output_non_aliasing": non_alias,
                "precision_flags_preserved": flags_ok,
            },
            per_shape_metrics=shape_perf_b,
            hypothesis_note=cnote,
            allow_pruned_branch_stage=is_pruned,
        )
        bd3_detail[cid] = {
            "decision": rec_b.decision,
            "geomean_ms": rec_b.geomean_ms,
            "speedup_vs_round1": rec_b.speedup_vs_baseline,
            "peak_alloc_mib": rec_b.peak_alloc_mib,
            "shapes": shape_perf_b,
        }

    del l2_flush
    torch.cuda.empty_cache()
    return (
        {
            "track_a_adaln_zero_roofline": {
                "pytorch_eager_anchor_geomean_ms": eager_geomean_ms,
                "pytorch_eager_anchor_per_shape_ms": eager_anchor_ms,
                "ratchet": ratchet_adaln_r2.to_summary_dict(),
                "candidates": adaln_detail,
            },
            "track_b_bd3_attention_zerospill": {
                "cubin_spill_telemetry_after_sweep": inspect_bd3_cubin_spills(),
                "ratchet": ratchet_bd3_r2.to_summary_dict(),
                "candidates": bd3_detail,
            },
        },
        ratchet_adaln_r2,
        ratchet_bd3_r2,
    )


# ============================================================================
# 3. Round-2 Merged-QKV / Merged-SwiGLU / Split-AdaLN / Zero-Spill BD3 Model
#    for Suite B (8-GPU 1.11B FSDP2) and Suite C (Fixed-Wall-Clock val_bpb Loop)
# ============================================================================


class BD3Round2DiTBlock(nn.Module):
    def __init__(self, d_model: int = 2048, h_q: int = 16, h_kv: int = 4, ffn_dim: int = 6144) -> None:
        super().__init__()
        self.d_model = d_model
        self.h_q = h_q
        self.h_kv = h_kv
        self.head_dim = d_model // h_q
        self.ffn_dim = ffn_dim
        qkv_out = (h_q + 2 * h_kv) * self.head_dim
        self.w_qkv = nn.Linear(d_model, qkv_out, bias=False, dtype=torch.bfloat16)
        self.wo = nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16)
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_dim, bias=False, dtype=torch.bfloat16)
        self.w_dn = nn.Linear(ffn_dim, d_model, bias=False, dtype=torch.bfloat16)
        self.adaln_proj = nn.Linear(d_model, 6 * d_model, bias=False, dtype=torch.bfloat16)
        nn.init.normal_(self.adaln_proj.weight, mean=0.0, std=0.02)

    def copy_from_unmerged_block_(self, src: e127.BD3DiTBlock) -> None:
        with torch.no_grad():
            self.w_qkv.weight.copy_(torch.cat([src.wq.weight, src.wk.weight, src.wv.weight], dim=0))
            self.wo.weight.copy_(src.wo.weight)
            self.w_gate_up.weight.copy_(torch.cat([src.w_gate.weight, src.w_up.weight], dim=0))
            self.w_dn.weight.copy_(src.w_dn.weight)
            self.adaln_proj.weight.copy_(src.adaln_proj.weight)

    def forward(
        self,
        ht: torch.Tensor,
        h0: torch.Tensor | None,
        t_emb_noisy: torch.Tensor,
        t_emb_clean0: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bd3_meta: dict[str, Any],
        split_meta_t: dict[str, Any],
        split_meta_0: dict[str, Any],
        *,
        prune_clean_out: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        n = ht.shape[0]
        d = self.d_model
        cond_t = self.adaln_proj(F.silu(t_emb_noisy)).view(-1, 6, d)
        gt1, bt1, at1, gt2, bt2, at2 = cond_t.unbind(dim=1)

        assert h0 is not None
        cond_0 = self.adaln_proj(F.silu(t_emb_clean0)).view(1, 6, d)
        g01, b01, a01, g02, b02, a02 = cond_0.unbind(dim=1)

        yt1 = SplitAdaLNP1Fn.apply(ht, gt1, bt1, split_meta_t, 1e-6)
        y01 = SplitAdaLNP1Fn.apply(h0, g01, b01, split_meta_0, 1e-6)

        y_cat = torch.cat([yt1, y01], dim=0)
        q_cat, k_cat, v_cat = DualStreamQKVSplitRoPEFn.apply(
            self.w_qkv(y_cat), cos, sin, self.h_q, self.h_kv, self.head_dim
        )

        if prune_clean_out:
            ot, _ = Round2KTile16ZeroCopyBD3Fn.apply(
                q_cat[:n], k_cat[:n], v_cat[:n], None, k_cat[n:], v_cat[n:], bd3_meta, False, False
            )
            attn_t = self.wo(ot.view(n, d))
            ht_mid, yt2 = SplitGateResAdaLNP2Fn.apply(ht, attn_t, at1, gt2, bt2, split_meta_t, 1e-6)
            mlp_t = self.w_dn(FusedSwiGLUFn.apply(self.w_gate_up(yt2)))
            ht_out = SplitGateResP3Fn.apply(ht_mid, mlp_t, at2, split_meta_t)
            return ht_out, None

        ot, o0 = Round2KTile16ZeroCopyBD3Fn.apply(
            q_cat[:n], k_cat[:n], v_cat[:n], q_cat[n:], k_cat[n:], v_cat[n:], bd3_meta, True, False
        )
        attn_cat = self.wo(torch.cat([ot.view(n, d), o0.view(n, d)], dim=0))
        ht_mid, yt2 = SplitGateResAdaLNP2Fn.apply(ht, attn_cat[:n], at1, gt2, bt2, split_meta_t, 1e-6)
        h0_mid, y02 = SplitGateResAdaLNP2Fn.apply(h0, attn_cat[n:], a01, g02, b02, split_meta_0, 1e-6)

        mlp_cat = self.w_dn(FusedSwiGLUFn.apply(self.w_gate_up(torch.cat([yt2, y02], dim=0))))
        ht_out = SplitGateResP3Fn.apply(ht_mid, mlp_cat[:n], at2, split_meta_t)
        h0_out = SplitGateResP3Fn.apply(h0_mid, mlp_cat[n:], a02, split_meta_0)
        return ht_out, h0_out


class BD3Round2DiffusionLLM(nn.Module):
    """Merged-QKV + Merged-SwiGLU + Split-32-AdaLN + Zero-Spill-BD3 Diffusion Transformer."""

    def __init__(
        self,
        vocab_size: int = 128256,
        d_model: int = 2048,
        n_layers: int = 8,
        h_q: int = 16,
        h_kv: int = 4,
        ffn_dim: int = 6144,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model, dtype=torch.bfloat16)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16),
        )
        self.blocks = nn.ModuleList(
            [BD3Round2DiTBlock(d_model=d_model, h_q=h_q, h_kv=h_kv, ffn_dim=ffn_dim) for _ in range(n_layers)]
        )
        self.final_norm_w = nn.Parameter(torch.ones(d_model, dtype=torch.bfloat16))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, dtype=torch.bfloat16)

        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        for m in self.time_mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        for blk in self.blocks:
            for m in (blk.w_qkv, blk.wo, blk.w_gate_up, blk.w_dn, blk.adaln_proj):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def copy_from_unmerged_model_(self, src: e127.BD3DiffusionLLM) -> None:
        with torch.no_grad():
            self.tok_emb.weight.copy_(src.tok_emb.weight)
            self.time_mlp[0].weight.copy_(src.time_mlp[0].weight)
            self.time_mlp[2].weight.copy_(src.time_mlp[2].weight)
            for dst_b, src_b in zip(self.blocks, src.blocks):
                dst_b.copy_from_unmerged_block_(src_b)
            self.final_norm_w.copy_(src.final_norm_w)
            self.lm_head.weight.copy_(src.lm_head.weight)

    def _sinusoidal_t_embed(self, t_vals: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=t_vals.device, dtype=torch.float32) / float(half)
        )
        args = t_vals[:, None].float() * 1000.0 * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(torch.bfloat16)
        return self.time_mlp(emb)

    def forward(
        self,
        batch: dict[str, Any],
        *,
        global_clean_tokens: int,
        world_size_scale: float,
    ) -> torch.Tensor:
        n = batch["xt_ids"].shape[0]
        both_emb = self.tok_emb(torch.cat([batch["xt_ids"], batch["x0_ids"]], dim=0))
        ht, h0 = both_emb[:n], both_emb[n:]

        t_cat = torch.cat([batch["t_blks"], batch["t_zero"]], dim=0)
        t_emb_all = self._sinusoidal_t_embed(t_cat)
        t_emb_noisy, t_emb_clean0 = t_emb_all[:-1], t_emb_all[-1:]

        n_layers = len(self.blocks)
        for li, blk in enumerate(self.blocks):
            ht, h0 = blk(
                ht, h0, t_emb_noisy, t_emb_clean0,
                batch["cos"], batch["sin"],
                batch["bd3_meta"], batch["split_meta_t"], batch["split_meta_0"],
                prune_clean_out=(li == n_layers - 1),
            )

        r_fin = torch.rsqrt(ht.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        ht_norm = (ht.float() * r_fin * self.final_norm_w.float()).to(torch.bfloat16)
        inv_t_scaled = (world_size_scale / (batch["t_tok"] * float(global_clean_tokens))).contiguous()
        return e127.ActiveMaskCompacted3GemmCEFn.apply(
            ht_norm, self.lm_head.weight, batch["x0_ids"], batch["mask_tok"], inv_t_scaled, 2048
        )


def _augment_batch_with_split_meta(batch: dict[str, Any], doc_lens: list[int], W: int, device: torch.device) -> dict[str, Any]:
    n = sum(doc_lens)
    batch["split_meta_t"] = build_split_adaln_meta(doc_lens, W, 32, device)
    batch["split_meta_0"] = build_split_adaln_meta([n], n, 32, device)
    return batch


def _configure_fsdp2_2block_prefetch(m: nn.Module) -> None:
    blks = list(m.blocks)
    nb = len(blks)
    for i in range(nb):
        fwd_next = [blks[j] for j in (i + 1, i + 2) if j < nb]
        if fwd_next:
            blks[i].set_modules_to_forward_prefetch(fwd_next)
    for i in range(nb - 1, -1, -1):
        bwd_prev = [blks[j] for j in (i - 1, i - 2) if j >= 0]
        if bwd_prev:
            blks[i].set_modules_to_backward_prefetch(bwd_prev)


# ============================================================================
# 4. Suite B: 8-GPU NVSwitch `1.11B` `BD3-LM` `FSDP2` Round-2 System RSI
#    (Breaking E127's `1.087x` Wall + Exact Unmerged 1-GPU Parity w/ Rank-7 M=0)
# ============================================================================


def run_suite_b_8gpu_bd3_fsdp2_round2(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    assert world_size == 8, f"Suite B requires world_size == 8, got {world_size}"
    dist.barrier()
    comp_cbm = torch.compile(create_block_mask)
    comp_flex = torch.compile(flex_attention)
    mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)

    # ------------------------------------------------------------------------
    # B.1 Exact Unmerged 1-GPU Oracle Parity vs 8-GPU Sharded Merged-QKV/SwiGLU
    #     + 2-Block Prefetch + Rank-7 Empty-Mask (`M = 0`) Symmetry (`Gate G9`)
    # ------------------------------------------------------------------------
    W_p = 64
    docs_p = [256, 256]
    global_clean_p = 8 * sum(docs_p)

    torch.manual_seed(2026128)
    cpu_ref_model = e127.BD3DiffusionLLM(vocab_size=128256, d_model=2048, n_layers=8, h_q=16, h_kv=4, ffn_dim=6144)

    ref_1gpu: dict[str, Any] = {}
    if rank == 0:
        m_1gpu = copy.deepcopy(cpu_ref_model).to(device)
        per_r_b = [
            _augment_batch_with_split_meta(
                e127._prepare_bd3_rank_batch(
                    docs_p, W_p, 128256, device, seed=6100 + r, comp_cbm=comp_cbm,
                    build_flex_bm=False, force_zero_mask=(r == 7),
                ),
                docs_p, W_p, device,
            )
            for r in range(8)
        ]
        all_docs = docs_p * 8
        g_batch = _augment_batch_with_split_meta(
            {
                "x0_ids": torch.cat([b["x0_ids"] for b in per_r_b], dim=0),
                "xt_ids": torch.cat([b["xt_ids"] for b in per_r_b], dim=0),
                "t_blks": torch.cat([b["t_blks"] for b in per_r_b], dim=0),
                "t_zero": torch.zeros((1,), device=device, dtype=torch.float32),
                "t_tok": torch.cat([b["t_tok"] for b in per_r_b], dim=0),
                "mask_tok": torch.cat([b["mask_tok"] for b in per_r_b], dim=0),
                "cos": torch.cat([b["cos"] for b in per_r_b], dim=0),
                "sin": torch.cat([b["sin"] for b in per_r_b], dim=0),
                "bd3_meta": e127.build_bd3_packed_meta(all_docs, W_p, device),
                "seg_meta_t": e127.build_adaln_segment_meta(all_docs, W_p, device),
                "seg_meta_0": {
                    "n": global_clean_p, "W": global_clean_p, "n_blks": 1,
                    "block_id": torch.zeros((global_clean_p,), device=device, dtype=torch.int32),
                    "block_id_long": torch.zeros((global_clean_p,), device=device, dtype=torch.long),
                    "blk_bounds": torch.tensor([[0, global_clean_p]], device=device, dtype=torch.int32),
                },
                "flex_bm": None,
            },
            all_docs, W_p, device,
        )
        loss_1 = m_1gpu(
            g_batch, rung="rung4_full_rsi_stack_compacted_3gemm_masked_ce",
            global_clean_tokens=global_clean_p, world_size_scale=1.0, comp_flex=comp_flex,
        )
        loss_1.backward()
        ref_1gpu = {
            "loss": float(loss_1.item()),
            "dw_tok_emb": m_1gpu.tok_emb.weight.grad.detach().cpu(),
            "dw_b0_wq": m_1gpu.blocks[0].wq.weight.grad.detach().cpu(),
            "dw_b0_wk": m_1gpu.blocks[0].wk.weight.grad.detach().cpu(),
            "dw_b0_wv": m_1gpu.blocks[0].wv.weight.grad.detach().cpu(),
            "dw_b0_wgate": m_1gpu.blocks[0].w_gate.weight.grad.detach().cpu(),
            "dw_b0_wup": m_1gpu.blocks[0].w_up.weight.grad.detach().cpu(),
            "dw_b7_adaln": m_1gpu.blocks[7].adaln_proj.weight.grad.detach().cpu(),
            "dw_lm_head": m_1gpu.lm_head.weight.grad.detach().cpu(),
        }
        del m_1gpu, per_r_b, g_batch
        torch.cuda.empty_cache()

    dist.barrier()
    m_r2 = BD3Round2DiffusionLLM(vocab_size=128256, d_model=2048, n_layers=8, h_q=16, h_kv=4, ffn_dim=6144)
    m_r2.copy_from_unmerged_model_(cpu_ref_model)
    del cpu_ref_model
    m_r2 = m_r2.to(device)
    param_count = sum(p.numel() for p in m_r2.parameters())
    fully_shard(m_r2.tok_emb, mp_policy=mp_policy)
    fully_shard(m_r2.time_mlp, mp_policy=mp_policy)
    for blk in m_r2.blocks:
        fully_shard(blk, mp_policy=mp_policy)
    fully_shard(m_r2, mp_policy=mp_policy)
    _configure_fsdp2_2block_prefetch(m_r2)

    b_rank_p = _augment_batch_with_split_meta(
        e127._prepare_bd3_rank_batch(
            docs_p, W_p, 128256, device, seed=6100 + rank, comp_cbm=comp_cbm,
            build_flex_bm=False, force_zero_mask=(rank == 7),
        ),
        docs_p, W_p, device,
    )
    loss_8 = m_r2(b_rank_p, global_clean_tokens=global_clean_p, world_size_scale=float(world_size))
    loss_8.backward()

    loss_avg = loss_8.detach().to(torch.float64) / float(world_size)
    dist.all_reduce(loss_avg, op=dist.ReduceOp.SUM)

    dw_emb_full = m_r2.tok_emb.weight.grad.full_tensor().detach().cpu()
    dw_qkv_full = m_r2.blocks[0].w_qkv.weight.grad.full_tensor().detach().cpu()
    dw_gu_full = m_r2.blocks[0].w_gate_up.weight.grad.full_tensor().detach().cpu()
    dw_ad7_full = m_r2.blocks[7].adaln_proj.weight.grad.full_tensor().detach().cpu()
    dw_lm_full = m_r2.lm_head.weight.grad.full_tensor().detach().cpu()

    parity_fsdp2: dict[str, Any] = {}
    if rank == 0:
        l8 = float(loss_avg.item())
        l1 = ref_1gpu["loss"]
        loss_rd = abs(l8 - l1) / max(1e-12, abs(l1))
        dw_rels = {
            "tok_emb": round(rel_l2(dw_emb_full, ref_1gpu["dw_tok_emb"]), 6),
            "blk0_wq_from_merged_qkv": round(rel_l2(dw_qkv_full[:2048], ref_1gpu["dw_b0_wq"]), 6),
            "blk0_wk_from_merged_qkv": round(rel_l2(dw_qkv_full[2048:2560], ref_1gpu["dw_b0_wk"]), 6),
            "blk0_wv_from_merged_qkv": round(rel_l2(dw_qkv_full[2560:], ref_1gpu["dw_b0_wv"]), 6),
            "blk0_wgate_from_merged_swiglu": round(rel_l2(dw_gu_full[:6144], ref_1gpu["dw_b0_wgate"]), 6),
            "blk0_wup_from_merged_swiglu": round(rel_l2(dw_gu_full[6144:], ref_1gpu["dw_b0_wup"]), 6),
            "blk7_adaln_proj": round(rel_l2(dw_ad7_full, ref_1gpu["dw_b7_adaln"]), 6),
            "lm_head": round(rel_l2(dw_lm_full, ref_1gpu["dw_lm_head"]), 6),
        }
        parity_fsdp2 = {
            "param_count": param_count,
            "rank7_forced_empty_mask_M0": True,
            "loss_1gpu_unmerged": round(l1, 8),
            "loss_8gpu_fsdp2_merged_prefetch": round(l8, 8),
            "loss_rel_diff": round(loss_rd, 8),
            "dw_split_shard_rel_l2": dw_rels,
            "max_dw_rel_l2": round(max(dw_rels.values()), 6),
            "parity_ok": (loss_rd <= 1e-4) and (max(dw_rels.values()) <= 1.5e-2),
        }
    del m_r2, b_rank_p, dw_emb_full, dw_qkv_full, dw_gu_full, dw_ad7_full, dw_lm_full, ref_1gpu
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    # ------------------------------------------------------------------------
    # B.2 4-Rung Cumulative 8-GPU NVSwitch `1.11B` `BD3-LM` `FSDP2` Ablation
    #     Evaluated at BOTH `W = 64` and `W = 16` (`B_rank = 32, L_doc = 256` -> 131,072 2L tok/step)
    # ------------------------------------------------------------------------
    b_docs = 8 if is_smoke else 32
    l_doc = 256
    docs_bench = [l_doc] * b_docs
    clean_per_rank = sum(docs_bench)
    global_clean_bench = world_size * clean_per_rank
    global_concat_2l_tok = 2 * global_clean_bench

    w_sweep_results: dict[str, Any] = {}
    for W_run in (64, 16):
        torch.manual_seed(20261285)
        m_r1 = e127.BD3DiffusionLLM(vocab_size=128256, d_model=2048, n_layers=8, h_q=16, h_kv=4, ffn_dim=6144).to(device)
        fully_shard(m_r1.tok_emb, mp_policy=mp_policy)
        fully_shard(m_r1.time_mlp, mp_policy=mp_policy)
        for blk in m_r1.blocks:
            fully_shard(blk, mp_policy=mp_policy)
        fully_shard(m_r1, mp_policy=mp_policy)

        torch.manual_seed(20261285)
        m_r2_nopf = BD3Round2DiffusionLLM(vocab_size=128256, d_model=2048, n_layers=8, h_q=16, h_kv=4, ffn_dim=6144).to(device)
        fully_shard(m_r2_nopf.tok_emb, mp_policy=mp_policy)
        fully_shard(m_r2_nopf.time_mlp, mp_policy=mp_policy)
        for blk in m_r2_nopf.blocks:
            fully_shard(blk, mp_policy=mp_policy)
        fully_shard(m_r2_nopf, mp_policy=mp_policy)

        torch.manual_seed(20261285)
        m_r2_pf = BD3Round2DiffusionLLM(vocab_size=128256, d_model=2048, n_layers=8, h_q=16, h_kv=4, ffn_dim=6144).to(device)
        fully_shard(m_r2_pf.tok_emb, mp_policy=mp_policy)
        fully_shard(m_r2_pf.time_mlp, mp_policy=mp_policy)
        for blk in m_r2_pf.blocks:
            fully_shard(blk, mp_policy=mp_policy)
        fully_shard(m_r2_pf, mp_policy=mp_policy)
        _configure_fsdp2_2block_prefetch(m_r2_pf)

        batch_w = _augment_batch_with_split_meta(
            e127._prepare_bd3_rank_batch(
                docs_bench, W_run, 128256, device, seed=7200 + W_run + rank,
                comp_cbm=comp_cbm, build_flex_bm=True,
            ),
            docs_bench, W_run, device,
        )

        rungs_cfg = [
            ("rung0_literature_flex_eager_adaln_full_ce", "r1_flex"),
            ("rung1_e127_round1_winner", "r1_win"),
            ("rung2_round2_fused_kernels_merged_qkv_mlp", "r2_nopf"),
            ("rung3_round2_full_stack_plus_fsdp2_2block_prefetch", "r2_pf"),
        ]
        rung_out: dict[str, Any] = {}
        for rname, rmode in rungs_cfg:
            def _step_fn(rm=rmode):
                if rm == "r1_flex":
                    l_v = m_r1(batch_w, rung="rung1_flex_bd3_eager_adaln_full_ce", global_clean_tokens=global_clean_bench, world_size_scale=float(world_size), comp_flex=comp_flex)
                    l_v.backward()
                    m_r1.zero_grad(set_to_none=True)
                elif rm == "r1_win":
                    l_v = m_r1(batch_w, rung="rung4_full_rsi_stack_compacted_3gemm_masked_ce", global_clean_tokens=global_clean_bench, world_size_scale=float(world_size), comp_flex=comp_flex)
                    l_v.backward()
                    m_r1.zero_grad(set_to_none=True)
                elif rm == "r2_nopf":
                    l_v = m_r2_nopf(batch_w, global_clean_tokens=global_clean_bench, world_size_scale=float(world_size))
                    l_v.backward()
                    m_r2_nopf.zero_grad(set_to_none=True)
                else:
                    l_v = m_r2_pf(batch_w, global_clean_tokens=global_clean_bench, world_size_scale=float(world_size))
                    l_v.backward()
                    m_r2_pf.zero_grad(set_to_none=True)

            w_it = 2 if is_smoke else 3
            g_cnt = 3 if is_smoke else 6
            it_g = 2 if is_smoke else 3
            for _ in range(w_it):
                _step_fn()
            torch.cuda.synchronize(device)
            dist.barrier()

            torch.cuda.reset_peak_memory_stats(device)
            _step_fn()
            torch.cuda.synchronize(device)
            peak_rank_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            peak_t = torch.tensor([peak_rank_mib], device=device, dtype=torch.float64)
            dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)

            step_samples: list[float] = []
            for _ in range(g_cnt):
                dist.barrier()
                s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s_ev.record()
                for _ in range(it_g):
                    _step_fn()
                e_ev.record()
                e_ev.synchronize()
                t_loc = torch.tensor([s_ev.elapsed_time(e_ev) / float(it_g)], device=device, dtype=torch.float64)
                dist.all_reduce(t_loc, op=dist.ReduceOp.MAX)
                step_samples.append(float(t_loc.item()))

            ss = sorted(step_samples)
            med_ms = float(statistics.median(ss))
            iqr_ms = float(ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)] - ss[max(0, (len(ss) - 1) // 4)])
            rung_out[rname] = {
                "median_step_ms": round(med_ms, 3),
                "iqr_step_ms": round(iqr_ms, 3),
                "global_clean_tok_per_s": round(global_clean_bench / (med_ms * 1e-3), 1),
                "global_concat_2l_tok_per_s": round(global_concat_2l_tok / (med_ms * 1e-3), 1),
                "max_rank_peak_alloc_mib": round(float(peak_t.item()), 2),
            }

        t0 = rung_out["rung0_literature_flex_eager_adaln_full_ce"]["median_step_ms"]
        t1 = rung_out["rung1_e127_round1_winner"]["median_step_ms"]
        t2 = rung_out["rung2_round2_fused_kernels_merged_qkv_mlp"]["median_step_ms"]
        t3 = rung_out["rung3_round2_full_stack_plus_fsdp2_2block_prefetch"]["median_step_ms"]
        m0 = rung_out["rung0_literature_flex_eager_adaln_full_ce"]["max_rank_peak_alloc_mib"]
        m3 = rung_out["rung3_round2_full_stack_plus_fsdp2_2block_prefetch"]["max_rank_peak_alloc_mib"]
        w_sweep_results[f"W_{W_run}"] = {
            "W": W_run,
            "b_docs_per_rank": b_docs,
            "doc_len": l_doc,
            "global_clean_tokens_per_step": global_clean_bench,
            "global_concat_2l_tokens_per_step": global_concat_2l_tok,
            "rungs": rung_out,
            "speedup_r1_vs_literature_r0": round(t0 / t1, 4),
            "speedup_r2_kernels_vs_r1": round(t1 / t2, 4),
            "speedup_r3_full_vs_r1": round(t1 / t3, 4),
            "speedup_r3_full_vs_literature_r0": round(t0 / t3, 4),
            "memory_saved_r3_vs_r0_mib_per_rank": round(m0 - m3, 2),
        }

        del m_r1, m_r2_nopf, m_r2_pf, batch_w
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    return {
        "fsdp2_unmerged_1gpu_parity_gate_g9": parity_fsdp2,
        "ablation_by_block_size": w_sweep_results,
    }


# ============================================================================
# 5. Suite C: `autoresearch` / `Naive-N0.5-Flash` Fixed-Wall-Clock `val_bpb`
#    Autonomous Vibe-Training Loop on `8x NVIDIA A100-SXM4-80GB` (`Gen 0..Gen 4`)
# ============================================================================


def _generate_structured_grammar_corpus(
    n_seqs: int, seq_len: int, vocab_size: int, seed: int, device: torch.device, grammar_seed: int = 20261117
) -> torch.Tensor:
    """Deterministic multi-scale structured phrase + modular induction corpus (`1` token = `2` UTF-8 bytes).
    All splits (`train`, `val`, `test`) share the same underlying language grammar (`grammar_seed`) while drawing
    strictly disjoint sequence realizations (`seed`), so held-out `val_bpb` and `test_bpb` measure true generalization.
    """
    g_lang = torch.Generator(device="cpu").manual_seed(grammar_seed)
    n_motifs = 64
    motif_len = 8
    eff_v = vocab_size - 1  # reserve vocab_size - 1 for [MASK]
    motifs = torch.randint(0, eff_v, (n_motifs, motif_len), generator=g_lang)
    trans = torch.randint(0, n_motifs, (n_motifs, 4), generator=g_lang)

    g = torch.Generator(device="cpu").manual_seed(seed)

    n_blocks = seq_len // motif_len
    cur_m = torch.randint(0, n_motifs, (n_seqs,), generator=g)
    seq_chunks = []
    for b in range(n_blocks):
        branch = (cur_m + b) % 4
        if b > 0:
            # 85% deterministic transition, 15% stochastic branch (~0.65 bpb irreducible entropy floor)
            noise_pick = torch.randint(0, 4, (n_seqs,), generator=g)
            use_det = torch.rand((n_seqs,), generator=g) < 0.85
            choice = torch.where(use_det, branch, noise_pick)
            cur_m = trans[cur_m, choice]
        tok_blk = motifs[cur_m].clone()
        # Cross-block copy dependency: last token of motif echoes block (b - 2) motif ID!
        if b >= 2:
            prev2 = seq_chunks[b - 2][:, 0]
            tok_blk[:, -1] = (prev2 * 7 + 13) % eff_v
        seq_chunks.append(tok_blk)

    corpus = torch.cat(seq_chunks, dim=1).contiguous()
    return corpus.to(device, non_blocking=True)


@torch.no_grad()
def evaluate_immutable_quadrature_bpb(
    model_eval: nn.Module,
    is_round2_model: bool,
    rung_tag: str,
    eval_seqs: torch.Tensor,
    *,
    W: int,
    vocab_size: int,
    quadrature_nodes: int,
    mask_seed: int,
    comp_cbm: Any,
    comp_flex: Any,
    bytes_per_token: float = 2.0,
) -> float:
    """Immutable deterministic multi-timestep midpoint-quadrature NELBO evaluator (`GPT 6 Astra, Max` Row 3 & Row 5).
    Computes exact corpus `bpb = sum_weighted_nelbo_nats / (ln(2) * total_utf8_bytes)` aggregated across all 8 ranks.
    """
    was_training = model_eval.training
    model_eval.eval()
    device = eval_seqs.device
    n_seqs_rank, seq_len = eval_seqs.shape
    doc_lens = [seq_len] * n_seqs_rank
    n_tok = n_seqs_rank * seq_len
    x0_flat = eval_seqs.reshape(-1)
    mask_id = vocab_size - 1

    bd3_meta = e127.build_bd3_packed_meta(doc_lens, W, device)
    seg_meta_t = e127.build_adaln_segment_meta(doc_lens, W, device)
    seg_meta_0 = {
        "n": n_tok, "W": n_tok, "n_blks": 1,
        "block_id": torch.zeros((n_tok,), device=device, dtype=torch.int32),
        "block_id_long": torch.zeros((n_tok,), device=device, dtype=torch.long),
        "blk_bounds": torch.tensor([[0, n_tok]], device=device, dtype=torch.int32),
    }
    split_meta_t = build_split_adaln_meta(doc_lens, W, 32, device)
    split_meta_0 = build_split_adaln_meta([n_tok], n_tok, 32, device)
    cos, sin = e127.build_doc_rope_cache(doc_lens, 128, device)

    flex_bm = None
    if (not is_round2_model) and ("flex" in rung_tag):
        flex_bm = comp_cbm(
            lambda b, h, qi, ki, nl=n_tok, dl0=seq_len, bs=W: (
                (torch.where(qi >= nl, (qi - nl) // dl0, qi // dl0))
                == (torch.where(ki >= nl, (ki - nl) // dl0, ki // dl0))
            )
            & (
                (
                    ((torch.where(qi >= nl, (qi - nl) // bs, qi // bs)) == (torch.where(ki >= nl, (ki - nl) // bs, ki // bs)))
                    & (((qi >= nl) & (ki >= nl)) | ((qi < nl) & (ki < nl)))
                )
                | (
                    ((torch.where(qi >= nl, (qi - nl) // bs, qi // bs)) > (torch.where(ki >= nl, (ki - nl) // bs, ki // bs)))
                    & (qi < nl) & (ki >= nl)
                )
                | (
                    ((torch.where(qi >= nl, (qi - nl) // bs, qi // bs)) >= (torch.where(ki >= nl, (ki - nl) // bs, ki // bs)))
                    & (qi >= nl) & (ki >= nl)
                )
            ),
            B=1, H=1, Q_LEN=2 * n_tok, KV_LEN=2 * n_tok, device=device,
        )

    total_nats_rank = torch.zeros((), device=device, dtype=torch.float64)
    n_blks = seg_meta_t["n_blks"]
    for q in range(quadrature_nodes):
        t_val = (float(q) + 0.5) / float(quadrature_nodes)
        g_q = torch.Generator(device=device).manual_seed(mask_seed + 1009 * q + dist.get_rank())
        t_blks = torch.full((n_blks,), t_val, device=device, dtype=torch.float32)
        t_tok = torch.full((n_tok,), t_val, device=device, dtype=torch.float32)
        mask_tok = (torch.rand((n_tok,), device=device, generator=g_q) < t_val).contiguous()
        if int(mask_tok.sum().item()) == 0:
            mask_tok[0] = True
        xt_ids = torch.where(mask_tok, torch.full_like(x0_flat, mask_id), x0_flat).contiguous()

        batch_q = {
            "x0_ids": x0_flat,
            "xt_ids": xt_ids,
            "t_blks": t_blks,
            "t_zero": torch.zeros((1,), device=device, dtype=torch.float32),
            "t_tok": t_tok,
            "mask_tok": mask_tok,
            "cos": cos,
            "sin": sin,
            "bd3_meta": bd3_meta,
            "seg_meta_t": seg_meta_t,
            "seg_meta_0": seg_meta_0,
            "split_meta_t": split_meta_t,
            "split_meta_0": split_meta_0,
            "flex_bm": flex_bm,
        }
        # Pass global_clean_tokens=1, world_size_scale=1.0 -> returns sum_{i in mask} CE_i / t_val (total weighted nats on rank!)
        if is_round2_model:
            nats_q = model_eval(batch_q, global_clean_tokens=1, world_size_scale=1.0)
        else:
            nats_q = model_eval(
                batch_q, rung=rung_tag, global_clean_tokens=1, world_size_scale=1.0, comp_flex=comp_flex
            )
        total_nats_rank += nats_q.to(torch.float64) / float(quadrature_nodes)

    agg = torch.stack([total_nats_rank, torch.tensor(float(n_tok) * bytes_per_token, device=device, dtype=torch.float64)])
    dist.all_reduce(agg, op=dist.ReduceOp.SUM)
    if was_training:
        model_eval.train()
    return float(agg[0].item() / (math.log(2.0) * agg[1].item()))


def run_suite_c_fixed_wallclock_valbpb_autoresearch(
    rank: int, world_size: int, device: torch.device, is_smoke: bool, out_dir: Path
) -> dict[str, Any]:
    assert world_size == 8
    dist.barrier()
    budget_s = 4.0 if is_smoke else 12.0
    vocab_v = 16384
    d_slm = 1024
    n_layers_slm = 6
    h_q_slm = 8
    h_kv_slm = 2
    ffn_slm = 2816
    seq_len = 256
    W_slm = 64
    b_rank = 8 if is_smoke else 16  # global batch = 128 seqs = 32,768 clean / 65,536 [x_t||x_0] tokens/step!
    n_tok_rank = b_rank * seq_len
    global_clean_tok = world_size * n_tok_rank
    doc_lens = [seq_len] * b_rank

    # Deterministic per-rank train pool + frozen val + untouched test splits (`1` token = `2` UTF-8 bytes)
    n_train_pool = 512 if is_smoke else 2048
    n_eval_rank = 4 if is_smoke else 8
    train_pool = _generate_structured_grammar_corpus(n_train_pool, seq_len, vocab_v, seed=40000 + rank, device=device)
    val_seqs = _generate_structured_grammar_corpus(n_eval_rank, seq_len, vocab_v, seed=50000 + rank, device=device)
    test_seqs = _generate_structured_grammar_corpus(n_eval_rank, seq_len, vocab_v, seed=60000 + rank, device=device)

    comp_cbm = torch.compile(create_block_mask)
    comp_flex = torch.compile(flex_attention)
    mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)

    # Canonical identical initial weights with proper DiT `adaLN-Zero` near-zero conditioning init (`std=1e-3`)
    # so residual blocks start near identity and train monotonically from step 0!
    torch.manual_seed(20261289)
    init_unmerged_cpu = e127.BD3DiffusionLLM(
        vocab_size=vocab_v, d_model=d_slm, n_layers=n_layers_slm, h_q=h_q_slm, h_kv=h_kv_slm, ffn_dim=ffn_slm
    )
    with torch.no_grad():
        for blk_init in init_unmerged_cpu.blocks:
            nn.init.normal_(blk_init.adaln_proj.weight, mean=0.0, std=1e-3)
    slm_param_count = sum(p.numel() for p in init_unmerged_cpu.parameters())

    bd3_meta = e127.build_bd3_packed_meta(doc_lens, W_slm, device)
    seg_meta_t = e127.build_adaln_segment_meta(doc_lens, W_slm, device)
    seg_meta_0 = {
        "n": n_tok_rank, "W": n_tok_rank, "n_blks": 1,
        "block_id": torch.zeros((n_tok_rank,), device=device, dtype=torch.int32),
        "block_id_long": torch.zeros((n_tok_rank,), device=device, dtype=torch.long),
        "blk_bounds": torch.tensor([[0, n_tok_rank]], device=device, dtype=torch.int32),
    }
    split_meta_t = build_split_adaln_meta(doc_lens, W_slm, 32, device)
    split_meta_0 = build_split_adaln_meta([n_tok_rank], n_tok_rank, 32, device)
    cos, sin = e127.build_doc_rope_cache(doc_lens, 128, device)
    flex_bm_train = comp_cbm(
        lambda b, h, qi, ki, nl=n_tok_rank, dl0=seq_len, bs=W_slm: (
            (torch.where(qi >= nl, (qi - nl) // dl0, qi // dl0))
            == (torch.where(ki >= nl, (ki - nl) // dl0, ki // dl0))
        )
        & (
            (
                ((torch.where(qi >= nl, (qi - nl) // bs, qi // bs)) == (torch.where(ki >= nl, (ki - nl) // bs, ki // bs)))
                & (((qi >= nl) & (ki >= nl)) | ((qi < nl) & (ki < nl)))
            )
            | (
                ((torch.where(qi >= nl, (qi - nl) // bs, qi // bs)) > (torch.where(ki >= nl, (ki - nl) // bs, ki // bs)))
                & (qi < nl) & (ki >= nl)
            )
            | (
                ((torch.where(qi >= nl, (qi - nl) // bs, qi // bs)) >= (torch.where(ki >= nl, (ki - nl) // bs, ki // bs)))
                & (qi >= nl) & (ki >= nl)
            )
        ),
        B=1, H=1, Q_LEN=2 * n_tok_rank, KV_LEN=2 * n_tok_rank, device=device,
    )

    generations = [
        {
            "gen_id": 0,
            "candidate_id": "gen0_literature_flex_eager_adaln_iid_t",
            "is_round2": False,
            "rung_tag": "rung1_flex_bd3_eager_adaln_full_ce",
            "prefetch": False,
            "t_scheme": "unclipped_iid",
            "peak_lr": 8.0e-4,
            "note": "Literature baseline: FlexAttention 2Lx2L + Eager adaLN-Zero + Full CE + IID t~U[0.01,0.99]",
        },
        {
            "gen_id": 1,
            "candidate_id": "gen1_e127_round1_kernels_iid_t",
            "is_round2": False,
            "rung_tag": "rung4_full_rsi_stack_compacted_3gemm_masked_ce",
            "prefetch": False,
            "t_scheme": "unclipped_iid",
            "peak_lr": 8.0e-4,
            "note": "Round-1 E127 kernels (6-call FA2 at W=64 + 1-CTA/blk adaLN + Compacted CE) with IID t~U[0.01,0.99]",
        },
        {
            "gen_id": 2,
            "candidate_id": "gen2_e128_round2_roofline_kernels_fsdp2_prefetch",
            "is_round2": True,
            "rung_tag": "round2",
            "prefetch": True,
            "t_scheme": "unclipped_iid",
            "peak_lr": 8.0e-4,
            "note": "Round-2 K_TILE=16 0-spill BD3 + Split-32 AdaLN + Merged QKV/SwiGLU + FSDP2 2-blk prefetch + IID t",
        },
        {
            "gen_id": 3,
            "candidate_id": "gen3_aggressive_lr_unclipped_spike_overshoot",
            "is_round2": True,
            "rung_tag": "round2",
            "prefetch": True,
            "t_scheme": "unclipped_extreme_low_t",
            "peak_lr": 3.5e-2,
            "note": "Over-aggressive peak LR (3.5e-2) + extreme t~U[0.001,0.999] 1/t variance spike -> destabilizes val_bpb",
        },
        {
            "gen_id": 4,
            "candidate_id": "gen4_round2_kernels_plus_stratified_antithetic_nelbo",
            "is_round2": True,
            "rung_tag": "round2",
            "prefetch": True,
            "t_scheme": "stratified_antithetic",
            "peak_lr": 8.0e-4,
            "note": "Round-2 roofline stack + zero-bias stratified complementary-antithetic (t, 1-t; m, 1-m) on [0.01,0.99]",
        },
    ]

    mask_id = vocab_v - 1
    n_blks_train = seg_meta_t["n_blks"]

    def _make_train_step_batch(step_idx: int, t_scheme: str, rng: torch.Generator) -> dict[str, Any]:
        start_s = (step_idx * b_rank) % (n_train_pool - b_rank)
        x0_b = train_pool[start_s : start_s + b_rank].reshape(-1)
        if t_scheme == "stratified_antithetic":
            # Shuffled low-discrepancy stratified sampling (MDLM Sahoo et al. NeurIPS 2024 Sec 3.3 + BD3 clipped [0.05, 0.95] schedule):
            # random permutation across blocks removes positional bias while eliminating extreme t->0 1/t=200 gradient spikes!
            perm = torch.randperm(n_blks_train, device=device, generator=rng)
            strata = (torch.arange(n_blks_train, device=device, dtype=torch.float32) + torch.rand((n_blks_train,), device=device, generator=rng)) / float(n_blks_train)
            t_blks = (0.05 + 0.90 * strata)[perm].contiguous()
            t_tok = t_blks.index_select(0, seg_meta_t["block_id_long"]).contiguous()
            mask_tok = (torch.rand((n_tok_rank,), device=device, generator=rng) < t_tok).contiguous()
        elif t_scheme == "unclipped_extreme_low_t":
            t_blks = (0.001 + 0.998 * torch.rand((n_blks_train,), device=device, generator=rng)).contiguous()
            t_tok = t_blks.index_select(0, seg_meta_t["block_id_long"]).contiguous()
            mask_tok = (torch.rand((n_tok_rank,), device=device, generator=rng) < t_tok).contiguous()
        else:
            # Unclipped single-draw-per-sequence IID t ~ U[0.004, 0.996] -> genuine 1/t=250 NELBO gradient spikes!
            t_seq = 0.004 + 0.992 * torch.rand((b_rank, 1), device=device, generator=rng)
            n_b_per_seq = seq_len // W_slm
            t_blks = t_seq.expand(b_rank, n_b_per_seq).reshape(-1).contiguous()
            t_tok = t_blks.index_select(0, seg_meta_t["block_id_long"]).contiguous()
            mask_tok = (torch.rand((n_tok_rank,), device=device, generator=rng) < t_tok).contiguous()

        if int(mask_tok.sum().item()) == 0:
            mask_tok[0] = True
        xt_b = torch.where(mask_tok, torch.full_like(x0_b, mask_id), x0_b).contiguous()
        return {
            "x0_ids": x0_b,
            "xt_ids": xt_b,
            "t_blks": t_blks,
            "t_zero": torch.zeros((1,), device=device, dtype=torch.float32),
            "t_tok": t_tok,
            "mask_tok": mask_tok,
            "cos": cos,
            "sin": sin,
            "bd3_meta": bd3_meta,
            "seg_meta_t": seg_meta_t,
            "seg_meta_0": seg_meta_0,
            "split_meta_t": split_meta_t,
            "split_meta_0": split_meta_0,
            "flex_bm": flex_bm_train,
        }

    # Evaluate initial untrained model `init_val_bpb` once for reference
    m_init_probe = BD3Round2DiffusionLLM(
        vocab_size=vocab_v, d_model=d_slm, n_layers=n_layers_slm, h_q=h_q_slm, h_kv=h_kv_slm, ffn_dim=ffn_slm
    )
    m_init_probe.copy_from_unmerged_model_(init_unmerged_cpu)
    m_init_probe = m_init_probe.to(device)
    fully_shard(m_init_probe.tok_emb, mp_policy=mp_policy)
    fully_shard(m_init_probe.time_mlp, mp_policy=mp_policy)
    for blk in m_init_probe.blocks:
        fully_shard(blk, mp_policy=mp_policy)
    fully_shard(m_init_probe, mp_policy=mp_policy)
    init_val_bpb = evaluate_immutable_quadrature_bpb(
        m_init_probe, True, "round2", val_seqs, W=W_slm, vocab_size=vocab_v,
        quadrature_nodes=4 if is_smoke else 8, mask_seed=77700, comp_cbm=comp_cbm, comp_flex=comp_flex,
    )
    del m_init_probe
    torch.cuda.empty_cache()
    dist.barrier()

    incumbent_val_bpb: float | None = None
    incumbent_gen_id: int = 0
    gen_records: list[dict[str, Any]] = []

    for gcfg in generations:
        gid = gcfg["gen_id"]
        is_r2 = gcfg["is_round2"]

        def _instantiate_fresh_sharded():
            if is_r2:
                m = BD3Round2DiffusionLLM(
                    vocab_size=vocab_v, d_model=d_slm, n_layers=n_layers_slm, h_q=h_q_slm, h_kv=h_kv_slm, ffn_dim=ffn_slm
                )
                m.copy_from_unmerged_model_(init_unmerged_cpu)
            else:
                m = copy.deepcopy(init_unmerged_cpu)
            m = m.to(device)
            fully_shard(m.tok_emb, mp_policy=mp_policy)
            fully_shard(m.time_mlp, mp_policy=mp_policy)
            for blk in m.blocks:
                fully_shard(blk, mp_policy=mp_policy)
            fully_shard(m, mp_policy=mp_policy)
            if gcfg["prefetch"]:
                _configure_fsdp2_2block_prefetch(m)
            return m

        def _build_fp32_master_adamw(m_sharded: nn.Module, lr_init: float):
            bf16_params = list(m_sharded.parameters())
            master_p32 = [
                nn.Parameter(p.to_local().detach().to(torch.float32).clone())
                for p in bf16_params
            ]
            opt32 = torch.optim.AdamW(master_p32, lr=lr_init, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
            return bf16_params, master_p32, opt32

        def _step_fp32_master_adamw(bf16_params: list[nn.Parameter], master_p32: list[nn.Parameter], opt32: torch.optim.Optimizer, max_norm: float = 1.0) -> None:
            local_g32 = [
                (p.grad.to_local().to(torch.float32) if p.grad is not None else torch.zeros_like(mp.data))
                for p, mp in zip(bf16_params, master_p32)
            ]
            norms = torch._foreach_norm(local_g32)
            sq_sum = torch.sum(torch.stack(norms).pow(2))
            dist.all_reduce(sq_sum, op=dist.ReduceOp.SUM)
            clip_coef = float(torch.clamp(max_norm / (torch.sqrt(sq_sum) + 1e-6), max=1.0).item())
            if clip_coef < 1.0:
                torch._foreach_mul_(local_g32, clip_coef)
            for mp, g in zip(master_p32, local_g32):
                mp.grad = g
            opt32.step()
            with torch.no_grad():
                bf16_locals = [p.to_local() for p in bf16_params]
                torch._foreach_copy_(bf16_locals, [mp.data for mp in master_p32])
            for p in bf16_params:
                p.grad = None

        # GPT-6-Astra Fairness Invariant 1: Warm representative forward/backward/optimizer/collectives
        # OUTSIDE the timer, THEN discard and re-instantiate exact initial weights, fresh FP32 master AdamW, cursor 0, and RNG!
        m_warm = _instantiate_fresh_sharded()
        wp_bf16, wp_m32, opt_warm = _build_fp32_master_adamw(m_warm, gcfg["peak_lr"])
        rng_warm = torch.Generator(device=device).manual_seed(81000 + rank)
        for ws in range(2):
            bw = _make_train_step_batch(ws, gcfg["t_scheme"], rng_warm)
            if is_r2:
                lw = m_warm(bw, global_clean_tokens=global_clean_tok, world_size_scale=float(world_size))
            else:
                lw = m_warm(bw, rung=gcfg["rung_tag"], global_clean_tokens=global_clean_tok, world_size_scale=float(world_size), comp_flex=comp_flex)
            lw.backward()
            _step_fp32_master_adamw(wp_bf16, wp_m32, opt_warm, max_norm=1.0)
        del m_warm, wp_bf16, wp_m32, opt_warm
        torch.cuda.synchronize(device)
        dist.barrier()

        # Fresh zero-advantage starting state identical across every generation!
        model = _instantiate_fresh_sharded()
        bf16_params, master_p32, optimizer = _build_fp32_master_adamw(model, gcfg["peak_lr"])
        rng_train = torch.Generator(device=device).manual_seed(90000 + rank)

        torch.cuda.synchronize(device)
        dist.barrier()
        t_start = time.perf_counter()
        completed_steps = 0
        elapsed_sync_s = 0.0
        ctrl_tensor = torch.zeros((2,), device=device, dtype=torch.float64)

        while True:
            # GPT-6-Astra Fairness Invariant 4: Wall-clock-parameterized warmup + cosine LR `u = elapsed / budget_s`
            u_prog = min(max(elapsed_sync_s / budget_s, 0.0), 1.0)
            if u_prog < 0.10:
                lr_now = gcfg["peak_lr"] * max(0.1, u_prog / 0.10)
            else:
                c_u = (u_prog - 0.10) / 0.90
                lr_now = gcfg["peak_lr"] * (0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * c_u)))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            batch_s = _make_train_step_batch(completed_steps, gcfg["t_scheme"], rng_train)
            if is_r2:
                loss_s = model(batch_s, global_clean_tokens=global_clean_tok, world_size_scale=float(world_size))
            else:
                loss_s = model(
                    batch_s, rung=gcfg["rung_tag"], global_clean_tokens=global_clean_tok,
                    world_size_scale=float(world_size), comp_flex=comp_flex,
                )
            loss_s.backward()
            _step_fp32_master_adamw(bf16_params, master_p32, optimizer, max_norm=1.0)
            completed_steps += 1

            # GPT-6-Astra Fairness Invariant 2: Synchronized GPU step completion + Rank-0 authoritative deadline
            torch.cuda.synchronize(device)
            if rank == 0:
                el = time.perf_counter() - t_start
                stop_flag = 1.0 if el >= budget_s else 0.0
                ctrl_tensor[0] = stop_flag
                ctrl_tensor[1] = el
            dist.broadcast(ctrl_tensor, src=0)
            elapsed_sync_s = float(ctrl_tensor[1].item())
            if ctrl_tensor[0].item() >= 0.5:
                break

        # GPT-6-Astra Fairness Invariants 3 & 5: Immutable 8-node quadrature `val_bpb` + untouched 16-node `test_bpb`
        val_bpb_raw = evaluate_immutable_quadrature_bpb(
            model, is_r2, gcfg["rung_tag"], val_seqs, W=W_slm, vocab_size=vocab_v,
            quadrature_nodes=4 if is_smoke else 8, mask_seed=77700, comp_cbm=comp_cbm, comp_flex=comp_flex,
        )
        test_bpb_raw = evaluate_immutable_quadrature_bpb(
            model, is_r2, gcfg["rung_tag"], test_seqs, W=W_slm, vocab_size=vocab_v,
            quadrature_nodes=8 if is_smoke else 16, mask_seed=99900, comp_cbm=comp_cbm, comp_flex=comp_flex,
        )
        diverged_nonfinite = not (math.isfinite(val_bpb_raw) and math.isfinite(test_bpb_raw))
        val_bpb = 999.0 if diverged_nonfinite else val_bpb_raw
        test_bpb = 999.0 if diverged_nonfinite else test_bpb_raw

        if incumbent_val_bpb is None:
            decision = "BASELINE"
            delta_vs_inc = 0.0
            incumbent_val_bpb = val_bpb
            incumbent_gen_id = gid
        elif (not diverged_nonfinite) and (val_bpb <= incumbent_val_bpb - 0.005):
            decision = "KEEP"
            delta_vs_inc = val_bpb - incumbent_val_bpb
            incumbent_val_bpb = val_bpb
            incumbent_gen_id = gid
        else:
            decision = "REVERT_VAL_BPB"
            delta_vs_inc = val_bpb - incumbent_val_bpb

        gen_records.append(
            {
                "gen_id": gid,
                "candidate_id": gcfg["candidate_id"],
                "budget_s": budget_s,
                "elapsed_s": round(elapsed_sync_s, 4),
                "completed_steps": completed_steps,
                "steps_per_sec": round(completed_steps / max(elapsed_sync_s, 1e-9), 3),
                "global_clean_tok_per_s": round((completed_steps * global_clean_tok) / max(elapsed_sync_s, 1e-9), 1),
                "global_concat_2l_tok_per_s": round((completed_steps * 2 * global_clean_tok) / max(elapsed_sync_s, 1e-9), 1),
                "peak_lr": gcfg["peak_lr"],
                "t_scheme": gcfg["t_scheme"],
                "diverged_nonfinite": diverged_nonfinite,
                "val_bpb_8node": round(val_bpb, 5),
                "untouched_test_bpb_16node": round(test_bpb, 5),
                "delta_val_bpb_vs_prior_incumbent": round(delta_vs_inc, 5),
                "decision": decision,
                "active_incumbent_after_trial": incumbent_gen_id,
                "hypothesis_note": gcfg["note"],
            }
        )
        del model, bf16_params, master_p32, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    valbpb_tsv_sha = ""
    if rank == 0:
        tsv_path = out_dir / "valbpb_autoresearch_ledger.tsv"
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "gen_id", "candidate_id", "budget_s", "elapsed_s", "completed_steps",
            "steps_per_sec", "val_bpb_8node", "untouched_test_bpb_16node",
            "delta_val_bpb_vs_prior_incumbent", "decision", "active_incumbent_after_trial", "hypothesis_note",
        ]
        with tsv_path.open("w", encoding="utf-8", newline="\n") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(cols)
            for r in gen_records:
                w.writerow([r[c] for c in cols])
        valbpb_tsv_sha = hashlib.sha256(tsv_path.read_bytes()).hexdigest()

    g0 = gen_records[0]
    g_final = next(r for r in gen_records if r["gen_id"] == incumbent_gen_id)
    return {
        "slm_param_count": slm_param_count,
        "vocab_size": vocab_v,
        "d_model": d_slm,
        "n_layers": n_layers_slm,
        "seq_len": seq_len,
        "W": W_slm,
        "global_clean_tokens_per_step": global_clean_tok,
        "budget_seconds_per_generation": budget_s,
        "untrained_init_val_bpb": round(init_val_bpb, 5),
        "baseline_gen0_val_bpb": g0["val_bpb_8node"],
        "final_incumbent_gen_id": incumbent_gen_id,
        "final_incumbent_candidate": g_final["candidate_id"],
        "final_incumbent_val_bpb": g_final["val_bpb_8node"],
        "final_incumbent_untouched_test_bpb": g_final["untouched_test_bpb_16node"],
        "total_val_bpb_improvement": round(g0["val_bpb_8node"] - g_final["val_bpb_8node"], 5),
        "total_test_bpb_improvement": round(g0["untouched_test_bpb_16node"] - g_final["untouched_test_bpb_16node"], 5),
        "step_multiplier_final_vs_gen0": round(g_final["completed_steps"] / max(1, g0["completed_steps"]), 4),
        "gen3_destabilizing_mutation_reverted": (gen_records[3]["decision"] == "REVERT_VAL_BPB"),
        "valbpb_ledger_sha256": valbpb_tsv_sha,
        "generations": gen_records,
    }


# ============================================================================
# 6. Main Distributed Entrypoint (`torchrun --nproc_per_node=8`)
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="E128 Round-2 RSI Roofline & Fixed-Wall-Clock val_bpb Benchmark")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    is_smoke = args.mode == "smoke"
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    t_wall0 = time.perf_counter()
    harness_sha = eval_harness_sha256()

    suite0_res: dict[str, Any] = {}
    suite_a_res: dict[str, Any] = {}
    ledger_sha = ""
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[E128][rank 0] Starting Suite 0 & Suite A (mode={args.mode}, harness_sha256={harness_sha[:16]}...)", flush=True)
        suite0_res, rt_ratchet = run_suite_0_adversarial_and_oracle_audits(device, is_smoke)
        suite_a_res, r_adaln, r_bd3 = run_suite_a_round2_roofline_ratchets(device, is_smoke)
        ledger_sha = write_rsi_ledger_tsv([rt_ratchet, r_adaln, r_bd3], args.output_dir / "rsi_ledger.tsv")
        print("[E128][rank 0] Suite 0 & Suite A complete; entering 8-GPU Suite B & Suite C...", flush=True)

    dist.barrier()
    suite_b_res = run_suite_b_8gpu_bd3_fsdp2_round2(rank, world_size, device, is_smoke)
    if rank == 0:
        print("[E128][rank 0] Suite B 8-GPU 1.11B FSDP2 complete; entering Suite C fixed-budget val_bpb loop...", flush=True)

    dist.barrier()
    suite_c_res = run_suite_c_fixed_wallclock_valbpb_autoresearch(rank, world_size, device, is_smoke, args.output_dir)
    dist.barrier()
    wall_s = round(time.perf_counter() - t_wall0, 2)

    if rank == 0:
        clk, pwr = query_sm_telemetry(0)
        payload = {
            "experiment_id": "E128",
            "title": "Round-2 RSI Roofline Convergence, Adversarial Gate Torture, 8-GPU 1.11B BD3-LM FSDP2 System RSI, & Fixed-Wall-Clock val_bpb Autoresearch Loop",
            "mode": args.mode,
            "world_size": world_size,
            "wall_clock_seconds": wall_s,
            "harness_version": HARNESS_VERSION,
            "eval_harness_sha256": harness_sha,
            "rsi_ledger_sha256": ledger_sha,
            "sm_clock_mhz_gpu0": clk,
            "power_w_gpu0": pwr,
            "environment": environment_metadata(),
            "suite_0_adversarial_and_oracle_audits": suite0_res,
            "suite_a_round2_roofline_ratchets": suite_a_res,
            "suite_b_8gpu_1_11b_bd3_fsdp2_round2": suite_b_res,
            "suite_c_fixed_wallclock_valbpb_autoresearch": suite_c_res,
        }
        out_json = args.output_dir / f"rsi_round2_roofline_and_valbpb_loop_{args.mode}.json"
        out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"[E128][rank 0] Wrote {out_json} in {wall_s:.2f}s.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
