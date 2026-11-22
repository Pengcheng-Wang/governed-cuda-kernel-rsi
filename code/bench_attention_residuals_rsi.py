#!/usr/bin/env python3
"""
E132: Round-6 Recursive Self-Improvement (`RSI`) Study of Depth-Wise Selective Residual
Aggregation (`Attention Residuals` / `Block AttnRes`, Moonshot Kimi Team, March 2026,
`arXiv:2603.15031`, official repo `MoonshotAI/Attention-Residuals` + community Triton
reference `Hydra-2P` `manishklach/attnres-kernel-lab`):

  - Suite 0: Cryptographic Harness Verification (`eval_harness_sha256`), `4/4` Adversarial
    Red-Team Exploits (`Fig. 2` `i=0` duplicate-source bug, omitted `RMSNorm` radial Jacobian,
    `TF32` precision mutation, unscaled partition merge), Community `Hydra-2P` `(N*M)`-way
    `tl.atomic_add` contention & SM80 `BD=4096, nw=16` `26`-spill forensics vs `14/14`
    `0`-spill Round-6 RSI cubins, Multi-Shape `FP32`/`BF16` Parity (including step-0 `w_l=0`,
    zero-vector `v=0`, ragged `D=1,264`), and `7` `FP64` `O(h^2)` Richardson Checks.
  - Suite A: Two Multi-Shape `RSIVectorRatchet` Tracks:
      * Track 1: Batched Two-Phase `Block AttnRes` + Superfused Sublayer `Pre-RMSNorm`
      * Track 2: End-to-End Multi-Block Sequential Transformer Stack (`N_B = 8` Blocks +
        Final `AttnRes` Readout, `RSIPhase1UnbindInterBlockFn + RSIPhase2SuperfusedPreNormFn`)
  - Suite B: 8-GPU NVSwitch Interleaved `1F1B` Pipeline Parallelism (`P = 8, V = 4`, `32`
    Virtual Stages, `142`-Tick Megatron-LM Causal Schedule) with Cross-Stage Virtual-Rank
    Block Caching (`arXiv:2603.15031` Sec. 4.1 `Eq. 7-8` & `Fig. 3`, `9.93x` comm reduction
    vs `Full AttnRes` and `2.60x` vs Naive `Block AttnRes`, exact `Gate G9` 1-GPU parity).
  - Suite C: Fixed-Wall-Clock (`15.0s`) 8xA100 DDP `val_bpb` `autoresearch` Loop (`5`
    Generations + Single-Touch `test_bpb`) + `5`-Arm Matched-Step Iso-Param Deep-`24`-Layer
    (`48`-Sublayer) Residual-Topology & PreNorm-Dilution Scaling Law (`Standard PreNorm`,
    `DenseFormer`, `Highway`, `Block AttnRes N=8`, `Full AttnRes N=48`).

Governed by `PROGRAM.md` and byte-identical frozen `code/rsi_kernel_harness.py`
(`eval_harness_sha256 == 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
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
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_local_rank_env = os.environ.get("LOCAL_RANK", "0")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/inductor_e132_r{_local_rank_env}")
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_e132_r{_local_rank_env}")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func

for _dyn_attr, _dyn_val in (
    ("recompile_limit", 64),
    ("cache_size_limit", 64),
    ("accumulated_recompile_limit", 256),
    ("accumulated_cache_size_limit", 256),
):
    if hasattr(torch._dynamo.config, _dyn_attr):
        setattr(torch._dynamo.config, _dyn_attr, _dyn_val)

from attnres_rsi_kernels import (
    BrokenNoRadialJacobianBlockAttnResFn,
    FusedSublayerRMSNormFn,
    Hydra2PCommunityAttnResFn,
    RSIBatchedTwoPhaseBlockAttnResFn,
    RSIPhase1InterBlockFn,
    RSIPhase1UnbindInterBlockFn,
    RSIPhase2SuperfusedPreNormFn,
    audit_naive_sm80_d4096_nw16_spill,
    collect_compiled_kernel_spill_stats,
    compile_friendly_block_attnres_prenorm_ref,
    eager_block_attnres_prenorm_ref,
    eager_rmsnorm_ref,
    fp64_block_attnres_oracle,
    official_fig2_single_attnres_ref,
    rsi_two_phase_block_attnres_batched,
)
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

EXPECTED_HARNESS_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def enforce_strict_ieee_precision() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


# =============================================================================
# Suite 0: Cryptographic Harness, 4/4 Red-Team Exploits, Hydra-2P & Spill Audit,
#          Multi-Shape Parity, and 7 FP64 O(h^2) Richardson Checks
# =============================================================================


def run_suite0_audit(device: torch.device) -> Dict[str, Any]:
    enforce_strict_ieee_precision()
    actual_sha = eval_harness_sha256()
    if actual_sha != EXPECTED_HARNESS_SHA256:
        raise RuntimeError(f"Immutable harness SHA256 mismatch: {actual_sha} != {EXPECTED_HARNESS_SHA256}")

    # 1. Four adversarial red-team exploits
    gen = torch.Generator(device=device)
    gen.manual_seed(20261123)
    S, N, M, D = 6, 8, 128, 512
    wq = (torch.randn((S, D), device=device, dtype=torch.float32, generator=gen) * 0.25).requires_grad_(True)
    gk = (1.0 + 0.1 * torch.randn((S, D), device=device, dtype=torch.float32, generator=gen)).requires_grad_(True)
    gs = (1.0 + 0.1 * torch.randn((S, D), device=device, dtype=torch.float32, generator=gen)).requires_grad_(True)
    bl = (torch.randn((N, M, D), device=device, dtype=torch.float32, generator=gen) * 0.35).requires_grad_(True)
    pa = bl[0].unsqueeze(0).repeat(S, 1, 1).detach().clone()
    pa[1:] = torch.randn((S - 1, M, D), device=device, dtype=torch.float32, generator=gen) * 0.35
    pa.requires_grad_(True)
    hp = torch.tensor([False] + [True] * (S - 1), device=device, dtype=torch.bool)
    dy = torch.randn((S, M, D), device=device, dtype=torch.float32, generator=gen)

    orc = fp64_block_attnres_oracle(wq, gk, bl, pa, hp, dy, gamma_subs=gs)
    red_team: List[Dict[str, Any]] = []

    # Exploit 1: Moonshot Fig. 2 `i=0` duplicate-source bug (`blocks + [partial_block]` when `partial_block = b_0`)
    h0_e1 = official_fig2_single_attnres_ref(
        list(bl.detach().unbind(0)), None, wq.detach()[0], gk.detach()[0], 1e-6, bug_duplicate_first_block=True
    )
    y0_e1 = eager_rmsnorm_ref(h0_e1, gs.detach()[0:1], 1e-6)
    c_e1 = compare_tensor_to_fp64(y0_e1, orc["out"][0], rel_tol=1e-4, name="fig2_s0_out")
    red_team.append(
        {
            "exploit": "exploit_1_moonshot_fig2_i0_duplicate_source_bug",
            "rejected": not c_e1["ok"],
            "decision": "REJECT_CORRECTNESS" if not c_e1["ok"] else "MISSED",
            "corrupted_s0_rel_l2": c_e1["rel_l2"],
        }
    )

    # Exploit 2: Omitted RMSNorm radial Jacobian (`0.0` forward error, corrupted `dblocks`!)
    wq2 = wq.detach().clone().requires_grad_(True)
    gk2 = gk.detach().clone().requires_grad_(True)
    bl2 = bl.detach().clone().requires_grad_(True)
    pa2 = pa.detach().clone().requires_grad_(True)
    gs2 = gs.detach().clone().requires_grad_(True)
    y_e2 = BrokenNoRadialJacobianBlockAttnResFn.apply(wq2, gk2, bl2, pa2, hp, gs2)
    (y_e2 * dy).sum().backward()
    fwd_e2 = compare_tensor_to_fp64(y_e2, orc["out"], rel_tol=1e-4, name="e2_fwd")
    bwd_e2 = compare_tensor_to_fp64(bl2.grad, orc["dblocks"], rel_tol=1e-4, name="e2_dblocks")
    red_team.append(
        {
            "exploit": "exploit_2_omitted_rmsnorm_radial_jacobian",
            "rejected": (fwd_e2["ok"] and (not bwd_e2["ok"])),
            "decision": "REJECT_CORRECTNESS" if not bwd_e2["ok"] else "MISSED",
            "forward_rel_l2": fwd_e2["rel_l2"],
            "corrupted_dblocks_rel_l2": bwd_e2["rel_l2"],
        }
    )

    # Exploit 3: TF32 precision flag tampering caught by PrecisionStateSnapshot
    snap0 = PrecisionStateSnapshot.capture()
    tamper_caught = False
    try:
        def _tamper_slot():
            torch.backends.cuda.matmul.allow_tf32 = True
            return rsi_two_phase_block_attnres_batched(wq.detach(), gk.detach(), bl.detach(), pa.detach(), hp, gamma_subs=gs.detach())
        benchmark_rotating_cuda([_tamper_slot], warmup=1, groups=1, iters_per_group=1, gpu_idx=device.index or 0)
    except RuntimeError:
        tamper_caught = True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = snap0.allow_tf32_matmul
        torch.backends.cudnn.allow_tf32 = snap0.allow_tf32_cudnn
    red_team.append(
        {
            "exploit": "exploit_3_tf32_precision_flag_tampering",
            "rejected": tamper_caught,
            "decision": "REJECT_ENV_TAMPER" if tamper_caught else "MISSED",
        }
    )

    # Exploit 4: Unscaled Two-Phase partition merge (averaging `O1_bar` and `partial` without `logaddexp`)
    o1_only, _ = RSIPhase1InterBlockFn.apply(wq.detach(), gk.detach(), bl.detach(), 1e-6, True)
    h_unscaled = torch.where(hp.view(-1, 1, 1), 0.5 * o1_only + 0.5 * pa.detach(), o1_only)
    y_e4 = eager_rmsnorm_ref(h_unscaled, gs.detach().unsqueeze(1), 1e-6)
    c_e4 = compare_tensor_to_fp64(y_e4, orc["out"], rel_tol=1e-4, name="unscaled_merge")
    red_team.append(
        {
            "exploit": "exploit_4_unscaled_two_phase_partition_merge",
            "rejected": not c_e4["ok"],
            "decision": "REJECT_CORRECTNESS" if not c_e4["ok"] else "MISSED",
            "corrupted_out_rel_l2": c_e4["rel_l2"],
        }
    )

    # 2. Cubin spill audit (14/14 zero-spill RSI kernels vs naive BD=4096, nw=16 26-spill kernel)
    rsi_spill_stats = collect_compiled_kernel_spill_stats(device)
    naive_d4096_spill = audit_naive_sm80_d4096_nw16_spill(device)

    # 3. Community Hydra-2P `(N*M)`-way atomic contention forensics across D in {1024, 2048, 4096}
    hydra2p_forensics: List[Dict[str, Any]] = []
    for d_aud in (1024, 2048, 4096):
        s_a, n_a, m_a = 6, 8, 4096
        wq_a = (torch.randn(s_a, d_aud, device=device, dtype=torch.float32) * 0.2).requires_grad_(True)
        gk_a = torch.ones(s_a, d_aud, device=device, dtype=torch.float32, requires_grad=True)
        gs_a = torch.ones(s_a, d_aud, device=device, dtype=torch.float32, requires_grad=True)
        bl_a = (torch.randn(n_a, m_a, d_aud, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        pa_a = (torch.randn(s_a, m_a, d_aud, device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
        hp_a = torch.tensor([False] + [True] * (s_a - 1), device=device, dtype=torch.bool)
        dy_a = torch.randn(s_a, m_a, d_aud, device=device, dtype=torch.bfloat16)

        def _h2p():
            wq_a.grad = gk_a.grad = gs_a.grad = bl_a.grad = pa_a.grad = None
            h_ = Hydra2PCommunityAttnResFn.apply(wq_a, gk_a, bl_a, pa_a, hp_a, 1e-6)
            FusedSublayerRMSNormFn.apply(h_, gs_a, 1e-6).backward(dy_a)

        def _rsi():
            wq_a.grad = gk_a.grad = gs_a.grad = bl_a.grad = pa_a.grad = None
            rsi_two_phase_block_attnres_batched(wq_a, gk_a, bl_a, pa_a, hp_a, gamma_subs=gs_a).backward(dy_a)

        def _eag():
            wq_a.grad = gk_a.grad = gs_a.grad = bl_a.grad = pa_a.grad = None
            eager_block_attnres_prenorm_ref(wq_a, gk_a, bl_a, pa_a, hp_a, gamma_subs=gs_a).backward(dy_a)

        t_eag = benchmark_rotating_cuda([_eag], warmup=3, groups=5, iters_per_group=4, gpu_idx=device.index or 0)["median_ms"]
        t_h2p = benchmark_rotating_cuda([_h2p], warmup=3, groups=5, iters_per_group=4, gpu_idx=device.index or 0)["median_ms"]
        t_rsi = benchmark_rotating_cuda([_rsi], warmup=3, groups=5, iters_per_group=4, gpu_idx=device.index or 0)["median_ms"]
        hydra2p_forensics.append(
            {
                "S": s_a,
                "N": n_a,
                "M": m_a,
                "D": d_aud,
                "dq_atomic_writers_per_elem": (n_a + 1) * m_a,
                "dv_atomic_writers_per_elem": s_a,
                "eager_fig2_ms": t_eag,
                "hydra2p_atomic_ms": t_h2p,
                "rsi_superfused_ms": t_rsi,
                "rsi_speedup_vs_hydra2p": round(t_h2p / max(t_rsi, 1e-9), 3),
                "hydra2p_slower_than_eager": bool(t_h2p > t_eag) if d_aud <= 2048 else False,
            }
        )

    # 4. Multi-shape FP32 & BF16 parity (batched + sequential unbind)
    parity_shapes = [
        ("canonical_d1024", 6, 8, 256, 1024, False),
        ("canonical_d2048", 6, 8, 256, 2048, False),
        ("canonical_d4096", 4, 8, 256, 4096, False),
        ("edge_step0_w0_v0_ragged_d1264", 6, 7, 256, 1264, True),
    ]
    parity_checks: List[Dict[str, Any]] = []
    max_fp32_rel = 0.0
    max_bf16_rel = 0.0
    for sname, sp, np_, mp, dp, is_edge in parity_shapes:
        for dt, dt_name, tol in ((torch.float32, "fp32", 2e-5), (torch.bfloat16, "bf16", 1.5e-2)):
            g_p = torch.Generator(device=device)
            g_p.manual_seed(3100 + dp + (1 if dt == torch.bfloat16 else 0))
            wq_p = torch.randn((sp, dp), device=device, dtype=torch.float32, generator=g_p) * 0.25
            if is_edge:
                wq_p[:2].zero_()  # Moonshot mandatory step-0 `w_l = 0` uniform init!
            wq_p.requires_grad_(True)
            gk_p = (1.0 + 0.1 * torch.randn((sp, dp), device=device, dtype=torch.float32, generator=g_p)).requires_grad_(True)
            gs_p = (1.0 + 0.1 * torch.randn((sp, dp), device=device, dtype=torch.float32, generator=g_p)).requires_grad_(True)
            bl_p = (torch.randn((np_, mp, dp), device=device, dtype=dt, generator=g_p) * 0.35)
            if is_edge:
                bl_p[0, 5, :].zero_()  # Exact zero-row `eps_key` guard!
            bl_p.requires_grad_(True)
            pa_p = (torch.randn((sp, mp, dp), device=device, dtype=dt, generator=g_p) * 0.35).requires_grad_(True)
            hp_p = torch.tensor([False] + [True] * (sp - 1), device=device, dtype=torch.bool)
            dy_p = torch.randn((sp, mp, dp), device=device, dtype=dt, generator=g_p)

            orc_p = fp64_block_attnres_oracle(wq_p, gk_p, bl_p, pa_p, hp_p, dy_p, gamma_subs=gs_p)

            # Batched Two-Phase Superfused
            yb = rsi_two_phase_block_attnres_batched(wq_p, gk_p, bl_p, pa_p, hp_p, gamma_subs=gs_p)
            (yb.float() * dy_p.float()).sum().backward()
            for tnm, cand, ref in (
                ("out", yb, orc_p["out"]),
                ("dwq", wq_p.grad, orc_p["dw_queries"]),
                ("dgk", gk_p.grad, orc_p["dgamma_keys"]),
                ("dbl", bl_p.grad, orc_p["dblocks"]),
                ("dpa", pa_p.grad, orc_p["dpartials"]),
                ("dgs", gs_p.grad, orc_p["dgamma_subs"]),
            ):
                chk = compare_tensor_to_fp64(cand, ref, rel_tol=tol, name=f"{sname}_{dt_name}_batched_{tnm}")
                parity_checks.append(chk)
                if dt == torch.float32:
                    max_fp32_rel = max(max_fp32_rel, float(chk["rel_l2"]))
                else:
                    max_bf16_rel = max(max_bf16_rel, float(chk["rel_l2"]))

            # Sequential Unbind + Per-Sublayer Superfused
            wq_p.grad = gk_p.grad = gs_p.grad = bl_p.grad = pa_p.grad = None
            ub = RSIPhase1UnbindInterBlockFn.apply(wq_p, gk_p, bl_p, 1e-6, True)
            ys_list = []
            for s_i in range(sp):
                p_arg = pa_p[s_i] if s_i > 0 else None
                ys_list.append(
                    RSIPhase2SuperfusedPreNormFn.apply(
                        wq_p[s_i], gk_p[s_i], ub[s_i], ub[sp + s_i], p_arg, gs_p[s_i], dt, 1e-6, 1e-6
                    )
                )
            yseq = torch.stack(ys_list, dim=0)
            (yseq.float() * dy_p.float()).sum().backward()
            chk_seq = compare_tensor_to_fp64(bl_p.grad, orc_p["dblocks"], rel_tol=tol, name=f"{sname}_{dt_name}_seq_dbl")
            parity_checks.append(chk_seq)
            if dt == torch.float32:
                max_fp32_rel = max(max_fp32_rel, float(chk_seq["rel_l2"]))
            else:
                max_bf16_rel = max(max_bf16_rel, float(chk_seq["rel_l2"]))

    # 5. Seven FP64 O(h^2) Richardson extrapolation checks
    g_r = torch.Generator(device=device)
    g_r.manual_seed(20261124)
    Sr, Nr, Mr, Dr = 4, 5, 32, 96
    wq_r = (torch.randn((Sr, Dr), device=device, dtype=torch.float64, generator=g_r) * 0.35).requires_grad_(True)
    gk_r = (1.0 + 0.15 * torch.randn((Sr, Dr), device=device, dtype=torch.float64, generator=g_r)).requires_grad_(True)
    gs_r = (1.0 + 0.15 * torch.randn((Sr, Dr), device=device, dtype=torch.float64, generator=g_r)).requires_grad_(True)
    bl_r = (torch.randn((Nr, Mr, Dr), device=device, dtype=torch.float64, generator=g_r) * 0.4).requires_grad_(True)
    pa_r = (torch.randn((Sr, Mr, Dr), device=device, dtype=torch.float64, generator=g_r) * 0.4).requires_grad_(True)
    hp_r = torch.tensor([False] + [True] * (Sr - 1), device=device, dtype=torch.bool)
    dy_r = torch.randn((Sr, Mr, Dr), device=device, dtype=torch.float64, generator=g_r)

    y_r = rsi_two_phase_block_attnres_batched(wq_r, gk_r, bl_r, pa_r, hp_r, gamma_subs=gs_r)
    (y_r * dy_r).sum().backward()

    richardson_checks: List[Dict[str, Any]] = [
        richardson_unit_rms_audit(
            lambda x: (rsi_two_phase_block_attnres_batched(x, gk_r.detach(), bl_r.detach(), pa_r.detach(), hp_r, gamma_subs=gs_r.detach()) * dy_r).sum(),
            wq_r.detach(), wq_r.grad.detach(), name="w_queries_general",
        ),
        richardson_unit_rms_audit(
            lambda x: (rsi_two_phase_block_attnres_batched(wq_r.detach(), x, bl_r.detach(), pa_r.detach(), hp_r, gamma_subs=gs_r.detach()) * dy_r).sum(),
            gk_r.detach(), gk_r.grad.detach(), name="gamma_keys",
        ),
        richardson_unit_rms_audit(
            lambda x: (rsi_two_phase_block_attnres_batched(wq_r.detach(), gk_r.detach(), x, pa_r.detach(), hp_r, gamma_subs=gs_r.detach()) * dy_r).sum(),
            bl_r.detach(), bl_r.grad.detach(), name="blocks_radial_rmsnorm",
        ),
        richardson_unit_rms_audit(
            lambda x: (rsi_two_phase_block_attnres_batched(wq_r.detach(), gk_r.detach(), bl_r.detach(), x, hp_r, gamma_subs=gs_r.detach()) * dy_r).sum(),
            pa_r.detach(), pa_r.grad.detach(), name="partials_radial_rmsnorm",
        ),
    ]

    # Check 5: Nonlinear smooth objective `cos(y * dy)` so d^3L / d(gamma_subs)^3 != 0 (`O(h^2)` ~ 4.000x!)
    gs_nl = gs_r.detach().clone().requires_grad_(True)
    loss_gs_nl = torch.cos(
        rsi_two_phase_block_attnres_batched(wq_r.detach(), gk_r.detach(), bl_r.detach(), pa_r.detach(), hp_r, gamma_subs=gs_nl) * dy_r
    ).sum()
    (dgs_nl,) = torch.autograd.grad(loss_gs_nl, (gs_nl,))
    richardson_checks.append(
        richardson_unit_rms_audit(
            lambda x: torch.cos(
                rsi_two_phase_block_attnres_batched(wq_r.detach(), gk_r.detach(), bl_r.detach(), pa_r.detach(), hp_r, gamma_subs=x) * dy_r
            ).sum(),
            gs_nl.detach(), dgs_nl.detach(), name="gamma_subs_superfused",
        )
    )

    # Check 6: Step-0 exact `w_queries = 0` uniform init (`O(h^2)` symmetry!)
    wq_zero = torch.zeros_like(wq_r.detach()).requires_grad_(True)
    y_z = rsi_two_phase_block_attnres_batched(wq_zero, gk_r.detach(), bl_r.detach(), pa_r.detach(), hp_r, gamma_subs=gs_r.detach())
    (y_z * dy_r).sum().backward()
    richardson_checks.append(
        richardson_unit_rms_audit(
            lambda x: (rsi_two_phase_block_attnres_batched(x, gk_r.detach(), bl_r.detach(), pa_r.detach(), hp_r, gamma_subs=gs_r.detach()) * dy_r).sum(),
            wq_zero.detach(), wq_zero.grad.detach(), h_coarse=1.5e-3, h_fine=7.5e-4, name="w_queries_step0_zero_init",
        )
    )

    # Check 7: Sequential 2-block stack + Final AttnRes Readout
    w_lin = torch.randn((2, 2, Dr, Dr), device=device, dtype=torch.float64, generator=g_r) * (0.3 / math.sqrt(Dr))
    wq_s = (torch.randn((2, 2, Dr), device=device, dtype=torch.float64, generator=g_r) * 0.3).requires_grad_(True)
    gk_s = torch.ones((2, 2, Dr), device=device, dtype=torch.float64)
    gs_s = torch.ones((2, 2, Dr), device=device, dtype=torch.float64)
    wq_ro = (torch.randn((1, Dr), device=device, dtype=torch.float64, generator=g_r) * 0.3)
    gk_ro = torch.ones((1, Dr), device=device, dtype=torch.float64)
    b0_s = (torch.randn((Mr, Dr), device=device, dtype=torch.float64, generator=g_r) * 0.4).requires_grad_(True)
    dy_ro = torch.randn((Mr, Dr), device=device, dtype=torch.float64, generator=g_r)

    def _seq2_loss(b0_in: torch.Tensor) -> torch.Tensor:
        blocks_acc = [b0_in]
        for bi in range(2):
            bl_st = torch.stack(blocks_acc, dim=0)
            ub_ = RSIPhase1UnbindInterBlockFn.apply(wq_s[bi], gk_s[bi], bl_st, 1e-6, True)
            p_cur: Optional[torch.Tensor] = None
            for si in range(2):
                y_norm = RSIPhase2SuperfusedPreNormFn.apply(
                    wq_s[bi, si], gk_s[bi, si], ub_[si], ub_[2 + si], p_cur, gs_s[bi, si], torch.float64, 1e-6, 1e-6
                )
                f_out = torch.tanh(y_norm @ w_lin[bi, si])
                p_cur = f_out if p_cur is None else (p_cur + f_out)
            assert p_cur is not None
            blocks_acc.append(p_cur)
        bl_final = torch.stack(blocks_acc, dim=0)
        ub_ro = RSIPhase1UnbindInterBlockFn.apply(wq_ro, gk_ro, bl_final, 1e-6, True)
        return (ub_ro[0] * dy_ro).sum()

    loss_seq = _seq2_loss(b0_s)
    (db0_s,) = torch.autograd.grad(loss_seq, (b0_s,))
    richardson_checks.append(
        richardson_unit_rms_audit(_seq2_loss, b0_s.detach(), db0_s.detach(), name="sequential_2block_stack_readout")
    )

    gc.collect()
    torch.cuda.empty_cache()
    return {
        "eval_harness_sha256": actual_sha,
        "harness_version": HARNESS_VERSION,
        "red_team_exploits": red_team,
        "red_team_rejected_count": sum(1 for r in red_team if r["rejected"]),
        "red_team_total": len(red_team),
        "rsi_cubin_spill_audit": rsi_spill_stats,
        "naive_d4096_nw16_spill_audit": naive_d4096_spill,
        "hydra2p_atomic_forensics": hydra2p_forensics,
        "num_parity_checks": len(parity_checks),
        "all_parity_passed": all(bool(c["ok"]) for c in parity_checks),
        "max_fp32_rel_l2": round(max_fp32_rel, 8),
        "max_bf16_rel_l2": round(max_bf16_rel, 8),
        "richardson_checks": richardson_checks,
        "all_richardson_passed": all(bool(c["ok"]) for c in richardson_checks),
    }


# =============================================================================
# Suite A: Track 1 (Batched Two-Phase Block AttnRes + Superfused PreNorm) &
#          Track 2 (End-to-End Multi-Block Sequential Transformer Stack)
# =============================================================================


def _seq_stack_forward(
    mode_name: str,
    b0: torch.Tensor,
    wq_all: torch.Tensor,      # [NB, S, D] fp32
    gk_all: torch.Tensor,      # [NB, S, D] fp32
    gs_all: torch.Tensor,      # [NB, S, D] fp32
    w_down: torch.Tensor,      # [NB, S, D, R] bf16
    w_up: torch.Tensor,        # [NB, S, R, D] bf16
    wq_ro: torch.Tensor,       # [1, D] fp32
    gk_ro: torch.Tensor,       # [1, D] fp32
    gs_ro: torch.Tensor,       # [1, D] fp32
    compiled_sub_fn: Optional[Callable] = None,
) -> torch.Tensor:
    NB, S, D = wq_all.shape
    blocks_list: List[torch.Tensor] = [b0]

    for bi in range(NB):
        bl_stack = torch.stack(blocks_list, dim=0)
        if mode_name == "t0_seq_moonshot_fig2_eager":
            w_dt = torch.float64 if b0.dtype == torch.float64 else torch.float32
            p_cur: Optional[torch.Tensor] = None
            for si in range(S):
                src = bl_stack if p_cur is None else torch.cat([bl_stack, p_cur.unsqueeze(0)], dim=0)
                k_norm = eager_rmsnorm_ref(src, gk_all[bi, si].view(1, 1, D), 1e-6)
                logits = torch.einsum("d,nmd->nm", wq_all[bi, si].to(w_dt), k_norm.to(w_dt))
                alpha = torch.softmax(logits, dim=0)
                h_s = torch.einsum("nm,nmd->md", alpha, src.to(w_dt)).to(b0.dtype)
                y_s = eager_rmsnorm_ref(h_s, gs_all[bi, si].view(1, D), 1e-6)
                f_out = F.silu(y_s @ w_down[bi, si]) @ w_up[bi, si]
                p_cur = f_out if p_cur is None else (p_cur + f_out)
            assert p_cur is not None
            blocks_list.append(p_cur)

        elif mode_name == "t1_seq_torch_compile":
            assert compiled_sub_fn is not None
            p_cur = None
            for si in range(S):
                src = bl_stack if p_cur is None else torch.cat([bl_stack, p_cur.unsqueeze(0)], dim=0)
                y_s = compiled_sub_fn(wq_all[bi, si], gk_all[bi, si], src, gs_all[bi, si])
                f_out = F.silu(y_s @ w_down[bi, si]) @ w_up[bi, si]
                p_cur = f_out if p_cur is None else (p_cur + f_out)
            assert p_cur is not None
            blocks_list.append(p_cur)

        elif mode_name == "t2_seq_hydra2p_atomic":
            p_cur = None
            hp1 = torch.ones((1,), device=b0.device, dtype=torch.bool)
            hp0 = torch.zeros((1,), device=b0.device, dtype=torch.bool)
            for si in range(S):
                p_in = bl_stack[0:1] if p_cur is None else p_cur.unsqueeze(0)
                h_s = Hydra2PCommunityAttnResFn.apply(
                    wq_all[bi, si : si + 1], gk_all[bi, si : si + 1], bl_stack, p_in, hp0 if p_cur is None else hp1, 1e-6
                )[0]
                y_s = FusedSublayerRMSNormFn.apply(h_s, gs_all[bi, si], 1e-6)
                f_out = F.silu(y_s @ w_down[bi, si]) @ w_up[bi, si]
                p_cur = f_out if p_cur is None else (p_cur + f_out)
            assert p_cur is not None
            blocks_list.append(p_cur)

        elif mode_name == "t3_seq_naive_slice_selectbwd":
            o1_b, lse1_b = RSIPhase1InterBlockFn.apply(wq_all[bi], gk_all[bi], bl_stack, 1e-6, True)
            p_cur = None
            for si in range(S):
                # Naive `o1_b[si]` Python slicing triggers `S` `[S, M, D]` `SelectBackward0` zero-fills!
                y_s = RSIPhase2SuperfusedPreNormFn.apply(
                    wq_all[bi, si], gk_all[bi, si], o1_b[si], lse1_b[si], p_cur, gs_all[bi, si], b0.dtype, 1e-6, 1e-6
                )
                f_out = F.silu(y_s @ w_down[bi, si]) @ w_up[bi, si]
                p_cur = f_out if p_cur is None else (p_cur + f_out)
            assert p_cur is not None
            blocks_list.append(p_cur)

        elif mode_name == "t4_seq_rsi_unbind_superfused":
            ub = RSIPhase1UnbindInterBlockFn.apply(wq_all[bi], gk_all[bi], bl_stack, 1e-6, True)
            p_cur = None
            for si in range(S):
                y_s = RSIPhase2SuperfusedPreNormFn.apply(
                    wq_all[bi, si], gk_all[bi, si], ub[si], ub[S + si], p_cur, gs_all[bi, si], b0.dtype, 1e-6, 1e-6
                )
                f_out = F.silu(y_s @ w_down[bi, si]) @ w_up[bi, si]
                p_cur = f_out if p_cur is None else (p_cur + f_out)
            assert p_cur is not None
            blocks_list.append(p_cur)
        else:
            raise ValueError(f"Unknown Track 2 mode: {mode_name}")

    # Final Readout AttnRes over `[b_0, b_1, ..., b_{NB}]` + Output Pre-RMSNorm (`arXiv:2603.15031` Sec. 3.2)
    bl_ro = torch.stack(blocks_list, dim=0)
    if mode_name in ("t0_seq_moonshot_fig2_eager", "t1_seq_torch_compile"):
        w_dt = torch.float64 if b0.dtype == torch.float64 else torch.float32
        kn = eager_rmsnorm_ref(bl_ro, gk_ro.view(1, 1, D), 1e-6)
        lg = torch.einsum("d,nmd->nm", wq_ro[0].to(w_dt), kn.to(w_dt))
        h_ro = torch.einsum("nm,nmd->md", torch.softmax(lg, dim=0), bl_ro.to(w_dt)).to(b0.dtype)
        return eager_rmsnorm_ref(h_ro, gs_ro.view(1, D), 1e-6)
    ub_ro = RSIPhase1UnbindInterBlockFn.apply(wq_ro, gk_ro, bl_ro, 1e-6, True)
    return RSIPhase2SuperfusedPreNormFn.apply(wq_ro[0], gk_ro[0], ub_ro[0], ub_ro[1], None, gs_ro[0], b0.dtype, 1e-6, 1e-6)


def _compile_seq_sublayer_attnres_prenorm(wq_s: torch.Tensor, gk_s: torch.Tensor, src: torch.Tensor, gs_s: torch.Tensor) -> torch.Tensor:
    vf = src.float()
    rv = torch.rsqrt((vf * vf).mean(dim=-1, keepdim=True) + 1e-6)
    logits = torch.einsum("d,nmd->nm", wq_s.float() * gk_s.float(), vf * rv)
    alpha = torch.softmax(logits, dim=0)
    h = torch.einsum("nm,nmd->md", alpha, vf)
    rh = torch.rsqrt((h * h).mean(dim=-1, keepdim=True) + 1e-6)
    return (h * rh * gs_s.float().unsqueeze(0)).to(src.dtype)


def run_suite_a_ratchets(device: torch.device, mode: str, out_dir: Path) -> Dict[str, Any]:
    enforce_strict_ieee_precision()
    l2_flush = torch.empty((16 * 1024 * 1024,), device=device, dtype=torch.float32)  # 64 MiB > 40 MiB A100 L2
    warmup = 3 if mode == "smoke" else 4
    groups = 5 if mode == "smoke" else 7
    iters = 4 if mode == "smoke" else 6

    # ---------------- Track 1: Batched Two-Phase Block AttnRes + Superfused PreNorm ----------------
    t1_shapes: List[Tuple[str, int, int, int, int]] = (
        [
            ("s6_n8_m4096_d1024", 6, 8, 4096, 1024),
            ("s6_n8_m4096_d2048", 6, 8, 4096, 2048),
        ]
        if mode == "smoke"
        else [
            ("s6_n4_m8192_d1024", 6, 4, 8192, 1024),
            ("s6_n8_m8192_d2048", 6, 8, 8192, 2048),
            ("s8_n8_m4096_d2048", 8, 8, 4096, 2048),
            ("s6_n8_m8192_d4096", 6, 8, 8192, 4096),
        ]
    )
    ratchet_t1 = RSIVectorRatchet("track1_batched_two_phase_block_attnres_prenorm", [x[0] for x in t1_shapes])
    comp_t1 = torch.compile(compile_friendly_block_attnres_prenorm_ref, fullgraph=True, dynamic=False)

    def _cand_t0(wq, gk, bl, pa, hp, gs):
        return eager_block_attnres_prenorm_ref(wq, gk, bl, pa, hp, gamma_subs=gs)

    def _cand_t1(wq, gk, bl, pa, hp, gs):
        return comp_t1(wq, gk, bl, pa, hp, gamma_subs=gs)

    def _cand_t2(wq, gk, bl, pa, hp, gs):
        h = Hydra2PCommunityAttnResFn.apply(wq, gk, bl, pa, hp, 1e-6)
        return FusedSublayerRMSNormFn.apply(h, gs, 1e-6)

    def _cand_t3(wq, gk, bl, pa, hp, gs):
        return BrokenNoRadialJacobianBlockAttnResFn.apply(wq, gk, bl, pa, hp, gs)

    def _cand_t4(wq, gk, bl, pa, hp, gs):
        h = rsi_two_phase_block_attnres_batched(wq, gk, bl, pa, hp, gamma_subs=None)
        return FusedSublayerRMSNormFn.apply(h, gs, 1e-6)

    def _cand_t5(wq, gk, bl, pa, hp, gs):
        return rsi_two_phase_block_attnres_batched(wq, gk, bl, pa, hp, gamma_subs=gs)

    t1_candidates = [
        (0, "t0_moonshot_fig2_eager", _cand_t0, hash_callable_sources(eager_block_attnres_prenorm_ref), "Official Moonshot Fig. 2 eager stack + RMSNorm + einsum"),
        (1, "t1_torch_compile_fullgraph", _cand_t1, hash_callable_sources(compile_friendly_block_attnres_prenorm_ref), "fullgraph=True TorchInductor fusion over block & partial logits"),
        (2, "t2_hydra2p_community_atomic", _cand_t2, hash_callable_sources(Hydra2PCommunityAttnResFn), "Community Hydra-2P Triton Two-Phase with (N*M)-way global tl.atomic_add backward"),
        (3, "t3_broken_no_radial_rmsnorm", _cand_t3, hash_callable_sources(BrokenNoRadialJacobianBlockAttnResFn), "Adversarial candidate omitting RMSNorm radial Jacobian (-rho * v)"),
        (4, "t4_rsi_lockfree_two_phase", _cand_t4, hash_callable_sources(RSIBatchedTwoPhaseBlockAttnResFn, FusedSublayerRMSNormFn), "Round-6 RSI 0-spill 0-atomic Split-M Two-Phase + separate Triton RMSNorm"),
        (5, "t5_rsi_superfused_prenorm", _cand_t5, hash_callable_sources(RSIBatchedTwoPhaseBlockAttnResFn), "Round-6 RSI Superfused Phase-2 + Sublayer Pre-RMSNorm (zero h_s & zero O1_bar retention)"),
    ]

    # Pre-build verification input for Track 1
    gv = torch.Generator(device=device)
    gv.manual_seed(4100)
    wq_v = (torch.randn((6, 1024), device=device, dtype=torch.float32, generator=gv) * 0.25).requires_grad_(True)
    gk_v = (1.0 + 0.1 * torch.randn((6, 1024), device=device, dtype=torch.float32, generator=gv)).requires_grad_(True)
    gs_v = (1.0 + 0.1 * torch.randn((6, 1024), device=device, dtype=torch.float32, generator=gv)).requires_grad_(True)
    bl_v = (torch.randn((8, 256, 1024), device=device, dtype=torch.bfloat16, generator=gv) * 0.35).requires_grad_(True)
    pa_v = (torch.randn((6, 256, 1024), device=device, dtype=torch.bfloat16, generator=gv) * 0.35).requires_grad_(True)
    hp_v = torch.tensor([False] + [True] * 5, device=device, dtype=torch.bool)
    dy_v = torch.randn((6, 256, 1024), device=device, dtype=torch.bfloat16, generator=gv)
    orc_t1 = fp64_block_attnres_oracle(wq_v, gk_v, bl_v, pa_v, hp_v, dy_v, gamma_subs=gs_v)

    for tid, cid, cfn, csha, note in t1_candidates:
        wq_c = wq_v.detach().clone().requires_grad_(True)
        gk_c = gk_v.detach().clone().requires_grad_(True)
        gs_c = gs_v.detach().clone().requires_grad_(True)
        bl_c = bl_v.detach().clone().requires_grad_(True)
        pa_c = pa_v.detach().clone().requires_grad_(True)
        yc = cfn(wq_c, gk_c, bl_c, pa_c, hp_v, gs_c)
        (yc.float() * dy_v.float()).sum().backward()
        pchecks = [
            compare_tensor_to_fp64(yc, orc_t1["out"], rel_tol=1.5e-2, name="out"),
            compare_tensor_to_fp64(wq_c.grad, orc_t1["dw_queries"], rel_tol=1.5e-2, name="dwq"),
            compare_tensor_to_fp64(gk_c.grad, orc_t1["dgamma_keys"], rel_tol=1.5e-2, name="dgk"),
            compare_tensor_to_fp64(bl_c.grad, orc_t1["dblocks"], rel_tol=1.5e-2, name="dbl"),
            compare_tensor_to_fp64(pa_c.grad, orc_t1["dpartials"], rel_tol=1.5e-2, name="dpa"),
            compare_tensor_to_fp64(gs_c.grad, orc_t1["dgamma_subs"], rel_tol=1.5e-2, name="dgs"),
        ]
        per_shape: Dict[str, Dict[str, Any]] = {}
        for skey, s_s, n_s, m_s, d_s in t1_shapes:
            slots = []
            for sl in range(3):
                wq_s = (torch.randn((s_s, d_s), device=device, dtype=torch.float32) * 0.2).requires_grad_(True)
                gk_s = torch.ones((s_s, d_s), device=device, dtype=torch.float32, requires_grad=True)
                gs_s = torch.ones((s_s, d_s), device=device, dtype=torch.float32, requires_grad=True)
                bl_s = (torch.randn((n_s, m_s, d_s), device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
                pa_s = (torch.randn((s_s, m_s, d_s), device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
                hp_s = torch.tensor([False] + [True] * (s_s - 1), device=device, dtype=torch.bool)
                dy_s = torch.randn((s_s, m_s, d_s), device=device, dtype=torch.bfloat16)

                def _make_fn(_wq=wq_s, _gk=gk_s, _bl=bl_s, _pa=pa_s, _hp=hp_s, _gs=gs_s, _dy=dy_s, _f=cfn):
                    def _step():
                        _wq.grad = _gk.grad = _gs.grad = _bl.grad = _pa.grad = None
                        out_ = _f(_wq, _gk, _bl, _pa, _hp, _gs)
                        out_.backward(_dy)
                    return _step

                slots.append(_make_fn())
            per_shape[skey] = benchmark_rotating_cuda(
                slots, warmup=warmup, groups=groups, iters_per_group=iters, gpu_idx=device.index or 0, l2_flush_buffer=l2_flush
            )
            del slots
            gc.collect()
            torch.cuda.empty_cache()

        ratchet_t1.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=csha,
            parity_checks=pchecks,
            richardson_checks=[],
            anti_cheat_checks={"ieee_fp32_preserved": not torch.backends.cuda.matmul.allow_tf32, "rotating_3slots_l2_flushed": True},
            per_shape_metrics=per_shape,
            hypothesis_note=note,
        )

    # ---------------- Track 2: End-to-End Multi-Block Sequential Transformer Stack ----------------
    t2_shapes: List[Tuple[str, int, int, int, int]] = (
        [
            ("m2048_d1024_s4_nb6", 2048, 1024, 4, 6),
            ("m2048_d1536_s4_nb6", 2048, 1536, 4, 6),
        ]
        if mode == "smoke"
        else [
            ("m4096_d1024_s4_nb8", 4096, 1024, 4, 8),
            ("m4096_d1536_s4_nb8", 4096, 1536, 4, 8),
            ("m4096_d2048_s4_nb8", 4096, 2048, 4, 8),
        ]
    )
    ratchet_t2 = RSIVectorRatchet("track2_sequential_multiblock_transformer_stack", [x[0] for x in t2_shapes])
    comp_sub = torch.compile(_compile_seq_sublayer_attnres_prenorm, dynamic=True)

    t2_candidates = [
        (0, "t0_seq_moonshot_fig2_eager", "Official sequential per-sublayer torch.stack + RMSNorm + einsum (32 stacks/fwd)"),
        (1, "t1_seq_torch_compile", "TorchInductor compiled per-sublayer AttnRes + Pre-RMSNorm"),
        (2, "t2_seq_hydra2p_atomic", "Community Hydra-2P sequential Two-Phase with atomic backward"),
        (3, "t3_seq_naive_slice_selectbwd", "Round-6 Phase-1 + naive Python o1_bar[s] slicing (S SelectBackward0 zero-fills/block)"),
        (4, "t4_seq_rsi_unbind_superfused", "Round-6 RSIPhase1UnbindInterBlockFn + RSIPhase2SuperfusedPreNormFn (zero SelectBackward0, zero O1_bar saved)"),
    ]

    # FP64 reference oracle on compact sequential stack `(M=128, D=256, S=4, NB=4)`
    gs2_gen = torch.Generator(device=device)
    gs2_gen.manual_seed(4200)
    Mv2, Dv2, Sv2, NBv2, Rv2 = 128, 256, 4, 4, 64
    b0_v2 = (torch.randn((Mv2, Dv2), device=device, dtype=torch.bfloat16, generator=gs2_gen) * 0.3).requires_grad_(True)
    wq_v2 = (torch.randn((NBv2, Sv2, Dv2), device=device, dtype=torch.float32, generator=gs2_gen) * 0.2).requires_grad_(True)
    gk_v2 = torch.ones((NBv2, Sv2, Dv2), device=device, dtype=torch.float32, requires_grad=True)
    gs_v2 = torch.ones((NBv2, Sv2, Dv2), device=device, dtype=torch.float32, requires_grad=True)
    wd_v2 = (torch.randn((NBv2, Sv2, Dv2, Rv2), device=device, dtype=torch.bfloat16, generator=gs2_gen) * (0.4 / math.sqrt(Dv2))).requires_grad_(True)
    wu_v2 = (torch.randn((NBv2, Sv2, Rv2, Dv2), device=device, dtype=torch.bfloat16, generator=gs2_gen) * (0.4 / math.sqrt(Rv2))).requires_grad_(True)
    wq_ro_v2 = (torch.randn((1, Dv2), device=device, dtype=torch.float32, generator=gs2_gen) * 0.2).requires_grad_(True)
    gk_ro_v2 = torch.ones((1, Dv2), device=device, dtype=torch.float32, requires_grad=True)
    gs_ro_v2 = torch.ones((1, Dv2), device=device, dtype=torch.float32, requires_grad=True)
    dy_v2 = torch.randn((Mv2, Dv2), device=device, dtype=torch.bfloat16, generator=gs2_gen)

    # Compute FP64 sequential stack oracle
    b0_64 = b0_v2.detach().to(torch.float64).requires_grad_(True)
    wq_64 = wq_v2.detach().to(torch.float64).requires_grad_(True)
    wd_64 = wd_v2.detach().to(torch.float64).requires_grad_(True)
    y_orc2 = _seq_stack_forward(
        "t0_seq_moonshot_fig2_eager",
        b0_64, wq_64, gk_v2.detach().to(torch.float64), gs_v2.detach().to(torch.float64),
        wd_64, wu_v2.detach().to(torch.float64),
        wq_ro_v2.detach().to(torch.float64), gk_ro_v2.detach().to(torch.float64), gs_ro_v2.detach().to(torch.float64),
    )
    (y_orc2 * dy_v2.to(torch.float64)).sum().backward()

    t2_peak_hbm_by_cand: Dict[str, Dict[str, float]] = {}
    for tid, cid, note in t2_candidates:
        # Verify BOTH strict FP32 parity (`<= 2e-5`) AND 16-sublayer (32-GEMM) BF16 parity (`<= 7.5e-2`) against FP64 oracle!
        b0_f32 = b0_64.detach().to(torch.float32).requires_grad_(True)
        wq_f32 = wq_64.detach().to(torch.float32).requires_grad_(True)
        wd_f32 = wd_64.detach().to(torch.float32).requires_grad_(True)
        yf32 = _seq_stack_forward(
            cid, b0_f32, wq_f32, gk_v2.detach(), gs_v2.detach(), wd_f32, wu_v2.detach().float(),
            wq_ro_v2.detach(), gk_ro_v2.detach(), gs_ro_v2.detach(), compiled_sub_fn=comp_sub,
        )
        (yf32 * dy_v2.float()).sum().backward()

        b0_c = b0_v2.detach().clone().requires_grad_(True)
        wq_c = wq_v2.detach().clone().requires_grad_(True)
        wd_c = wd_v2.detach().clone().requires_grad_(True)
        yc = _seq_stack_forward(
            cid, b0_c, wq_c, gk_v2.detach(), gs_v2.detach(), wd_c, wu_v2.detach(),
            wq_ro_v2.detach(), gk_ro_v2.detach(), gs_ro_v2.detach(), compiled_sub_fn=comp_sub,
        )
        (yc.float() * dy_v2.float()).sum().backward()
        pchecks2 = [
            compare_tensor_to_fp64(yf32, y_orc2, rel_tol=2e-5, name="seq_fp32_out"),
            compare_tensor_to_fp64(b0_f32.grad, b0_64.grad, rel_tol=2e-5, name="seq_fp32_db0"),
            compare_tensor_to_fp64(wq_f32.grad, wq_64.grad, rel_tol=2e-5, name="seq_fp32_dwq"),
            compare_tensor_to_fp64(wd_f32.grad, wd_64.grad, rel_tol=2e-5, name="seq_fp32_dwd"),
            compare_tensor_to_fp64(yc, y_orc2, rel_tol=2.5e-2, name="seq_bf16_out"),
            compare_tensor_to_fp64(b0_c.grad, b0_64.grad, rel_tol=7.5e-2, name="seq_bf16_db0"),
            compare_tensor_to_fp64(wq_c.grad, wq_64.grad, rel_tol=7.5e-2, name="seq_bf16_dwq"),
            compare_tensor_to_fp64(wd_c.grad, wd_64.grad, rel_tol=7.5e-2, name="seq_bf16_dwd"),
        ]

        per_shape2: Dict[str, Dict[str, Any]] = {}
        t2_peak_hbm_by_cand[cid] = {}
        for skey, m_s, d_s, s_s, nb_s in t2_shapes:
            r_s = max(64, d_s // 8)
            slots2 = []
            for _ in range(2):
                b0_s = (torch.randn((m_s, d_s), device=device, dtype=torch.bfloat16) * 0.3).requires_grad_(True)
                wq_s = (torch.randn((nb_s, s_s, d_s), device=device, dtype=torch.float32) * 0.2).requires_grad_(True)
                gk_s = torch.ones((nb_s, s_s, d_s), device=device, dtype=torch.float32, requires_grad=True)
                gs_s = torch.ones((nb_s, s_s, d_s), device=device, dtype=torch.float32, requires_grad=True)
                wd_s = (torch.randn((nb_s, s_s, d_s, r_s), device=device, dtype=torch.bfloat16) * (0.4 / math.sqrt(d_s))).requires_grad_(True)
                wu_s = (torch.randn((nb_s, s_s, r_s, d_s), device=device, dtype=torch.bfloat16) * (0.4 / math.sqrt(r_s))).requires_grad_(True)
                wq_ro = (torch.randn((1, d_s), device=device, dtype=torch.float32) * 0.2).requires_grad_(True)
                gk_ro = torch.ones((1, d_s), device=device, dtype=torch.float32, requires_grad=True)
                gs_ro = torch.ones((1, d_s), device=device, dtype=torch.float32, requires_grad=True)
                dy_s = torch.randn((m_s, d_s), device=device, dtype=torch.bfloat16)

                def _make_seq_fn(
                    _cid=cid, _b0=b0_s, _wq=wq_s, _gk=gk_s, _gs=gs_s, _wd=wd_s, _wu=wu_s,
                    _wqro=wq_ro, _gkro=gk_ro, _gsro=gs_ro, _dy=dy_s
                ):
                    def _step():
                        for t_ in (_b0, _wq, _gk, _gs, _wd, _wu, _wqro, _gkro, _gsro):
                            t_.grad = None
                        out_ = _seq_stack_forward(
                            _cid, _b0, _wq, _gk, _gs, _wd, _wu, _wqro, _gkro, _gsro, compiled_sub_fn=comp_sub
                        )
                        out_.backward(_dy)
                    return _step

                slots2.append(_make_seq_fn())
            res_s = benchmark_rotating_cuda(
                slots2, warmup=warmup, groups=groups, iters_per_group=iters, gpu_idx=device.index or 0, l2_flush_buffer=l2_flush
            )
            per_shape2[skey] = res_s
            t2_peak_hbm_by_cand[cid][skey] = float(res_s["peak_alloc_mib"])
            del slots2
            gc.collect()
            torch.cuda.empty_cache()

        ratchet_t2.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=hash_callable_sources(_seq_stack_forward, cid),
            parity_checks=pchecks2,
            richardson_checks=[],
            anti_cheat_checks={"ieee_fp32_preserved": not torch.backends.cuda.matmul.allow_tf32, "rotating_2slots_l2_flushed": True},
            per_shape_metrics=per_shape2,
            hypothesis_note=note,
        )

    ledger_path = out_dir / "rsi_ledger.tsv"
    ledger_sha = write_rsi_ledger_tsv([ratchet_t1, ratchet_t2], ledger_path)
    return {
        "rsi_ledger_tsv": str(ledger_path.name),
        "rsi_ledger_sha256": ledger_sha,
        "track1": ratchet_t1.to_summary_dict(),
        "track2": ratchet_t2.to_summary_dict(),
        "track2_peak_hbm_by_candidate_mib": t2_peak_hbm_by_cand,
    }


# =============================================================================
# Suite B: 8-GPU NVSwitch Interleaved 1F1B Pipeline Parallelism (`P=8, V=4`)
#          with Cross-Stage Virtual-Rank Block Caching (`arXiv:2603.15031` Sec. 4.1)
# =============================================================================


class PPStageBlockModule(nn.Module):
    """Virtual pipeline stage `vs in 0..31` owning `1` complete `AttnRes` Block (`S = 4` sublayers).
    Stage `vs = 0` also owns initial projection `W_in` creating `b_0`; stage `vs = 31` also owns
    Final Readout `AttnRes` (`wq_ro, gk_ro, gs_ro`) over `[b_0, ..., b_32]`."""

    def __init__(self, vs: int, total_vs: int, D: int, S: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.vs = vs
        self.total_vs = total_vs
        self.D = D
        self.S = S
        gen = torch.Generator(device=device)
        gen.manual_seed(7700 + vs)

        R = max(64, D // 4)
        self.w_in = (
            nn.Parameter(torch.randn((D, D), device=device, dtype=dtype, generator=gen) * (0.5 / math.sqrt(D)))
            if vs == 0
            else None
        )
        # Non-zero initialized depth queries so `Gate G9` exercises full non-trivial softmax routing!
        self.wq = nn.Parameter(torch.randn((S, D), device=device, dtype=torch.float32, generator=gen) * 0.25)
        self.gk = nn.Parameter(1.0 + 0.1 * torch.randn((S, D), device=device, dtype=torch.float32, generator=gen))
        self.gs = nn.Parameter(1.0 + 0.1 * torch.randn((S, D), device=device, dtype=torch.float32, generator=gen))
        self.w_down = nn.Parameter(torch.randn((S, D, R), device=device, dtype=dtype, generator=gen) * (0.45 / math.sqrt(D)))
        self.w_up = nn.Parameter(torch.randn((S, R, D), device=device, dtype=dtype, generator=gen) * (0.45 / math.sqrt(R)))

        if vs == total_vs - 1:
            self.wq_ro = nn.Parameter(torch.randn((1, D), device=device, dtype=torch.float32, generator=gen) * 0.25)
            self.gk_ro = nn.Parameter(1.0 + 0.1 * torch.randn((1, D), device=device, dtype=torch.float32, generator=gen))
            self.gs_ro = nn.Parameter(1.0 + 0.1 * torch.randn((1, D), device=device, dtype=torch.float32, generator=gen))
        else:
            self.wq_ro = self.gk_ro = self.gs_ro = None

    def forward_stage(
        self,
        hist_tensors: List[torch.Tensor],
        kernel_mode: str,
        x_mb: Optional[torch.Tensor] = None,
        target_mb: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Executes stage `vs` returning `(produced_tensors, loss_if_last_stage)`."""
        if self.vs == 0:
            assert x_mb is not None and self.w_in is not None
            b0 = x_mb @ self.w_in
            hist = [b0]
        else:
            hist = hist_tensors

        if kernel_mode == "standard_prenorm":
            h_cur = hist[0]
            for si in range(self.S):
                y_s = FusedSublayerRMSNormFn.apply(h_cur, self.gs[si], 1e-6)
                h_cur = h_cur + (F.silu(y_s @ self.w_down[si]) @ self.w_up[si])
            if self.vs == self.total_vs - 1:
                assert target_mb is not None and self.gs_ro is not None
                y_fin = FusedSublayerRMSNormFn.apply(h_cur, self.gs_ro[0], 1e-6)
                loss = ((y_fin.float() - target_mb.float()) ** 2).mean()
                return [h_cur], loss
            return [h_cur], None

        if kernel_mode == "full_attnres_eager":
            # Full AttnRes (with zero-O1_bar fused Triton kernels for fair Triton-vs-Triton PP comparison):
            # attends over EVERY prior sublayer output (`4 * vs + si + 1` sources up to 128!)
            sub_sources = list(hist)
            for si in range(self.S):
                bl_st = torch.stack(sub_sources, dim=0)
                ub_f = RSIPhase1UnbindInterBlockFn.apply(self.wq[si : si + 1], self.gk[si : si + 1], bl_st, 1e-6, True)
                y_s = RSIPhase2SuperfusedPreNormFn.apply(
                    self.wq[si], self.gk[si], ub_f[0], ub_f[1], None, self.gs[si], bl_st.dtype, 1e-6, 1e-6
                )
                f_out = F.silu(y_s @ self.w_down[si]) @ self.w_up[si]
                sub_sources.append(f_out)
            if self.vs == self.total_vs - 1:
                assert target_mb is not None and self.wq_ro is not None and self.gk_ro is not None and self.gs_ro is not None
                bl_ro = torch.stack(sub_sources, dim=0)
                ub_ro = RSIPhase1UnbindInterBlockFn.apply(self.wq_ro, self.gk_ro, bl_ro, 1e-6, True)
                y_fin = RSIPhase2SuperfusedPreNormFn.apply(
                    self.wq_ro[0], self.gk_ro[0], ub_ro[0], ub_ro[1], None, self.gs_ro[0], bl_ro.dtype, 1e-6, 1e-6
                )
                loss = ((y_fin.float() - target_mb.float()) ** 2).mean()
                return sub_sources[-self.S :], loss
            return ([hist[0]] + sub_sources[-self.S :]) if self.vs == 0 else sub_sources[-self.S :], None

        # Block AttnRes (`eager_block` vs `rsi_block`)
        bl_stack = torch.stack(hist, dim=0)
        p_cur = None
        if kernel_mode == "eager_block":
            for si in range(self.S):
                src = bl_stack if p_cur is None else torch.cat([bl_stack, p_cur.unsqueeze(0)], dim=0)
                kn = eager_rmsnorm_ref(src, self.gk[si].view(1, 1, self.D), 1e-6)
                lg = torch.einsum("d,nmd->nm", self.wq[si].float(), kn.float())
                h_s = torch.einsum("nm,nmd->md", torch.softmax(lg, dim=0), src.float()).to(bl_stack.dtype)
                y_s = eager_rmsnorm_ref(h_s, self.gs[si].view(1, self.D), 1e-6)
                f_out = F.silu(y_s @ self.w_down[si]) @ self.w_up[si]
                p_cur = f_out if p_cur is None else (p_cur + f_out)
        elif kernel_mode == "rsi_block":
            ub = RSIPhase1UnbindInterBlockFn.apply(self.wq, self.gk, bl_stack, 1e-6, True)
            for si in range(self.S):
                y_s = RSIPhase2SuperfusedPreNormFn.apply(
                    self.wq[si], self.gk[si], ub[si], ub[self.S + si], p_cur, self.gs[si], bl_stack.dtype, 1e-6, 1e-6
                )
                f_out = F.silu(y_s @ self.w_down[si]) @ self.w_up[si]
                p_cur = f_out if p_cur is None else (p_cur + f_out)
        else:
            raise ValueError(f"Unknown Suite B kernel_mode: {kernel_mode}")

        assert p_cur is not None
        if self.vs == self.total_vs - 1:
            assert target_mb is not None and self.wq_ro is not None and self.gk_ro is not None and self.gs_ro is not None
            bl_ro = torch.stack(hist + [p_cur], dim=0)
            if kernel_mode == "eager_block":
                kn_ro = eager_rmsnorm_ref(bl_ro, self.gk_ro.view(1, 1, self.D), 1e-6)
                lg_ro = torch.einsum("d,nmd->nm", self.wq_ro[0].float(), kn_ro.float())
                h_ro = torch.einsum("nm,nmd->md", torch.softmax(lg_ro, dim=0), bl_ro.float()).to(bl_ro.dtype)
                y_fin = eager_rmsnorm_ref(h_ro, self.gs_ro.view(1, self.D), 1e-6)
            else:
                ub_ro = RSIPhase1UnbindInterBlockFn.apply(self.wq_ro, self.gk_ro, bl_ro, 1e-6, True)
                y_fin = RSIPhase2SuperfusedPreNormFn.apply(
                    self.wq_ro[0], self.gk_ro[0], ub_ro[0], ub_ro[1], None, self.gs_ro[0], bl_ro.dtype, 1e-6, 1e-6
                )
            loss = ((y_fin.float() - target_mb.float()) ** 2).mean()
            return [p_cur], loss
        return ([hist[0], p_cur] if self.vs == 0 else [p_cur]), None


def build_interleaved_1f1b_schedule(P: int, V: int, N_mb: int) -> Tuple[List[List[Optional[Tuple[str, int, int]]]], Dict[int, List[Tuple[str, int, int, int, int]]]]:
    """Builds exact `142`-tick Megatron-LM causal Interleaved `1F1B` schedule for `(P, V, N_mb)`
    plus deterministic per-tick matched P2P communication plan `comm_plan[tick]`."""
    assert N_mb % P == 0
    tot = N_mb * V
    fwd_q: Dict[int, List[Tuple[int, int]]] = {r: [] for r in range(P)}
    bwd_q: Dict[int, List[Tuple[int, int]]] = {r: [] for r in range(P)}
    for g in range(N_mb // P):
        for v in range(V):
            for k in range(P):
                mb = g * P + k
                for r in range(P):
                    fwd_q[r].append((mb, v * P + r))
        for v in range(V - 1, -1, -1):
            for k in range(P):
                mb = g * P + k
                for r in range(P):
                    bwd_q[r].append((mb, v * P + r))

    fi = [0] * P
    bi = [0] * P
    warmup = [min(tot, (V - 1) * P + 1 + 2 * (P - 1 - r)) for r in range(P)]
    next_steady = ["B"] * P
    fwd_done = set()
    bwd_done = set()
    ticks: List[List[Optional[Tuple[str, int, int]]]] = []

    while any(fi[r] < tot or bi[r] < tot for r in range(P)):
        step: List[Optional[Tuple[str, int, int]]] = [None] * P
        for r in range(P):
            want = None
            if fi[r] < warmup[r]:
                want = "F"
            elif fi[r] < tot and bi[r] < tot:
                want = next_steady[r]
            elif bi[r] < tot:
                want = "B"
            elif fi[r] < tot:
                want = "F"
            if want == "F":
                mb, vs = fwd_q[r][fi[r]]
                if vs == 0 or (mb, vs - 1) in fwd_done:
                    step[r] = ("F", mb, vs)
            elif want == "B":
                mb, vs = bwd_q[r][bi[r]]
                if (mb, vs) in fwd_done and (vs == P * V - 1 or (mb, vs + 1) in bwd_done):
                    step[r] = ("B", mb, vs)
        if not any(x is not None for x in step):
            raise RuntimeError(f"1F1B schedule deadlock: fi={fi}, bi={bi}")
        for r in range(P):
            if step[r] is not None:
                op, mb, vs = step[r]
                if op == "F":
                    fwd_done.add((mb, vs))
                    fi[r] += 1
                    if fi[r] > warmup[r]:
                        next_steady[r] = "B"
                else:
                    bwd_done.add((mb, vs))
                    bi[r] += 1
                    if fi[r] < tot:
                        next_steady[r] = "F"
        ticks.append(step)

    # Schedule every P2P transfer at `t_cons - 1` right before consumer executes at `t_cons`!
    comm_plan: Dict[int, List[Tuple[str, int, int, int, int]]] = {t: [] for t in range(len(ticks))}
    for t_cons, step in enumerate(ticks):
        for r, item in enumerate(step):
            if item is None:
                continue
            op, mb, vs = item
            if op == "F" and vs > 0:
                src_r = (vs - 1) % P
                comm_plan[t_cons - 1].append(("F", mb, vs - 1, src_r, r))
            elif op == "B" and vs < P * V - 1:
                src_r = (vs + 1) % P
                comm_plan[t_cons - 1].append(("B", mb, vs + 1, src_r, r))
    return ticks, comm_plan


def sent_block_keys_for_hop(vs_from: int, P: int, comm_mode: str, S: int) -> List[int]:
    """Exact tensor keys sent across hop `vs_from -> vs_from + 1`:
      - `standard_prenorm`: `[0]` (1 rolling hidden tensor)
      - `full_attnres_naive`: `0 .. S * (vs_from + 1)` (`4*(vs_from + 1) + 1` sublayer tensors!)
      - `block_naive`: `0 .. vs_from + 1` (`vs_from + 2` completed blocks: `Eq. 7`)
      - `block_cross_stage_cache` (`Eq. 8` & `Fig. 3`):
          * `v = 0` (`vs_from + 1 < P`): `0 .. vs_from + 1`
          * `v >= 1` (`vs_from + 1 >= P`): `dst` already cached `0 .. (vs_from + 1 - P + 1)`,
            so ONLY the `P - 1 = 7` incremental blocks `(vs_from + 3 - P) .. (vs_from + 1)` are sent!
    """
    vs_to = vs_from + 1
    if comm_mode == "standard_prenorm":
        return [0]
    if comm_mode == "full_attnres_naive":
        return list(range(S * vs_to + 1))
    if comm_mode == "block_naive":
        return list(range(vs_to + 1))
    if comm_mode in ("block_cross_stage_cache", "broken_cache_no_local_grad_accum"):
        if vs_to < P:
            return list(range(vs_to + 1))
        already_cached_max = vs_to - P + 1
        return list(range(already_cached_max + 1, vs_to + 1))
    raise ValueError(f"Unknown comm_mode: {comm_mode}")


def execute_interleaved_1f1b_step(
    rank: int,
    P: int,
    V: int,
    N_mb: int,
    M_mb: int,
    D: int,
    S: int,
    dtype: torch.dtype,
    device: torch.device,
    local_stages: Dict[int, PPStageBlockModule],
    ticks: List[List[Optional[Tuple[str, int, int]]]],
    comm_plan: Dict[int, List[Tuple[str, int, int, int, int]]],
    kernel_mode: str,
    comm_mode: str,
    x_microbatches_r0: Optional[List[torch.Tensor]],
    target_microbatches_last: Optional[List[torch.Tensor]],
) -> Dict[str, Any]:
    """Executes one complete `142`-tick Interleaved `1F1B` forward+backward step across all `P = 8` GPUs."""
    total_vs = P * V
    for mod in local_stages.values():
        mod.zero_grad(set_to_none=True)

    # Per-rank local activation & gradient caches indexed by `mb` -> `block_key` -> Tensor
    fwd_cache: List[Dict[int, torch.Tensor]] = [{} for _ in range(N_mb)]
    grad_cache: List[Dict[int, torch.Tensor]] = [{} for _ in range(N_mb)]
    # Saved per-(mb, vs) autograd graph handles so backward runs stage-by-stage and frees activations immediately!
    stage_saved: Dict[Tuple[int, int], Tuple[List[int], List[torch.Tensor], List[int], List[torch.Tensor], Optional[torch.Tensor]]] = {}

    # For `naive` comm modes (`block_naive`, `full_attnres_naive`, `standard_prenorm`), keep per-(mb, vs) inbox/outbox
    naive_fwd_box: Dict[Tuple[int, int], Dict[int, torch.Tensor]] = {}
    naive_bwd_box: Dict[Tuple[int, int], Dict[int, torch.Tensor]] = {}

    fwd_p2p_tensors = 0
    bwd_p2p_tensors = 0
    max_hop_tensors = 0
    loss_accum = torch.zeros((), device=device, dtype=torch.float32)

    use_cross_cache = comm_mode in ("block_cross_stage_cache", "broken_cache_no_local_grad_accum")
    broken_cache_bwd = comm_mode == "broken_cache_no_local_grad_accum"

    for t_idx, step in enumerate(ticks):
        my_op = step[rank]
        if my_op is not None:
            op, mb, vs = my_op
            stage_mod = local_stages[vs]
            if op == "F":
                if vs == 0:
                    assert x_microbatches_r0 is not None
                    x_in = x_microbatches_r0[mb].detach().requires_grad_(False)
                    in_keys: List[int] = []
                    in_leaves: List[torch.Tensor] = []
                    produced, loss_mb = stage_mod.forward_stage([], kernel_mode, x_mb=x_in, target_mb=None)
                    if kernel_mode == "standard_prenorm":
                        out_keys = [0]
                    elif kernel_mode == "full_attnres_eager":
                        out_keys = list(range(S + 1))
                    else:
                        out_keys = [0, 1]
                else:
                    if kernel_mode == "standard_prenorm":
                        in_keys = [0]
                    elif kernel_mode == "full_attnres_eager":
                        in_keys = list(range(S * vs + 1))
                    else:
                        in_keys = list(range(vs + 1))

                    src_dict = fwd_cache[mb] if use_cross_cache else naive_fwd_box.pop((mb, vs))
                    in_leaves = [src_dict[k].detach().requires_grad_(True) for k in in_keys]
                    tgt_in = target_microbatches_last[mb] if (vs == total_vs - 1 and target_microbatches_last is not None) else None
                    produced, loss_mb = stage_mod.forward_stage(in_leaves, kernel_mode, x_mb=None, target_mb=tgt_in)

                    if kernel_mode == "standard_prenorm":
                        out_keys = [0]
                    elif kernel_mode == "full_attnres_eager":
                        out_keys = list(range(S * vs + 1, S * (vs + 1) + 1)) if vs < total_vs - 1 else []
                    else:
                        out_keys = [vs + 1] if vs < total_vs - 1 else []

                if loss_mb is not None:
                    scaled_loss = loss_mb / float(N_mb)
                    loss_accum = loss_accum + scaled_loss.detach()
                else:
                    scaled_loss = None

                if use_cross_cache:
                    for ok_i, ot in zip(out_keys, produced):
                        fwd_cache[mb][ok_i] = ot.detach()
                else:
                    if vs < total_vs - 1:
                        out_map = {} if kernel_mode == "standard_prenorm" else {k: lf.detach() for k, lf in zip(in_keys, in_leaves)}
                        for ok_i, ot in zip(out_keys, produced):
                            out_map[ok_i] = ot.detach()
                        naive_fwd_box[(mb, vs)] = out_map

                stage_saved[(mb, vs)] = (in_keys, in_leaves, out_keys, produced, scaled_loss)

            else:  # op == "B"
                in_keys, in_leaves, out_keys, produced, scaled_loss = stage_saved.pop((mb, vs))
                bmap_rem: Dict[int, torch.Tensor] = {}
                if vs == total_vs - 1:
                    assert scaled_loss is not None
                    scaled_loss.backward()
                else:
                    if use_cross_cache:
                        out_grads = [grad_cache[mb].pop(ok_i) for ok_i in out_keys]
                    else:
                        bmap_rem = naive_bwd_box.pop((mb, vs))
                        out_grads = [bmap_rem.pop(ok_i) for ok_i in out_keys]
                    torch.autograd.backward(produced, out_grads)

                if vs > 0:
                    if use_cross_cache:
                        for k, lf in zip(in_keys, in_leaves):
                            assert lf.grad is not None
                            if broken_cache_bwd:
                                # Red-team bug: overwrites instead of accumulating cross-virtual-chunk gradients!
                                grad_cache[mb][k] = lf.grad.detach()
                            else:
                                prev_g = grad_cache[mb].get(k)
                                grad_cache[mb][k] = lf.grad.detach() if prev_g is None else (prev_g + lf.grad.detach())
                    else:
                        if kernel_mode == "standard_prenorm":
                            bmap_out = {0: in_leaves[0].grad.detach()}
                        else:
                            bmap_out = dict(bmap_rem)
                            for k, lf in zip(in_keys, in_leaves):
                                assert lf.grad is not None
                                prev_g = bmap_out.get(k)
                                bmap_out[k] = lf.grad.detach() if prev_g is None else (prev_g + lf.grad.detach())
                        naive_bwd_box[(mb, vs)] = bmap_out

                if use_cross_cache and vs < P:
                    # First virtual chunk `v = 0` (`vs < P`): last consumer of `fwd_cache[mb]` on this rank!
                    fwd_cache[mb].clear()

        # Execute deterministic matched P2P transfers scheduled at end of tick `t_idx`
        events_t = comm_plan.get(t_idx, [])
        if events_t:
            p2p_ops: List[dist.P2POp] = []
            pending_recvs: List[Tuple[str, int, int, List[int], List[torch.Tensor]]] = []
            for dir_flag, mb_c, vs_src, src_r, dst_r in events_t:
                if dir_flag == "F":
                    keys = sent_block_keys_for_hop(vs_src, P, comm_mode, S)
                    fwd_p2p_tensors += len(keys)
                    max_hop_tensors = max(max_hop_tensors, len(keys))
                    if rank == src_r:
                        src_map = fwd_cache[mb_c] if use_cross_cache else naive_fwd_box.pop((mb_c, vs_src))
                        for k in keys:
                            p2p_ops.append(dist.P2POp(dist.isend, src_map[k].contiguous(), dst_r))
                    elif rank == dst_r:
                        rbufs = [torch.empty((M_mb, D), device=device, dtype=dtype) for _ in keys]
                        for rb in rbufs:
                            p2p_ops.append(dist.P2POp(dist.irecv, rb, src_r))
                        pending_recvs.append(("F", mb_c, vs_src + 1, keys, rbufs))
                else:  # dir_flag == "B" (`vs_src -> vs_src - 1`)
                    vs_dst = vs_src - 1
                    keys = sent_block_keys_for_hop(vs_dst, P, comm_mode, S)
                    bwd_p2p_tensors += len(keys)
                    max_hop_tensors = max(max_hop_tensors, len(keys))
                    if rank == src_r:
                        if use_cross_cache:
                            for k in keys:
                                gt = grad_cache[mb_c].pop(k).to(dtype).contiguous()
                                p2p_ops.append(dist.P2POp(dist.isend, gt, dst_r))
                        else:
                            bmap_s = naive_bwd_box.pop((mb_c, vs_src))
                            for k in keys:
                                p2p_ops.append(dist.P2POp(dist.isend, bmap_s[k].to(dtype).contiguous(), dst_r))
                    elif rank == dst_r:
                        rbufs = [torch.empty((M_mb, D), device=device, dtype=dtype) for _ in keys]
                        for rb in rbufs:
                            p2p_ops.append(dist.P2POp(dist.irecv, rb, src_r))
                        pending_recvs.append(("B", mb_c, vs_dst, keys, rbufs))

            if p2p_ops:
                reqs = dist.batch_isend_irecv(p2p_ops)
                for rq in reqs:
                    rq.wait()

            for dir_flag, mb_c, vs_target, keys, rbufs in pending_recvs:
                if dir_flag == "F":
                    if use_cross_cache:
                        for k, rb in zip(keys, rbufs):
                            fwd_cache[mb_c][k] = rb
                    else:
                        naive_fwd_box[(mb_c, vs_target)] = {k: rb for k, rb in zip(keys, rbufs)}
                else:
                    if use_cross_cache:
                        for k, rb in zip(keys, rbufs):
                            if broken_cache_bwd:
                                grad_cache[mb_c][k] = rb
                            else:
                                prev_g = grad_cache[mb_c].get(k)
                                grad_cache[mb_c][k] = rb if prev_g is None else (prev_g + rb)
                    else:
                        naive_bwd_box[(mb_c, vs_target)] = {k: rb for k, rb in zip(keys, rbufs)}

    if use_cross_cache and not broken_cache_bwd:
        for mb_i, gdict in enumerate(grad_cache):
            if gdict:
                raise RuntimeError(f"Rank {rank} leaked unconsumed cached gradients in mb={mb_i}: keys={list(gdict.keys())}")

    # Broadcast scalar loss from last stage (`rank == P - 1`) to all ranks
    dist.broadcast(loss_accum, src=P - 1)
    bytes_per_tensor = M_mb * D * (4 if dtype == torch.float32 else 2)
    return {
        "loss": float(loss_accum.item()),
        "fwd_p2p_tensors_total": fwd_p2p_tensors,
        "bwd_p2p_tensors_total": bwd_p2p_tensors,
        "fwd_p2p_tensors_per_mb": round(fwd_p2p_tensors / float(N_mb), 2),
        "max_hop_tensors": max_hop_tensors,
        "total_p2p_mib": round((fwd_p2p_tensors + bwd_p2p_tensors) * bytes_per_tensor / (1024.0 * 1024.0), 3),
    }


def _run_1gpu_monolithic_32stage_oracle(
    P: int, V: int, N_mb: int, M_mb: int, D: int, S: int, dtype: torch.dtype, device: torch.device,
    x_mbs: List[torch.Tensor], tgt_mbs: List[torch.Tensor],
) -> Tuple[float, Dict[int, Dict[str, torch.Tensor]]]:
    total_vs = P * V
    stages = [PPStageBlockModule(vs, total_vs, D, S, dtype, device) for vs in range(total_vs)]
    for st in stages:
        st.zero_grad(set_to_none=True)
    loss_tot = 0.0
    for mb in range(N_mb):
        blocks_acc: List[torch.Tensor] = []
        loss_mb: Optional[torch.Tensor] = None
        for vs in range(total_vs):
            prod, l_mb = stages[vs].forward_stage(
                blocks_acc,
                "rsi_block",
                x_mb=(x_mbs[mb] if vs == 0 else None),
                target_mb=(tgt_mbs[mb] if vs == total_vs - 1 else None),
            )
            blocks_acc.extend(prod)
            if l_mb is not None:
                loss_mb = l_mb / float(N_mb)
        assert loss_mb is not None
        loss_tot += float(loss_mb.detach().item())
        loss_mb.backward()

    grads_by_vs: Dict[int, Dict[str, torch.Tensor]] = {}
    for vs, st in enumerate(stages):
        gd: Dict[str, torch.Tensor] = {
            "wq": st.wq.grad.detach().to(torch.float64).cpu(),
            "gk": st.gk.grad.detach().to(torch.float64).cpu(),
            "gs": st.gs.grad.detach().to(torch.float64).cpu(),
            "w_down": st.w_down.grad.detach().to(torch.float64).cpu(),
            "w_up": st.w_up.grad.detach().to(torch.float64).cpu(),
        }
        if st.w_in is not None and st.w_in.grad is not None:
            gd["w_in"] = st.w_in.grad.detach().to(torch.float64).cpu()
        if st.wq_ro is not None and st.wq_ro.grad is not None:
            gd["wq_ro"] = st.wq_ro.grad.detach().to(torch.float64).cpu()
        grads_by_vs[vs] = gd
    return loss_tot, grads_by_vs


def run_suite_b_pipeline_parallel_caching(rank: int, world_size: int, device: torch.device, mode: str) -> Dict[str, Any]:
    enforce_strict_ieee_precision()
    if world_size != 8:
        raise RuntimeError(f"Suite B requires 8xA100 Pipeline Parallelism (world_size=8), got {world_size}")

    P, V, S, N_mb = 8, 4, 4, 16
    total_vs = P * V
    ticks, comm_plan = build_interleaved_1f1b_schedule(P, V, N_mb)

    # 1. Gate G9 Single-GPU 32-Stage Oracle Verification (FP32 & BF16 + Broken Cached Backward Red-Team Audit)
    M_g9, D_g9 = 128, 256
    g9_results: Dict[str, Any] = {}
    for dt, dt_label, tol in ((torch.float32, "fp32", 2e-5), (torch.bfloat16, "bf16", 3.0e-2)):
        g_in = torch.Generator(device=device)
        g_in.manual_seed(8800)
        x_mbs = [torch.randn((M_g9, D_g9), device=device, dtype=dt, generator=g_in) * 0.4 for _ in range(N_mb)]
        tgt_mbs = [torch.randn((M_g9, D_g9), device=device, dtype=dt, generator=g_in) * 0.4 for _ in range(N_mb)]

        orc_loss = 0.0
        orc_grads: Dict[int, Dict[str, torch.Tensor]] = {}
        if rank == 0:
            orc_loss, orc_grads = _run_1gpu_monolithic_32stage_oracle(P, V, N_mb, M_g9, D_g9, S, dt, device, x_mbs, tgt_mbs)
        dist.barrier()

        local_st_g9 = {
            v * P + rank: PPStageBlockModule(v * P + rank, total_vs, D_g9, S, dt, device) for v in range(V)
        }
        step_g9 = execute_interleaved_1f1b_step(
            rank, P, V, N_mb, M_g9, D_g9, S, dt, device, local_st_g9, ticks, comm_plan,
            kernel_mode="rsi_block", comm_mode="block_cross_stage_cache",
            x_microbatches_r0=(x_mbs if rank == 0 else None),
            target_microbatches_last=(tgt_mbs if rank == P - 1 else None),
        )

        # Gather all 32 stage gradients to rank 0 for exhaustive Gate G9 comparison
        local_pack: Dict[int, Dict[str, torch.Tensor]] = {}
        for vs_i, st_i in local_st_g9.items():
            d_i = {
                "wq": st_i.wq.grad.detach().to(torch.float64).cpu(),
                "gk": st_i.gk.grad.detach().to(torch.float64).cpu(),
                "gs": st_i.gs.grad.detach().to(torch.float64).cpu(),
                "w_down": st_i.w_down.grad.detach().to(torch.float64).cpu(),
                "w_up": st_i.w_up.grad.detach().to(torch.float64).cpu(),
            }
            if st_i.w_in is not None and st_i.w_in.grad is not None:
                d_i["w_in"] = st_i.w_in.grad.detach().to(torch.float64).cpu()
            if st_i.wq_ro is not None and st_i.wq_ro.grad is not None:
                d_i["wq_ro"] = st_i.wq_ro.grad.detach().to(torch.float64).cpu()
            local_pack[vs_i] = d_i

        gathered: List[Optional[Dict[int, Dict[str, torch.Tensor]]]] = [None] * world_size
        dist.all_gather_object(gathered, local_pack)

        if rank == 0:
            merged: Dict[int, Dict[str, torch.Tensor]] = {}
            for gp in gathered:
                assert gp is not None
                merged.update(gp)
            loss_rel = abs(step_g9["loss"] - orc_loss) / max(abs(orc_loss), 1e-12)
            max_wq_rel = max(compare_tensor_to_fp64(merged[vs_]["wq"], orc_grads[vs_]["wq"])["rel_l2"] for vs_ in range(total_vs))
            max_gk_rel = max(compare_tensor_to_fp64(merged[vs_]["gk"], orc_grads[vs_]["gk"])["rel_l2"] for vs_ in range(total_vs))
            max_sub_rel = max(
                max(
                    compare_tensor_to_fp64(merged[vs_]["w_down"], orc_grads[vs_]["w_down"])["rel_l2"],
                    compare_tensor_to_fp64(merged[vs_]["w_up"], orc_grads[vs_]["w_up"])["rel_l2"],
                )
                for vs_ in range(total_vs)
            )
            win_rel = compare_tensor_to_fp64(merged[0]["w_in"], orc_grads[0]["w_in"])["rel_l2"]
            ro_rel = compare_tensor_to_fp64(merged[total_vs - 1]["wq_ro"], orc_grads[total_vs - 1]["wq_ro"])["rel_l2"]
            worst_all = max(loss_rel, max_wq_rel, max_gk_rel, max_sub_rel, win_rel, ro_rel)
            g9_results[dt_label] = {
                "passed": bool(worst_all <= tol),
                "tol": tol,
                "loss_rel_diff": round(loss_rel, 8),
                "w_in_grad_rel_l2": round(win_rel, 8),
                "max_wq_grad_rel_l2": round(max_wq_rel, 8),
                "max_gk_grad_rel_l2": round(max_gk_rel, 8),
                "max_sublayer_grad_rel_l2": round(max_sub_rel, 8),
                "readout_wq_grad_rel_l2": round(ro_rel, 8),
                "max_overall_rel_l2": round(worst_all, 8),
            }

        # Also audit broken cached backward (`broken_cache_no_local_grad_accum`) in FP32 to prove Gate G9 catches missing cross-chunk grad accumulation!
        if dt == torch.float32:
            step_broken = execute_interleaved_1f1b_step(
                rank, P, V, N_mb, M_g9, D_g9, S, dt, device, local_st_g9, ticks, comm_plan,
                kernel_mode="rsi_block", comm_mode="broken_cache_no_local_grad_accum",
                x_microbatches_r0=(x_mbs if rank == 0 else None),
                target_microbatches_last=(tgt_mbs if rank == P - 1 else None),
            )
            if rank == 0:
                corrupted_win = compare_tensor_to_fp64(
                    local_st_g9[0].w_in.grad.detach().cpu(), orc_grads[0]["w_in"]
                )["rel_l2"]
                g9_results["broken_cached_bwd_red_team"] = {
                    "forward_loss_exact": abs(step_broken["loss"] - orc_loss) <= 1e-6,
                    "corrupted_w_in_grad_rel_l2": round(corrupted_win, 6),
                    "caught_by_gate_g9": bool(corrupted_win > 0.10),
                }

    # 2. Production 8-GPU NVSwitch Interleaved 1F1B Pipeline Benchmark (`P=8, V=4, 32 Virtual Stages`)
    M_pp = 512 if mode == "smoke" else 960
    D_pp = 1536 if mode == "smoke" else 2048
    n_warm_pp = 2
    n_timed_pp = 4 if mode == "smoke" else 7

    g_pp = torch.Generator(device=device)
    g_pp.manual_seed(9100 + rank)
    x_pp = [torch.randn((M_pp, D_pp), device=device, dtype=torch.bfloat16, generator=g_pp) * 0.3 for _ in range(N_mb)] if rank == 0 else None
    tgt_pp = [torch.randn((M_pp, D_pp), device=device, dtype=torch.bfloat16, generator=g_pp) * 0.3 for _ in range(N_mb)] if rank == P - 1 else None

    local_st_pp = {
        v * P + rank: PPStageBlockModule(v * P + rank, total_vs, D_pp, S, torch.bfloat16, device) for v in range(V)
    }

    arms_pp = [
        ("p0_full_attnres_naive_pp", "full_attnres_eager", "full_attnres_naive"),
        ("p1_eager_block_attnres_naive_pp", "eager_block", "block_naive"),
        ("p2_rsi_block_attnres_naive_pp", "rsi_block", "block_naive"),
        ("p3_rsi_block_attnres_cross_stage_cache", "rsi_block", "block_cross_stage_cache"),
        ("p4_standard_prenorm_reference_pp", "standard_prenorm", "standard_prenorm"),
    ]

    arm_metrics: Dict[str, Any] = {}
    for arm_name, kmode, cmode in arms_pp:
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        last_meta: Dict[str, Any] = {}
        for _ in range(n_warm_pp):
            last_meta = execute_interleaved_1f1b_step(
                rank, P, V, N_mb, M_pp, D_pp, S, torch.bfloat16, device, local_st_pp, ticks, comm_plan, kmode, cmode, x_pp, tgt_pp
            )
        torch.cuda.synchronize(device)
        dist.barrier()

        torch.cuda.reset_peak_memory_stats(device)
        samples_pp: List[float] = []
        for _ in range(n_timed_pp):
            dist.barrier()
            t0 = time.perf_counter()
            last_meta = execute_interleaved_1f1b_step(
                rank, P, V, N_mb, M_pp, D_pp, S, torch.bfloat16, device, local_st_pp, ticks, comm_plan, kmode, cmode, x_pp, tgt_pp
            )
            torch.cuda.synchronize(device)
            dist.barrier()
            samples_pp.append((time.perf_counter() - t0) * 1000.0)

        local_pk_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        pk_t = torch.tensor([local_pk_mib], device=device, dtype=torch.float64)
        dist.all_reduce(pk_t, op=dist.ReduceOp.MAX)
        max_rank_pk_mib = float(pk_t.item())

        ss = sorted(samples_pp)
        q25 = ss[max(0, (len(ss) - 1) // 4)]
        q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
        arm_metrics[arm_name] = {
            "median_ms": round(float(statistics.median(ss)), 3),
            "iqr_ms": round(float(q75 - q25), 3),
            "min_ms": round(float(ss[0]), 3),
            "max_ms": round(float(ss[-1]), 3),
            "max_rank_peak_alloc_mib": round(max_rank_pk_mib, 2),
            **last_meta,
        }

    del local_st_pp
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    if rank != 0:
        return {}

    p0 = arm_metrics["p0_full_attnres_naive_pp"]
    p1 = arm_metrics["p1_eager_block_attnres_naive_pp"]
    p2 = arm_metrics["p2_rsi_block_attnres_naive_pp"]
    p3 = arm_metrics["p3_rsi_block_attnres_cross_stage_cache"]
    p4 = arm_metrics["p4_standard_prenorm_reference_pp"]
    return {
        "topology": {
            "P_stages": P,
            "V_virtual_chunks": V,
            "total_virtual_stages": total_vs,
            "S_sublayers_per_stage": S,
            "total_residual_sublayers": total_vs * S,
            "N_microbatches": N_mb,
            "M_tokens_per_mb": M_pp,
            "D_hidden": D_pp,
            "schedule_ticks": len(ticks),
            "ideal_ticks": 2 * N_mb * V,
            "bubble_ticks": len(ticks) - 2 * N_mb * V,
            "bubble_fraction": round((len(ticks) - 2 * N_mb * V) / float(len(ticks)), 4),
        },
        "gate_g9_single_gpu_parity": g9_results,
        "arms": arm_metrics,
        "speedups_and_reductions": {
            "speedup_p3_vs_p0_full_attnres": round(p0["median_ms"] / p3["median_ms"], 3),
            "speedup_p3_vs_p1_eager_block_naive": round(p1["median_ms"] / p3["median_ms"], 3),
            "speedup_p3_vs_p2_pure_cross_stage_cache_gain": round(p2["median_ms"] / p3["median_ms"], 3),
            "comm_reduction_p3_vs_p0_full_attnres": round(p0["total_p2p_mib"] / p3["total_p2p_mib"], 3),
            "comm_reduction_p3_vs_p1_naive_block": round(p1["total_p2p_mib"] / p3["total_p2p_mib"], 3),
            "peak_hop_reduction_p3_vs_p0": round(p0["max_hop_tensors"] / float(p3["max_hop_tensors"]), 3),
            "peak_hop_reduction_p3_vs_p1": round(p1["max_hop_tensors"] / float(p3["max_hop_tensors"]), 3),
            "hbm_saved_mib_per_rank_vs_p0": round(p0["max_rank_peak_alloc_mib"] - p3["max_rank_peak_alloc_mib"], 2),
            "hbm_saved_mib_per_rank_vs_p1": round(p1["max_rank_peak_alloc_mib"] - p3["max_rank_peak_alloc_mib"], 2),
            "overhead_ratio_p3_vs_standard_prenorm_pp": round(p3["median_ms"] / p4["median_ms"], 3),
        },
    }


# =============================================================================
# Suite C: Fixed-Wall-Clock (`15.0s`) 8xA100 DDP `val_bpb` RSI Loop +
#          5-Arm Matched-Step Deep-24-Layer Residual-Topology & PreNorm-Dilution Law
# =============================================================================


def build_multidepth_compositional_splits(vocab_size: int = 1024, seq_len: int = 192) -> Dict[str, Any]:
    """Deterministic multi-scale synthetic language corpus (`train / val / untouched test`):
    Each sequence interleaves Zipfian Markov prose with context-dependent multi-depth queries:
      - `TAG_SHALLOW`: target equals direct lexical key (`k1`) placed earlier in sequence
        (directly recoverable from un-diluted token embedding `b_0` via 1 attention hop, degraded by
        deep `PreNorm` dilution!).
      - `TAG_MID`: target equals pairwise XOR-hash `f_mid(k1, k2)` requiring 2-hop composition.
      - `TAG_DEEP`: target equals 4-way non-associative composition `f_deep(k1, k2, k3, k4)`
        requiring multi-layer deep composition across stacked attention+SwiGLU layers!
    Crucially, because `TAG_SHALLOW`, `TAG_MID`, and `TAG_DEEP` occur at variable positions in
    the SAME sequence, static depth weights (`DenseFormer`) cannot adapt per token, whereas
    content-dependent `Block AttnRes (N=8)` and `Full AttnRes (N=48)` dynamically route depth!"""
    g = torch.Generator()
    g.manual_seed(20260316)

    TAG_SHALLOW = 1
    TAG_MID = 2
    TAG_DEEP = 3
    ARROW_TOK = 4
    SEP_TOK = 5
    ENT_BASE = 16
    N_ENTS = 24
    PROSE_BASE = 256
    N_PROSE = vocab_size - PROSE_BASE

    # Zipfian bigram prose transition distribution
    ranks = torch.arange(1, N_PROSE + 1, dtype=torch.float32)
    zipf = (1.0 / (ranks ** 1.05))
    zipf = zipf / zipf.sum()
    prose_trans = (0.75 * zipf.unsqueeze(0) + 0.25 * torch.rand((N_PROSE, N_PROSE), generator=g) * zipf.unsqueeze(0))
    prose_trans = prose_trans / prose_trans.sum(dim=-1, keepdim=True)

    # Deterministic byte lengths per token (`bpb` conversion)
    tok_bytes = torch.full((vocab_size,), 4.0, dtype=torch.float32)
    tok_bytes[:PROSE_BASE] = 3.5
    tok_bytes[PROSE_BASE:] = 3.0 + 2.0 * (torch.arange(N_PROSE, dtype=torch.float32) / float(N_PROSE))

    def _make_split(n_seqs: int, split_seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Each sequence defines two explicit 3-hop in-context pointer chains in the header
        (`e0 -> e1, e1 -> e2, e2 -> e3` and `e4 -> e5, e5 -> e6, e6 -> e7`, shuffled per sequence!)
        interleaved with Zipfian Markov prose and 12 context-dependent depth queries:
          - `[TAG_SHALLOW, root] -> hop_1` (1-hop direct lexical recall; hurt by PreNorm dilution of `b_0`)
          - `[TAG_MID,     root] -> hop_2` (2-hop pointer composition across early+mid blocks)
          - `[TAG_DEEP,    root] -> hop_3` (3-hop pointer composition across full 24-layer stack)
        Every query is 100% deductively solvable from the sequence's own prefix without memorizing tables!"""
        gs = torch.Generator()
        gs.manual_seed(split_seed)
        seqs = torch.empty((n_seqs, seq_len + 1), dtype=torch.long)
        scale_mask = torch.zeros((n_seqs, seq_len), dtype=torch.long)

        for i in range(n_seqs):
            ents = (ENT_BASE + torch.randperm(N_ENTS, generator=gs)[:8]).tolist()
            e0, e1, e2, e3, e4, e5, e6, e7 = ents
            edges = [
                (e0, e1), (e1, e2), (e2, e3),
                (e4, e5), (e5, e6), (e6, e7),
            ]
            perm_e = torch.randperm(len(edges), generator=gs).tolist()
            header: List[int] = []
            for idx_e in perm_e:
                u_tok, v_tok = edges[idx_e]
                header.extend([u_tok, ARROW_TOK, v_tok, SEP_TOK])

            row = [0] * (seq_len + 1)
            for pos_h, tok_h in enumerate(header):
                row[pos_h] = tok_h

            cur_p = int(torch.multinomial(zipf, 1, generator=gs).item())
            pos = len(header)
            q_slots = {
                32:  (TAG_SHALLOW, e0, e1, 1),
                46:  (TAG_MID,     e0, e2, 2),
                60:  (TAG_DEEP,    e0, e3, 3),
                76:  (TAG_SHALLOW, e4, e5, 1),
                92:  (TAG_MID,     e4, e6, 2),
                108: (TAG_DEEP,    e4, e7, 3),
                124: (TAG_SHALLOW, e1, e2, 1),
                140: (TAG_MID,     e1, e3, 2),
                156: (TAG_SHALLOW, e5, e6, 1),
                170: (TAG_MID,     e5, e7, 2),
                182: (TAG_DEEP,    e0, e3, 3),
            }
            while pos < seq_len + 1:
                if pos in q_slots:
                    q_tag, q_root, ans, stype = q_slots[pos]
                    if pos + 2 < seq_len + 1:
                        row[pos] = q_tag
                        row[pos + 1] = q_root
                        row[pos + 2] = ans
                        if pos + 1 < seq_len:
                            scale_mask[i, pos + 1] = stype
                        pos += 3
                        continue
                cur_p = int(torch.multinomial(prose_trans[cur_p], 1, generator=gs).item())
                row[pos] = PROSE_BASE + cur_p
                pos += 1
            seqs[i] = torch.tensor(row, dtype=torch.long)
        return seqs, scale_mask

    train_seqs, train_mask = _make_split(4096, 51001)
    val_seqs, val_mask = _make_split(512, 52002)
    test_seqs, test_mask = _make_split(512, 53003)
    return {
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "tok_bytes": tok_bytes,
        "train": (train_seqs, train_mask),
        "val": (val_seqs, val_mask),
        "test": (test_seqs, test_mask),
    }


class Deep24LayerResidualLM(nn.Module):
    """Deep 24-Transformer-Layer (`48` Residual Sublayers: `24` Causal FlashAttention + `24` SwiGLU MLPs)
    supporting `5` residual topologies (`standard_prenorm`, `denseformer_static`, `highway_sigmoid_gate`,
    `block_attnres_n8`, `full_attnres_n48`) and `4` `Block AttnRes` kernel/init regimes."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 24,
        n_heads: int = 4,
        n_blocks: int = 8,
        topology: str = "block_attnres_n8",
        kernel_backend: str = "rsi_superfused",
        zero_query_init: bool = True,
        use_key_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        assert n_layers % n_blocks == 0
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_sublayers = 2 * n_layers  # 48 residual sublayers!
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_blocks = n_blocks
        self.sub_per_block = self.n_sublayers // n_blocks  # 6 sublayers/block
        self.topology = topology
        self.kernel_backend = kernel_backend
        self.use_key_rmsnorm = use_key_rmsnorm

        d_ff = int(2 * d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        self.qkv_proj = nn.ModuleList([nn.Linear(d_model, 3 * d_model, bias=False) for _ in range(n_layers)])
        self.out_proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(n_layers)])
        self.gate_up_proj = nn.ModuleList([nn.Linear(d_model, 2 * d_ff, bias=False) for _ in range(n_layers)])
        self.down_proj = nn.ModuleList([nn.Linear(d_ff, d_model, bias=False) for _ in range(n_layers)])

        # Scale residual output projections (`GPT-2 / Llama` `1 / sqrt(2 * n_layers)` init)
        with torch.no_grad():
            res_scale = 1.0 / math.sqrt(float(self.n_sublayers))
            for l_i in range(n_layers):
                self.out_proj[l_i].weight.mul_(res_scale)
                self.down_proj[l_i].weight.mul_(res_scale)

        self.sub_gammas = nn.Parameter(torch.ones(self.n_sublayers, d_model, dtype=torch.float32))
        self.final_gamma = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying

        # Topology-specific parameters
        if topology == "block_attnres_n8":
            wq_init = (
                torch.zeros(n_blocks, self.sub_per_block, d_model, dtype=torch.float32)
                if zero_query_init
                else torch.randn(n_blocks, self.sub_per_block, d_model, dtype=torch.float32) * 1.5
            )
            ro_init = (
                torch.zeros(1, d_model, dtype=torch.float32)
                if zero_query_init
                else torch.randn(1, d_model, dtype=torch.float32) * 1.5
            )
            self.w_queries = nn.Parameter(wq_init)
            self.wq_readout = nn.Parameter(ro_init)
            if use_key_rmsnorm:
                self.gamma_keys = nn.Parameter(torch.ones(n_blocks, self.sub_per_block, d_model, dtype=torch.float32))
                self.gk_readout = nn.Parameter(torch.ones(1, d_model, dtype=torch.float32))
            else:
                self.register_buffer("gamma_keys", torch.ones(n_blocks, self.sub_per_block, d_model, dtype=torch.float32))
                self.register_buffer("gk_readout", torch.ones(1, d_model, dtype=torch.float32))
        elif topology == "full_attnres_n48":
            self.w_queries = nn.Parameter(torch.zeros(self.n_sublayers, d_model, dtype=torch.float32))
            self.gamma_keys = nn.Parameter(torch.ones(self.n_sublayers, d_model, dtype=torch.float32))
            self.wq_readout = nn.Parameter(torch.zeros(1, d_model, dtype=torch.float32))
            self.gk_readout = nn.Parameter(torch.ones(1, d_model, dtype=torch.float32))
        elif topology == "denseformer_static":
            # Static depth-mixing scalars (`1` per prior source at each sublayer, uniform init)
            self.static_logits = nn.ParameterList(
                [nn.Parameter(torch.zeros(si + 1, dtype=torch.float32)) for si in range(self.n_sublayers + 1)]
            )
        elif topology == "highway_sigmoid_gate":
            self.gate_projs = nn.ModuleList([nn.Linear(d_model, 1, bias=True) for _ in range(self.n_sublayers)])
            for gp in self.gate_projs:
                nn.init.zeros_(gp.weight)
                nn.init.constant_(gp.bias, -1.0)

    def _run_sublayer_compute(self, sub_idx: int, y_norm_flat: torch.Tensor, B: int, T: int) -> torch.Tensor:
        layer_idx = sub_idx // 2
        if sub_idx % 2 == 0:
            qkv = self.qkv_proj[layer_idx](y_norm_flat).view(B, T, 3, self.n_heads, self.head_dim)
            attn_out = flash_attn_func(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], causal=True)
            return self.out_proj[layer_idx](attn_out.reshape(B * T, self.d_model))
        gu = self.gate_up_proj[layer_idx](y_norm_flat)
        g_part, u_part = gu.chunk(2, dim=-1)
        return self.down_proj[layer_idx](F.silu(g_part) * u_part)

    def forward(self, idx: torch.Tensor, collect_diagnostics: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, T = idx.shape
        M = B * T
        D = self.d_model
        b0 = (self.tok_emb(idx) + self.pos_emb[:, :T, :]).to(torch.bfloat16).reshape(M, D)

        sub_hidden_norms: List[float] = []
        sub_outputs_for_grad: List[torch.Tensor] = []

        if self.topology == "standard_prenorm":
            h = b0
            for si in range(self.n_sublayers):
                if collect_diagnostics:
                    sub_hidden_norms.append(float(torch.sqrt((h.detach().float() ** 2).mean()).item()))
                y_n = FusedSublayerRMSNormFn.apply(h, self.sub_gammas[si], 1e-6)
                f_out = self._run_sublayer_compute(si, y_n, B, T)
                if collect_diagnostics and f_out.requires_grad:
                    f_out.retain_grad()
                    sub_outputs_for_grad.append(f_out)
                h = h + f_out
            if collect_diagnostics:
                sub_hidden_norms.append(float(torch.sqrt((h.detach().float() ** 2).mean()).item()))
            y_fin = FusedSublayerRMSNormFn.apply(h, self.final_gamma, 1e-6)

        elif self.topology == "highway_sigmoid_gate":
            h = b0
            for si in range(self.n_sublayers):
                if collect_diagnostics:
                    sub_hidden_norms.append(float(torch.sqrt((h.detach().float() ** 2).mean()).item()))
                y_n = FusedSublayerRMSNormFn.apply(h, self.sub_gammas[si], 1e-6)
                f_out = self._run_sublayer_compute(si, y_n, B, T)
                if collect_diagnostics and f_out.requires_grad:
                    f_out.retain_grad()
                    sub_outputs_for_grad.append(f_out)
                gate = torch.sigmoid(self.gate_projs[si](y_n.float())).to(torch.bfloat16)
                h = (1.0 - gate) * h + gate * f_out
            if collect_diagnostics:
                sub_hidden_norms.append(float(torch.sqrt((h.detach().float() ** 2).mean()).item()))
            y_fin = FusedSublayerRMSNormFn.apply(h, self.final_gamma, 1e-6)

        elif self.topology == "denseformer_static":
            sources: List[torch.Tensor] = [b0]
            for si in range(self.n_sublayers):
                w_st = torch.softmax(self.static_logits[si], dim=0)
                st_stack = torch.stack(sources, dim=0)
                h = torch.einsum("n,nmd->md", w_st, st_stack.float()).to(torch.bfloat16)
                if collect_diagnostics:
                    sub_hidden_norms.append(float(torch.sqrt((h.detach().float() ** 2).mean()).item()))
                y_n = FusedSublayerRMSNormFn.apply(h, self.sub_gammas[si], 1e-6)
                f_out = self._run_sublayer_compute(si, y_n, B, T)
                if collect_diagnostics and f_out.requires_grad:
                    f_out.retain_grad()
                    sub_outputs_for_grad.append(f_out)
                sources.append(f_out)
            w_fin = torch.softmax(self.static_logits[self.n_sublayers], dim=0)
            h_fin = torch.einsum("n,nmd->md", w_fin, torch.stack(sources, dim=0).float()).to(torch.bfloat16)
            if collect_diagnostics:
                sub_hidden_norms.append(float(torch.sqrt((h_fin.detach().float() ** 2).mean()).item()))
            y_fin = FusedSublayerRMSNormFn.apply(h_fin, self.final_gamma, 1e-6)

        elif self.topology == "full_attnres_n48":
            sources = [b0]
            for si in range(self.n_sublayers):
                bl_st = torch.stack(sources, dim=0)
                if self.kernel_backend == "rsi_superfused":
                    ub = RSIPhase1UnbindInterBlockFn.apply(
                        self.w_queries[si : si + 1], self.gamma_keys[si : si + 1], bl_st, 1e-6, True
                    )
                    if collect_diagnostics:
                        sub_hidden_norms.append(float(torch.sqrt((ub[0].detach().float() ** 2).mean()).item()))
                    y_n = RSIPhase2SuperfusedPreNormFn.apply(
                        self.w_queries[si], self.gamma_keys[si], ub[0], ub[1], None,
                        self.sub_gammas[si], torch.bfloat16, 1e-6, 1e-6,
                    )
                else:
                    kn = eager_rmsnorm_ref(bl_st, self.gamma_keys[si].view(1, 1, D), 1e-6)
                    lg = torch.einsum("d,nmd->nm", self.w_queries[si].float(), kn.float())
                    h = torch.einsum("nm,nmd->md", torch.softmax(lg, dim=0), bl_st.float()).to(torch.bfloat16)
                    if collect_diagnostics:
                        sub_hidden_norms.append(float(torch.sqrt((h.detach().float() ** 2).mean()).item()))
                    y_n = eager_rmsnorm_ref(h, self.sub_gammas[si].view(1, D), 1e-6)
                f_out = self._run_sublayer_compute(si, y_n, B, T)
                if collect_diagnostics and f_out.requires_grad:
                    f_out.retain_grad()
                    sub_outputs_for_grad.append(f_out)
                sources.append(f_out)

            bl_ro = torch.stack(sources, dim=0)
            ub_ro = RSIPhase1UnbindInterBlockFn.apply(self.wq_readout, self.gk_readout, bl_ro, 1e-6, True)
            if collect_diagnostics:
                sub_hidden_norms.append(float(torch.sqrt((ub_ro[0].detach().float() ** 2).mean()).item()))
            y_fin = RSIPhase2SuperfusedPreNormFn.apply(
                self.wq_readout[0], self.gk_readout[0], ub_ro[0], ub_ro[1], None,
                self.final_gamma, torch.bfloat16, 1e-6, 1e-6,
            )

        elif self.topology == "block_attnres_n8":
            blocks_list: List[torch.Tensor] = [b0]
            S_b = self.sub_per_block
            hp1 = torch.ones((1,), device=b0.device, dtype=torch.bool)
            hp0 = torch.zeros((1,), device=b0.device, dtype=torch.bool)

            for bi in range(self.n_blocks):
                bl_st = torch.stack(blocks_list, dim=0)
                if self.kernel_backend == "rsi_superfused" and self.use_key_rmsnorm:
                    ub = RSIPhase1UnbindInterBlockFn.apply(self.w_queries[bi], self.gamma_keys[bi], bl_st, 1e-6, True)
                    p_cur: Optional[torch.Tensor] = None
                    for si_b in range(S_b):
                        g_si = bi * S_b + si_b
                        if collect_diagnostics:
                            with torch.no_grad():
                                h_diag = ub[si_b].detach() if p_cur is None else (
                                    RSIPhase2SuperfusedPreNormFn.apply(
                                        self.w_queries[bi, si_b].detach(), self.gamma_keys[bi, si_b].detach(),
                                        ub[si_b].detach(), ub[S_b + si_b].detach(), p_cur.detach(), None, torch.bfloat16, 1e-6, 1e-6,
                                    )
                                )
                                sub_hidden_norms.append(float(torch.sqrt((h_diag.float() ** 2).mean()).item()))
                        y_n = RSIPhase2SuperfusedPreNormFn.apply(
                            self.w_queries[bi, si_b], self.gamma_keys[bi, si_b],
                            ub[si_b], ub[S_b + si_b], p_cur, self.sub_gammas[g_si],
                            torch.bfloat16, 1e-6, 1e-6,
                        )
                        f_out = self._run_sublayer_compute(g_si, y_n, B, T)
                        if collect_diagnostics and f_out.requires_grad:
                            f_out.retain_grad()
                            sub_outputs_for_grad.append(f_out)
                        p_cur = f_out if p_cur is None else (p_cur + f_out)
                    assert p_cur is not None
                    blocks_list.append(p_cur)

                elif self.kernel_backend == "hydra2p_atomic":
                    p_cur = None
                    for si_b in range(S_b):
                        g_si = bi * S_b + si_b
                        p_in = bl_st[0:1] if p_cur is None else p_cur.unsqueeze(0)
                        h = Hydra2PCommunityAttnResFn.apply(
                            self.w_queries[bi, si_b : si_b + 1], self.gamma_keys[bi, si_b : si_b + 1],
                            bl_st, p_in, hp0 if p_cur is None else hp1, 1e-6,
                        )[0]
                        y_n = FusedSublayerRMSNormFn.apply(h, self.sub_gammas[g_si], 1e-6)
                        f_out = self._run_sublayer_compute(g_si, y_n, B, T)
                        p_cur = f_out if p_cur is None else (p_cur + f_out)
                    assert p_cur is not None
                    blocks_list.append(p_cur)

                else:
                    # Eager Fig. 2 OR `no_key_rmsnorm` ablation (`Gen 3`)
                    p_cur = None
                    for si_b in range(S_b):
                        g_si = bi * S_b + si_b
                        src = bl_st if p_cur is None else torch.cat([bl_st, p_cur.unsqueeze(0)], dim=0)
                        keys_t = (
                            eager_rmsnorm_ref(src, self.gamma_keys[bi, si_b].view(1, 1, D), 1e-6)
                            if self.use_key_rmsnorm
                            else src.float()
                        )
                        lg = torch.einsum("d,nmd->nm", self.w_queries[bi, si_b].float(), keys_t.float())
                        h = torch.einsum("nm,nmd->md", torch.softmax(lg, dim=0), src.float()).to(torch.bfloat16)
                        y_n = FusedSublayerRMSNormFn.apply(h, self.sub_gammas[g_si], 1e-6)
                        f_out = self._run_sublayer_compute(g_si, y_n, B, T)
                        p_cur = f_out if p_cur is None else (p_cur + f_out)
                    assert p_cur is not None
                    blocks_list.append(p_cur)

            bl_ro = torch.stack(blocks_list, dim=0)
            if self.kernel_backend == "rsi_superfused" and self.use_key_rmsnorm:
                ub_ro = RSIPhase1UnbindInterBlockFn.apply(self.wq_readout, self.gk_readout, bl_ro, 1e-6, True)
                if collect_diagnostics:
                    sub_hidden_norms.append(float(torch.sqrt((ub_ro[0].detach().float() ** 2).mean()).item()))
                y_fin = RSIPhase2SuperfusedPreNormFn.apply(
                    self.wq_readout[0], self.gk_readout[0], ub_ro[0], ub_ro[1], None,
                    self.final_gamma, torch.bfloat16, 1e-6, 1e-6,
                )
            else:
                keys_ro = eager_rmsnorm_ref(bl_ro, self.gk_readout.view(1, 1, D), 1e-6) if self.use_key_rmsnorm else bl_ro.float()
                lg_ro = torch.einsum("d,nmd->nm", self.wq_readout[0].float(), keys_ro.float())
                h_ro = torch.einsum("nm,nmd->md", torch.softmax(lg_ro, dim=0), bl_ro.float()).to(torch.bfloat16)
                y_fin = FusedSublayerRMSNormFn.apply(h_ro, self.final_gamma, 1e-6)
        else:
            raise ValueError(f"Unknown topology: {self.topology}")

        logits = self.lm_head(y_fin.to(self.lm_head.weight.dtype)).view(B, T, self.vocab_size)
        return logits, {
            "sub_hidden_norms": sub_hidden_norms,
            "sub_outputs_for_grad": sub_outputs_for_grad,
        }


@torch.no_grad()
def evaluate_lm_split_ddp(
    model: Deep24LayerResidualLM,
    seqs_cpu: torch.Tensor,
    mask_cpu: torch.Tensor,
    tok_bytes_gpu: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, float]:
    model.eval()
    n_tot = seqs_cpu.shape[0]
    per_rank = n_tot // world_size
    s_idx = rank * per_rank
    e_idx = s_idx + per_rank

    nll_sum = 0.0
    bytes_sum = 0.0
    corr = [0, 0, 0, 0]
    cnt = [0, 0, 0, 0]

    for b0 in range(s_idx, e_idx, batch_size):
        b1 = min(e_idx, b0 + batch_size)
        batch = seqs_cpu[b0:b1].to(device, non_blocking=True)
        bmask = mask_cpu[b0:b1].to(device, non_blocking=True)
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        logits, _ = model(inp, collect_diagnostics=False)
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum")
        nll_sum += float(ce.item())
        bytes_sum += float(tok_bytes_gpu[tgt].sum().item())

        preds = logits.argmax(dim=-1)
        for stype in (1, 2, 3):
            sel = bmask == stype
            c_i = int(sel.sum().item())
            if c_i > 0:
                cnt[stype] += c_i
                corr[stype] += int((preds[sel] == tgt[sel]).sum().item())

    stats = torch.tensor(
        [nll_sum, bytes_sum, corr[1], cnt[1], corr[2], cnt[2], corr[3], cnt[3]],
        device=device, dtype=torch.float64,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    tot_nll, tot_bytes = float(stats[0].item()), float(stats[1].item())
    bpb = (tot_nll / math.log(2.0)) / max(tot_bytes, 1.0)
    sh_acc = float(stats[2].item()) / max(float(stats[3].item()), 1.0)
    md_acc = float(stats[4].item()) / max(float(stats[5].item()), 1.0)
    dp_acc = float(stats[6].item()) / max(float(stats[7].item()), 1.0)
    all_trig_c = float(stats[2].item() + stats[4].item() + stats[6].item())
    all_trig_n = max(float(stats[3].item() + stats[5].item() + stats[7].item()), 1.0)
    model.train()
    return {
        "bpb": round(bpb, 5),
        "shallow_acc": round(sh_acc, 4),
        "mid_acc": round(md_acc, 4),
        "deep_acc": round(dp_acc, 4),
        "trigger_acc": round(all_trig_c / all_trig_n, 4),
    }


def collect_dilution_diagnostics_ddp(
    raw_model: Deep24LayerResidualLM,
    seqs_cpu: torch.Tensor,
    rank: int,
    device: torch.device,
) -> Dict[str, float]:
    """Measures Moonshot `arXiv:2603.15031` Fig. 5(b) hidden-norm depth growth ratio
    `||h_final||_rms / ||h_1||_rms` and Fig. 5(c) cross-depth sublayer gradient norm CV."""
    raw_model.zero_grad(set_to_none=True)
    batch = seqs_cpu[rank * 8 : (rank + 1) * 8].to(device)
    logits, diag = raw_model(batch[:, :-1], collect_diagnostics=True)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
    loss.backward()

    h_norms = diag["sub_hidden_norms"]
    norm_growth = h_norms[-1] / max(h_norms[0], 1e-8) if len(h_norms) >= 2 else 1.0
    g_norms = [
        float(torch.linalg.norm(t.grad.detach().float()).item())
        for t in diag["sub_outputs_for_grad"]
        if t.grad is not None
    ]
    g_mean = sum(g_norms) / max(len(g_norms), 1)
    g_std = math.sqrt(sum((x - g_mean) ** 2 for x in g_norms) / max(len(g_norms), 1))
    grad_cv = g_std / max(g_mean, 1e-12)

    q_rms = 0.0
    if hasattr(raw_model, "w_queries"):
        q_rms = float(torch.sqrt((raw_model.w_queries.detach().float() ** 2).mean()).item())

    vec = torch.tensor([norm_growth, grad_cv, q_rms], device=device, dtype=torch.float64)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    vec /= float(dist.get_world_size())
    raw_model.zero_grad(set_to_none=True)
    return {
        "hidden_norm_depth_growth_ratio": round(float(vec[0].item()), 4),
        "sublayer_grad_norm_cv": round(float(vec[1].item()), 4),
        "learned_query_rms": round(float(vec[2].item()), 5),
    }


def run_suite_c_valbpb_and_topology_law(
    rank: int, world_size: int, device: torch.device, mode: str, out_dir: Path
) -> Dict[str, Any]:
    enforce_strict_ieee_precision()
    data = build_multidepth_compositional_splits(vocab_size=1024, seq_len=192)
    vocab_size = int(data["vocab_size"])
    tok_bytes_gpu = data["tok_bytes"].to(device)
    train_seqs, train_mask = data["train"]
    val_seqs, val_mask = data["val"]
    test_seqs, test_mask = data["test"]

    budget_s = 6.0 if mode == "smoke" else 15.0
    batch_per_gpu = 16
    lr = 2.2e-3

    def _build_and_train(
        *,
        seed: int,
        topology: str,
        kernel_backend: str,
        zero_query_init: bool,
        use_key_rmsnorm: bool,
        d_model_c: int = 256,
        n_heads_c: int = 4,
        base_lr: float = 2.2e-3,
        fixed_seconds: Optional[float] = None,
        fixed_steps: Optional[int] = None,
    ) -> Tuple[Deep24LayerResidualLM, Dict[str, Any]]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        raw = Deep24LayerResidualLM(
            vocab_size=vocab_size,
            d_model=d_model_c,
            n_layers=24,
            n_heads=n_heads_c,
            n_blocks=8,
            topology=topology,
            kernel_backend=kernel_backend,
            zero_query_init=zero_query_init,
            use_key_rmsnorm=use_key_rmsnorm,
        ).to(device)
        # Cast base projections to BF16 while keeping embeddings/master norms FP32-compatible
        for m_sub in (raw.qkv_proj, raw.out_proj, raw.gate_up_proj, raw.down_proj):
            m_sub.to(torch.bfloat16)

        base_lrs = (base_lr, base_lr, base_lr * 2.5)

        def _make_opt(mod_ddp: nn.Module) -> torch.optim.Optimizer:
            routing_keys = ("w_queries", "wq_readout", "static_logits", "gate_projs")
            route_p, norm_p, mat_p = [], [], []
            for p_name, p_val in mod_ddp.named_parameters():
                if not p_val.requires_grad:
                    continue
                if any(rk in p_name for rk in routing_keys):
                    route_p.append(p_val)
                elif p_val.ndim <= 1 or "gamma" in p_name or "pos_emb" in p_name:
                    norm_p.append(p_val)
                else:
                    mat_p.append(p_val)
            return torch.optim.AdamW(
                [
                    {"params": mat_p, "lr": base_lrs[0], "weight_decay": 0.01},
                    {"params": norm_p, "lr": base_lrs[1], "weight_decay": 0.0},
                    {"params": route_p, "lr": base_lrs[2], "weight_decay": 0.0},
                ],
                betas=(0.9, 0.95),
            )

        ddp = nn.parallel.DistributedDataParallel(raw, device_ids=[rank], broadcast_buffers=False)
        opt = _make_opt(ddp)

        n_train = train_seqs.shape[0]
        # Warmup 2 untimed steps so kernel/DDP buckets initialize cleanly before wall-clock timer!
        for w_i in range(2):
            off = ((w_i * world_size + rank) * batch_per_gpu) % (n_train - batch_per_gpu)
            b_w = train_seqs[off : off + batch_per_gpu].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            lg_w, _ = ddp(b_w[:, :-1])
            ls_w = F.cross_entropy(lg_w.float().reshape(-1, vocab_size), b_w[:, 1:].reshape(-1))
            ls_w.backward()
            opt.step()

        # Re-seed exact identical initial model weights & optimizer state right before timed/matched run!
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        raw_reset = Deep24LayerResidualLM(
            vocab_size=vocab_size,
            d_model=d_model_c,
            n_layers=24,
            n_heads=n_heads_c,
            n_blocks=8,
            topology=topology,
            kernel_backend=kernel_backend,
            zero_query_init=zero_query_init,
            use_key_rmsnorm=use_key_rmsnorm,
        ).to(device)
        for m_sub in (raw_reset.qkv_proj, raw_reset.out_proj, raw_reset.gate_up_proj, raw_reset.down_proj):
            m_sub.to(torch.bfloat16)
        raw.load_state_dict(raw_reset.state_dict())
        del raw_reset
        opt = _make_opt(ddp)

        torch.cuda.synchronize(device)
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        t_start = time.perf_counter()
        steps_done = 0

        while True:
            if fixed_steps is not None and steps_done >= fixed_steps:
                break
            if fixed_seconds is not None:
                # Synchronized rank-0 stop flag + PROGRAM.md Sec. 5 Rule #4 wall-clock cosine schedule!
                el_now = time.perf_counter() - t_start
                sync_vec = torch.tensor(
                    [1.0 if el_now >= fixed_seconds else 0.0, min(max(el_now / fixed_seconds, 0.0), 1.0)],
                    device=device, dtype=torch.float32,
                )
                dist.broadcast(sync_vec, src=0)
                if float(sync_vec[0].item()) >= 0.5 and steps_done >= 4:
                    break
                u_prog = float(sync_vec[1].item())
                if u_prog < 0.08:
                    lr_mult = (u_prog + 0.01) / 0.09
                else:
                    lr_mult = 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * (u_prog - 0.08) / 0.92))
                for pg, blr in zip(opt.param_groups, base_lrs):
                    pg["lr"] = blr * lr_mult

            g_step = torch.Generator()
            g_step.manual_seed(seed * 10000 + steps_done * 37 + rank)
            idx_sel = torch.randint(0, n_train, (batch_per_gpu,), generator=g_step)
            batch = train_seqs[idx_sel].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits, _ = ddp(batch[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            opt.step()
            steps_done += 1

        torch.cuda.synchronize(device)
        dist.barrier()
        wall_s = time.perf_counter() - t_start
        pk_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        val_m = evaluate_lm_split_ddp(raw, val_seqs, val_mask, tok_bytes_gpu, rank, world_size, device)
        return raw, {
            "steps_completed": steps_done,
            "wall_clock_s": round(wall_s, 3),
            "steps_per_sec": round(steps_done / max(wall_s, 1e-6), 2),
            "peak_alloc_mib": round(pk_mib, 2),
            "val_bpb": val_m["bpb"],
            "val_shallow_acc": val_m["shallow_acc"],
            "val_mid_acc": val_m["mid_acc"],
            "val_deep_acc": val_m["deep_acc"],
            "val_trigger_acc": val_m["trigger_acc"],
        }

    # ---------------- Part 1: 5-Gen Fixed-Wall-Clock `val_bpb` RSI Loop ----------------
    generations_spec = [
        (0, "gen0_moonshot_fig2_eager", "block_attnres_n8", "eager_fig2", True, True, "Official Moonshot Fig. 2 eager stack + RMSNorm + einsum"),
        (1, "gen1_hydra2p_community_atomic", "block_attnres_n8", "hydra2p_atomic", True, True, "Community Hydra-2P Triton Two-Phase with (N*M)-way atomic backward"),
        (2, "gen2_rsi_superfused_two_phase", "block_attnres_n8", "rsi_superfused", True, True, "Round-6 RSI 0-spill Unbind Phase-1 + Superfused Phase-2 Pre-RMSNorm (w_l=0 + key RMSNorm)"),
        (3, "gen3_nonzero_query_no_key_rmsnorm", "block_attnres_n8", "eager_fig2", False, False, "Violates Moonshot Table 4 invariants (non-zero w_l~N(0,1.5) init + no key RMSNorm)"),
        (4, "gen4_unbatched_full_attnres_n48", "full_attnres_n48", "rsi_superfused", True, True, "48-source per-sublayer Full AttnRes (O(L) depth reads/sublayer vs Block AttnRes N=8)"),
    ]

    gen_records: List[Dict[str, Any]] = []
    incumbent_bpb = float("inf")
    incumbent_gen_id = 0
    gen0_state: Optional[Dict[str, torch.Tensor]] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_cfg: Optional[Tuple[str, str, bool, bool]] = None

    d_part1, h_part1, lr_part1 = 512, 8, 1.8e-3
    for gid, gname, gtopo, gback, gzinit, gkrmsn, gnote in generations_spec:
        raw_g, met_g = _build_and_train(
            seed=6100,
            topology=gtopo,
            kernel_backend=gback,
            zero_query_init=gzinit,
            use_key_rmsnorm=gkrmsn,
            d_model_c=d_part1,
            n_heads_c=h_part1,
            base_lr=lr_part1,
            fixed_seconds=budget_s,
        )
        if gid == 0:
            decision = "BASELINE"
            incumbent_bpb = met_g["val_bpb"]
            incumbent_gen_id = gid
            gen0_state = {k: v.detach().cpu() for k, v in raw_g.state_dict().items()}
            best_state = gen0_state
            best_cfg = (gtopo, gback, gzinit, gkrmsn)
        elif met_g["val_bpb"] <= incumbent_bpb - 0.005:
            decision = "KEEP"
            incumbent_bpb = met_g["val_bpb"]
            incumbent_gen_id = gid
            best_state = {k: v.detach().cpu() for k, v in raw_g.state_dict().items()}
            best_cfg = (gtopo, gback, gzinit, gkrmsn)
        else:
            decision = "REVERT_VAL_BPB"

        rec = {
            "gen_id": gid,
            "candidate_id": gname,
            "topology": gtopo,
            "kernel_backend": gback,
            "zero_query_init": gzinit,
            "use_key_rmsnorm": gkrmsn,
            "budget_s": budget_s,
            **met_g,
            "incumbent_val_bpb": round(incumbent_bpb, 5),
            "decision": decision,
            "note": gnote,
        }
        gen_records.append(rec)
        del raw_g
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    # Single-touch `test_bpb` evaluated strictly ONCE on Gen 0 vs winning incumbent!
    assert gen0_state is not None and best_state is not None and best_cfg is not None
    eval_g0 = Deep24LayerResidualLM(
        vocab_size=vocab_size, d_model=d_part1, n_heads=h_part1,
        topology="block_attnres_n8", kernel_backend="eager_fig2",
    ).to(device)
    for m_sub in (eval_g0.qkv_proj, eval_g0.out_proj, eval_g0.gate_up_proj, eval_g0.down_proj):
        m_sub.to(torch.bfloat16)
    eval_g0.load_state_dict({k: v.to(device) for k, v in gen0_state.items()})
    test_g0 = evaluate_lm_split_ddp(eval_g0, test_seqs, test_mask, tok_bytes_gpu, rank, world_size, device)
    del eval_g0

    eval_best = Deep24LayerResidualLM(
        vocab_size=vocab_size, d_model=d_part1, n_heads=h_part1,
        topology=best_cfg[0], kernel_backend=best_cfg[1],
        zero_query_init=best_cfg[2], use_key_rmsnorm=best_cfg[3],
    ).to(device)
    for m_sub in (eval_best.qkv_proj, eval_best.out_proj, eval_best.gate_up_proj, eval_best.down_proj):
        m_sub.to(torch.bfloat16)
    eval_best.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_best = evaluate_lm_split_ddp(eval_best, test_seqs, test_mask, tok_bytes_gpu, rank, world_size, device)
    del eval_best
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- Part 2: 5-Arm Matched-Step Deep-24-Layer Residual-Topology Law ----------------
    matched_steps = 100 if mode == "smoke" else 240
    topology_arms = [
        ("R0_standard_prenorm", "standard_prenorm", "rsi_superfused"),
        ("R1_denseformer_static", "denseformer_static", "rsi_superfused"),
        ("R2_highway_sigmoid_gate", "highway_sigmoid_gate", "rsi_superfused"),
        ("R3_block_attnres_n8", "block_attnres_n8", "rsi_superfused"),
        ("R4_full_attnres_n48", "full_attnres_n48", "rsi_superfused"),
    ]
    topology_results: Dict[str, Any] = {}
    for arm_id, topo_name, kback in topology_arms:
        raw_arm, met_arm = _build_and_train(
            seed=6200,
            topology=topo_name,
            kernel_backend=kback,
            zero_query_init=True,
            use_key_rmsnorm=True,
            fixed_steps=matched_steps,
        )
        test_arm = evaluate_lm_split_ddp(raw_arm, test_seqs, test_mask, tok_bytes_gpu, rank, world_size, device)
        dil_diag = collect_dilution_diagnostics_ddp(raw_arm, val_seqs, rank, device)
        n_params = sum(p.numel() for p in raw_arm.parameters())
        topology_results[arm_id] = {
            "topology": topo_name,
            "num_parameters": n_params,
            **met_arm,
            "test_bpb": test_arm["bpb"],
            "test_shallow_acc": test_arm["shallow_acc"],
            "test_mid_acc": test_arm["mid_acc"],
            "test_deep_acc": test_arm["deep_acc"],
            "test_trigger_acc": test_arm["trigger_acc"],
            **dil_diag,
        }
        del raw_arm
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    if rank != 0:
        return {}

    valbpb_tsv = out_dir / "valbpb_autoresearch_ledger.tsv"
    valbpb_tsv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "gen_id", "candidate_id", "topology", "kernel_backend", "budget_s",
        "steps_completed", "steps_per_sec", "val_bpb", "val_trigger_acc",
        "incumbent_val_bpb", "peak_alloc_mib", "decision", "eval_harness_sha256",
    ]
    with valbpb_tsv.open("w", encoding="utf-8", newline="\n") as fh:
        wr = csv.writer(fh, delimiter="\t", lineterminator="\n")
        wr.writerow(fields)
        for r in gen_records:
            wr.writerow([r[k] for k in fields[:-1]] + [EXPECTED_HARNESS_SHA256])

    g0_rec = gen_records[0]
    win_rec = gen_records[incumbent_gen_id]
    return {
        "budget_seconds": budget_s,
        "valbpb_ledger_tsv": valbpb_tsv.name,
        "generations": gen_records,
        "winning_gen_id": incumbent_gen_id,
        "winning_candidate_id": win_rec["candidate_id"],
        "step_multiplier_vs_gen0": round(win_rec["steps_completed"] / max(float(g0_rec["steps_completed"]), 1.0), 3),
        "val_bpb_improvement_vs_gen0": round(g0_rec["val_bpb"] - win_rec["val_bpb"], 5),
        "single_touch_test_audit": {
            "gen0_test_bpb": test_g0["bpb"],
            "gen0_test_trigger_acc": test_g0["trigger_acc"],
            "winner_test_bpb": test_best["bpb"],
            "winner_test_trigger_acc": test_best["trigger_acc"],
            "test_bpb_improvement": round(test_g0["bpb"] - test_best["bpb"], 5),
            "single_touch_enforced": True,
        },
        "matched_step_residual_topology_law": {
            "matched_steps": matched_steps,
            "arms": topology_results,
        },
    }


# =============================================================================
# Main Orchestrator (`torchrun --nproc_per_node=8`)
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="E132 Moonshot Kimi Attention Residuals Round-6 RSI Benchmark")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--json-out", type=str, required=True)
    args = parser.parse_args()

    enforce_strict_ieee_precision()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    out_dir = Path(args.out_dir).resolve()
    json_out = Path(args.json_out).resolve()
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_out.parent.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    t_wall0 = time.perf_counter()
    suite0_res: Dict[str, Any] = {}
    suite_a_res: Dict[str, Any] = {}

    if rank == 0:
        print(f"[E132][Rank 0] Starting Suite 0 (Harness + Red-Team + Hydra-2P & Spill Forensics + Parity + Richardson)...", flush=True)
        suite0_res = run_suite0_audit(device)
        print(
            f"[E132][Rank 0] Suite 0 done: red_team={suite0_res['red_team_rejected_count']}/{suite0_res['red_team_total']}, "
            f"spills={suite0_res['rsi_cubin_spill_audit']['total_cubin_spills']}, "
            f"parity={suite0_res['all_parity_passed']}, richardson={suite0_res['all_richardson_passed']}",
            flush=True,
        )
        print(f"[E132][Rank 0] Starting Suite A (Track 1 Batched Two-Phase & Track 2 Multi-Block Sequential Stack)...", flush=True)
        suite_a_res = run_suite_a_ratchets(device, args.mode, out_dir)
        print(
            f"[E132][Rank 0] Suite A done: T1={suite_a_res['track1']['final_speedup_vs_baseline']}x, "
            f"T2={suite_a_res['track2']['final_speedup_vs_baseline']}x",
            flush=True,
        )
    dist.barrier()

    if rank == 0:
        print(f"[E132][8-GPU] Starting Suite B (Interleaved 1F1B P=8, V=4 Cross-Stage Block Caching + Gate G9)...", flush=True)
    suite_b_res = run_suite_b_pipeline_parallel_caching(rank, world_size, device, args.mode)
    dist.barrier()
    if rank == 0:
        sp_b = suite_b_res["speedups_and_reductions"]
        print(
            f"[E132][8-GPU] Suite B done: speedup_vs_full={sp_b['speedup_p3_vs_p0_full_attnres']}x, "
            f"speedup_vs_eager_block={sp_b['speedup_p3_vs_p1_eager_block_naive']}x, "
            f"cache_gain={sp_b['speedup_p3_vs_p2_pure_cross_stage_cache_gain']}x, "
            f"comm_red={sp_b['comm_reduction_p3_vs_p0_full_attnres']}x",
            flush=True,
        )
        print(f"[E132][8-GPU] Starting Suite C ({args.mode} Fixed-Wall-Clock val_bpb RSI Loop + 5-Arm Deep-24-Layer Law)...", flush=True)

    suite_c_res = run_suite_c_valbpb_and_topology_law(rank, world_size, device, args.mode, out_dir)
    dist.barrier()
    total_elapsed = round(time.perf_counter() - t_wall0, 2)

    if rank == 0:
        clk, pwr = query_sm_telemetry(0)
        payload = {
            "experiment_id": "E132",
            "title": "Round-6 RSI Study: Moonshot Kimi Attention Residuals (Two-Phase Superfused Block AttnRes & Interleaved 1F1B Cross-Stage Caching)",
            "paper_arxiv": "2603.15031",
            "official_repo": "MoonshotAI/Attention-Residuals",
            "community_triton_ref": "manishklach/attnres-kernel-lab (Hydra-2P)",
            "mode": args.mode,
            "world_size": world_size,
            "eval_harness_sha256": eval_harness_sha256(),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_wall_clock_s": total_elapsed,
            "sm_clock_mhz_end": clk,
            "power_w_end": pwr,
            "environment": environment_metadata(),
            "suite0": suite0_res,
            "suite_a": suite_a_res,
            "suite_b": suite_b_res,
            "suite_c": suite_c_res,
        }
        json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[E132][Rank 0] Completed {args.mode} in {total_elapsed:.2f}s -> {json_out}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

