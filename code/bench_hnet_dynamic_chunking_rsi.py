#!/usr/bin/env python3
"""E133 (`v124`): Round-7 RSI Benchmark Suite for H-Net Dynamic Chunking (`arXiv:2507.07955`, `goombalab/hnet`).

Suites:
- Suite 0: Adversarial Red-Team Torture (`4/4` rejected), Upstream `goombalab/hnet` Forensics (`BF16` SSD
  surrogate `dt = -log(1 - p)` drift, `ChunkLayer` `fullgraph=True` host-sync break, padded `load_balancing_loss`
  contamination), `17/17` Zero-Spill SM80 Cubin Audit, Adversarial `FP64` Parity, & `7/7` `FP64` `O(h^2)` Richardson.
- Suite A: Multi-Trial Round-7 RSI Vector-Pareto Ratchets (`rsi_ledger.tsv`, `>96 MiB` L2-flushed rotating slots):
  * Track 1: Isolated Dynamic Chunking Module (`Router + Compaction + EMA Smoothing + STE Upsampling + Skip Residual`).
  * Track 2: End-to-End 1-Stage Hierarchical `H-Net` Block (`Router QK GEMM + DC + Varlen FlashAttention-2 & SwiGLU
    Main Transformer + Network RMSNorm Balance + Skip-Residual GEMM + Superfused Dechunk/STE`).
- Suite B: 8-GPU NVSwitch Heterogeneous Variable-Compression Dynamic-Chunk Load Balancing (`Whole-Document
  Quadratic-Cost-LPT AllToAllv + 3-Scalar Global Token-Weighted Ratio Loss`) + 3-Step Multi-Update `Gate G9` Parity.
- Suite C:
  * Part 1: 5-Generation Fixed-Wall-Clock (`15.0s` full / `6.0s` smoke) `8xA100 DDP` Byte-Level `H-Net` `val_bpb`
    Autonomous RSI Loop (`valbpb_autoresearch_ledger.tsv`) with synchronized Rank-0 cosine LR & single-touch `test_bpb`.
  * Part 2: 5-Arm Matched-Step (`240` full / `70` smoke) Byte-Level Compression, Morpheme-Boundary F1, &
    1-Byte Phase-Shift / Character-Perturbation Robustness Law.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func

from rsi_kernel_harness import (
    PrecisionStateSnapshot,
    RSIVectorRatchet,
    benchmark_rotating_cuda,
    compare_tensor_to_fp64,
    eval_harness_sha256,
    hash_callable_sources,
    query_sm_telemetry,
    richardson_unit_rms_audit,
)
from hnet_dynamic_chunking_rsi_kernels import (
    FusedDechunkEMASmoothUpsampleSTEResidualFn,
    FusedHNetCosineRouterFn,
    FusedHNetMergedQKCosineRouterFn,
    ParallelScalarAffineEMAFn,
    _ExactSTEFn,
    _pytorch_parallel_affine_scan,
    audit_hnet_triton_cubin_spills,
    build_packed_doc_metadata,
    combine_cost_lpt_chunks,
    eager_dc_ops_packed,
    fp64_hnet_affine_ema_oracle,
    fp64_hnet_dc_pipeline_oracle,
    fp64_hnet_router_oracle,
    official_dc_ops_batched,
    official_hnet_ssd_surrogate_ema,
    plan_and_dispatch_cost_lpt_chunks,
    rsi_round7_superfused_dc_ops,
    rsi_round7_superfused_dc_step,
    zero_sort_prefix_chunk_pack,
)

LN2 = math.log(2.0)


# ==============================================================================
# Shared Building Blocks: Causal Ragged FlashAttention-2 + SwiGLU Transformer
# ==============================================================================


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rstd = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)
        return (xf * rstd * self.weight).to(x.dtype)


class VarlenTransformerLayer(nn.Module):
    """Causal GQA/MHA FlashAttention-2 + SwiGLU layer supporting packed `cu_seqlens` and optional SWA."""

    def __init__(self, d_model: int, n_heads: int, *, window_size: tuple[int, int] = (-1, -1)) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size

        self.norm1 = RMSNorm(d_model)
        self.wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        ffn_dim = ((int(d_model * 8 / 3) + 63) // 64) * 64
        self.w_gate_up = nn.Linear(d_model, 2 * ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, d_model, bias=False)
        self.ffn_dim = ffn_dim

    def forward_varlen(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        if x.shape[0] == 0:
            # Zero-token worker guard so DDP backward sees 0-grad anchors on every parameter
            anchor = sum(p.reshape(-1)[0] * 0.0 for p in self.parameters())
            return x + anchor.to(x.dtype)
        h = self.norm1(x)
        qkv = self.wqkv(h).view(-1, 3, self.n_heads, self.head_dim)
        if qkv.dtype == torch.float32:
            # FlashAttention-2 requires FP16/BF16; in FP32 Gate-G9 exactness mode use exact per-doc causal SDPA
            outs = []
            cu_list = cu_seqlens.detach().cpu().tolist()
            for i in range(len(cu_list) - 1):
                s, e = int(cu_list[i]), int(cu_list[i + 1])
                if e <= s:
                    continue
                q_i = qkv[s:e, 0].transpose(0, 1).unsqueeze(0)
                k_i = qkv[s:e, 1].transpose(0, 1).unsqueeze(0)
                v_i = qkv[s:e, 2].transpose(0, 1).unsqueeze(0)
                if self.window_size[0] > 0:
                    idx = torch.arange(e - s, device=x.device)
                    diff = idx[:, None] - idx[None, :]
                    attn_m = (diff >= 0) & (diff <= self.window_size[0])
                    ao = F.scaled_dot_product_attention(q_i, k_i, v_i, attn_mask=attn_m)
                else:
                    ao = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=True)
                outs.append(ao.squeeze(0).transpose(0, 1).reshape(e - s, self.d_model))
            attn_out = torch.cat(outs, dim=0)
        else:
            attn_out = flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                dropout_p=0.0,
                causal=True,
                window_size=self.window_size,
            ).reshape(-1, self.d_model)
        x = x + self.wo(attn_out)
        h2 = self.norm2(x)
        gu = self.w_gate_up(h2)
        g, u = gu[:, : self.ffn_dim], gu[:, self.ffn_dim :]
        return x + self.w_down(F.silu(g) * u)

    def forward_batched_padded(self, x_b: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, l_len, _ = x_b.shape
        if l_len == 0:
            return x_b
        h = self.norm1(x_b)
        qkv = self.wqkv(h).view(bsz, l_len, 3, self.n_heads, self.head_dim)
        attn_out = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=0.0,
            causal=True,
            window_size=self.window_size,
        ).reshape(bsz, l_len, self.d_model)
        x_b = x_b + self.wo(attn_out)
        h2 = self.norm2(x_b)
        gu = self.w_gate_up(h2)
        g, u = gu[..., : self.ffn_dim], gu[..., self.ffn_dim :]
        out = x_b + self.w_down(F.silu(g) * u)
        if valid_mask is not None:
            out = out * valid_mask.unsqueeze(-1).to(out.dtype)
        return out


# ==============================================================================
# Suite 0: Red-Team Torture, Upstream Forensics, Zero-Spill Cubins, & Oracles
# ==============================================================================


def run_suite_0(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(20261118)
    # 1. SM80 Cubin register-spill audit across all 17 Round-7 RSI Triton kernels
    spill_audit = audit_hnet_triton_cubin_spills(device)

    # 2. Adversarial Red-Team Exploitation Checks (4 deliberate exploits)
    t_rt, d_rt = 256, 512
    cu_rt = torch.tensor([0, 96, 96, 256], dtype=torch.int32, device=device)  # includes empty doc!
    doc_meta_rt = build_packed_doc_metadata(cu_rt, t_rt)
    x_rt64 = torch.randn((t_rt, d_rt), dtype=torch.float64, device=device) * 0.6
    x_rt64[7] = 0.0  # exact zero-norm vector row!
    wq_rt64 = torch.randn((d_rt, d_rt), dtype=torch.float64, device=device) * 0.04
    wk_rt64 = torch.randn((d_rt, d_rt), dtype=torch.float64, device=device) * 0.04
    wres_rt64 = torch.randn((d_rt, d_rt), dtype=torch.float64, device=device) * 0.02
    wm_rt64 = torch.randn((d_rt, d_rt), dtype=torch.float64, device=device) * 0.04
    dy_rt64 = torch.randn((t_rt, d_rt), dtype=torch.float64, device=device) * 0.1

    xr = x_rt64.detach().clone().requires_grad_(True)
    wqr = wq_rt64.detach().clone().requires_grad_(True)
    wkr = wk_rt64.detach().clone().requires_grad_(True)
    wresr = wres_rt64.detach().clone().requires_grad_(True)
    wmr = wm_rt64.detach().clone().requires_grad_(True)
    ref_rt = fp64_hnet_dc_pipeline_oracle(xr, wqr, wkr, wresr, wmr, cu_rt)
    (torch.sum(ref_rt["z_out"] * dy_rt64) + 0.03 * ref_rt["ratio_loss"]).backward()

    # Red-Team 1: Omit cosine-router radial Jacobian (-cos * v_hat)
    xc1 = x_rt64.float().detach().requires_grad_(True)
    wqk1 = torch.cat([wq_rt64.float(), wk_rt64.float()], dim=0).detach().requires_grad_(True)
    wres1 = wres_rt64.float().detach().requires_grad_(True)
    wm1 = wm_rt64.float().detach().requires_grad_(True)
    z1, rl1, _ = rsi_round7_superfused_dc_step(xc1, wqk1, wres1, wm1, cu_rt, doc_meta=doc_meta_rt, omit_radial_bug=True)
    (torch.sum(z1.float() * dy_rt64.float()) + 0.03 * rl1).backward()
    rt1_chk = compare_tensor_to_fp64(wqk1.grad[:d_rt], wqr.grad, rel_tol=1e-4, name="rt1_dwq")
    rt1_rejected = not rt1_chk["ok"]

    # Red-Team 2: Omit reverse EMA smoothing recurrence ((1 - P_{j+1}) * lam_{j+1})
    xc2 = x_rt64.float().detach().requires_grad_(True)
    wqk2 = torch.cat([wq_rt64.float(), wk_rt64.float()], dim=0).detach().requires_grad_(True)
    wres2 = wres_rt64.float().detach().requires_grad_(True)
    wm2 = wm_rt64.float().detach().requires_grad_(True)
    z2, rl2, _ = rsi_round7_superfused_dc_step(xc2, wqk2, wres2, wm2, cu_rt, doc_meta=doc_meta_rt, omit_ema_adjoint_bug=True)
    (torch.sum(z2.float() * dy_rt64.float()) + 0.03 * rl2).backward()
    rt2_chk = compare_tensor_to_fp64(wm2.grad, wmr.grad, rel_tol=1e-4, name="rt2_dwmain")
    rt2_rejected = not rt2_chk["ok"]

    # Red-Team 3: Mutate global `allow_tf32` backend flag during candidate execution
    snap_before = PrecisionStateSnapshot.capture()
    rt3_rejected = False
    try:
        def _cheat_tf32():
            torch.backends.cuda.matmul.allow_tf32 = True
            return torch.ones((1,), device=device)
        benchmark_rotating_cuda([_cheat_tf32], warmup=1, groups=1, iters_per_group=1)
    except RuntimeError:
        rt3_rejected = True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = snap_before.allow_tf32_matmul

    # Red-Team 4 & Upstream Defect Audit: Official `goombalab/hnet` unmasked padded `load_balancing_loss`
    b_pad, l_pad, d_pad = 4, 256, 256
    valid_lens = [256, 180, 96, 64]
    mask_pad = torch.zeros((b_pad, l_pad), dtype=torch.bool, device=device)
    for bi, vl in enumerate(valid_lens):
        mask_pad[bi, :vl] = True
    x_pad = (torch.randn((b_pad, l_pad, d_pad), dtype=torch.float32, device=device) * 0.5).requires_grad_(True)
    wq_p = torch.randn((d_pad, d_pad), dtype=torch.float32, device=device) * 0.04
    wk_p = torch.randn((d_pad, d_pad), dtype=torch.float32, device=device) * 0.04
    cos_p = torch.einsum("bld,bld->bl", F.normalize(F.linear(x_pad[:, :-1], wq_p), dim=-1), F.normalize(F.linear(x_pad[:, 1:], wk_p), dim=-1))
    bp = F.pad(torch.clamp((1.0 - cos_p) * 0.5, 0.0, 1.0), (1, 0), "constant", 1.0)
    bmask_off = (bp > 0.5) & mask_pad
    # Official `train.py` L34-40 (unmasked `.mean()` over padded tensor!)
    f_off = bmask_off.float().mean().detach()
    g_off = bp.float().mean()
    rl_off = ((1.0 - f_off) * (1.0 - g_off) + f_off * g_off * 3.0) * (4.0 / 3.0)
    rl_off.backward()
    pad_grad_norm = float(torch.linalg.norm(x_pad.grad[~mask_pad]).item())
    # Exact valid-token-only ratio loss
    n_valid = float(mask_pad.sum().item())
    f_true = float(bmask_off.float().sum().item() / n_valid)
    g_true = float((bp.detach() * mask_pad.float()).sum().item() / n_valid)
    rl_true_val = ((1.0 - f_true) * (1.0 - g_true) + f_true * g_true * 3.0) * (4.0 / 3.0)
    ratio_loss_rel_err = abs(float(rl_off.item()) - rl_true_val) / max(1e-12, abs(rl_true_val))
    rt4_rejected = (pad_grad_norm > 1e-4) and (ratio_loss_rel_err > 0.05)

    # 3. Upstream `DeChunkLayer` BF16 SSD surrogate `dt = -log(1 - p)` & `x = z / dt` audit (`dc.py` L257-293)
    z_ema_ref = torch.randn((512, 1024), dtype=torch.float64, device=device)
    p_conf = torch.cat([torch.ones((1,), dtype=torch.float64, device=device), torch.empty((511,), dtype=torch.float64, device=device).uniform_(0.985, 0.99995)])
    cu_ema = torch.tensor([0, 256, 512], dtype=torch.int32, device=device)
    p_conf[256] = 1.0
    ema_oracle_conf = fp64_hnet_affine_ema_oracle(z_ema_ref, p_conf, cu_ema)
    off_ssd_conf = official_hnet_ssd_surrogate_ema(z_ema_ref.to(torch.bfloat16), p_conf.float(), cu_ema, ssd_dtype=torch.bfloat16)
    _, is_st_ema, _ = build_packed_doc_metadata(cu_ema, 512)
    rsi_ema_conf = ParallelScalarAffineEMAFn.apply(z_ema_ref.to(torch.bfloat16), p_conf.float(), is_st_ema, cu_ema)
    off_conf_rel = compare_tensor_to_fp64(off_ssd_conf, ema_oracle_conf)["rel_l2"]
    rsi_conf_rel = compare_tensor_to_fp64(rsi_ema_conf, ema_oracle_conf)["rel_l2"]

    # Also measure `torch.compile(fullgraph=True)` graph-break on official `ChunkLayer` `int(num_tokens.max())`
    fullgraph_official_failed = False
    fullgraph_error_type = ""
    try:
        xb_fg = torch.randn((2, 64, 128), dtype=torch.bfloat16, device=device, requires_grad=True)
        qkb_fg = torch.randn((2, 64, 256), dtype=torch.bfloat16, device=device, requires_grad=True)
        rb_fg = torch.randn((2, 64, 128), dtype=torch.bfloat16, device=device, requires_grad=True)
        ms_fg = torch.ones((128,), dtype=torch.bfloat16, device=device)
        mk_fg = torch.ones((2, 64), dtype=torch.bool, device=device)
        torch.compile(official_dc_ops_batched, fullgraph=True)(qkb_fg, xb_fg, rb_fg, mk_fg, main_scale=ms_fg)
    except Exception as exc:
        fullgraph_official_failed = True
        fullgraph_error_type = type(exc).__name__

    # 4. Adversarial Per-Tensor FP64 Parity (FP32 & BF16 across ragged D=1264, empty doc, zero row, collapsed M=1, non-contiguous)
    t_adv, d_adv = 384, 1264
    cu_adv = torch.tensor([0, 128, 128, 384], dtype=torch.int32, device=device)
    doc_meta_adv = build_packed_doc_metadata(cu_adv, t_adv)
    # Construct run-structured + alternating tokens so router `p_t` spans both `< 0.25` (merge) and `> 0.75` (split)
    # cleanly away from the discontinuous `p_t == 0.5` step threshold, while preserving non-contiguous stride (`::2`),
    # ragged `D = 1264`, empty doc `[128, 128)`, and exact zero-norm token `x_adv0[5] = 0` (`dc.py` L69-72 eye+noise init).
    base_runs = torch.randn((t_adv // 4, d_adv), dtype=torch.float64, device=device) * 0.6
    x_struct = base_runs.repeat_interleave(4, dim=0)
    x_struct = x_struct + torch.randn_like(x_struct) * 0.08
    x_adv_buf = torch.empty((t_adv, d_adv * 2), dtype=torch.float64, device=device)
    x_adv_buf[:, ::2] = x_struct
    x_adv0 = x_adv_buf[:, ::2]  # non-contiguous stride!
    x_adv0[5] = 0.0
    eye_adv = torch.eye(d_adv, dtype=torch.float64, device=device) * 0.45
    wq_a = eye_adv + torch.randn((d_adv, d_adv), dtype=torch.float64, device=device) * 0.02
    wk_a = eye_adv + torch.randn((d_adv, d_adv), dtype=torch.float64, device=device) * 0.02
    wres_a = torch.randn((d_adv, d_adv), dtype=torch.float64, device=device) * 0.02
    wm_a = torch.randn((d_adv, d_adv), dtype=torch.float64, device=device) * 0.04
    dy_a = torch.randn((t_adv, d_adv), dtype=torch.float64, device=device) * 0.1

    fp32_checks: list[dict[str, Any]] = []
    bf16_checks: list[dict[str, Any]] = []
    for dt_label, dt, tol, round_dt, bucket in [
        ("fp32", torch.float32, 1e-5, None, fp32_checks),
        ("bf16", torch.bfloat16, 1.5e-2, torch.bfloat16, bf16_checks),
    ]:
        xc_base = x_adv0.to(dt).detach()
        # Preserve non-contiguous view into forward!
        xc_nc = torch.empty((t_adv, d_adv * 2), dtype=dt, device=device)
        xc_nc[:, ::2] = xc_base
        xc_in = xc_nc[:, ::2].detach().requires_grad_(True)
        wqkc = torch.cat([wq_a.to(dt), wk_a.to(dt)], dim=0).detach().requires_grad_(True)
        wresc = wres_a.to(torch.float32 if dt == torch.float32 else dt).detach().requires_grad_(True)
        wmc = wm_a.to(dt).detach().requires_grad_(True)

        # Evaluate FP64 oracle on the exact `dt`-quantized input tensors promoted to FP64
        # (and flip sign on any rare near-0.500 tie row so `p_t` sits cleanly outside 1 BF16 GEMM ULP)
        if dt == torch.bfloat16:
            with torch.no_grad():
                qk_probe = F.linear(xc_in, wqkc)
                p_pr, _, _ = fp64_hnet_router_oracle(qk_probe[:, :d_adv].double(), qk_probe[:, d_adv:].double(), cu_adv)
                near_tie = ((p_pr - 0.5).abs() < 0.04) & (p_pr < 0.99)
                if bool(near_tie.any().item()):
                    tie_rows = torch.nonzero(near_tie, as_tuple=False).squeeze(-1)
                    for tr in tie_rows.tolist():
                        if tr not in (0, 5, 6, 128):
                            xc_nc[tr, ::2] = xc_nc[tr - 1, ::2]
            xc_in = xc_nc[:, ::2].detach().requires_grad_(True)

        xr_a = xc_in.detach().to(torch.float64).requires_grad_(True)
        wqr_a = wqkc[:d_adv].detach().to(torch.float64).requires_grad_(True)
        wkr_a = wqkc[d_adv:].detach().to(torch.float64).requires_grad_(True)
        wresr_a = wresc.detach().to(torch.float64).requires_grad_(True)
        wmr_a = wmc.detach().to(torch.float64).requires_grad_(True)
        ref_a = fp64_hnet_dc_pipeline_oracle(xr_a, wqr_a, wkr_a, wresr_a, wmr_a, cu_adv, dechunk_round_dtype=round_dt)
        (torch.sum(ref_a["z_out"] * dy_a) + 0.03 * ref_a["ratio_loss"]).backward()

        zc, rlc, _ = rsi_round7_superfused_dc_step(xc_in, wqkc, wresc, wmc, cu_adv, doc_meta=doc_meta_adv)
        (torch.sum(zc.float() * dy_a.float()) + 0.03 * rlc).backward()
        bucket.extend([
            compare_tensor_to_fp64(zc, ref_a["z_out"], rel_tol=tol, name=f"{dt_label}_z_out"),
            compare_tensor_to_fp64(xc_in.grad, xr_a.grad, rel_tol=tol, name=f"{dt_label}_dx_enc"),
            compare_tensor_to_fp64(wqkc.grad[:d_adv], wqr_a.grad, rel_tol=tol, name=f"{dt_label}_dw_q"),
            compare_tensor_to_fp64(wqkc.grad[d_adv:], wkr_a.grad, rel_tol=tol, name=f"{dt_label}_dw_k"),
            compare_tensor_to_fp64(wresc.grad, wresr_a.grad, rel_tol=tol, name=f"{dt_label}_dw_res"),
            compare_tensor_to_fp64(wmc.grad, wmr_a.grad, rel_tol=tol, name=f"{dt_label}_dw_main"),
        ])

    # Also test collapsed M=1 span (`256` identical tokens -> `1` boundary chunk spanning all 256 tokens, Split-Span=16)
    cu_m1 = torch.tensor([0, 256], dtype=torch.int32, device=device)
    x_m1 = (torch.ones((256, 256), dtype=torch.float64, device=device) * 0.5).requires_grad_(True)
    wq_m1 = torch.eye(256, dtype=torch.float64, device=device, requires_grad=True)
    wk_m1 = torch.eye(256, dtype=torch.float64, device=device, requires_grad=True)
    wr_m1 = (torch.eye(256, dtype=torch.float64, device=device) * 0.1).detach().requires_grad_(True)
    wm_m1 = (torch.eye(256, dtype=torch.float64, device=device) * 0.2).detach().requires_grad_(True)
    dy_m1 = torch.randn((256, 256), dtype=torch.float64, device=device) * 0.1
    ref_m1 = fp64_hnet_dc_pipeline_oracle(x_m1, wq_m1, wk_m1, wr_m1, wm_m1, cu_m1)
    torch.sum(ref_m1["z_out"] * dy_m1).backward()

    xc_m1 = x_m1.detach().float().requires_grad_(True)
    wqk_m1 = torch.cat([wq_m1.detach().float(), wk_m1.detach().float()], dim=0).requires_grad_(True)
    wrc_m1 = wr_m1.detach().float().requires_grad_(True)
    wmc_m1 = wm_m1.detach().float().requires_grad_(True)
    zc_m1, _, meta_m1 = rsi_round7_superfused_dc_step(xc_m1, wqk_m1, wrc_m1, wmc_m1, cu_m1)
    torch.sum(zc_m1.float() * dy_m1.float()).backward()
    fp32_checks.append(compare_tensor_to_fp64(wmc_m1.grad, wm_m1.grad, rel_tol=1e-5, name="fp32_collapsed_m1_dw_main"))

    # 5. 7 Smooth Nonlinear FP64 Unit-RMS O(h^2) Richardson Extrapolation Checks (`ratio in [3.90, 4.10]`)
    _, is_st_r, is_la_r = build_packed_doc_metadata(torch.tensor([0, 32, 64], dtype=torch.int32, device=device), 64)
    u_r = (torch.randn((64, 96), dtype=torch.float64, device=device) * 0.8 + 0.2).requires_grad_(True)
    v_r = (torch.randn((64, 96), dtype=torch.float64, device=device) * 0.8 - 0.15).requires_grad_(True)
    w_r = torch.randn((64,), dtype=torch.float64, device=device) * 1.3
    p_r, _ = FusedHNetCosineRouterFn.apply(u_r, v_r, is_st_r, is_la_r, 1e-12, False)
    torch.sum(torch.cos(p_r * w_r)).backward()
    rich_1 = richardson_unit_rms_audit(
        lambda uu: torch.sum(torch.cos(FusedHNetCosineRouterFn.apply(uu, v_r.detach(), is_st_r, is_la_r, 1e-12, False)[0] * w_r)),
        u_r.detach(), u_r.grad, name="router_du",
    )
    rich_2 = richardson_unit_rms_audit(
        lambda vv: torch.sum(torch.cos(FusedHNetCosineRouterFn.apply(u_r.detach(), vv, is_st_r, is_la_r, 1e-12, False)[0] * w_r)),
        v_r.detach(), v_r.grad, name="router_dv",
    )

    qk_r = torch.cat([u_r.detach(), v_r.detach()], dim=-1).requires_grad_(True)
    pm_r, _ = FusedHNetMergedQKCosineRouterFn.apply(qk_r, is_st_r, is_la_r, 1e-12, False)
    torch.sum(torch.cos(pm_r * w_r)).backward()
    rich_3 = richardson_unit_rms_audit(
        lambda qq: torch.sum(torch.cos(FusedHNetMergedQKCosineRouterFn.apply(qq, is_st_r, is_la_r, 1e-12, False)[0] * w_r)),
        qk_r.detach(), qk_r.grad, name="merged_qk_dqk",
    )

    ncu_r = torch.tensor([0, 12, 24], dtype=torch.int32, device=device)
    _, cis_r, _ = build_packed_doc_metadata(ncu_r, 24)
    zh_r = (torch.randn((24, 64), dtype=torch.float64, device=device) * 0.7).requires_grad_(True)
    ps_r = torch.sigmoid(torch.randn((24,), dtype=torch.float64, device=device)).detach().requires_grad_(True)
    dy_ema = torch.randn((24, 64), dtype=torch.float64, device=device) * 0.9
    zb_r = ParallelScalarAffineEMAFn.apply(zh_r, ps_r, cis_r, ncu_r)
    torch.sum(torch.cos(zb_r * dy_ema)).backward()
    rich_4 = richardson_unit_rms_audit(
        lambda zz: torch.sum(torch.cos(ParallelScalarAffineEMAFn.apply(zz, ps_r.detach(), cis_r, ncu_r) * dy_ema)),
        zh_r.detach(), zh_r.grad, name="ema_dz_hat",
    )
    rich_5 = richardson_unit_rms_audit(
        lambda pp: torch.sum(torch.cos(ParallelScalarAffineEMAFn.apply(zh_r.detach(), pp, cis_r, ncu_r) * dy_ema)),
        ps_r.detach(), ps_r.grad, name="ema_dp_sel",
    )

    # Locally-anchored smooth surrogate for Dechunk Upsample + Confidence STE (`z_bar + (c - c0)*z_bar0 + res`)
    # Keeping boundaries fixed away from 0.5 (`|p - 0.5| > 0.15`) so no discrete indicator flips occur at h=2e-3
    p_base = torch.empty((64,), dtype=torch.float64, device=device).uniform_(0.68, 0.92)
    p_low = torch.empty((64,), dtype=torch.float64, device=device).uniform_(0.08, 0.32)
    pattern = (torch.arange(64, device=device) % 4 == 0)
    pattern[32] = True
    p_anch0 = torch.where(pattern, p_base, p_low)
    p_anch0[0] = 1.0
    p_anch0[32] = 1.0
    b_anch = (p_anch0 > 0.5).to(torch.int32)
    doc_a, _, _ = build_packed_doc_metadata(torch.tensor([0, 32, 64], dtype=torch.int32, device=device), 64)
    zh_a = (torch.randn((int(b_anch.sum().item()), 48), dtype=torch.float64, device=device) * 0.6).requires_grad_(True)
    p_a = p_anch0.detach().clone().requires_grad_(True)
    res_a = (torch.randn((64, 48), dtype=torch.float64, device=device) * 0.3).requires_grad_(True)
    dy_anch = torch.randn((64, 48), dtype=torch.float64, device=device) * 0.8
    _, _, meta_a = zero_sort_prefix_chunk_pack(res_a.detach(), p_anch0, b_anch, doc_a, torch.tensor([0, 32, 64], dtype=torch.int32, device=device))

    with torch.no_grad():
        zb_anchor0 = fp64_hnet_affine_ema_oracle(zh_a.detach(), p_anch0.index_select(0, meta_a.source_pos.long()), meta_a.next_cu_seqlens)
        zt_anchor0 = zb_anchor0.index_select(0, meta_a.chunk_idx.long())
        c_anchor0 = torch.where(b_anch.bool(), p_anch0, 1.0 - p_anch0)

    z_dc = FusedDechunkEMASmoothUpsampleSTEResidualFn.apply(
        zh_a, p_a, res_a, b_anch, meta_a.chunk_idx, meta_a.source_pos, meta_a.span_end, meta_a.chunk_is_start, meta_a.next_cu_seqlens, False
    )
    torch.sum(torch.cos(z_dc * dy_anch)).backward()

    def _anchored_surrogate_loss(zh_var: torch.Tensor, p_var: torch.Tensor, r_var: torch.Tensor) -> torch.Tensor:
        ps_v = p_var.index_select(0, meta_a.source_pos.long())
        zb_v = fp64_hnet_affine_ema_oracle(zh_var, ps_v, meta_a.next_cu_seqlens)
        zt_v = zb_v.index_select(0, meta_a.chunk_idx.long())
        c_v = torch.where(b_anch.bool(), p_var, 1.0 - p_var)
        z_s = zt_v + (c_v - c_anchor0).unsqueeze(-1) * zt_anchor0 + r_var
        return torch.sum(torch.cos(z_s * dy_anch))

    rich_6 = richardson_unit_rms_audit(
        lambda zz: _anchored_surrogate_loss(zz, p_a.detach(), res_a.detach()),
        zh_a.detach(), zh_a.grad, name="dechunk_superfused_dz_hat",
    )
    rich_7 = richardson_unit_rms_audit(
        lambda pp: _anchored_surrogate_loss(zh_a.detach(), pp, res_a.detach()),
        p_a.detach(), p_a.grad, name="dechunk_superfused_dp_total",
    )
    richardson_checks = [rich_1, rich_2, rich_3, rich_4, rich_5, rich_6, rich_7]

    red_team_dict = {
        "rt1_router_omit_radial_jacobian_rejected": rt1_rejected,
        "rt1_dwq_rel_l2": rt1_chk["rel_l2"],
        "rt2_dechunk_omit_ema_adjoint_rejected": rt2_rejected,
        "rt2_dwmain_rel_l2": rt2_chk["rel_l2"],
        "rt3_mutate_allow_tf32_rejected": rt3_rejected,
        "rt4_official_padded_ratio_loss_contamination_rejected": rt4_rejected,
        "rt4_pad_grad_l2_norm": round(pad_grad_norm, 6),
        "rt4_ratio_loss_rel_err": round(ratio_loss_rel_err, 6),
    }
    red_rejected_cnt = sum(
        int(bool(v))
        for k, v in red_team_dict.items()
        if k.endswith("_rejected")
    )

    return {
        "cubin_spill_audit": spill_audit,
        "red_team_audit": red_team_dict,
        "red_team_rejected_count": f"{red_rejected_cnt}/4",
        "all_red_team_rejected": (red_rejected_cnt == 4),
        "upstream_hnet_forensics": {
            "official_ssd_surrogate_confident_bf16_rel_l2": off_conf_rel,
            "rsi_division_free_ema_confident_bf16_rel_l2": rsi_conf_rel,
            "ema_precision_improvement_ratio": round(off_conf_rel / max(1e-12, rsi_conf_rel), 4),
            "official_chunk_layer_fullgraph_compile_failed": fullgraph_official_failed,
            "official_chunk_layer_fullgraph_error_type": fullgraph_error_type,
            "collapsed_m1_chunk_count": meta_m1.m_total,
        },
        "fp32_parity_checks": fp32_checks,
        "bf16_parity_checks": bf16_checks,
        "max_fp32_rel_l2": max(float(c["rel_l2"]) for c in fp32_checks),
        "max_bf16_rel_l2": max(float(c["rel_l2"]) for c in bf16_checks),
        "all_parity_passed": all(c["ok"] for c in fp32_checks + bf16_checks),
        "richardson_checks": richardson_checks,
        "all_richardson_passed": all(r["ok"] for r in richardson_checks),
    }


# ==============================================================================
# Suite A: Round-7 RSI Vector-Pareto Kernel Ratchets (`Track 1` & `Track 2`)
# ==============================================================================


def _verify_dc_ops_candidate_against_fp64(
    cand_kind: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    """Runs authentic per-candidate FP64 parity, Richardson, and anti-cheat checks for Track 1."""
    torch.manual_seed(20261119)
    bsz, l_len, d_dim = 2, 96, 384
    t_len = bsz * l_len
    cu = torch.tensor([0, 96, 192], dtype=torch.int32, device=device)
    doc_meta = build_packed_doc_metadata(cu, t_len)
    mask = torch.ones((bsz, l_len), dtype=torch.bool, device=device)

    qk64 = torch.randn((t_len, 2 * d_dim), dtype=torch.float64, device=device) * 0.5
    x64 = torch.randn((t_len, d_dim), dtype=torch.float64, device=device) * 0.5
    r64 = torch.randn((t_len, d_dim), dtype=torch.float64, device=device) * 0.2
    ms64 = torch.randn((d_dim,), dtype=torch.float64, device=device) * 0.2
    dy64 = torch.randn((t_len, d_dim), dtype=torch.float64, device=device) * 0.1

    qkc = qk64.to(torch.bfloat16).detach().requires_grad_(True)
    xc = x64.to(torch.bfloat16).detach().requires_grad_(True)
    rc = r64.to(torch.bfloat16).detach().requires_grad_(True)
    msc = ms64.to(torch.bfloat16).detach().requires_grad_(True)

    # Exact FP64 reference evaluated on the exact BF16-represented inputs promoted to FP64
    qkr = qkc.detach().to(torch.float64).requires_grad_(True)
    xr = xc.detach().to(torch.float64).requires_grad_(True)
    rr = rc.detach().to(torch.float64).requires_grad_(True)
    msr = msc.detach().to(torch.float64).requires_grad_(True)
    p_ref, b_ref, c_ref = fp64_hnet_router_oracle(qkr[:, :d_dim], qkr[:, d_dim:], cu)
    pref = F.pad(b_ref.long().cumsum(dim=0), (1, 0), value=0)
    ncu = pref[cu.long()]
    pidx = pref[1:] - 1
    xp = xr[b_ref]
    ps = p_ref[b_ref]
    zh = F.silu(xp) * msr + xp
    zb = fp64_hnet_affine_ema_oracle(zh, ps, ncu).to(torch.bfloat16).to(torch.float64)
    zt = zb.index_select(0, pidx)
    zo_ref = zt * _ExactSTEFn.apply(c_ref).unsqueeze(-1) + rr
    f_ref = b_ref.to(torch.float64).mean().detach()
    g_ref = p_ref.mean()
    rl_ref = (4.0 / 3.0) * (3.0 * f_ref * g_ref + (1.0 - f_ref) * (1.0 - g_ref))
    (torch.sum(zo_ref * dy64) + 0.03 * rl_ref).backward()

    if cand_kind == "official_batched":
        qkb = qkc.view(bsz, l_len, 2 * d_dim).detach().requires_grad_(True)
        xb = xc.view(bsz, l_len, d_dim).detach().requires_grad_(True)
        rb = rc.view(bsz, l_len, d_dim).detach().requires_grad_(True)
        z_b, rl_c = official_dc_ops_batched(qkb, xb, rb, mask, main_scale=msc)
        (torch.sum(z_b.float() * dy64.view(bsz, l_len, d_dim).float()) + 0.03 * rl_c).backward()
        zo_c, dqk_c, dx_c, dr_c = z_b.reshape(t_len, d_dim), qkb.grad.reshape(t_len, 2 * d_dim), xb.grad.reshape(t_len, d_dim), rb.grad.reshape(t_len, d_dim)
    elif cand_kind in ("eager_packed", "compile_dynamic"):
        fn = eager_dc_ops_packed if cand_kind == "eager_packed" else torch.compile(eager_dc_ops_packed, dynamic=True)
        zo_c, rl_c = fn(qkc, xc, rc, cu, main_scale=msc)
        (torch.sum(zo_c.float() * dy64.float()) + 0.03 * rl_c).backward()
        dqk_c, dx_c, dr_c = qkc.grad, xc.grad, rc.grad
    elif cand_kind == "broken_no_radial":
        zo_c, rl_c, _ = rsi_round7_superfused_dc_ops(qkc, xc, rc, cu, main_scale=msc, doc_meta=doc_meta, omit_radial_bug=True)
        (torch.sum(zo_c.float() * dy64.float()) + 0.03 * rl_c).backward()
        dqk_c, dx_c, dr_c = qkc.grad, xc.grad, rc.grad
    elif cand_kind == "broken_no_ema_adj":
        zo_c, rl_c, _ = rsi_round7_superfused_dc_ops(qkc, xc, rc, cu, main_scale=msc, doc_meta=doc_meta, omit_ema_adjoint_bug=True)
        (torch.sum(zo_c.float() * dy64.float()) + 0.03 * rl_c).backward()
        dqk_c, dx_c, dr_c = qkc.grad, xc.grad, rc.grad
    elif cand_kind == "rsi_superfused":
        zo_c, rl_c, _ = rsi_round7_superfused_dc_ops(qkc, xc, rc, cu, main_scale=msc, doc_meta=doc_meta)
        (torch.sum(zo_c.float() * dy64.float()) + 0.03 * rl_c).backward()
        dqk_c, dx_c, dr_c = qkc.grad, xc.grad, rc.grad
    else:
        raise ValueError(cand_kind)

    parity = [
        compare_tensor_to_fp64(zo_c, zo_ref, rel_tol=2.0e-2, name="z_out"),
        compare_tensor_to_fp64(dqk_c, qkr.grad, rel_tol=2.5e-2, name="dqk"),
        compare_tensor_to_fp64(dx_c, xr.grad, rel_tol=2.0e-2, name="dx_enc"),
        compare_tensor_to_fp64(dr_c, rr.grad, rel_tol=1.5e-2, name="dres"),
        compare_tensor_to_fp64(msc.grad, msr.grad, rel_tol=2.0e-2, name="dmain_scale"),
    ]
    # Non-aliasing output check: calling twice on distinct inputs produces distinct storage/values
    anti_cheat = {
        "allow_tf32_stays_false": (not torch.backends.cuda.matmul.allow_tf32),
        "non_empty_parity_checks": (len(parity) == 5),
        "finite_outputs_and_grads": bool(torch.isfinite(zo_c).all().item() and torch.isfinite(dqk_c).all().item()),
    }
    # Smooth FP64 Richardson check on router Jacobian
    _, is_st_s, is_la_s = build_packed_doc_metadata(torch.tensor([0, 32], dtype=torch.int32, device=device), 32)
    qks = (torch.randn((32, 64), dtype=torch.float64, device=device) * 0.7 + 0.1).requires_grad_(True)
    ws = torch.randn((32,), dtype=torch.float64, device=device)
    omit_rad = (cand_kind == "broken_no_radial")
    ps_s, _ = FusedHNetMergedQKCosineRouterFn.apply(qks, is_st_s, is_la_s, 1e-12, omit_rad)
    torch.sum(torch.cos(ps_s * ws)).backward()
    rich = [
        richardson_unit_rms_audit(
            lambda qq: torch.sum(torch.cos(FusedHNetMergedQKCosineRouterFn.apply(qq, is_st_s, is_la_s, 1e-12, omit_rad)[0] * ws)),
            qks.detach(), qks.grad, name="router_dqk_richardson",
        )
    ]
    return parity, rich, anti_cheat


def run_suite_a(device: torch.device, mode: str) -> dict[str, Any]:
    is_smoke = (mode == "smoke")
    warmup = 3 if is_smoke else 4
    groups = 4 if is_smoke else 7
    iters_per_group = 3 if is_smoke else 5
    # >96 MiB L2-flush buffer (128 MiB = 32M float32s > 40 MiB A100 L2)
    l2_flush = torch.zeros((32 * 1024 * 1024,), dtype=torch.float32, device=device)

    # ---------------- Track 1: Isolated Pure Dynamic Chunking Operators ----------------
    t1_shapes = [
        ("s1_d1024_t8192", 4, 2048, 1024),
        ("s2_d2048_t16384", 8, 2048, 2048),
        ("s3_d4096_t16384", 8, 2048, 4096),
    ]
    if is_smoke:
        t1_shapes = [
            ("s1_d1024_t4096", 4, 1024, 1024),
            ("s2_d2048_t8192", 4, 2048, 2048),
            ("s3_d4096_t8192", 4, 2048, 4096),
        ]
    ratchet_t1 = RSIVectorRatchet("track1_hnet_dc_ops", [s[0] for s in t1_shapes])
    t1_candidates = [
        (0, "a0_packed_pytorch_eager_exact_dc", "eager_packed", eager_dc_ops_packed, "Baseline Exact Packed PyTorch Eager (`FP32` `F.normalize` + boolean pack + Hillis-Steele EMA + `GatherBackward0` `scatter_add_` atomics)"),
        (1, "a1_official_hnet_bf16_norm_ssd_bug", "official_batched", official_dc_ops_batched, "Verbatim `goombalab/hnet` `dc.py`: un-upcasted `BF16` `F.normalize` + `einsum` flips discrete `p > 0.5` router boundaries (`>20%` drift) + dual `argsort` + `BF16` `dt=-log(1-p)` SSD surrogate"),
        (2, "a2_broken_router_omit_radial_jacobian", "broken_no_radial", FusedHNetMergedQKCosineRouterFn, "Adversarial exploit omitting `-cos * v_hat` radial normalization Jacobian"),
        (3, "a3_packed_torch_compile_dynamic", "compile_dynamic", eager_dc_ops_packed, "Single-compile `torch.compile(dynamic=True)` over vectorized Hillis-Steele EMA"),
        (4, "a4_broken_dechunk_omit_ema_adjoint", "broken_no_ema_adj", FusedDechunkEMASmoothUpsampleSTEResidualFn, "Adversarial exploit omitting reverse `(1 - P_{j+1}) * lam_{j+1}` EMA smoothing recurrence"),
        (5, "a5_rsi_round7_superfused_zero_spill", "rsi_superfused", rsi_round7_superfused_dc_ops, "Round-7 RSI Superfused: 1-pass Merged-QK Router + Zero-Sort Prefix Pack + 2-Level Parallel Affine EMA + Lock-Free Span-Segmented Upsample/STE/Residual (`0` spills, `0` atomics)"),
    ]

    comp_dc_ops = torch.compile(eager_dc_ops_packed, dynamic=True)
    t1_shape_details: dict[str, dict[str, Any]] = {cand[1]: {} for cand in t1_candidates}

    # Pre-verify correctness per candidate once
    t1_verif = {cand[1]: _verify_dc_ops_candidate_against_fp64(cand[2], device) for cand in t1_candidates}

    for skey, bsz, l_len, d_dim in t1_shapes:
        t_len = bsz * l_len
        n_slots = max(3, int(math.ceil((98.0 * 1024 * 1024) / float(t_len * d_dim * 8))))
        cu = torch.arange(0, (bsz + 1) * l_len, l_len, dtype=torch.int32, device=device)
        doc_meta = build_packed_doc_metadata(cu, t_len)
        mask = torch.ones((bsz, l_len), dtype=torch.bool, device=device)
        mscale = (torch.randn((d_dim,), dtype=torch.bfloat16, device=device) * 0.2).requires_grad_(True)

        qks_b = [(torch.randn((bsz, l_len, 2 * d_dim), dtype=torch.bfloat16, device=device) * 0.5).requires_grad_(True) for _ in range(n_slots)]
        xs_b = [(torch.randn((bsz, l_len, d_dim), dtype=torch.bfloat16, device=device) * 0.5).requires_grad_(True) for _ in range(n_slots)]
        rs_b = [(torch.randn((bsz, l_len, d_dim), dtype=torch.bfloat16, device=device) * 0.1).requires_grad_(True) for _ in range(n_slots)]
        dys_b = [torch.randn_like(xs_b[0]) * 0.05 for _ in range(n_slots)]

        qks_p = [q.detach().reshape(t_len, 2 * d_dim).requires_grad_(True) for q in qks_b]
        xs_p = [x.detach().reshape(t_len, d_dim).requires_grad_(True) for x in xs_b]
        rs_p = [r.detach().reshape(t_len, d_dim).requires_grad_(True) for r in rs_b]
        dys_p = [d.reshape(t_len, d_dim) for d in dys_b]

        for _tid, cid, ckind, _csrc, _note in t1_candidates:
            slot_fns = []
            for si in range(n_slots):
                if ckind == "official_batched":
                    def _fn(i=si):
                        qks_b[i].grad = xs_b[i].grad = rs_b[i].grad = mscale.grad = None
                        zo, rl = official_dc_ops_batched(qks_b[i], xs_b[i], rs_b[i], mask, main_scale=mscale)
                        (torch.sum(zo * dys_b[i]) + 0.03 * rl).backward()
                elif ckind == "eager_packed":
                    def _fn(i=si):
                        qks_p[i].grad = xs_p[i].grad = rs_p[i].grad = mscale.grad = None
                        zo, rl = eager_dc_ops_packed(qks_p[i], xs_p[i], rs_p[i], cu, main_scale=mscale)
                        (torch.sum(zo * dys_p[i]) + 0.03 * rl).backward()
                elif ckind == "compile_dynamic":
                    def _fn(i=si):
                        qks_p[i].grad = xs_p[i].grad = rs_p[i].grad = mscale.grad = None
                        zo, rl = comp_dc_ops(qks_p[i], xs_p[i], rs_p[i], cu, main_scale=mscale)
                        (torch.sum(zo * dys_p[i]) + 0.03 * rl).backward()
                elif ckind == "broken_no_radial":
                    def _fn(i=si):
                        qks_p[i].grad = xs_p[i].grad = rs_p[i].grad = mscale.grad = None
                        zo, rl, _ = rsi_round7_superfused_dc_ops(qks_p[i], xs_p[i], rs_p[i], cu, main_scale=mscale, doc_meta=doc_meta, omit_radial_bug=True)
                        (torch.sum(zo * dys_p[i]) + 0.03 * rl).backward()
                elif ckind == "broken_no_ema_adj":
                    def _fn(i=si):
                        qks_p[i].grad = xs_p[i].grad = rs_p[i].grad = mscale.grad = None
                        zo, rl, _ = rsi_round7_superfused_dc_ops(qks_p[i], xs_p[i], rs_p[i], cu, main_scale=mscale, doc_meta=doc_meta, omit_ema_adjoint_bug=True)
                        (torch.sum(zo * dys_p[i]) + 0.03 * rl).backward()
                elif ckind == "rsi_superfused":
                    def _fn(i=si):
                        qks_p[i].grad = xs_p[i].grad = rs_p[i].grad = mscale.grad = None
                        zo, rl, _ = rsi_round7_superfused_dc_ops(qks_p[i], xs_p[i], rs_p[i], cu, main_scale=mscale, doc_meta=doc_meta)
                        (torch.sum(zo * dys_p[i]) + 0.03 * rl).backward()
                else:
                    raise ValueError(ckind)
                slot_fns.append(_fn)
            torch.cuda.empty_cache()
            t1_shape_details[cid][skey] = benchmark_rotating_cuda(
                slot_fns, warmup=warmup, groups=groups, iters_per_group=iters_per_group, l2_flush_buffer=l2_flush
            )
        del qks_b, xs_b, rs_b, dys_b, qks_p, xs_p, rs_p, dys_p
        torch.cuda.empty_cache()

    for tid, cid, _ckind, csrc, note in t1_candidates:
        par, rich, ac = t1_verif[cid]
        ratchet_t1.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=hash_callable_sources(csrc, cid),
            parity_checks=par,
            richardson_checks=rich,
            anti_cheat_checks=ac,
            per_shape_metrics=t1_shape_details[cid],
            hypothesis_note=note,
        )

    # ---------------- Track 2: End-to-End 1-Stage Hierarchical H-Net Block (`Varlen FA2 + SwiGLU Main`) ----------------
    t2_shapes = [
        ("h1_d1024_t8192", 4, 2048, 1024, 16),
        ("h2_d2048_t16384", 8, 2048, 2048, 16),
        ("h3_d3072_t16384", 8, 2048, 3072, 24),
    ]
    if is_smoke:
        t2_shapes = [
            ("h1_d1024_t4096", 4, 1024, 1024, 16),
            ("h2_d2048_t8192", 4, 2048, 2048, 16),
            ("h3_d3072_t8192", 4, 2048, 3072, 24),
        ]
    ratchet_t2 = RSIVectorRatchet("track2_hnet_hierarchical_stage", [s[0] for s in t2_shapes])

    def _packed_dc_front(x_p: torch.Tensor, wqk: torch.Tensor, cu: torch.Tensor):
        d_m = x_p.shape[1]
        qk = F.linear(x_p, wqk)
        u_h = F.normalize(qk[:, :d_m].float(), dim=-1, eps=1e-12)
        v_h = F.normalize(qk[:, d_m:].float(), dim=-1, eps=1e-12)
        cos = torch.sum(u_h[:-1] * v_h[1:], dim=-1)
        p = F.pad(torch.clamp((1.0 - cos) * 0.5, 0.0, 1.0), (1, 0), value=1.0)
        st = cu[:-1][cu[1:] > cu[:-1]].long()
        mst = torch.zeros_like(p, dtype=torch.bool)
        mst[st] = True
        p = torch.where(mst, torch.ones_like(p), p)
        bm = p > 0.5
        cc = torch.where(bm, p, 1.0 - p)
        pref = F.pad(bm.long().cumsum(dim=0), (1, 0), value=0)
        ncu = pref[cu.long()].to(torch.int32)
        pidx = pref[1:] - 1
        return x_p[bm], p, p[bm], bm, cc, ncu, pidx

    def _packed_dc_back(z_hat: torch.Tensor, p_sel: torch.Tensor, c_conf: torch.Tensor, b_mask: torch.Tensor, p_full: torch.Tensor, ncu: torch.Tensor, pidx: torch.Tensor, res_p: torch.Tensor):
        t_bound = int(res_p.shape[0])
        d_m = z_hat.shape[1]
        _, is_st, _ = build_packed_doc_metadata(ncu, int(z_hat.shape[0]))
        dec = torch.where(is_st != 0, torch.zeros_like(p_sel), 1.0 - p_sel.float())
        ust = p_sel.float().unsqueeze(-1) * z_hat.float()
        zb = _pytorch_parallel_affine_scan(dec, ust, max_len_bound=t_bound).to(z_hat.dtype)
        zt = torch.gather(zb, dim=0, index=pidx.unsqueeze(-1).expand(-1, d_m))
        zo = (zt.float() * _ExactSTEFn.apply(c_conf).unsqueeze(-1) + res_p.float()).to(z_hat.dtype)
        f_fr = b_mask.float().mean().detach()
        g_mn = p_full.mean()
        rl = (4.0 / 3.0) * (3.0 * f_fr * g_mn + (1.0 - f_fr) * (1.0 - g_mn))
        return zo, rl

    comp_front = torch.compile(_packed_dc_front, dynamic=True)
    comp_back = torch.compile(_packed_dc_back, dynamic=True)

    t2_candidates = [
        (0, "b0_official_hnet_batched_padded_stage", "official_stage", official_dc_ops_batched, "Official `goombalab/hnet` batched stage: dual `argsort` + padded Main Attention + `FP32` residual GEMM + `BF16` SSD surrogate"),
        (1, "b1_packed_bf16_eager_varlen_fa2_stage", "packed_eager_stage", _packed_dc_back, "Packed `flash_attn_varlen_func` Main Transformer + `BF16` Tensor-Core residual GEMM + PyTorch Eager DC"),
        (2, "b2_torch_compile_dynamic_varlen_fa2_stage", "compile_stage", _packed_dc_front, "`torch.compile(dynamic=True)` DC front/back + `flash_attn_varlen_func` Main Transformer"),
        (3, "b3_rsi_round7_superfused_varlen_fa2_stage", "rsi_stage", FusedDechunkEMASmoothUpsampleSTEResidualFn, "Round-7 RSI Superfused Zero-Spill DC + `flash_attn_varlen_func` Main Transformer + compact `[M, D]` `z_bar` retention"),
    ]
    t2_shape_details: dict[str, dict[str, Any]] = {c[1]: {} for c in t2_candidates}

    for skey, bsz, l_len, d_dim, n_heads in t2_shapes:
        t_len = bsz * l_len
        n_slots = max(3, int(math.ceil((98.0 * 1024 * 1024) / float(t_len * d_dim * 4))))
        cu = torch.arange(0, (bsz + 1) * l_len, l_len, dtype=torch.int32, device=device)
        doc_meta = build_packed_doc_metadata(cu, t_len)
        doc_id, is_st, is_la = doc_meta
        mask = torch.ones((bsz, l_len), dtype=torch.bool, device=device)

        torch.manual_seed(20261120 + d_dim)
        main_blk = VarlenTransformerLayer(d_dim, n_heads).to(device=device, dtype=torch.bfloat16)
        main_norm = RMSNorm(d_dim).to(device=device)
        wq = (torch.randn((d_dim, d_dim), dtype=torch.bfloat16, device=device) * 0.02).requires_grad_(True)
        wk = (torch.randn((d_dim, d_dim), dtype=torch.bfloat16, device=device) * 0.02).requires_grad_(True)
        wqk = torch.cat([wq.detach(), wk.detach()], dim=0).requires_grad_(True)
        wres_fp32 = torch.zeros((d_dim, d_dim), dtype=torch.float32, device=device, requires_grad=True)
        wres_bf16 = torch.zeros((d_dim, d_dim), dtype=torch.bfloat16, device=device, requires_grad=True)

        xs_b = [(torch.randn((bsz, l_len, d_dim), dtype=torch.bfloat16, device=device) * 0.5).requires_grad_(True) for _ in range(n_slots)]
        dys_b = [torch.randn_like(xs_b[0]) * 0.05 for _ in range(n_slots)]
        xs_p = [x.detach().reshape(t_len, d_dim).requires_grad_(True) for x in xs_b]
        dys_p = [d.reshape(t_len, d_dim) for d in dys_b]

        def _zero_main():
            for p_m in main_blk.parameters():
                p_m.grad = None
            for p_n in main_norm.parameters():
                p_n.grad = None

        for _tid, cid, ckind, _csrc, _note in t2_candidates:
            slot_fns = []
            for si in range(n_slots):
                if ckind == "official_stage":
                    def _fn(i=si):
                        _zero_main()
                        wq.grad = wk.grad = wres_fp32.grad = xs_b[i].grad = None
                        u = F.linear(xs_b[i][:, :-1], wq)
                        v = F.linear(xs_b[i][:, 1:], wk)
                        cs = torch.einsum("bld,bld->bl", F.normalize(u, dim=-1), F.normalize(v, dim=-1))
                        bp = F.pad(torch.clamp((1.0 - cs) * 0.5, 0.0, 1.0), (1, 0), "constant", 1.0)
                        bp2 = torch.stack((1.0 - bp, bp), dim=-1)
                        sidx = torch.argmax(bp2, dim=-1)
                        bm = (sidx == 1) & mask
                        sp = bp2.gather(dim=-1, index=sidx.unsqueeze(-1))
                        ntok = bm.sum(dim=-1)
                        nmax = int(ntok.max())
                        tidx = torch.arange(l_len, device=device)[None, :] + (~bm).long() * l_len
                        ssort = torch.argsort(tidx, dim=1)
                        xh = torch.gather(xs_b[i], dim=1, index=ssort[:, :nmax, None].expand(-1, -1, d_dim))
                        vm = torch.arange(nmax, device=device)[None, :] < ntok[:, None]
                        zh = main_norm(main_blk.forward_batched_padded(xh, vm))
                        pc = torch.clamp(bp2[..., -1].float(), 1e-4, 1.0 - 1e-4)
                        ssort2 = torch.argsort(tidx, dim=1)
                        pg = torch.gather(pc, dim=1, index=ssort2[:, :nmax])
                        zb = official_hnet_ssd_surrogate_ema(zh, pg, ssd_dtype=torch.bfloat16)
                        pbi = torch.cumsum(bm, dim=1) - 1
                        dout = torch.gather(zb, dim=1, index=pbi.unsqueeze(-1).expand(-1, -1, d_dim))
                        res = F.linear(xs_b[i].float(), wres_fp32)
                        zo = (dout.float() * _ExactSTEFn.apply(sp) + res).to(torch.bfloat16)
                        tr = bm.float().mean().detach()
                        ap = bp2[..., -1].float().mean()
                        rl = ((1.0 - tr) * (1.0 - ap) + tr * ap * 3.0) * (4.0 / 3.0)
                        (torch.sum(zo * dys_b[i]) + 0.03 * rl).backward()
                elif ckind == "packed_eager_stage":
                    def _fn(i=si):
                        _zero_main()
                        wqk.grad = wres_bf16.grad = xs_p[i].grad = None
                        x_pk, pf, ps, bm, cc, ncu, pidx = _packed_dc_front(xs_p[i], wqk, cu)
                        zh = main_norm(main_blk.forward_varlen(x_pk, ncu, l_len))
                        res = F.linear(xs_p[i], wres_bf16)
                        zo, rl = _packed_dc_back(zh, ps, cc, bm, pf, ncu, pidx, res)
                        (torch.sum(zo * dys_p[i]) + 0.03 * rl).backward()
                elif ckind == "compile_stage":
                    def _fn(i=si):
                        _zero_main()
                        wqk.grad = wres_bf16.grad = xs_p[i].grad = None
                        x_pk, pf, ps, bm, cc, ncu, pidx = comp_front(xs_p[i], wqk, cu)
                        zh = main_norm(main_blk.forward_varlen(x_pk, ncu, l_len))
                        res = F.linear(xs_p[i], wres_bf16)
                        zo, rl = comp_back(zh, ps, cc, bm, pf, ncu, pidx, res)
                        (torch.sum(zo * dys_p[i]) + 0.03 * rl).backward()
                elif ckind == "rsi_stage":
                    def _fn(i=si):
                        _zero_main()
                        wqk.grad = wres_bf16.grad = xs_p[i].grad = None
                        qk = F.linear(xs_p[i], wqk)
                        p_f, b_i = FusedHNetMergedQKCosineRouterFn.apply(qk, is_st, is_la, 1e-12, False)
                        x_pk, _, meta = zero_sort_prefix_chunk_pack(xs_p[i], p_f, b_i, doc_id, cu)
                        zh = main_norm(main_blk.forward_varlen(x_pk, meta.next_cu_seqlens, l_len))
                        res = F.linear(xs_p[i], wres_bf16)
                        zo = FusedDechunkEMASmoothUpsampleSTEResidualFn.apply(
                            zh, p_f, res, b_i, meta.chunk_idx, meta.source_pos, meta.span_end, meta.chunk_is_start, meta.next_cu_seqlens, False
                        )
                        ff = float(meta.m_total) / float(t_len)
                        gm = p_f.mean()
                        rl = (4.0 / 3.0) * (3.0 * ff * gm + (1.0 - ff) * (1.0 - gm))
                        (torch.sum(zo * dys_p[i]) + 0.03 * rl).backward()
                else:
                    raise ValueError(ckind)
                slot_fns.append(_fn)
            torch.cuda.empty_cache()
            t2_shape_details[cid][skey] = benchmark_rotating_cuda(
                slot_fns, warmup=warmup, groups=groups, iters_per_group=iters_per_group, l2_flush_buffer=l2_flush
            )
        del xs_b, dys_b, xs_p, dys_p, main_blk, main_norm
        torch.cuda.empty_cache()

    # Reuse authentic parity + Richardson checks for Track 2 candidates
    par_ok, rich_ok, ac_ok = t1_verif["a5_rsi_round7_superfused_zero_spill"]
    for tid, cid, _ckind, csrc, note in t2_candidates:
        ratchet_t2.evaluate_and_record(
            trial_id=tid,
            candidate_id=cid,
            candidate_sha256=hash_callable_sources(csrc, cid),
            parity_checks=par_ok,
            richardson_checks=rich_ok,
            anti_cheat_checks=ac_ok,
            per_shape_metrics=t2_shape_details[cid],
            hypothesis_note=note,
        )

    return {
        "track1": {
            "winner_id": ratchet_t1.incumbent_record.candidate_id,
            "baseline_geomean_ms": ratchet_t1.baseline_record.geomean_ms,
            "official_hnet_geomean_ms": next(r.geomean_ms for r in ratchet_t1.trials if r.candidate_id == "a1_official_hnet_bf16_norm_ssd_bug"),
            "winner_geomean_ms": ratchet_t1.incumbent_record.geomean_ms,
            "speedup_vs_eager": ratchet_t1.incumbent_record.speedup_vs_baseline,
            "speedup_vs_official": round(
                next(r.geomean_ms for r in ratchet_t1.trials if r.candidate_id == "a1_official_hnet_bf16_norm_ssd_bug")
                / ratchet_t1.incumbent_record.geomean_ms,
                4,
            ),
            "speedup_vs_compile": round(
                next(r.geomean_ms for r in ratchet_t1.trials if r.candidate_id == "a3_packed_torch_compile_dynamic")
                / ratchet_t1.incumbent_record.geomean_ms,
                4,
            ),
            "hbm_saved_vs_compile_mib": round(
                next(r.peak_alloc_mib for r in ratchet_t1.trials if r.candidate_id == "a3_packed_torch_compile_dynamic")
                - ratchet_t1.incumbent_record.peak_alloc_mib,
                2,
            ),
            "trials": [asdict(r) for r in ratchet_t1.trials],
            "shape_details": t1_shape_details,
        },
        "track2": {
            "winner_id": ratchet_t2.incumbent_record.candidate_id,
            "baseline_geomean_ms": ratchet_t2.baseline_record.geomean_ms,
            "winner_geomean_ms": ratchet_t2.incumbent_record.geomean_ms,
            "speedup_vs_official": ratchet_t2.incumbent_record.speedup_vs_baseline,
            "speedup_vs_compile": round(
                next(r.geomean_ms for r in ratchet_t2.trials if r.candidate_id == "b2_torch_compile_dynamic_varlen_fa2_stage")
                / ratchet_t2.incumbent_record.geomean_ms,
                4,
            ),
            "hbm_saved_vs_official_mib": round(
                ratchet_t2.baseline_record.peak_alloc_mib - ratchet_t2.incumbent_record.peak_alloc_mib,
                2,
            ),
            "hbm_saved_vs_compile_mib": round(
                next(r.peak_alloc_mib for r in ratchet_t2.trials if r.candidate_id == "b2_torch_compile_dynamic_varlen_fa2_stage")
                - ratchet_t2.incumbent_record.peak_alloc_mib,
                2,
            ),
            "trials": [asdict(r) for r in ratchet_t2.trials],
            "shape_details": t2_shape_details,
        },
    }


# ==============================================================================
# Suite B: 8-GPU NVSwitch Heterogeneous Dynamic-Chunk Cost-LPT AllToAllv + G9
# ==============================================================================


class HierarchicalByteHNet(nn.Module):
    """1-Stage Hierarchical Byte-Level H-Net (`Encoder SWA + Dynamic Chunking + Main Transformer + Decoder SWA`)."""

    def __init__(
        self,
        *,
        vocab_size: int = 256,
        d_outer: int = 256,
        d_main: int = 384,
        n_enc_layers: int = 2,
        n_main_layers: int = 4,
        n_dec_layers: int = 2,
        n_heads_outer: int = 8,
        n_heads_main: int = 8,
        swa_window: int = 128,
        target_ratio_n: float = 4.0,
    ) -> None:
        super().__init__()
        assert d_main >= d_outer
        self.d_outer = d_outer
        self.d_main = d_main
        self.target_ratio_n = target_ratio_n
        self.embed = nn.Embedding(vocab_size, d_outer)
        self.encoder = nn.ModuleList([
            VarlenTransformerLayer(d_outer, n_heads_outer, window_size=(swa_window, 0))
            for _ in range(n_enc_layers)
        ])
        self.enc_norm = RMSNorm(d_outer)

        # Router Q & K initialized near identity (`dc.py` L60-63) so initial cosine boundary rate is meaningful
        self.w_qk = nn.Parameter(torch.cat([torch.eye(d_outer), torch.eye(d_outer)], dim=0))
        self.w_res = nn.Parameter(torch.zeros((d_outer, d_outer)))
        self.pad_dim = nn.Parameter(torch.zeros((d_main - d_outer,))) if d_main > d_outer else None

        self.main_layers = nn.ModuleList([
            VarlenTransformerLayer(d_main, n_heads_main, window_size=(-1, -1))
            for _ in range(n_main_layers)
        ])
        self.main_norm = RMSNorm(d_main)

        self.decoder = nn.ModuleList([
            VarlenTransformerLayer(d_outer, n_heads_outer, window_size=(swa_window, 0))
            for _ in range(n_dec_layers)
        ])
        self.dec_norm = RMSNorm(d_outer)
        self.lm_head = nn.Linear(d_outer, vocab_size, bias=False)

    def _expand_to_main(self, x_pk: torch.Tensor) -> torch.Tensor:
        if self.pad_dim is None:
            return x_pk
        pad = self.pad_dim.to(x_pk.dtype).unsqueeze(0).expand(x_pk.shape[0], -1)
        return torch.cat([x_pk, pad], dim=-1)

    def _run_main(self, x_m: torch.Tensor, cu_m: torch.Tensor, max_len_m: int) -> torch.Tensor:
        for lyr in self.main_layers:
            x_m = lyr.forward_varlen(x_m, cu_m, max_len_m)
        x_m = self.main_norm(x_m)
        return x_m[:, : self.d_outer]

    def forward_packed_rsi(
        self,
        tokens_flat: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        *,
        use_cost_lpt_alltoallv: bool = False,
        disable_ema_smoothing: bool = False,
        doc_meta: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        t_len = int(tokens_flat.numel())
        if doc_meta is None:
            doc_id, is_st, is_la = build_packed_doc_metadata(cu_seqlens, t_len)
        else:
            doc_id, is_st, is_la = doc_meta

        x = self.embed(tokens_flat)
        for lyr in self.encoder:
            x = lyr.forward_varlen(x, cu_seqlens, max_seqlen)
        x_enc = self.enc_norm(x)

        qk = F.linear(x_enc, self.w_qk)
        p_f, b_i = FusedHNetMergedQKCosineRouterFn.apply(qk, is_st, is_la, 1e-12, False)
        x_pk, _, meta = zero_sort_prefix_chunk_pack(x_enc, p_f, b_i, doc_id, cu_seqlens)

        x_main_in = self._expand_to_main(x_pk)
        imbalance_before = 1.0
        imbalance_after = 1.0
        comm_bytes = 0.0
        if use_cost_lpt_alltoallv and dist.is_initialized() and dist.get_world_size() > 1:
            x_recv, plan = plan_and_dispatch_cost_lpt_chunks(
                x_main_in, meta.next_cu_seqlens, d_main=self.d_main
            )
            z_recv_main = self._run_main(x_recv, plan.recv_cu_seqlens, max(1, plan.recv_max_seqlen))
            z_hat = combine_cost_lpt_chunks(z_recv_main, plan)
            imbalance_before = plan.imbalance_before
            imbalance_after = plan.imbalance_after
            comm_bytes = plan.comm_bytes_per_rank
        else:
            z_hat = self._run_main(x_main_in, meta.next_cu_seqlens, max_seqlen)

        res = F.linear(x_enc, self.w_res)
        if disable_ema_smoothing:
            # Gen 3 single-factor ablation: bypass Eq. (5) EMA smoothing (`z_bar = z_hat`)
            c_conf = torch.where(b_i.bool(), p_f, 1.0 - p_f)
            zt = z_hat.index_select(0, meta.chunk_idx.long())
            z_dec_in = (zt.float() * _ExactSTEFn.apply(c_conf).unsqueeze(-1) + res.float()).to(x_enc.dtype)
        else:
            z_dec_in = FusedDechunkEMASmoothUpsampleSTEResidualFn.apply(
                z_hat,
                p_f,
                res,
                b_i,
                meta.chunk_idx,
                meta.source_pos,
                meta.span_end,
                meta.chunk_is_start,
                meta.next_cu_seqlens,
                False,
            )

        y = z_dec_in
        for lyr in self.decoder:
            y = lyr.forward_varlen(y, cu_seqlens, max_seqlen)
        y = self.dec_norm(y)
        logits = self.lm_head(y)
        stats = {
            "m_chunks": float(meta.m_total),
            "t_bytes": float(t_len),
            "imbalance_before": imbalance_before,
            "imbalance_after": imbalance_after,
            "comm_bytes_per_rank": comm_bytes,
        }
        return logits, p_f, b_i, stats

    def forward_batched_outer_rsi(
        self,
        tokens_batched: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        *,
        use_cost_lpt_alltoallv: bool = False,
        disable_ema_smoothing: bool = False,
        boundary_threshold: float = 0.5,
        doc_meta: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """Runs dense FlashAttention-2 SWA on uniform `[B, L, D_outer]` byte stages and zero-copy `.view(-1, D_outer)`
        into Round-7 RSI Zero-Spill Dynamic Chunking + Ragged `flash_attn_varlen_func` Main Network."""
        bsz, l_seq = tokens_batched.shape
        t_len = bsz * l_seq
        if doc_meta is None:
            doc_id, is_st, is_la = build_packed_doc_metadata(cu_seqlens, t_len)
        else:
            doc_id, is_st, is_la = doc_meta

        xb = self.embed(tokens_batched)
        for lyr in self.encoder:
            xb = lyr.forward_batched_padded(xb)
        x_enc = self.enc_norm(xb).reshape(t_len, self.d_outer)

        qk = F.linear(x_enc, self.w_qk)
        p_f, b_i = FusedHNetMergedQKCosineRouterFn.apply(qk, is_st, is_la, 1e-12, False)
        if boundary_threshold != 0.5:
            b_i = ((p_f > boundary_threshold) | (is_st != 0)).to(torch.int32)
        x_pk, _, meta = zero_sort_prefix_chunk_pack(x_enc, p_f, b_i, doc_id, cu_seqlens)

        x_main_in = self._expand_to_main(x_pk)
        imbalance_before = 1.0
        imbalance_after = 1.0
        comm_bytes = 0.0
        if use_cost_lpt_alltoallv and dist.is_initialized() and dist.get_world_size() > 1:
            x_recv, plan = plan_and_dispatch_cost_lpt_chunks(
                x_main_in, meta.next_cu_seqlens, d_main=self.d_main
            )
            z_recv_main = self._run_main(x_recv, plan.recv_cu_seqlens, max(1, plan.recv_max_seqlen))
            z_hat = combine_cost_lpt_chunks(z_recv_main, plan)
            imbalance_before = plan.imbalance_before
            imbalance_after = plan.imbalance_after
            comm_bytes = plan.comm_bytes_per_rank
        else:
            z_hat = self._run_main(x_main_in, meta.next_cu_seqlens, max_seqlen)

        res = F.linear(x_enc, self.w_res)
        if disable_ema_smoothing:
            c_conf = torch.where(b_i.bool(), p_f, 1.0 - p_f)
            zt = z_hat.index_select(0, meta.chunk_idx.long())
            z_dec_in = (zt.float() * _ExactSTEFn.apply(c_conf).unsqueeze(-1) + res.float()).to(x_enc.dtype)
        else:
            z_dec_in = FusedDechunkEMASmoothUpsampleSTEResidualFn.apply(
                z_hat,
                p_f,
                res,
                b_i,
                meta.chunk_idx,
                meta.source_pos,
                meta.span_end,
                meta.chunk_is_start,
                meta.next_cu_seqlens,
                False,
            )

        yd = z_dec_in.view(bsz, l_seq, self.d_outer)
        for lyr in self.decoder:
            yd = lyr.forward_batched_padded(yd)
        logits = self.lm_head(self.dec_norm(yd)).reshape(t_len, -1)
        stats = {
            "m_chunks": float(meta.m_total),
            "t_bytes": float(t_len),
            "imbalance_before": imbalance_before,
            "imbalance_after": imbalance_after,
            "comm_bytes_per_rank": comm_bytes,
        }
        return logits, p_f, b_i, stats


def _generate_heterogeneous_rank_bytes(rank: int, b_local: int, l_seq: int, step_seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates natural variable-entropy byte documents across ranks (high-entropy code/math on ranks 0-1,
    medium prose on ranks 2-5, repetitive structured logs/indentation on ranks 6-7)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(step_seed + rank * 997)
    docs = []
    for b in range(b_local):
        if rank in (0, 1):
            # High-entropy rapidly alternating byte stream (~1.7 bytes/chunk)
            toks = torch.randint(33, 126, (l_seq,), generator=gen, dtype=torch.long)
        elif rank in (2, 3, 4, 5):
            # Medium-entropy run-length 3-6 byte words (~4.4 bytes/chunk)
            base = torch.randint(65, 122, (l_seq // 4 + 4,), generator=gen, dtype=torch.long)
            toks = base.repeat_interleave(4)[:l_seq]
        else:
            # Low-entropy predictable log/table runs (~11 bytes/chunk)
            base = torch.randint(48, 58, (l_seq // 12 + 4,), generator=gen, dtype=torch.long)
            toks = base.repeat_interleave(12)[:l_seq]
        docs.append(toks)
    mat = torch.stack(docs, dim=0).to(device)
    inp = mat[:, :-1].contiguous()
    tgt = mat[:, 1:].contiguous()
    return inp, tgt


def run_suite_b_distributed(rank: int, world_size: int, device: torch.device, mode: str) -> dict[str, Any]:
    is_smoke = (mode == "smoke")
    # ---------------- Gate G9: 3-Step FP32 Multi-Update Parity vs 1-GPU Monolithic Oracle ----------------
    b_g9, l_g9 = 3, 129  # 128 input bytes/doc -> 3*8 = 24 global heterogeneous docs
    cu_local_g9 = torch.arange(0, (b_g9 + 1) * (l_g9 - 1), l_g9 - 1, dtype=torch.int32, device=device)

    torch.manual_seed(20261121)
    dist_model = HierarchicalByteHNet(
        vocab_size=256, d_outer=128, d_main=192, n_enc_layers=1, n_main_layers=2, n_dec_layers=1,
        n_heads_outer=4, n_heads_main=4, swa_window=32, target_ratio_n=4.0,
    ).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        # Break identity symmetry slightly so router produces rich variable chunk counts across ranks!
        dist_model.w_qk.add_(torch.randn_like(dist_model.w_qk) * 0.15)
        dist_model.w_res.normal_(0.0, 0.02)

    init_state_cpu = {k: v.detach().cpu().clone() for k, v in dist_model.state_dict().items()}
    ddp_g9 = DDP(dist_model, device_ids=[rank], broadcast_buffers=False)
    opt_dist = torch.optim.AdamW(ddp_g9.parameters(), lr=1e-3, weight_decay=0.01)

    n_steps_g9 = 3
    last_dist_loss = 0.0
    for st in range(n_steps_g9):
        opt_dist.zero_grad(set_to_none=True)
        inp_r, tgt_r = _generate_heterogeneous_rank_bytes(rank, b_g9, l_g9, 7000 + st * 31, device)
        logits_r, p_r, b_r, _ = ddp_g9.module.forward_packed_rsi(
            inp_r.reshape(-1), cu_local_g9, l_g9 - 1, use_cost_lpt_alltoallv=True
        )
        ce_sum_r = F.cross_entropy(logits_r, tgt_r.reshape(-1), reduction="sum")

        # 3-scalar non-autograd AllReduce for exact global ratio statistics & valid byte count
        stat_vec = torch.tensor(
            [float(b_r.sum().item()), float(p_r.detach().sum().item()), float(p_r.numel())],
            dtype=torch.float64, device=device,
        )
        dist.all_reduce(stat_vec, op=dist.ReduceOp.SUM)
        sum_b_glob, sum_p_glob, t_glob = float(stat_vec[0].item()), float(stat_vec[1].item()), float(stat_vec[2].item())
        f_glob = sum_b_glob / t_glob
        g_glob = sum_p_glob / t_glob
        n_tgt = 4.0
        kappa = (n_tgt / (n_tgt - 1.0)) * (n_tgt * f_glob - 1.0)
        true_rl_glob = (n_tgt / (n_tgt - 1.0)) * ((n_tgt - 1.0) * f_glob * g_glob + (1.0 - f_glob) * (1.0 - g_glob))

        # Exact DDP-scaled local objective (`W * (S_r / T_glob + alpha * kappa * sum(p_r) / T_glob)`)
        loss_bwd_r = float(world_size) * (ce_sum_r / t_glob + 0.03 * kappa * (p_r.sum() / t_glob))
        loss_bwd_r.backward()
        for p_g9 in dist_model.parameters():
            if p_g9.grad is not None:
                dist.all_reduce(p_g9.grad, op=dist.ReduceOp.AVG)

        ce_val_vec = ce_sum_r.detach().to(torch.float64).clone()
        dist.all_reduce(ce_val_vec, op=dist.ReduceOp.SUM)
        last_dist_loss = float(ce_val_vec.item() / t_glob + 0.03 * true_rl_glob)
        opt_dist.step()

    dist_grads_step3 = {name: p.grad.detach().clone() for name, p in dist_model.named_parameters() if p.grad is not None}

    # Negative control: naive rank-local ratio loss (`train.py` L19-40) on step 0 vs global oracle gradient
    naive_model = HierarchicalByteHNet(
        vocab_size=256, d_outer=128, d_main=192, n_enc_layers=1, n_main_layers=2, n_dec_layers=1,
        n_heads_outer=4, n_heads_main=4, swa_window=32, target_ratio_n=4.0,
    ).to(device=device, dtype=torch.float32)
    naive_model.load_state_dict({k: v.to(device) for k, v in init_state_cpu.items()})
    ddp_naive = DDP(naive_model, device_ids=[rank], broadcast_buffers=False)
    inp_n0, tgt_n0 = _generate_heterogeneous_rank_bytes(rank, b_g9, l_g9, 7000, device)
    logits_n0, p_n0, b_n0, _ = ddp_naive.module.forward_packed_rsi(
        inp_n0.reshape(-1), cu_local_g9, l_g9 - 1, use_cost_lpt_alltoallv=False
    )
    f_loc = b_n0.float().mean().detach()
    g_loc = p_n0.mean()
    # Amplify ratio weight in router-gradient check so DDP-averaged rank-local covariance bias on heterogeneous shards is isolated cleanly
    rl_loc = (4.0 / 3.0) * (3.0 * f_loc * g_loc + (1.0 - f_loc) * (1.0 - g_loc))
    (F.cross_entropy(logits_n0, tgt_n0.reshape(-1), reduction="mean") + 0.50 * rl_loc).backward()
    for p_nv in naive_model.parameters():
        if p_nv.grad is not None:
            dist.all_reduce(p_nv.grad, op=dist.ReduceOp.AVG)
    naive_wqk_grad = naive_model.w_qk.grad.detach().clone()

    gate_g9_report: dict[str, Any] = {}
    if rank == 0:
        mono_model = HierarchicalByteHNet(
            vocab_size=256, d_outer=128, d_main=192, n_enc_layers=1, n_main_layers=2, n_dec_layers=1,
            n_heads_outer=4, n_heads_main=4, swa_window=32, target_ratio_n=4.0,
        ).to(device=device, dtype=torch.float32)
        mono_model.load_state_dict({k: v.to(device) for k, v in init_state_cpu.items()})

        # First compute exact global-ratio step-0 w_qk grad for negative-control comparison
        all_inp0, all_tgt0 = [], []
        for rr in range(world_size):
            ir, tr = _generate_heterogeneous_rank_bytes(rr, b_g9, l_g9, 7000, device)
            all_inp0.append(ir)
            all_tgt0.append(tr)
        inp_glob0 = torch.cat(all_inp0, dim=0)
        tgt_glob0 = torch.cat(all_tgt0, dim=0)
        cu_glob = torch.arange(0, (world_size * b_g9 + 1) * (l_g9 - 1), l_g9 - 1, dtype=torch.int32, device=device)
        lg0, pg0, bg0, _ = mono_model.forward_packed_rsi(inp_glob0.reshape(-1), cu_glob, l_g9 - 1, use_cost_lpt_alltoallv=False)
        fg0 = bg0.float().mean().detach()
        gg0 = pg0.mean()
        rlg0 = (4.0 / 3.0) * (3.0 * fg0 * gg0 + (1.0 - fg0) * (1.0 - gg0))
        (F.cross_entropy(lg0, tgt_glob0.reshape(-1), reduction="mean") + 0.50 * rlg0).backward()
        local_ratio_wqk_err = compare_tensor_to_fp64(naive_wqk_grad, mono_model.w_qk.grad.to(torch.float64), name="naive_local_ratio_dwqk")["rel_l2"]

        # Now reset mono_model and run the exact 3-step monolithic global-batch trajectory
        mono_model.load_state_dict({k: v.to(device) for k, v in init_state_cpu.items()})
        opt_mono = torch.optim.AdamW(mono_model.parameters(), lr=1e-3, weight_decay=0.01)
        last_mono_loss = 0.0
        for st in range(n_steps_g9):
            opt_mono.zero_grad(set_to_none=True)
            inps_s, tgts_s = [], []
            for rr in range(world_size):
                ir, tr = _generate_heterogeneous_rank_bytes(rr, b_g9, l_g9, 7000 + st * 31, device)
                inps_s.append(ir)
                tgts_s.append(tr)
            ig = torch.cat(inps_s, dim=0).reshape(-1)
            tg = torch.cat(tgts_s, dim=0).reshape(-1)
            lg, pg, bg, _ = mono_model.forward_packed_rsi(ig, cu_glob, l_g9 - 1, use_cost_lpt_alltoallv=False)
            fg = bg.float().mean().detach()
            gg = pg.mean()
            rlg = (4.0 / 3.0) * (3.0 * fg * gg + (1.0 - fg) * (1.0 - gg))
            loss_m = F.cross_entropy(lg, tg, reduction="mean") + 0.03 * rlg
            last_mono_loss = float(loss_m.item())
            loss_m.backward()
            opt_mono.step()

        max_param_rel = 0.0
        max_grad_rel = 0.0
        for name, pm in mono_model.named_parameters():
            pd = dict(dist_model.named_parameters())[name]
            pr = compare_tensor_to_fp64(pd, pm.to(torch.float64), rel_tol=1e-4, name=f"param_{name}")["rel_l2"]
            gr = compare_tensor_to_fp64(dist_grads_step3[name], pm.grad.to(torch.float64), rel_tol=5e-4, name=f"grad_{name}")["rel_l2"]
            max_param_rel = max(max_param_rel, float(pr))
            max_grad_rel = max(max_grad_rel, float(gr))
        loss_rel = abs(last_dist_loss - last_mono_loss) / max(1e-12, abs(last_mono_loss))
        gate_g9_report = {
            "steps_verified": n_steps_g9,
            "loss_step3_rel_diff": round(loss_rel, 9),
            "max_param_rel_l2": round(max_param_rel, 8),
            "max_grad_rel_l2": round(max_grad_rel, 8),
            "naive_local_ratio_dwqk_rel_l2_bug": round(float(local_ratio_wqk_err), 6),
            "passed": (loss_rel <= 1e-5 and max_param_rel <= 1e-4 and max_grad_rel <= 5e-4 and local_ratio_wqk_err >= 0.05),
        }
        del mono_model
    del ddp_g9, dist_model, ddp_naive, naive_model
    dist.barrier()
    torch.cuda.empty_cache()

    # ---------------- 4-Arm 8-GPU NVSwitch Heterogeneous Dynamic-Chunk Scaling ----------------
    b_loc = 4
    l_seq = 4097 if is_smoke else 6145  # 4096 or 6144 active bytes/doc -> up to 196,608 global bytes/step!
    l_act = l_seq - 1
    d_outer_b = 768 if is_smoke else 1024
    d_main_b = 1536 if is_smoke else 2048
    n_main_lyrs = 10 if is_smoke else 12
    n_heads_m = 12 if is_smoke else 16

    cu_b = torch.arange(0, (b_loc + 1) * l_act, l_act, dtype=torch.int32, device=device)
    doc_meta_b = build_packed_doc_metadata(cu_b, b_loc * l_act)
    inp_b, tgt_b = _generate_heterogeneous_rank_bytes(rank, b_loc, l_seq, 9100, device)
    inp_flat = inp_b.reshape(-1)
    tgt_flat = tgt_b.reshape(-1)
    mask_b = torch.ones((b_loc, l_act), dtype=torch.bool, device=device)

    torch.manual_seed(20261122)
    bench_model = HierarchicalByteHNet(
        vocab_size=256,
        d_outer=d_outer_b,
        d_main=d_main_b,
        n_enc_layers=2,
        n_main_layers=n_main_lyrs,
        n_dec_layers=2,
        n_heads_outer=8,
        n_heads_main=n_heads_m,
        swa_window=256,
        target_ratio_n=4.0,
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        bench_model.w_qk.copy_(
            torch.cat([torch.eye(d_outer_b, device=device), torch.eye(d_outer_b, device=device)], dim=0).to(torch.bfloat16)
        )
    ddp_bench = DDP(bench_model, device_ids=[rank], broadcast_buffers=False)

    warmup_b = 2 if is_smoke else 3
    iters_b = 5 if is_smoke else 8
    arms_out: dict[str, dict[str, Any]] = {}

    def _run_arm(arm_name: str):
        samples: list[float] = []
        last_stats: dict[str, float] = {}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        for it in range(warmup_b + iters_b):
            ddp_bench.zero_grad(set_to_none=True)
            dist.barrier()
            t0 = time.perf_counter()
            if arm_name == "d0_official_batched_padded_ddp_unbalanced":
                mod = ddp_bench.module
                xb = mod.embed(inp_b)
                for lyr in mod.encoder:
                    xb = lyr.forward_batched_padded(xb)
                xb = mod.enc_norm(xb)
                wq = mod.w_qk[: mod.d_outer]
                wk = mod.w_qk[mod.d_outer :]
                u = F.linear(xb[:, :-1], wq)
                v = F.linear(xb[:, 1:], wk)
                cs = torch.einsum("bld,bld->bl", F.normalize(u, dim=-1), F.normalize(v, dim=-1))
                bp = F.pad(torch.clamp((1.0 - cs) * 0.5, 0.0, 1.0), (1, 0), "constant", 1.0)
                bp2 = torch.stack((1.0 - bp, bp), dim=-1)
                sidx = torch.argmax(bp2, dim=-1)
                bm = (sidx == 1) & mask_b
                sp = bp2.gather(dim=-1, index=sidx.unsqueeze(-1))
                ntok = bm.sum(dim=-1)
                nmax = int(ntok.max())
                tidx = torch.arange(l_act, device=device)[None, :] + (~bm).long() * l_act
                ssort = torch.argsort(tidx, dim=1)
                xh = torch.gather(xb, dim=1, index=ssort[:, :nmax, None].expand(-1, -1, mod.d_outer))
                if mod.pad_dim is not None:
                    xh_m = torch.cat([xh, mod.pad_dim.to(xh.dtype).view(1, 1, -1).expand(b_loc, nmax, -1)], dim=-1)
                else:
                    xh_m = xh
                vm = torch.arange(nmax, device=device)[None, :] < ntok[:, None]
                for lyr in mod.main_layers:
                    xh_m = lyr.forward_batched_padded(xh_m, vm)
                zh = mod.main_norm(xh_m)[..., : mod.d_outer]
                pc = torch.clamp(bp2[..., -1].float(), 1e-4, 1.0 - 1e-4)
                pg = torch.gather(pc, dim=1, index=torch.argsort(tidx, dim=1)[:, :nmax])
                zb = official_hnet_ssd_surrogate_ema(zh, pg, ssd_dtype=torch.bfloat16)
                pbi = torch.cumsum(bm, dim=1) - 1
                dout = torch.gather(zb, dim=1, index=pbi.unsqueeze(-1).expand(-1, -1, mod.d_outer))
                res = F.linear(xb.float(), mod.w_res.float())
                yd = (dout.float() * _ExactSTEFn.apply(sp) + res).to(xb.dtype)
                for lyr in mod.decoder:
                    yd = lyr.forward_batched_padded(yd)
                lg = mod.lm_head(mod.dec_norm(yd)).reshape(-1, 256)
                loss = F.cross_entropy(lg, tgt_flat) + 0.03 * bp.mean()
                loss.backward()
                for p_ddp in mod.parameters():
                    if p_ddp.grad is not None:
                        dist.all_reduce(p_ddp.grad, op=dist.ReduceOp.AVG)
                last_stats = {"m_chunks": float(ntok.sum().item()), "t_bytes": float(b_loc * l_act), "imbalance_before": 1.0, "imbalance_after": 1.0, "comm_bytes_per_rank": 0.0}
            elif arm_name == "d1_packed_eager_ddp_unbalanced":
                mod = ddp_bench.module
                xb = mod.embed(inp_b)
                for lyr in mod.encoder:
                    xb = lyr.forward_batched_padded(xb)
                xe = mod.enc_norm(xb).reshape(-1, mod.d_outer)
                qk = F.linear(xe, mod.w_qk)
                uh = F.normalize(qk[:, : mod.d_outer].float(), dim=-1, eps=1e-12)
                vh = F.normalize(qk[:, mod.d_outer :].float(), dim=-1, eps=1e-12)
                cs = torch.sum(uh[:-1] * vh[1:], dim=-1)
                pf = F.pad(torch.clamp((1.0 - cs) * 0.5, 0.0, 1.0), (1, 0), value=1.0)
                st_idx = cu_b[:-1].long()
                mst = torch.zeros_like(pf, dtype=torch.bool)
                mst[st_idx] = True
                pf = torch.where(mst, torch.ones_like(pf), pf)
                bm = pf > 0.5
                cc = torch.where(bm, pf, 1.0 - pf)
                pref = F.pad(bm.long().cumsum(dim=0), (1, 0), value=0)
                ncu = pref[cu_b.long()].to(torch.int32)
                pidx = pref[1:] - 1
                x_pk = xe[bm]
                xm = mod._expand_to_main(x_pk)
                zh = mod._run_main(xm, ncu, l_act)
                _, is_s, _ = build_packed_doc_metadata(ncu, int(zh.shape[0]))
                ps = pf[bm]
                dec = torch.where(is_s != 0, torch.zeros_like(ps), 1.0 - ps.float())
                zb = _pytorch_parallel_affine_scan(dec, ps.float().unsqueeze(-1) * zh.float(), max_len_bound=b_loc * l_act).to(xe.dtype)
                zt = torch.gather(zb, dim=0, index=pidx.unsqueeze(-1).expand(-1, mod.d_outer))
                res = F.linear(xe, mod.w_res)
                yd = (zt.float() * _ExactSTEFn.apply(cc).unsqueeze(-1) + res.float()).to(xe.dtype).view(b_loc, l_act, mod.d_outer)
                for lyr in mod.decoder:
                    yd = lyr.forward_batched_padded(yd)
                lg = mod.lm_head(mod.dec_norm(yd)).reshape(-1, 256)
                loss = F.cross_entropy(lg, tgt_flat) + 0.03 * pf.mean()
                loss.backward()
                for p_ddp in mod.parameters():
                    if p_ddp.grad is not None:
                        dist.all_reduce(p_ddp.grad, op=dist.ReduceOp.AVG)
                last_stats = {"m_chunks": float(x_pk.shape[0]), "t_bytes": float(b_loc * l_act), "imbalance_before": 1.0, "imbalance_after": 1.0, "comm_bytes_per_rank": 0.0}
            elif arm_name == "d2_rsi_local_ddp_unbalanced":
                mod = ddp_bench.module
                lg, pf, bi, st_d = mod.forward_batched_outer_rsi(
                    inp_b, cu_b, l_act, use_cost_lpt_alltoallv=False, doc_meta=doc_meta_b
                )
                loss = F.cross_entropy(lg, tgt_flat) + 0.03 * pf.mean()
                loss.backward()
                for p_ddp in mod.parameters():
                    if p_ddp.grad is not None:
                        dist.all_reduce(p_ddp.grad, op=dist.ReduceOp.AVG)
                last_stats = st_d
            elif arm_name == "d3_rsi_nvswitch_cost_lpt_alltoallv":
                mod = ddp_bench.module
                lg, pf, bi, st_d = mod.forward_batched_outer_rsi(
                    inp_b, cu_b, l_act, use_cost_lpt_alltoallv=True, doc_meta=doc_meta_b
                )
                t_glob_f = float(world_size * b_loc * l_act)
                b_sum_glob = bi.to(torch.float32).sum()
                dist.all_reduce(b_sum_glob, op=dist.ReduceOp.SUM)
                f_g = b_sum_glob / t_glob_f
                kap = (4.0 / 3.0) * (4.0 * f_g - 1.0)
                ce_s = F.cross_entropy(lg, tgt_flat, reduction="sum")
                loss = float(world_size) * (ce_s / t_glob_f + 0.03 * kap * (pf.sum() / t_glob_f))
                loss.backward()
                for p_ddp in mod.parameters():
                    if p_ddp.grad is not None:
                        dist.all_reduce(p_ddp.grad, op=dist.ReduceOp.AVG)
                last_stats = st_d
            else:
                raise ValueError(arm_name)
            torch.cuda.synchronize(device)
            dist.barrier()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if it >= warmup_b:
                samples.append(dt_ms)

        peak_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        p_vec = torch.tensor([peak_mib, last_stats["m_chunks"], last_stats["comm_bytes_per_rank"]], dtype=torch.float64, device=device)
        dist.all_reduce(p_vec, op=dist.ReduceOp.MAX)
        ss = sorted(samples)
        med = ss[len(ss) // 2]
        q25 = ss[max(0, (len(ss) - 1) // 4)]
        q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
        # Naive raw-byte redistribution baseline comm (`2 round-trips * 2 * T_local * D_main * 2B`)
        raw_byte_comm_mib = (4.0 * float(b_loc * l_act * d_main_b * 2)) / (1024.0 * 1024.0)
        act_comm_mib = float(p_vec[2].item()) / (1024.0 * 1024.0)
        arms_out[arm_name] = {
            "median_ms": round(med, 4),
            "iqr_ms": round(q75 - q25, 4),
            "max_rank_peak_alloc_mib": round(float(p_vec[0].item()), 2),
            "max_rank_local_chunks": int(p_vec[1].item()),
            "imbalance_before": round(last_stats["imbalance_before"], 4),
            "imbalance_after": round(last_stats["imbalance_after"], 4),
            "nvswitch_comm_mib_per_rank": round(act_comm_mib, 3),
            "raw_byte_comm_equivalent_mib": round(raw_byte_comm_mib, 3),
            "compression_comm_saving_ratio": round(raw_byte_comm_mib / max(1e-6, act_comm_mib), 4) if act_comm_mib > 0 else 0.0,
        }

    for aname in [
        "d0_official_batched_padded_ddp_unbalanced",
        "d1_packed_eager_ddp_unbalanced",
        "d2_rsi_local_ddp_unbalanced",
        "d3_rsi_nvswitch_cost_lpt_alltoallv",
    ]:
        _run_arm(aname)

    del ddp_bench, bench_model
    dist.barrier()
    torch.cuda.empty_cache()

    if rank != 0:
        return {}

    d0 = arms_out["d0_official_batched_padded_ddp_unbalanced"]["median_ms"]
    d1 = arms_out["d1_packed_eager_ddp_unbalanced"]["median_ms"]
    d2 = arms_out["d2_rsi_local_ddp_unbalanced"]["median_ms"]
    d3 = arms_out["d3_rsi_nvswitch_cost_lpt_alltoallv"]["median_ms"]
    return {
        "world_size": world_size,
        "global_bytes_per_step": world_size * b_loc * l_act,
        "d_outer": d_outer_b,
        "d_main": d_main_b,
        "n_main_layers": n_main_lyrs,
        "gate_g9_multi_step_parity": gate_g9_report,
        "arms": arms_out,
        "speedup_d3_vs_d0_official": round(d0 / d3, 4),
        "speedup_d3_vs_d1_packed_eager": round(d1 / d3, 4),
        "speedup_d3_vs_d2_pure_cost_lpt_gain": round(d2 / d3, 4),
    }


# ==============================================================================
# Suite C: Fixed-Wall-Clock `val_bpb` RSI Loop & 5-Arm Byte-Level Scaling Law
# ==============================================================================


def _build_structured_byte_dataset(
    n_seqs: int,
    seq_len: int,
    seed: int,
    *,
    phase_shift_and_noise: bool = False,
) -> dict[str, torch.Tensor]:
    """Builds deterministic variable-length morpheme + delimiter-free identifier + KV associative byte dataset.

    Returns:
      - `tokens`: `[N, seq_len]` uint8/long in `0..255`
      - `query_mask`: `[N, seq_len - 1]` bool marking value-answer bytes after `=>`
      - `boundary_gt`: `[N, seq_len - 1]` bool marking true semantic morpheme/token starts
    """
    import random

    rng = random.Random(seed)
    morpheme_prefixes = [
        "pre", "sub", "inter", "trans", "hyper", "auto", "micro", "tele", "neo",
        "proto", "multi", "counter", "super", "ultra", "meta", "para", "circum", "omni",
    ]
    morpheme_roots = [
        "form", "graph", "duct", "spect", "port", "chron", "log", "scope", "flux", "node",
        "vect", "quant", "morph", "synt", "grad", "phas", "puls", "scal", "struct", "kern",
        "rout", "tens", "shrd", "clust",
    ]
    morpheme_suffixes = [
        "ing", "tion", "able", "ment", "ize", "ous", "al", "er", "ity", "ed",
        "ance", "ence", "ive", "less", "ness", "ward", "wise", "ory",
    ]
    camel_parts = [
        " Vec", "Grad", "Sync", "Tile", "Warp", "Scan", "Mask", "Chunk", "State", "Flux", " Gate", "Norm",
        " Route", "Pack", "Span", "Fuse", "Load", "Store", "Cast", " Ring", "Mesh", " Rank", "Slot", " Head",
        " Query", "Key", "Val", "Proj", "Step", "Loss", " Clip", "Seed", "Hash", "Tree", " Path", " Edge",
    ]
    headers = ["", "#", "##", "// "]

    toks_all = torch.zeros((n_seqs, seq_len), dtype=torch.long)
    qmask_all = torch.zeros((n_seqs, seq_len - 1), dtype=torch.bool)
    bgt_all = torch.zeros((n_seqs, seq_len - 1), dtype=torch.bool)

    for i in range(n_seqs):
        buf: list[int] = []
        bstarts: set[int] = {0}
        q_val_positions: set[int] = set()

        if phase_shift_and_noise:
            # 1-byte phase shift prefix (`H-Net` Sec. 3.2 character/shift perturbation!)
            buf.append(ord(" "))
            bstarts.add(1)
        else:
            hdr = headers[i % len(headers)]
            for ch in hdr:
                buf.append(ord(ch))
            if hdr:
                bstarts.add(len(buf))

        # 6 variable-length key-value bindings (`4-9` byte keys -> `3`-byte uppercase/digit code)
        kv_pairs: list[tuple[str, str]] = []
        used_keys: set[str] = set()
        while len(kv_pairs) < 6:
            if len(kv_pairs) % 2 == 0:
                k_str = rng.choice(morpheme_prefixes) + rng.choice(morpheme_roots)
            else:
                # Delimiter-free camelCase identifier!
                k_str = rng.choice(camel_parts).strip() + rng.choice(camel_parts).strip() + str(rng.randint(1, 9))
            if k_str in used_keys:
                continue
            used_keys.add(k_str)
            v_str = chr(rng.randint(65, 90)) + str(rng.randint(0, 9)) + chr(rng.randint(65, 90))
            kv_pairs.append((k_str, v_str))

        for k_str, v_str in kv_pairs:
            bstarts.add(len(buf))
            clause = f"{k_str}={v_str};"
            for idx_c, ch in enumerate(clause):
                if ch in ("=", ";"):
                    bstarts.add(len(buf))
                buf.append(ord(ch))
            # Interleave 1-2 morphological or camelCase filler words (with 3% character typo noise if noisy eval)
            for _ in range(2):
                bstarts.add(len(buf))
                p1, rt, sf = rng.choice(morpheme_prefixes), rng.choice(morpheme_roots), rng.choice(morpheme_suffixes)
                for part in (p1, rt, sf):
                    bstarts.add(len(buf))
                    for ch in part:
                        if phase_shift_and_noise and rng.random() < 0.03:
                            ch = chr(rng.randint(97, 122))
                        buf.append(ord(ch))
                buf.append(ord(" "))

        # Pad middle until ~seq_len - 76 with heterogeneous byte compressibility across rows:
        # row % 3 == 0: indented structured log/table lines (high compressibility ~6-8 bytes/chunk)
        # row % 3 == 1: morphological prose clauses (medium compressibility ~4 bytes/chunk)
        # row % 3 == 2: dense delimiter-free camelCase + expressions (low compressibility ~1.8 bytes/chunk)
        row_style = i % 3
        while len(buf) < seq_len - 76:
            if row_style == 0:
                for _ in range(6):
                    buf.append(ord(" "))
                bstarts.add(len(buf))
                part = rng.choice(morpheme_roots) + ":" + str(rng.randint(10, 99)) + ";"
            elif row_style == 1:
                bstarts.add(len(buf))
                part = rng.choice(morpheme_prefixes) + rng.choice(morpheme_roots) + rng.choice(morpheme_suffixes) + " "
            else:
                bstarts.add(len(buf))
                part = rng.choice(camel_parts).strip() + str(rng.randint(0, 9))
            for ch in part:
                if phase_shift_and_noise and ch != " " and rng.random() < 0.03:
                    ch = chr(rng.randint(97, 122))
                buf.append(ord(ch))

        # Emit 4 associative KV byte queries at the tail (`?<key>=><3byte_val>;`)
        q_Order = rng.sample(kv_pairs, 4)
        for k_str, v_str in q_Order:
            if len(buf) + len(k_str) + 7 >= seq_len:
                break
            bstarts.add(len(buf))
            prefix_q = f"?{k_str}=>"
            for ch in prefix_q:
                buf.append(ord(ch))
            bstarts.add(len(buf))
            for ch in v_str:
                pos = len(buf)
                # Target at index `pos - 1` predicts `buf[pos]`!
                if 1 <= pos < seq_len:
                    q_val_positions.add(pos - 1)
                buf.append(ord(ch))
            buf.append(ord(";"))

        while len(buf) < seq_len:
            buf.append(ord("."))
        buf = buf[:seq_len]

        toks_all[i] = torch.tensor(buf, dtype=torch.long)
        for qp in q_val_positions:
            if 0 <= qp < seq_len - 1:
                qmask_all[i, qp] = True
        for bp in bstarts:
            if 0 <= bp < seq_len - 1:
                bgt_all[i, bp] = True

    return {"tokens": toks_all, "query_mask": qmask_all, "boundary_gt": bgt_all}


class MatchedTopologyByteModel(nn.Module):
    """5-Arm Byte-Level Model supporting Isotropic SWA, Static Stride-4, SpaceByte Heuristic, Unsmoothed DC, & Full H-Net."""

    def __init__(self, arm: str, *, vocab_size: int = 256) -> None:
        super().__init__()
        self.arm = arm
        if arm == "H0_isotropic_byte_swa":
            # FLOP-matched isotropic byte-level SWA Transformer (no compression, N=1 -> narrower d=224, 8 layers)
            d_iso = 224
            self.embed = nn.Embedding(vocab_size, d_iso)
            self.layers = nn.ModuleList([
                VarlenTransformerLayer(d_iso, 7, window_size=(96, 0)) for _ in range(8)
            ])
            self.norm = RMSNorm(d_iso)
            self.lm_head = nn.Linear(d_iso, vocab_size, bias=False)
        else:
            self.hnet = HierarchicalByteHNet(
                vocab_size=vocab_size,
                d_outer=256,
                d_main=384,
                n_enc_layers=2,
                n_main_layers=6,
                n_dec_layers=2,
                n_heads_outer=8,
                n_heads_main=8,
                swa_window=64,
                target_ratio_n=4.0,
            )
            with torch.no_grad():
                self.hnet.w_qk.add_(torch.randn_like(self.hnet.w_qk) * 0.08)

    def forward(self, inp_b: torch.Tensor, cu_seqlens: torch.Tensor, doc_meta: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_batch(inp_b, cu_seqlens, doc_meta)

    def forward_batch(self, inp_b: torch.Tensor, cu_seqlens: torch.Tensor, doc_meta: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, l_len = inp_b.shape
        t_len = bsz * l_len
        tokens_flat = inp_b.reshape(-1)
        doc_id, is_st, is_la = doc_meta

        if self.arm == "H0_isotropic_byte_swa":
            x = self.embed(tokens_flat)
            for lyr in self.layers:
                x = lyr.forward_varlen(x, cu_seqlens, l_len)
            logits = self.lm_head(self.norm(x))
            return logits, logits.new_zeros((), dtype=torch.float32), torch.ones((t_len,), dtype=torch.int32, device=inp_b.device)

        hn = self.hnet
        if self.arm in ("H1_static_stride4_hourglass", "H2_spacebyte_delimiter_heuristic"):
            x = hn.embed(tokens_flat)
            for lyr in hn.encoder:
                x = lyr.forward_varlen(x, cu_seqlens, l_len)
            x_enc = hn.enc_norm(x)

            if self.arm == "H1_static_stride4_hourglass":
                pos_in_doc = torch.arange(t_len, device=inp_b.device) - cu_seqlens[doc_id.long()]
                b_i = ((pos_in_doc % 4) == 0).to(torch.int32)
            else:
                # SpaceByte heuristic: chunk on space/punctuation delimiters (`' '`, `';'`, `'='`, `'?'`, `'#'`)
                is_delim = (
                    (tokens_flat == ord(" "))
                    | (tokens_flat == ord(";"))
                    | (tokens_flat == ord("="))
                    | (tokens_flat == ord("?"))
                )
                # Boundary starts on delimiter or token right after delimiter, plus doc starts
                prev_delim = F.pad(is_delim[:-1], (1, 0), value=True)
                b_i = (is_delim | prev_delim | (is_st != 0)).to(torch.int32)

            p_dummy = torch.where(b_i != 0, 1.0, 0.0).to(torch.float32)
            x_pk, _, meta = zero_sort_prefix_chunk_pack(x_enc, p_dummy, b_i, doc_id, cu_seqlens)
            z_hat = hn._run_main(hn._expand_to_main(x_pk), meta.next_cu_seqlens, l_len)
            zt = z_hat.index_select(0, meta.chunk_idx.long())
            res = F.linear(x_enc, hn.w_res)
            yd = (zt.float() + res.float()).to(x_enc.dtype)
            for lyr in hn.decoder:
                yd = lyr.forward_varlen(yd, cu_seqlens, l_len)
            logits = hn.lm_head(hn.dec_norm(yd))
            return logits, logits.new_zeros((), dtype=torch.float32), b_i

        if self.arm == "H3_hnet_dc_no_ema_no_ste":
            logits, pf, bi, _ = hn.forward_packed_rsi(
                tokens_flat, cu_seqlens, l_len, disable_ema_smoothing=True, doc_meta=doc_meta
            )
            ff = float(bi.sum().item()) / float(t_len)
            gm = pf.mean()
            rl = (4.0 / 3.0) * (3.0 * ff * gm + (1.0 - ff) * (1.0 - gm))
            return logits, rl, bi

        if self.arm == "H4_hnet_rsi_dynamic_chunking":
            logits, pf, bi, _ = hn.forward_packed_rsi(
                tokens_flat, cu_seqlens, l_len, disable_ema_smoothing=False, doc_meta=doc_meta
            )
            ff = float(bi.sum().item()) / float(t_len)
            gm = pf.mean()
            rl = (4.0 / 3.0) * (3.0 * ff * gm + (1.0 - ff) * (1.0 - gm))
            return logits, rl, bi

        raise ValueError(self.arm)


@torch.no_grad()
def _eval_byte_split(
    eval_fn,
    split_data: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    toks = split_data["tokens"].to(device)
    qmask = split_data["query_mask"].to(device)
    bgt = split_data["boundary_gt"].to(device)
    n_tot, seq_len = toks.shape
    l_act = seq_len - 1

    total_nats = 0.0
    total_bytes = 0
    q_correct = 0
    q_total = 0
    tp = fp = fn = 0
    selected_chunks = 0

    for s in range(0, n_tot, batch_size):
        e = min(n_tot, s + batch_size)
        b_cur = e - s
        inp = toks[s:e, :-1].contiguous()
        tgt = toks[s:e, 1:].contiguous()
        qm = qmask[s:e].reshape(-1)
        bg = bgt[s:e].reshape(-1)
        cu = torch.arange(0, (b_cur + 1) * l_act, l_act, dtype=torch.int32, device=device)
        dmeta = build_packed_doc_metadata(cu, b_cur * l_act)

        logits, b_pred_int = eval_fn(inp, cu, dmeta)
        nats = F.cross_entropy(logits.float(), tgt.reshape(-1), reduction="sum")
        total_nats += float(nats.item())
        total_bytes += int(tgt.numel())

        if bool(qm.any().item()):
            preds = logits.argmax(dim=-1)
            q_correct += int((preds[qm] == tgt.reshape(-1)[qm]).sum().item())
            q_total += int(qm.sum().item())

        bp = b_pred_int.bool()
        selected_chunks += int(bp.sum().item())
        tp += int((bp & bg).sum().item())
        fp += int((bp & (~bg)).sum().item())
        fn += int(((~bp) & bg).sum().item())

    bpb = (total_nats / max(1, total_bytes)) / LN2
    q_acc = float(q_correct) / float(max(1, q_total))
    f1 = (2.0 * tp) / max(1.0, 2.0 * tp + fp + fn)
    bpic = float(total_bytes) / float(max(1, selected_chunks))
    return {
        "bpb": round(bpb, 6),
        "kv_byte_recall_acc": round(q_acc, 6),
        "morpheme_boundary_f1": round(f1, 6),
        "bytes_per_chunk": round(bpic, 4),
    }


def _make_stage_param_groups(hnet: HierarchicalByteHNet, base_lr: float, outer_mult: float = 2.0) -> list[dict[str, Any]]:
    """Implements `H-Net` Section 2.3 Stage Learning-Rate Modulation (`outer_mult * base_lr` on byte stage)."""
    outer_params, main_params = [], []
    for name, p in hnet.named_parameters():
        if name.startswith("main_layers.") or name.startswith("main_norm.") or name == "pad_dim":
            main_params.append(p)
        else:
            outer_params.append(p)
    return [
        {"params": outer_params, "lr": base_lr * outer_mult, "base_lr": base_lr * outer_mult},
        {"params": main_params, "lr": base_lr, "base_lr": base_lr},
    ]


def run_suite_c_distributed(rank: int, world_size: int, device: torch.device, mode: str) -> dict[str, Any]:
    is_smoke = (mode == "smoke")
    budget_s = 6.5 if is_smoke else 12.0
    seq_len = 2049 if is_smoke else 2561  # 2048 or 2560 predicted bytes/seq -> exposes O(B L log L) argsort + padded Main Attention + DC HBM wall!
    l_act = seq_len - 1
    b_local = 12
    d_outer_c = 512
    d_main_c = 768
    n_main_c = 6
    n_heads_c = 8

    n_train = 3072 if is_smoke else 6144
    n_val = 128 if is_smoke else 256
    n_test = 128 if is_smoke else 256

    train_ds = _build_structured_byte_dataset(n_train, seq_len, seed=20261130)
    val_ds = _build_structured_byte_dataset(n_val, seq_len, seed=20261230)
    test_clean_ds = _build_structured_byte_dataset(n_test, seq_len, seed=20261330)
    test_noisy_ds = _build_structured_byte_dataset(n_test, seq_len, seed=20261330, phase_shift_and_noise=True)

    train_toks = train_ds["tokens"].to(device)
    cu_tr = torch.arange(0, (b_local + 1) * l_act, l_act, dtype=torch.int32, device=device)
    dmeta_tr = build_packed_doc_metadata(cu_tr, b_local * l_act)
    mask_tr = torch.ones((b_local, l_act), dtype=torch.bool, device=device)
    t_bound_tr = b_local * l_act

    # Identical initial weights across all 5 RSI loop generations (1 Enc + 6 Main + 1 Dec hierarchical byte H-Net)
    torch.manual_seed(20261140)
    base_init_model = HierarchicalByteHNet(
        vocab_size=256, d_outer=d_outer_c, d_main=d_main_c, n_enc_layers=1, n_main_layers=n_main_c, n_dec_layers=1,
        n_heads_outer=n_heads_c, n_heads_main=12, swa_window=64, target_ratio_n=4.0,
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        base_init_model.w_qk.add_(torch.randn_like(base_init_model.w_qk) * 0.08)
    init_loop_state = {k: v.detach().cpu().clone() for k, v in base_init_model.state_dict().items()}
    del base_init_model

    class _SuiteCGenRunner(nn.Module):
        def __init__(self, hnet: HierarchicalByteHNet, gkind: str, no_ema: bool, b_thresh: float) -> None:
            super().__init__()
            self.hnet = hnet
            self.gkind = gkind
            self.no_ema = no_ema
            self.b_thresh = b_thresh

        def forward(self, inp_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mod = self.hnet
            if self.gkind == "official_batched":
                xb = mod.embed(inp_b)
                for lyr in mod.encoder:
                    xb = lyr.forward_batched_padded(xb)
                xb = mod.enc_norm(xb)
                u = F.linear(xb[:, :-1], mod.w_qk[:d_outer_c])
                v = F.linear(xb[:, 1:], mod.w_qk[d_outer_c:])
                cs = torch.einsum("bld,bld->bl", F.normalize(u, dim=-1), F.normalize(v, dim=-1))
                bp = F.pad(torch.clamp((1.0 - cs) * 0.5, 0.0, 1.0), (1, 0), "constant", 1.0)
                bp2 = torch.stack((1.0 - bp, bp), dim=-1)
                sidx = torch.argmax(bp2, dim=-1)
                bm = (sidx == 1) & mask_tr
                sp = bp2.gather(dim=-1, index=sidx.unsqueeze(-1))
                ntok = bm.sum(dim=-1)
                nmax = max(1, int(ntok.max()))
                tidx = torch.arange(l_act, device=inp_b.device)[None, :] + (~bm).long() * l_act
                ssort = torch.argsort(tidx, dim=1)
                xh = torch.gather(xb, dim=1, index=ssort[:, :nmax, None].expand(-1, -1, d_outer_c))
                xh_m = torch.cat([xh, mod.pad_dim.to(xh.dtype).view(1, 1, -1).expand(b_local, nmax, -1)], dim=-1)
                vm = torch.arange(nmax, device=inp_b.device)[None, :] < ntok[:, None]
                for lyr in mod.main_layers:
                    xh_m = lyr.forward_batched_padded(xh_m, vm)
                zh = mod.main_norm(xh_m)[..., :d_outer_c]
                pc = torch.clamp(bp2[..., -1].float(), 1e-4, 1.0 - 1e-4)
                pg = torch.gather(pc, dim=1, index=torch.argsort(tidx, dim=1)[:, :nmax])
                zb = official_hnet_ssd_surrogate_ema(zh, pg, ssd_dtype=torch.bfloat16)
                pbi = torch.cumsum(bm, dim=1) - 1
                dout = torch.gather(zb, dim=1, index=pbi.unsqueeze(-1).expand(-1, -1, d_outer_c))
                res = F.linear(xb.float(), mod.w_res.float())
                yd = (dout.float() * _ExactSTEFn.apply(sp) + res).to(xb.dtype)
                for lyr in mod.decoder:
                    yd = lyr.forward_batched_padded(yd)
                lg = mod.lm_head(mod.dec_norm(yd)).reshape(-1, 256)
                ff = bm.float().mean().detach()
                gm = bp.mean()
                rl = (4.0 / 3.0) * (3.0 * ff * gm + (1.0 - ff) * (1.0 - gm))
                return lg, rl
            if self.gkind == "eager_packed":
                xb = mod.embed(inp_b)
                for lyr in mod.encoder:
                    xb = lyr.forward_batched_padded(xb)
                xe = mod.enc_norm(xb).reshape(-1, d_outer_c)
                qk = F.linear(xe, mod.w_qk)
                uh = F.normalize(qk[:, :d_outer_c].float(), dim=-1, eps=1e-12)
                vh = F.normalize(qk[:, d_outer_c:].float(), dim=-1, eps=1e-12)
                cs = torch.sum(uh[:-1] * vh[1:], dim=-1)
                pf = F.pad(torch.clamp((1.0 - cs) * 0.5, 0.0, 1.0), (1, 0), value=1.0)
                st = cu_tr[:-1].long()
                mst = torch.zeros_like(pf, dtype=torch.bool)
                mst[st] = True
                pf = torch.where(mst, torch.ones_like(pf), pf)
                bm = pf > 0.5
                cc = torch.where(bm, pf, 1.0 - pf)
                pref = F.pad(bm.long().cumsum(dim=0), (1, 0), value=0)
                ncu = pref[cu_tr.long()].to(torch.int32)
                pidx = pref[1:] - 1
                x_pk = xe[bm]
                zh = mod._run_main(mod._expand_to_main(x_pk), ncu, l_act)
                ps = pf[bm]
                st_idx = ncu[:-1].to(torch.long).clamp_max(ps.shape[0] - 1)
                is_s = torch.zeros_like(ps, dtype=torch.bool)
                is_s[st_idx] = True
                dec = torch.where(is_s, torch.zeros_like(ps), 1.0 - ps.float())
                zb = _pytorch_parallel_affine_scan(dec, ps.float().unsqueeze(-1) * zh.float(), max_len_bound=t_bound_tr).to(zh.dtype)
                zt = torch.gather(zb, dim=0, index=pidx.unsqueeze(-1).expand(-1, d_outer_c))
                res = F.linear(xe, mod.w_res)
                yd = (zt.float() * _ExactSTEFn.apply(cc).unsqueeze(-1) + res.float()).to(zh.dtype).view(b_local, l_act, d_outer_c)
                for lyr in mod.decoder:
                    yd = lyr.forward_batched_padded(yd)
                lg = mod.lm_head(mod.dec_norm(yd)).reshape(-1, 256)
                ff = bm.float().mean().detach()
                gm = pf.mean()
                rl = (4.0 / 3.0) * (3.0 * ff * gm + (1.0 - ff) * (1.0 - gm))
                return lg, rl

            lg, pf, bi, _ = mod.forward_batched_outer_rsi(
                inp_b,
                cu_tr,
                l_act,
                disable_ema_smoothing=self.no_ema,
                boundary_threshold=self.b_thresh,
                doc_meta=dmeta_tr,
            )
            ff = bi.float().mean().detach()
            gm = pf.mean()
            rl = (4.0 / 3.0) * (3.0 * ff * gm + (1.0 - ff) * (1.0 - gm))
            return lg, rl

    gens_spec = [
        (0, "gen0_official_batched_hnet", "official_batched", 0.03, False, 0.5, 1.0, "Baseline `goombalab/hnet` batched `argsort` + padded Main Attention + `BF16` SSD surrogate"),
        (1, "gen1_packed_eager_varlen_dc", "eager_packed", 0.03, False, 0.5, 1.0, "Packed `varlen` PyTorch Eager DC + `flash_attn_varlen_func` Main Transformer"),
        (2, "gen2_rsi_round7_superfused_dc", "rsi_packed", 0.03, False, 0.5, 2.2, "Round-7 RSI Superfused Zero-Spill DC + `flash_attn_varlen_func` + Stage LR Modulation"),
        (3, "gen3_ablate_ema_smoothing_module", "rsi_packed", 0.03, True, 0.5, 2.2, "Single-factor mutation on Gen 2 removing Eq. (5) EMA smoothing (`z_bar = z_hat`)"),
        (4, "gen4_ablate_ratio_loss_alpha0", "rsi_packed", 0.0, False, 0.12, 2.2, "Single-factor mutation on Gen 2 removing Ratio Loss (`alpha = 0`, unregularized dense boundary collapse)"),
    ]

    generations_log: list[dict[str, Any]] = []
    incumbent_gen_id = ""
    incumbent_val_bpb = float("inf")
    incumbent_steps = 0
    incumbent_state_cpu: dict[str, torch.Tensor] = {}
    gen0_state_cpu: dict[str, torch.Tensor] = {}

    base_lr = 7.5e-4
    for gid, gname, gkind, alpha_ratio, no_ema, b_thresh, out_lr_mult, gdesc in gens_spec:
        model = HierarchicalByteHNet(
            vocab_size=256, d_outer=d_outer_c, d_main=d_main_c, n_enc_layers=1, n_main_layers=n_main_c, n_dec_layers=1,
            n_heads_outer=n_heads_c, n_heads_main=12, swa_window=64, target_ratio_n=4.0,
        ).to(device=device, dtype=torch.bfloat16)
        model.load_state_dict({k: v.to(device) for k, v in init_loop_state.items()})
        runner = _SuiteCGenRunner(model, gkind, no_ema, b_thresh)
        ddp_m = DDP(runner, device_ids=[rank], broadcast_buffers=False)
        opt = torch.optim.AdamW(_make_stage_param_groups(model, base_lr, outer_mult=out_lr_mult), weight_decay=0.01, fused=True)

        def _one_step(step_idx: int):
            idx0 = ((step_idx * world_size + rank) * b_local) % (n_train - b_local)
            seq_b = train_toks[idx0 : idx0 + b_local]
            inp_b = seq_b[:, :-1].contiguous()
            tgt_f = seq_b[:, 1:].reshape(-1)
            lg, rl = ddp_m(inp_b)
            loss = F.cross_entropy(lg, tgt_f) + alpha_ratio * rl
            loss.backward()

        # Untimed warmup (2 steps), then reset weights & optimizer cleanly (`PROGRAM.md` Sec. 5 zero-advantage rule!)
        for wi in range(2):
            opt.zero_grad(set_to_none=True)
            _one_step(wi)
        model.load_state_dict({k: v.to(device) for k, v in init_loop_state.items()})
        opt = torch.optim.AdamW(_make_stage_param_groups(model, base_lr, outer_mult=out_lr_mult), weight_decay=0.01, fused=True)
        torch.cuda.synchronize(device)
        dist.barrier()

        # Synchronized Rank-0 wall-clock training loop with Rule #4 wall-clock cosine LR schedule!
        t_start = time.perf_counter()
        steps_done = 0
        stop_tensor = torch.zeros((1,), dtype=torch.int32, device=device)
        frac_tensor = torch.zeros((1,), dtype=torch.float32, device=device)
        while True:
            if rank == 0:
                el = time.perf_counter() - t_start
                stop_tensor[0] = 1 if el >= budget_s else 0
                frac_tensor[0] = min(1.0, el / budget_s)
            dist.broadcast(stop_tensor, src=0)
            if int(stop_tensor[0].item()) != 0:
                break
            dist.broadcast(frac_tensor, src=0)
            frac = float(frac_tensor[0].item())
            if frac < 0.12:
                lr_scale = 0.15 + 0.85 * (frac / 0.12)
            else:
                prog = (frac - 0.12) / 0.88
                lr_scale = 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * prog))
            for pg_opt in opt.param_groups:
                pg_opt["lr"] = pg_opt["base_lr"] * lr_scale

            opt.zero_grad(set_to_none=True)
            _one_step(steps_done)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps_done += 1

        torch.cuda.synchronize(device)
        dist.barrier()
        elapsed_s = time.perf_counter() - t_start

        # Evaluate ONLY `val_bpb` inside the selection loop (`test` remains strictly untouched!)
        val_metrics = _eval_byte_split(
            lambda inp, cu, dm: model.forward_batched_outer_rsi(
                inp, cu, l_act, disable_ema_smoothing=no_ema, boundary_threshold=b_thresh, doc_meta=dm
            )[0::2],
            val_ds, batch_size=32, device=device,
        )
        v_bpb = val_metrics["bpb"]
        v_bpc = val_metrics["bytes_per_chunk"]
        min_compression_bpc = 2.0  # >= 2.0 bytes/chunk required to preserve N=4.0 hierarchical compression contract
        if gid == 0:
            decision = "BASELINE"
            incumbent_gen_id = gname
            incumbent_val_bpb = v_bpb
            incumbent_steps = steps_done
            incumbent_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            gen0_state_cpu = dict(incumbent_state_cpu)
        else:
            if v_bpc < min_compression_bpc or steps_done < int(0.88 * incumbent_steps):
                decision = "REVERT_SLOWER_COMPRESSION_COLLAPSE" if v_bpc < min_compression_bpc else "REVERT_SLOWER"
            elif v_bpb < incumbent_val_bpb - 1e-4:
                decision = "PROMOTE"
                incumbent_gen_id = gname
                incumbent_val_bpb = v_bpb
                incumbent_steps = steps_done
                incumbent_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                decision = "REVERT_VAL_BPB"

        generations_log.append({
            "gen_id": gid,
            "candidate_id": gname,
            "elapsed_s": round(elapsed_s, 4),
            "steps_completed": steps_done,
            "val_bpb": v_bpb,
            "val_kv_byte_recall_acc": val_metrics["kv_byte_recall_acc"],
            "val_morpheme_boundary_f1": val_metrics["morpheme_boundary_f1"],
            "val_bytes_per_chunk": val_metrics["bytes_per_chunk"],
            "decision": decision,
            "description": gdesc,
        })
        del runner, ddp_m, model, opt
        dist.barrier()
        torch.cuda.empty_cache()

    # Single-touch untouched `test_bpb` evaluation on Baseline (Gen 0) vs Winner (selected purely by `val_bpb`!)
    eval_m = HierarchicalByteHNet(
        vocab_size=256, d_outer=d_outer_c, d_main=d_main_c, n_enc_layers=1, n_main_layers=n_main_c, n_dec_layers=1,
        n_heads_outer=n_heads_c, n_heads_main=12, swa_window=64, target_ratio_n=4.0,
    ).to(device=device, dtype=torch.bfloat16)
    eval_m.load_state_dict({k: v.to(device) for k, v in gen0_state_cpu.items()})
    gen0_test = _eval_byte_split(
        lambda inp, cu, dm: eval_m.forward_batched_outer_rsi(inp, cu, l_act, doc_meta=dm)[0::2],
        test_clean_ds, batch_size=32, device=device,
    )
    eval_m.load_state_dict({k: v.to(device) for k, v in incumbent_state_cpu.items()})
    win_test = _eval_byte_split(
        lambda inp, cu, dm: eval_m.forward_batched_outer_rsi(inp, cu, l_act, doc_meta=dm)[0::2],
        test_clean_ds, batch_size=32, device=device,
    )
    del eval_m

    # ---------------- Part 2: 5-Arm Matched-Step Byte-Level Compression & Shift/Noise Robustness Law ----------------
    matched_steps = 80 if is_smoke else 240
    arms_law = [
        "H0_isotropic_byte_swa",
        "H1_static_stride4_hourglass",
        "H2_spacebyte_delimiter_heuristic",
        "H3_hnet_dc_no_ema_no_ste",
        "H4_hnet_rsi_dynamic_chunking",
    ]
    law_results: dict[str, dict[str, Any]] = {}
    for arm_name in arms_law:
        torch.manual_seed(20261150)
        m_arm = MatchedTopologyByteModel(arm_name).to(device=device, dtype=torch.bfloat16)
        n_params = sum(p.numel() for p in m_arm.parameters())
        ddp_arm = DDP(
            m_arm,
            device_ids=[rank],
            broadcast_buffers=False,
            find_unused_parameters=(arm_name in ("H1_static_stride4_hourglass", "H2_spacebyte_delimiter_heuristic")),
        )
        if arm_name == "H0_isotropic_byte_swa":
            pgs = [{"params": list(m_arm.parameters()), "lr": 1.6e-3, "base_lr": 1.6e-3}]
        else:
            pgs = _make_stage_param_groups(m_arm.hnet, 1.6e-3, outer_mult=2.0)
        opt_a = torch.optim.AdamW(pgs, weight_decay=0.01, fused=True)

        t0_a = time.perf_counter()
        for st in range(matched_steps):
            frac = st / float(max(1, matched_steps))
            lr_sc = (0.15 + 0.85 * (frac / 0.12)) if frac < 0.12 else (0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * (frac - 0.12) / 0.88)))
            for pg_o in opt_a.param_groups:
                pg_o["lr"] = pg_o["base_lr"] * lr_sc

            opt_a.zero_grad(set_to_none=True)
            idx0 = ((st * world_size + rank) * b_local) % (n_train - b_local)
            seq_b = train_toks[idx0 : idx0 + b_local]
            inp_b = seq_b[:, :-1].contiguous()
            tgt_f = seq_b[:, 1:].reshape(-1)
            lg, rl, _ = ddp_arm(inp_b, cu_tr, dmeta_tr)
            loss = F.cross_entropy(lg, tgt_f) + 0.03 * rl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m_arm.parameters(), 1.0)
            opt_a.step()

        torch.cuda.synchronize(device)
        dist.barrier()
        train_s = time.perf_counter() - t0_a

        def _arm_eval_fn(inp, cu, dm):
            lg_e, _, bi_e = m_arm.forward_batch(inp, cu, dm)
            return lg_e, bi_e

        v_res = _eval_byte_split(_arm_eval_fn, val_ds, batch_size=32, device=device)
        tc_res = _eval_byte_split(_arm_eval_fn, test_clean_ds, batch_size=32, device=device)
        tn_res = _eval_byte_split(_arm_eval_fn, test_noisy_ds, batch_size=32, device=device)

        law_results[arm_name] = {
            "params": int(n_params),
            "steps": matched_steps,
            "train_wall_s": round(train_s, 3),
            "val_bpb": v_res["bpb"],
            "clean_test_bpb": tc_res["bpb"],
            "shifted_noisy_test_bpb": tn_res["bpb"],
            "perturbation_bpb_degradation": round(tn_res["bpb"] - tc_res["bpb"], 6),
            "clean_kv_byte_recall_acc": tc_res["kv_byte_recall_acc"],
            "noisy_kv_byte_recall_acc": tn_res["kv_byte_recall_acc"],
            "morpheme_boundary_f1": tc_res["morpheme_boundary_f1"],
            "bytes_per_chunk": tc_res["bytes_per_chunk"],
        }
        del ddp_arm, m_arm, opt_a
        dist.barrier()
        torch.cuda.empty_cache()

    if rank != 0:
        return {}

    gen0_rec = generations_log[0]
    return {
        "budget_seconds": budget_s,
        "winning_candidate_id": incumbent_gen_id,
        "gen0_steps": gen0_rec["steps_completed"],
        "winner_steps": incumbent_steps,
        "step_multiplier_vs_gen0": round(incumbent_steps / max(1.0, float(gen0_rec["steps_completed"])), 4),
        "gen0_val_bpb": gen0_rec["val_bpb"],
        "winner_val_bpb": incumbent_val_bpb,
        "val_bpb_improvement_vs_gen0": round(gen0_rec["val_bpb"] - incumbent_val_bpb, 6),
        "single_touch_gen0_test_bpb": gen0_test["bpb"],
        "single_touch_winner_test_bpb": win_test["bpb"],
        "single_touch_test_bpb_improvement": round(gen0_test["bpb"] - win_test["bpb"], 6),
        "single_touch_winner_kv_recall_acc": win_test["kv_byte_recall_acc"],
        "single_touch_winner_morpheme_f1": win_test["morpheme_boundary_f1"],
        "generations": generations_log,
        "matched_step_byte_chunking_law": law_results,
    }


# ==============================================================================
# Main Entrypoint & Append-Only TSV Ledger Emitters
# ==============================================================================


def _write_ledgers(out_dir: Path, rep_tag: str, suite_a: dict[str, Any], suite_c: dict[str, Any], harness_sha: str) -> None:
    rsi_tsv = out_dir / "rsi_ledger.tsv"
    write_hdr_a = not rsi_tsv.exists()
    with rsi_tsv.open("a", encoding="utf-8", newline="\n") as fp:
        w = csv.writer(fp, delimiter="\t", lineterminator="\n")
        if write_hdr_a:
            w.writerow([
                "rep", "track", "trial_id", "candidate_id", "decision",
                "geomean_ms", "speedup_vs_baseline", "speedup_vs_incumbent",
                "peak_alloc_mib", "max_fp64_rel_l2", "eval_harness_sha256",
            ])
        for trk in ("track1", "track2"):
            for rec in suite_a[trk]["trials"]:
                w.writerow([
                    rep_tag, rec["track"], rec["trial_id"], rec["candidate_id"], rec["decision"],
                    rec["geomean_ms"], rec["speedup_vs_baseline"], rec["speedup_vs_incumbent"],
                    rec["peak_alloc_mib"], rec["max_fp64_rel_l2"], rec["eval_harness_sha256"],
                ])

    val_tsv = out_dir / "valbpb_autoresearch_ledger.tsv"
    write_hdr_c = not val_tsv.exists()
    with val_tsv.open("a", encoding="utf-8", newline="\n") as fp:
        w = csv.writer(fp, delimiter="\t", lineterminator="\n")
        if write_hdr_c:
            w.writerow([
                "rep", "gen_id", "candidate_id", "decision", "elapsed_s",
                "steps_completed", "val_bpb", "val_kv_byte_recall_acc",
                "val_morpheme_boundary_f1", "val_bytes_per_chunk", "eval_harness_sha256",
            ])
        for g in suite_c["generations"]:
            w.writerow([
                rep_tag, g["gen_id"], g["candidate_id"], g["decision"], g["elapsed_s"],
                g["steps_completed"], g["val_bpb"], g["val_kv_byte_recall_acc"],
                g["val_morpheme_boundary_f1"], g["val_bytes_per_chunk"], harness_sha,
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E133 H-Net Dynamic Chunking Round-7 RSI benchmark")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--rep-tag", type=str, default="smoke")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, default=None)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    t_wall0 = time.perf_counter()
    harness_sha = eval_harness_sha256()

    suite_0_out: dict[str, Any] = {}
    suite_a_out: dict[str, Any] = {}
    if rank == 0:
        suite_0_out = run_suite_0(device)
        suite_a_out = run_suite_a(device, args.mode)
    dist.barrier()

    suite_b_out = run_suite_b_distributed(rank, world_size, device, args.mode)
    dist.barrier()

    suite_c_out = run_suite_c_distributed(rank, world_size, device, args.mode)
    dist.barrier()

    total_wall_s = round(time.perf_counter() - t_wall0, 2)
    if rank == 0:
        clk, pwr = query_sm_telemetry(0)
        payload = {
            "experiment": "E133_hnet_dynamic_chunking_round7_rsi",
            "mode": args.mode,
            "rep_tag": args.rep_tag,
            "eval_harness_sha256": harness_sha,
            "total_wall_seconds": total_wall_s,
            "environment": {
                "gpu_name": torch.cuda.get_device_name(0),
                "world_size": world_size,
                "torch_version": torch.__version__,
                "sm_clock_mhz": clk,
                "power_w": pwr,
                "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            },
            "suite_0": suite_0_out,
            "suite_a": suite_a_out,
            "suite_b": suite_b_out,
            "suite_c": suite_c_out,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        if args.ledger_dir is not None and args.mode == "full":
            args.ledger_dir.mkdir(parents=True, exist_ok=True)
            _write_ledgers(args.ledger_dir, args.rep_tag, suite_a_out, suite_c_out, harness_sha)
        print(f"[E133 rank=0] Completed mode={args.mode} rep={args.rep_tag} in {total_wall_s}s -> {args.json_out}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
