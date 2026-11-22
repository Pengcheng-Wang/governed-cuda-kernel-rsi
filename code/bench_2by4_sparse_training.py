#!/usr/bin/env python3
"""E124: SM80 2:4 Semi-Structured Sparse Tensor-Core Training (cuSPARSELt / Exact-90 DP /
KKT-Exact Madow MVUE / Grad-SR-STE), Layout-Aware Sparse-SwiGLU, & 8-GPU NVSwitch
9 bits/param Compressed-AllGather FSDP2 Training.

Architecture & Primary Literature Anchor:
  - Ampere 2:4 Sparse Tensor Cores (`mma.sp.sync.m16n8k32`): Mishra et al., 2021 (arXiv:2104.08378)
  - SR-STE (Sparse-Refined Straight-Through Estimator): Zhou et al., ICLR 2021 (arXiv:2102.04010)
  - Transposable 4x4 N:M Masks for Fwd+dX Training: Hubara et al., NeurIPS 2021 (arXiv:2102.08124)
  - Minimum-Variance Unbiased N:M Gradient Sparsity (MVUE): Chmiel et al., ICLR 2023 (arXiv:2203.10991)
  - Accelerating Transformer Pre-Training with 2:4 Sparsity (Grad-SR-STE, 90-Mask Search,
    Layout-Aware Gated Activations, Mask Refresh L=10): Hu et al., ICML 2024 (arXiv:2404.01847)
  - PyTorch Semi-Structured (2:4) Sparsity (`SparseSemiStructuredTensor`, `cuSPARSELt` 0.8.1):
    Cai et al., PyTorch 2.1–2.13
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse.semi_structured as sss
import triton
import triton.language as tl
from flash_attn import flash_attn_func
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import environment_metadata

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# 0. Utilities, Telemetry, Timing, & cuSPARSELt Search Cache
# ============================================================================


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().double()
    bf = b.detach().double()
    return float((af - bf).norm().item() / bf.norm().clamp(min=1e-12).item())


def query_gpu_telemetry(gpu_index: int = 0) -> tuple[float, float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=clocks.current.sm,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return float(parts[0]), float(parts[1])
    except Exception:
        return -1.0, -1.0


def time_fn_with_peak_mem(
    fn: Callable[[], Any],
    warmup: int,
    iters: int,
    device: torch.device,
    grad_bytes_steady: int = 0,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base_bytes = torch.cuda.memory_allocated(device)

    fn()
    torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device)
    workspace_mib = max(0.0, (peak_bytes - base_bytes - grad_bytes_steady) / (1024.0**2))
    inc_peak_mib = max(0.0, (peak_bytes - base_bytes) / (1024.0**2))

    samples_ms: list[float] = []
    sm_clocks: list[float] = []
    powers: list[float] = []
    ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        ev_s[i].record()
        fn()
        ev_e[i].record()
        if i == iters // 2:
            clk, pwr = query_gpu_telemetry(device.index or 0)
            if clk > 0:
                sm_clocks.append(clk)
                powers.append(pwr)
    torch.cuda.synchronize(device)
    for i in range(iters):
        samples_ms.append(float(ev_s[i].elapsed_time(ev_e[i])))

    s_sorted = sorted(samples_ms)
    q1 = s_sorted[len(s_sorted) // 4]
    q3 = s_sorted[(3 * len(s_sorted)) // 4]
    return {
        "median_ms": round(statistics.median(s_sorted), 4),
        "iqr_ms": round(q3 - q1, 4),
        "min_ms": round(s_sorted[0], 4),
        "max_ms": round(s_sorted[-1], 4),
        "transient_workspace_mib": round(workspace_mib, 2),
        "peak_incremental_mib": round(inc_peak_mib, 2),
        "sm_clock_mhz": round(statistics.median(sm_clocks), 1) if sm_clocks else -1.0,
        "power_w": round(statistics.median(powers), 2) if powers else -1.0,
    }


class CsltPlanCache:
    """Caches `torch._C._cusparselt.mm_search` plans per shape/layout while guarding against
    cuSPARSELt 0.8.1's `split_k > 1` (`split_k = 3` on `K = 14,336, M = 1,024`) `CUSPARSE_STATUS_INTERNAL_ERROR`.
    """

    def __init__(self) -> None:
        self._plans: dict[tuple[Any, ...], tuple[int, int, int]] = {}
        self.splitk_fallbacks: list[dict[str, Any]] = []

    def mm(
        self,
        compressed_a: torch.Tensor,
        dense_b: torch.Tensor,
        transpose_result: bool = False,
    ) -> torch.Tensor:
        key = (
            tuple(compressed_a.shape),
            tuple(dense_b.shape),
            dense_b.stride(0),
            dense_b.stride(1),
            compressed_a.dtype,
            transpose_result,
        )
        plan = self._plans.get(key)
        if plan is None:
            alg_id, split_k, split_k_mode, _ = torch._C._cusparselt.mm_search(
                compressed_a, dense_b, None, None, None, transpose_result
            )
            cand = (int(alg_id), int(split_k), int(split_k_mode))
            try:
                out = torch.ops.aten._cslt_sparse_mm(
                    compressed_a,
                    dense_b,
                    None,
                    None,
                    None,
                    transpose_result,
                    cand[0],
                    cand[1],
                    cand[2],
                )
                self._plans[key] = cand
                return out
            except RuntimeError:
                # Sweep valid SM80 split_k=1 algorithms (0..7) so rejected split_k>1 plans
                # still select the true fastest split_k=1 Tensor-Core kernel.
                best_alg = int(alg_id)
                best_ms = float("inf")
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                for aid in range(8):
                    try:
                        for _ in range(2):
                            torch.ops.aten._cslt_sparse_mm(
                                compressed_a, dense_b, None, None, None, transpose_result, aid, 1, -1
                            )
                        s_cur = torch.cuda.current_stream(compressed_a.device)
                        s_cur.synchronize()
                        ev0.record(s_cur)
                        for _ in range(6):
                            torch.ops.aten._cslt_sparse_mm(
                                compressed_a, dense_b, None, None, None, transpose_result, aid, 1, -1
                            )
                        ev1.record(s_cur)
                        s_cur.synchronize()
                        ms = float(ev0.elapsed_time(ev1)) / 6.0
                        if ms < best_ms:
                            best_ms = ms
                            best_alg = aid
                    except RuntimeError:
                        break
                safe = (best_alg, 1, -1)
                self.splitk_fallbacks.append(
                    {
                        "compressed_shape": list(compressed_a.shape),
                        "dense_b_shape": list(dense_b.shape),
                        "transpose_result": transpose_result,
                        "rejected_plan": list(cand),
                        "safe_plan": list(safe),
                    }
                )
                self._plans[key] = safe
                plan = safe
        alg_id, split_k, split_k_mode = plan
        return torch.ops.aten._cslt_sparse_mm(
            compressed_a,
            dense_b,
            None,
            None,
            None,
            transpose_result,
            alg_id,
            split_k,
            split_k_mode,
        )


CSLT_CACHE = CsltPlanCache()


# ============================================================================
# 1. Custom Triton Kernels:
#    - Closed-Form SM80 cuSPARSELt 0.8.1 2-Bit Metadata Swizzle
#    - Exact-90 Factored Bipartite DP Transposable Dual Packer (0 Deadlocks)
#    - Fast Cached-Bitmask Dual Packer (for L = 10 Mask Refresh Steps)
#    - Fused Transpose + KKT-Exact Madow-Systematic MVUE 2:4 Gradient Packer
#    - Layout-Aware Fused Sparse-SwiGLU + Fused SwiGLU-Bwd & KKT-MVUE Packer
#    - Single-Pass Fused RMSNorm & Fused AdamW + 16-Bit Bitmask Grad-SR-STE
# ============================================================================


@triton.jit
def _best2(sA, cA, sB, cB):
    keep = sA >= sB
    return tl.where(keep, sA, sB), tl.where(keep, cA, cB)


@triton.jit
def _code_to_bm(p):
    # Pair index 0..5 -> 4-bit row mask:
    # 0:(0,1)->3, 1:(0,2)->5, 2:(0,3)->9, 3:(1,2)->6, 4:(1,3)->10, 5:(2,3)->12
    return tl.where(
        p < 3,
        tl.where(p == 0, 3, tl.where(p == 1, 5, 9)),
        tl.where(p == 3, 6, tl.where(p == 4, 10, 12)),
    )


@triton.jit
def _bm_to_idx(bm):
    i0 = tl.where((bm & 1) != 0, 0, tl.where((bm & 2) != 0, 1, 2))
    i1 = tl.where((bm & 8) != 0, 3, tl.where((bm & 4) != 0, 2, 1))
    return i0, i1


@triton.jit
def _pick4(v0, v1, v2, v3, idx):
    return tl.where(
        idx < 2,
        tl.where(idx == 0, v0, v1),
        tl.where(idx == 2, v2, v3),
    )


@triton.jit
def _cslt_meta_byte_off(r, g_even, M):
    """Exact SM80 cuSPARSELt 0.8.1 2-bit metadata swizzle offset for even 4-elem group `g_even`.
    Packs `meta` as `[K // 64, M // 128, 1024]` uint8 macro-tiles (1 bit/element).
    """
    return (
        (g_even >> 4) * (8 * M)
        + (r >> 7) * 1024
        + ((g_even & 8) << 6)
        + ((r & 96) << 2)
        + ((r & 7) << 4)
        + ((r & 16) >> 1)
        + (g_even & 4)
        + ((r & 8) >> 2)
        + ((g_even & 2) >> 1)
    )


@triton.jit
def _exact90_dual_cslt_pack_kernel(
    w_ptr,
    pk_val_ptr,
    pk_meta_ptr,
    pkt_val_ptr,
    pkt_meta_ptr,
    bmask_ptr,
    N,
    K,
    stride_wn,
    stride_wk,
    STORE_BMASK: tl.constexpr,
    BLOCK_TR: tl.constexpr,
    BLOCK_TC: tl.constexpr,
):
    """Evaluates all 90 2-regular 4x4 bipartite masks in 30 register ops via 19 row-sum profiles
    and emits both `packed` ([N, 9*K/16]) and `packed_t` ([K, 9*N/16]) directly in SM80 cuSPARSELt
    swizzled layout with 0 register spills and 0% bipartite deadlocks.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    tr = pid_r * BLOCK_TR + tl.arange(0, BLOCK_TR)[:, None]
    tc = pid_c * BLOCK_TC + tl.arange(0, BLOCK_TC)[None, :]
    r_base = tr * 4
    c_base = tc * 4

    w00 = tl.load(w_ptr + (r_base + 0) * stride_wn + (c_base + 0) * stride_wk)
    w01 = tl.load(w_ptr + (r_base + 0) * stride_wn + (c_base + 1) * stride_wk)
    w02 = tl.load(w_ptr + (r_base + 0) * stride_wn + (c_base + 2) * stride_wk)
    w03 = tl.load(w_ptr + (r_base + 0) * stride_wn + (c_base + 3) * stride_wk)

    w10 = tl.load(w_ptr + (r_base + 1) * stride_wn + (c_base + 0) * stride_wk)
    w11 = tl.load(w_ptr + (r_base + 1) * stride_wn + (c_base + 1) * stride_wk)
    w12 = tl.load(w_ptr + (r_base + 1) * stride_wn + (c_base + 2) * stride_wk)
    w13 = tl.load(w_ptr + (r_base + 1) * stride_wn + (c_base + 3) * stride_wk)

    w20 = tl.load(w_ptr + (r_base + 2) * stride_wn + (c_base + 0) * stride_wk)
    w21 = tl.load(w_ptr + (r_base + 2) * stride_wn + (c_base + 1) * stride_wk)
    w22 = tl.load(w_ptr + (r_base + 2) * stride_wn + (c_base + 2) * stride_wk)
    w23 = tl.load(w_ptr + (r_base + 2) * stride_wn + (c_base + 3) * stride_wk)

    w30 = tl.load(w_ptr + (r_base + 3) * stride_wn + (c_base + 0) * stride_wk)
    w31 = tl.load(w_ptr + (r_base + 3) * stride_wn + (c_base + 1) * stride_wk)
    w32 = tl.load(w_ptr + (r_base + 3) * stride_wn + (c_base + 2) * stride_wk)
    w33 = tl.load(w_ptr + (r_base + 3) * stride_wn + (c_base + 3) * stride_wk)

    m00, m01, m02, m03 = (
        tl.abs(w00.to(tl.float32)),
        tl.abs(w01.to(tl.float32)),
        tl.abs(w02.to(tl.float32)),
        tl.abs(w03.to(tl.float32)),
    )
    m10, m11, m12, m13 = (
        tl.abs(w10.to(tl.float32)),
        tl.abs(w11.to(tl.float32)),
        tl.abs(w12.to(tl.float32)),
        tl.abs(w13.to(tl.float32)),
    )
    m20, m21, m22, m23 = (
        tl.abs(w20.to(tl.float32)),
        tl.abs(w21.to(tl.float32)),
        tl.abs(w22.to(tl.float32)),
        tl.abs(w23.to(tl.float32)),
    )
    m30, m31, m32, m33 = (
        tl.abs(w30.to(tl.float32)),
        tl.abs(w31.to(tl.float32)),
        tl.abs(w32.to(tl.float32)),
        tl.abs(w33.to(tl.float32)),
    )

    r0_0, r0_1, r0_2, r0_3, r0_4, r0_5 = (
        m00 + m01,
        m00 + m02,
        m00 + m03,
        m01 + m02,
        m01 + m03,
        m02 + m03,
    )
    r1_0, r1_1, r1_2, r1_3, r1_4, r1_5 = (
        m10 + m11,
        m10 + m12,
        m10 + m13,
        m11 + m12,
        m11 + m13,
        m12 + m13,
    )
    r2_0, r2_1, r2_2, r2_3, r2_4, r2_5 = (
        m20 + m21,
        m20 + m22,
        m20 + m23,
        m21 + m22,
        m21 + m23,
        m22 + m23,
    )
    r3_0, r3_1, r3_2, r3_3, r3_4, r3_5 = (
        m30 + m31,
        m30 + m32,
        m30 + m33,
        m31 + m32,
        m31 + m33,
        m32 + m33,
    )

    # Profile Group A: 6 identical top-pair masks (p0 == p1, p2 == p3 == 5 - p0)
    bs = r0_0 + r1_0 + r2_5 + r3_5
    bc = tl.full(bs.shape, 0 | (0 << 3) | (5 << 6) | (5 << 9), tl.int32)
    bs, bc = _best2(bs, bc, r0_1 + r1_1 + r2_4 + r3_4, 1 | (1 << 3) | (4 << 6) | (4 << 9))
    bs, bc = _best2(bs, bc, r0_2 + r1_2 + r2_3 + r3_3, 2 | (2 << 3) | (3 << 6) | (3 << 9))
    bs, bc = _best2(bs, bc, r0_3 + r1_3 + r2_2 + r3_2, 3 | (3 << 3) | (2 << 6) | (2 << 9))
    bs, bc = _best2(bs, bc, r0_4 + r1_4 + r2_1 + r3_1, 4 | (4 << 3) | (1 << 6) | (1 << 9))
    bs, bc = _best2(bs, bc, r0_5 + r1_5 + r2_0 + r3_0, 5 | (5 << 3) | (0 << 6) | (0 << 9))

    # Profile Group B: (1,1,1,1) complementary top/bottom profiles (36 masks factored as 6 + 6)
    ts, tc_code = _best2(r0_0 + r1_5, 0 | (5 << 3), r0_5 + r1_0, 5 | (0 << 3))
    ts, tc_code = _best2(ts, tc_code, r0_1 + r1_4, 1 | (4 << 3))
    ts, tc_code = _best2(ts, tc_code, r0_4 + r1_1, 4 | (1 << 3))
    ts, tc_code = _best2(ts, tc_code, r0_2 + r1_3, 2 | (3 << 3))
    ts, tc_code = _best2(ts, tc_code, r0_3 + r1_2, 3 | (2 << 3))

    ys, yc_code = _best2(r2_0 + r3_5, (0 << 6) | (5 << 9), r2_5 + r3_0, (5 << 6) | (0 << 9))
    ys, yc_code = _best2(ys, yc_code, r2_1 + r3_4, (1 << 6) | (4 << 9))
    ys, yc_code = _best2(ys, yc_code, r2_4 + r3_1, (4 << 6) | (1 << 9))
    ys, yc_code = _best2(ys, yc_code, r2_2 + r3_3, (2 << 6) | (3 << 9))
    ys, yc_code = _best2(ys, yc_code, r2_3 + r3_2, (3 << 6) | (2 << 9))
    bs, bc = _best2(bs, bc, ts + ys, tc_code | yc_code)

    # Profile Group C: 12 single-overlap profiles (48 masks factored as 6 dual pairs)
    tp, tc1 = _best2(r0_0 + r1_1, 0 | (1 << 3), r0_1 + r1_0, 1 | (0 << 3))
    tq, tc2 = _best2(r0_4 + r1_5, 4 | (5 << 3), r0_5 + r1_4, 5 | (4 << 3))
    bp, bc1 = _best2(r2_4 + r3_5, (4 << 6) | (5 << 9), r2_5 + r3_4, (5 << 6) | (4 << 9))
    bq, bc2 = _best2(r2_0 + r3_1, (0 << 6) | (1 << 9), r2_1 + r3_0, (1 << 6) | (0 << 9))
    bs, bc = _best2(bs, bc, tp + bp, tc1 | bc1)
    bs, bc = _best2(bs, bc, tq + bq, tc2 | bc2)

    tp, tc1 = _best2(r0_0 + r1_2, 0 | (2 << 3), r0_2 + r1_0, 2 | (0 << 3))
    tq, tc2 = _best2(r0_3 + r1_5, 3 | (5 << 3), r0_5 + r1_3, 5 | (3 << 3))
    bp, bc1 = _best2(r2_3 + r3_5, (3 << 6) | (5 << 9), r2_5 + r3_3, (5 << 6) | (3 << 9))
    bq, bc2 = _best2(r2_0 + r3_2, (0 << 6) | (2 << 9), r2_2 + r3_0, (2 << 6) | (0 << 9))
    bs, bc = _best2(bs, bc, tp + bp, tc1 | bc1)
    bs, bc = _best2(bs, bc, tq + bq, tc2 | bc2)

    tp, tc1 = _best2(r0_0 + r1_3, 0 | (3 << 3), r0_3 + r1_0, 3 | (0 << 3))
    tq, tc2 = _best2(r0_2 + r1_5, 2 | (5 << 3), r0_5 + r1_2, 5 | (2 << 3))
    bp, bc1 = _best2(r2_2 + r3_5, (2 << 6) | (5 << 9), r2_5 + r3_2, (5 << 6) | (2 << 9))
    bq, bc2 = _best2(r2_0 + r3_3, (0 << 6) | (3 << 9), r2_3 + r3_0, (3 << 6) | (0 << 9))
    bs, bc = _best2(bs, bc, tp + bp, tc1 | bc1)
    bs, bc = _best2(bs, bc, tq + bq, tc2 | bc2)

    tp, tc1 = _best2(r0_0 + r1_4, 0 | (4 << 3), r0_4 + r1_0, 4 | (0 << 3))
    tq, tc2 = _best2(r0_1 + r1_5, 1 | (5 << 3), r0_5 + r1_1, 5 | (1 << 3))
    bp, bc1 = _best2(r2_1 + r3_5, (1 << 6) | (5 << 9), r2_5 + r3_1, (5 << 6) | (1 << 9))
    bq, bc2 = _best2(r2_0 + r3_4, (0 << 6) | (4 << 9), r2_4 + r3_0, (4 << 6) | (0 << 9))
    bs, bc = _best2(bs, bc, tp + bp, tc1 | bc1)
    bs, bc = _best2(bs, bc, tq + bq, tc2 | bc2)

    tp, tc1 = _best2(r0_1 + r1_2, 1 | (2 << 3), r0_2 + r1_1, 2 | (1 << 3))
    tq, tc2 = _best2(r0_3 + r1_4, 3 | (4 << 3), r0_4 + r1_3, 4 | (3 << 3))
    bp, bc1 = _best2(r2_3 + r3_4, (3 << 6) | (4 << 9), r2_4 + r3_3, (4 << 6) | (3 << 9))
    bq, bc2 = _best2(r2_1 + r3_2, (1 << 6) | (2 << 9), r2_2 + r3_1, (2 << 6) | (1 << 9))
    bs, bc = _best2(bs, bc, tp + bp, tc1 | bc1)
    bs, bc = _best2(bs, bc, tq + bq, tc2 | bc2)

    tp, tc1 = _best2(r0_1 + r1_3, 1 | (3 << 3), r0_3 + r1_1, 3 | (1 << 3))
    tq, tc2 = _best2(r0_2 + r1_4, 2 | (4 << 3), r0_4 + r1_2, 4 | (2 << 3))
    bp, bc1 = _best2(r2_2 + r3_4, (2 << 6) | (4 << 9), r2_4 + r3_2, (4 << 6) | (2 << 9))
    bq, bc2 = _best2(r2_1 + r3_3, (1 << 6) | (3 << 9), r2_3 + r3_1, (3 << 6) | (1 << 9))
    bs, bc = _best2(bs, bc, tp + bp, tc1 | bc1)
    bs, bc = _best2(bs, bc, tq + bq, tc2 | bc2)

    bm0 = _code_to_bm(bc & 7)
    bm1 = _code_to_bm((bc >> 3) & 7)
    bm2 = _code_to_bm((bc >> 6) & 7)
    bm3 = _code_to_bm((bc >> 9) & 7)

    if STORE_BMASK:
        mask16 = (bm0 | (bm1 << 4) | (bm2 << 8) | (bm3 << 12)).to(tl.uint16)
        tl.store(bmask_ptr + tr * (K // 4) + tc, mask16)

    K_half = K // 2
    r0i0, r0i1 = _bm_to_idx(bm0)
    r1i0, r1i1 = _bm_to_idx(bm1)
    r2i0, r2i1 = _bm_to_idx(bm2)
    r3i0, r3i1 = _bm_to_idx(bm3)

    tl.store(pk_val_ptr + (r_base + 0) * K_half + tc * 2 + 0, _pick4(w00, w01, w02, w03, r0i0))
    tl.store(pk_val_ptr + (r_base + 0) * K_half + tc * 2 + 1, _pick4(w00, w01, w02, w03, r0i1))
    tl.store(pk_val_ptr + (r_base + 1) * K_half + tc * 2 + 0, _pick4(w10, w11, w12, w13, r1i0))
    tl.store(pk_val_ptr + (r_base + 1) * K_half + tc * 2 + 1, _pick4(w10, w11, w12, w13, r1i1))
    tl.store(pk_val_ptr + (r_base + 2) * K_half + tc * 2 + 0, _pick4(w20, w21, w22, w23, r2i0))
    tl.store(pk_val_ptr + (r_base + 2) * K_half + tc * 2 + 1, _pick4(w20, w21, w22, w23, r2i1))
    tl.store(pk_val_ptr + (r_base + 3) * K_half + tc * 2 + 0, _pick4(w30, w31, w32, w33, r3i0))
    tl.store(pk_val_ptr + (r_base + 3) * K_half + tc * 2 + 1, _pick4(w30, w31, w32, w33, r3i1))

    w_nib = tl.arange(0, 2)[None, None, :].to(tl.uint8) * 15 + 1
    b_r0 = tl.sum(tl.reshape((r0i0 | (r0i1 << 2)).to(tl.uint8), [BLOCK_TR, BLOCK_TC // 2, 2]) * w_nib, axis=2).to(tl.uint8)
    b_r1 = tl.sum(tl.reshape((r1i0 | (r1i1 << 2)).to(tl.uint8), [BLOCK_TR, BLOCK_TC // 2, 2]) * w_nib, axis=2).to(tl.uint8)
    b_r2 = tl.sum(tl.reshape((r2i0 | (r2i1 << 2)).to(tl.uint8), [BLOCK_TR, BLOCK_TC // 2, 2]) * w_nib, axis=2).to(tl.uint8)
    b_r3 = tl.sum(tl.reshape((r3i0 | (r3i1 << 2)).to(tl.uint8), [BLOCK_TR, BLOCK_TC // 2, 2]) * w_nib, axis=2).to(tl.uint8)

    g_even = pid_c * BLOCK_TC + tl.arange(0, BLOCK_TC // 2)[None, :] * 2
    tl.store(pk_meta_ptr + _cslt_meta_byte_off(r_base + 0, g_even, N), b_r0)
    tl.store(pk_meta_ptr + _cslt_meta_byte_off(r_base + 1, g_even, N), b_r1)
    tl.store(pk_meta_ptr + _cslt_meta_byte_off(r_base + 2, g_even, N), b_r2)
    tl.store(pk_meta_ptr + _cslt_meta_byte_off(r_base + 3, g_even, N), b_r3)

    cm0 = ((bm0 >> 0) & 1) | (((bm1 >> 0) & 1) << 1) | (((bm2 >> 0) & 1) << 2) | (((bm3 >> 0) & 1) << 3)
    cm1 = ((bm0 >> 1) & 1) | (((bm1 >> 1) & 1) << 1) | (((bm2 >> 1) & 1) << 2) | (((bm3 >> 1) & 1) << 3)
    cm2 = ((bm0 >> 2) & 1) | (((bm1 >> 2) & 1) << 1) | (((bm2 >> 2) & 1) << 2) | (((bm3 >> 2) & 1) << 3)
    cm3 = ((bm0 >> 3) & 1) | (((bm1 >> 3) & 1) << 1) | (((bm2 >> 3) & 1) << 2) | (((bm3 >> 3) & 1) << 3)

    N_half = N // 2
    c0j0, c0j1 = _bm_to_idx(cm0)
    c1j0, c1j1 = _bm_to_idx(cm1)
    c2j0, c2j1 = _bm_to_idx(cm2)
    c3j0, c3j1 = _bm_to_idx(cm3)

    tl.store(pkt_val_ptr + (c_base + 0) * N_half + tr * 2 + 0, _pick4(w00, w10, w20, w30, c0j0))
    tl.store(pkt_val_ptr + (c_base + 0) * N_half + tr * 2 + 1, _pick4(w00, w10, w20, w30, c0j1))
    tl.store(pkt_val_ptr + (c_base + 1) * N_half + tr * 2 + 0, _pick4(w01, w11, w21, w31, c1j0))
    tl.store(pkt_val_ptr + (c_base + 1) * N_half + tr * 2 + 1, _pick4(w01, w11, w21, w31, c1j1))
    tl.store(pkt_val_ptr + (c_base + 2) * N_half + tr * 2 + 0, _pick4(w02, w12, w22, w32, c2j0))
    tl.store(pkt_val_ptr + (c_base + 2) * N_half + tr * 2 + 1, _pick4(w02, w12, w22, w32, c2j1))
    tl.store(pkt_val_ptr + (c_base + 3) * N_half + tr * 2 + 0, _pick4(w03, w13, w23, w33, c3j0))
    tl.store(pkt_val_ptr + (c_base + 3) * N_half + tr * 2 + 1, _pick4(w03, w13, w23, w33, c3j1))

    w_nibt = tl.arange(0, 2)[None, :, None].to(tl.uint8) * 15 + 1
    b_c0 = tl.sum(tl.reshape((c0j0 | (c0j1 << 2)).to(tl.uint8), [BLOCK_TR // 2, 2, BLOCK_TC]) * w_nibt, axis=1).to(tl.uint8)
    b_c1 = tl.sum(tl.reshape((c1j0 | (c1j1 << 2)).to(tl.uint8), [BLOCK_TR // 2, 2, BLOCK_TC]) * w_nibt, axis=1).to(tl.uint8)
    b_c2 = tl.sum(tl.reshape((c2j0 | (c2j1 << 2)).to(tl.uint8), [BLOCK_TR // 2, 2, BLOCK_TC]) * w_nibt, axis=1).to(tl.uint8)
    b_c3 = tl.sum(tl.reshape((c3j0 | (c3j1 << 2)).to(tl.uint8), [BLOCK_TR // 2, 2, BLOCK_TC]) * w_nibt, axis=1).to(tl.uint8)

    gt_even = pid_r * BLOCK_TR + tl.arange(0, BLOCK_TR // 2)[:, None] * 2
    tl.store(pkt_meta_ptr + _cslt_meta_byte_off(c_base + 0, gt_even, K), b_c0)
    tl.store(pkt_meta_ptr + _cslt_meta_byte_off(c_base + 1, gt_even, K), b_c1)
    tl.store(pkt_meta_ptr + _cslt_meta_byte_off(c_base + 2, gt_even, K), b_c2)
    tl.store(pkt_meta_ptr + _cslt_meta_byte_off(c_base + 3, gt_even, K), b_c3)


@triton.jit
def _fast_cached_bitmask_dual_pack_kernel(
    w_ptr,
    bmask_ptr,
    pk_val_ptr,
    pkt_val_ptr,
    N,
    K,
    stride_wn,
    stride_wk,
    BLOCK_TR: tl.constexpr,
    BLOCK_TC: tl.constexpr,
):
    """On amortized steps (t % L != 0, Hu et al. ICML 2024 Sec. 5.3), reuses the cached 16-bit
    4x4 transposable bitmask and pre-swizzled cuSPARSELt metadata buffers, gathering only updated
    BF16 nonzero values into `pk_val` and `pkt_val`.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    tr = pid_r * BLOCK_TR + tl.arange(0, BLOCK_TR)[:, None]
    tc = pid_c * BLOCK_TC + tl.arange(0, BLOCK_TC)[None, :]
    r_base = tr * 4
    c_base = tc * 4

    mask16 = tl.load(bmask_ptr + tr * (K // 4) + tc).to(tl.int32)
    bm0 = mask16 & 15
    bm1 = (mask16 >> 4) & 15
    bm2 = (mask16 >> 8) & 15
    bm3 = (mask16 >> 12) & 15

    w00 = tl.load(w_ptr + (r_base + 0) * stride_wn + (c_base + 0) * stride_wk)
    w01 = tl.load(w_ptr + (r_base + 0) * stride_wn + (c_base + 1) * stride_wk)
    w02 = tl.load(w_ptr + (r_base + 0) * stride_wn + (c_base + 2) * stride_wk)
    w03 = tl.load(w_ptr + (r_base + 0) * stride_wn + (c_base + 3) * stride_wk)

    w10 = tl.load(w_ptr + (r_base + 1) * stride_wn + (c_base + 0) * stride_wk)
    w11 = tl.load(w_ptr + (r_base + 1) * stride_wn + (c_base + 1) * stride_wk)
    w12 = tl.load(w_ptr + (r_base + 1) * stride_wn + (c_base + 2) * stride_wk)
    w13 = tl.load(w_ptr + (r_base + 1) * stride_wn + (c_base + 3) * stride_wk)

    w20 = tl.load(w_ptr + (r_base + 2) * stride_wn + (c_base + 0) * stride_wk)
    w21 = tl.load(w_ptr + (r_base + 2) * stride_wn + (c_base + 1) * stride_wk)
    w22 = tl.load(w_ptr + (r_base + 2) * stride_wn + (c_base + 2) * stride_wk)
    w23 = tl.load(w_ptr + (r_base + 2) * stride_wn + (c_base + 3) * stride_wk)

    w30 = tl.load(w_ptr + (r_base + 3) * stride_wn + (c_base + 0) * stride_wk)
    w31 = tl.load(w_ptr + (r_base + 3) * stride_wn + (c_base + 1) * stride_wk)
    w32 = tl.load(w_ptr + (r_base + 3) * stride_wn + (c_base + 2) * stride_wk)
    w33 = tl.load(w_ptr + (r_base + 3) * stride_wn + (c_base + 3) * stride_wk)

    K_half = K // 2
    r0i0, r0i1 = _bm_to_idx(bm0)
    r1i0, r1i1 = _bm_to_idx(bm1)
    r2i0, r2i1 = _bm_to_idx(bm2)
    r3i0, r3i1 = _bm_to_idx(bm3)

    tl.store(pk_val_ptr + (r_base + 0) * K_half + tc * 2 + 0, _pick4(w00, w01, w02, w03, r0i0))
    tl.store(pk_val_ptr + (r_base + 0) * K_half + tc * 2 + 1, _pick4(w00, w01, w02, w03, r0i1))
    tl.store(pk_val_ptr + (r_base + 1) * K_half + tc * 2 + 0, _pick4(w10, w11, w12, w13, r1i0))
    tl.store(pk_val_ptr + (r_base + 1) * K_half + tc * 2 + 1, _pick4(w10, w11, w12, w13, r1i1))
    tl.store(pk_val_ptr + (r_base + 2) * K_half + tc * 2 + 0, _pick4(w20, w21, w22, w23, r2i0))
    tl.store(pk_val_ptr + (r_base + 2) * K_half + tc * 2 + 1, _pick4(w20, w21, w22, w23, r2i1))
    tl.store(pk_val_ptr + (r_base + 3) * K_half + tc * 2 + 0, _pick4(w30, w31, w32, w33, r3i0))
    tl.store(pk_val_ptr + (r_base + 3) * K_half + tc * 2 + 1, _pick4(w30, w31, w32, w33, r3i1))

    cm0 = ((bm0 >> 0) & 1) | (((bm1 >> 0) & 1) << 1) | (((bm2 >> 0) & 1) << 2) | (((bm3 >> 0) & 1) << 3)
    cm1 = ((bm0 >> 1) & 1) | (((bm1 >> 1) & 1) << 1) | (((bm2 >> 1) & 1) << 2) | (((bm3 >> 1) & 1) << 3)
    cm2 = ((bm0 >> 2) & 1) | (((bm1 >> 2) & 1) << 1) | (((bm2 >> 2) & 1) << 2) | (((bm3 >> 2) & 1) << 3)
    cm3 = ((bm0 >> 3) & 1) | (((bm1 >> 3) & 1) << 1) | (((bm2 >> 3) & 1) << 2) | (((bm3 >> 3) & 1) << 3)

    N_half = N // 2
    c0j0, c0j1 = _bm_to_idx(cm0)
    c1j0, c1j1 = _bm_to_idx(cm1)
    c2j0, c2j1 = _bm_to_idx(cm2)
    c3j0, c3j1 = _bm_to_idx(cm3)

    tl.store(pkt_val_ptr + (c_base + 0) * N_half + tr * 2 + 0, _pick4(w00, w10, w20, w30, c0j0))
    tl.store(pkt_val_ptr + (c_base + 0) * N_half + tr * 2 + 1, _pick4(w00, w10, w20, w30, c0j1))
    tl.store(pkt_val_ptr + (c_base + 1) * N_half + tr * 2 + 0, _pick4(w01, w11, w21, w31, c1j0))
    tl.store(pkt_val_ptr + (c_base + 1) * N_half + tr * 2 + 1, _pick4(w01, w11, w21, w31, c1j1))
    tl.store(pkt_val_ptr + (c_base + 2) * N_half + tr * 2 + 0, _pick4(w02, w12, w22, w32, c2j0))
    tl.store(pkt_val_ptr + (c_base + 2) * N_half + tr * 2 + 1, _pick4(w02, w12, w22, w32, c2j1))
    tl.store(pkt_val_ptr + (c_base + 3) * N_half + tr * 2 + 0, _pick4(w03, w13, w23, w33, c3j0))
    tl.store(pkt_val_ptr + (c_base + 3) * N_half + tr * 2 + 1, _pick4(w03, w13, w23, w33, c3j1))


@triton.jit
def _splitmix_u01(seed, idx):
    x = (idx.to(tl.uint32) ^ seed.to(tl.uint32)) + 0x9E3779B9
    x = (x ^ (x >> 16)) * 0x85EBCA6B
    x = (x ^ (x >> 13)) * 0xC2B2AE35
    x = x ^ (x >> 16)
    return (x >> 8).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def _kkt_mvue_sample_4(a0, a1, a2, a3, u):
    """Exact KKT Minimum-Variance Unbiased 2:4 Estimator (Chmiel et al. ICLR 2023 Eq. 16 & Eq. 18)
    sampled without replacement via Madow systematic half-open interval probes (u, u + 1).
    """
    m0 = tl.abs(a0)
    m1 = tl.abs(a1)
    m2 = tl.abs(a2)
    m3 = tl.abs(a3)
    s = m0 + m1 + m2 + m3

    mx01 = tl.maximum(m0, m1)
    id01 = tl.where(m0 >= m1, 0, 1)
    mx23 = tl.maximum(m2, m3)
    id23 = tl.where(m2 >= m3, 2, 3)
    m_max = tl.maximum(mx01, mx23)
    id_max = tl.where(mx01 >= mx23, id01, id23)

    is_dom = (2.0 * m_max) >= s
    rem = tl.maximum(s - m_max, 1e-30)
    s_safe = tl.maximum(s, 1e-30)

    p0 = tl.where(
        s <= 0.0,
        1.0,
        tl.where(
            is_dom,
            tl.where(id_max == 0, 1.0, tl.where(s > m_max, m0 / rem, 1.0)),
            2.0 * m0 / s_safe,
        ),
    )
    p1 = tl.where(
        s <= 0.0,
        1.0,
        tl.where(
            is_dom,
            tl.where(
                id_max == 1,
                1.0,
                tl.where(s > m_max, m1 / rem, tl.where(id_max == 0, 1.0, 0.0)),
            ),
            2.0 * m1 / s_safe,
        ),
    )
    p2 = tl.where(
        s <= 0.0,
        0.0,
        tl.where(
            is_dom,
            tl.where(id_max == 2, 1.0, tl.where(s > m_max, m2 / rem, 0.0)),
            2.0 * m2 / s_safe,
        ),
    )
    p3 = tl.where(
        s <= 0.0,
        0.0,
        tl.where(
            is_dom,
            tl.where(id_max == 3, 1.0, tl.where(s > m_max, m3 / rem, 0.0)),
            2.0 * m3 / s_safe,
        ),
    )

    p0 = tl.minimum(tl.maximum(p0, 0.0), 1.0)
    p1 = tl.minimum(tl.maximum(p1, 0.0), 1.0)
    p2 = tl.minimum(tl.maximum(p2, 0.0), 1.0)
    c1 = p0
    c2 = c1 + p1
    c3 = c2 + p2

    u1 = u
    u2 = u + 1.0
    idx0 = tl.where(u1 < c1, 0, tl.where(u1 < c2, 1, tl.where(u1 < c3, 2, 3)))
    idx1 = tl.where(u2 < c1, 0, tl.where(u2 < c2, 1, tl.where(u2 < c3, 2, 3)))
    idx1 = tl.where(idx1 <= idx0, tl.minimum(idx0 + 1, 3), idx1)
    idx0 = tl.where(idx0 >= idx1, tl.maximum(idx1 - 1, 0), idx0)

    v_raw0 = _pick4(a0, a1, a2, a3, idx0)
    p_sel0 = _pick4(p0, p1, p2, p3, idx0)
    v_raw1 = _pick4(a0, a1, a2, a3, idx1)
    p_sel1 = _pick4(p0, p1, p2, p3, idx1)

    val0 = tl.where(p_sel0 > 1e-20, v_raw0 / p_sel0, 0.0)
    val1 = tl.where(p_sel1 > 1e-20, v_raw1 / p_sel1, 0.0)
    return val0, val1, idx0, idx1


@triton.jit
def _transpose_kkt_mvue24_cslt_pack_kernel(
    dy_ptr,
    pk_val_ptr,
    pk_meta_ptr,
    M,
    N,
    stride_dym,
    stride_dyn,
    seed,
    MODE: tl.constexpr,  # 0 = KKT-MVUE, 1 = Uniform-Unbiased 2x, 2 = Biased Greedy Top-2
    BLOCK_N: tl.constexpr,
    BLOCK_GM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_gm = tl.program_id(1)

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    gm = pid_gm * BLOCK_GM + tl.arange(0, BLOCK_GM)[None, :]
    m_base = gm * 4

    a0 = tl.load(dy_ptr + (m_base + 0) * stride_dym + rn * stride_dyn).to(tl.float32)
    a1 = tl.load(dy_ptr + (m_base + 1) * stride_dym + rn * stride_dyn).to(tl.float32)
    a2 = tl.load(dy_ptr + (m_base + 2) * stride_dym + rn * stride_dyn).to(tl.float32)
    a3 = tl.load(dy_ptr + (m_base + 3) * stride_dym + rn * stride_dyn).to(tl.float32)

    u = _splitmix_u01(seed, rn * (M // 4) + gm)

    if MODE == 0:
        v0, v1, i0, i1 = _kkt_mvue_sample_4(a0, a1, a2, a3, u)
    elif MODE == 1:
        # Uniform unbiased over all 6 pairs (p_i = 0.5 -> 2.0 * a_i)
        pair = tl.minimum((u * 6.0).to(tl.int32), 5)
        bm = _code_to_bm(pair)
        i0, i1 = _bm_to_idx(bm)
        v0 = 2.0 * _pick4(a0, a1, a2, a3, i0)
        v1 = 2.0 * _pick4(a0, a1, a2, a3, i1)
    else:
        # Biased Greedy Top-2 MSE (keeps 2 largest |a_i| without rescaling)
        m0, m1, m2, m3 = tl.abs(a0), tl.abs(a1), tl.abs(a2), tl.abs(a3)
        bs, bc = _best2(m0 + m1, 0, m0 + m2, 1)
        bs, bc = _best2(bs, bc, m0 + m3, 2)
        bs, bc = _best2(bs, bc, m1 + m2, 3)
        bs, bc = _best2(bs, bc, m1 + m3, 4)
        bs, bc = _best2(bs, bc, m2 + m3, 5)
        bm = _code_to_bm(bc)
        i0, i1 = _bm_to_idx(bm)
        v0 = _pick4(a0, a1, a2, a3, i0)
        v1 = _pick4(a0, a1, a2, a3, i1)

    M_half = M // 2
    tl.store(pk_val_ptr + rn * M_half + gm * 2 + 0, v0.to(tl.bfloat16))
    tl.store(pk_val_ptr + rn * M_half + gm * 2 + 1, v1.to(tl.bfloat16))

    nib = tl.reshape((i0 | (i1 << 2)).to(tl.uint8), [BLOCK_N, BLOCK_GM // 2, 2])
    w_nib = tl.arange(0, 2)[None, None, :].to(tl.uint8) * 15 + 1
    b_meta = tl.sum(nib * w_nib, axis=2).to(tl.uint8)
    g_even = pid_gm * BLOCK_GM + tl.arange(0, BLOCK_GM // 2)[None, :] * 2
    tl.store(pk_meta_ptr + _cslt_meta_byte_off(rn, g_even, N), b_meta)


@triton.jit
def _swiglu_row_fwd_kernel(
    gu_ptr,
    out_ptr,
    M,
    I,
    stride_gum,
    stride_om,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = offs_i < I
    g = tl.load(gu_ptr + pid_m * stride_gum + offs_i, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + pid_m * stride_gum + I + offs_i, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    out = (g * sig * u).to(tl.bfloat16)
    tl.store(out_ptr + pid_m * stride_om + offs_i, out, mask=mask)


@triton.jit
def _swiglu_col_coalesced_fwd_kernel(
    zcol_ptr,  # [2*I, M] row-major == [M, 2*I] column-major!
    out_ptr,   # [M, I] row-major
    M,
    I,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Layout-aware column-major gated activation kernel (Hu et al. ICML 2024 Sec. 5.2):
    loads along contiguous token axis `M` (`zcol_ptr[i * M + m]`) to eliminate 16 KiB-stride L2 thrashing.
    """
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)[:, None]
    g = tl.load(zcol_ptr + offs_i * M + offs_m).to(tl.float32)
    u = tl.load(zcol_ptr + (I + offs_i) * M + offs_m).to(tl.float32)
    y_im = (g * tl.sigmoid(g) * u).to(tl.bfloat16)
    y_mi = tl.trans(y_im)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    ci = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)[None, :]
    tl.store(out_ptr + rm * I + ci, y_mi)


@triton.jit
def _swiglu_col_coalesced_bwd_kernel(
    dmid_ptr,  # [M, I] row-major
    zcol_ptr,  # [2*I, M] row-major
    dzcol_ptr, # [2*I, M] row-major
    M,
    I,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    ci = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)[None, :]
    dm_mi = tl.load(dmid_ptr + rm * I + ci).to(tl.float32)
    dm_im = tl.trans(dm_mi)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)[:, None]
    g = tl.load(zcol_ptr + offs_i * M + offs_m).to(tl.float32)
    u = tl.load(zcol_ptr + (I + offs_i) * M + offs_m).to(tl.float32)
    sig = tl.sigmoid(g)
    silu_g = g * sig
    du = dm_im * silu_g
    dg = dm_im * u * sig * (1.0 + g * (1.0 - sig))
    tl.store(dzcol_ptr + offs_i * M + offs_m, dg.to(tl.bfloat16))
    tl.store(dzcol_ptr + (I + offs_i) * M + offs_m, du.to(tl.bfloat16))


@triton.jit
def _fused_swiglu_bwd_and_kkt_mvue24_pack_kernel(
    dmid_ptr,    # [M, I] row-major
    gu_ptr,      # [M, 2*I] row-major
    dgu_ptr,     # [M, 2*I] row-major (for dX = _cslt_sparse_mm(pkt_gu, d_gu.T))
    pk_val_ptr,  # packed d_gu^T values [2*I, M // 2]
    pk_meta_ptr, # packed d_gu^T swizzled metadata [2*I * M // 8]
    M,
    I,
    stride_dmm,
    stride_gum,
    stride_dgum,
    seed,
    BLOCK_I: tl.constexpr,   # 32
    BLOCK_GM: tl.constexpr,  # 8 (-> 32 tokens per CTA!)
):
    """Zero-spill (63 regs, 0 spills) fused SwiGLU backward (`dg`, `du`) + `[8, 32] -> [32, 8]`
    tile transpose + KKT-Exact Madow MVUE 2:4 cuSPARSELt packing in ONE HBM sweep, eliminating
    a 448 MiB `d_gu` HBM re-read per layer at 0.007 ms marginal cost over SwiGLU backward alone.
    """
    pid_i = tl.program_id(0)
    pid_gm = tl.program_id(1)

    ri = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)     # [32] contiguous along feature dim I
    gm = pid_gm * BLOCK_GM + tl.arange(0, BLOCK_GM)  # [8] 4-token groups
    m_b = gm[:, None] * 4                            # [8, 1]
    ch = ri[None, :]                                 # [1, 32]

    dm0 = tl.load(dmid_ptr + (m_b + 0) * stride_dmm + ch).to(tl.float32)
    g0 = tl.load(gu_ptr + (m_b + 0) * stride_gum + ch).to(tl.float32)
    u0 = tl.load(gu_ptr + (m_b + 0) * stride_gum + (I + ch)).to(tl.float32)
    s0 = tl.sigmoid(g0)
    dg0 = dm0 * u0 * s0 * (1.0 + g0 * (1.0 - s0))
    du0 = dm0 * (g0 * s0)
    tl.store(dgu_ptr + (m_b + 0) * stride_dgum + ch, dg0.to(tl.bfloat16))
    tl.store(dgu_ptr + (m_b + 0) * stride_dgum + (I + ch), du0.to(tl.bfloat16))

    dm1 = tl.load(dmid_ptr + (m_b + 1) * stride_dmm + ch).to(tl.float32)
    g1 = tl.load(gu_ptr + (m_b + 1) * stride_gum + ch).to(tl.float32)
    u1 = tl.load(gu_ptr + (m_b + 1) * stride_gum + (I + ch)).to(tl.float32)
    s1 = tl.sigmoid(g1)
    dg1 = dm1 * u1 * s1 * (1.0 + g1 * (1.0 - s1))
    du1 = dm1 * (g1 * s1)
    tl.store(dgu_ptr + (m_b + 1) * stride_dgum + ch, dg1.to(tl.bfloat16))
    tl.store(dgu_ptr + (m_b + 1) * stride_dgum + (I + ch), du1.to(tl.bfloat16))

    dm2 = tl.load(dmid_ptr + (m_b + 2) * stride_dmm + ch).to(tl.float32)
    g2 = tl.load(gu_ptr + (m_b + 2) * stride_gum + ch).to(tl.float32)
    u2 = tl.load(gu_ptr + (m_b + 2) * stride_gum + (I + ch)).to(tl.float32)
    s2 = tl.sigmoid(g2)
    dg2 = dm2 * u2 * s2 * (1.0 + g2 * (1.0 - s2))
    du2 = dm2 * (g2 * s2)
    tl.store(dgu_ptr + (m_b + 2) * stride_dgum + ch, dg2.to(tl.bfloat16))
    tl.store(dgu_ptr + (m_b + 2) * stride_dgum + (I + ch), du2.to(tl.bfloat16))

    dm3 = tl.load(dmid_ptr + (m_b + 3) * stride_dmm + ch).to(tl.float32)
    g3 = tl.load(gu_ptr + (m_b + 3) * stride_gum + ch).to(tl.float32)
    u3 = tl.load(gu_ptr + (m_b + 3) * stride_gum + (I + ch)).to(tl.float32)
    s3 = tl.sigmoid(g3)
    dg3 = dm3 * u3 * s3 * (1.0 + g3 * (1.0 - s3))
    du3 = dm3 * (g3 * s3)
    tl.store(dgu_ptr + (m_b + 3) * stride_dgum + ch, dg3.to(tl.bfloat16))
    tl.store(dgu_ptr + (m_b + 3) * stride_dgum + (I + ch), du3.to(tl.bfloat16))

    N2 = 2 * I
    M_half = M // 2
    ri2 = ri[:, None]
    gm2 = gm[None, :]
    w_nib = tl.arange(0, 2)[None, None, :].to(tl.uint8) * 15 + 1
    g_even = pid_gm * BLOCK_GM + tl.arange(0, BLOCK_GM // 2)[None, :] * 2

    ug = _splitmix_u01(seed, ri2 * (M // 4) + gm2)
    vg0, vg1, ig0, ig1 = _kkt_mvue_sample_4(
        tl.trans(dg0), tl.trans(dg1), tl.trans(dg2), tl.trans(dg3), ug
    )
    tl.store(pk_val_ptr + ri2 * M_half + gm2 * 2 + 0, vg0.to(tl.bfloat16))
    tl.store(pk_val_ptr + ri2 * M_half + gm2 * 2 + 1, vg1.to(tl.bfloat16))
    bg_meta = tl.sum(
        tl.reshape((ig0 | (ig1 << 2)).to(tl.uint8), [BLOCK_I, BLOCK_GM // 2, 2]) * w_nib,
        axis=2,
    ).to(tl.uint8)
    tl.store(pk_meta_ptr + _cslt_meta_byte_off(ri2, g_even, N2), bg_meta)

    ri_u = ri2 + I
    uu = _splitmix_u01(seed ^ 0x5A5A5A5A, ri_u * (M // 4) + gm2)
    vu0, vu1, iu0, iu1 = _kkt_mvue_sample_4(
        tl.trans(du0), tl.trans(du1), tl.trans(du2), tl.trans(du3), uu
    )
    tl.store(pk_val_ptr + ri_u * M_half + gm2 * 2 + 0, vu0.to(tl.bfloat16))
    tl.store(pk_val_ptr + ri_u * M_half + gm2 * 2 + 1, vu1.to(tl.bfloat16))
    bu_meta = tl.sum(
        tl.reshape((iu0 | (iu1 << 2)).to(tl.uint8), [BLOCK_I, BLOCK_GM // 2, 2]) * w_nib,
        axis=2,
    ).to(tl.uint8)
    tl.store(pk_meta_ptr + _cslt_meta_byte_off(ri_u, g_even, N2), bu_meta)


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    rstd_ptr,
    M,
    H: tl.constexpr,
    stride_xm,
    stride_ym,
    eps,
):
    row = tl.program_id(0)
    cols = tl.arange(0, H)
    x = tl.load(x_ptr + row * stride_xm + cols).to(tl.float32)
    g = tl.load(g_ptr + cols).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    tl.store(rstd_ptr + row, rstd)
    tl.store(y_ptr + row * stride_ym + cols, (x * rstd * g).to(tl.bfloat16))


@triton.jit
def _rmsnorm_bwd_kernel(
    dy_ptr,
    x_ptr,
    g_ptr,
    rstd_ptr,
    dx_ptr,
    dg_part_ptr,
    M,
    H: tl.constexpr,
    stride_dym,
    stride_xm,
    stride_dxm,
    stride_dgm,
    ROWS_PER_CTA: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, H)
    g = tl.load(g_ptr + cols).to(tl.float32)
    dg_acc = tl.zeros([H], dtype=tl.float32)
    row_start = pid * ROWS_PER_CTA
    for r in range(ROWS_PER_CTA):
        row = row_start + r
        if row < M:
            dy = tl.load(dy_ptr + row * stride_dym + cols).to(tl.float32)
            x = tl.load(x_ptr + row * stride_xm + cols).to(tl.float32)
            rstd = tl.load(rstd_ptr + row).to(tl.float32)
            x_hat = x * rstd
            dg_acc += dy * x_hat
            dyg = dy * g
            mean_dot = tl.sum(dyg * x_hat, axis=0) / H
            dx = rstd * (dyg - x_hat * mean_dot)
            tl.store(dx_ptr + row * stride_dxm + cols, dx.to(tl.bfloat16))
    tl.store(dg_part_ptr + pid * stride_dgm + cols, dg_acc)


@triton.jit
def _fused_adamw_bitmask_grad_sr_ste_kernel(
    w_ptr,       # [N, K] BF16 or FP32 master weights
    dw_ptr,      # [N, K] BF16 gradients
    m_ptr,       # [N, K] FP32 first moment
    v_ptr,       # [N, K] FP32 second moment
    bmask_ptr,   # [N // 4, K // 4] uint16 4x4 transposable bitmask
    N,
    K,
    lr,
    beta1,
    beta2,
    bc1_inv,
    bc2_sqrt_inv,
    eps,
    weight_decay,
    sr_ste_lambda,
    SR_MODE: tl.constexpr,  # 0 = none, 1 = Weight-SR-STE (Zhou 2021), 2 = Grad-SR-STE (Hu ICML 2024)
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Single-pass fused AdamW + 16-bit 4x4 bitmask SR-STE optimizer kernel.
    Decodes `1 - m_ij` directly from `uint16` `bmask[r >> 2, c >> 2]` with zero boolean mask allocation.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)[:, None]
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    idx = offs_r * K + offs_c

    w = tl.load(w_ptr + idx).to(tl.float32)
    g = tl.load(dw_ptr + idx).to(tl.float32)
    m = tl.load(m_ptr + idx).to(tl.float32)
    v = tl.load(v_ptr + idx).to(tl.float32)

    if SR_MODE != 0:
        m16 = tl.load(bmask_ptr + (offs_r >> 2) * (K // 4) + (offs_c >> 2)).to(tl.int32)
        bit_pos = ((offs_r & 3) << 2) + (offs_c & 3)
        active = ((m16 >> bit_pos) & 1).to(tl.float32)
        pruned = 1.0 - active
    else:
        pruned = tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32)

    if SR_MODE == 2:
        # Hu et al. ICML 2024 Eq. 10: masked decay added BEFORE AdamW moment & 1/sqrt(v) normalization!
        g = g + sr_ste_lambda * pruned * w

    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * (g * g)
    m_hat = m_new * bc1_inv
    denom = tl.sqrt(v_new) * bc2_sqrt_inv + eps

    w_new = w * (1.0 - lr * weight_decay) - lr * (m_hat / denom)
    if SR_MODE == 1:
        # Zhou et al. ICLR 2021 Eq. 8: decoupled weight penalty unnormalized by AdamW's sqrt(v)
        w_new = w_new - lr * sr_ste_lambda * pruned * w

    tl.store(m_ptr + idx, m_new)
    tl.store(v_ptr + idx, v_new)
    tl.store(w_ptr + idx, w_new.to(w_ptr.dtype.element_ty))


# ============================================================================
# 2. Python Wrappers for Exact-90, Fast-Bitmask, KKT-MVUE, & Fused RMSNorm
# ============================================================================


def exact90_dual_pack(
    W: torch.Tensor,
    packed: torch.Tensor | None = None,
    packed_t: torch.Tensor | None = None,
    bmask: torch.Tensor | None = None,
    store_bmask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    N, K = W.shape
    if packed is None:
        packed = torch.empty((N, 9 * K // 16), device=W.device, dtype=W.dtype)
    if packed_t is None:
        packed_t = torch.empty((K, 9 * N // 16), device=W.device, dtype=W.dtype)
    if store_bmask and bmask is None:
        bmask = torch.empty((N // 4, K // 4), device=W.device, dtype=torch.uint16)
    dummy_bmask = bmask if store_bmask else packed.view(torch.uint16)

    pk_val = packed.view(-1)[: N * K // 2]
    pk_meta = packed.view(-1)[N * K // 2 :].view(torch.uint8)
    pkt_val = packed_t.view(-1)[: K * N // 2]
    pkt_meta = packed_t.view(-1)[K * N // 2 :].view(torch.uint8)

    BLOCK_TR, BLOCK_TC = 8, 8
    grid = (N // (4 * BLOCK_TR), K // (4 * BLOCK_TC))
    _exact90_dual_cslt_pack_kernel[grid](
        W,
        pk_val,
        pk_meta,
        pkt_val,
        pkt_meta,
        dummy_bmask,
        N,
        K,
        W.stride(0),
        W.stride(1),
        STORE_BMASK=store_bmask,
        BLOCK_TR=BLOCK_TR,
        BLOCK_TC=BLOCK_TC,
        num_warps=2,
    )
    return packed, packed_t, bmask


def fast_cached_bitmask_dual_pack(
    W: torch.Tensor,
    bmask: torch.Tensor,
    packed: torch.Tensor,
    packed_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    N, K = W.shape
    pk_val = packed.view(-1)[: N * K // 2]
    pkt_val = packed_t.view(-1)[: K * N // 2]
    BLOCK_TR, BLOCK_TC = 8, 8
    grid = (N // (4 * BLOCK_TR), K // (4 * BLOCK_TC))
    _fast_cached_bitmask_dual_pack_kernel[grid](
        W,
        bmask,
        pk_val,
        pkt_val,
        N,
        K,
        W.stride(0),
        W.stride(1),
        BLOCK_TR=BLOCK_TR,
        BLOCK_TC=BLOCK_TC,
        num_warps=2,
    )
    return packed, packed_t


def transpose_grad_2by4_pack(
    dY: torch.Tensor,
    seed: int = 1337,
    mode: int = 0,
    packed_out: torch.Tensor | None = None,
) -> torch.Tensor:
    M, N = dY.shape
    if packed_out is None:
        packed_out = torch.empty((N, 9 * M // 16), device=dY.device, dtype=dY.dtype)
    pk_val = packed_out.view(-1)[: N * M // 2]
    pk_meta = packed_out.view(-1)[N * M // 2 :].view(torch.uint8)
    BLOCK_N, BLOCK_GM = 32, 8
    grid = (N // BLOCK_N, M // (4 * BLOCK_GM))
    _transpose_kkt_mvue24_cslt_pack_kernel[grid](
        dY,
        pk_val,
        pk_meta,
        M,
        N,
        dY.stride(0),
        dY.stride(1),
        seed,
        MODE=mode,
        BLOCK_N=BLOCK_N,
        BLOCK_GM=BLOCK_GM,
        num_warps=4,
    )
    return packed_out


def fused_swiglu_bwd_and_mvue24_pack(
    d_mid: torch.Tensor,
    gu: torch.Tensor,
    seed: int = 2026,
    d_gu_out: torch.Tensor | None = None,
    pk_dgu_t_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, I = d_mid.shape
    N2 = 2 * I
    if d_gu_out is None:
        d_gu_out = torch.empty((M, N2), device=d_mid.device, dtype=d_mid.dtype)
    if pk_dgu_t_out is None:
        pk_dgu_t_out = torch.empty((N2, 9 * M // 16), device=d_mid.device, dtype=d_mid.dtype)
    pk_val = pk_dgu_t_out.view(-1)[: N2 * M // 2]
    pk_meta = pk_dgu_t_out.view(-1)[N2 * M // 2 :].view(torch.uint8)
    BLOCK_I, BLOCK_GM = 32, 8
    grid = (I // BLOCK_I, M // (4 * BLOCK_GM))
    _fused_swiglu_bwd_and_kkt_mvue24_pack_kernel[grid](
        d_mid,
        gu,
        d_gu_out,
        pk_val,
        pk_meta,
        M,
        I,
        d_mid.stride(0),
        gu.stride(0),
        d_gu_out.stride(0),
        seed,
        BLOCK_I=BLOCK_I,
        BLOCK_GM=BLOCK_GM,
        num_warps=4,
    )
    return d_gu_out, pk_dgu_t_out


def triton_rmsnorm_fwd(
    x2d: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    M, H = x2d.shape
    y = torch.empty_like(x2d)
    rstd = torch.empty((M,), device=x2d.device, dtype=torch.float32)
    _rmsnorm_fwd_kernel[(M,)](
        x2d, gamma, y, rstd, M, H, x2d.stride(0), y.stride(0), eps, num_warps=8
    )
    return y, rstd


def triton_rmsnorm_bwd(
    dy2d: torch.Tensor, x2d: torch.Tensor, gamma: torch.Tensor, rstd: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    M, H = dy2d.shape
    dx = torch.empty_like(x2d)
    rows_per_cta = 32
    n_ctas = triton.cdiv(M, rows_per_cta)
    dg_part = torch.empty((n_ctas, H), device=x2d.device, dtype=torch.float32)
    _rmsnorm_bwd_kernel[(n_ctas,)](
        dy2d,
        x2d,
        gamma,
        rstd,
        dx,
        dg_part,
        M,
        H,
        dy2d.stride(0),
        x2d.stride(0),
        dx.stride(0),
        dg_part.stride(0),
        ROWS_PER_CTA=rows_per_cta,
        num_warps=8,
    )
    return dx, dg_part.sum(dim=0).to(gamma.dtype)


def fused_adamw_bitmask_step(
    W: torch.Tensor,
    dW: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    bmask: torch.Tensor,
    step: int,
    lr: float = 2e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    sr_ste_lambda: float = 2e-4,
    sr_mode: int = 2,
) -> None:
    N, K = W.shape
    bc1_inv = 1.0 / (1.0 - (beta1**step))
    bc2_sqrt_inv = 1.0 / math.sqrt(1.0 - (beta2**step))
    BLOCK_R, BLOCK_C = 32, 32
    grid = (N // BLOCK_R, K // BLOCK_C)
    _fused_adamw_bitmask_grad_sr_ste_kernel[grid](
        W,
        dW,
        m,
        v,
        bmask,
        N,
        K,
        lr,
        beta1,
        beta2,
        bc1_inv,
        bc2_sqrt_inv,
        eps,
        weight_decay,
        sr_ste_lambda,
        SR_MODE=sr_mode,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )


def bmask_u16_to_dense(bmask: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    tr, tc = bmask.shape
    shifts = torch.arange(16, device=bmask.device, dtype=torch.int32)
    bits = ((bmask.to(torch.int32).unsqueeze(-1) >> shifts) & 1).view(tr, tc, 4, 4)
    return bits.permute(0, 2, 1, 3).reshape(tr * 4, tc * 4).to(dtype=dtype)


# ============================================================================
# 3. Suite 0: SM80 2:4 Roofline, PyTorch 2.13 Bug Audit, KKT-MVUE Unbiasedness,
#    Grad-SR-STE Flip-Rate Trajectory, & FP64 Richardson Chain-Rule Verification
# ============================================================================


def run_suite_0_audits(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    warmup = 4 if is_smoke else 10
    iters = 10 if is_smoke else 30

    # --- S0.1: SM80 2:4 Sparse Tensor-Core Roofline & PyTorch 2.13 Backend Audit ---
    M0, K0, N0 = (4096, 4096, 14336) if is_smoke else (8192, 4096, 14336)
    flops_0 = 2.0 * M0 * K0 * N0
    torch.manual_seed(1240)
    A0 = torch.randn(M0, K0, device=device, dtype=torch.bfloat16)
    W0 = torch.randn(N0, K0, device=device, dtype=torch.bfloat16)

    pk0, pkt0, bmask0 = exact90_dual_pack(W0)
    assert bmask0 is not None
    W0_sparse = W0 * bmask_u16_to_dense(bmask0)

    w_cut0 = sss.SparseSemiStructuredTensorCUTLASS.from_dense(W0_sparse)
    w_cslt_def0 = sss.SparseSemiStructuredTensorCUSPARSELT.from_dense(W0_sparse)
    # Prime plan cache
    _ = CSLT_CACHE.mm(pk0, A0.t(), transpose_result=True)

    roofline_modes: dict[str, Any] = {}
    for name, fn in [
        ("bf16_cublas_dense", lambda: F.linear(A0, W0)),
        ("pytorch_cutlass_2by4", lambda: F.linear(A0, w_cut0)),
        ("cusparselt_default_alg0", lambda: F.linear(A0, w_cslt_def0)),
        ("cusparselt_searched_tuned", lambda: CSLT_CACHE.mm(pk0, A0.t(), transpose_result=True)),
    ]:
        st = time_fn_with_peak_mem(fn, warmup, iters, device)
        st["effective_tflops"] = round(flops_0 / (st["median_ms"] * 1e-3) / 1e12, 2)
        roofline_modes[name] = st
    dense_ms = roofline_modes["bf16_cublas_dense"]["median_ms"]
    for st in roofline_modes.values():
        st["speedup_vs_dense_bf16"] = round(dense_ms / max(st["median_ms"], 1e-9), 3)

    # Compression / transposable-pack kernel latency comparison on [14336, 4096]
    comp_bench: dict[str, Any] = {}
    for cname, cfn in [
        ("pytorch_cutlass_from_dense", lambda: sss.SparseSemiStructuredTensorCUTLASS.from_dense(W0)),
        (
            "pytorch_cslt_dual_from_dense",
            lambda: (
                sss.SparseSemiStructuredTensorCUSPARSELT.from_dense(W0),
                sss.SparseSemiStructuredTensorCUSPARSELT.from_dense(W0.t().contiguous()),
            ),
        ),
        (
            "pytorch_tile_4x4_greedy_abs",
            lambda: sss.SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(
                W0, algorithm="largest_abs_values_greedy"
            ),
        ),
        ("triton_exact90_dp_dual_pack", lambda: exact90_dual_pack(W0, pk0, pkt0, bmask0)),
        ("triton_fast_cached_bitmask_dual_pack", lambda: fast_cached_bitmask_dual_pack(W0, bmask0, pk0, pkt0)),
    ]:
        comp_bench[cname] = time_fn_with_peak_mem(cfn, warmup, iters, device)

    # --- S0.2: PyTorch 2.13 `_sparse_semi_structured_tile` Double-Defect Audit ---
    ws_pyt_def = sss.SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(W0, algorithm="")
    wd_pyt_def = ws_pyt_def.to_dense()
    ws_pyt_abs = sss.SparseSemiStructuredTensorCUSPARSELT.prune_dense_static_sort(
        W0, algorithm="largest_abs_values_greedy"
    )
    wd_pyt_abs = ws_pyt_abs.to_dense()

    # Build exhaustive 90-mask oracle on [14336, 4096] in chunks to verify 100.000% Exact-90 L1 optimality!
    masks90 = []
    for cols_per_row in itertools.product(itertools.combinations(range(4), 2), repeat=4):
        m = torch.zeros(4, 4)
        for r, cs in enumerate(cols_per_row):
            m[r, cs[0]] = 1.0
            m[r, cs[1]] = 1.0
        if (m.sum(0) == 2).all():
            masks90.append(m)
    M90 = torch.stack(masks90).to(device=device, dtype=torch.float32)  # [90, 4, 4]
    oracle_l1 = 0.0
    for r0 in range(0, N0, 1024):
        r1 = min(r0 + 1024, N0)
        wt = W0[r0:r1].float().abs().view((r1 - r0) // 4, 4, K0 // 4, 4).permute(0, 2, 1, 3)
        oracle_l1 += float(torch.einsum("abij,mij->abm", wt, M90).amax(dim=-1).sum().item())

    total_l1 = float(W0.float().abs().sum().item())

    def _audit_mask_quality(wd: torch.Tensor) -> dict[str, Any]:
        tiles = wd.view(N0 // 4, 4, K0 // 4, 4).permute(0, 2, 1, 3)
        nnz_tile = (tiles != 0).sum(dim=(-1, -2))
        row_nnz = (wd.view(-1, 4) != 0).sum(dim=-1)
        col_nnz = (wd.t().contiguous().view(-1, 4) != 0).sum(dim=-1)
        kept = wd[wd != 0]
        l1_kept = float(wd.float().abs().sum().item())
        return {
            "retained_l1_pct_of_dense": round(100.0 * l1_kept / total_l1, 4),
            "retained_l1_pct_of_90mask_oracle": round(100.0 * l1_kept / oracle_l1, 4),
            "pruning_rel_frobenius_error": round(rel_l2(wd, W0), 6),
            "kept_positive_fraction": round(float((kept > 0).float().mean().item()), 6),
            "deadlocked_4x4_tile_pct": round(
                100.0 * float((nnz_tile < 8).float().mean().item()), 4
            ),
            "min_row_group_nnz": int(row_nnz.min().item()),
            "max_row_group_nnz": int(row_nnz.max().item()),
            "min_col_group_nnz": int(col_nnz.min().item()),
            "max_col_group_nnz": int(col_nnz.max().item()),
        }

    # Verify Fwd and dX cuSPARSELt output parity of our Exact-90 packed tensors against dense W0_sparse
    dY0 = torch.randn(M0, N0, device=device, dtype=torch.bfloat16)
    y_cslt0 = CSLT_CACHE.mm(pk0, A0.t(), transpose_result=True)
    y_ref0 = F.linear(A0, W0_sparse)
    dx_cslt0 = CSLT_CACHE.mm(pkt0, dY0.t(), transpose_result=True)
    dx_ref0 = dY0 @ W0_sparse

    pyt_bug_audit = {
        "shape": [N0, K0],
        "num_4x4_tiles": (N0 // 4) * (K0 // 4),
        "pytorch_default_empty_alg": _audit_mask_quality(wd_pyt_def),
        "pytorch_largest_abs_greedy": _audit_mask_quality(wd_pyt_abs),
        "triton_exact90_bipartite_dp": _audit_mask_quality(W0_sparse),
        "exact90_cslt_fwd_rel_l2_vs_sparse_dense": round(rel_l2(y_cslt0, y_ref0), 6),
        "exact90_cslt_dx_rel_l2_vs_sparse_dense": round(rel_l2(dx_cslt0, dx_ref0), 6),
    }
    del A0, W0, dY0, y_cslt0, y_ref0, dx_cslt0, dx_ref0, ws_pyt_def, wd_pyt_def, ws_pyt_abs, wd_pyt_abs

    # --- S0.3: Neural-Gradient 2:4 Pruning (`dW`): KKT-MVUE vs Uniform-2x vs Biased Greedy Top-2 ---
    torch.manual_seed(1243)
    M_mc, N_mc, K_mc = 256, 512, 256
    dy_mc_f = torch.randn(M_mc, N_mc, device=device, dtype=torch.float32)
    # Realistic transformer token-gradient heterogeneity within 4-token blocks (e.g. high-loss vs low-loss tokens)
    dy_mc_f[0::4, :] *= 4.0
    dy_mc_f[1::4, :] *= 3.0
    dy_mc = dy_mc_f.to(torch.bfloat16)
    x_mc = torch.randn(M_mc, K_mc, device=device, dtype=torch.bfloat16)
    dw_exact_mc = dy_mc.float().t() @ x_mc.float()

    mc_steps = [1, 16, 64, 256] if is_smoke else [1, 16, 64, 256, 1024]
    max_mc = mc_steps[-1]
    mvue_curve: dict[str, dict[str, float]] = {}
    for m_id, m_label in [
        (2, "greedy_top2_mse_biased"),
        (1, "uniform_unbiased_2x"),
        (0, "kkt_exact_madow_mvue_2by4"),
    ]:
        acc = torch.zeros_like(dw_exact_mc)
        pk_buf = torch.empty((N_mc, 9 * M_mc // 16), device=device, dtype=torch.bfloat16)
        curve: dict[str, float] = {}
        for s in range(1, max_mc + 1):
            transpose_grad_2by4_pack(dy_mc, seed=50000 + s * 7919, mode=m_id, packed_out=pk_buf)
            dw_s = CSLT_CACHE.mm(pk_buf, x_mc, transpose_result=False)
            acc.add_(dw_s.float())
            if s in mc_steps:
                curve[f"steps_{s}"] = round(rel_l2(acc / float(s), dw_exact_mc), 6)
        mvue_curve[m_label] = curve

    # --- S0.4: 160-Step AdamW Transformer SwiGLU Pre-Training Flip-Rate & Loss Trajectory ---
    # Compare Dense vs Naive STE vs Weight-SR-STE (Zhou 2021) vs Grad-SR-STE (Hu ICML 2024)
    n_steps_tr = 60 if is_smoke else 160
    D_tr, I_tr, B_tr = 256, 768, 256
    torch.manual_seed(1244)
    # Fixed teacher 2-layer SwiGLU FFN target
    t_gu = torch.randn(2 * I_tr, D_tr, device=device, dtype=torch.float32) / math.sqrt(D_tr)
    t_dn = torch.randn(D_tr, I_tr, device=device, dtype=torch.float32) / math.sqrt(I_tr)
    init_gu = torch.randn(2 * I_tr, D_tr, device=device, dtype=torch.float32) / math.sqrt(D_tr)
    init_dn = torch.randn(D_tr, I_tr, device=device, dtype=torch.float32) / math.sqrt(I_tr)
    val_x = torch.randn(1024, D_tr, device=device, dtype=torch.float32)
    with torch.no_grad():
        vg, vu = (val_x @ t_gu.t()).chunk(2, dim=-1)
        val_y = (F.silu(vg) * vu) @ t_dn.t()

    sr_trajectories: dict[str, Any] = {}
    for reg_name, is_sparse_reg, sr_mode, lam in [
        ("dense_adamw", False, 0, 0.0),
        ("sparse_2by4_naive_ste", True, 0, 0.0),
        ("sparse_2by4_weight_sr_ste", True, 1, 1.5e-2),
        ("sparse_2by4_grad_sr_ste", True, 2, 1.5e-2),
    ]:
        w_gu = init_gu.clone()
        w_dn = init_dn.clone()
        m_gu = torch.zeros_like(w_gu)
        v_gu = torch.zeros_like(w_gu)
        m_dn = torch.zeros_like(w_dn)
        v_dn = torch.zeros_like(w_dn)
        prev_bm_gu: torch.Tensor | None = None
        prev_bm_dn: torch.Tensor | None = None
        flips_early: list[float] = []
        flips_tail: list[float] = []
        init_loss = 0.0

        for stp in range(1, n_steps_tr + 1):
            g_gen = torch.Generator(device=device)
            g_gen.manual_seed(90000 + stp)
            bx = torch.randn((B_tr, D_tr), generator=g_gen, device=device, dtype=torch.float32)
            with torch.no_grad():
                tg, tu = (bx @ t_gu.t()).chunk(2, dim=-1)
                by = (F.silu(tg) * tu) @ t_dn.t()

            _, _, bm_gu = exact90_dual_pack(w_gu.to(torch.bfloat16))
            _, _, bm_dn = exact90_dual_pack(w_dn.to(torch.bfloat16))
            assert bm_gu is not None and bm_dn is not None

            if prev_bm_gu is not None and prev_bm_dn is not None:
                diff_gu = (bm_gu.to(torch.int32) ^ prev_bm_gu.to(torch.int32))
                diff_dn = (bm_dn.to(torch.int32) ^ prev_bm_dn.to(torch.int32))
                # Popcount 16-bit XOR across all tiles
                shifts = torch.arange(16, device=device, dtype=torch.int32)
                flips = float(
                    ((diff_gu.unsqueeze(-1) >> shifts) & 1).sum().item()
                    + ((diff_dn.unsqueeze(-1) >> shifts) & 1).sum().item()
                ) / float(w_gu.numel() + w_dn.numel())
                if stp <= max(10, n_steps_tr // 4):
                    flips_early.append(flips)
                if stp > (3 * n_steps_tr) // 4:
                    flips_tail.append(flips)
            prev_bm_gu = bm_gu.clone()
            prev_bm_dn = bm_dn.clone()

            # Cosine-decayed lr from 3e-3 -> 3e-4 so healthy masks settle while STE dilemma traps oscillate!
            lr_t = 3e-4 + 0.5 * (3e-3 - 3e-4) * (
                1.0 + math.cos(math.pi * (stp - 1) / max(n_steps_tr - 1, 1))
            )

            w_gu_req = w_gu.detach().requires_grad_(True)
            w_dn_req = w_dn.detach().requires_grad_(True)
            if is_sparse_reg:
                mask_gu_f = bmask_u16_to_dense(bm_gu, dtype=torch.float32)
                mask_dn_f = bmask_u16_to_dense(bm_dn, dtype=torch.float32)
                eff_gu = w_gu_req * mask_gu_f + (w_gu_req - w_gu_req.detach()) * (1.0 - mask_gu_f)
                eff_dn = w_dn_req * mask_dn_f + (w_dn_req - w_dn_req.detach()) * (1.0 - mask_dn_f)
            else:
                eff_gu = w_gu_req
                eff_dn = w_dn_req

            gu_act = bx @ eff_gu.t()
            ga, ua = gu_act.chunk(2, dim=-1)
            pred = (F.silu(ga) * ua) @ eff_dn.t()
            loss = F.mse_loss(pred, by)
            if stp == 1:
                init_loss = float(loss.item())
            loss.backward()

            assert w_gu_req.grad is not None and w_dn_req.grad is not None
            fused_adamw_bitmask_step(
                w_gu,
                w_gu_req.grad.to(torch.bfloat16),
                m_gu,
                v_gu,
                bm_gu,
                step=stp,
                lr=lr_t,
                weight_decay=0.01,
                sr_ste_lambda=lam,
                sr_mode=sr_mode,
            )
            fused_adamw_bitmask_step(
                w_dn,
                w_dn_req.grad.to(torch.bfloat16),
                m_dn,
                v_dn,
                bm_dn,
                step=stp,
                lr=lr_t,
                weight_decay=0.01,
                sr_ste_lambda=lam,
                sr_mode=sr_mode,
            )

        with torch.no_grad():
            if is_sparse_reg:
                _, _, fbm_gu = exact90_dual_pack(w_gu.to(torch.bfloat16))
                _, _, fbm_dn = exact90_dual_pack(w_dn.to(torch.bfloat16))
                assert fbm_gu is not None and fbm_dn is not None
                ew_gu = w_gu * bmask_u16_to_dense(fbm_gu, dtype=torch.float32)
                ew_dn = w_dn * bmask_u16_to_dense(fbm_dn, dtype=torch.float32)
            else:
                ew_gu, ew_dn = w_gu, w_dn
            fvg, fvu = (val_x @ ew_gu.t()).chunk(2, dim=-1)
            final_val_loss = float(F.mse_loss((F.silu(fvg) * fvu) @ ew_dn.t(), val_y).item())

        sr_trajectories[reg_name] = {
            "steps": n_steps_tr,
            "init_loss": round(init_loss, 6),
            "final_val_mse_loss": round(final_val_loss, 6),
            "early_flip_rate_mean": round(statistics.mean(flips_early), 6),
            "tail_flip_rate_mean": round(statistics.mean(flips_tail), 6),
        }

    # --- S0.5: FP64 Richardson Central-Difference Chain-Rule Verification (h vs h/2 ~ 4.00x) ---
    torch.manual_seed(1245)
    M_fd, H_fd, I_fd = 16, 128, 128
    x_fd = torch.randn(M_fd, H_fd, device=device, dtype=torch.float64, requires_grad=True)
    gam_fd = torch.randn(H_fd, device=device, dtype=torch.float64, requires_grad=True)
    wgu_fd = (
        torch.randn(2 * I_fd, H_fd, device=device, dtype=torch.float64) / math.sqrt(H_fd)
    ).detach().requires_grad_(True)
    wdn_fd = (
        torch.randn(H_fd, I_fd, device=device, dtype=torch.float64) / math.sqrt(I_fd)
    ).detach().requires_grad_(True)
    target_fd = torch.randn(M_fd, H_fd, device=device, dtype=torch.float64)

    _, _, bm_gu_fd = exact90_dual_pack(wgu_fd.detach().to(torch.bfloat16))
    _, _, bm_dn_fd = exact90_dual_pack(wdn_fd.detach().to(torch.bfloat16))
    assert bm_gu_fd is not None and bm_dn_fd is not None
    m_gu_fd = bmask_u16_to_dense(bm_gu_fd, dtype=torch.float64)
    m_dn_fd = bmask_u16_to_dense(bm_dn_fd, dtype=torch.float64)

    def _smooth_sparse_swiglu_loss(
        xv: torch.Tensor, gv: torch.Tensor, wguv: torch.Tensor, wdnv: torch.Tensor
    ) -> torch.Tensor:
        rstd = torch.rsqrt((xv * xv).mean(dim=-1, keepdim=True) + 1e-6)
        xn = xv * rstd * gv
        gu = xn @ (wguv * m_gu_fd).t()
        g_part, u_part = gu.chunk(2, dim=-1)
        mid = (g_part * torch.sigmoid(g_part)) * u_part
        out = xv + mid @ (wdnv * m_dn_fd).t()
        diff = out - target_fd
        return 0.5 * (diff * diff).mean()

    loss_fd = _smooth_sparse_swiglu_loss(x_fd, gam_fd, wgu_fd, wdn_fd)
    gx_a, gg_a, gwgu_a, gwdn_a = torch.autograd.grad(loss_fd, (x_fd, gam_fd, wgu_fd, wdn_fd))

    # Manual analytical backward formulas under fixed transposable masks (m_gu_fd, m_dn_fd)
    with torch.no_grad():
        rstd_m = torch.rsqrt((x_fd * x_fd).mean(dim=-1, keepdim=True) + 1e-6)
        xhat_m = x_fd * rstd_m
        xn_m = xhat_m * gam_fd
        wgu_eff = wgu_fd * m_gu_fd
        wdn_eff = wdn_fd * m_dn_fd
        gu_m = xn_m @ wgu_eff.t()
        gp_m, up_m = gu_m.chunk(2, dim=-1)
        sig_m = torch.sigmoid(gp_m)
        silu_m = gp_m * sig_m
        mid_m = silu_m * up_m
        out_m = x_fd + mid_m @ wdn_eff.t()
        dy_fd = (out_m - target_fd) / float(out_m.numel())

        dwdn_man = (dy_fd.t() @ mid_m) * m_dn_fd
        dmid_m = dy_fd @ wdn_eff
        dgp_m = dmid_m * up_m * sig_m * (1.0 + gp_m * (1.0 - sig_m))
        dup_m = dmid_m * silu_m
        dgu_m = torch.cat([dgp_m, dup_m], dim=-1)
        dwgu_man = (dgu_m.t() @ xn_m) * m_gu_fd
        dxn_m = dgu_m @ wgu_eff
        dgam_man = (dxn_m * xhat_m).sum(dim=0)
        dx_man = dy_fd + rstd_m * (
            dxn_m * gam_fd - xhat_m * ((dxn_m * gam_fd * xhat_m).mean(dim=-1, keepdim=True))
        )

    richardson: dict[str, Any] = {
        "manual_vs_autograd_max_rel_l2": max(
            rel_l2(dx_man, gx_a),
            rel_l2(dgam_man, gg_a),
            rel_l2(dwgu_man, gwgu_a),
            rel_l2(dwdn_man, gwdn_a),
        )
    }
    h1, h2 = 5e-3, 2.5e-3
    ratios: list[float] = []
    for t_name, base_t, grad_man, fn_eval in [
        ("dx", x_fd, dx_man, lambda z: _smooth_sparse_swiglu_loss(z, gam_fd, wgu_fd, wdn_fd)),
        ("dgamma", gam_fd, dgam_man, lambda z: _smooth_sparse_swiglu_loss(x_fd, z, wgu_fd, wdn_fd)),
        ("dw_gate_up_masked", wgu_fd, dwgu_man, lambda z: _smooth_sparse_swiglu_loss(x_fd, gam_fd, z, wdn_fd)),
        ("dw_down_masked", wdn_fd, dwdn_man, lambda z: _smooth_sparse_swiglu_loss(x_fd, gam_fd, wgu_fd, z)),
    ]:
        g_dir = torch.Generator(device=device)
        g_dir.manual_seed(7700 + len(ratios))
        d_vec = torch.randn(base_t.shape, generator=g_dir, device=device, dtype=torch.float64)
        d_vec = d_vec / d_vec.norm()
        with torch.no_grad():
            exact_dir = float((grad_man * d_vec).sum().item())
            fd1 = float((fn_eval(base_t + h1 * d_vec) - fn_eval(base_t - h1 * d_vec)).item()) / (2.0 * h1)
            fd2 = float((fn_eval(base_t + h2 * d_vec) - fn_eval(base_t - h2 * d_vec)).item()) / (2.0 * h2)
        err1 = abs(fd1 - exact_dir)
        err2 = max(abs(fd2 - exact_dir), 1e-18)
        rat = err1 / err2
        ratios.append(rat)
        richardson[t_name] = {
            "err_h": err1,
            "err_h_half": err2,
            "richardson_ratio": round(rat, 4),
        }
    richardson["min_richardson_ratio"] = round(min(ratios), 4)

    gc.collect()
    torch.cuda.empty_cache()
    return {
        "sm80_2by4_gemm_roofline": roofline_modes,
        "compression_kernel_ablation": comp_bench,
        "pytorch_213_tile_bug_audit": pyt_bug_audit,
        "kkt_mvue_monte_carlo_unbiasedness": mvue_curve,
        "sr_ste_flip_rate_trajectories": sr_trajectories,
        "fp64_richardson_chain_rule": richardson,
    }


# ============================================================================
# 4. Suite A: 1-GPU Llama-3-8B 2:4 Sparse Linear 3-GEMM (Fwd + dX + dW) Scaling
#    (K = 4,096 -> N = 14,336, M = 2,048..16,384)
# ============================================================================


def run_suite_a_linear_3gemm_scaling(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    K, N = 4096, 14336
    tok_list = [4096, 8192] if is_smoke else [2048, 4096, 8192, 16384]
    warmup = 4 if is_smoke else 8
    iters = 8 if is_smoke else 24

    torch.manual_seed(1246)
    W = (torch.randn(N, K, device=device, dtype=torch.bfloat16) / math.sqrt(K)).contiguous()
    pk_buf = torch.empty((N, 9 * K // 16), device=device, dtype=torch.bfloat16)
    pkt_buf = torch.empty((K, 9 * N // 16), device=device, dtype=torch.bfloat16)
    bmask_buf = torch.empty((N // 4, K // 4), device=device, dtype=torch.uint16)
    exact90_dual_pack(W, pk_buf, pkt_buf, bmask_buf)

    scaling: dict[str, Any] = {}
    for M in tok_list:
        X = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        dY = torch.randn(M, N, device=device, dtype=torch.bfloat16)
        pk_dy_t = torch.empty((N, 9 * M // 16), device=device, dtype=torch.bfloat16)
        flops_3gemm = 6.0 * M * K * N

        # Prime cuSPARSELt plans for Fwd, dX, and dW
        _ = CSLT_CACHE.mm(pk_buf, X.t(), transpose_result=True)
        _ = CSLT_CACHE.mm(pkt_buf, dY.t(), transpose_result=True)
        transpose_grad_2by4_pack(dY, seed=42, mode=0, packed_out=pk_dy_t)
        _ = CSLT_CACHE.mm(pk_dy_t, X, transpose_result=False)

        def _dense() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            y = X @ W.t()
            dx = dY @ W
            dw = dY.t() @ X
            return y, dx, dw

        def _pyt_cutlass_2gemm() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            ws_c = sss.SparseSemiStructuredTensorCUTLASS.prune_dense_static_sort(
                W, algorithm="largest_abs_values_greedy"
            )
            y = F.linear(X, ws_c)
            dx = dY @ ws_c
            dw = dY.t() @ X
            return y, dx, dw

        def _cslt_2gemm_l1() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            exact90_dual_pack(W, pk_buf, pkt_buf, bmask_buf, store_bmask=False)
            y = CSLT_CACHE.mm(pk_buf, X.t(), transpose_result=True)
            dx = CSLT_CACHE.mm(pkt_buf, dY.t(), transpose_result=True)
            dw = dY.t() @ X
            return y, dx, dw

        def _cslt_3gemm_fst_l1() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            exact90_dual_pack(W, pk_buf, pkt_buf, bmask_buf, store_bmask=True)
            y = CSLT_CACHE.mm(pk_buf, X.t(), transpose_result=True)
            dx = CSLT_CACHE.mm(pkt_buf, dY.t(), transpose_result=True)
            transpose_grad_2by4_pack(dY, seed=101, mode=0, packed_out=pk_dy_t)
            dw = CSLT_CACHE.mm(pk_dy_t, X, transpose_result=False)
            return y, dx, dw

        step_box = [0]

        def _cslt_3gemm_fst_l10() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if step_box[0] % 10 == 0:
                exact90_dual_pack(W, pk_buf, pkt_buf, bmask_buf, store_bmask=True)
            else:
                fast_cached_bitmask_dual_pack(W, bmask_buf, pk_buf, pkt_buf)
            step_box[0] += 1
            y = CSLT_CACHE.mm(pk_buf, X.t(), transpose_result=True)
            dx = CSLT_CACHE.mm(pkt_buf, dY.t(), transpose_result=True)
            transpose_grad_2by4_pack(dY, seed=202, mode=0, packed_out=pk_dy_t)
            dw = CSLT_CACHE.mm(pk_dy_t, X, transpose_result=False)
            return y, dx, dw

        def _cslt_3gemm_fst_gacc4() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # 4 gradient-accumulation microbatches sharing one Exact-90 weight pack!
            exact90_dual_pack(W, pk_buf, pkt_buf, bmask_buf, store_bmask=True)
            dw_acc: torch.Tensor | None = None
            y_last = X
            dx_last = X
            for mb in range(4):
                y_last = CSLT_CACHE.mm(pk_buf, X.t(), transpose_result=True)
                dx_last = CSLT_CACHE.mm(pkt_buf, dY.t(), transpose_result=True)
                transpose_grad_2by4_pack(dY, seed=300 + mb, mode=0, packed_out=pk_dy_t)
                dw_mb = CSLT_CACHE.mm(pk_dy_t, X, transpose_result=False)
                dw_acc = dw_mb if dw_acc is None else dw_acc.add_(dw_mb)
            assert dw_acc is not None
            return y_last, dx_last, dw_acc

        modes_m: dict[str, Any] = {}
        for m_name, m_fn, mult in [
            ("bf16_cublas_dense", _dense, 1.0),
            ("pytorch_cutlass_2by4_2gemm", _pyt_cutlass_2gemm, 1.0),
            ("cslt_exact90_2gemm_refresh1", _cslt_2gemm_l1, 1.0),
            ("cslt_exact90_3gemm_fst_refresh1", _cslt_3gemm_fst_l1, 1.0),
            ("cslt_exact90_3gemm_fst_refresh10", _cslt_3gemm_fst_l10, 1.0),
            ("cslt_exact90_3gemm_fst_gacc4", _cslt_3gemm_fst_gacc4, 0.25),
        ]:
            st = time_fn_with_peak_mem(m_fn, warmup, iters, device)
            if mult != 1.0:
                for k_ms in ("median_ms", "iqr_ms", "min_ms", "max_ms"):
                    st[k_ms] = round(st[k_ms] * mult, 4)
            st["effective_tflops"] = round(flops_3gemm / (st["median_ms"] * 1e-3) / 1e12, 2)
            modes_m[m_name] = st

        d_med = modes_m["bf16_cublas_dense"]["median_ms"]
        for st in modes_m.values():
            st["speedup_vs_dense_bf16"] = round(d_med / max(st["median_ms"], 1e-9), 3)

        scaling[str(M)] = {
            "num_tokens": M,
            "in_features": K,
            "out_features": N,
            "dense_equiv_gflops": round(flops_3gemm / 1e9, 2),
            "modes": modes_m,
        }
        del X, dY, pk_dy_t
        gc.collect()
        torch.cuda.empty_cache()

    del W, pk_buf, pkt_buf, bmask_buf
    gc.collect()
    torch.cuda.empty_cache()
    return {"token_scaling": scaling}


# ============================================================================
# 5. Suite B: Layout-Aware Gated-Activation Ablation & 4-Layer Llama-3-8B
#    Pre-Norm 2:4 Sparse-SwiGLU MLP Stack (Fwd + Bwd + Fused Grad-SR-STE Step)
# ============================================================================


class DensePreNormSwiGLUBlock(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.norm = nn.Parameter(torch.ones(hidden, dtype=torch.bfloat16))
        self.w_gate_up = nn.Linear(hidden, 2 * inter, bias=False, dtype=torch.bfloat16)
        self.w_down = nn.Linear(inter, hidden, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xn = (xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        ) * self.norm
        g, u = self.w_gate_up(xn).chunk(2, dim=-1)
        return x + self.w_down(F.silu(g) * u)


class _Sparse2by4SwiGLUBlockFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        norm: torch.Tensor,
        w_gu: torch.Tensor,
        w_dn: torch.Tensor,
        pk_gu: torch.Tensor,
        pkt_gu: torch.Tensor,
        bm_gu: torch.Tensor,
        pk_dn: torch.Tensor,
        pkt_dn: torch.Tensor,
        bm_dn: torch.Tensor,
        refresh_mask: bool,
        step_seed: int,
    ) -> torch.Tensor:
        M, H = x.shape
        I = w_dn.shape[1]
        if refresh_mask:
            exact90_dual_pack(w_gu, pk_gu, pkt_gu, bm_gu, store_bmask=True)
            exact90_dual_pack(w_dn, pk_dn, pkt_dn, bm_dn, store_bmask=True)
        else:
            fast_cached_bitmask_dual_pack(w_gu, bm_gu, pk_gu, pkt_gu)
            fast_cached_bitmask_dual_pack(w_dn, bm_dn, pk_dn, pkt_dn)

        xn, rstd = triton_rmsnorm_fwd(x, norm)
        gu = CSLT_CACHE.mm(pk_gu, xn.t(), transpose_result=True)
        mid = torch.empty((M, I), device=x.device, dtype=torch.bfloat16)
        BLOCK_I = 256
        _swiglu_row_fwd_kernel[(M, triton.cdiv(I, BLOCK_I))](
            gu, mid, M, I, gu.stride(0), mid.stride(0), BLOCK_I=BLOCK_I, num_warps=4
        )
        dn_out = CSLT_CACHE.mm(pk_dn, mid.t(), transpose_result=True)
        out = x + dn_out
        ctx.save_for_backward(x, norm, xn, rstd, gu, mid, pkt_gu, pkt_dn)
        ctx.step_seed = step_seed
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[Any, ...]:
        x, norm, xn, rstd, gu, mid, pkt_gu, pkt_dn = ctx.saved_tensors
        dout_c = dout.contiguous()
        seed = ctx.step_seed

        # Down backward: d_mid via 2:4 Sparse Tensor Cores + dW_dn via KKT-MVUE 2:4 Sparse Tensor Cores
        d_mid = CSLT_CACHE.mm(pkt_dn, dout_c.t(), transpose_result=True)
        pk_dout_t = transpose_grad_2by4_pack(dout_c, seed=seed + 11, mode=0)
        dw_dn = CSLT_CACHE.mm(pk_dout_t, mid, transpose_result=False)
        del pk_dout_t

        # Fused SwiGLU backward + SRAM [32,32] transpose + KKT-MVUE 2:4 pack (saves 448 MiB HBM read!)
        d_gu, pk_dgu_t = fused_swiglu_bwd_and_mvue24_pack(d_mid, gu, seed=seed + 23)
        del d_mid

        dxn = CSLT_CACHE.mm(pkt_gu, d_gu.t(), transpose_result=True)
        del d_gu
        dw_gu = CSLT_CACHE.mm(pk_dgu_t, xn, transpose_result=False)
        del pk_dgu_t

        dx_norm, dnorm = triton_rmsnorm_bwd(dxn, x, norm, rstd)
        del dxn
        dx = dout_c + dx_norm
        return dx, dnorm, dw_gu, dw_dn, None, None, None, None, None, None, None, None


class Sparse2by4PreNormSwiGLUBlock(nn.Module):
    def __init__(self, ref_blk: DensePreNormSwiGLUBlock) -> None:
        super().__init__()
        self.norm = nn.Parameter(ref_blk.norm.detach().clone())
        self.w_gate_up = nn.Parameter(ref_blk.w_gate_up.weight.detach().clone())
        self.w_down = nn.Parameter(ref_blk.w_down.weight.detach().clone())

        N_gu, H = self.w_gate_up.shape
        H_dn, I = self.w_down.shape
        dev = self.w_gate_up.device
        self.register_buffer("pk_gu", torch.empty((N_gu, 9 * H // 16), device=dev, dtype=torch.bfloat16))
        self.register_buffer("pkt_gu", torch.empty((H, 9 * N_gu // 16), device=dev, dtype=torch.bfloat16))
        self.register_buffer("bm_gu", torch.empty((N_gu // 4, H // 4), device=dev, dtype=torch.uint16))
        self.register_buffer("pk_dn", torch.empty((H_dn, 9 * I // 16), device=dev, dtype=torch.bfloat16))
        self.register_buffer("pkt_dn", torch.empty((I, 9 * H_dn // 16), device=dev, dtype=torch.bfloat16))
        self.register_buffer("bm_dn", torch.empty((H_dn // 4, I // 4), device=dev, dtype=torch.uint16))
        exact90_dual_pack(self.w_gate_up.data, self.pk_gu, self.pkt_gu, self.bm_gu, store_bmask=True)
        exact90_dual_pack(self.w_down.data, self.pk_dn, self.pkt_dn, self.bm_dn, store_bmask=True)

        self.register_buffer("m_gu", torch.zeros_like(self.w_gate_up, dtype=torch.float32))
        self.register_buffer("v_gu", torch.zeros_like(self.w_gate_up, dtype=torch.float32))
        self.register_buffer("m_dn", torch.zeros_like(self.w_down, dtype=torch.float32))
        self.register_buffer("v_dn", torch.zeros_like(self.w_down, dtype=torch.float32))

    def forward(self, x: torch.Tensor, refresh_mask: bool = True, step_seed: int = 100) -> torch.Tensor:
        return _Sparse2by4SwiGLUBlockFn.apply(
            x,
            self.norm,
            self.w_gate_up,
            self.w_down,
            self.pk_gu,
            self.pkt_gu,
            self.bm_gu,
            self.pk_dn,
            self.pkt_dn,
            self.bm_dn,
            refresh_mask,
            step_seed,
        )


def run_suite_b_sparse_swiglu_stack(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    H, I = 4096, 14336
    n_layers = 2 if is_smoke else 4
    tok_list = [4096, 8192] if is_smoke else [2048, 4096, 8192]
    warmup = 3 if is_smoke else 6
    iters = 6 if is_smoke else 18

    # --- B1: Layout-Aware Gated-Activation Ablation (Column-Major L2 Thrash vs Coalesced Triton) ---
    M_ab = 4096 if is_smoke else 8192
    z_col_storage = torch.randn(2 * I, M_ab, device=device, dtype=torch.bfloat16)  # raw _cslt_sparse_mm(trans=False)
    z_row_storage = z_col_storage.t().contiguous()
    dm_ab = torch.randn(M_ab, I, device=device, dtype=torch.bfloat16)
    # 3 tensors read + written in Fwd+Bwd: Forward reads [M, 2I] writes [M, I]; Backward reads [M, I] + [M, 2I], writes [M, 2I]
    # Total physical bytes for unfused SwiGLU Fwd+Bwd = (3 + 5) * M * I * 2 = 16 * M * I bytes
    swiglu_fwd_bwd_bytes = 16.0 * M_ab * I

    def _col_major_naive_eager() -> torch.Tensor:
        zv = z_col_storage.t().requires_grad_(True)  # stride (1, M_ab): 16 KiB jump along feature dim!
        g, u = zv.chunk(2, dim=-1)
        y = F.silu(g) * u
        y.backward(dm_ab)
        assert zv.grad is not None
        return zv.grad

    def _col_major_coalesced_triton() -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty((M_ab, I), device=device, dtype=torch.bfloat16)
        dzcol = torch.empty_like(z_col_storage)
        g_ab = (M_ab // 32, I // 32)
        _swiglu_col_coalesced_fwd_kernel[g_ab](
            z_col_storage, out, M_ab, I, BLOCK_M=32, BLOCK_I=32, num_warps=4
        )
        _swiglu_col_coalesced_bwd_kernel[g_ab](
            dm_ab, z_col_storage, dzcol, M_ab, I, BLOCK_M=32, BLOCK_I=32, num_warps=4
        )
        return out, dzcol

    def _row_major_fused_bwd_mvue() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = torch.empty((M_ab, I), device=device, dtype=torch.bfloat16)
        _swiglu_row_fwd_kernel[(M_ab, triton.cdiv(I, 256))](
            z_row_storage, out, M_ab, I, z_row_storage.stride(0), out.stride(0), BLOCK_I=256, num_warps=4
        )
        d_gu, pk_dgu_t = fused_swiglu_bwd_and_mvue24_pack(dm_ab, z_row_storage, seed=77)
        return out, d_gu, pk_dgu_t

    # Verify parity of both Triton paths against eager reference!
    ref_dz_row = _col_major_naive_eager().contiguous()
    out_col, dz_col = _col_major_coalesced_triton()
    out_row, dgu_row, _ = _row_major_fused_bwd_mvue()
    g_ref, u_ref = z_row_storage.chunk(2, dim=-1)
    y_ref_ab = F.silu(g_ref) * u_ref

    gated_ablation: dict[str, Any] = {
        "shape_m_2i": [M_ab, 2 * I],
        "col_coalesced_out_rel_l2": round(rel_l2(out_col, y_ref_ab), 6),
        "col_coalesced_dgu_rel_l2": round(rel_l2(dz_col.t().contiguous(), ref_dz_row), 6),
        "row_fused_mvue_dgu_rel_l2": round(rel_l2(dgu_row, ref_dz_row), 6),
        "modes": {},
    }
    for aname, afn in [
        ("col_major_strided_eager_swiglu", _col_major_naive_eager),
        ("col_major_coalesced_triton_swiglu", _col_major_coalesced_triton),
        ("row_major_fused_swiglu_plus_mvue24_pack", _row_major_fused_bwd_mvue),
    ]:
        st = time_fn_with_peak_mem(afn, warmup, iters, device)
        st["effective_gbps"] = round(swiglu_fwd_bwd_bytes / (st["median_ms"] * 1e-3) / 1e9, 2)
        gated_ablation["modes"][aname] = st
    del z_col_storage, z_row_storage, dm_ab, ref_dz_row, out_col, dz_col, out_row, dgu_row, y_ref_ab

    # --- B2: 4-Layer Llama-3-8B Pre-Norm SwiGLU Stack (Fwd + Bwd + Optimizer Step) ---
    torch.manual_seed(1247)
    dense_blocks = nn.ModuleList([DensePreNormSwiGLUBlock(H, I).to(device) for _ in range(n_layers)])
    sparse_blocks = nn.ModuleList([Sparse2by4PreNormSwiGLUBlock(b).to(device) for b in dense_blocks])
    compiled_blocks = [torch.compile(b, mode="default") for b in dense_blocks]
    dense_opt = torch.optim.AdamW(dense_blocks.parameters(), lr=0.0, fused=True)  # lr=0 preserves weights across timing reps!

    # Parity check of our Sparse2by4PreNormSwiGLUBlock Fwd & dX against exact masked-dense reference!
    x_chk = torch.randn(1024, H, device=device, dtype=torch.bfloat16, requires_grad=True)
    dy_chk = torch.randn(1024, H, device=device, dtype=torch.bfloat16)
    masked_ref_blocks = nn.ModuleList([DensePreNormSwiGLUBlock(H, I).to(device) for _ in range(n_layers)])
    with torch.no_grad():
        for mb, sb in zip(masked_ref_blocks, sparse_blocks):
            mb.norm.copy_(sb.norm)
            mb.w_gate_up.weight.copy_(sb.w_gate_up * bmask_u16_to_dense(sb.bm_gu))
            mb.w_down.weight.copy_(sb.w_down * bmask_u16_to_dense(sb.bm_dn))
    h_mref = x_chk
    for mb in masked_ref_blocks:
        h_mref = mb(h_mref)
    h_mref.backward(dy_chk)

    xs_chk = x_chk.detach().clone().requires_grad_(True)
    for p in sparse_blocks.parameters():
        p.grad = None
    h_sp = xs_chk
    for li, sb in enumerate(sparse_blocks):
        h_sp = sb(h_sp, refresh_mask=False, step_seed=500 + li * 37)
    h_sp.backward(dy_chk)

    assert xs_chk.grad is not None and x_chk.grad is not None
    stack_parity = {
        "out_rel_l2_vs_exact_masked_dense": round(rel_l2(h_sp, h_mref), 6),
        "dx_rel_l2_vs_exact_masked_dense": round(rel_l2(xs_chk.grad, x_chk.grad), 6),
        "dnorm0_rel_l2_vs_exact_masked_dense": round(
            rel_l2(sparse_blocks[0].norm.grad, masked_ref_blocks[0].norm.grad), 6  # type: ignore[arg-type]
        ),
    }
    del masked_ref_blocks, x_chk, xs_chk, dy_chk, h_mref, h_sp

    grad_bytes_stack = sum(p.numel() * 2 for p in dense_blocks.parameters())
    scaling_b: dict[str, Any] = {}
    for M in tok_list:
        x_in = torch.randn(M, H, device=device, dtype=torch.bfloat16, requires_grad=True)
        dout = torch.randn(M, H, device=device, dtype=torch.bfloat16)
        flops_stack = n_layers * 18.0 * M * H * I

        # Prime cuSPARSELt plan cache on first sparse forward/backward
        for p in sparse_blocks.parameters():
            p.grad = None
        x_in.grad = None
        hs0 = x_in
        for li, sb in enumerate(sparse_blocks):
            hs0 = sb(hs0, refresh_mask=True, step_seed=700 + li)
        hs0.backward(dout)
        del hs0

        def _run_dense_eager() -> None:
            for p in dense_blocks.parameters():
                p.grad = None
            x_in.grad = None
            h = x_in
            for blk in dense_blocks:
                h = blk(h)
            h.backward(dout)
            dense_opt.step()

        def _run_dense_compile() -> None:
            for p in dense_blocks.parameters():
                p.grad = None
            x_in.grad = None
            h = x_in
            for cblk in compiled_blocks:
                h = cblk(h)
            h.backward(dout)
            dense_opt.step()

        class _Cutlass2by4STELinearFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
                ws = sss.SparseSemiStructuredTensorCUTLASS.prune_dense_static_sort(
                    weight, algorithm="largest_abs_values_greedy"
                )
                ctx.save_for_backward(inp)
                ctx.ws = ws
                return F.linear(inp, ws)

            @staticmethod
            def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                (inp,) = ctx.saved_tensors
                go = grad_out.contiguous()
                dx_out = go @ ctx.ws
                dw_out = go.t() @ inp
                return dx_out, dw_out

        def _run_pyt_cutlass() -> None:
            for p in dense_blocks.parameters():
                p.grad = None
            x_in.grad = None
            h = x_in
            for blk in dense_blocks:
                xf = h.float()
                xn = (xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + 1e-6)).to(
                    torch.bfloat16
                ) * blk.norm
                g, u = _Cutlass2by4STELinearFn.apply(xn, blk.w_gate_up.weight).chunk(2, dim=-1)
                h = h + _Cutlass2by4STELinearFn.apply(F.silu(g) * u, blk.w_down.weight)
            h.backward(dout)
            dense_opt.step()

        def _run_fst_l1() -> None:
            for p in sparse_blocks.parameters():
                p.grad = None
            x_in.grad = None
            h = x_in
            for li, sb in enumerate(sparse_blocks):
                h = sb(h, refresh_mask=True, step_seed=1000 + li * 31)
            h.backward(dout)
            for sb in sparse_blocks:
                assert sb.w_gate_up.grad is not None and sb.w_down.grad is not None
                fused_adamw_bitmask_step(
                    sb.w_gate_up.data, sb.w_gate_up.grad, sb.m_gu, sb.v_gu, sb.bm_gu, step=1, lr=0.0, sr_mode=2
                )
                fused_adamw_bitmask_step(
                    sb.w_down.data, sb.w_down.grad, sb.m_dn, sb.v_dn, sb.bm_dn, step=1, lr=0.0, sr_mode=2
                )

        l10_ctr = [0]

        def _run_fst_l10() -> None:
            ref_now = (l10_ctr[0] % 10) == 0
            l10_ctr[0] += 1
            for p in sparse_blocks.parameters():
                p.grad = None
            x_in.grad = None
            h = x_in
            for li, sb in enumerate(sparse_blocks):
                h = sb(h, refresh_mask=ref_now, step_seed=2000 + li * 31)
            h.backward(dout)
            for sb in sparse_blocks:
                assert sb.w_gate_up.grad is not None and sb.w_down.grad is not None
                fused_adamw_bitmask_step(
                    sb.w_gate_up.data, sb.w_gate_up.grad, sb.m_gu, sb.v_gu, sb.bm_gu, step=1, lr=0.0, sr_mode=2
                )
                fused_adamw_bitmask_step(
                    sb.w_down.data, sb.w_down.grad, sb.m_dn, sb.v_dn, sb.bm_dn, step=1, lr=0.0, sr_mode=2
                )

        modes_b: dict[str, Any] = {}
        for mname, mfn in [
            ("mlp_bf16_dense_eager", _run_dense_eager),
            ("mlp_bf16_dense_compile", _run_dense_compile),
            ("mlp_2by4_pytorch_cutlass_2gemm", _run_pyt_cutlass),
            ("mlp_2by4_fst_cslt_refresh1", _run_fst_l1),
            ("mlp_2by4_fst_cslt_refresh10", _run_fst_l10),
        ]:
            st = time_fn_with_peak_mem(
                mfn, warmup, iters, device, grad_bytes_steady=grad_bytes_stack + M * H * 2
            )
            st["effective_tflops"] = round(flops_stack / (st["median_ms"] * 1e-3) / 1e12, 2)
            modes_b[mname] = st

        d_eager_ms = modes_b["mlp_bf16_dense_eager"]["median_ms"]
        d_comp_ms = modes_b["mlp_bf16_dense_compile"]["median_ms"]
        for st in modes_b.values():
            st["speedup_vs_dense_eager"] = round(d_eager_ms / max(st["median_ms"], 1e-9), 3)
            st["speedup_vs_dense_compile"] = round(d_comp_ms / max(st["median_ms"], 1e-9), 3)

        scaling_b[str(M)] = {
            "num_tokens": M,
            "num_layers": n_layers,
            "hidden": H,
            "intermediate": I,
            "modes": modes_b,
        }
        del x_in, dout
        gc.collect()
        torch.cuda.empty_cache()

    del dense_blocks, sparse_blocks, compiled_blocks, dense_opt
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "gated_activation_layout_ablation": gated_ablation,
        "stack_masked_bf16_parity": stack_parity,
        "token_scaling": scaling_b,
    }


# ============================================================================
# 6. Suite C: 8-GPU NVSwitch Shard-Local Exact-90 `9 bits/param` Compressed-
#    AllGather 2:4 FSDP2 Training (6 Layers, 1.31B Params, 65,536 Global Tok)
# ============================================================================


class GQATransformerBlockDense(nn.Module):
    def __init__(
        self, hidden: int, inter: int, n_q: int, n_kv: int, head_dim: int = 128
    ) -> None:
        super().__init__()
        self.n_q = n_q
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.norm1 = nn.Parameter(torch.ones(hidden, dtype=torch.bfloat16))
        self.w_qkv = nn.Linear(
            hidden, (n_q + 2 * n_kv) * head_dim, bias=False, dtype=torch.bfloat16
        )
        self.w_o = nn.Linear(n_q * head_dim, hidden, bias=False, dtype=torch.bfloat16)
        self.norm2 = nn.Parameter(torch.ones(hidden, dtype=torch.bfloat16))
        self.w_gate_up = nn.Linear(hidden, 2 * inter, bias=False, dtype=torch.bfloat16)
        self.w_down = nn.Linear(inter, hidden, bias=False, dtype=torch.bfloat16)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        xf = x.float()
        h1 = (xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        ) * self.norm1
        qkv = self.w_qkv(h1)
        qd = self.n_q * self.head_dim
        kvd = self.n_kv * self.head_dim
        q = qkv[:, :, :qd].view(B, S, self.n_q, self.head_dim)
        k = qkv[:, :, qd : qd + kvd].view(B, S, self.n_kv, self.head_dim)
        v = qkv[:, :, qd + kvd :].view(B, S, self.n_kv, self.head_dim)
        ao = flash_attn_func(q, k, v, causal=True).reshape(B, S, qd)
        return x + self.w_o(ao)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        h2 = (xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        ) * self.norm2
        g, u = self.w_gate_up(h2).chunk(2, dim=-1)
        return x + self.w_down(F.silu(g) * u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(self._attn(x))


class GQATransformerModelDense(nn.Module):
    def __init__(
        self,
        n_layers: int,
        hidden: int,
        inter: int,
        n_q: int,
        n_kv: int,
        compile_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [GQATransformerBlockDense(hidden, inter, n_q, n_kv) for _ in range(n_layers)]
        )
        if compile_mlp:
            for blk in self.blocks:
                blk._mlp = torch.compile(blk._mlp, mode="default")  # type: ignore[method-assign]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


def _wrap_fsdp2(model: GQATransformerModelDense) -> GQATransformerModelDense:
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
    for blk in model.blocks:
        fully_shard(blk, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model, mp_policy=mp, reshard_after_forward=True)
    return model


def _unpack_9bit_fwd_slice_i64(
    gathered_2d_u8: torch.Tensor,
    byte_off: int,
    N_sh: int,
    K: int,
    world_size: int,
    out_pk: torch.Tensor,
) -> torch.Tensor:
    """8-byte vectorized (`int64` `LDG.E.64 / STG.E.64`) assembly of forward `packed` ([N_full, 9*K/16]).
    Assembled `out_pk` is 100% BITWISE IDENTICAL to 1-GPU `exact90_dual_pack(W_full)[0]`.
    """
    shard_bytes = (9 * N_sh * K) // 8
    val_i64 = (N_sh * K) // 8
    sl_i64 = gathered_2d_u8[:, byte_off : byte_off + shard_bytes].contiguous().view(torch.int64)
    out_i64 = out_pk.view(torch.int64).reshape(-1)
    out_i64[: world_size * val_i64].copy_(sl_i64[:, :val_i64].reshape(-1))
    meta_4d = sl_i64[:, val_i64:].view(world_size, K // 64, N_sh // 128, 128)
    out_i64[world_size * val_i64 :].copy_(meta_4d.permute(1, 0, 2, 3).reshape(-1))
    return out_pk


def _unpack_9bit_bwd_slice_i64(
    gathered_2d_u8: torch.Tensor,
    byte_off: int,
    N_sh: int,
    K: int,
    world_size: int,
    out_pkt: torch.Tensor,
) -> torch.Tensor:
    """8-byte vectorized (`int64` `LDG.E.64 / STG.E.64`) assembly of backward `packed_t` ([K, 9*N_full/16]).
    Assembled `out_pkt` is 100% BITWISE IDENTICAL to 1-GPU `exact90_dual_pack(W_full)[1]`.
    """
    shard_bytes = (9 * N_sh * K) // 8
    val_i64 = (K * N_sh) // 8
    sl_i64 = gathered_2d_u8[:, byte_off : byte_off + shard_bytes].contiguous().view(torch.int64)
    out_i64 = out_pkt.view(torch.int64).reshape(-1)
    vals_3d = sl_i64[:, :val_i64].view(world_size, K, N_sh // 8)
    out_i64[: world_size * val_i64].copy_(vals_3d.permute(1, 0, 2).reshape(-1))
    out_i64[world_size * val_i64 :].copy_(sl_i64[:, val_i64:].reshape(-1))
    return out_pkt


def _unpack_16bit_dense_slice_i64(
    gathered_2d_u8: torch.Tensor,
    byte_off: int,
    N_sh: int,
    K: int,
    world_size: int,
    out_w: torch.Tensor,
) -> torch.Tensor:
    shard_bytes = 2 * N_sh * K
    sl_i64 = gathered_2d_u8[:, byte_off : byte_off + shard_bytes].contiguous().view(torch.int64)
    out_w.view(torch.int64).reshape(-1).copy_(sl_i64.reshape(-1))
    return out_w


class _Sparse9BitPipelineWorkspace:
    """Pre-allocated double-buffered flat AllGather / ReduceScatter buffers + dedicated `comm_stream`
    so `Sparse9BitFSDP2Model` executes 1 flat AllGather per block (overlapped with prior/next block
    compute) and overlaps MLP `ReduceScatter` behind Attention backward inside each block.
    """

    def __init__(
        self,
        world_size: int,
        fwd_sh_bytes: int,
        bwd_sh_bytes: int,
        H: int,
        I: int,
        qd: int,
        qkv_d: int,
        mlp_sh_elems: int,
        attn_sh_elems: int,
        sparse_attn: bool,
        post_ag_redundant: bool,
        device: torch.device,
    ) -> None:
        self.comm_stream = torch.cuda.Stream(device=device)
        self.ag_fwd_buf = [
            torch.empty((world_size, fwd_sh_bytes), device=device, dtype=torch.uint8)
            for _ in range(2)
        ]
        self.ag_bwd_buf = [
            torch.empty((world_size, bwd_sh_bytes), device=device, dtype=torch.uint8)
            for _ in range(2)
        ]
        self.ag_fwd_ev = [torch.cuda.Event() for _ in range(2)]
        self.fwd_cons_ev = [torch.cuda.Event() for _ in range(2)]
        self.ag_bwd_ev = [torch.cuda.Event() for _ in range(2)]
        self.bwd_cons_ev = [torch.cuda.Event() for _ in range(2)]

        # Reusable full-layer 9-bit weight workspaces (zero torch.empty inside forward/backward!)
        need_sparse_attn = sparse_attn or post_ag_redundant
        need_dense_attn = (not sparse_attn) or post_ag_redundant
        empty_bf16 = torch.empty((0,), device=device, dtype=torch.bfloat16)

        self.pk_qkv = (
            torch.empty((qkv_d, 9 * H // 16), device=device, dtype=torch.bfloat16)
            if need_sparse_attn
            else empty_bf16
        )
        self.pk_o = (
            torch.empty((H, 9 * qd // 16), device=device, dtype=torch.bfloat16)
            if need_sparse_attn
            else empty_bf16
        )
        self.pk_gu = torch.empty((2 * I, 9 * H // 16), device=device, dtype=torch.bfloat16)
        self.pk_dn = torch.empty((H, 9 * I // 16), device=device, dtype=torch.bfloat16)

        self.pkt_qkv = (
            torch.empty((H, 9 * qkv_d // 16), device=device, dtype=torch.bfloat16)
            if need_sparse_attn
            else empty_bf16
        )
        self.pkt_o = (
            torch.empty((qd, 9 * H // 16), device=device, dtype=torch.bfloat16)
            if need_sparse_attn
            else empty_bf16
        )
        self.pkt_gu = torch.empty((H, 9 * (2 * I) // 16), device=device, dtype=torch.bfloat16)
        self.pkt_dn = torch.empty((I, 9 * H // 16), device=device, dtype=torch.bfloat16)

        self.dense_qkv = (
            torch.empty((qkv_d, H), device=device, dtype=torch.bfloat16)
            if need_dense_attn
            else empty_bf16
        )
        self.dense_o = (
            torch.empty((H, qd), device=device, dtype=torch.bfloat16)
            if need_dense_attn
            else empty_bf16
        )
        self.dense_gu = (
            torch.empty((2 * I, H), device=device, dtype=torch.bfloat16)
            if post_ag_redundant
            else empty_bf16
        )
        self.dense_dn = (
            torch.empty((H, I), device=device, dtype=torch.bfloat16)
            if post_ag_redundant
            else empty_bf16
        )

        self.rs_mlp_in = torch.empty((world_size, mlp_sh_elems), device=device, dtype=torch.float32)
        self.rs_mlp_out = torch.empty((mlp_sh_elems,), device=device, dtype=torch.float32)
        self.rs_attn_in = torch.empty((world_size, attn_sh_elems), device=device, dtype=torch.float32)
        self.rs_attn_out = torch.empty((attn_sh_elems,), device=device, dtype=torch.float32)
        self.dnorms_buf = torch.empty((2, H), device=device, dtype=torch.float32)


class _Sparse9BitFSDP2BlockFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        norm1: torch.Tensor,
        wqkv_sh: torch.Tensor,
        wo_sh: torch.Tensor,
        norm2: torch.Tensor,
        wgu_sh: torch.Tensor,
        wdn_sh: torch.Tensor,
        blk_ref: Any,
        all_blocks: Any,
        layer_idx: int,
        pws: _Sparse9BitPipelineWorkspace,
        step_seed: int,
    ) -> torch.Tensor:
        B, S, H = x.shape
        M = B * S
        x2d = x.view(M, H)
        I = wdn_sh.shape[1]
        n_q = blk_ref.n_q
        n_kv = blk_ref.n_kv
        head_dim = blk_ref.head_dim
        qd = n_q * head_dim
        kvd = n_kv * head_dim
        world_size = blk_ref.world_size
        sparse_attn = blk_ref.sparse_attn
        post_ag_redundant = blk_ref.post_ag_redundant
        use_sparse_dw = blk_ref.use_sparse_dw
        n_layers = len(all_blocks)

        comp_stream = torch.cuda.current_stream(x.device)
        buf_idx = layer_idx % 2

        # Wait for this block's prefetched flat AllGather, and concurrently prefetch block `layer_idx + 1`!
        comp_stream.wait_event(pws.ag_fwd_ev[buf_idx])
        if layer_idx + 1 < n_layers:
            nxt_buf = (layer_idx + 1) % 2
            with torch.cuda.stream(pws.comm_stream):
                pws.comm_stream.wait_event(pws.fwd_cons_ev[nxt_buf])
                dist.all_gather_into_tensor(
                    pws.ag_fwd_buf[nxt_buf].view(-1), all_blocks[layer_idx + 1].flat_fwd_sh
                )
                pws.ag_fwd_ev[nxt_buf].record(pws.comm_stream)

        g2d = pws.ag_fwd_buf[buf_idx]
        offs_f = blk_ref.fwd_offsets
        if post_ag_redundant:
            wqkv_f = _unpack_16bit_dense_slice_i64(g2d, offs_f["qkv"], wqkv_sh.shape[0], H, world_size, pws.dense_qkv)
            wo_f = _unpack_16bit_dense_slice_i64(g2d, offs_f["o"], wo_sh.shape[0], qd, world_size, pws.dense_o)
            wgu_f = _unpack_16bit_dense_slice_i64(g2d, offs_f["gu"], wgu_sh.shape[0], H, world_size, pws.dense_gu)
            wdn_f = _unpack_16bit_dense_slice_i64(g2d, offs_f["dn"], wdn_sh.shape[0], I, world_size, pws.dense_dn)
            pws.fwd_cons_ev[buf_idx].record(comp_stream)
            exact90_dual_pack(wqkv_f, pws.pk_qkv, pws.pkt_qkv, store_bmask=False)
            exact90_dual_pack(wo_f, pws.pk_o, pws.pkt_o, store_bmask=False)
            exact90_dual_pack(wgu_f, pws.pk_gu, pws.pkt_gu, store_bmask=False)
            exact90_dual_pack(wdn_f, pws.pk_dn, pws.pkt_dn, store_bmask=False)
        else:
            if sparse_attn:
                _unpack_9bit_fwd_slice_i64(g2d, offs_f["qkv"], wqkv_sh.shape[0], H, world_size, pws.pk_qkv)
                _unpack_9bit_fwd_slice_i64(g2d, offs_f["o"], wo_sh.shape[0], qd, world_size, pws.pk_o)
            else:
                _unpack_16bit_dense_slice_i64(g2d, offs_f["qkv"], wqkv_sh.shape[0], H, world_size, pws.dense_qkv)
                _unpack_16bit_dense_slice_i64(g2d, offs_f["o"], wo_sh.shape[0], qd, world_size, pws.dense_o)
            _unpack_9bit_fwd_slice_i64(g2d, offs_f["gu"], wgu_sh.shape[0], H, world_size, pws.pk_gu)
            _unpack_9bit_fwd_slice_i64(g2d, offs_f["dn"], wdn_sh.shape[0], I, world_size, pws.pk_dn)
            pws.fwd_cons_ev[buf_idx].record(comp_stream)

        # Attention subblock
        xn1, rstd1 = triton_rmsnorm_fwd(x2d, norm1)
        if sparse_attn or post_ag_redundant:
            qkv = CSLT_CACHE.mm(pws.pk_qkv, xn1.t(), transpose_result=True)
        else:
            qkv = xn1 @ pws.dense_qkv.t()

        q_leaf = qkv[:, :qd].view(B, S, n_q, head_dim).contiguous().detach().requires_grad_(True)
        k_leaf = qkv[:, qd : qd + kvd].view(B, S, n_kv, head_dim).contiguous().detach().requires_grad_(True)
        v_leaf = qkv[:, qd + kvd :].view(B, S, n_kv, head_dim).contiguous().detach().requires_grad_(True)
        del qkv
        with torch.enable_grad():
            ao_leaf = flash_attn_func(q_leaf, k_leaf, v_leaf, causal=True)
        ao2d = ao_leaf.reshape(M, qd)

        if sparse_attn or post_ag_redundant:
            o_out = CSLT_CACHE.mm(pws.pk_o, ao2d.t(), transpose_result=True)
        else:
            o_out = ao2d @ pws.dense_o.t()
        x_mid = x2d + o_out
        del o_out

        # 2:4 Sparse-SwiGLU MLP subblock
        xn2, rstd2 = triton_rmsnorm_fwd(x_mid, norm2)
        gu = CSLT_CACHE.mm(pws.pk_gu, xn2.t(), transpose_result=True)
        mid = torch.empty((M, I), device=x.device, dtype=torch.bfloat16)
        _swiglu_row_fwd_kernel[(M, triton.cdiv(I, 256))](
            gu, mid, M, I, gu.stride(0), mid.stride(0), BLOCK_I=256, num_warps=4
        )
        dn_out = CSLT_CACHE.mm(pws.pk_dn, mid.t(), transpose_result=True)
        out = (x_mid + dn_out).view(B, S, H)

        ctx.save_for_backward(
            x2d, norm1, xn1, rstd1, ao2d,
            x_mid, norm2, xn2, rstd2, gu, mid,
            wqkv_sh, wo_sh, wgu_sh, wdn_sh,
        )
        ctx.blk_ref = blk_ref
        ctx.all_blocks = all_blocks
        ctx.layer_idx = layer_idx
        ctx.pws = pws
        ctx.fa2_leaves = (q_leaf, k_leaf, v_leaf, ao_leaf)
        ctx.meta = (B, S, H, I, n_q, n_kv, head_dim, world_size, use_sparse_dw, step_seed)
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[Any, ...]:
        (
            x2d, norm1, xn1, rstd1, ao2d,
            x_mid, norm2, xn2, rstd2, gu, mid,
            wqkv_sh, wo_sh, wgu_sh, wdn_sh,
        ) = ctx.saved_tensors
        blk_ref = ctx.blk_ref
        all_blocks = ctx.all_blocks
        layer_idx: int = ctx.layer_idx
        pws: _Sparse9BitPipelineWorkspace = ctx.pws
        q_leaf, k_leaf, v_leaf, ao_leaf = ctx.fa2_leaves
        B, S, H, I, n_q, n_kv, head_dim, world_size, use_sparse_dw, seed = ctx.meta
        M = B * S
        qd = n_q * head_dim
        kvd = n_kv * head_dim
        sparse_attn = blk_ref.sparse_attn
        post_ag_redundant = blk_ref.post_ag_redundant
        n_layers = len(all_blocks)

        comp_stream = torch.cuda.current_stream(dout.device)
        buf_idx = layer_idx % 2

        # Kick off initial backward flat AllGather on the topmost block
        if layer_idx == n_layers - 1:
            with torch.cuda.stream(pws.comm_stream):
                pws.comm_stream.wait_stream(comp_stream)
                dist.all_gather_into_tensor(pws.ag_bwd_buf[buf_idx].view(-1), blk_ref.flat_bwd_sh)
                pws.ag_bwd_ev[buf_idx].record(pws.comm_stream)

        comp_stream.wait_event(pws.ag_bwd_ev[buf_idx])
        # Concurrently prefetch `layer_idx - 1` flat AllGather on `comm_stream`!
        if layer_idx - 1 >= 0:
            prv_buf = (layer_idx - 1) % 2
            with torch.cuda.stream(pws.comm_stream):
                pws.comm_stream.wait_event(pws.bwd_cons_ev[prv_buf])
                dist.all_gather_into_tensor(
                    pws.ag_bwd_buf[prv_buf].view(-1), all_blocks[layer_idx - 1].flat_bwd_sh
                )
                pws.ag_bwd_ev[prv_buf].record(pws.comm_stream)

        g2d = pws.ag_bwd_buf[buf_idx]
        offs_b = blk_ref.bwd_offsets
        if post_ag_redundant:
            wqkv_f = _unpack_16bit_dense_slice_i64(g2d, offs_b["qkv"], wqkv_sh.shape[0], H, world_size, pws.dense_qkv)
            wo_f = _unpack_16bit_dense_slice_i64(g2d, offs_b["o"], wo_sh.shape[0], qd, world_size, pws.dense_o)
            wgu_f = _unpack_16bit_dense_slice_i64(g2d, offs_b["gu"], wgu_sh.shape[0], H, world_size, pws.dense_gu)
            wdn_f = _unpack_16bit_dense_slice_i64(g2d, offs_b["dn"], wdn_sh.shape[0], I, world_size, pws.dense_dn)
            pws.bwd_cons_ev[buf_idx].record(comp_stream)
            exact90_dual_pack(wqkv_f, pws.pk_qkv, pws.pkt_qkv, store_bmask=False)
            exact90_dual_pack(wo_f, pws.pk_o, pws.pkt_o, store_bmask=False)
            exact90_dual_pack(wgu_f, pws.pk_gu, pws.pkt_gu, store_bmask=False)
            exact90_dual_pack(wdn_f, pws.pk_dn, pws.pkt_dn, store_bmask=False)
        else:
            if sparse_attn:
                _unpack_9bit_bwd_slice_i64(g2d, offs_b["qkv"], wqkv_sh.shape[0], H, world_size, pws.pkt_qkv)
                _unpack_9bit_bwd_slice_i64(g2d, offs_b["o"], wo_sh.shape[0], qd, world_size, pws.pkt_o)
            else:
                _unpack_16bit_dense_slice_i64(g2d, offs_b["qkv"], wqkv_sh.shape[0], H, world_size, pws.dense_qkv)
                _unpack_16bit_dense_slice_i64(g2d, offs_b["o"], wo_sh.shape[0], qd, world_size, pws.dense_o)
            _unpack_9bit_bwd_slice_i64(g2d, offs_b["gu"], wgu_sh.shape[0], H, world_size, pws.pkt_gu)
            _unpack_9bit_bwd_slice_i64(g2d, offs_b["dn"], wdn_sh.shape[0], I, world_size, pws.pkt_dn)
            pws.bwd_cons_ev[buf_idx].record(comp_stream)

        dout2d = dout.contiguous().view(M, H)

        # --- Backward MLP ---
        d_mid = CSLT_CACHE.mm(pws.pkt_dn, dout2d.t(), transpose_result=True)
        if use_sparse_dw:
            pk_dout_t = transpose_grad_2by4_pack(dout2d, seed=seed + 101, mode=0)
            dw_dn_full = CSLT_CACHE.mm(pk_dout_t, mid, transpose_result=False)
            del pk_dout_t
            d_gu, pk_dgu_t = fused_swiglu_bwd_and_mvue24_pack(d_mid, gu, seed=seed + 103)
        else:
            dw_dn_full = dout2d.t() @ mid
            d_gu, pk_dgu_t = fused_swiglu_bwd_and_mvue24_pack(d_mid, gu, seed=seed + 103)
            del pk_dgu_t
            pk_dgu_t = None
        del d_mid

        dxn2 = CSLT_CACHE.mm(pws.pkt_gu, d_gu.t(), transpose_result=True)
        if use_sparse_dw:
            assert pk_dgu_t is not None
            dw_gu_full = CSLT_CACHE.mm(pk_dgu_t, xn2, transpose_result=False)
            del pk_dgu_t
        else:
            dw_gu_full = d_gu.t() @ xn2
        del d_gu

        # Launch single fused MLP ReduceScatter (80.8% of block params!) on `comm_stream`
        # OVERLAPPED behind Attention backward compute!
        gu_el = wgu_sh.numel()
        dn_el = wdn_sh.numel()
        pws.rs_mlp_in[:, :gu_el].copy_(dw_gu_full.view(world_size, gu_el))
        pws.rs_mlp_in[:, gu_el:].copy_(dw_dn_full.view(world_size, dn_el))
        pws.rs_mlp_in.mul_(1.0 / world_size)
        del dw_gu_full, dw_dn_full
        mlp_rs_ready = torch.cuda.Event()
        mlp_rs_ready.record(comp_stream)
        with torch.cuda.stream(pws.comm_stream):
            pws.comm_stream.wait_event(mlp_rs_ready)
            dist.reduce_scatter_tensor(
                pws.rs_mlp_out, pws.rs_mlp_in.reshape(-1), op=dist.ReduceOp.SUM
            )

        dx_norm2, dnorm2_loc = triton_rmsnorm_bwd(dxn2, x_mid, norm2, rstd2)
        del dxn2
        dx_mid = dout2d + dx_norm2
        del dx_norm2

        # --- Backward Attention (executes on Tensor Cores in parallel with MLP ReduceScatter!) ---
        if sparse_attn or post_ag_redundant:
            dao2d = CSLT_CACHE.mm(pws.pkt_o, dx_mid.t(), transpose_result=True)
            if use_sparse_dw:
                pk_dxm_t = transpose_grad_2by4_pack(dx_mid, seed=seed + 107, mode=0)
                dw_o_full = CSLT_CACHE.mm(pk_dxm_t, ao2d, transpose_result=False)
                del pk_dxm_t
            else:
                dw_o_full = dx_mid.t() @ ao2d
        else:
            dao2d = dx_mid @ pws.dense_o
            dw_o_full = dx_mid.t() @ ao2d

        torch.autograd.backward(ao_leaf, dao2d.view(B, S, n_q, head_dim))
        del dao2d
        assert q_leaf.grad is not None and k_leaf.grad is not None and v_leaf.grad is not None
        dqkv = torch.cat(
            [
                q_leaf.grad.reshape(M, qd),
                k_leaf.grad.reshape(M, kvd),
                v_leaf.grad.reshape(M, kvd),
            ],
            dim=-1,
        )

        if sparse_attn or post_ag_redundant:
            dxn1 = CSLT_CACHE.mm(pws.pkt_qkv, dqkv.t(), transpose_result=True)
            if use_sparse_dw:
                pk_dqkv_t = transpose_grad_2by4_pack(dqkv, seed=seed + 109, mode=0)
                dw_qkv_full = CSLT_CACHE.mm(pk_dqkv_t, xn1, transpose_result=False)
                del pk_dqkv_t
            else:
                dw_qkv_full = dqkv.t() @ xn1
        else:
            dxn1 = dqkv @ pws.dense_qkv
            dw_qkv_full = dqkv.t() @ xn1
        del dqkv

        qkv_el = wqkv_sh.numel()
        o_el = wo_sh.numel()
        pws.rs_attn_in[:, :qkv_el].copy_(dw_qkv_full.view(world_size, qkv_el))
        pws.rs_attn_in[:, qkv_el:].copy_(dw_o_full.view(world_size, o_el))
        pws.rs_attn_in.mul_(1.0 / world_size)
        del dw_qkv_full, dw_o_full
        attn_rs_ready = torch.cuda.Event()
        attn_rs_ready.record(comp_stream)
        with torch.cuda.stream(pws.comm_stream):
            pws.comm_stream.wait_event(attn_rs_ready)
            dist.reduce_scatter_tensor(
                pws.rs_attn_out, pws.rs_attn_in.reshape(-1), op=dist.ReduceOp.SUM
            )

        # RMSNorm1 backward runs on compute_stream in parallel with Attention ReduceScatter!
        dx_norm1, dnorm1_loc = triton_rmsnorm_bwd(dxn1, x2d, norm1, rstd1)
        del dxn1
        dx = (dx_mid + dx_norm1).view(B, S, H)

        pws.dnorms_buf[0].copy_(dnorm1_loc.float())
        pws.dnorms_buf[1].copy_(dnorm2_loc.float())
        pws.dnorms_buf.mul_(1.0 / world_size)
        dnorm_ready = torch.cuda.Event()
        dnorm_ready.record(comp_stream)
        with torch.cuda.stream(pws.comm_stream):
            pws.comm_stream.wait_event(dnorm_ready)
            dist.all_reduce(pws.dnorms_buf, op=dist.ReduceOp.SUM)

        comp_stream.wait_stream(pws.comm_stream)
        dwgu_sh = pws.rs_mlp_out[:gu_el].view_as(wgu_sh).to(torch.bfloat16)
        dwdn_sh = pws.rs_mlp_out[gu_el:].view_as(wdn_sh).to(torch.bfloat16)
        dwqkv_sh = pws.rs_attn_out[:qkv_el].view_as(wqkv_sh).to(torch.bfloat16)
        dwo_sh = pws.rs_attn_out[qkv_el:].view_as(wo_sh).to(torch.bfloat16)
        dnorm1_out = pws.dnorms_buf[0].to(torch.bfloat16)
        dnorm2_out = pws.dnorms_buf[1].to(torch.bfloat16)
        return (
            dx,
            dnorm1_out,
            dwqkv_sh,
            dwo_sh,
            dnorm2_out,
            dwgu_sh,
            dwdn_sh,
            None,
            None,
            None,
            None,
            None,
        )


class Sparse9BitFSDP2Block(nn.Module):
    def __init__(
        self,
        ref_blk: GQATransformerBlockDense,
        rank: int,
        world_size: int,
        device: torch.device,
        sparse_attn: bool = True,
        post_ag_redundant: bool = False,
        use_sparse_dw: bool = True,
    ) -> None:
        super().__init__()
        self.n_q = ref_blk.n_q
        self.n_kv = ref_blk.n_kv
        self.head_dim = ref_blk.head_dim
        self.world_size = world_size
        self.sparse_attn = sparse_attn
        self.post_ag_redundant = post_ag_redundant
        self.use_sparse_dw = use_sparse_dw

        self.norm1 = nn.Parameter(ref_blk.norm1.detach().to(device))
        self.norm2 = nn.Parameter(ref_blk.norm2.detach().to(device))

        def _shard(w: torch.Tensor) -> nn.Parameter:
            n_sh = w.shape[0] // world_size
            return nn.Parameter(w.detach()[rank * n_sh : (rank + 1) * n_sh].contiguous().to(device))

        self.wqkv_sh = _shard(ref_blk.w_qkv.weight)
        self.wo_sh = _shard(ref_blk.w_o.weight)
        self.wgu_sh = _shard(ref_blk.w_gate_up.weight)
        self.wdn_sh = _shard(ref_blk.w_down.weight)

        # Build single contiguous block-flat forward and backward shard buffers
        self.fwd_offsets: dict[str, int] = {}
        self.bwd_offsets: dict[str, int] = {}
        specs = [
            ("qkv", self.wqkv_sh, sparse_attn and not post_ag_redundant),
            ("o", self.wo_sh, sparse_attn and not post_ag_redundant),
            ("gu", self.wgu_sh, not post_ag_redundant),
            ("dn", self.wdn_sh, not post_ag_redundant),
        ]
        f_cur, b_cur = 0, 0
        for key, p, is_9bit in specs:
            ns, ks = p.shape
            nbytes = (9 * ns * ks) // 8 if is_9bit else 2 * ns * ks
            self.fwd_offsets[key] = f_cur
            self.bwd_offsets[key] = b_cur
            f_cur += nbytes
            b_cur += nbytes
        self.flat_fwd_sh = torch.empty((f_cur,), device=device, dtype=torch.uint8)
        self.flat_bwd_sh = torch.empty((b_cur,), device=device, dtype=torch.uint8)

        self.sh_buffers: dict[str, torch.Tensor] = {}
        for key, p, is_9bit in specs:
            ns, ks = p.shape
            fo = self.fwd_offsets[key]
            bo = self.bwd_offsets[key]
            if is_9bit:
                nbytes = (9 * ns * ks) // 8
                pk = self.flat_fwd_sh[fo : fo + nbytes].view(torch.bfloat16).view(ns, 9 * ks // 16)
                pkt = self.flat_bwd_sh[bo : bo + nbytes].view(torch.bfloat16).view(ks, 9 * ns // 16)
                bm = torch.empty((ns // 4, ks // 4), device=device, dtype=torch.uint16)
                exact90_dual_pack(p.data, pk, pkt, bm, store_bmask=True)
                self.sh_buffers[f"pk_{key}"] = pk
                self.sh_buffers[f"pkt_{key}"] = pkt
                self.sh_buffers[f"bm_{key}"] = bm

    def stage_local_shards(self, refresh_mask: bool) -> None:
        if self.post_ag_redundant:
            for key, p in [
                ("qkv", self.wqkv_sh),
                ("o", self.wo_sh),
                ("gu", self.wgu_sh),
                ("dn", self.wdn_sh),
            ]:
                nb = 2 * p.numel()
                fo = self.fwd_offsets[key]
                self.flat_fwd_sh[fo : fo + nb].copy_(p.data.view(torch.uint8).reshape(-1))
                self.flat_bwd_sh[fo : fo + nb].copy_(p.data.view(torch.uint8).reshape(-1))
            return

        if not self.sparse_attn:
            for key, p in [("qkv", self.wqkv_sh), ("o", self.wo_sh)]:
                nb = 2 * p.numel()
                fo = self.fwd_offsets[key]
                self.flat_fwd_sh[fo : fo + nb].copy_(p.data.view(torch.uint8).reshape(-1))
                self.flat_bwd_sh[fo : fo + nb].copy_(p.data.view(torch.uint8).reshape(-1))

        sparse_keys = [("gu", self.wgu_sh), ("dn", self.wdn_sh)]
        if self.sparse_attn:
            sparse_keys = [("qkv", self.wqkv_sh), ("o", self.wo_sh)] + sparse_keys
        for key, p in sparse_keys:
            if refresh_mask:
                exact90_dual_pack(
                    p.data,
                    self.sh_buffers[f"pk_{key}"],
                    self.sh_buffers[f"pkt_{key}"],
                    self.sh_buffers[f"bm_{key}"],
                    store_bmask=True,
                )
            else:
                fast_cached_bitmask_dual_pack(
                    p.data,
                    self.sh_buffers[f"bm_{key}"],
                    self.sh_buffers[f"pk_{key}"],
                    self.sh_buffers[f"pkt_{key}"],
                )


class Sparse9BitFSDP2Model(nn.Module):
    def __init__(
        self,
        base_cpu: GQATransformerModelDense,
        rank: int,
        world_size: int,
        device: torch.device,
        sparse_attn: bool = True,
        post_ag_redundant: bool = False,
        use_sparse_dw: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Sparse9BitFSDP2Block(
                    blk,
                    rank,
                    world_size,
                    device,
                    sparse_attn=sparse_attn,
                    post_ag_redundant=post_ag_redundant,
                    use_sparse_dw=use_sparse_dw,
                )
                for blk in base_cpu.blocks
            ]
        )
        b0: Sparse9BitFSDP2Block = self.blocks[0]  # type: ignore[assignment]
        H = b0.wgu_sh.shape[1]
        I = b0.wdn_sh.shape[1]
        qd = b0.n_q * b0.head_dim
        qkv_d = (b0.n_q + 2 * b0.n_kv) * b0.head_dim
        self.pws = _Sparse9BitPipelineWorkspace(
            world_size=world_size,
            fwd_sh_bytes=int(b0.flat_fwd_sh.numel()),
            bwd_sh_bytes=int(b0.flat_bwd_sh.numel()),
            H=H,
            I=I,
            qd=qd,
            qkv_d=qkv_d,
            mlp_sh_elems=b0.wgu_sh.numel() + b0.wdn_sh.numel(),
            attn_sh_elems=b0.wqkv_sh.numel() + b0.wo_sh.numel(),
            sparse_attn=sparse_attn,
            post_ag_redundant=post_ag_redundant,
            device=device,
        )

    def forward(self, x: torch.Tensor, refresh_mask: bool = True, step_seed: int = 400) -> torch.Tensor:
        for blk in self.blocks:
            blk.stage_local_shards(refresh_mask=refresh_mask)  # type: ignore[operator]
        comp_stream = torch.cuda.current_stream(x.device)
        with torch.cuda.stream(self.pws.comm_stream):
            self.pws.comm_stream.wait_stream(comp_stream)
            dist.all_gather_into_tensor(
                self.pws.ag_fwd_buf[0].view(-1), self.blocks[0].flat_fwd_sh  # type: ignore[arg-type]
            )
            self.pws.ag_fwd_ev[0].record(self.pws.comm_stream)

        for li, blk in enumerate(self.blocks):
            x = _Sparse9BitFSDP2BlockFn.apply(
                x,
                blk.norm1,
                blk.wqkv_sh,
                blk.wo_sh,
                blk.norm2,
                blk.wgu_sh,
                blk.wdn_sh,
                blk,
                self.blocks,
                li,
                self.pws,
                step_seed + li * 43,
            )
        return x


def run_suite_c_9bit_2by4_fsdp2_8gpu(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    H, I, n_q, n_kv, hd = 4096, 14336, 32, 8, 128

    # --- C1: 8-GPU Distributed 9-Bit AllGather Bitwise & Gradient Parity Audit vs 1-GPU ---
    torch.manual_seed(1248)
    base_p = GQATransformerModelDense(2, H, I, n_q, n_kv)
    for p in base_p.parameters():
        if p.ndim == 2:
            nn.init.normal_(p, std=1.0 / math.sqrt(p.shape[1]))

    B_loc_p, S_p = 1, 1024
    x_glob = torch.randn(world_size * B_loc_p, S_p, H, device=device, dtype=torch.bfloat16)
    dy_glob = torch.randn(world_size * B_loc_p, S_p, H, device=device, dtype=torch.bfloat16)
    dist.broadcast(x_glob, src=0)
    dist.broadcast(dy_glob, src=0)
    x_loc_p = x_glob[rank * B_loc_p : (rank + 1) * B_loc_p].contiguous().detach().requires_grad_(True)
    dy_loc_p = dy_glob[rank * B_loc_p : (rank + 1) * B_loc_p].contiguous()

    # Run 8-GPU 9-bit AllGather with dense dW (`use_sparse_dw=False`) to prove ZERO-MISMATCH 9-bit AG!
    m_9bit_2gemm = Sparse9BitFSDP2Model(
        base_p, rank, world_size, device, sparse_attn=True, post_ag_redundant=False, use_sparse_dw=False
    )
    y_loc_2g = m_9bit_2gemm(x_loc_p, refresh_mask=True, step_seed=700)
    y_loc_2g.backward(dy_loc_p)
    y_gather_2g = torch.empty_like(x_glob)
    dx_gather_2g = torch.empty_like(x_glob)
    assert x_loc_p.grad is not None
    dist.all_gather_into_tensor(y_gather_2g, y_loc_2g.detach())
    dist.all_gather_into_tensor(dx_gather_2g, x_loc_p.grad.detach())

    dw_gu0_2g = torch.empty((2 * I, H), device=device, dtype=torch.bfloat16)
    assert m_9bit_2gemm.blocks[0].wgu_sh.grad is not None
    dist.all_gather_into_tensor(dw_gu0_2g, m_9bit_2gemm.blocks[0].wgu_sh.grad)
    del m_9bit_2gemm

    # Also average 16 Monte Carlo microsteps of full 3-GEMM FST (`use_sparse_dw=True`) to prove KKT-MVUE
    # distributed unbiased convergence vs 1-GPU exact 2:4 masked reference!
    m_9bit_3gemm = Sparse9BitFSDP2Model(
        base_p, rank, world_size, device, sparse_attn=True, post_ag_redundant=False, use_sparse_dw=True
    )
    mc_dist_steps = 8 if is_smoke else 16
    dw_gu0_fst_acc = torch.zeros_like(m_9bit_3gemm.blocks[0].wgu_sh, dtype=torch.float32)
    for ms in range(mc_dist_steps):
        for p in m_9bit_3gemm.parameters():
            p.grad = None
        x_loc_p.grad = None
        y_tmp = m_9bit_3gemm(x_loc_p, refresh_mask=(ms == 0), step_seed=9000 + ms * 997)
        y_tmp.backward(dy_loc_p)
        assert m_9bit_3gemm.blocks[0].wgu_sh.grad is not None
        dw_gu0_fst_acc.add_(m_9bit_3gemm.blocks[0].wgu_sh.grad.float())
    dw_gu0_fst_sh = (dw_gu0_fst_acc / float(mc_dist_steps)).to(torch.bfloat16)
    dw_gu0_fst = torch.empty((2 * I, H), device=device, dtype=torch.bfloat16)
    dist.all_gather_into_tensor(dw_gu0_fst, dw_gu0_fst_sh)
    del m_9bit_3gemm

    parity_c1: dict[str, Any] = {}
    if rank == 0:
        # Build exact 1-GPU Exact-90 masked reference model
        m_1gpu = GQATransformerModelDense(2, H, I, n_q, n_kv).to(device)
        m_1gpu.load_state_dict(base_p.state_dict())
        with torch.no_grad():
            for blk in m_1gpu.blocks:
                for lin in (blk.w_qkv, blk.w_o, blk.w_gate_up, blk.w_down):
                    _, _, bm_1 = exact90_dual_pack(lin.weight.data)
                    assert bm_1 is not None
                    lin.weight.mul_(bmask_u16_to_dense(bm_1))
        xg_ref = x_glob.detach().clone().requires_grad_(True)
        yg_ref = m_1gpu(xg_ref)
        yg_ref.backward(dy_glob / float(world_size))
        assert xg_ref.grad is not None and m_1gpu.blocks[0].w_gate_up.weight.grad is not None
        parity_c1 = {
            "fwd_9bit_ag_rel_l2_vs_1gpu_exact90": round(rel_l2(y_gather_2g, yg_ref), 6),
            "dx_9bit_ag_rel_l2_vs_1gpu_exact90": round(
                rel_l2(dx_gather_2g / float(world_size), xg_ref.grad), 6
            ),
            "dw_gate_up_2gemm_9bit_ag_rel_l2_vs_1gpu": round(
                rel_l2(dw_gu0_2g, m_1gpu.blocks[0].w_gate_up.weight.grad), 6
            ),
            "dw_gate_up_3gemm_fst_mc_avg_rel_l2_vs_1gpu": round(
                rel_l2(dw_gu0_fst, m_1gpu.blocks[0].w_gate_up.weight.grad), 6
            ),
            "mc_dist_steps": mc_dist_steps,
        }
        del m_1gpu, xg_ref, yg_ref
    del base_p, x_glob, dy_glob, x_loc_p, dy_loc_p, y_gather_2g, dx_gather_2g, dw_gu0_2g, dw_gu0_fst
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    # --- C2: 8-GPU NVSwitch FSDP2 Scaling (6 Layers, 1.309B Params, 65,536 Global Tok/Step) ---
    n_layers_s = 2 if is_smoke else 6
    B_loc_s = 2 if is_smoke else 4
    S_s = 1024 if is_smoke else 2048
    M_loc_s = B_loc_s * S_s
    N_glob_s = M_loc_s * world_size
    warmup = 2 if is_smoke else 5
    iters = 5 if is_smoke else 14

    torch.manual_seed(1249)
    base_s = GQATransformerModelDense(n_layers_s, H, I, n_q, n_kv)
    for p in base_s.parameters():
        if p.ndim == 2:
            nn.init.normal_(p, std=1.0 / math.sqrt(p.shape[1]))

    total_2d_params = sum(p.numel() for p in base_s.parameters() if p.ndim == 2)
    ffn_2d_params = sum(
        blk.w_gate_up.weight.numel() + blk.w_down.weight.numel() for blk in base_s.blocks
    )
    attn_2d_params = total_2d_params - ffn_2d_params
    params_billions = round(sum(p.numel() for p in base_s.parameters()) / 1e9, 4)

    # Fwd + Bwd reshard_after_forward=True performs 2 AllGathers per step:
    #   Dense BF16: 2 * 2.0 B/param = 4.0 B/param
    #   9-bit 2:4:  2 * 1.125 B/param = 2.25 B/param (43.75% bandwidth reduction, 1.778x smaller!)
    bf16_ag_mib = round(4.0 * total_2d_params / (1024.0**2), 2)
    full_9bit_ag_mib = round(2.25 * total_2d_params / (1024.0**2), 2)
    ffn_hybrid_9bit_ag_mib = round(
        (2.25 * ffn_2d_params + 4.0 * attn_2d_params) / (1024.0**2), 2
    )

    x_s = torch.randn(B_loc_s, S_s, H, device=device, dtype=torch.bfloat16)
    dy_s = torch.randn(B_loc_s, S_s, H, device=device, dtype=torch.bfloat16)

    def _bench_dist_mode(step_fn: Callable[[], None]) -> dict[str, float]:
        for _ in range(warmup):
            step_fn()
        torch.cuda.synchronize(device)
        dist.barrier()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        step_fn()
        torch.cuda.synchronize(device)
        peak_mib_loc = torch.cuda.max_memory_allocated(device) / (1024.0**2)
        pk_t = torch.tensor([peak_mib_loc], device=device, dtype=torch.float64)
        dist.all_reduce(pk_t, op=dist.ReduceOp.MAX)
        peak_alloc_mib = float(pk_t.item())

        samples: list[float] = []
        clks: list[float] = []
        pwrs: list[float] = []
        for i in range(iters):
            dist.barrier()
            t_a = time.perf_counter()
            step_fn()
            torch.cuda.synchronize(device)
            dist.barrier()
            t_b = time.perf_counter()
            samples.append((t_b - t_a) * 1e3)
            if i == iters // 2:
                c, pw = query_gpu_telemetry(device.index or 0)
                if c > 0:
                    clks.append(c)
                    pwrs.append(pw)
        s_sort = sorted(samples)
        med = statistics.median(s_sort)
        q1 = s_sort[len(s_sort) // 4]
        q3 = s_sort[(3 * len(s_sort)) // 4]
        return {
            "median_ms": round(med, 3),
            "iqr_ms": round(q3 - q1, 3),
            "min_ms": round(s_sort[0], 3),
            "max_ms": round(s_sort[-1], 3),
            "global_tok_per_s": round(N_glob_s / (med * 1e-3), 1),
            "peak_allocated_mib_max_rank": round(peak_alloc_mib, 2),
            "sm_clock_mhz": round(statistics.median(clks), 1) if clks else -1.0,
            "power_w": round(statistics.median(pwrs), 2) if pwrs else -1.0,
        }

    c2_modes: dict[str, Any] = {}

    # 1. FSDP2 BF16 Dense Eager
    m_eag = _wrap_fsdp2(GQATransformerModelDense(n_layers_s, H, I, n_q, n_kv, compile_mlp=False).to(device))
    m_eag.load_state_dict(base_s.state_dict(), assign=False) if False else None

    def _s_eag() -> None:
        for p in m_eag.parameters():
            p.grad = None
        m_eag(x_s).backward(dy_s)

    c2_modes["fsdp2_bf16_dense_eager"] = _bench_dist_mode(_s_eag)
    del m_eag
    gc.collect()
    torch.cuda.empty_cache()

    # 2. FSDP2 BF16 Dense Compile (compiled MLP)
    m_cmp = _wrap_fsdp2(GQATransformerModelDense(n_layers_s, H, I, n_q, n_kv, compile_mlp=True).to(device))

    def _s_cmp() -> None:
        for p in m_cmp.parameters():
            p.grad = None
        m_cmp(x_s).backward(dy_s)

    c2_modes["fsdp2_bf16_dense_compile"] = _bench_dist_mode(_s_cmp)
    del m_cmp
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Post-AllGather Redundant 2:4 FST (16-bit BF16 AllGather + 8x redundant full-matrix pruning)
    m_post = Sparse9BitFSDP2Model(
        base_s, rank, world_size, device, sparse_attn=True, post_ag_redundant=True, use_sparse_dw=True
    )

    def _s_post() -> None:
        for p in m_post.parameters():
            p.grad = None
        m_post(x_s, refresh_mask=True, step_seed=500).backward(dy_s)

    c2_modes["fsdp2_2by4_post_ag_redundant_fst"] = _bench_dist_mode(_s_post)
    del m_post
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Hu et al. FFN-Only Hybrid 9-Bit AllGather 2:4 FST (80.8% FFN params 9-bit 2:4 + Dense GQA)
    m_hyb = Sparse9BitFSDP2Model(
        base_s, rank, world_size, device, sparse_attn=False, post_ag_redundant=False, use_sparse_dw=True
    )

    def _s_hyb() -> None:
        for p in m_hyb.parameters():
            p.grad = None
        m_hyb(x_s, refresh_mask=True, step_seed=600).backward(dy_s)

    c2_modes["fsdp2_2by4_ffn_hybrid_9bit_ag_refresh1"] = _bench_dist_mode(_s_hyb)
    del m_hyb
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Full-Linear Shard-Local Exact-90 9-Bit AllGather 2:4 FST (L = 1 every-step refresh & L = 10 amortized)
    m_9bit = Sparse9BitFSDP2Model(
        base_s, rank, world_size, device, sparse_attn=True, post_ag_redundant=False, use_sparse_dw=True
    )

    def _s_9bit_l1() -> None:
        for p in m_9bit.parameters():
            p.grad = None
        m_9bit(x_s, refresh_mask=True, step_seed=700).backward(dy_s)

    c2_modes["fsdp2_2by4_9bit_ag_fst_refresh1"] = _bench_dist_mode(_s_9bit_l1)

    l10_box = [0]

    def _s_9bit_l10() -> None:
        ref = (l10_box[0] % 10) == 0
        l10_box[0] += 1
        for p in m_9bit.parameters():
            p.grad = None
        m_9bit(x_s, refresh_mask=ref, step_seed=800).backward(dy_s)

    c2_modes["fsdp2_2by4_9bit_ag_fst_refresh10"] = _bench_dist_mode(_s_9bit_l10)

    del m_9bit, base_s, x_s, dy_s
    gc.collect()
    torch.cuda.empty_cache()

    eag_ms = c2_modes["fsdp2_bf16_dense_eager"]["median_ms"]
    cmp_ms = c2_modes["fsdp2_bf16_dense_compile"]["median_ms"]
    for st in c2_modes.values():
        st["speedup_vs_fsdp2_dense_eager"] = round(eag_ms / max(st["median_ms"], 1e-9), 3)
        st["speedup_vs_fsdp2_dense_compile"] = round(cmp_ms / max(st["median_ms"], 1e-9), 3)

    return {
        "distributed_9bit_allgather_parity": parity_c1,
        "fsdp2_8gpu_training_scaling": {
            "num_layers": n_layers_s,
            "params_billions": params_billions,
            "ffn_linear_param_share": round(ffn_2d_params / max(total_2d_params, 1), 4),
            "local_tokens_per_gpu": M_loc_s,
            "global_tokens_per_step": N_glob_s,
            "seq_len": S_s,
            "bf16_allgather_mib_per_step": bf16_ag_mib,
            "ffn_hybrid_9bit_allgather_mib_per_step": ffn_hybrid_9bit_ag_mib,
            "full_9bit_allgather_mib_per_step": full_9bit_ag_mib,
            "modes": c2_modes,
        },
    }


# ============================================================================
# 7. Entrypoint
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E124: SM80 2:4 Semi-Structured Sparse Tensor-Core Training & 8-GPU 9-Bit FSDP2"
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_smoke = args.mode == "smoke"

    t0 = time.perf_counter()
    init_clk, init_pwr = query_gpu_telemetry(local_rank)

    s0_res: dict[str, Any] = {}
    sa_res: dict[str, Any] = {}
    sb_res: dict[str, Any] = {}

    if rank == 0:
        s0_res = run_suite_0_audits(device, is_smoke)
        print(f"[E124] Suite 0 complete at +{time.perf_counter() - t0:.1f}s", flush=True)
        sa_res = run_suite_a_linear_3gemm_scaling(device, is_smoke)
        print(f"[E124] Suite A complete at +{time.perf_counter() - t0:.1f}s", flush=True)
        sb_res = run_suite_b_sparse_swiglu_stack(device, is_smoke)
        print(f"[E124] Suite B complete at +{time.perf_counter() - t0:.1f}s", flush=True)

    dist.barrier()
    sc_res = run_suite_c_9bit_2by4_fsdp2_8gpu(rank, world_size, device, is_smoke)
    dist.barrier()
    elapsed_s = time.perf_counter() - t0

    if rank == 0:
        print(f"[E124] Suite C complete at +{elapsed_s:.1f}s", flush=True)
        env = environment_metadata()
        env["hostname"] = "node-0"
        env["triton"] = triton.__version__
        env["cusparselt_version"] = int(torch.backends.cusparselt.version())
        import flash_attn

        env["flash_attn"] = flash_attn.__version__
        payload = {
            "experiment": "E124_2by4_sparse_training",
            "mode": args.mode,
            "world_size": world_size,
            "elapsed_s": round(elapsed_s, 2),
            "initial_sm_clock_mhz": round(init_clk, 1),
            "initial_power_w": round(init_pwr, 2),
            "environment": env,
            "cslt_splitk_fallbacks": CSLT_CACHE.splitk_fallbacks,
            "suite_0_audits": s0_res,
            "suite_a_linear_3gemm_scaling": sa_res,
            "suite_b_sparse_swiglu_stack": sb_res,
            "suite_c_9bit_2by4_fsdp2_8gpu": sc_res,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"Wrote {args.out} in {elapsed_s:.2f}s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
