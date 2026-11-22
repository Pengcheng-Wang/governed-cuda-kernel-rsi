#!/usr/bin/env python3
"""E123: Block-Hadamard INT8 Tensor-Core Training (SwitchBack / Jetfire / COAT / Halo),
SM80 In-Loop Promotion vs Orthogonal Hadamard Roofline, Stochastic-Rounded 3-GEMM Backprop,
COAT Compressed INT8 Activation Saving, & 8-GPU Quantized-AllGather FSDP2 Training.

Architecture & Primary Literature Anchor:
  - SwitchBack (8-bit Linear Training): Dettmers et al., ICML 2023 (arXiv:2304.13013)
  - Jetfire (Full INT8 Dataflow & Block Quantization): Xi et al., ICML 2024 (arXiv:2403.12422)
  - DeepSeek-V3 (1x128 x 128x128 Block Scaling & Promotion): DeepSeek-AI, 2024 (arXiv:2412.19437)
  - QuaRot (Outlier-Free Hadamard Rotations): Ashkboos et al., NeurIPS 2024 (arXiv:2404.00456)
  - COAT (Compressed Activations for Low-Precision Training): Xi et al., ICLR 2025 (arXiv:2410.19313)
  - Halo (Hadamard Low-Precision LLM Training): Ashkboos et al., 2025 (arXiv:2501.02625)
"""

from __future__ import annotations

import argparse
import gc
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

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from flash_attn import flash_attn_func
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint as ckpt_fn
from vllm import _custom_ops as vllm_ops

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import environment_metadata

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True


# ============================================================================
# 0. Utilities, Orthogonal Hadamard Builders, & NVML Telemetry
# ============================================================================


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().double()
    bf = b.detach().double()
    return float((af - bf).norm().item() / bf.norm().clamp(min=1e-12).item())


def sylvester_hadamard(
    n: int, device: torch.device | str, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    steps = int(round(math.log2(n)))
    assert (1 << steps) == n, f"n={n} must be a power of 2"
    h = torch.tensor([[1.0]], device=device, dtype=torch.float64)
    for _ in range(steps):
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return (h / math.sqrt(n)).to(dtype=dtype).contiguous()


def rademacher_signs(
    n: int, seed: int, device: torch.device | str, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    s = (torch.randint(0, 2, (n,), generator=g, dtype=torch.int64) * 2 - 1).to(
        dtype=dtype
    )
    return s.to(device=device).contiguous()


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
        return 0.0, 0.0


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
    base_mem = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    peak_inc_mib = max(
        0.0, (torch.cuda.max_memory_allocated(device) - base_mem) / (1024.0**2)
    )
    transient_ws_mib = max(0.0, peak_inc_mib - (grad_bytes_steady / (1024.0**2)))

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize(device)
    samples = [starts[i].elapsed_time(ends[i]) for i in range(iters)]
    sm_clk, pwr_w = query_gpu_telemetry(device.index or 0)
    ss = sorted(samples)
    q25 = ss[max(0, (len(ss) - 1) // 4)]
    q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
    return {
        "median_ms": round(float(statistics.median(samples)), 4),
        "iqr_ms": round(float(q75 - q25), 4),
        "peak_inc_mib": round(float(peak_inc_mib), 2),
        "transient_workspace_mib": round(float(transient_ws_mib), 2),
        "sm_clock_after_mhz": round(sm_clk, 1),
        "power_after_w": round(pwr_w, 2),
    }


# ============================================================================
# 1. Custom Triton SM80 INT8 Tensor-Core GEMM, Block-Hadamard, Kronecker,
#    Fused rstd-Invariant RMSNorm->INT8, & COAT SwiGLU Kernels
# ============================================================================


@triton.jit
def _fast_sr_noise(seed, idx):
    """3-multiply Murmur3/SplitMix32 integer hash -> unbiased uniform [-0.5, 0.5)."""
    u = (idx.to(tl.uint32) ^ seed.to(tl.uint32)) * 0x9E3779B9
    u = (u ^ (u >> 16)) * 0x85EBCA6B
    u = (u ^ (u >> 13)) * 0xC2B2AE35
    u = u ^ (u >> 16)
    return (u & 0x00FFFFFF).to(tl.float32) * (1.0 / 16777216.0) - 0.5


@triton.jit
def _triton_int8_gemm_kernel(
    A_ptr,
    B_ptr,
    SA_ptr,
    SB_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sam,
    stride_sak,
    stride_sbn,
    stride_sbk,
    stride_cm,
    stride_cn,
    IN_LOOP_PROMOTE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Coalesced SM80 mma.sync.m16n8k32.row.col.s32.s8.s8.s32 GEMM: C = A @ B^T.
    Supports both SwitchBack/Halo epilogue-only scaling (IN_LOOP_PROMOTE=False)
    and Jetfire/DeepSeek-V3 1x128 x 128x128 inner-K CUDA-core promotion (IN_LOOP_PROMOTE=True).
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = rm % M
    offs_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    if IN_LOOP_PROMOTE:
        acc_f32 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        nb_idx = pid_n * BLOCK_N // 128
        for kt in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            dot_i32 = tl.dot(a, tl.trans(b))
            kg = (kt * BLOCK_K) // 128
            sa = tl.load(SA_ptr + offs_m * stride_sam + kg * stride_sak).to(tl.float32)
            sb = tl.load(SB_ptr + nb_idx * stride_sbn + kg * stride_sbk).to(tl.float32)
            acc_f32 += dot_i32.to(tl.float32) * (sa[:, None] * sb)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        out = acc_f32.to(tl.bfloat16)
    else:
        acc_i32 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc_i32 += tl.dot(a, tl.trans(b))
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        sa = tl.load(SA_ptr + offs_m * stride_sam).to(tl.float32)
        sb = tl.load(SB_ptr + offs_n * stride_sbn).to(tl.float32)
        out = (acc_i32.to(tl.float32) * (sa[:, None] * sb[None, :])).to(tl.bfloat16)

    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, out, mask=mask_c)


@triton.jit
def _jetfire_block128_quant_act_kernel(
    X_ptr,
    Q_ptr,
    S_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_qm,
    stride_qk,
    stride_sm,
    stride_sk,
    SEED,
    STOCHASTIC_ROUND: tl.constexpr,
):
    """Per 1x128 block quantization of [M, K] BF16 -> [M, K] INT8 + [M, K//128] FP32."""
    pid_m = tl.program_id(0)
    pid_kb = tl.program_id(1)
    offs_m = pid_m * 64 + tl.arange(0, 64)
    offs_k = pid_kb * 128 + tl.arange(0, 128)
    mask_m = offs_m < M
    x = tl.load(
        X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    scale = tl.maximum(amax, 1e-8) / 127.0
    tl.store(S_ptr + offs_m * stride_sm + pid_kb * stride_sk, scale, mask=mask_m)
    scaled = x * (1.0 / scale)[:, None]
    if STOCHASTIC_ROUND:
        rnd = _fast_sr_noise(SEED, offs_m[:, None] * K + offs_k[None, :])
        qf = tl.extra.cuda.libdevice.rint(scaled + rnd)
    else:
        qf = tl.extra.cuda.libdevice.rint(scaled)
    q = tl.minimum(tl.maximum(qf, -127.0), 127.0).to(tl.int8)
    tl.store(
        Q_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
        q,
        mask=mask_m[:, None],
    )


@triton.jit
def _jetfire_block128_quant_wt_kernel(
    W_ptr,
    Q_ptr,
    S_ptr,
    N,
    K,
    stride_wn,
    stride_wk,
    stride_qn,
    stride_qk,
    stride_sn,
    stride_sk,
    TRANSPOSE_OUT: tl.constexpr,
):
    """Per 128x128 block quantization of [N, K] BF16 -> [N, K] (or [K, N]) INT8 + block scales."""
    pid_nb = tl.program_id(0)
    pid_kb = tl.program_id(1)
    offs_n = pid_nb * 128 + tl.arange(0, 128)
    offs_k = pid_kb * 128 + tl.arange(0, 128)
    w = tl.load(
        W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    ).to(tl.float32)
    amax = tl.max(tl.abs(w))
    scale = tl.maximum(amax, 1e-8) / 127.0
    q = tl.minimum(
        tl.maximum(tl.extra.cuda.libdevice.rint(w / scale), -127.0), 127.0
    ).to(tl.int8)
    if TRANSPOSE_OUT:
        tl.store(S_ptr + pid_kb * stride_sn + pid_nb * stride_sk, scale)
        tl.store(
            Q_ptr + offs_k[:, None] * stride_qn + offs_n[None, :] * stride_qk,
            tl.trans(q),
        )
    else:
        tl.store(S_ptr + pid_nb * stride_sn + pid_kb * stride_sk, scale)
        tl.store(
            Q_ptr + offs_n[:, None] * stride_qn + offs_k[None, :] * stride_qk,
            q,
        )


@triton.jit
def _naive_spilling_fp32_atomic_dual_had_amax_kernel(
    A_ptr,
    H_ptr,
    SignR_ptr,
    SignC_ptr,
    RowAmaxF32_ptr,
    ColAmaxF32_ptr,
    R,
    C,
    stride_ar,
    stride_ac,
):
    """Baseline negative-control quantizer pass demonstrating SM80 FP32 atomicCAS spin-loop."""
    pid_rb = tl.program_id(0)
    pid_cb = tl.program_id(1)
    offs_r = pid_rb * 128 + tl.arange(0, 128)
    offs_c = pid_cb * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 128)
    a = tl.load(
        A_ptr + offs_r[:, None] * stride_ar + offs_c[None, :] * stride_ac
    ).to(tl.bfloat16)
    h = tl.load(H_ptr + offs_k[:, None] * 128 + offs_k[None, :])
    sc = tl.load(SignC_ptr + offs_c).to(tl.bfloat16)
    r_rot = tl.dot((a * sc[None, :]).to(tl.bfloat16), h)
    tl.atomic_max(RowAmaxF32_ptr + offs_r, tl.max(tl.abs(r_rot), axis=1))
    sr = tl.load(SignR_ptr + offs_r).to(tl.bfloat16)
    c_rot = tl.dot(tl.trans((a * sr[:, None]).to(tl.bfloat16)), h)
    tl.atomic_max(ColAmaxF32_ptr + offs_c, tl.max(tl.abs(c_rot), axis=1))


@triton.jit
def _naive_spilling_philox10_dual_had_quant_kernel(
    A_ptr,
    H_ptr,
    SignR_ptr,
    SignC_ptr,
    RowAmaxF32_ptr,
    ColAmaxF32_ptr,
    QRow_ptr,
    SRow_ptr,
    QColT_ptr,
    SColT_ptr,
    R,
    C,
    stride_ar,
    stride_ac,
    stride_qrr,
    stride_qrc,
    stride_qcc,
    stride_qcr,
    SEED,
):
    """Baseline negative-control dual quantizer (>255 reg local-memory spill + 10-round Philox)."""
    pid_rb = tl.program_id(0)
    pid_cb = tl.program_id(1)
    offs_r = pid_rb * 128 + tl.arange(0, 128)
    offs_c = pid_cb * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 128)
    a = tl.load(
        A_ptr + offs_r[:, None] * stride_ar + offs_c[None, :] * stride_ac
    ).to(tl.bfloat16)
    h = tl.load(H_ptr + offs_k[:, None] * 128 + offs_k[None, :])

    sc = tl.load(SignC_ptr + offs_c).to(tl.bfloat16)
    r_rot = tl.dot((a * sc[None, :]).to(tl.bfloat16), h)
    r_scale = tl.maximum(tl.load(RowAmaxF32_ptr + offs_r), 1e-8) / 127.0
    if pid_cb == 0:
        tl.store(SRow_ptr + offs_r, r_scale)
    rnd_r = tl.rand(SEED, offs_r[:, None] * C + offs_c[None, :]) - 0.5
    qr = tl.minimum(
        tl.maximum(
            tl.extra.cuda.libdevice.rint(r_rot / r_scale[:, None] + rnd_r), -127.0
        ),
        127.0,
    ).to(tl.int8)
    tl.store(QRow_ptr + offs_r[:, None] * stride_qrr + offs_c[None, :] * stride_qrc, qr)

    sr = tl.load(SignR_ptr + offs_r).to(tl.bfloat16)
    c_rot = tl.dot(tl.trans((a * sr[:, None]).to(tl.bfloat16)), h)
    c_scale = tl.maximum(tl.load(ColAmaxF32_ptr + offs_c), 1e-8) / 127.0
    if pid_rb == 0:
        tl.store(SColT_ptr + offs_c, c_scale)
    rnd_c = tl.rand(SEED + 7919, offs_c[:, None] * R + offs_r[None, :]) - 0.5
    qc = tl.minimum(
        tl.maximum(
            tl.extra.cuda.libdevice.rint(c_rot / c_scale[:, None] + rnd_c), -127.0
        ),
        127.0,
    ).to(tl.int8)
    tl.store(QColT_ptr + offs_c[:, None] * stride_qcc + offs_r[None, :] * stride_qcr, qc)


@triton.jit
def _row_had128_amax_64x128(
    A_ptr,
    H_ptr,
    SignC_ptr,
    RowAmaxI32_ptr,
    R,
    C,
    stride_ar,
    stride_ac,
    USE_HADAMARD: tl.constexpr,
):
    pid_rb = tl.program_id(0)
    pid_cb = tl.program_id(1)
    offs_r = pid_rb * 64 + tl.arange(0, 64)
    offs_c = pid_cb * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 128)
    a = tl.load(
        A_ptr + offs_r[:, None] * stride_ar + offs_c[None, :] * stride_ac
    ).to(tl.bfloat16)
    if USE_HADAMARD:
        sc = tl.load(SignC_ptr + offs_c).to(tl.bfloat16)
        h = tl.load(H_ptr + offs_k[:, None] * 128 + offs_k[None, :])
        r_rot = tl.dot((a * sc[None, :]).to(tl.bfloat16), h)
    else:
        r_rot = a.to(tl.float32)
    tl.atomic_max(
        RowAmaxI32_ptr + offs_r,
        tl.max(tl.abs(r_rot), axis=1).to(tl.int32, bitcast=True),
    )


@triton.jit
def _row_had128_quant_64x128(
    A_ptr,
    H_ptr,
    SignC_ptr,
    RowAmaxI32_ptr,
    QRow_ptr,
    SRow_ptr,
    R,
    C,
    stride_ar,
    stride_ac,
    stride_qrr,
    stride_qrc,
    SEED,
    USE_HADAMARD: tl.constexpr,
    STOCHASTIC_ROUND: tl.constexpr,
):
    pid_rb = tl.program_id(0)
    pid_cb = tl.program_id(1)
    offs_r = pid_rb * 64 + tl.arange(0, 64)
    offs_c = pid_cb * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 128)
    a = tl.load(
        A_ptr + offs_r[:, None] * stride_ar + offs_c[None, :] * stride_ac
    ).to(tl.bfloat16)
    if USE_HADAMARD:
        sc = tl.load(SignC_ptr + offs_c).to(tl.bfloat16)
        h = tl.load(H_ptr + offs_k[:, None] * 128 + offs_k[None, :])
        r_rot = tl.dot((a * sc[None, :]).to(tl.bfloat16), h)
    else:
        r_rot = a.to(tl.float32)
    r_amax = tl.load(RowAmaxI32_ptr + offs_r).to(tl.float32, bitcast=True)
    r_scale = tl.maximum(r_amax, 1e-8) / 127.0
    if pid_cb == 0:
        tl.store(SRow_ptr + offs_r, r_scale)
    scaled = r_rot * (1.0 / r_scale)[:, None]
    if STOCHASTIC_ROUND:
        rnd = _fast_sr_noise(SEED, offs_r[:, None] * C + offs_c[None, :])
        qf = tl.extra.cuda.libdevice.rint(scaled + rnd)
    else:
        qf = tl.extra.cuda.libdevice.rint(scaled)
    qr = tl.minimum(tl.maximum(qf, -127.0), 127.0).to(tl.int8)
    tl.store(QRow_ptr + offs_r[:, None] * stride_qrr + offs_c[None, :] * stride_qrc, qr)


@triton.jit
def _col_had128_amax_128x64(
    A_ptr,
    Sin_ptr,
    H_ptr,
    SignR_ptr,
    ColAmaxI32_ptr,
    R,
    C,
    stride_ar,
    stride_ac,
    INPUT_IS_INT8: tl.constexpr,
    USE_HADAMARD: tl.constexpr,
):
    pid_rb = tl.program_id(0)
    pid_cb = tl.program_id(1)
    offs_r = pid_rb * 128 + tl.arange(0, 128)
    offs_c = pid_cb * 64 + tl.arange(0, 64)
    offs_k = tl.arange(0, 128)
    a_raw = tl.load(A_ptr + offs_r[:, None] * stride_ar + offs_c[None, :] * stride_ac)
    if INPUT_IS_INT8:
        sin = tl.load(Sin_ptr + offs_r).to(tl.float32)
        a = (a_raw.to(tl.float32) * sin[:, None]).to(tl.bfloat16)
    else:
        a = a_raw.to(tl.bfloat16)
    if USE_HADAMARD:
        sr = tl.load(SignR_ptr + offs_r).to(tl.bfloat16)
        h = tl.load(H_ptr + offs_k[:, None] * 128 + offs_k[None, :])
        c_rot = tl.dot(tl.trans((a * sr[:, None]).to(tl.bfloat16)), h)
    else:
        c_rot = tl.trans(a).to(tl.float32)
    tl.atomic_max(
        ColAmaxI32_ptr + offs_c,
        tl.max(tl.abs(c_rot), axis=1).to(tl.int32, bitcast=True),
    )


@triton.jit
def _col_had128_quant_128x64(
    A_ptr,
    Sin_ptr,
    H_ptr,
    SignR_ptr,
    ColAmaxI32_ptr,
    QColT_ptr,
    SColT_ptr,
    R,
    C,
    stride_ar,
    stride_ac,
    stride_qcc,
    stride_qcr,
    SEED,
    INPUT_IS_INT8: tl.constexpr,
    USE_HADAMARD: tl.constexpr,
    STOCHASTIC_ROUND: tl.constexpr,
):
    pid_rb = tl.program_id(0)
    pid_cb = tl.program_id(1)
    offs_r = pid_rb * 128 + tl.arange(0, 128)
    offs_c = pid_cb * 64 + tl.arange(0, 64)
    offs_k = tl.arange(0, 128)
    a_raw = tl.load(A_ptr + offs_r[:, None] * stride_ar + offs_c[None, :] * stride_ac)
    if INPUT_IS_INT8:
        sin = tl.load(Sin_ptr + offs_r).to(tl.float32)
        a = (a_raw.to(tl.float32) * sin[:, None]).to(tl.bfloat16)
    else:
        a = a_raw.to(tl.bfloat16)
    if USE_HADAMARD:
        sr = tl.load(SignR_ptr + offs_r).to(tl.bfloat16)
        h = tl.load(H_ptr + offs_k[:, None] * 128 + offs_k[None, :])
        c_rot = tl.dot(tl.trans((a * sr[:, None]).to(tl.bfloat16)), h)
    else:
        c_rot = tl.trans(a).to(tl.float32)
    c_amax = tl.load(ColAmaxI32_ptr + offs_c).to(tl.float32, bitcast=True)
    c_scale = tl.maximum(c_amax, 1e-8) / 127.0
    if pid_rb == 0:
        tl.store(SColT_ptr + offs_c, c_scale)
    scaled_c = c_rot * (1.0 / c_scale)[:, None]
    if STOCHASTIC_ROUND:
        rnd_c = _fast_sr_noise(SEED + 7919, offs_c[:, None] * R + offs_r[None, :])
        qfc = tl.extra.cuda.libdevice.rint(scaled_c + rnd_c)
    else:
        qfc = tl.extra.cuda.libdevice.rint(scaled_c)
    qc = tl.minimum(tl.maximum(qfc, -127.0), 127.0).to(tl.int8)
    tl.store(QColT_ptr + offs_c[:, None] * stride_qcc + offs_r[None, :] * stride_qcr, qc)


@triton.jit
def _col_had128_nontransposed_shard_quant_128x64(
    A_ptr,
    H_ptr,
    SignR_ptr,
    ColAmaxI32_ptr,
    QShard_ptr,
    SScale_ptr,
    R_shard,
    C,
    stride_ar,
    stride_ac,
    stride_qsr,
    stride_qsc,
):
    """For 8-GPU Quantized-AllGather FSDP2 Bwd: rotates local [N/8, K] shard along N
    using globally reduced ColAmaxI32[K] and stores [N/8, K] INT8 ready for AllGather!
    """
    pid_rb = tl.program_id(0)
    pid_cb = tl.program_id(1)
    offs_r = pid_rb * 128 + tl.arange(0, 128)
    offs_c = pid_cb * 64 + tl.arange(0, 64)
    offs_k = tl.arange(0, 128)

    a = tl.load(
        A_ptr + offs_r[:, None] * stride_ar + offs_c[None, :] * stride_ac
    ).to(tl.bfloat16)
    sr = tl.load(SignR_ptr + offs_r).to(tl.bfloat16)
    h = tl.load(H_ptr + offs_k[:, None] * 128 + offs_k[None, :])
    # H128 is symmetric: ((A * sr)^T @ H128)^T = H128 @ (A * sr) of shape [128_r, 64_c]!
    n_rot = tl.dot(h, (a * sr[:, None]).to(tl.bfloat16))
    c_amax = tl.load(ColAmaxI32_ptr + offs_c).to(tl.float32, bitcast=True)
    c_scale = tl.maximum(c_amax, 1e-8) / 127.0
    if pid_rb == 0:
        tl.store(SScale_ptr + offs_c, c_scale)
    qf = tl.extra.cuda.libdevice.rint(n_rot * (1.0 / c_scale)[None, :])
    q = tl.minimum(tl.maximum(qf, -127.0), 127.0).to(tl.int8)
    tl.store(
        QShard_ptr + offs_r[:, None] * stride_qsr + offs_c[None, :] * stride_qsc, q
    )


@triton.jit
def _transpose_int8_128x128_kernel(
    In_ptr,
    Out_ptr,
    R,
    C,
    stride_ir,
    stride_ic,
    stride_oc,
    stride_or,
):
    """SRAM-tiled 128x128 INT8 matrix transpose [R, C] -> [C, R] after INT8 AllGather."""
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_r = pid_r * 128 + tl.arange(0, 128)
    offs_c = pid_c * 128 + tl.arange(0, 128)
    tile = tl.load(In_ptr + offs_r[:, None] * stride_ir + offs_c[None, :] * stride_ic)
    tl.store(
        Out_ptr + offs_c[:, None] * stride_oc + offs_r[None, :] * stride_or,
        tl.trans(tile),
    )


@triton.jit
def _inv_row_had128_64x128(
    Y_ptr,
    H_ptr,
    Sign_ptr,
    Out_ptr,
    R,
    C,
    stride_yr,
    stride_yc,
    stride_or,
    stride_oc,
    OUT_FP32: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    pid_rb = tl.program_id(0)
    pid_cb = tl.program_id(1)
    offs_r = pid_rb * 64 + tl.arange(0, 64)
    offs_c = pid_cb * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 128)
    h = tl.load(H_ptr + offs_k[:, None] * 128 + offs_k[None, :])
    s = tl.load(Sign_ptr + offs_c).to(tl.float32)
    y = tl.load(
        Y_ptr + offs_r[:, None] * stride_yr + offs_c[None, :] * stride_yc
    ).to(tl.bfloat16)
    val = tl.dot(y, h) * s[None, :]
    out_ptrs = Out_ptr + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc
    if ACCUMULATE:
        prev = tl.load(out_ptrs).to(tl.float32)
        val = prev + val
    if OUT_FP32:
        tl.store(out_ptrs, val)
    else:
        tl.store(out_ptrs, val.to(tl.bfloat16))


@triton.jit
def _dequant_inv_row_had128_64x128(
    Q_ptr,
    Scale_ptr,
    H_ptr,
    Sign_ptr,
    Out_ptr,
    R,
    C,
    stride_qr,
    stride_qc,
    stride_or,
    stride_oc,
):
    """Reconstructs BF16 X from saved COAT INT8 (Qx, sx) via inverse H128 on Tensor Cores."""
    pid_rb = tl.program_id(0)
    pid_cb = tl.program_id(1)
    offs_r = pid_rb * 64 + tl.arange(0, 64)
    offs_c = pid_cb * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 128)
    q = tl.load(Q_ptr + offs_r[:, None] * stride_qr + offs_c[None, :] * stride_qc)
    sx = tl.load(Scale_ptr + offs_r).to(tl.float32)
    x_rot = (q.to(tl.float32) * sx[:, None]).to(tl.bfloat16)
    h = tl.load(H_ptr + offs_k[:, None] * 128 + offs_k[None, :])
    s = tl.load(Sign_ptr + offs_c).to(tl.float32)
    out = (tl.dot(x_rot, h) * s[None, :]).to(tl.bfloat16)
    tl.store(Out_ptr + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc, out)


@triton.jit
def _kronecker_hadamard_quant_row_kernel(
    X_ptr,
    H1_ptr,
    H2_ptr,
    Sign_ptr,
    Q_ptr,
    Scale_ptr,
    M,
    stride_xm,
    stride_qm,
    SEED,
    STOCHASTIC_ROUND: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
):
    """Single-CTA In-Register 2D Kronecker Tensor-Core Randomized Hadamard:
    vec(H_D1 @ (X_m * S) @ H_D2) = ((H_D1 (x) H_D2) diag(S)) x_m -> INT8 + FP32 scale!
    """
    row = tl.program_id(0)
    r1 = tl.arange(0, D1)
    r2 = tl.arange(0, D2)
    offs = r1[:, None] * D2 + r2[None, :]

    x = tl.load(X_ptr + row * stride_xm + offs).to(tl.bfloat16)
    s = tl.load(Sign_ptr + offs).to(tl.bfloat16)
    xs = (x * s).to(tl.bfloat16)

    h1 = tl.load(H1_ptr + r1[:, None] * D1 + r1[None, :])
    h2 = tl.load(H2_ptr + r2[:, None] * D2 + r2[None, :])

    tmp = tl.dot(h1, xs).to(tl.bfloat16)
    y = tl.dot(tmp, h2)

    amax = tl.max(tl.abs(y))
    scale = tl.maximum(amax, 1e-8) / 127.0
    scaled = y / scale
    if STOCHASTIC_ROUND:
        rnd = _fast_sr_noise(SEED, row * (D1 * D2) + offs)
        qf = tl.extra.cuda.libdevice.rint(scaled + rnd)
    else:
        qf = tl.extra.cuda.libdevice.rint(scaled)
    q = tl.minimum(tl.maximum(qf, -127.0), 127.0).to(tl.int8)
    tl.store(Q_ptr + row * stride_qm + offs, q)
    tl.store(Scale_ptr + row, scale)


# ----------------------------------------------------------------------------
# Fused rstd-Invariant RMSNorm + Row-Hadamard-128 -> INT8 Forward & Backward
# ----------------------------------------------------------------------------


@triton.jit
def _fused_rmsnorm_had128_fwd_kernel(
    X_ptr,
    Gamma_ptr,
    H_ptr,
    Sign_ptr,
    QX_ptr,
    SX_ptr,
    Rstd_ptr,
    M,
    H_DIM,
    stride_xm,
    stride_qxm,
    EPS,
):
    """Exploits rstd linearity: (x * rstd * gamma * sign) @ H128 = rstd * ((x * gamma * sign) @ H128)!
    Pass 1 computes BOTH sum(x^2) AND unnormalized Hadamard amax in ONE HBM sweep!
    Pass 2 hits hot L2 cache (<28 MiB across 108 SMs) to store (QX [1 B/elem], SX, Rstd),
    eliminating saving BF16 X and BF16 X_norm!
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * 32 + tl.arange(0, 32)
    offs_k = tl.arange(0, 128)
    h = tl.load(H_ptr + offs_k[:, None] * 128 + offs_k[None, :])

    sum_sq = tl.zeros((32,), dtype=tl.float32)
    u_amax = tl.zeros((32,), dtype=tl.float32)
    num_hb = H_DIM // 128
    for hb in range(0, num_hb):
        offs_c = hb * 128 + offs_k
        x = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_c[None, :]).to(
            tl.float32
        )
        sum_sq += tl.sum(x * x, axis=1)
        gs = (
            tl.load(Gamma_ptr + offs_c).to(tl.float32)
            * tl.load(Sign_ptr + offs_c).to(tl.float32)
        ).to(tl.bfloat16)
        u_rot = tl.dot((x.to(tl.bfloat16) * gs[None, :]).to(tl.bfloat16), h)
        u_amax = tl.maximum(u_amax, tl.max(tl.abs(u_rot), axis=1))

    rstd = tl.math.rsqrt(sum_sq * (1.0 / H_DIM) + EPS)
    u_scale = tl.maximum(u_amax, 1e-8) / 127.0
    inv_u_scale = 1.0 / u_scale
    sx = rstd * u_scale
    tl.store(SX_ptr + offs_m, sx)
    tl.store(Rstd_ptr + offs_m, rstd)

    for hb in range(0, num_hb):
        offs_c = hb * 128 + offs_k
        x = tl.load(
            X_ptr + offs_m[:, None] * stride_xm + offs_c[None, :],
            cache_modifier=".ca",
        ).to(tl.bfloat16)
        gs = (
            tl.load(Gamma_ptr + offs_c).to(tl.float32)
            * tl.load(Sign_ptr + offs_c).to(tl.float32)
        ).to(tl.bfloat16)
        u_rot = tl.dot((x * gs[None, :]).to(tl.bfloat16), h)
        qf = tl.extra.cuda.libdevice.rint(u_rot * inv_u_scale[:, None])
        q = tl.minimum(tl.maximum(qf, -127.0), 127.0).to(tl.int8)
        tl.store(QX_ptr + offs_m[:, None] * stride_qxm + offs_c[None, :], q)


@triton.jit
def _fused_rmsnorm_from_qnorm_bwd_kernel(
    DY_ptr,
    QX_ptr,
    SX_ptr,
    Rstd_ptr,
    Gamma_ptr,
    H_ptr,
    Sign_ptr,
    DX_ptr,
    DGammaPartial_ptr,
    M,
    H_DIM,
    stride_dym,
    stride_qxm,
    stride_dxm,
    stride_dgp,
):
    """Reconstructs x_hat = ((QX * SX) @ H128 * Sign) / Gamma directly from 1 B/elem QX,
    computing exact RMSNorm dx and partial FP32 dgamma with ZERO saved BF16 X!
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * 32 + tl.arange(0, 32)
    offs_k = tl.arange(0, 128)
    h = tl.load(H_ptr + offs_k[:, None] * 128 + offs_k[None, :])
    sx = tl.load(SX_ptr + offs_m).to(tl.float32)
    rstd = tl.load(Rstd_ptr + offs_m).to(tl.float32)

    dot_dy_xhat = tl.zeros((32,), dtype=tl.float32)
    num_hb = H_DIM // 128
    for hb in range(0, num_hb):
        offs_c = hb * 128 + offs_k
        q = tl.load(QX_ptr + offs_m[:, None] * stride_qxm + offs_c[None, :])
        y_rot = (q.to(tl.float32) * sx[:, None]).to(tl.bfloat16)
        s = tl.load(Sign_ptr + offs_c).to(tl.float32)
        g = tl.load(Gamma_ptr + offs_c).to(tl.float32)
        y_norm = tl.dot(y_rot, h) * s[None, :]
        x_hat = y_norm / g[None, :]
        dy = tl.load(DY_ptr + offs_m[:, None] * stride_dym + offs_c[None, :]).to(
            tl.float32
        )
        dg_blk = tl.sum(dy * x_hat, axis=0)
        tl.store(DGammaPartial_ptr + pid_m * stride_dgp + offs_c, dg_blk)
        dot_dy_xhat += tl.sum((dy * g[None, :]) * x_hat, axis=1)

    mean_dot = dot_dy_xhat * (1.0 / H_DIM)
    for hb in range(0, num_hb):
        offs_c = hb * 128 + offs_k
        q = tl.load(
            QX_ptr + offs_m[:, None] * stride_qxm + offs_c[None, :],
            cache_modifier=".ca",
        )
        y_rot = (q.to(tl.float32) * sx[:, None]).to(tl.bfloat16)
        s = tl.load(Sign_ptr + offs_c).to(tl.float32)
        g = tl.load(Gamma_ptr + offs_c).to(tl.float32)
        x_hat = (tl.dot(y_rot, h) * s[None, :]) / g[None, :]
        dy = tl.load(
            DY_ptr + offs_m[:, None] * stride_dym + offs_c[None, :],
            cache_modifier=".ca",
        ).to(tl.float32)
        dx = rstd[:, None] * (dy * g[None, :] - x_hat * mean_dot[:, None])
        tl.store(
            DX_ptr + offs_m[:, None] * stride_dxm + offs_c[None, :],
            dx.to(tl.bfloat16),
        )


# ----------------------------------------------------------------------------
# COAT Fused SwiGLU + Block-INT8 Gate/Up Compression + Hadamard-128 Mid Quant
# ----------------------------------------------------------------------------


@triton.jit
def _coat_swiglu_fwd_amax_and_compress_gu_kernel(
    GU_ptr,
    H_ptr,
    SignI_ptr,
    QG_ptr,
    SG_ptr,
    QU_ptr,
    SU_ptr,
    MidAmaxI32_ptr,
    M,
    I_DIM,
    stride_gum,
    stride_qim,
    stride_sim,
):
    """Zero-spill 32x128 tile: reads BF16 gate_up [M, 2*I] ONCE:
    1. Compresses gate & up into 1x128 COAT INT8 (QG, SG, QU, SU: 1.016 B/elem saved!),
    2. Computes mid = silu(gate)*up, rotates (mid * SignI) @ H128 on Tensor Cores,
       and updates MidAmaxI32[M] via hardware RED.MAX.S32!
    """
    pid_mb = tl.program_id(0)
    pid_ib = tl.program_id(1)
    offs_m = pid_mb * 32 + tl.arange(0, 32)
    offs_i = pid_ib * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 128)

    g = tl.load(GU_ptr + offs_m[:, None] * stride_gum + offs_i[None, :]).to(tl.float32)
    u = tl.load(
        GU_ptr + offs_m[:, None] * stride_gum + (I_DIM + offs_i[None, :])
    ).to(tl.float32)

    sg = tl.maximum(tl.max(tl.abs(g), axis=1), 1e-8) / 127.0
    su = tl.maximum(tl.max(tl.abs(u), axis=1), 1e-8) / 127.0
    qg = tl.minimum(
        tl.maximum(tl.extra.cuda.libdevice.rint(g * (1.0 / sg)[:, None]), -127.0), 127.0
    ).to(tl.int8)
    qu = tl.minimum(
        tl.maximum(tl.extra.cuda.libdevice.rint(u * (1.0 / su)[:, None]), -127.0), 127.0
    ).to(tl.int8)

    tl.store(QG_ptr + offs_m[:, None] * stride_qim + offs_i[None, :], qg)
    tl.store(QU_ptr + offs_m[:, None] * stride_qim + offs_i[None, :], qu)
    tl.store(SG_ptr + offs_m * stride_sim + pid_ib, sg.to(tl.bfloat16))
    tl.store(SU_ptr + offs_m * stride_sim + pid_ib, su.to(tl.bfloat16))

    mid = (g * tl.sigmoid(g) * u).to(tl.bfloat16)
    si = tl.load(SignI_ptr + offs_i).to(tl.bfloat16)
    h = tl.load(H_ptr + offs_k[:, None] * 128 + offs_k[None, :])
    m_rot = tl.dot((mid * si[None, :]).to(tl.bfloat16), h)
    tl.atomic_max(
        MidAmaxI32_ptr + offs_m,
        tl.max(tl.abs(m_rot), axis=1).to(tl.int32, bitcast=True),
    )


@triton.jit
def _coat_swiglu_fwd_quant_mid_from_compressed_kernel(
    QG_ptr,
    SG_ptr,
    QU_ptr,
    SU_ptr,
    H_ptr,
    SignI_ptr,
    MidAmaxI32_ptr,
    QMid_ptr,
    SMid_ptr,
    M,
    I_DIM,
    stride_qim,
    stride_sim,
):
    """Zero-spill 32x128 Pass 2 reads 2x-smaller INT8 (QG, QU) from hot L2 cache to write QMid [M, I] INT8!"""
    pid_mb = tl.program_id(0)
    pid_ib = tl.program_id(1)
    offs_m = pid_mb * 32 + tl.arange(0, 32)
    offs_i = pid_ib * 128 + tl.arange(0, 128)
    offs_k = tl.arange(0, 128)

    qg = tl.load(QG_ptr + offs_m[:, None] * stride_qim + offs_i[None, :]).to(tl.float32)
    qu = tl.load(QU_ptr + offs_m[:, None] * stride_qim + offs_i[None, :]).to(tl.float32)
    sg = tl.load(SG_ptr + offs_m * stride_sim + pid_ib).to(tl.float32)
    su = tl.load(SU_ptr + offs_m * stride_sim + pid_ib).to(tl.float32)
    g = qg * sg[:, None]
    u = qu * su[:, None]

    mid = (g * tl.sigmoid(g) * u).to(tl.bfloat16)
    si = tl.load(SignI_ptr + offs_i).to(tl.bfloat16)
    h = tl.load(H_ptr + offs_k[:, None] * 128 + offs_k[None, :])
    m_rot = tl.dot((mid * si[None, :]).to(tl.bfloat16), h)

    m_amax = tl.load(MidAmaxI32_ptr + offs_m).to(tl.float32, bitcast=True)
    m_scale = tl.maximum(m_amax, 1e-8) / 127.0
    if pid_ib == 0:
        tl.store(SMid_ptr + offs_m, m_scale)
    qf = tl.extra.cuda.libdevice.rint(m_rot * (1.0 / m_scale)[:, None])
    qm = tl.minimum(tl.maximum(qf, -127.0), 127.0).to(tl.int8)
    tl.store(QMid_ptr + offs_m[:, None] * stride_qim + offs_i[None, :], qm)


@triton.jit
def _coat_swiglu_bwd_from_compressed_kernel(
    DMid_ptr,
    QG_ptr,
    SG_ptr,
    QU_ptr,
    SU_ptr,
    DGU_ptr,
    M,
    I_DIM,
    stride_dm,
    stride_qim,
    stride_sim,
    stride_dgum,
):
    """Reads d_mid [M, I] BF16 + COAT compressed INT8 (QG, QU) (half the HBM bytes!)
    and writes d_gate_up [M, 2*I] BF16.
    """
    pid_mb = tl.program_id(0)
    pid_ib = tl.program_id(1)
    offs_m = pid_mb * 64 + tl.arange(0, 64)
    offs_i = pid_ib * 128 + tl.arange(0, 128)

    dm = tl.load(DMid_ptr + offs_m[:, None] * stride_dm + offs_i[None, :]).to(
        tl.float32
    )
    qg = tl.load(QG_ptr + offs_m[:, None] * stride_qim + offs_i[None, :]).to(tl.float32)
    qu = tl.load(QU_ptr + offs_m[:, None] * stride_qim + offs_i[None, :]).to(tl.float32)
    sg = tl.load(SG_ptr + offs_m * stride_sim + pid_ib).to(tl.float32)
    su = tl.load(SU_ptr + offs_m * stride_sim + pid_ib).to(tl.float32)
    g = qg * sg[:, None]
    u = qu * su[:, None]

    sig = tl.sigmoid(g)
    silu_g = g * sig
    du = (dm * silu_g).to(tl.bfloat16)
    dg = (dm * u * sig * (1.0 + g * (1.0 - sig))).to(tl.bfloat16)

    tl.store(DGU_ptr + offs_m[:, None] * stride_dgum + offs_i[None, :], dg)
    tl.store(DGU_ptr + offs_m[:, None] * stride_dgum + (I_DIM + offs_i[None, :]), du)


# ============================================================================
# 2. High-Level Kernel Dispatchers & Autograd Wrappers
# ============================================================================


class HaloWorkspace:
    """Reusable zero-allocation scratchpad for atomic amax reductions."""

    def __init__(self, max_dim: int, device: torch.device) -> None:
        self.amax_i32 = torch.zeros((max_dim,), device=device, dtype=torch.int32)
        self.H128 = sylvester_hadamard(128, device=device, dtype=torch.bfloat16)

    def quant_row(
        self,
        inp: torch.Tensor,
        sign_c: torch.Tensor,
        use_hadamard: bool = True,
        sr: bool = False,
        seed: int = 17,
        q_out: torch.Tensor | None = None,
        s_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        R, C = inp.shape
        if q_out is None:
            q_out = torch.empty((R, C), device=inp.device, dtype=torch.int8)
        if s_out is None:
            s_out = torch.empty((R, 1), device=inp.device, dtype=torch.float32)
        ra = self.amax_i32[:R]
        ra.zero_()
        g = (R // 64, C // 128)
        _row_had128_amax_64x128[g](
            inp,
            self.H128,
            sign_c,
            ra,
            R,
            C,
            inp.stride(0),
            inp.stride(1),
            USE_HADAMARD=use_hadamard,
            num_warps=4,
        )
        _row_had128_quant_64x128[g](
            inp,
            self.H128,
            sign_c,
            ra,
            q_out,
            s_out,
            R,
            C,
            inp.stride(0),
            inp.stride(1),
            q_out.stride(0),
            q_out.stride(1),
            seed,
            USE_HADAMARD=use_hadamard,
            STOCHASTIC_ROUND=sr,
            num_warps=4,
        )
        return q_out, s_out

    def quant_col_transposed(
        self,
        inp: torch.Tensor,
        sign_r: torch.Tensor,
        sin: torch.Tensor | None = None,
        use_hadamard: bool = True,
        sr: bool = False,
        seed: int = 31,
        qt_out: torch.Tensor | None = None,
        st_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        R, C = inp.shape
        is_i8 = inp.dtype == torch.int8
        sin_t = sin if is_i8 else inp
        if qt_out is None:
            qt_out = torch.empty((C, R), device=inp.device, dtype=torch.int8)
        if st_out is None:
            st_out = torch.empty((C, 1), device=inp.device, dtype=torch.float32)
        ca = self.amax_i32[:C]
        ca.zero_()
        g = (R // 128, C // 64)
        _col_had128_amax_128x64[g](
            inp,
            sin_t,
            self.H128,
            sign_r,
            ca,
            R,
            C,
            inp.stride(0),
            inp.stride(1),
            INPUT_IS_INT8=is_i8,
            USE_HADAMARD=use_hadamard,
            num_warps=4,
        )
        _col_had128_quant_128x64[g](
            inp,
            sin_t,
            self.H128,
            sign_r,
            ca,
            qt_out,
            st_out,
            R,
            C,
            inp.stride(0),
            inp.stride(1),
            qt_out.stride(0),
            qt_out.stride(1),
            seed,
            INPUT_IS_INT8=is_i8,
            USE_HADAMARD=use_hadamard,
            STOCHASTIC_ROUND=sr,
            num_warps=4,
        )
        return qt_out, st_out

    def inv_row_had128(
        self,
        y_rot: torch.Tensor,
        sign_c: torch.Tensor,
        out: torch.Tensor | None = None,
        out_fp32: bool = False,
        accumulate: bool = False,
    ) -> torch.Tensor:
        R, C = y_rot.shape
        if out is None:
            out = torch.empty(
                (R, C),
                device=y_rot.device,
                dtype=torch.float32 if out_fp32 else torch.bfloat16,
            )
        _inv_row_had128_64x128[(R // 64, C // 128)](
            y_rot,
            self.H128,
            sign_c,
            out,
            R,
            C,
            y_rot.stride(0),
            y_rot.stride(1),
            out.stride(0),
            out.stride(1),
            OUT_FP32=out_fp32,
            ACCUMULATE=accumulate,
            num_warps=4,
        )
        return out


def triton_epilogue_int8_mm(
    a_i8: torch.Tensor,
    b_row_i8: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
) -> torch.Tensor:
    M, K = a_i8.shape
    N, _ = b_row_i8.shape
    out = torch.empty((M, N), device=a_i8.device, dtype=torch.bfloat16)
    bm, bn, bk = 128, 256, 128
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _triton_int8_gemm_kernel[grid](
        a_i8,
        b_row_i8,
        sa,
        sb,
        out,
        M,
        N,
        K,
        a_i8.stride(0),
        a_i8.stride(1),
        b_row_i8.stride(0),
        b_row_i8.stride(1),
        sa.stride(0),
        0,
        sb.stride(0),
        0,
        out.stride(0),
        out.stride(1),
        IN_LOOP_PROMOTE=False,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        num_warps=8,
        num_stages=3,
    )
    return out


def triton_jetfire_block128_mm(
    a_i8: torch.Tensor,
    b_row_i8: torch.Tensor,
    sa_blk: torch.Tensor,
    sb_blk: torch.Tensor,
) -> torch.Tensor:
    M, K = a_i8.shape
    N, _ = b_row_i8.shape
    out = torch.empty((M, N), device=a_i8.device, dtype=torch.bfloat16)
    bm, bn, bk = 128, 128, 128
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _triton_int8_gemm_kernel[grid](
        a_i8,
        b_row_i8,
        sa_blk,
        sb_blk,
        out,
        M,
        N,
        K,
        a_i8.stride(0),
        a_i8.stride(1),
        b_row_i8.stride(0),
        b_row_i8.stride(1),
        sa_blk.stride(0),
        sa_blk.stride(1),
        sb_blk.stride(0),
        sb_blk.stride(1),
        out.stride(0),
        out.stride(1),
        IN_LOOP_PROMOTE=True,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=8,
        num_warps=8,
        num_stages=3,
    )
    return out


# ============================================================================
# 3. Suite 0: Hardware-PTX, Outlier, SR-INT8 Accumulation, & FP64 Audits
# ============================================================================


def run_suite_0_audits(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    ws = HaloWorkspace(65536, device)
    M, K, N = (2048, 4096, 14336) if is_smoke else (4096, 4096, 14336)
    warmup = 4 if is_smoke else 8
    iters = 10 if is_smoke else 25
    flops_1mm = 2.0 * M * N * K

    torch.manual_seed(12300)
    x_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.02
    x_i8 = torch.randint(-127, 127, (M, K), device=device, dtype=torch.int8)
    w_i8_row = torch.randint(-127, 127, (N, K), device=device, dtype=torch.int8)
    w_i8_col = w_i8_row.T  # zero-copy col-major [K, N] view!
    sx = torch.full((M, 1), 0.01, device=device, dtype=torch.float32)
    sw = torch.full((N, 1), 0.01, device=device, dtype=torch.float32)
    sa_blk = torch.full((M, K // 128), 0.01, device=device, dtype=torch.float32)
    sb_blk = torch.full((N // 128, K // 128), 0.01, device=device, dtype=torch.float32)

    def _gemm_stat(fn: Callable[[], Any]) -> dict[str, float]:
        st = time_fn_with_peak_mem(fn, warmup, iters, device)
        st["tops_or_tflops"] = round(flops_1mm / (st["median_ms"] * 1e-3) / 1e12, 2)
        return st

    # 0.1 Resolving the E6 torch._int_mm Negative Control & SM80 INT8 Roofline
    roofline = {
        "shape": {"M": M, "K": K, "N": N},
        "bf16_cublas": _gemm_stat(lambda: x_bf16 @ w_bf16.T),
        "torch_int_mm_e6_contiguous_copy_and_dequant": _gemm_stat(
            lambda: (
                torch._int_mm(x_i8, w_i8_row.t().contiguous()).float()
                * (sx * sw.T)
            ).to(torch.bfloat16)
        ),
        "torch_int_mm_col_major_raw_s32": _gemm_stat(
            lambda: torch._int_mm(x_i8, w_i8_col)
        ),
        "triton_jetfire_inloop_promote_1x128x128": _gemm_stat(
            lambda: triton_jetfire_block128_mm(x_i8, w_i8_row, sa_blk, sb_blk)
        ),
        "triton_epilogue_only_coalesced_int8": _gemm_stat(
            lambda: triton_epilogue_int8_mm(x_i8, w_i8_row, sx, sw)
        ),
        "cutlass_sm80_scaled_mm_int8_to_bf16": _gemm_stat(
            lambda: vllm_ops.cutlass_scaled_mm(
                x_i8, w_i8_col, sx, sw.T, torch.bfloat16
            )
        ),
    }

    # 0.2 SM80 PTX Atomic & Register-Spill Quantizer Ablation on dY [M, N]
    dy_ab = torch.randn(M, N, device=device, dtype=torch.bfloat16)
    sm_ab = rademacher_signs(M, 101, device)
    sn_ab = rademacher_signs(N, 102, device)
    ra_f32 = torch.zeros(M, device=device, dtype=torch.float32)
    ca_f32 = torch.zeros(N, device=device, dtype=torch.float32)
    qdy_r = torch.empty((M, N), device=device, dtype=torch.int8)
    sdy_r = torch.empty((M, 1), device=device, dtype=torch.float32)
    qdy_c = torch.empty((N, M), device=device, dtype=torch.int8)
    sdy_c = torch.empty((N, 1), device=device, dtype=torch.float32)

    def _run_naive_spilling_dual() -> None:
        ra_f32.zero_()
        ca_f32.zero_()
        g = (M // 128, N // 128)
        _naive_spilling_fp32_atomic_dual_had_amax_kernel[g](
            dy_ab,
            ws.H128,
            sm_ab,
            sn_ab,
            ra_f32,
            ca_f32,
            M,
            N,
            dy_ab.stride(0),
            dy_ab.stride(1),
            num_warps=4,
        )
        _naive_spilling_philox10_dual_had_quant_kernel[g](
            dy_ab,
            ws.H128,
            sm_ab,
            sn_ab,
            ra_f32,
            ca_f32,
            qdy_r,
            sdy_r,
            qdy_c,
            sdy_c,
            M,
            N,
            dy_ab.stride(0),
            dy_ab.stride(1),
            qdy_r.stride(0),
            qdy_r.stride(1),
            qdy_c.stride(0),
            qdy_c.stride(1),
            77,
            num_warps=4,
        )

    def _run_fast_split_s32_dual() -> None:
        ws.quant_row(dy_ab, sn_ab, use_hadamard=True, sr=True, seed=77, q_out=qdy_r, s_out=sdy_r)
        ws.quant_col_transposed(
            dy_ab, sm_ab, use_hadamard=True, sr=True, seed=88, qt_out=qdy_c, st_out=sdy_c
        )

    t_naive_q = time_fn_with_peak_mem(_run_naive_spilling_dual, 2, 5 if is_smoke else 10, device)
    t_fast_q = time_fn_with_peak_mem(_run_fast_split_s32_dual, warmup, iters, device)
    quantizer_ablation = {
        "shape": {"M": M, "N": N},
        "naive_fp32_atomic_philox10_spilling_ms": t_naive_q["median_ms"],
        "split_s32_bitcast_splitmix_zero_spill_ms": t_fast_q["median_ms"],
        "speedup": round(
            t_naive_q["median_ms"] / max(t_fast_q["median_ms"], 1e-6), 4
        ),
    }

    # 0.3 Outlier Stress Fidelity (3-GEMM + 2D Kronecker vs Jetfire) & 64-Step SR-INT8 vs RTN-INT8
    M_o, K_o, N_o = 2048, 4096, 16384
    torch.manual_seed(12303)
    xo = (torch.randn(M_o, K_o, device=device, dtype=torch.float32) * 0.5)
    out_cols = torch.tensor([3, 17, 129, 511, 1024, 2049, 3111, 4001], device=device)
    xo[:, out_cols] *= 65.0
    xo_bf16 = xo.to(torch.bfloat16)
    wo = (torch.randn(N_o, K_o, device=device, dtype=torch.float32) * 0.02)
    wo[:, out_cols[:4]] *= 12.0
    wo_bf16 = wo.to(torch.bfloat16)
    dyo = torch.randn(M_o, N_o, device=device, dtype=torch.float32) * 0.5
    dyo[[0, 64, 256], :] *= 45.0
    dyo[:, [5, 1025, 8193]] *= 40.0
    dyo_bf16 = dyo.to(torch.bfloat16)

    y_ref = xo_bf16.float() @ wo_bf16.float().T
    dx_ref = dyo_bf16.float() @ wo_bf16.float()
    dw_ref = dyo_bf16.float().T @ xo_bf16.float()

    sk_o = rademacher_signs(K_o, 201, device)
    sn_o = rademacher_signs(N_o, 202, device)
    sm_o = rademacher_signs(M_o, 203, device)

    def _eval_3gemm_parity(use_had: bool, sr: bool) -> dict[str, float]:
        qx_t, sx_t = ws.quant_row(xo_bf16, sk_o, use_hadamard=use_had, sr=False, seed=1)
        qw_r, sw_r = ws.quant_row(wo_bf16, sk_o, use_hadamard=use_had, sr=False, seed=2)
        y_t = vllm_ops.cutlass_scaled_mm(qx_t, qw_r.T, sx_t, sw_r.T, torch.bfloat16)

        qdy_r_t, sdy_r_t = ws.quant_row(dyo_bf16, sn_o, use_hadamard=use_had, sr=sr, seed=3)
        qw_c, sw_c = ws.quant_col_transposed(wo_bf16, sn_o, use_hadamard=use_had, sr=False, seed=4)
        dx_t = vllm_ops.cutlass_scaled_mm(qdy_r_t, qw_c.T, sdy_r_t, sw_c.T, torch.bfloat16)

        qdy_c_t, sdy_c_t = ws.quant_col_transposed(dyo_bf16, sm_o, use_hadamard=use_had, sr=sr, seed=5)
        qx_c_t, sx_c_t = ws.quant_col_transposed(qx_t, sm_o, sin=sx_t, use_hadamard=use_had, sr=sr, seed=6)
        dw_rot = vllm_ops.cutlass_scaled_mm(qdy_c_t, qx_c_t.T, sdy_c_t, sx_c_t.T, torch.bfloat16)
        dw_t = ws.inv_row_had128(dw_rot, sk_o) if use_had else dw_rot
        return {
            "y_rel_l2": round(rel_l2(y_t, y_ref), 6),
            "dx_rel_l2": round(rel_l2(dx_t, dx_ref), 6),
            "dw_rel_l2": round(rel_l2(dw_t, dw_ref), 6),
        }

    # Also compare Jetfire 1x128x128 Fwd vs 2D Kronecker (H64 x H64) Fwd on the same 65x outlier X!
    qjx = torch.empty((M_o, K_o), device=device, dtype=torch.int8)
    sjx = torch.empty((M_o, K_o // 128), device=device, dtype=torch.float32)
    qjw = torch.empty((N_o, K_o), device=device, dtype=torch.int8)
    sjw = torch.empty((N_o // 128, K_o // 128), device=device, dtype=torch.float32)
    _jetfire_block128_quant_act_kernel[(M_o // 64, K_o // 128)](
        xo_bf16, qjx, sjx, M_o, K_o, xo_bf16.stride(0), xo_bf16.stride(1), qjx.stride(0), qjx.stride(1), sjx.stride(0), sjx.stride(1), 1, STOCHASTIC_ROUND=False, num_warps=4
    )
    _jetfire_block128_quant_wt_kernel[(N_o // 128, K_o // 128)](
        wo_bf16, qjw, sjw, N_o, K_o, wo_bf16.stride(0), wo_bf16.stride(1), qjw.stride(0), qjw.stride(1), sjw.stride(0), sjw.stride(1), TRANSPOSE_OUT=False, num_warps=4
    )
    y_jetfire = triton_jetfire_block128_mm(qjx, qjw, sjx, sjw)

    H64 = sylvester_hadamard(64, device=device, dtype=torch.bfloat16)
    qx_kr = torch.empty((M_o, K_o), device=device, dtype=torch.int8)
    sx_kr = torch.empty((M_o, 1), device=device, dtype=torch.float32)
    qw_kr = torch.empty((N_o, K_o), device=device, dtype=torch.int8)
    sw_kr = torch.empty((N_o, 1), device=device, dtype=torch.float32)
    _kronecker_hadamard_quant_row_kernel[(M_o,)](
        xo_bf16, H64, H64, sk_o, qx_kr, sx_kr, M_o, xo_bf16.stride(0), qx_kr.stride(0), 11, STOCHASTIC_ROUND=False, D1=64, D2=64, num_warps=4
    )
    _kronecker_hadamard_quant_row_kernel[(N_o,)](
        wo_bf16, H64, H64, sk_o, qw_kr, sw_kr, N_o, wo_bf16.stride(0), qw_kr.stride(0), 22, STOCHASTIC_ROUND=False, D1=64, D2=64, num_warps=4
    )
    y_kron = vllm_ops.cutlass_scaled_mm(qx_kr, qw_kr.T, sx_kr, sw_kr.T, torch.bfloat16)

    # 64-Step Small-Signal Sub-LSB Gradient Accumulation Bias Audit (RTN-INT8 vs SR-INT8)
    # Construct a row where element 0 is 127.0 (fixing scale = 1.0) and elements 1..511 hold subtle
    # steady learning signal 0.32 (< 0.5 LSB!). Deterministic RTN rounds 0.32 -> 0 EVERY step (100% signal loss!),
    # whereas SplitMix32 SR-INT8 rounds to 1 with probability 0.32 and converges as 1/sqrt(K_steps)!
    R_sr, C_sr = 256, 512
    sig_row = torch.full((R_sr, C_sr), 0.32, device=device, dtype=torch.bfloat16)
    sig_row[:, 0] = 127.0
    ones_s = torch.ones(C_sr, device=device, dtype=torch.bfloat16)
    acc_rtn = torch.zeros((R_sr, C_sr), device=device, dtype=torch.float32)
    acc_sr = torch.zeros((R_sr, C_sr), device=device, dtype=torch.float32)
    K_steps = 256
    for step_idx in range(K_steps):
        qr_d, sr_d = ws.quant_row(sig_row, ones_s, use_hadamard=False, sr=False, seed=1000 + step_idx)
        qr_s, sr_s = ws.quant_row(sig_row, ones_s, use_hadamard=False, sr=True, seed=1000 + step_idx)
        acc_rtn += qr_d.float() * sr_d
        acc_sr += qr_s.float() * sr_s
    acc_rtn /= K_steps
    acc_sr /= K_steps
    sub_lsb_true = sig_row[:, 1:].float()
    rtn_sub_lsb_err = rel_l2(acc_rtn[:, 1:], sub_lsb_true)
    sr_sub_lsb_err = rel_l2(acc_sr[:, 1:], sub_lsb_true)

    outlier_and_sr_audit = {
        "bf16_3gemm_parity": {
            "y_rel_l2": round(rel_l2(xo_bf16 @ wo_bf16.T, y_ref), 6),
            "dx_rel_l2": round(rel_l2(dyo_bf16 @ wo_bf16, dx_ref), 6),
            "dw_rel_l2": round(rel_l2(dyo_bf16.T @ xo_bf16, dw_ref), 6),
        },
        "unrotated_switchback_3gemm_int8": _eval_3gemm_parity(use_had=False, sr=False),
        "halo_block128_hadamard_sr_3gemm_int8": _eval_3gemm_parity(use_had=True, sr=True),
        "jetfire_1x128x128_fwd_y_rel_l2": round(rel_l2(y_jetfire, y_ref), 6),
        "kronecker_64x64_hadamard_fwd_y_rel_l2": round(rel_l2(y_kron, y_ref), 6),
        "sub_lsb_256step_accumulation_audit": {
            "steps": K_steps,
            "rtn_int8_sub_lsb_rel_l2_bias": round(rtn_sub_lsb_err, 6),
            "sr_int8_sub_lsb_rel_l2_err": round(sr_sub_lsb_err, 6),
        },
    }

    # 0.4 Exact FP64 Richardson Central-Difference (h vs h/2 -> 4.000x O(h^2))
    # Use smooth non-linear outer objective cosh(out - target).sum() so d^3L/dtheta^3 != 0
    # across all 4 rotated-space derivatives (including the outer linear projection w_dn!).
    torch.manual_seed(12304)
    M_fd, H_fd, I_fd = 8, 32, 64
    H_h = sylvester_hadamard(H_fd, device=device, dtype=torch.float64)
    H_i = sylvester_hadamard(I_fd, device=device, dtype=torch.float64)
    sh_fd = rademacher_signs(H_fd, 301, device, dtype=torch.float64)
    si_fd = rademacher_signs(I_fd, 302, device, dtype=torch.float64)

    x_fd = torch.randn(M_fd, H_fd, device=device, dtype=torch.float64, requires_grad=True)
    gm_fd = (1.0 + 0.1 * torch.randn(H_fd, device=device, dtype=torch.float64)).detach().requires_grad_(True)
    wgu_fd = (0.25 * torch.randn(2 * I_fd, H_fd, device=device, dtype=torch.float64)).detach().requires_grad_(True)
    wdn_fd = (0.25 * torch.randn(H_fd, I_fd, device=device, dtype=torch.float64)).detach().requires_grad_(True)
    tgt_fd = torch.randn(M_fd, H_fd, device=device, dtype=torch.float64)

    def _had_mlp_fp64(
        xx: torch.Tensor, gg: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
    ) -> torch.Tensor:
        rstd = torch.rsqrt((xx * xx).mean(dim=-1, keepdim=True) + 1e-6)
        u_rot = rstd * ((xx * (gg * sh_fd)) @ H_h)
        w1_rot = (w1 * sh_fd) @ H_h
        gu = u_rot @ w1_rot.T
        g_part, u_part = gu[:, :I_fd], gu[:, I_fd:]
        mid = F.silu(g_part) * u_part
        mid_rot = (mid * si_fd) @ H_i
        w2_rot = (w2 * si_fd) @ H_i
        out = xx + mid_rot @ w2_rot.T
        return torch.cosh(out - tgt_fd).sum()

    loss_fd = _had_mlp_fp64(x_fd, gm_fd, wgu_fd, wdn_fd)
    loss_fd.backward()

    with torch.no_grad():
        rstd_v = torch.rsqrt((x_fd * x_fd).mean(dim=-1, keepdim=True) + 1e-6)
        u_rot_v = rstd_v * ((x_fd * (gm_fd * sh_fd)) @ H_h)
        w1_rot_v = (wgu_fd * sh_fd) @ H_h
        gu_v = u_rot_v @ w1_rot_v.T
        gp_v, up_v = gu_v[:, :I_fd], gu_v[:, I_fd:]
        sig_v = torch.sigmoid(gp_v)
        mid_v = (gp_v * sig_v) * up_v
        mid_rot_v = (mid_v * si_fd) @ H_i
        w2_rot_v = (wdn_fd * si_fd) @ H_i
        out_v = x_fd + mid_rot_v @ w2_rot_v.T
        dout_fd = torch.sinh(out_v - tgt_fd)

        d_mid_rot = dout_fd @ w2_rot_v
        dw2_ana = ((dout_fd.T @ mid_rot_v) @ H_i) * si_fd
        d_mid = (d_mid_rot @ H_i) * si_fd
        du_v = d_mid * (gp_v * sig_v)
        dg_v = d_mid * up_v * sig_v * (1.0 + gp_v * (1.0 - sig_v))
        d_gu = torch.cat([dg_v, du_v], dim=-1)
        dw1_ana = ((d_gu.T @ u_rot_v) @ H_h) * sh_fd
        du_rot = d_gu @ w1_rot_v
        dy_norm = (du_rot @ H_h) * sh_fd
        x_hat_v = x_fd * rstd_v
        dgm_ana = (dy_norm * x_hat_v).sum(dim=0)
        dx_ana = dout_fd + rstd_v * (
            dy_norm * gm_fd - x_hat_v * ((dy_norm * gm_fd) * x_hat_v).mean(dim=-1, keepdim=True)
        )

    ratios_fd: dict[str, float] = {}
    max_ana_vs_autograd = 0.0
    for name, param, g_ana, fn_eval in [
        ("dx", x_fd, dx_ana, lambda p: _had_mlp_fp64(p, gm_fd, wgu_fd, wdn_fd)),
        ("dgamma", gm_fd, dgm_ana, lambda p: _had_mlp_fp64(x_fd, p, wgu_fd, wdn_fd)),
        ("dw_gate_up_rotated_identity", wgu_fd, dw1_ana, lambda p: _had_mlp_fp64(x_fd, gm_fd, p, wdn_fd)),
        ("dw_down_rotated_identity", wdn_fd, dw2_ana, lambda p: _had_mlp_fp64(x_fd, gm_fd, wgu_fd, p)),
    ]:
        max_ana_vs_autograd = max(max_ana_vs_autograd, rel_l2(g_ana, param.grad))
        torch.manual_seed(9900)
        direction = torch.randn_like(param)
        direction = direction / direction.norm()
        ana_dir = float((g_ana * direction).sum().item())
        errs = []
        for h_step in (2e-3, 1e-3):
            with torch.no_grad():
                f_plus = float(fn_eval(param + h_step * direction).item())
                f_minus = float(fn_eval(param - h_step * direction).item())
            fd_val = (f_plus - f_minus) / (2.0 * h_step)
            errs.append(abs(fd_val - ana_dir))
        ratios_fd[name] = round(errs[0] / max(errs[1], 1e-30), 4)

    # Also verify BF16 parity of our Fused RMSNorm->INT8 + COAT SwiGLU kernels vs FP32!
    M_p, H_p, I_p = 512, 4096, 14336
    xp = torch.randn(M_p, H_p, device=device, dtype=torch.bfloat16)
    gmp = torch.ones(H_p, device=device, dtype=torch.bfloat16) * 0.9 + 0.1 * torch.randn(H_p, device=device, dtype=torch.bfloat16)
    shp = rademacher_signs(H_p, 401, device)
    sip = rademacher_signs(I_p, 402, device)
    qx_p = torch.empty((M_p, H_p), device=device, dtype=torch.int8)
    sx_p = torch.empty((M_p, 1), device=device, dtype=torch.float32)
    rstd_p = torch.empty((M_p,), device=device, dtype=torch.float32)
    _fused_rmsnorm_had128_fwd_kernel[(M_p // 32,)](
        xp, gmp, ws.H128, shp, qx_p, sx_p, rstd_p, M_p, H_p, xp.stride(0), qx_p.stride(0), 1e-6, num_warps=4
    )
    dyp = torch.randn(M_p, H_p, device=device, dtype=torch.bfloat16)
    dxp = torch.empty_like(xp)
    dgp_part = torch.empty((M_p // 32, H_p), device=device, dtype=torch.float32)
    _fused_rmsnorm_from_qnorm_bwd_kernel[(M_p // 32,)](
        dyp, qx_p, sx_p, rstd_p, gmp, ws.H128, shp, dxp, dgp_part, M_p, H_p, dyp.stride(0), qx_p.stride(0), dxp.stride(0), dgp_part.stride(0), num_warps=4
    )
    dgp = dgp_part.sum(dim=0)

    xp_ref = xp.detach().float().requires_grad_(True)
    gmp_ref = gmp.detach().float().requires_grad_(True)
    yn_ref = xp_ref * torch.rsqrt((xp_ref * xp_ref).mean(dim=-1, keepdim=True) + 1e-6) * gmp_ref
    (yn_ref * dyp.float()).sum().backward()

    gup = torch.randn(M_p, 2 * I_p, device=device, dtype=torch.bfloat16)
    qg_p = torch.empty((M_p, I_p), device=device, dtype=torch.int8)
    qu_p = torch.empty((M_p, I_p), device=device, dtype=torch.int8)
    sg_p = torch.empty((M_p, I_p // 128), device=device, dtype=torch.bfloat16)
    su_p = torch.empty((M_p, I_p // 128), device=device, dtype=torch.bfloat16)
    qmid_p = torch.empty((M_p, I_p), device=device, dtype=torch.int8)
    smid_p = torch.empty((M_p, 1), device=device, dtype=torch.float32)
    ma_p = ws.amax_i32[:M_p]
    ma_p.zero_()
    _coat_swiglu_fwd_amax_and_compress_gu_kernel[(M_p // 32, I_p // 128)](
        gup, ws.H128, sip, qg_p, sg_p, qu_p, su_p, ma_p, M_p, I_p, gup.stride(0), qg_p.stride(0), sg_p.stride(0), num_warps=4
    )
    _coat_swiglu_fwd_quant_mid_from_compressed_kernel[(M_p // 32, I_p // 128)](
        qg_p, sg_p, qu_p, su_p, ws.H128, sip, ma_p, qmid_p, smid_p, M_p, I_p, qg_p.stride(0), sg_p.stride(0), num_warps=4
    )
    dmid_p = torch.randn(M_p, I_p, device=device, dtype=torch.bfloat16)
    dgu_p = torch.empty_like(gup)
    _coat_swiglu_bwd_from_compressed_kernel[(M_p // 64, I_p // 128)](
        dmid_p, qg_p, sg_p, qu_p, su_p, dgu_p, M_p, I_p, dmid_p.stride(0), qg_p.stride(0), sg_p.stride(0), dgu_p.stride(0), num_warps=4
    )
    gup_ref = gup.detach().float().requires_grad_(True)
    mid_ref = F.silu(gup_ref[:, :I_p]) * gup_ref[:, I_p:]
    (mid_ref * dmid_p.float()).sum().backward()
    mid_rec = torch.empty((M_p, I_p), device=device, dtype=torch.bfloat16)
    _dequant_inv_row_had128_64x128[(M_p // 64, I_p // 128)](
        qmid_p, smid_p, ws.H128, sip, mid_rec, M_p, I_p, qmid_p.stride(0), qmid_p.stride(1), mid_rec.stride(0), mid_rec.stride(1), num_warps=4
    )

    coat_kernel_parity = {
        "rmsnorm_qnorm_dx_rel_l2": round(rel_l2(dxp, xp_ref.grad), 6),
        "rmsnorm_qnorm_dgamma_rel_l2": round(rel_l2(dgp, gmp_ref.grad), 6),
        "coat_swiglu_mid_rel_l2": round(rel_l2(mid_rec, mid_ref), 6),
        "coat_swiglu_dgate_up_rel_l2": round(rel_l2(dgu_p, gup_ref.grad), 6),
    }

    return {
        "sm80_int8_tensor_core_roofline": roofline,
        "sm80_ptx_quantizer_ablation": quantizer_ablation,
        "outlier_and_sr_accumulation_audit": outlier_and_sr_audit,
        "fp64_chain_rule_refinement": {
            "ratios_h_vs_half_h": ratios_fd,
            "min_refinement_ratio": round(min(ratios_fd.values()), 4),
            "max_analytical_vs_autograd_rel_l2": float(f"{max_ana_vs_autograd:.3e}"),
        },
        "coat_fused_kernel_parity": coat_kernel_parity,
    }


# ============================================================================
# 4. Suite A: Single-GPU Llama-3-8B Projection (K=4096 -> N=14336) 3-GEMM Scaling
# ============================================================================


def run_suite_a_linear_3gemm_scaling(
    device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    ws = HaloWorkspace(65536, device)
    K, N = 4096, 14336
    m_list = [2048, 8192] if is_smoke else [2048, 4096, 8192, 16384]
    warmup = 3 if is_smoke else 6
    iters = 8 if is_smoke else 20

    sk = rademacher_signs(K, 501, device)
    sn = rademacher_signs(N, 502, device)
    torch.manual_seed(12310)
    w = (torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.02)

    scaling_out: dict[str, Any] = {}
    for M in m_list:
        sm = rademacher_signs(M, 503 + M, device)
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        x[:, [3, 17, 511, 2049]] *= 40.0
        dy = torch.randn(M, N, device=device, dtype=torch.bfloat16)
        dy[[0, 64], :] *= 30.0
        flops_3gemm = 6.0 * M * N * K

        # Pre-allocate steady outputs / buffers so timing reflects kernel execution
        qx = torch.empty((M, K), device=device, dtype=torch.int8)
        sx = torch.empty((M, 1), device=device, dtype=torch.float32)
        qw_r = torch.empty((N, K), device=device, dtype=torch.int8)
        sw_r = torch.empty((N, 1), device=device, dtype=torch.float32)
        qw_c = torch.empty((K, N), device=device, dtype=torch.int8)
        sw_c = torch.empty((K, 1), device=device, dtype=torch.float32)
        qdy_r = torch.empty((M, N), device=device, dtype=torch.int8)
        sdy_r = torch.empty((M, 1), device=device, dtype=torch.float32)
        qdy_c = torch.empty((N, M), device=device, dtype=torch.int8)
        sdy_c = torch.empty((N, 1), device=device, dtype=torch.float32)
        qx_c = torch.empty((K, M), device=device, dtype=torch.int8)
        sx_c = torch.empty((K, 1), device=device, dtype=torch.float32)
        dw_buf = torch.empty((N, K), device=device, dtype=torch.bfloat16)
        x_rec = torch.empty((M, K), device=device, dtype=torch.bfloat16)

        # Jetfire 1x128 x 128x128 buffers
        qjx = torch.empty((M, K), device=device, dtype=torch.int8)
        sjx = torch.empty((M, K // 128), device=device, dtype=torch.float32)
        qjw_r = torch.empty((N, K), device=device, dtype=torch.int8)
        sjw_r = torch.empty((N // 128, K // 128), device=device, dtype=torch.float32)
        qjdy_r = torch.empty((M, N), device=device, dtype=torch.int8)
        sjdy_r = torch.empty((M, N // 128), device=device, dtype=torch.float32)
        qjw_c = torch.empty((K, N), device=device, dtype=torch.int8)
        sjw_c = torch.empty((K // 128, N // 128), device=device, dtype=torch.float32)
        qjdy_t = torch.empty((N, M), device=device, dtype=torch.int8)
        sjdy_t = torch.empty((N, M // 128), device=device, dtype=torch.float32)
        qjx_t = torch.empty((K, M), device=device, dtype=torch.int8)
        sjx_t = torch.empty((K // 128, M // 128), device=device, dtype=torch.float32)

        def _m_bf16() -> None:
            y = x @ w.T
            dx = dy @ w
            dw = dy.T @ x
            _ = (y, dx, dw)

        def _m_jetfire() -> None:
            _jetfire_block128_quant_act_kernel[(M // 64, K // 128)](
                x, qjx, sjx, M, K, x.stride(0), x.stride(1), qjx.stride(0), qjx.stride(1), sjx.stride(0), sjx.stride(1), 1, STOCHASTIC_ROUND=False, num_warps=4
            )
            _jetfire_block128_quant_wt_kernel[(N // 128, K // 128)](
                w, qjw_r, sjw_r, N, K, w.stride(0), w.stride(1), qjw_r.stride(0), qjw_r.stride(1), sjw_r.stride(0), sjw_r.stride(1), TRANSPOSE_OUT=False, num_warps=4
            )
            y = triton_jetfire_block128_mm(qjx, qjw_r, sjx, sjw_r)
            _jetfire_block128_quant_act_kernel[(M // 64, N // 128)](
                dy, qjdy_r, sjdy_r, M, N, dy.stride(0), dy.stride(1), qjdy_r.stride(0), qjdy_r.stride(1), sjdy_r.stride(0), sjdy_r.stride(1), 2, STOCHASTIC_ROUND=True, num_warps=4
            )
            _jetfire_block128_quant_wt_kernel[(N // 128, K // 128)](
                w, qjw_c, sjw_c, N, K, w.stride(0), w.stride(1), qjw_c.stride(0), qjw_c.stride(1), sjw_c.stride(0), sjw_c.stride(1), TRANSPOSE_OUT=True, num_warps=4
            )
            dx = triton_jetfire_block128_mm(qjdy_r, qjw_c, sjdy_r, sjw_c)
            dy_t = dy.t().contiguous()
            _jetfire_block128_quant_act_kernel[(N // 64, M // 128)](
                dy_t, qjdy_t, sjdy_t, N, M, dy_t.stride(0), dy_t.stride(1), qjdy_t.stride(0), qjdy_t.stride(1), sjdy_t.stride(0), sjdy_t.stride(1), 3, STOCHASTIC_ROUND=True, num_warps=4
            )
            _jetfire_block128_quant_wt_kernel[(M // 128, K // 128)](
                x, qjx_t, sjx_t, M, K, x.stride(0), x.stride(1), qjx_t.stride(0), qjx_t.stride(1), sjx_t.stride(0), sjx_t.stride(1), TRANSPOSE_OUT=True, num_warps=4
            )
            dw = triton_jetfire_block128_mm(qjdy_t, qjx_t, sjdy_t, sjx_t)
            _ = (y, dx, dw)

        def _m_switchback_hybrid() -> None:
            ws.quant_row(x, sk, use_hadamard=True, sr=False, seed=1, q_out=qx, s_out=sx)
            ws.quant_row(w, sk, use_hadamard=True, sr=False, seed=2, q_out=qw_r, s_out=sw_r)
            y = vllm_ops.cutlass_scaled_mm(qx, qw_r.T, sx, sw_r.T, torch.bfloat16)
            ws.quant_row(dy, sn, use_hadamard=True, sr=True, seed=3, q_out=qdy_r, s_out=sdy_r)
            ws.quant_col_transposed(w, sn, use_hadamard=True, sr=False, seed=4, qt_out=qw_c, st_out=sw_c)
            dx = vllm_ops.cutlass_scaled_mm(qdy_r, qw_c.T, sdy_r, sw_c.T, torch.bfloat16)
            _dequant_inv_row_had128_64x128[(M // 64, K // 128)](
                qx, sx, ws.H128, sk, x_rec, M, K, qx.stride(0), qx.stride(1), x_rec.stride(0), x_rec.stride(1), num_warps=4
            )
            dw = dy.T @ x_rec
            _ = (y, dx, dw)

        def _m_triton_pure_halo() -> None:
            ws.quant_row(x, sk, use_hadamard=True, sr=False, seed=1, q_out=qx, s_out=sx)
            ws.quant_row(w, sk, use_hadamard=True, sr=False, seed=2, q_out=qw_r, s_out=sw_r)
            y = triton_epilogue_int8_mm(qx, qw_r, sx, sw_r)
            ws.quant_row(dy, sn, use_hadamard=True, sr=True, seed=3, q_out=qdy_r, s_out=sdy_r)
            ws.quant_col_transposed(w, sn, use_hadamard=True, sr=False, seed=4, qt_out=qw_c, st_out=sw_c)
            dx = triton_epilogue_int8_mm(qdy_r, qw_c, sdy_r, sw_c)
            ws.quant_col_transposed(dy, sm, use_hadamard=True, sr=True, seed=5, qt_out=qdy_c, st_out=sdy_c)
            ws.quant_col_transposed(qx, sm, sin=sx, use_hadamard=True, sr=True, seed=6, qt_out=qx_c, st_out=sx_c)
            dw_rot = triton_epilogue_int8_mm(qdy_c, qx_c, sdy_c, sx_c)
            ws.inv_row_had128(dw_rot, sk, out=dw_buf)
            _ = (y, dx, dw_buf)

        def _m_halo_cutlass_gacc1() -> None:
            ws.quant_row(x, sk, use_hadamard=True, sr=False, seed=1, q_out=qx, s_out=sx)
            ws.quant_row(w, sk, use_hadamard=True, sr=False, seed=2, q_out=qw_r, s_out=sw_r)
            y = vllm_ops.cutlass_scaled_mm(qx, qw_r.T, sx, sw_r.T, torch.bfloat16)
            ws.quant_row(dy, sn, use_hadamard=True, sr=True, seed=3, q_out=qdy_r, s_out=sdy_r)
            ws.quant_col_transposed(w, sn, use_hadamard=True, sr=False, seed=4, qt_out=qw_c, st_out=sw_c)
            dx = vllm_ops.cutlass_scaled_mm(qdy_r, qw_c.T, sdy_r, sw_c.T, torch.bfloat16)
            ws.quant_col_transposed(dy, sm, use_hadamard=True, sr=True, seed=5, qt_out=qdy_c, st_out=sdy_c)
            ws.quant_col_transposed(qx, sm, sin=sx, use_hadamard=True, sr=True, seed=6, qt_out=qx_c, st_out=sx_c)
            dw_rot = vllm_ops.cutlass_scaled_mm(qdy_c, qx_c.T, sdy_c, sx_c.T, torch.bfloat16)
            ws.inv_row_had128(dw_rot, sk, out=dw_buf)
            _ = (y, dx, dw_buf)

        def _m_halo_cutlass_gacc4_per_mb() -> None:
            # 4-microbatch weight-stationary gradient accumulation (quantize W once per 4 microbatches!)
            ws.quant_row(w, sk, use_hadamard=True, sr=False, seed=2, q_out=qw_r, s_out=sw_r)
            ws.quant_col_transposed(w, sn, use_hadamard=True, sr=False, seed=4, qt_out=qw_c, st_out=sw_c)
            dw_buf.zero_()
            for mb in range(4):
                ws.quant_row(x, sk, use_hadamard=True, sr=False, seed=10 + mb, q_out=qx, s_out=sx)
                y = vllm_ops.cutlass_scaled_mm(qx, qw_r.T, sx, sw_r.T, torch.bfloat16)
                ws.quant_row(dy, sn, use_hadamard=True, sr=True, seed=20 + mb, q_out=qdy_r, s_out=sdy_r)
                dx = vllm_ops.cutlass_scaled_mm(qdy_r, qw_c.T, sdy_r, sw_c.T, torch.bfloat16)
                ws.quant_col_transposed(dy, sm, use_hadamard=True, sr=True, seed=30 + mb, qt_out=qdy_c, st_out=sdy_c)
                ws.quant_col_transposed(qx, sm, sin=sx, use_hadamard=True, sr=True, seed=40 + mb, qt_out=qx_c, st_out=sx_c)
                dw_rot = vllm_ops.cutlass_scaled_mm(qdy_c, qx_c.T, sdy_c, sx_c.T, torch.bfloat16)
                ws.inv_row_had128(dw_rot, sk, out=dw_buf, accumulate=(mb > 0))
                _ = (y, dx)

        modes_m: dict[str, Any] = {}
        for mk, fn, scale_div, saved_act_bytes in [
            ("bf16_cublas_eager", _m_bf16, 1.0, M * K * 2),
            ("triton_jetfire_block128_3gemm_int8", _m_jetfire, 1.0, M * K + M * (K // 128) * 4),
            ("switchback_hybrid_hadamard_int8", _m_switchback_hybrid, 1.0, M * K + M * 4),
            ("triton_pure_halo_3gemm_sr_int8", _m_triton_pure_halo, 1.0, M * K + M * 4),
            ("halo_cutlass_3gemm_sr_int8_gacc1", _m_halo_cutlass_gacc1, 1.0, M * K + M * 4),
            ("halo_cutlass_3gemm_sr_int8_gacc4", _m_halo_cutlass_gacc4_per_mb, 4.0, M * K + M * 4),
        ]:
            st = time_fn_with_peak_mem(fn, warmup, max(4, int(iters // scale_div)), device)
            ms_per_mb = st["median_ms"] / scale_div
            iqr_per_mb = st["iqr_ms"] / scale_div
            modes_m[mk] = {
                "median_ms": round(ms_per_mb, 4),
                "iqr_ms": round(iqr_per_mb, 4),
                "effective_tflops": round(flops_3gemm / (ms_per_mb * 1e-3) / 1e12, 2),
                "saved_activation_mib": round(saved_act_bytes / (1024.0**2), 2),
                "sm_clock_after_mhz": st["sm_clock_after_mhz"],
                "power_after_w": st["power_after_w"],
            }

        scaling_out[str(M)] = {
            "num_tokens": M,
            "in_features": K,
            "out_features": N,
            "modes": modes_m,
        }
        del x, dy, qx, qw_r, qw_c, qdy_r, qdy_c, qx_c, dw_buf, x_rec
        del qjx, qjw_r, qjdy_r, qjw_c, qjdy_t, qjx_t
        gc.collect()
        torch.cuda.empty_cache()

    return {"linear_3gemm_token_scaling": scaling_out}


# ============================================================================
# 5. Suite B: 4-Layer Llama-3-8B Pre-Norm SwiGLU Stack & COAT INT8 Activations
# ============================================================================


class BF16SwiGLUBlock(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden, dtype=torch.bfloat16))
        self.w_gate_up = nn.Linear(hidden, 2 * inter, bias=False, dtype=torch.bfloat16)
        self.w_down = nn.Linear(inter, hidden, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        yn = (xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        ) * self.gamma
        gu = self.w_gate_up(yn)
        g, u = gu.chunk(2, dim=-1)
        mid = F.silu(g) * u
        return x + self.w_down(mid)


class _HaloCoatSwiGLUBlockFn(torch.autograd.Function):
    """Zero-BF16-Activation-Saved Pre-Norm SwiGLU Block:
    Saves ONLY:
      - qx_norm [M, H] INT8 + sx_norm [M, 1] + rstd [M]  (1.002 B/elem; replaces BOTH x and x_norm!)
      - q_gate, q_up [M, I] INT8 + s_gate, s_up [M, I//128] BF16 (1.016 B/elem COAT!)
      - q_mid [M, I] INT8 + s_mid [M, 1] FP32 (1.0003 B/elem!)
    Total saved activation bytes/tok: 47,636 B vs 94,208 B (BF16 Eager, 1.98x smaller)
    and 65,536 B (BF16 torch.compile, 1.38x smaller), with ZERO forward GEMM recomputation!
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        gamma: torch.Tensor,
        w_gu: torch.Tensor,
        w_dn: torch.Tensor,
        sh: torch.Tensor,
        s2i: torch.Tensor,
        si: torch.Tensor,
        sm: torch.Tensor,
        ws: HaloWorkspace,
    ) -> torch.Tensor:
        M, H = x.shape
        two_I = w_gu.shape[0]
        I = two_I // 2

        qx_norm = torch.empty((M, H), device=x.device, dtype=torch.int8)
        sx_norm = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        rstd = torch.empty((M,), device=x.device, dtype=torch.float32)
        _fused_rmsnorm_had128_fwd_kernel[(M // 32,)](
            x,
            gamma,
            ws.H128,
            sh,
            qx_norm,
            sx_norm,
            rstd,
            M,
            H,
            x.stride(0),
            qx_norm.stride(0),
            1e-6,
            num_warps=4,
        )

        qw_gu_r, sw_gu_r = ws.quant_row(w_gu, sh, use_hadamard=True, sr=False, seed=101)
        gu = vllm_ops.cutlass_scaled_mm(
            qx_norm, qw_gu_r.T, sx_norm, sw_gu_r.T, torch.bfloat16
        )

        qg = torch.empty((M, I), device=x.device, dtype=torch.int8)
        qu = torch.empty((M, I), device=x.device, dtype=torch.int8)
        sg = torch.empty((M, I // 128), device=x.device, dtype=torch.bfloat16)
        su = torch.empty((M, I // 128), device=x.device, dtype=torch.bfloat16)
        q_mid = torch.empty((M, I), device=x.device, dtype=torch.int8)
        s_mid = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        mid_amax = ws.amax_i32[:M]
        mid_amax.zero_()
        g_coat = (M // 32, I // 128)
        _coat_swiglu_fwd_amax_and_compress_gu_kernel[g_coat](
            gu,
            ws.H128,
            si,
            qg,
            sg,
            qu,
            su,
            mid_amax,
            M,
            I,
            gu.stride(0),
            qg.stride(0),
            sg.stride(0),
            num_warps=4,
        )
        del gu
        _coat_swiglu_fwd_quant_mid_from_compressed_kernel[g_coat](
            qg,
            sg,
            qu,
            su,
            ws.H128,
            si,
            mid_amax,
            q_mid,
            s_mid,
            M,
            I,
            qg.stride(0),
            sg.stride(0),
            num_warps=4,
        )

        qw_dn_r, sw_dn_r = ws.quant_row(w_dn, si, use_hadamard=True, sr=False, seed=102)
        dn_out = vllm_ops.cutlass_scaled_mm(
            q_mid, qw_dn_r.T, s_mid, sw_dn_r.T, torch.bfloat16
        )
        out = x + dn_out

        ctx.save_for_backward(
            qx_norm, sx_norm, rstd, qg, sg, qu, su, q_mid, s_mid, gamma, w_gu, w_dn
        )
        ctx.sh = sh
        ctx.s2i = s2i
        ctx.si = si
        ctx.sm = sm
        ctx.ws = ws
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[Any, ...]:
        (
            qx_norm,
            sx_norm,
            rstd,
            qg,
            sg,
            qu,
            su,
            q_mid,
            s_mid,
            gamma,
            w_gu,
            w_dn,
        ) = ctx.saved_tensors
        ws: HaloWorkspace = ctx.ws
        sh, s2i, si, sm = ctx.sh, ctx.s2i, ctx.si, ctx.sm
        dout_c = dout.contiguous()
        M, H = dout_c.shape
        I = q_mid.shape[1]

        # 1. Down projection backward (d_mid & dw_dn)
        qdout_r, sdout_r = ws.quant_row(dout_c, sh, use_hadamard=True, sr=True, seed=201)
        qw_dn_c, sw_dn_c = ws.quant_col_transposed(
            w_dn, sh, use_hadamard=True, sr=False, seed=202
        )
        d_mid = vllm_ops.cutlass_scaled_mm(
            qdout_r, qw_dn_c.T, sdout_r, sw_dn_c.T, torch.bfloat16
        )

        qdout_c, sdout_c = ws.quant_col_transposed(
            dout_c, sm, use_hadamard=True, sr=True, seed=203
        )
        qmid_c, smid_c = ws.quant_col_transposed(
            q_mid, sm, sin=s_mid, use_hadamard=True, sr=True, seed=204
        )
        dw_dn_rot = vllm_ops.cutlass_scaled_mm(
            qdout_c, qmid_c.T, sdout_c, smid_c.T, torch.bfloat16
        )
        dw_dn = ws.inv_row_had128(dw_dn_rot, si)
        del qdout_r, qw_dn_c, qdout_c, qmid_c, dw_dn_rot

        # 2. COAT compressed SwiGLU backward -> d_gu [M, 2*I]
        d_gu = torch.empty((M, 2 * I), device=dout_c.device, dtype=torch.bfloat16)
        _coat_swiglu_bwd_from_compressed_kernel[(M // 64, I // 128)](
            d_mid,
            qg,
            sg,
            qu,
            su,
            d_gu,
            M,
            I,
            d_mid.stride(0),
            qg.stride(0),
            sg.stride(0),
            d_gu.stride(0),
            num_warps=4,
        )
        del d_mid

        # 3. GateUp projection backward (dy_norm & dw_gu)
        qdgu_r, sdgu_r = ws.quant_row(d_gu, s2i, use_hadamard=True, sr=True, seed=205)
        qw_gu_c, sw_gu_c = ws.quant_col_transposed(
            w_gu, s2i, use_hadamard=True, sr=False, seed=206
        )
        dy_norm = vllm_ops.cutlass_scaled_mm(
            qdgu_r, qw_gu_c.T, sdgu_r, sw_gu_c.T, torch.bfloat16
        )

        qdgu_c, sdgu_c = ws.quant_col_transposed(
            d_gu, sm, use_hadamard=True, sr=True, seed=207
        )
        del d_gu, qdgu_r, qw_gu_c
        qx_c, sx_c = ws.quant_col_transposed(
            qx_norm, sm, sin=sx_norm, use_hadamard=True, sr=True, seed=208
        )
        dw_gu_rot = vllm_ops.cutlass_scaled_mm(
            qdgu_c, qx_c.T, sdgu_c, sx_c.T, torch.bfloat16
        )
        dw_gu = ws.inv_row_had128(dw_gu_rot, sh)
        del qdgu_c, qx_c, dw_gu_rot

        # 4. Fused RMSNorm backward directly from 1 B/elem qx_norm + residual!
        dx_norm = torch.empty_like(dout_c)
        dg_part = torch.empty((M // 32, H), device=dout_c.device, dtype=torch.float32)
        _fused_rmsnorm_from_qnorm_bwd_kernel[(M // 32,)](
            dy_norm,
            qx_norm,
            sx_norm,
            rstd,
            gamma,
            ws.H128,
            sh,
            dx_norm,
            dg_part,
            M,
            H,
            dy_norm.stride(0),
            qx_norm.stride(0),
            dx_norm.stride(0),
            dg_part.stride(0),
            num_warps=4,
        )
        dgamma = dg_part.sum(dim=0).to(torch.bfloat16)
        dx = dout_c + dx_norm
        return dx, dgamma, dw_gu, dw_dn, None, None, None, None, None


class HaloCoatSwiGLUBlock(nn.Module):
    def __init__(self, ref_blk: BF16SwiGLUBlock, ws: HaloWorkspace, idx: int) -> None:
        super().__init__()
        H = ref_blk.gamma.numel()
        two_I = ref_blk.w_gate_up.weight.shape[0]
        I = two_I // 2
        self.gamma = nn.Parameter(ref_blk.gamma.detach().clone())
        self.w_gate_up = nn.Parameter(ref_blk.w_gate_up.weight.detach().clone())
        self.w_down = nn.Parameter(ref_blk.w_down.weight.detach().clone())
        dev = self.gamma.device
        self.register_buffer("sh", rademacher_signs(H, 600 + idx * 7, dev))
        self.register_buffer("s2i", rademacher_signs(two_I, 601 + idx * 7, dev))
        self.register_buffer("si", rademacher_signs(I, 602 + idx * 7, dev))
        self.ws = ws

    def forward(self, x: torch.Tensor, sm: torch.Tensor) -> torch.Tensor:
        return _HaloCoatSwiGLUBlockFn.apply(
            x,
            self.gamma,
            self.w_gate_up,
            self.w_down,
            self.sh,
            self.s2i,
            self.si,
            sm,
            self.ws,
        )


def run_suite_b_coat_swiglu_mlp_stack(
    device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    ws = HaloWorkspace(65536, device)
    n_layers = 2 if is_smoke else 4
    H, I = 4096, 14336
    tok_list = [2048, 4096] if is_smoke else [2048, 4096, 8192]
    warmup = 3 if is_smoke else 5
    iters = 6 if is_smoke else 15

    torch.manual_seed(12320)
    ref_blocks = nn.ModuleList(
        [BF16SwiGLUBlock(H, I).to(device=device) for _ in range(n_layers)]
    )
    for blk in ref_blocks:
        nn.init.normal_(blk.w_gate_up.weight, std=0.02)
        nn.init.normal_(blk.w_down.weight, std=0.02)

    halo_blocks = nn.ModuleList(
        [HaloCoatSwiGLUBlock(ref_blocks[i], ws, i).to(device=device) for i in range(n_layers)]
    )
    compiled_blocks = nn.ModuleList(
        [torch.compile(ref_blocks[i], mode="default") for i in range(n_layers)]
    )

    grad_bytes = sum(p.numel() * p.element_size() for p in ref_blocks.parameters())

    # 4-layer stack gradient parity check at N=2048
    x_chk = torch.randn(2048, H, device=device, dtype=torch.bfloat16, requires_grad=True)
    dout_chk = torch.randn(2048, H, device=device, dtype=torch.bfloat16)
    sm_chk = rademacher_signs(2048, 700, device)
    for p in ref_blocks.parameters():
        p.grad = None
    h_ref = x_chk
    for blk in ref_blocks:
        h_ref = blk(h_ref)
    h_ref.backward(dout_chk)

    xh_chk = x_chk.detach().clone().requires_grad_(True)
    for p in halo_blocks.parameters():
        p.grad = None
    h_halo = xh_chk
    for blk in halo_blocks:
        h_halo = blk(h_halo, sm_chk)
    h_halo.backward(dout_chk)

    stack_parity = {
        "out_rel_l2": round(rel_l2(h_halo, h_ref), 6),
        "dx_rel_l2": round(rel_l2(xh_chk.grad, x_chk.grad), 6),
        "dw_layer0_gate_up_rel_l2": round(
            rel_l2(halo_blocks[0].w_gate_up.grad, ref_blocks[0].w_gate_up.weight.grad),
            6,
        ),
        "dw_last_down_rel_l2": round(
            rel_l2(halo_blocks[-1].w_down.grad, ref_blocks[-1].w_down.weight.grad),
            6,
        ),
    }
    del x_chk, xh_chk, dout_chk, h_ref, h_halo

    scaling_b: dict[str, Any] = {}
    for M in tok_list:
        x_in = torch.randn(M, H, device=device, dtype=torch.bfloat16, requires_grad=True)
        dout = torch.randn(M, H, device=device, dtype=torch.bfloat16)
        sm = rademacher_signs(M, 710 + M, device)
        # 3 Linear equivalents per SwiGLU block (2*I + I = 3*I), Fwd+dX+dW = 6 * M * H * (3*I) / 2?
        # GateUp: 6*M*H*(2*I), Down: 6*M*I*H => total 18 * M * H * I FLOPs per layer!
        flops_stack = n_layers * 18.0 * M * H * I

        def _measure_stack(
            fn_fwd_bwd: Callable[[], None], modules: nn.Module
        ) -> dict[str, float]:
            # Also measure exact persistent saved-activation HBM right between Fwd and Bwd!
            for p in modules.parameters():
                p.grad = None
            x_in.grad = None
            st = time_fn_with_peak_mem(
                fn_fwd_bwd, warmup, iters, device, grad_bytes_steady=grad_bytes + M * H * 2
            )
            st["effective_tflops"] = round(
                flops_stack / (st["median_ms"] * 1e-3) / 1e12, 2
            )
            return st

        def _run_eager() -> None:
            for p in ref_blocks.parameters():
                p.grad = None
            x_in.grad = None
            h = x_in
            for blk in ref_blocks:
                h = blk(h)
            h.backward(dout)

        def _run_compile() -> None:
            for p in ref_blocks.parameters():
                p.grad = None
            x_in.grad = None
            h = x_in
            for cblk in compiled_blocks:
                h = cblk(h)
            h.backward(dout)

        def _run_ckpt() -> None:
            for p in ref_blocks.parameters():
                p.grad = None
            x_in.grad = None
            h = x_in
            for blk in ref_blocks:
                h = ckpt_fn(blk, h, use_reentrant=False)
            h.backward(dout)

        def _run_halo_coat() -> None:
            for p in halo_blocks.parameters():
                p.grad = None
            x_in.grad = None
            h = x_in
            for hblk in halo_blocks:
                h = hblk(h, sm)
            h.backward(dout)

        # Measure exact forward saved-activation memory after forward() before backward()!
        def _saved_act_mib(fwd_only: Callable[[], torch.Tensor]) -> float:
            gc.collect()
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated(device)
            y_tmp = fwd_only()
            torch.cuda.synchronize(device)
            after = torch.cuda.memory_allocated(device)
            saved_mib = max(0.0, (after - base - y_tmp.numel() * y_tmp.element_size()) / (1024.0**2))
            del y_tmp
            gc.collect()
            torch.cuda.empty_cache()
            return round(saved_mib, 2)

        def _fwd_e() -> torch.Tensor:
            h = x_in
            for blk in ref_blocks:
                h = blk(h)
            return h

        def _fwd_c() -> torch.Tensor:
            h = x_in
            for cblk in compiled_blocks:
                h = cblk(h)
            return h

        def _fwd_k() -> torch.Tensor:
            h = x_in
            for blk in ref_blocks:
                h = ckpt_fn(blk, h, use_reentrant=False)
            return h

        def _fwd_h() -> torch.Tensor:
            h = x_in
            for hblk in halo_blocks:
                h = hblk(h, sm)
            return h

        m_eager = _measure_stack(_run_eager, ref_blocks)
        m_eager["saved_activations_mib"] = _saved_act_mib(_fwd_e)

        m_comp = _measure_stack(_run_compile, ref_blocks)
        m_comp["saved_activations_mib"] = _saved_act_mib(_fwd_c)

        m_ckpt = _measure_stack(_run_ckpt, ref_blocks)
        m_ckpt["saved_activations_mib"] = _saved_act_mib(_fwd_k)

        m_halo = _measure_stack(_run_halo_coat, halo_blocks)
        m_halo["saved_activations_mib"] = _saved_act_mib(_fwd_h)

        scaling_b[str(M)] = {
            "num_tokens": M,
            "num_layers": n_layers,
            "hidden": H,
            "intermediate": I,
            "modes": {
                "mlp_bf16_eager": m_eager,
                "mlp_bf16_torch_compile": m_comp,
                "mlp_bf16_full_checkpoint": m_ckpt,
                "mlp_halo_coat_int8_fused": m_halo,
            },
        }
        del x_in, dout
        gc.collect()
        torch.cuda.empty_cache()

    del ref_blocks, halo_blocks, compiled_blocks
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "stack_bf16_parity": stack_parity,
        "token_scaling": scaling_b,
    }


# ============================================================================
# 6. Suite C: 8-GPU NVSwitch Quantized-AllGather + COAT INT8 FSDP2 Training
# ============================================================================


class GQATransformerBlockBF16(nn.Module):
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
        B, S, H = x.shape
        xf = x.float()
        h1 = (xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        ) * self.norm1
        qkv = self.w_qkv(h1)
        q_dim = self.n_q * self.head_dim
        kv_dim = self.n_kv * self.head_dim
        q = qkv[:, :, :q_dim].view(B, S, self.n_q, self.head_dim)
        k = qkv[:, :, q_dim : q_dim + kv_dim].view(B, S, self.n_kv, self.head_dim)
        v = qkv[:, :, q_dim + kv_dim :].view(B, S, self.n_kv, self.head_dim)
        ao = flash_attn_func(q, k, v, causal=True).reshape(B, S, q_dim)
        return x + self.w_o(ao)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        h2 = (xf * torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        ) * self.norm2
        gu = self.w_gate_up(h2)
        g, u = gu.chunk(2, dim=-1)
        return x + self.w_down(F.silu(g) * u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(self._attn(x))


class GQATransformerModelBF16(nn.Module):
    def __init__(
        self,
        n_layers: int,
        hidden: int,
        inter: int,
        n_q: int,
        n_kv: int,
        use_ckpt: bool = False,
        compile_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [GQATransformerBlockBF16(hidden, inter, n_q, n_kv) for _ in range(n_layers)]
        )
        self.use_ckpt = use_ckpt
        if compile_mlp:
            for blk in self.blocks:
                blk._mlp = torch.compile(blk._mlp, mode="default")  # type: ignore[method-assign]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_ckpt:
                x = ckpt_fn(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x


def _quant_allgather_fwd_weight(
    w_shard: torch.Tensor,
    sign_k: torch.Tensor,
    ws: HaloWorkspace,
    world_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shard-local Hadamard along K (8x less compute!) + 1 B/param INT8 AllGather (2x less NVSwitch traffic!)."""
    N_sh, K = w_shard.shape
    N_full = N_sh * world_size
    qw_sh, sw_sh = ws.quant_row(w_shard, sign_k, use_hadamard=True, sr=False, seed=seed)
    qw_full = torch.empty((N_full, K), device=w_shard.device, dtype=torch.int8)
    sw_full = torch.empty((N_full, 1), device=w_shard.device, dtype=torch.float32)
    dist.all_gather_into_tensor(qw_full, qw_sh)
    dist.all_gather_into_tensor(sw_full, sw_sh)
    return qw_full, sw_full


def _quant_allgather_bwd_col_weight(
    w_shard: torch.Tensor,
    sign_n_shard: torch.Tensor,
    ws: HaloWorkspace,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mathematically bitwise-identical 8-GPU shard-local N-Hadamard + 16 KiB MAX AllReduce
    + 1 B/param INT8 AllGather + SRAM 128x128 INT8 transpose!
    Requires N_shard % 128 == 0 (satisfied by QKV=768, O=512, GateUp=3584, Down=512).
    """
    N_sh, K = w_shard.shape
    N_full = N_sh * world_size
    col_amax_i32 = ws.amax_i32[:K]
    col_amax_i32.zero_()
    g = (N_sh // 128, K // 64)
    _col_had128_amax_128x64[g](
        w_shard,
        w_shard,
        ws.H128,
        sign_n_shard,
        col_amax_i32,
        N_sh,
        K,
        w_shard.stride(0),
        w_shard.stride(1),
        INPUT_IS_INT8=False,
        USE_HADAMARD=True,
        num_warps=4,
    )
    dist.all_reduce(col_amax_i32, op=dist.ReduceOp.MAX)

    qw_nrot_sh = torch.empty((N_sh, K), device=w_shard.device, dtype=torch.int8)
    sw_c = torch.empty((K, 1), device=w_shard.device, dtype=torch.float32)
    _col_had128_nontransposed_shard_quant_128x64[g](
        w_shard,
        ws.H128,
        sign_n_shard,
        col_amax_i32,
        qw_nrot_sh,
        sw_c,
        N_sh,
        K,
        w_shard.stride(0),
        w_shard.stride(1),
        qw_nrot_sh.stride(0),
        qw_nrot_sh.stride(1),
        num_warps=4,
    )
    qw_nrot_full = torch.empty((N_full, K), device=w_shard.device, dtype=torch.int8)
    dist.all_gather_into_tensor(qw_nrot_full, qw_nrot_sh)
    del qw_nrot_sh
    qw_c = torch.empty((K, N_full), device=w_shard.device, dtype=torch.int8)
    _transpose_int8_128x128_kernel[(N_full // 128, K // 128)](
        qw_nrot_full,
        qw_c,
        N_full,
        K,
        qw_nrot_full.stride(0),
        qw_nrot_full.stride(1),
        qw_c.stride(0),
        qw_c.stride(1),
        num_warps=4,
    )
    del qw_nrot_full
    return qw_c, sw_c


def _fp32_reducescatter_dw(
    dw_rot: torch.Tensor,
    sign_k: torch.Tensor,
    ws: HaloWorkspace,
    world_size: int,
) -> torch.Tensor:
    N_full, K = dw_rot.shape
    dw_fp32 = ws.inv_row_had128(dw_rot, sign_k, out_fp32=True)
    dw_fp32.mul_(1.0 / world_size)
    dw_sh = torch.empty((N_full // world_size, K), device=dw_rot.device, dtype=torch.float32)
    dist.reduce_scatter_tensor(dw_sh, dw_fp32, op=dist.ReduceOp.SUM)
    return dw_sh.to(torch.bfloat16)


class _QuantFSDP2BlockFn(torch.autograd.Function):
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
        sh: torch.Tensor,
        sqkv: torch.Tensor,
        sqkv_sh: torch.Tensor,
        sh_sh: torch.Tensor,
        s2i: torch.Tensor,
        s2i_sh: torch.Tensor,
        si: torch.Tensor,
        sm: torch.Tensor,
        ws: HaloWorkspace,
        n_q: int,
        n_kv: int,
        head_dim: int,
        world_size: int,
    ) -> torch.Tensor:
        B, S, H = x.shape
        M = B * S
        x2d = x.view(M, H)

        # --- Subblock 1: Attention with INT8 AllGather & COAT INT8 Saved Norms/AttnOut ---
        qx1 = torch.empty((M, H), device=x.device, dtype=torch.int8)
        sx1 = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        rstd1 = torch.empty((M,), device=x.device, dtype=torch.float32)
        _fused_rmsnorm_had128_fwd_kernel[(M // 32,)](
            x2d, norm1, ws.H128, sh, qx1, sx1, rstd1, M, H, x2d.stride(0), qx1.stride(0), 1e-6, num_warps=4
        )
        qw_qkv_r, sw_qkv_r = _quant_allgather_fwd_weight(wqkv_sh, sh, ws, world_size, 11)
        qkv = vllm_ops.cutlass_scaled_mm(qx1, qw_qkv_r.T, sx1, sw_qkv_r.T, torch.bfloat16)
        del qw_qkv_r, sw_qkv_r

        q_dim = n_q * head_dim
        kv_dim = n_kv * head_dim
        q = qkv[:, :q_dim].view(B, S, n_q, head_dim).contiguous()
        k = qkv[:, q_dim : q_dim + kv_dim].view(B, S, n_kv, head_dim).contiguous()
        v = qkv[:, q_dim + kv_dim :].view(B, S, n_kv, head_dim).contiguous()
        del qkv

        ao, softmax_lse, _, _ = flash_attn_func(
            q, k, v, causal=True, return_attn_probs=False
        ) if False else (None, None, None, None)
        # Use flash_attn_func via detached leaf to save only FA2 minimal state while using INT8 O-proj!
        q_leaf = q.detach().requires_grad_(True)
        k_leaf = k.detach().requires_grad_(True)
        v_leaf = v.detach().requires_grad_(True)
        with torch.enable_grad():
            ao_leaf = flash_attn_func(q_leaf, k_leaf, v_leaf, causal=True)
        ao_2d = ao_leaf.reshape(M, q_dim)

        qao, sao = ws.quant_row(ao_2d, sh, use_hadamard=True, sr=False, seed=12)
        qw_o_r, sw_o_r = _quant_allgather_fwd_weight(wo_sh, sh, ws, world_size, 13)
        o_out = vllm_ops.cutlass_scaled_mm(qao, qw_o_r.T, sao, sw_o_r.T, torch.bfloat16)
        del qw_o_r, sw_o_r
        x_mid = x2d + o_out
        del o_out

        # --- Subblock 2: SwiGLU MLP with INT8 AllGather & COAT INT8 Saved Activations ---
        I = wdn_sh.shape[1]
        qx2 = torch.empty((M, H), device=x.device, dtype=torch.int8)
        sx2 = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        rstd2 = torch.empty((M,), device=x.device, dtype=torch.float32)
        _fused_rmsnorm_had128_fwd_kernel[(M // 32,)](
            x_mid, norm2, ws.H128, sh, qx2, sx2, rstd2, M, H, x_mid.stride(0), qx2.stride(0), 1e-6, num_warps=4
        )
        qw_gu_r, sw_gu_r = _quant_allgather_fwd_weight(wgu_sh, sh, ws, world_size, 14)
        gu = vllm_ops.cutlass_scaled_mm(qx2, qw_gu_r.T, sx2, sw_gu_r.T, torch.bfloat16)
        del qw_gu_r, sw_gu_r

        qg = torch.empty((M, I), device=x.device, dtype=torch.int8)
        qu = torch.empty((M, I), device=x.device, dtype=torch.int8)
        sg = torch.empty((M, I // 128), device=x.device, dtype=torch.bfloat16)
        su = torch.empty((M, I // 128), device=x.device, dtype=torch.bfloat16)
        q_mid = torch.empty((M, I), device=x.device, dtype=torch.int8)
        s_mid = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        mid_amax = ws.amax_i32[:M]
        mid_amax.zero_()
        g_coat = (M // 32, I // 128)
        _coat_swiglu_fwd_amax_and_compress_gu_kernel[g_coat](
            gu, ws.H128, si, qg, sg, qu, su, mid_amax, M, I, gu.stride(0), qg.stride(0), sg.stride(0), num_warps=4
        )
        del gu
        _coat_swiglu_fwd_quant_mid_from_compressed_kernel[g_coat](
            qg, sg, qu, su, ws.H128, si, mid_amax, q_mid, s_mid, M, I, qg.stride(0), sg.stride(0), num_warps=4
        )

        qw_dn_r, sw_dn_r = _quant_allgather_fwd_weight(wdn_sh, si, ws, world_size, 15)
        dn_out = vllm_ops.cutlass_scaled_mm(q_mid, qw_dn_r.T, s_mid, sw_dn_r.T, torch.bfloat16)
        del qw_dn_r, sw_dn_r
        out = (x_mid + dn_out).view(B, S, H)

        ctx.save_for_backward(
            qx1, sx1, rstd1, qao, sao,
            qx2, sx2, rstd2, qg, sg, qu, su, q_mid, s_mid,
            norm1, wqkv_sh, wo_sh, norm2, wgu_sh, wdn_sh,
        )
        ctx.fa2_leaves = (q_leaf, k_leaf, v_leaf, ao_leaf)
        ctx.signs = (sh, sqkv, sqkv_sh, sh_sh, s2i, s2i_sh, si, sm)
        ctx.ws = ws
        ctx.meta = (B, S, H, I, n_q, n_kv, head_dim, world_size)
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[Any, ...]:
        (
            qx1, sx1, rstd1, qao, sao,
            qx2, sx2, rstd2, qg, sg, qu, su, q_mid, s_mid,
            norm1, wqkv_sh, wo_sh, norm2, wgu_sh, wdn_sh,
        ) = ctx.saved_tensors
        q_leaf, k_leaf, v_leaf, ao_leaf = ctx.fa2_leaves
        sh, sqkv, sqkv_sh, sh_sh, s2i, s2i_sh, si, sm = ctx.signs
        ws: HaloWorkspace = ctx.ws
        B, S, H, I, n_q, n_kv, head_dim, world_size = ctx.meta
        M = B * S
        dout2d = dout.contiguous().view(M, H)

        # --- Backward MLP ---
        qdout_r, sdout_r = ws.quant_row(dout2d, sh, use_hadamard=True, sr=True, seed=301)
        qw_dn_c, sw_dn_c = _quant_allgather_bwd_col_weight(wdn_sh, sh_sh, ws, world_size)
        d_mid = vllm_ops.cutlass_scaled_mm(qdout_r, qw_dn_c.T, sdout_r, sw_dn_c.T, torch.bfloat16)
        del qdout_r, qw_dn_c, sw_dn_c

        qdout_c, sdout_c = ws.quant_col_transposed(dout2d, sm, use_hadamard=True, sr=True, seed=302)
        qmid_c, smid_c = ws.quant_col_transposed(q_mid, sm, sin=s_mid, use_hadamard=True, sr=True, seed=303)
        dw_dn_rot = vllm_ops.cutlass_scaled_mm(qdout_c, qmid_c.T, sdout_c, smid_c.T, torch.bfloat16)
        del qdout_c, qmid_c
        dwdn_sh = _fp32_reducescatter_dw(dw_dn_rot, si, ws, world_size)
        del dw_dn_rot

        d_gu = torch.empty((M, 2 * I), device=dout.device, dtype=torch.bfloat16)
        _coat_swiglu_bwd_from_compressed_kernel[(M // 64, I // 128)](
            d_mid, qg, sg, qu, su, d_gu, M, I, d_mid.stride(0), qg.stride(0), sg.stride(0), d_gu.stride(0), num_warps=4
        )
        del d_mid

        qdgu_r, sdgu_r = ws.quant_row(d_gu, s2i, use_hadamard=True, sr=True, seed=304)
        qw_gu_c, sw_gu_c = _quant_allgather_bwd_col_weight(wgu_sh, s2i_sh, ws, world_size)
        dy_norm2 = vllm_ops.cutlass_scaled_mm(qdgu_r, qw_gu_c.T, sdgu_r, sw_gu_c.T, torch.bfloat16)
        del qdgu_r, qw_gu_c, sw_gu_c

        qdgu_c, sdgu_c = ws.quant_col_transposed(d_gu, sm, use_hadamard=True, sr=True, seed=305)
        del d_gu
        qx2_c, sx2_c = ws.quant_col_transposed(qx2, sm, sin=sx2, use_hadamard=True, sr=True, seed=306)
        dw_gu_rot = vllm_ops.cutlass_scaled_mm(qdgu_c, qx2_c.T, sdgu_c, sx2_c.T, torch.bfloat16)
        del qdgu_c, qx2_c
        dwgu_sh = _fp32_reducescatter_dw(dw_gu_rot, sh, ws, world_size)
        del dw_gu_rot

        dx_norm2 = torch.empty_like(dout2d)
        dg2_part = torch.empty((M // 32, H), device=dout.device, dtype=torch.float32)
        _fused_rmsnorm_from_qnorm_bwd_kernel[(M // 32,)](
            dy_norm2, qx2, sx2, rstd2, norm2, ws.H128, sh, dx_norm2, dg2_part, M, H, dy_norm2.stride(0), qx2.stride(0), dx_norm2.stride(0), dg2_part.stride(0), num_warps=4
        )
        del dy_norm2
        dnorm2 = dg2_part.sum(dim=0) / world_size
        dist.all_reduce(dnorm2, op=dist.ReduceOp.SUM)
        dx_mid = dout2d + dx_norm2
        del dx_norm2

        # --- Backward Attention ---
        qdxm_r, sdxm_r = ws.quant_row(dx_mid, sh, use_hadamard=True, sr=True, seed=307)
        qw_o_c, sw_o_c = _quant_allgather_bwd_col_weight(wo_sh, sh_sh, ws, world_size)
        dao_2d = vllm_ops.cutlass_scaled_mm(qdxm_r, qw_o_c.T, sdxm_r, sw_o_c.T, torch.bfloat16)
        del qdxm_r, qw_o_c, sw_o_c

        qdxm_c, sdxm_c = ws.quant_col_transposed(dx_mid, sm, use_hadamard=True, sr=True, seed=308)
        qao_c, sao_c = ws.quant_col_transposed(qao, sm, sin=sao, use_hadamard=True, sr=True, seed=309)
        dw_o_rot = vllm_ops.cutlass_scaled_mm(qdxm_c, qao_c.T, sdxm_c, sao_c.T, torch.bfloat16)
        del qdxm_c, qao_c
        dwo_sh = _fp32_reducescatter_dw(dw_o_rot, sh, ws, world_size)
        del dw_o_rot

        torch.autograd.backward(ao_leaf, dao_2d.view(B, S, n_q, head_dim))
        del dao_2d
        dqkv = torch.cat(
            [
                q_leaf.grad.reshape(M, n_q * head_dim),
                k_leaf.grad.reshape(M, n_kv * head_dim),
                v_leaf.grad.reshape(M, n_kv * head_dim),
            ],
            dim=-1,
        )
        ctx.fa2_leaves = None
        del q_leaf, k_leaf, v_leaf, ao_leaf

        qdqkv_r, sdqkv_r = ws.quant_row(dqkv, sqkv, use_hadamard=True, sr=True, seed=310)
        qw_qkv_c, sw_qkv_c = _quant_allgather_bwd_col_weight(wqkv_sh, sqkv_sh, ws, world_size)
        dy_norm1 = vllm_ops.cutlass_scaled_mm(qdqkv_r, qw_qkv_c.T, sdqkv_r, sw_qkv_c.T, torch.bfloat16)
        del qdqkv_r, qw_qkv_c, sw_qkv_c

        qdqkv_c, sdqkv_c = ws.quant_col_transposed(dqkv, sm, use_hadamard=True, sr=True, seed=311)
        del dqkv
        qx1_c, sx1_c = ws.quant_col_transposed(qx1, sm, sin=sx1, use_hadamard=True, sr=True, seed=312)
        dw_qkv_rot = vllm_ops.cutlass_scaled_mm(qdqkv_c, qx1_c.T, sdqkv_c, sx1_c.T, torch.bfloat16)
        del qdqkv_c, qx1_c
        dwqkv_sh = _fp32_reducescatter_dw(dw_qkv_rot, sh, ws, world_size)
        del dw_qkv_rot

        dx_norm1 = torch.empty_like(dx_mid)
        dg1_part = torch.empty((M // 32, H), device=dout.device, dtype=torch.float32)
        _fused_rmsnorm_from_qnorm_bwd_kernel[(M // 32,)](
            dy_norm1, qx1, sx1, rstd1, norm1, ws.H128, sh, dx_norm1, dg1_part, M, H, dy_norm1.stride(0), qx1.stride(0), dx_norm1.stride(0), dg1_part.stride(0), num_warps=4
        )
        del dy_norm1
        dnorm1 = dg1_part.sum(dim=0) / world_size
        dist.all_reduce(dnorm1, op=dist.ReduceOp.SUM)
        dx = (dx_mid + dx_norm1).view(B, S, H)
        return (
            dx,
            dnorm1.to(torch.bfloat16),
            dwqkv_sh,
            dwo_sh,
            dnorm2.to(torch.bfloat16),
            dwgu_sh,
            dwdn_sh,
            *(None for _ in range(13)),
        )


class HaloCoatQuantFSDP2Model(nn.Module):
    """Owner-sharded 8-GPU Transformer storing 1/8 weight shards [N/8, K] per rank,
    executing shard-local Hadamard + 1 B/param INT8 AllGather + COAT INT8 activations + FP32 ReduceScatter.
    """

    def __init__(
        self,
        ref_cpu: GQATransformerModelBF16,
        rank: int,
        world_size: int,
        device: torch.device,
        ws: HaloWorkspace,
    ) -> None:
        super().__init__()
        self.ws = ws
        self.world_size = world_size
        self.layers = nn.ModuleList()

        for idx, blk in enumerate(ref_cpu.blocks):
            H = blk.norm1.numel()
            qkv_dim = blk.w_qkv.weight.shape[0]
            two_I = blk.w_gate_up.weight.shape[0]
            I = two_I // 2

            def _slice_sh(w: torch.Tensor) -> nn.Parameter:
                n_sh = w.shape[0] // world_size
                return nn.Parameter(
                    w[rank * n_sh : (rank + 1) * n_sh].detach().to(device=device).contiguous()
                )

            ld = nn.Module()
            ld.norm1 = nn.Parameter(blk.norm1.detach().to(device=device).clone())
            ld.wqkv_sh = _slice_sh(blk.w_qkv.weight)
            ld.wo_sh = _slice_sh(blk.w_o.weight)
            ld.norm2 = nn.Parameter(blk.norm2.detach().to(device=device).clone())
            ld.wgu_sh = _slice_sh(blk.w_gate_up.weight)
            ld.wdn_sh = _slice_sh(blk.w_down.weight)

            sh = rademacher_signs(H, 800 + idx * 11, device)
            sqkv = rademacher_signs(qkv_dim, 801 + idx * 11, device)
            s2i = rademacher_signs(two_I, 802 + idx * 11, device)
            si = rademacher_signs(I, 803 + idx * 11, device)

            ld.register_buffer("sh", sh)
            ld.register_buffer("sh_sh", sh[rank * (H // world_size) : (rank + 1) * (H // world_size)].contiguous())
            ld.register_buffer("sqkv", sqkv)
            ld.register_buffer("sqkv_sh", sqkv[rank * (qkv_dim // world_size) : (rank + 1) * (qkv_dim // world_size)].contiguous())
            ld.register_buffer("s2i", s2i)
            ld.register_buffer("s2i_sh", s2i[rank * (two_I // world_size) : (rank + 1) * (two_I // world_size)].contiguous())
            ld.register_buffer("si", si)
            ld.n_q = blk.n_q
            ld.n_kv = blk.n_kv
            ld.head_dim = blk.head_dim
            self.layers.append(ld)

    def forward(self, x: torch.Tensor, sm: torch.Tensor) -> torch.Tensor:
        for ld in self.layers:
            x = _QuantFSDP2BlockFn.apply(
                x,
                ld.norm1,
                ld.wqkv_sh,
                ld.wo_sh,
                ld.norm2,
                ld.wgu_sh,
                ld.wdn_sh,
                ld.sh,
                ld.sqkv,
                ld.sqkv_sh,
                ld.sh_sh,
                ld.s2i,
                ld.s2i_sh,
                ld.si,
                sm,
                self.ws,
                ld.n_q,
                ld.n_kv,
                ld.head_dim,
                self.world_size,
            )
        return x


def _wrap_fsdp2_gqa(model: GQATransformerModelBF16) -> GQATransformerModelBF16:
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in model.blocks:
        fully_shard(blk, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model, mp_policy=mp, reshard_after_forward=True)
    return model


def run_suite_c_quantized_fsdp2_8gpu(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    ws = HaloWorkspace(65536, device)

    # ------------------------------------------------------------------------
    # C.1 Exact 8-GPU Distributed Gradient Parity vs 1-GPU Global FP32/BF16
    #     (2-layer Llama-3-8B-geometry: H=2048, I=8192, 16Q/8KV, D=128)
    # ------------------------------------------------------------------------
    torch.manual_seed(12350)
    H_p, I_p, nq_p, nkv_p = 2048, 8192, 16, 8
    p_cpu = GQATransformerModelBF16(2, H_p, I_p, nq_p, nkv_p)
    for p in p_cpu.parameters():
        if p.ndim == 2:
            nn.init.normal_(p, std=0.02)

    B_glob_p, S_p = 8, 512
    x_glob_p = torch.randn(B_glob_p, S_p, H_p, dtype=torch.bfloat16)
    x_glob_p[:, :, [3, 17, 257, 1025]] *= 25.0
    dy_glob_p = torch.randn(B_glob_p, S_p, H_p, dtype=torch.bfloat16)
    dy_glob_p[:, [0, 32], :] *= 25.0

    # 1-GPU Global BF16 Reference (computed identically or broadcast from rank 0)
    ref_1gpu = GQATransformerModelBF16(2, H_p, I_p, nq_p, nkv_p).to(device=device)
    ref_1gpu.load_state_dict(p_cpu.state_dict())
    xg_dev = x_glob_p.to(device=device)
    dyg_dev = dy_glob_p.to(device=device)
    out_1gpu = ref_1gpu(xg_dev)
    (out_1gpu.float() * dyg_dev.float()).sum().div(world_size).backward()
    g1_wqkv0 = ref_1gpu.blocks[0].w_qkv.weight.grad.detach().clone()
    g1_wgu0 = ref_1gpu.blocks[0].w_gate_up.weight.grad.detach().clone()
    g1_wdn1 = ref_1gpu.blocks[1].w_down.weight.grad.detach().clone()
    del ref_1gpu, out_1gpu

    B_loc_p = B_glob_p // world_size
    x_loc_p = xg_dev[rank * B_loc_p : (rank + 1) * B_loc_p].contiguous()
    dy_loc_p = dyg_dev[rank * B_loc_p : (rank + 1) * B_loc_p].contiguous()
    del xg_dev, dyg_dev

    # 8-GPU Standard BF16 FSDP2
    fsdp_p = GQATransformerModelBF16(2, H_p, I_p, nq_p, nkv_p).to(device=device)
    fsdp_p.load_state_dict(p_cpu.state_dict())
    fsdp_p = _wrap_fsdp2_gqa(fsdp_p)
    out_fp = fsdp_p(x_loc_p)
    (out_fp.float() * dy_loc_p.float()).sum().backward()
    gf_wqkv0 = fsdp_p.blocks[0].w_qkv.weight.grad.full_tensor()
    gf_wgu0 = fsdp_p.blocks[0].w_gate_up.weight.grad.full_tensor()
    gf_wdn1 = fsdp_p.blocks[1].w_down.weight.grad.full_tensor()
    del fsdp_p, out_fp

    # 8-GPU Halo-COAT Quantized-AllGather FSDP2
    qfsdp_p = HaloCoatQuantFSDP2Model(p_cpu, rank, world_size, device, ws)
    sm_p = rademacher_signs(B_loc_p * S_p, 910, device)
    out_qp = qfsdp_p(x_loc_p, sm_p)
    (out_qp.float() * dy_loc_p.float()).sum().backward()

    def _gather_sh_grad(sh_g: torch.Tensor) -> torch.Tensor:
        full_g = torch.empty(
            (sh_g.shape[0] * world_size, sh_g.shape[1]),
            device=device,
            dtype=sh_g.dtype,
        )
        dist.all_gather_into_tensor(full_g, sh_g.contiguous())
        return full_g

    gq_wqkv0 = _gather_sh_grad(qfsdp_p.layers[0].wqkv_sh.grad)
    gq_wgu0 = _gather_sh_grad(qfsdp_p.layers[0].wgu_sh.grad)
    gq_wdn1 = _gather_sh_grad(qfsdp_p.layers[1].wdn_sh.grad)

    # Control: Naive Unrotated Per-Tensor RTN-INT8 Gradient ReduceScatter on Outlier-Spiked Gradients
    # (Demonstrating why naive unrotated per-tensor INT8 gradient communication fails!)
    raw_local_g = g1_wgu0.float()
    # Inject rank-specific outlier spike cancelling across ranks unless preserved:
    spike = torch.zeros_like(raw_local_g)
    spike[0, 0] = 50.0 * (1.0 if rank % 2 == 0 else -1.0) * raw_local_g.abs().max()
    noisy_rank_g = raw_local_g + spike
    sc_pt = noisy_rank_g.abs().max().clamp_min(1e-8) / 127.0
    q_pt = (noisy_rank_g / sc_pt).round().clamp(-127, 127).to(torch.int8)
    deq_pt = q_pt.float() * sc_pt / world_size
    dist.all_reduce(deq_pt, op=dist.ReduceOp.SUM)
    naive_pt_rtn_rs_rel_l2 = rel_l2(deq_pt, g1_wgu0)

    c1_parity = {
        "fsdp2_bf16_vs_1gpu": {
            "dw_layer0_wqkv_rel_l2": round(rel_l2(gf_wqkv0, g1_wqkv0), 6),
            "dw_layer0_wgu_rel_l2": round(rel_l2(gf_wgu0, g1_wgu0), 6),
            "dw_layer1_wdn_rel_l2": round(rel_l2(gf_wdn1, g1_wdn1), 6),
        },
        "halo_coat_quant_fsdp2_vs_1gpu": {
            "dw_layer0_wqkv_rel_l2": round(rel_l2(gq_wqkv0, g1_wqkv0), 6),
            "dw_layer0_wgu_rel_l2": round(rel_l2(gq_wgu0, g1_wgu0), 6),
            "dw_layer1_wdn_rel_l2": round(rel_l2(gq_wdn1, g1_wdn1), 6),
        },
        "naive_per_tensor_rtn_int8_grad_comm_bias_rel_l2": round(
            naive_pt_rtn_rs_rel_l2, 6
        ),
    }
    del qfsdp_p, p_cpu, g1_wqkv0, g1_wgu0, g1_wdn1, gf_wqkv0, gf_wgu0, gf_wdn1, gq_wqkv0, gq_wgu0, gq_wdn1
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # C.2 End-to-End 8-GPU Scaling: 6-Layer Llama-3-8B Geometry (1.39B Params)
    #     H = 4096, I = 14336, 32Q/8KV, D = 128; Global Tokens = 32,768 (4,096/GPU)
    # ------------------------------------------------------------------------
    n_layers_s = 2 if is_smoke else 6
    H_s, I_s, nq_s, nkv_s = 4096, 14336, 32, 8
    B_loc_s = 1 if is_smoke else 2
    S_s = 2048 if is_smoke else 4096
    M_loc_s = B_loc_s * S_s
    N_glob_s = M_loc_s * world_size
    warmup_s = 2 if is_smoke else 4
    iters_s = 5 if is_smoke else 12

    torch.manual_seed(12360)
    base_cpu = GQATransformerModelBF16(n_layers_s, H_s, I_s, nq_s, nkv_s)
    for p in base_cpu.parameters():
        if p.ndim == 2:
            nn.init.normal_(p, std=0.02)
    total_params = sum(p.numel() for p in base_cpu.parameters())
    params_billions = round(total_params / 1e9, 3)

    # AllGather bytes per step (reshard_after_forward=True -> Fwd + Bwd = 2 AllGathers):
    linear_params = sum(p.numel() for p in base_cpu.parameters() if p.ndim == 2)
    bf16_ag_mib_per_step = round((2.0 * linear_params * 2.0) / (1024.0**2), 1)
    int8_ag_mib_per_step = round((2.0 * linear_params * 1.001) / (1024.0**2), 1)

    torch.manual_seed(12370 + rank)
    x_loc_s = torch.randn(B_loc_s, S_s, H_s, device=device, dtype=torch.bfloat16)
    dy_loc_s = torch.randn(B_loc_s, S_s, H_s, device=device, dtype=torch.bfloat16)
    sm_s = rademacher_signs(M_loc_s, 950 + rank, device)

    def _bench_dist_mode(step_fn: Callable[[], None]) -> dict[str, float]:
        for _ in range(warmup_s):
            step_fn()
        torch.cuda.synchronize(device)
        dist.barrier()

        gc.collect()
        torch.cuda.empty_cache()
        base_m = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        step_fn()
        torch.cuda.synchronize(device)
        dist.barrier()
        loc_peak = max(
            0.0, (torch.cuda.max_memory_allocated(device) - base_m) / (1024.0**2)
        )
        pk_t = torch.tensor([loc_peak], device=device, dtype=torch.float64)
        dist.all_reduce(pk_t, op=dist.ReduceOp.MAX)
        max_rank_peak_mib = float(pk_t.item())

        times_ms: list[float] = []
        for _ in range(iters_s):
            dist.barrier()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            step_fn()
            e1.record()
            torch.cuda.synchronize(device)
            t_loc = torch.tensor([e0.elapsed_time(e1)], device=device, dtype=torch.float64)
            dist.all_reduce(t_loc, op=dist.ReduceOp.MAX)
            times_ms.append(float(t_loc.item()))

        med = float(statistics.median(times_ms))
        ss = sorted(times_ms)
        iqr = float(
            ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
            - ss[max(0, (len(ss) - 1) // 4)]
        )
        sm_clk, pwr_w = query_gpu_telemetry(device.index or 0)
        return {
            "step_ms_median": round(med, 4),
            "step_ms_iqr": round(iqr, 4),
            "per_rank_peak_inc_mib": round(max_rank_peak_mib, 2),
            "global_tokens_per_s": round(N_glob_s / (med * 1e-3), 1),
            "sm_clock_after_mhz": round(sm_clk, 1),
            "power_after_w": round(pwr_w, 2),
        }

    c2_modes: dict[str, Any] = {}

    # 1. FSDP2 BF16 Eager (reshard_after_forward=True)
    m_eag = GQATransformerModelBF16(
        n_layers_s, H_s, I_s, nq_s, nkv_s, use_ckpt=False, compile_mlp=False
    ).to(device=device)
    m_eag.load_state_dict(base_cpu.state_dict())
    m_eag = _wrap_fsdp2_gqa(m_eag)

    def _s_eag() -> None:
        for p in m_eag.parameters():
            p.grad = None
        y = m_eag(x_loc_s)
        y.backward(dy_loc_s)

    c2_modes["fsdp2_bf16_eager_reshard"] = _bench_dist_mode(_s_eag)
    del m_eag
    gc.collect()
    torch.cuda.empty_cache()

    # 2. FSDP2 BF16 Compiled MLP (reshard_after_forward=True)
    m_cmp = GQATransformerModelBF16(
        n_layers_s, H_s, I_s, nq_s, nkv_s, use_ckpt=False, compile_mlp=True
    ).to(device=device)
    m_cmp.load_state_dict(base_cpu.state_dict())
    m_cmp = _wrap_fsdp2_gqa(m_cmp)

    def _s_cmp() -> None:
        for p in m_cmp.parameters():
            p.grad = None
        y = m_cmp(x_loc_s)
        y.backward(dy_loc_s)

    c2_modes["fsdp2_bf16_compile_reshard"] = _bench_dist_mode(_s_cmp)
    del m_cmp
    gc.collect()
    torch.cuda.empty_cache()

    # 3. FSDP2 BF16 Full-Layer Activation Checkpointing
    m_ckp = GQATransformerModelBF16(
        n_layers_s, H_s, I_s, nq_s, nkv_s, use_ckpt=True, compile_mlp=False
    ).to(device=device)
    m_ckp.load_state_dict(base_cpu.state_dict())
    m_ckp = _wrap_fsdp2_gqa(m_ckp)

    def _s_ckp() -> None:
        for p in m_ckp.parameters():
            p.grad = None
        y = m_ckp(x_loc_s)
        y.backward(dy_loc_s)

    c2_modes["fsdp2_bf16_act_checkpoint"] = _bench_dist_mode(_s_ckp)
    del m_ckp
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Halo-COAT Quantized-AllGather (1 B/param INT8 AG + COAT INT8 Activations + SR-INT8 3-GEMM + FP32 RS)
    m_qfsdp = HaloCoatQuantFSDP2Model(base_cpu, rank, world_size, device, ws)

    def _s_qfsdp() -> None:
        for p in m_qfsdp.parameters():
            p.grad = None
        y = m_qfsdp(x_loc_s, sm_s)
        y.backward(dy_loc_s)

    c2_modes["halo_coat_quant_fsdp2"] = _bench_dist_mode(_s_qfsdp)
    del m_qfsdp, base_cpu
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "distributed_gradient_parity": c1_parity,
        "fsdp2_8gpu_training_scaling": {
            "num_layers": n_layers_s,
            "params_billions": params_billions,
            "local_tokens_per_gpu": M_loc_s,
            "global_tokens_per_step": N_glob_s,
            "seq_len": S_s,
            "bf16_allgather_mib_per_step": bf16_ag_mib_per_step,
            "int8_allgather_mib_per_step": int8_ag_mib_per_step,
            "modes": c2_modes,
        },
    }


# ============================================================================
# 7. Entrypoint
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E123: Block-Hadamard INT8 Tensor-Core Training, COAT INT8 Activations, & 8-GPU Quantized FSDP2"
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
        print(f"[E123] Suite 0 complete at +{time.perf_counter() - t0:.1f}s", flush=True)
        sa_res = run_suite_a_linear_3gemm_scaling(device, is_smoke)
        print(f"[E123] Suite A complete at +{time.perf_counter() - t0:.1f}s", flush=True)
        sb_res = run_suite_b_coat_swiglu_mlp_stack(device, is_smoke)
        print(f"[E123] Suite B complete at +{time.perf_counter() - t0:.1f}s", flush=True)

    dist.barrier()
    sc_res = run_suite_c_quantized_fsdp2_8gpu(rank, world_size, device, is_smoke)
    dist.barrier()
    elapsed_s = time.perf_counter() - t0

    if rank == 0:
        print(f"[E123] Suite C complete at +{elapsed_s:.1f}s", flush=True)
        env = environment_metadata()
        env["hostname"] = "node-0"
        env["triton"] = triton.__version__
        import flash_attn

        env["flash_attn"] = flash_attn.__version__
        payload = {
            "experiment": "E123_int8_quantized_training",
            "mode": args.mode,
            "world_size": world_size,
            "elapsed_s": round(elapsed_s, 2),
            "initial_sm_clock_mhz": round(init_clk, 1),
            "initial_power_w": round(init_pwr, 2),
            "environment": env,
            "suite_0_audits": s0_res,
            "suite_a_linear_3gemm_scaling": sa_res,
            "suite_b_coat_swiglu_mlp_stack": sb_res,
            "suite_c_quantized_fsdp2_8gpu": sc_res,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.out} in {elapsed_s:.2f}s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
