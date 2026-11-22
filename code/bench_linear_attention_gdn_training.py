from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
import triton
import triton.language as tl

from flash_attn import flash_attn_func
from common import environment_metadata

try:
    import pynvml  # type: ignore

    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False


# ============================================================================
# 1. Custom Fused Triton Kernels: Chunkwise Mamba-2 SSD / GLA (C = 64)
# ============================================================================


@triton.jit
def _ssd_gamma_cumsum_fwd_kernel(
    G_ptr,
    Gamma_ptr,
    S: tl.int32,
):
    c_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    t_base = bh_id.to(tl.int64) * S + c_id * 64
    c64 = tl.arange(0, 64)
    g64 = tl.load(G_ptr + t_base + c64).to(tl.float32)
    tl.store(Gamma_ptr + t_base + c64, tl.cumsum(g64, axis=0))


@triton.jit
def _ssd_state_scan_fwd_kernel(
    K_ptr,
    V_ptr,
    Gamma_ptr,
    S_init_ptr,
    S_ckpt_ptr,
    S_final_ptr,
    Nc: tl.int32,
    D: tl.constexpr,
    BV: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    vt_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    S = Nc * 64
    dk_offs = tl.arange(0, D)
    dv_offs = vt_id * BV + tl.arange(0, BV)
    c64 = tl.arange(0, 64)

    if HAS_INIT:
        s_base = bh_id.to(tl.int64) * D * D
        s = tl.load(
            S_init_ptr + s_base + dk_offs[:, None] * D + dv_offs[None, :]
        ).to(tl.float32)
    else:
        s = tl.zeros([D, BV], dtype=tl.float32)

    for c in range(0, Nc):
        ck_base = (bh_id.to(tl.int64) * Nc + c) * D * D
        tl.store(
            S_ckpt_ptr + ck_base + dk_offs[:, None] * D + dv_offs[None, :],
            s.to(tl.bfloat16),
        )
        t_base = bh_id.to(tl.int64) * S + c * 64
        kd_base = t_base * D
        gam = tl.load(Gamma_ptr + t_base + c64)
        g_end = tl.load(Gamma_ptr + t_base + 63)
        k_c = tl.load(
            K_ptr + kd_base + c64[:, None] * D + dk_offs[None, :]
        ).to(tl.float32)
        v_c = tl.load(
            V_ptr + kd_base + c64[:, None] * D + dv_offs[None, :]
        ).to(tl.bfloat16)
        k_end = (tl.exp(g_end - gam)[:, None] * k_c).to(tl.bfloat16)
        s = tl.exp(g_end) * s + tl.dot(tl.trans(k_end), v_c)

    sf_base = bh_id.to(tl.int64) * D * D
    tl.store(S_final_ptr + sf_base + dk_offs[:, None] * D + dv_offs[None, :], s)


@triton.jit
def _ssd_state_scan_bwd_kernel(
    Q_ptr,
    Gamma_ptr,
    dO_ptr,
    dS_init_ptr,
    dS_ckpt_ptr,
    dS_final_ptr,
    Nc: tl.int32,
    D: tl.constexpr,
    BV: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    vt_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    S = Nc * 64
    dk_offs = tl.arange(0, D)
    dv_offs = vt_id * BV + tl.arange(0, BV)
    c64 = tl.arange(0, 64)

    if HAS_INIT:
        s_base = bh_id.to(tl.int64) * D * D
        ds = tl.load(
            dS_init_ptr + s_base + dk_offs[:, None] * D + dv_offs[None, :]
        ).to(tl.float32)
    else:
        ds = tl.zeros([D, BV], dtype=tl.float32)

    for step_idx in range(0, Nc):
        c = (Nc - 1) - step_idx
        ck_base = (bh_id.to(tl.int64) * Nc + c) * D * D
        tl.store(
            dS_ckpt_ptr + ck_base + dk_offs[:, None] * D + dv_offs[None, :],
            ds.to(tl.bfloat16),
        )
        t_base = bh_id.to(tl.int64) * S + c * 64
        kd_base = t_base * D
        gam = tl.load(Gamma_ptr + t_base + c64)
        g_end = tl.load(Gamma_ptr + t_base + 63)
        q_c = tl.load(
            Q_ptr + kd_base + c64[:, None] * D + dk_offs[None, :]
        ).to(tl.float32)
        do_c = tl.load(
            dO_ptr + kd_base + c64[:, None] * D + dv_offs[None, :]
        ).to(tl.bfloat16)
        q_dec = (tl.exp(gam)[:, None] * q_c).to(tl.bfloat16)
        ds = tl.exp(g_end) * ds + tl.dot(tl.trans(q_dec), do_c)

    sf_base = bh_id.to(tl.int64) * D * D
    tl.store(
        dS_final_ptr + sf_base + dk_offs[:, None] * D + dv_offs[None, :], ds
    )


@triton.jit
def _ssd_chunk_bwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    S_ckpt_ptr,
    dS_ckpt_ptr,
    Gamma_ptr,
    dO_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    dG_ptr,
    Nc: tl.int32,
    D: tl.constexpr,
    BV: tl.constexpr,
):
    c_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    S = Nc * 64
    t_base = bh_id.to(tl.int64) * S + c_id * 64
    kd_base = t_base * D
    ck_base = (bh_id.to(tl.int64) * Nc + c_id) * D * D

    c64 = tl.arange(0, 64)
    dk_offs = tl.arange(0, D)
    ca_mask = c64[:, None] >= c64[None, :]

    gam = tl.load(Gamma_ptr + t_base + c64)
    g_end = tl.load(Gamma_ptr + t_base + 63)
    exp_g = tl.exp(gam)
    exp_end_minus_g = tl.exp(g_end - gam)

    q_c = tl.load(Q_ptr + kd_base + c64[:, None] * D + dk_offs[None, :]).to(
        tl.bfloat16
    )
    k_c = tl.load(K_ptr + kd_base + c64[:, None] * D + dk_offs[None, :]).to(
        tl.bfloat16
    )
    k_end = (exp_end_minus_g[:, None] * k_c.to(tl.float32)).to(tl.bfloat16)

    diff = tl.where(ca_mask, gam[:, None] - gam[None, :], 0.0)
    exp_ca = tl.where(ca_mask, tl.exp(diff), 0.0)
    m_qk = exp_ca * tl.dot(q_c, tl.trans(k_c))
    m_qk_t_bf16 = tl.trans(m_qk).to(tl.bfloat16)

    dm_qk = tl.zeros([64, 64], dtype=tl.float32)
    dq_inter_acc = tl.zeros([64, D], dtype=tl.float32)
    dk_end_acc = tl.zeros([64, D], dtype=tl.float32)
    s0_ds_dot = 0.0

    for v_start in tl.static_range(0, D, BV):
        dv_offs = v_start + tl.arange(0, BV)
        do_tile = tl.load(
            dO_ptr + kd_base + c64[:, None] * D + dv_offs[None, :]
        ).to(tl.bfloat16)
        v_tile = tl.load(
            V_ptr + kd_base + c64[:, None] * D + dv_offs[None, :]
        ).to(tl.bfloat16)
        s0_tile = tl.load(
            S_ckpt_ptr + ck_base + dk_offs[:, None] * D + dv_offs[None, :]
        ).to(tl.bfloat16)
        ds_tile = tl.load(
            dS_ckpt_ptr + ck_base + dk_offs[:, None] * D + dv_offs[None, :]
        ).to(tl.bfloat16)

        dm_qk += tl.dot(do_tile, tl.trans(v_tile))
        dq_inter_acc += tl.dot(do_tile, tl.trans(s0_tile))
        dk_end_acc += tl.dot(v_tile, tl.trans(ds_tile))
        s0_ds_dot += tl.sum(s0_tile.to(tl.float32) * ds_tile.to(tl.float32))

        dv_tile = tl.dot(m_qk_t_bf16, do_tile) + tl.dot(k_end, ds_tile)
        tl.store(
            dV_ptr + kd_base + c64[:, None] * D + dv_offs[None, :],
            dv_tile.to(tl.bfloat16),
        )

    dq_inter = exp_g[:, None] * dq_inter_acc
    d_qk = (dm_qk * exp_ca).to(tl.bfloat16)
    dq_c = dq_inter + tl.dot(d_qk, k_c)
    dk_c = tl.dot(tl.trans(d_qk), q_c) + exp_end_minus_g[:, None] * dk_end_acc

    tl.store(
        dQ_ptr + kd_base + c64[:, None] * D + dk_offs[None, :],
        dq_c.to(tl.bfloat16),
    )
    tl.store(
        dK_ptr + kd_base + c64[:, None] * D + dk_offs[None, :],
        dk_c.to(tl.bfloat16),
    )

    P_ca = dm_qk * m_qk
    d_exp_term = tl.sum(k_end.to(tl.float32) * dk_end_acc, axis=1)
    dg_end_tot = tl.exp(g_end) * s0_ds_dot + tl.sum(d_exp_term, axis=0)
    dgam = (
        tl.sum(q_c.to(tl.float32) * dq_inter, axis=1)
        + tl.sum(P_ca, axis=1)
        - tl.sum(P_ca, axis=0)
        - d_exp_term
        + tl.where(c64 == 63, dg_end_tot, 0.0)
    )
    tot_dgam = tl.sum(dgam, axis=0)
    dg_final = tot_dgam - tl.cumsum(dgam, axis=0) + dgam
    tl.store(dG_ptr + t_base + c64, dg_final)


# ============================================================================
# 2. Custom Fused Triton Kernels: Chunkwise WY Gated DeltaNet (C = 64)
# ============================================================================


@triton.jit
def _inv_unit_tril_16_tc(A_mm):
    """Exact 4-squaring nilpotent inversion of a 16x16 unit lower-triangular
    matrix `(I + A_mm)^-1 = (I - A)(I + A^2)(I + A^4)(I + A^8)` on SM80 Tensor Cores."""
    r = tl.arange(0, 16)
    eye = (r[:, None] == r[None, :]).to(tl.float32)
    N = -A_mm
    N2 = tl.dot(N.to(tl.bfloat16), N.to(tl.bfloat16))
    N4 = tl.dot(N2.to(tl.bfloat16), N2.to(tl.bfloat16))
    N8 = tl.dot(N4.to(tl.bfloat16), N4.to(tl.bfloat16))
    P1 = tl.dot((eye + N).to(tl.bfloat16), (eye + N2).to(tl.bfloat16))
    P2 = tl.dot((eye + N4).to(tl.bfloat16), (eye + N8).to(tl.bfloat16))
    return tl.dot(P1.to(tl.bfloat16), P2.to(tl.bfloat16))


@triton.jit
def _inv_unit_tril_16_ieee(A_mm):
    """FP32 IEEE CUDA-core variant used in Suite 0's Tensor-Core vs SIMT audit."""
    r = tl.arange(0, 16)
    eye = (r[:, None] == r[None, :]).to(tl.float32)
    N = -A_mm
    N2 = tl.dot(N, N, input_precision="ieee")
    N4 = tl.dot(N2, N2, input_precision="ieee")
    N8 = tl.dot(N4, N4, input_precision="ieee")
    P1 = tl.dot(eye + N, eye + N2, input_precision="ieee")
    P2 = tl.dot(eye + N4, eye + N8, input_precision="ieee")
    return tl.dot(P1, P2, input_precision="ieee")


@triton.jit
def _gdn_wy_prep_fwd_kernel(
    K_ptr,
    V_ptr,
    Beta_ptr,
    G_ptr,
    U_wy_ptr,
    W_wy_ptr,
    Gamma_ptr,
    S: tl.int32,
    D: tl.constexpr,
    USE_IEEE: tl.constexpr,
):
    c_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    t_base = bh_id.to(tl.int64) * S + c_id * 64
    kd_base = t_base * D

    r16 = tl.arange(0, 16)
    d_offs = tl.arange(0, D)
    st16 = r16[:, None] > r16[None, :]

    # Compute 4-block cumulative log-decay in registers (no global store->load race!)
    raw_g0 = tl.load(G_ptr + t_base + 0 + r16).to(tl.float32)
    raw_g1 = tl.load(G_ptr + t_base + 16 + r16).to(tl.float32)
    raw_g2 = tl.load(G_ptr + t_base + 32 + r16).to(tl.float32)
    raw_g3 = tl.load(G_ptr + t_base + 48 + r16).to(tl.float32)

    g0 = tl.cumsum(raw_g0, axis=0)
    s0 = tl.sum(raw_g0, axis=0)
    g1 = s0 + tl.cumsum(raw_g1, axis=0)
    s01 = s0 + tl.sum(raw_g1, axis=0)
    g2 = s01 + tl.cumsum(raw_g2, axis=0)
    s012 = s01 + tl.sum(raw_g2, axis=0)
    g3 = s012 + tl.cumsum(raw_g3, axis=0)

    tl.store(Gamma_ptr + t_base + 0 + r16, g0)
    tl.store(Gamma_ptr + t_base + 16 + r16, g1)
    tl.store(Gamma_ptr + t_base + 32 + r16, g2)
    tl.store(Gamma_ptr + t_base + 48 + r16, g3)

    b0 = tl.load(Beta_ptr + t_base + 0 + r16).to(tl.float32)
    b1 = tl.load(Beta_ptr + t_base + 16 + r16).to(tl.float32)
    b2 = tl.load(Beta_ptr + t_base + 32 + r16).to(tl.float32)
    b3 = tl.load(Beta_ptr + t_base + 48 + r16).to(tl.float32)

    k0 = tl.load(K_ptr + kd_base + (0 + r16)[:, None] * D + d_offs[None, :]).to(
        tl.bfloat16
    )
    k1 = tl.load(
        K_ptr + kd_base + (16 + r16)[:, None] * D + d_offs[None, :]
    ).to(tl.bfloat16)
    k2 = tl.load(
        K_ptr + kd_base + (32 + r16)[:, None] * D + d_offs[None, :]
    ).to(tl.bfloat16)
    k3 = tl.load(
        K_ptr + kd_base + (48 + r16)[:, None] * D + d_offs[None, :]
    ).to(tl.bfloat16)

    A00 = tl.where(
        st16,
        b0[:, None]
        * tl.exp(tl.where(st16, g0[:, None] - g0[None, :], 0.0))
        * tl.dot(k0, tl.trans(k0)),
        0.0,
    )
    A11 = tl.where(
        st16,
        b1[:, None]
        * tl.exp(tl.where(st16, g1[:, None] - g1[None, :], 0.0))
        * tl.dot(k1, tl.trans(k1)),
        0.0,
    )
    A22 = tl.where(
        st16,
        b2[:, None]
        * tl.exp(tl.where(st16, g2[:, None] - g2[None, :], 0.0))
        * tl.dot(k2, tl.trans(k2)),
        0.0,
    )
    A33 = tl.where(
        st16,
        b3[:, None]
        * tl.exp(tl.where(st16, g3[:, None] - g3[None, :], 0.0))
        * tl.dot(k3, tl.trans(k3)),
        0.0,
    )

    if USE_IEEE:
        T00_f = _inv_unit_tril_16_ieee(A00)
        T11_f = _inv_unit_tril_16_ieee(A11)
        T22_f = _inv_unit_tril_16_ieee(A22)
        T33_f = _inv_unit_tril_16_ieee(A33)
        A10_f = b1[:, None] * tl.exp(g1[:, None] - g0[None, :]) * tl.dot(k1, tl.trans(k0))
        A20_f = b2[:, None] * tl.exp(g2[:, None] - g0[None, :]) * tl.dot(k2, tl.trans(k0))
        A21_f = b2[:, None] * tl.exp(g2[:, None] - g1[None, :]) * tl.dot(k2, tl.trans(k1))
        A30_f = b3[:, None] * tl.exp(g3[:, None] - g0[None, :]) * tl.dot(k3, tl.trans(k0))
        A31_f = b3[:, None] * tl.exp(g3[:, None] - g1[None, :]) * tl.dot(k3, tl.trans(k1))
        A32_f = b3[:, None] * tl.exp(g3[:, None] - g2[None, :]) * tl.dot(k3, tl.trans(k2))

        W0 = tl.dot(T00_f, (b0 * tl.exp(g0))[:, None] * k0.to(tl.float32), input_precision="ieee")
        W1 = tl.dot(T11_f, (b1 * tl.exp(g1))[:, None] * k1.to(tl.float32) - tl.dot(A10_f, W0, input_precision="ieee"), input_precision="ieee")
        W2 = tl.dot(T22_f, (b2 * tl.exp(g2))[:, None] * k2.to(tl.float32) - tl.dot(A20_f, W0, input_precision="ieee") - tl.dot(A21_f, W1, input_precision="ieee"), input_precision="ieee")
        W3 = tl.dot(T33_f, (b3 * tl.exp(g3))[:, None] * k3.to(tl.float32) - tl.dot(A30_f, W0, input_precision="ieee") - tl.dot(A31_f, W1, input_precision="ieee") - tl.dot(A32_f, W2, input_precision="ieee"), input_precision="ieee")

        v0 = tl.load(V_ptr + kd_base + (0 + r16)[:, None] * D + d_offs[None, :]).to(tl.float32)
        v1 = tl.load(V_ptr + kd_base + (16 + r16)[:, None] * D + d_offs[None, :]).to(tl.float32)
        v2 = tl.load(V_ptr + kd_base + (32 + r16)[:, None] * D + d_offs[None, :]).to(tl.float32)
        v3 = tl.load(V_ptr + kd_base + (48 + r16)[:, None] * D + d_offs[None, :]).to(tl.float32)

        U0 = tl.dot(T00_f, b0[:, None] * v0, input_precision="ieee")
        U1 = tl.dot(T11_f, b1[:, None] * v1 - tl.dot(A10_f, U0, input_precision="ieee"), input_precision="ieee")
        U2 = tl.dot(T22_f, b2[:, None] * v2 - tl.dot(A20_f, U0, input_precision="ieee") - tl.dot(A21_f, U1, input_precision="ieee"), input_precision="ieee")
        U3 = tl.dot(T33_f, b3[:, None] * v3 - tl.dot(A30_f, U0, input_precision="ieee") - tl.dot(A31_f, U1, input_precision="ieee") - tl.dot(A32_f, U2, input_precision="ieee"), input_precision="ieee")
    else:
        T00 = _inv_unit_tril_16_tc(A00).to(tl.bfloat16)
        T11 = _inv_unit_tril_16_tc(A11).to(tl.bfloat16)
        T22 = _inv_unit_tril_16_tc(A22).to(tl.bfloat16)
        T33 = _inv_unit_tril_16_tc(A33).to(tl.bfloat16)

        A10 = (b1[:, None] * tl.exp(g1[:, None] - g0[None, :]) * tl.dot(k1, tl.trans(k0))).to(tl.bfloat16)
        A20 = (b2[:, None] * tl.exp(g2[:, None] - g0[None, :]) * tl.dot(k2, tl.trans(k0))).to(tl.bfloat16)
        A21 = (b2[:, None] * tl.exp(g2[:, None] - g1[None, :]) * tl.dot(k2, tl.trans(k1))).to(tl.bfloat16)
        A30 = (b3[:, None] * tl.exp(g3[:, None] - g0[None, :]) * tl.dot(k3, tl.trans(k0))).to(tl.bfloat16)
        A31 = (b3[:, None] * tl.exp(g3[:, None] - g1[None, :]) * tl.dot(k3, tl.trans(k1))).to(tl.bfloat16)
        A32 = (b3[:, None] * tl.exp(g3[:, None] - g2[None, :]) * tl.dot(k3, tl.trans(k2))).to(tl.bfloat16)

        W0 = tl.dot(T00, ((b0 * tl.exp(g0))[:, None] * k0.to(tl.float32)).to(tl.bfloat16)).to(tl.bfloat16)
        W1 = tl.dot(T11, ((b1 * tl.exp(g1))[:, None] * k1.to(tl.float32) - tl.dot(A10, W0)).to(tl.bfloat16)).to(tl.bfloat16)
        W2 = tl.dot(T22, ((b2 * tl.exp(g2))[:, None] * k2.to(tl.float32) - tl.dot(A20, W0) - tl.dot(A21, W1)).to(tl.bfloat16)).to(tl.bfloat16)
        W3 = tl.dot(T33, ((b3 * tl.exp(g3))[:, None] * k3.to(tl.float32) - tl.dot(A30, W0) - tl.dot(A31, W1) - tl.dot(A32, W2)).to(tl.bfloat16)).to(tl.bfloat16)

        v0 = tl.load(V_ptr + kd_base + (0 + r16)[:, None] * D + d_offs[None, :]).to(tl.float32)
        v1 = tl.load(V_ptr + kd_base + (16 + r16)[:, None] * D + d_offs[None, :]).to(tl.float32)
        v2 = tl.load(V_ptr + kd_base + (32 + r16)[:, None] * D + d_offs[None, :]).to(tl.float32)
        v3 = tl.load(V_ptr + kd_base + (48 + r16)[:, None] * D + d_offs[None, :]).to(tl.float32)

        U0 = tl.dot(T00, (b0[:, None] * v0).to(tl.bfloat16)).to(tl.bfloat16)
        U1 = tl.dot(T11, (b1[:, None] * v1 - tl.dot(A10, U0)).to(tl.bfloat16)).to(tl.bfloat16)
        U2 = tl.dot(T22, (b2[:, None] * v2 - tl.dot(A20, U0) - tl.dot(A21, U1)).to(tl.bfloat16)).to(tl.bfloat16)
        U3 = tl.dot(T33, (b3[:, None] * v3 - tl.dot(A30, U0) - tl.dot(A31, U1) - tl.dot(A32, U2)).to(tl.bfloat16)).to(tl.bfloat16)

    tl.store(W_wy_ptr + kd_base + (0 + r16)[:, None] * D + d_offs[None, :], W0.to(tl.bfloat16))
    tl.store(W_wy_ptr + kd_base + (16 + r16)[:, None] * D + d_offs[None, :], W1.to(tl.bfloat16))
    tl.store(W_wy_ptr + kd_base + (32 + r16)[:, None] * D + d_offs[None, :], W2.to(tl.bfloat16))
    tl.store(W_wy_ptr + kd_base + (48 + r16)[:, None] * D + d_offs[None, :], W3.to(tl.bfloat16))

    tl.store(U_wy_ptr + kd_base + (0 + r16)[:, None] * D + d_offs[None, :], U0.to(tl.bfloat16))
    tl.store(U_wy_ptr + kd_base + (16 + r16)[:, None] * D + d_offs[None, :], U1.to(tl.bfloat16))
    tl.store(U_wy_ptr + kd_base + (32 + r16)[:, None] * D + d_offs[None, :], U2.to(tl.bfloat16))
    tl.store(U_wy_ptr + kd_base + (48 + r16)[:, None] * D + d_offs[None, :], U3.to(tl.bfloat16))


@triton.jit
def _gdn_state_scan_fwd_kernel(
    K_ptr,
    U_wy_ptr,
    W_wy_ptr,
    Gamma_ptr,
    S_init_ptr,
    S_ckpt_ptr,
    U_corr_ptr,
    S_final_ptr,
    Nc: tl.int32,
    D: tl.constexpr,
    DV_STRIDE: tl.constexpr,
    BV: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    vt_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    S = Nc * 64
    dk_offs = tl.arange(0, D)
    dv_offs = vt_id * BV + tl.arange(0, BV)
    c64 = tl.arange(0, 64)

    if HAS_INIT:
        s_base = bh_id.to(tl.int64) * D * DV_STRIDE
        s = tl.load(
            S_init_ptr + s_base + dk_offs[:, None] * DV_STRIDE + dv_offs[None, :]
        ).to(tl.float32)
    else:
        s = tl.zeros([D, BV], dtype=tl.float32)

    for c in range(0, Nc):
        ck_base = (bh_id.to(tl.int64) * Nc + c) * D * DV_STRIDE
        tl.store(
            S_ckpt_ptr
            + ck_base
            + dk_offs[:, None] * DV_STRIDE
            + dv_offs[None, :],
            s.to(tl.bfloat16),
        )

        t_base = bh_id.to(tl.int64) * S + c * 64
        kd_base = t_base * D
        uv_base = t_base * DV_STRIDE

        w_wy = tl.load(
            W_wy_ptr + kd_base + c64[:, None] * D + dk_offs[None, :]
        ).to(tl.bfloat16)
        u_wy = tl.load(
            U_wy_ptr + uv_base + c64[:, None] * DV_STRIDE + dv_offs[None, :]
        ).to(tl.float32)
        u_corr = u_wy - tl.dot(w_wy, s.to(tl.bfloat16))
        tl.store(
            U_corr_ptr + uv_base + c64[:, None] * DV_STRIDE + dv_offs[None, :],
            u_corr.to(tl.bfloat16),
        )

        gam = tl.load(Gamma_ptr + t_base + c64)
        g_end = tl.load(Gamma_ptr + t_base + 63)
        k_c = tl.load(
            K_ptr + kd_base + c64[:, None] * D + dk_offs[None, :]
        ).to(tl.float32)
        k_end = (tl.exp(g_end - gam)[:, None] * k_c).to(tl.bfloat16)
        s = tl.exp(g_end) * s + tl.dot(tl.trans(k_end), u_corr.to(tl.bfloat16))

    sf_base = bh_id.to(tl.int64) * D * DV_STRIDE
    tl.store(
        S_final_ptr + sf_base + dk_offs[:, None] * DV_STRIDE + dv_offs[None, :],
        s,
    )


@triton.jit
def _gdn_output_fwd_kernel(
    Q_ptr,
    K_ptr,
    U_corr_ptr,
    S_ckpt_ptr,
    Gamma_ptr,
    O_ptr,
    Nc: tl.int32,
    D: tl.constexpr,
    BV: tl.constexpr,
):
    c_id = tl.program_id(0)
    vt_id = tl.program_id(1)
    bh_id = tl.program_id(2)
    S = Nc * 64
    t_base = bh_id.to(tl.int64) * S + c_id * 64
    kd_base = t_base * D
    ck_base = (bh_id.to(tl.int64) * Nc + c_id) * D * D

    c64 = tl.arange(0, 64)
    dk_offs = tl.arange(0, D)
    dv_offs = vt_id * BV + tl.arange(0, BV)
    ca_mask = c64[:, None] >= c64[None, :]

    gam = tl.load(Gamma_ptr + t_base + c64)
    q_c = tl.load(Q_ptr + kd_base + c64[:, None] * D + dk_offs[None, :]).to(
        tl.bfloat16
    )
    k_c = tl.load(K_ptr + kd_base + c64[:, None] * D + dk_offs[None, :]).to(
        tl.bfloat16
    )
    s0 = tl.load(
        S_ckpt_ptr + ck_base + dk_offs[:, None] * D + dv_offs[None, :]
    ).to(tl.bfloat16)
    u_c = tl.load(
        U_corr_ptr + kd_base + c64[:, None] * D + dv_offs[None, :]
    ).to(tl.bfloat16)

    q_dec = (tl.exp(gam)[:, None] * q_c.to(tl.float32)).to(tl.bfloat16)
    o_inter = tl.dot(q_dec, s0)

    diff = tl.where(ca_mask, gam[:, None] - gam[None, :], 0.0)
    m_qk = tl.where(
        ca_mask, tl.exp(diff) * tl.dot(q_c, tl.trans(k_c)), 0.0
    ).to(tl.bfloat16)
    o_intra = tl.dot(m_qk, u_c)
    tl.store(
        O_ptr + kd_base + c64[:, None] * D + dv_offs[None, :],
        (o_inter + o_intra).to(tl.bfloat16),
    )


@triton.jit
def _gdn_bwd_stage1_kernel(
    Q_ptr,
    K_ptr,
    U_corr_ptr,
    S_ckpt_ptr,
    Gamma_ptr,
    dO_ptr,
    dQ_ptr,
    dK_o_ptr,
    dU_intra_ptr,
    dGamma_o_ptr,
    Nc: tl.int32,
    D: tl.constexpr,
    BV: tl.constexpr,
):
    c_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    S = Nc * 64
    t_base = bh_id.to(tl.int64) * S + c_id * 64
    kd_base = t_base * D
    ck_base = (bh_id.to(tl.int64) * Nc + c_id) * D * D

    c64 = tl.arange(0, 64)
    dk_offs = tl.arange(0, D)
    ca_mask = c64[:, None] >= c64[None, :]

    gam = tl.load(Gamma_ptr + t_base + c64)
    exp_g = tl.exp(gam)
    q_c = tl.load(Q_ptr + kd_base + c64[:, None] * D + dk_offs[None, :]).to(
        tl.bfloat16
    )
    k_c = tl.load(K_ptr + kd_base + c64[:, None] * D + dk_offs[None, :]).to(
        tl.bfloat16
    )

    diff = tl.where(ca_mask, gam[:, None] - gam[None, :], 0.0)
    exp_ca = tl.where(ca_mask, tl.exp(diff), 0.0)
    m_qk = exp_ca * tl.dot(q_c, tl.trans(k_c))
    m_qk_t_bf16 = tl.trans(m_qk).to(tl.bfloat16)

    dm_qk = tl.zeros([64, 64], dtype=tl.float32)
    dq_inter_acc = tl.zeros([64, D], dtype=tl.float32)

    for v_start in tl.static_range(0, D, BV):
        dv_offs = v_start + tl.arange(0, BV)
        do_tile = tl.load(
            dO_ptr + kd_base + c64[:, None] * D + dv_offs[None, :]
        ).to(tl.bfloat16)
        uc_tile = tl.load(
            U_corr_ptr + kd_base + c64[:, None] * D + dv_offs[None, :]
        ).to(tl.bfloat16)
        s0_tile = tl.load(
            S_ckpt_ptr + ck_base + dk_offs[:, None] * D + dv_offs[None, :]
        ).to(tl.bfloat16)

        dm_qk += tl.dot(do_tile, tl.trans(uc_tile))
        dq_inter_acc += tl.dot(do_tile, tl.trans(s0_tile))
        du_in = tl.dot(m_qk_t_bf16, do_tile)
        tl.store(
            dU_intra_ptr + kd_base + c64[:, None] * D + dv_offs[None, :],
            du_in.to(tl.bfloat16),
        )

    dq_inter = exp_g[:, None] * dq_inter_acc
    d_qk = (dm_qk * exp_ca).to(tl.bfloat16)
    dq_c = dq_inter + tl.dot(d_qk, k_c)
    dk_o = tl.dot(tl.trans(d_qk), q_c)

    tl.store(
        dQ_ptr + kd_base + c64[:, None] * D + dk_offs[None, :],
        dq_c.to(tl.bfloat16),
    )
    tl.store(
        dK_o_ptr + kd_base + c64[:, None] * D + dk_offs[None, :],
        dk_o.to(tl.bfloat16),
    )

    P_ca = dm_qk * m_qk
    dgam = (
        tl.sum(q_c.to(tl.float32) * dq_inter, axis=1)
        + tl.sum(P_ca, axis=1)
        - tl.sum(P_ca, axis=0)
    )
    tl.store(dGamma_o_ptr + t_base + c64, dgam)


@triton.jit
def _gdn_state_scan_bwd_kernel(
    Q_ptr,
    K_ptr,
    W_wy_ptr,
    Gamma_ptr,
    dO_ptr,
    dU_intra_ptr,
    dS_init_ptr,
    dS_ckpt_ptr,
    dU_wy_ptr,
    dS_final_ptr,
    Nc: tl.int32,
    D: tl.constexpr,
    BV: tl.constexpr,
    HAS_INIT: tl.constexpr,
):
    vt_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    S = Nc * 64
    dk_offs = tl.arange(0, D)
    dv_offs = vt_id * BV + tl.arange(0, BV)
    c64 = tl.arange(0, 64)

    if HAS_INIT:
        s_base = bh_id.to(tl.int64) * D * D
        ds = tl.load(
            dS_init_ptr + s_base + dk_offs[:, None] * D + dv_offs[None, :]
        ).to(tl.float32)
    else:
        ds = tl.zeros([D, BV], dtype=tl.float32)

    for step_idx in range(0, Nc):
        c = (Nc - 1) - step_idx
        ck_base = (bh_id.to(tl.int64) * Nc + c) * D * D
        tl.store(
            dS_ckpt_ptr + ck_base + dk_offs[:, None] * D + dv_offs[None, :],
            ds.to(tl.bfloat16),
        )

        t_base = bh_id.to(tl.int64) * S + c * 64
        kd_base = t_base * D

        gam = tl.load(Gamma_ptr + t_base + c64)
        g_end = tl.load(Gamma_ptr + t_base + 63)
        k_c = tl.load(
            K_ptr + kd_base + c64[:, None] * D + dk_offs[None, :]
        ).to(tl.float32)
        k_end = (tl.exp(g_end - gam)[:, None] * k_c).to(tl.bfloat16)

        du_in = tl.load(
            dU_intra_ptr + kd_base + c64[:, None] * D + dv_offs[None, :]
        ).to(tl.float32)
        du_c = du_in + tl.dot(k_end, ds.to(tl.bfloat16))
        tl.store(
            dU_wy_ptr + kd_base + c64[:, None] * D + dv_offs[None, :],
            du_c.to(tl.bfloat16),
        )

        q_c = tl.load(
            Q_ptr + kd_base + c64[:, None] * D + dk_offs[None, :]
        ).to(tl.float32)
        q_dec = (tl.exp(gam)[:, None] * q_c).to(tl.bfloat16)
        do_c = tl.load(
            dO_ptr + kd_base + c64[:, None] * D + dv_offs[None, :]
        ).to(tl.bfloat16)
        w_wy = tl.load(
            W_wy_ptr + kd_base + c64[:, None] * D + dk_offs[None, :]
        ).to(tl.bfloat16)

        ds = (
            tl.exp(g_end) * ds
            + tl.dot(tl.trans(q_dec), do_c)
            - tl.dot(tl.trans(w_wy), du_c.to(tl.bfloat16))
        )

    sf_base = bh_id.to(tl.int64) * D * D
    tl.store(
        dS_final_ptr + sf_base + dk_offs[:, None] * D + dv_offs[None, :], ds
    )


@triton.jit
def _gdn_wy_prep_bwd_kernel(
    K_ptr,
    V_ptr,
    Beta_ptr,
    Gamma_ptr,
    U_wy_ptr,
    W_wy_ptr,
    U_corr_ptr,
    S_ckpt_ptr,
    dS_ckpt_ptr,
    dU_wy_ptr,
    dK_o_ptr,
    dGamma_o_ptr,
    dK_ptr,
    dV_ptr,
    dBeta_ptr,
    dG_ptr,
    Nc: tl.int32,
    D: tl.constexpr,
    BV: tl.constexpr,
):
    """100% register-pure (zero global scratch / zero intra-CTA races) WY prep
    backward via 4x16 block backward substitution on SM80 Tensor Cores."""
    c_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    S = Nc * 64
    t_base = bh_id.to(tl.int64) * S + c_id * 64
    kd_base = t_base * D
    ck_base = (bh_id.to(tl.int64) * Nc + c_id) * D * D

    r16 = tl.arange(0, 16)
    dk_offs = tl.arange(0, D)
    st16 = r16[:, None] > r16[None, :]

    g0 = tl.load(Gamma_ptr + t_base + 0 + r16)
    g1 = tl.load(Gamma_ptr + t_base + 16 + r16)
    g2 = tl.load(Gamma_ptr + t_base + 32 + r16)
    g3 = tl.load(Gamma_ptr + t_base + 48 + r16)
    g_end = tl.load(Gamma_ptr + t_base + 63)

    b0 = tl.load(Beta_ptr + t_base + 0 + r16).to(tl.float32)
    b1 = tl.load(Beta_ptr + t_base + 16 + r16).to(tl.float32)
    b2 = tl.load(Beta_ptr + t_base + 32 + r16).to(tl.float32)
    b3 = tl.load(Beta_ptr + t_base + 48 + r16).to(tl.float32)

    # Accumulate dw0..3 [16, D] and dke0..3 [16, D] directly in 16xD register tiles
    dw0 = tl.zeros([16, D], dtype=tl.float32)
    dw1 = tl.zeros([16, D], dtype=tl.float32)
    dw2 = tl.zeros([16, D], dtype=tl.float32)
    dw3 = tl.zeros([16, D], dtype=tl.float32)

    dke0 = tl.zeros([16, D], dtype=tl.float32)
    dke1 = tl.zeros([16, D], dtype=tl.float32)
    dke2 = tl.zeros([16, D], dtype=tl.float32)
    dke3 = tl.zeros([16, D], dtype=tl.float32)
    s0_ds_dot = 0.0

    for v_start in tl.static_range(0, D, BV):
        dv_offs = v_start + tl.arange(0, BV)
        s0_t = tl.load(
            S_ckpt_ptr + ck_base + dk_offs[:, None] * D + dv_offs[None, :]
        ).to(tl.bfloat16)
        ds_t = tl.load(
            dS_ckpt_ptr + ck_base + dk_offs[:, None] * D + dv_offs[None, :]
        ).to(tl.bfloat16)
        s0_ds_dot += tl.sum(s0_t.to(tl.float32) * ds_t.to(tl.float32))
        s0_t_tr = tl.trans(s0_t)
        ds_t_tr = tl.trans(ds_t)

        du0_t = tl.load(dU_wy_ptr + kd_base + (0 + r16)[:, None] * D + dv_offs[None, :]).to(tl.bfloat16)
        du1_t = tl.load(dU_wy_ptr + kd_base + (16 + r16)[:, None] * D + dv_offs[None, :]).to(tl.bfloat16)
        du2_t = tl.load(dU_wy_ptr + kd_base + (32 + r16)[:, None] * D + dv_offs[None, :]).to(tl.bfloat16)
        du3_t = tl.load(dU_wy_ptr + kd_base + (48 + r16)[:, None] * D + dv_offs[None, :]).to(tl.bfloat16)

        dw0 -= tl.dot(du0_t, s0_t_tr)
        dw1 -= tl.dot(du1_t, s0_t_tr)
        dw2 -= tl.dot(du2_t, s0_t_tr)
        dw3 -= tl.dot(du3_t, s0_t_tr)

        uc0_t = tl.load(U_corr_ptr + kd_base + (0 + r16)[:, None] * D + dv_offs[None, :]).to(tl.bfloat16)
        uc1_t = tl.load(U_corr_ptr + kd_base + (16 + r16)[:, None] * D + dv_offs[None, :]).to(tl.bfloat16)
        uc2_t = tl.load(U_corr_ptr + kd_base + (32 + r16)[:, None] * D + dv_offs[None, :]).to(tl.bfloat16)
        uc3_t = tl.load(U_corr_ptr + kd_base + (48 + r16)[:, None] * D + dv_offs[None, :]).to(tl.bfloat16)

        dke0 += tl.dot(uc0_t, ds_t_tr)
        dke1 += tl.dot(uc1_t, ds_t_tr)
        dke2 += tl.dot(uc2_t, ds_t_tr)
        dke3 += tl.dot(uc3_t, ds_t_tr)

    k0 = tl.load(K_ptr + kd_base + (0 + r16)[:, None] * D + dk_offs[None, :]).to(tl.bfloat16)
    k1 = tl.load(K_ptr + kd_base + (16 + r16)[:, None] * D + dk_offs[None, :]).to(tl.bfloat16)
    k2 = tl.load(K_ptr + kd_base + (32 + r16)[:, None] * D + dk_offs[None, :]).to(tl.bfloat16)
    k3 = tl.load(K_ptr + kd_base + (48 + r16)[:, None] * D + dk_offs[None, :]).to(tl.bfloat16)

    ee0 = tl.exp(g_end - g0)
    ee1 = tl.exp(g_end - g1)
    ee2 = tl.exp(g_end - g2)
    ee3 = tl.exp(g_end - g3)

    d_exp0 = tl.sum((ee0[:, None] * k0.to(tl.float32)) * dke0, axis=1)
    d_exp1 = tl.sum((ee1[:, None] * k1.to(tl.float32)) * dke1, axis=1)
    d_exp2 = tl.sum((ee2[:, None] * k2.to(tl.float32)) * dke2, axis=1)
    d_exp3 = tl.sum((ee3[:, None] * k3.to(tl.float32)) * dke3, axis=1)
    dg_end_tot = tl.exp(g_end) * s0_ds_dot + tl.sum(d_exp0 + d_exp1 + d_exp2 + d_exp3, axis=0)

    kkt00 = tl.dot(k0, tl.trans(k0)); exp00 = tl.where(st16, tl.exp(tl.where(st16, g0[:, None] - g0[None, :], 0.0)), 0.0); A00 = b0[:, None] * exp00 * kkt00
    kkt11 = tl.dot(k1, tl.trans(k1)); exp11 = tl.where(st16, tl.exp(tl.where(st16, g1[:, None] - g1[None, :], 0.0)), 0.0); A11 = b1[:, None] * exp11 * kkt11
    kkt22 = tl.dot(k2, tl.trans(k2)); exp22 = tl.where(st16, tl.exp(tl.where(st16, g2[:, None] - g2[None, :], 0.0)), 0.0); A22 = b2[:, None] * exp22 * kkt22
    kkt33 = tl.dot(k3, tl.trans(k3)); exp33 = tl.where(st16, tl.exp(tl.where(st16, g3[:, None] - g3[None, :], 0.0)), 0.0); A33 = b3[:, None] * exp33 * kkt33

    T00_t = tl.trans(_inv_unit_tril_16_tc(A00)).to(tl.bfloat16)
    T11_t = tl.trans(_inv_unit_tril_16_tc(A11)).to(tl.bfloat16)
    T22_t = tl.trans(_inv_unit_tril_16_tc(A22)).to(tl.bfloat16)
    T33_t = tl.trans(_inv_unit_tril_16_tc(A33)).to(tl.bfloat16)

    kkt10 = tl.dot(k1, tl.trans(k0)); exp10 = tl.exp(g1[:, None] - g0[None, :]); A10 = b1[:, None] * exp10 * kkt10
    kkt20 = tl.dot(k2, tl.trans(k0)); exp20 = tl.exp(g2[:, None] - g0[None, :]); A20 = b2[:, None] * exp20 * kkt20
    kkt21 = tl.dot(k2, tl.trans(k1)); exp21 = tl.exp(g2[:, None] - g1[None, :]); A21 = b2[:, None] * exp21 * kkt21
    kkt30 = tl.dot(k3, tl.trans(k0)); exp30 = tl.exp(g3[:, None] - g0[None, :]); A30 = b3[:, None] * exp30 * kkt30
    kkt31 = tl.dot(k3, tl.trans(k1)); exp31 = tl.exp(g3[:, None] - g1[None, :]); A31 = b3[:, None] * exp31 * kkt31
    kkt32 = tl.dot(k3, tl.trans(k2)); exp32 = tl.exp(g3[:, None] - g2[None, :]); A32 = b3[:, None] * exp32 * kkt32

    A10_t = tl.trans(A10).to(tl.bfloat16); A20_t = tl.trans(A20).to(tl.bfloat16); A21_t = tl.trans(A21).to(tl.bfloat16)
    A30_t = tl.trans(A30).to(tl.bfloat16); A31_t = tl.trans(A31).to(tl.bfloat16); A32_t = tl.trans(A32).to(tl.bfloat16)

    du0 = tl.load(dU_wy_ptr + kd_base + (0 + r16)[:, None] * D + dk_offs[None, :]).to(tl.float32)
    du1 = tl.load(dU_wy_ptr + kd_base + (16 + r16)[:, None] * D + dk_offs[None, :]).to(tl.float32)
    du2 = tl.load(dU_wy_ptr + kd_base + (32 + r16)[:, None] * D + dk_offs[None, :]).to(tl.float32)
    du3 = tl.load(dU_wy_ptr + kd_base + (48 + r16)[:, None] * D + dk_offs[None, :]).to(tl.float32)

    ZU3 = tl.dot(T33_t, du3.to(tl.bfloat16))
    ZU2 = tl.dot(T22_t, (du2 - tl.dot(A32_t, ZU3.to(tl.bfloat16))).to(tl.bfloat16))
    ZU1 = tl.dot(T11_t, (du1 - tl.dot(A21_t, ZU2.to(tl.bfloat16)) - tl.dot(A31_t, ZU3.to(tl.bfloat16))).to(tl.bfloat16))
    ZU0 = tl.dot(T00_t, (du0 - tl.dot(A10_t, ZU1.to(tl.bfloat16)) - tl.dot(A20_t, ZU2.to(tl.bfloat16)) - tl.dot(A30_t, ZU3.to(tl.bfloat16))).to(tl.bfloat16))

    ZW3 = tl.dot(T33_t, dw3.to(tl.bfloat16))
    ZW2 = tl.dot(T22_t, (dw2 - tl.dot(A32_t, ZW3.to(tl.bfloat16))).to(tl.bfloat16))
    ZW1 = tl.dot(T11_t, (dw1 - tl.dot(A21_t, ZW2.to(tl.bfloat16)) - tl.dot(A31_t, ZW3.to(tl.bfloat16))).to(tl.bfloat16))
    ZW0 = tl.dot(T00_t, (dw0 - tl.dot(A10_t, ZW1.to(tl.bfloat16)) - tl.dot(A20_t, ZW2.to(tl.bfloat16)) - tl.dot(A30_t, ZW3.to(tl.bfloat16))).to(tl.bfloat16))

    v0 = tl.load(V_ptr + kd_base + (0 + r16)[:, None] * D + dk_offs[None, :]).to(tl.float32)
    v1 = tl.load(V_ptr + kd_base + (16 + r16)[:, None] * D + dk_offs[None, :]).to(tl.float32)
    v2 = tl.load(V_ptr + kd_base + (32 + r16)[:, None] * D + dk_offs[None, :]).to(tl.float32)
    v3 = tl.load(V_ptr + kd_base + (48 + r16)[:, None] * D + dk_offs[None, :]).to(tl.float32)

    tl.store(dV_ptr + kd_base + (0 + r16)[:, None] * D + dk_offs[None, :], (b0[:, None] * ZU0).to(tl.bfloat16))
    tl.store(dV_ptr + kd_base + (16 + r16)[:, None] * D + dk_offs[None, :], (b1[:, None] * ZU1).to(tl.bfloat16))
    tl.store(dV_ptr + kd_base + (32 + r16)[:, None] * D + dk_offs[None, :], (b2[:, None] * ZU2).to(tl.bfloat16))
    tl.store(dV_ptr + kd_base + (48 + r16)[:, None] * D + dk_offs[None, :], (b3[:, None] * ZU3).to(tl.bfloat16))

    U0 = tl.load(U_wy_ptr + kd_base + (0 + r16)[:, None] * D + dk_offs[None, :]).to(tl.bfloat16)
    U1 = tl.load(U_wy_ptr + kd_base + (16 + r16)[:, None] * D + dk_offs[None, :]).to(tl.bfloat16)
    U2 = tl.load(U_wy_ptr + kd_base + (32 + r16)[:, None] * D + dk_offs[None, :]).to(tl.bfloat16)
    U3 = tl.load(U_wy_ptr + kd_base + (48 + r16)[:, None] * D + dk_offs[None, :]).to(tl.bfloat16)

    W0 = tl.load(W_wy_ptr + kd_base + (0 + r16)[:, None] * D + dk_offs[None, :]).to(tl.bfloat16)
    W1 = tl.load(W_wy_ptr + kd_base + (16 + r16)[:, None] * D + dk_offs[None, :]).to(tl.bfloat16)
    W2 = tl.load(W_wy_ptr + kd_base + (32 + r16)[:, None] * D + dk_offs[None, :]).to(tl.bfloat16)
    W3 = tl.load(W_wy_ptr + kd_base + (48 + r16)[:, None] * D + dk_offs[None, :]).to(tl.bfloat16)

    ZU0b = ZU0.to(tl.bfloat16); ZU1b = ZU1.to(tl.bfloat16); ZU2b = ZU2.to(tl.bfloat16); ZU3b = ZU3.to(tl.bfloat16)
    ZW0b = ZW0.to(tl.bfloat16); ZW1b = ZW1.to(tl.bfloat16); ZW2b = ZW2.to(tl.bfloat16); ZW3b = ZW3.to(tl.bfloat16)

    dA00 = tl.where(st16, -tl.dot(ZU0b, tl.trans(U0)) - tl.dot(ZW0b, tl.trans(W0)), 0.0)
    dA11 = tl.where(st16, -tl.dot(ZU1b, tl.trans(U1)) - tl.dot(ZW1b, tl.trans(W1)), 0.0)
    dA22 = tl.where(st16, -tl.dot(ZU2b, tl.trans(U2)) - tl.dot(ZW2b, tl.trans(W2)), 0.0)
    dA33 = tl.where(st16, -tl.dot(ZU3b, tl.trans(U3)) - tl.dot(ZW3b, tl.trans(W3)), 0.0)

    dA10 = -tl.dot(ZU1b, tl.trans(U0)) - tl.dot(ZW1b, tl.trans(W0))
    dA20 = -tl.dot(ZU2b, tl.trans(U0)) - tl.dot(ZW2b, tl.trans(W0))
    dA21 = -tl.dot(ZU2b, tl.trans(U1)) - tl.dot(ZW2b, tl.trans(W1))
    dA30 = -tl.dot(ZU3b, tl.trans(U0)) - tl.dot(ZW3b, tl.trans(W0))
    dA31 = -tl.dot(ZU3b, tl.trans(U1)) - tl.dot(ZW3b, tl.trans(W1))
    dA32 = -tl.dot(ZU3b, tl.trans(U2)) - tl.dot(ZW3b, tl.trans(W2))

    zwk0 = tl.sum(ZW0 * k0.to(tl.float32), axis=1); zwk1 = tl.sum(ZW1 * k1.to(tl.float32), axis=1)
    zwk2 = tl.sum(ZW2 * k2.to(tl.float32), axis=1); zwk3 = tl.sum(ZW3 * k3.to(tl.float32), axis=1)

    db0 = tl.sum(ZU0 * v0, axis=1) + tl.exp(g0) * zwk0 + tl.sum(dA00 * exp00 * kkt00, axis=1)
    db1 = tl.sum(ZU1 * v1, axis=1) + tl.exp(g1) * zwk1 + tl.sum(dA10 * exp10 * kkt10 + dA11 * exp11 * kkt11, axis=1)
    db2 = tl.sum(ZU2 * v2, axis=1) + tl.exp(g2) * zwk2 + tl.sum(dA20 * exp20 * kkt20 + dA21 * exp21 * kkt21 + dA22 * exp22 * kkt22, axis=1)
    db3 = tl.sum(ZU3 * v3, axis=1) + tl.exp(g3) * zwk3 + tl.sum(dA30 * exp30 * kkt30 + dA31 * exp31 * kkt31 + dA32 * exp32 * kkt32 + dA33 * exp33 * kkt33, axis=1)

    tl.store(dBeta_ptr + t_base + 0 + r16, db0)
    tl.store(dBeta_ptr + t_base + 16 + r16, db1)
    tl.store(dBeta_ptr + t_base + 32 + r16, db2)
    tl.store(dBeta_ptr + t_base + 48 + r16, db3)

    P00 = dA00 * A00; P11 = dA11 * A11; P22 = dA22 * A22; P33 = dA33 * A33
    P10 = dA10 * A10; P20 = dA20 * A20; P21 = dA21 * A21
    P30 = dA30 * A30; P31 = dA31 * A31; P32 = dA32 * A32

    dgo0 = tl.load(dGamma_o_ptr + t_base + 0 + r16)
    dgo1 = tl.load(dGamma_o_ptr + t_base + 16 + r16)
    dgo2 = tl.load(dGamma_o_ptr + t_base + 32 + r16)
    dgo3 = tl.load(dGamma_o_ptr + t_base + 48 + r16)

    dgam0 = dgo0 - d_exp0 + b0 * tl.exp(g0) * zwk0 + tl.sum(P00, axis=1) - tl.sum(P00 + P10 + P20 + P30, axis=0)
    dgam1 = dgo1 - d_exp1 + b1 * tl.exp(g1) * zwk1 + tl.sum(P10 + P11, axis=1) - tl.sum(P11 + P21 + P31, axis=0)
    dgam2 = dgo2 - d_exp2 + b2 * tl.exp(g2) * zwk2 + tl.sum(P20 + P21 + P22, axis=1) - tl.sum(P22 + P32, axis=0)
    dgam3 = dgo3 - d_exp3 + b3 * tl.exp(g3) * zwk3 + tl.sum(P30 + P31 + P32 + P33, axis=1) - tl.sum(P33, axis=0) + tl.where(r16 == 15, dg_end_tot, 0.0)

    # 4-block reverse cumsum in registers (0 global scratch!)
    s3 = tl.sum(dgam3, axis=0)
    dg3_out = s3 - tl.cumsum(dgam3, axis=0) + dgam3
    s2 = tl.sum(dgam2, axis=0)
    dg2_out = s3 + (s2 - tl.cumsum(dgam2, axis=0) + dgam2)
    s23 = s3 + s2
    s1 = tl.sum(dgam1, axis=0)
    dg1_out = s23 + (s1 - tl.cumsum(dgam1, axis=0) + dgam1)
    s123 = s23 + s1
    s0_g = tl.sum(dgam0, axis=0)
    dg0_out = s123 + (s0_g - tl.cumsum(dgam0, axis=0) + dgam0)

    tl.store(dG_ptr + t_base + 0 + r16, dg0_out)
    tl.store(dG_ptr + t_base + 16 + r16, dg1_out)
    tl.store(dG_ptr + t_base + 32 + r16, dg2_out)
    tl.store(dG_ptr + t_base + 48 + r16, dg3_out)

    dkkt00 = dA00 * b0[:, None] * exp00; dkkt11 = dA11 * b1[:, None] * exp11
    dkkt22 = dA22 * b2[:, None] * exp22; dkkt33 = dA33 * b3[:, None] * exp33
    dkkt10 = dA10 * b1[:, None] * exp10; dkkt20 = dA20 * b2[:, None] * exp20; dkkt21 = dA21 * b2[:, None] * exp21
    dkkt30 = dA30 * b3[:, None] * exp30; dkkt31 = dA31 * b3[:, None] * exp31; dkkt32 = dA32 * b3[:, None] * exp32

    S00 = (dkkt00 + tl.trans(dkkt00)).to(tl.bfloat16)
    S11 = (dkkt11 + tl.trans(dkkt11)).to(tl.bfloat16)
    S22 = (dkkt22 + tl.trans(dkkt22)).to(tl.bfloat16)
    S33 = (dkkt33 + tl.trans(dkkt33)).to(tl.bfloat16)

    dkA0 = tl.dot(S00, k0) + tl.dot(tl.trans(dkkt10).to(tl.bfloat16), k1) + tl.dot(tl.trans(dkkt20).to(tl.bfloat16), k2) + tl.dot(tl.trans(dkkt30).to(tl.bfloat16), k3)
    dkA1 = tl.dot(dkkt10.to(tl.bfloat16), k0) + tl.dot(S11, k1) + tl.dot(tl.trans(dkkt21).to(tl.bfloat16), k2) + tl.dot(tl.trans(dkkt31).to(tl.bfloat16), k3)
    dkA2 = tl.dot(dkkt20.to(tl.bfloat16), k0) + tl.dot(dkkt21.to(tl.bfloat16), k1) + tl.dot(S22, k2) + tl.dot(tl.trans(dkkt32).to(tl.bfloat16), k3)
    dkA3 = tl.dot(dkkt30.to(tl.bfloat16), k0) + tl.dot(dkkt31.to(tl.bfloat16), k1) + tl.dot(dkkt32.to(tl.bfloat16), k2) + tl.dot(S33, k3)

    dko0 = tl.load(dK_o_ptr + kd_base + (0 + r16)[:, None] * D + dk_offs[None, :]).to(tl.float32)
    dko1 = tl.load(dK_o_ptr + kd_base + (16 + r16)[:, None] * D + dk_offs[None, :]).to(tl.float32)
    dko2 = tl.load(dK_o_ptr + kd_base + (32 + r16)[:, None] * D + dk_offs[None, :]).to(tl.float32)
    dko3 = tl.load(dK_o_ptr + kd_base + (48 + r16)[:, None] * D + dk_offs[None, :]).to(tl.float32)

    tl.store(dK_ptr + kd_base + (0 + r16)[:, None] * D + dk_offs[None, :], (dko0 + ee0[:, None] * dke0 + (b0 * tl.exp(g0))[:, None] * ZW0 + dkA0).to(tl.bfloat16))
    tl.store(dK_ptr + kd_base + (16 + r16)[:, None] * D + dk_offs[None, :], (dko1 + ee1[:, None] * dke1 + (b1 * tl.exp(g1))[:, None] * ZW1 + dkA1).to(tl.bfloat16))
    tl.store(dK_ptr + kd_base + (32 + r16)[:, None] * D + dk_offs[None, :], (dko2 + ee2[:, None] * dke2 + (b2 * tl.exp(g2))[:, None] * ZW2 + dkA2).to(tl.bfloat16))
    tl.store(dK_ptr + kd_base + (48 + r16)[:, None] * D + dk_offs[None, :], (dko3 + ee3[:, None] * dke3 + (b3 * tl.exp(g3))[:, None] * ZW3 + dkA3).to(tl.bfloat16))


# ============================================================================
# 3. Autograd Wrappers: Single-GPU & 8-GPU Context-Parallel SSD/GLA & WY GDN
# ============================================================================


class _TritonChunkwiseSSDAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        B, H, S, D = q.shape
        Nc = S // 64
        BV = 64 if D == 64 else 32
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g_f32 = g.float().contiguous()

        gamma = torch.empty_like(g_f32)
        s_ckpt = torch.empty((B, H, Nc, D, D), device=q.device, dtype=torch.bfloat16)
        s_final = torch.empty((B, H, D, D), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)

        _ssd_gamma_cumsum_fwd_kernel[(Nc, B * H)](g_f32, gamma, S, num_warps=2)
        _ssd_state_scan_fwd_kernel[(D // BV, B * H)](
            k, v, gamma, s_final, s_ckpt, s_final, Nc, D=D, BV=BV, HAS_INIT=False, num_warps=4
        )
        _gdn_output_fwd_kernel[(Nc, D // BV, B * H)](
            q, k, v, s_ckpt, gamma, out, Nc, D=D, BV=BV, num_warps=4
        )
        ctx.save_for_backward(q, k, v, s_ckpt, gamma)
        ctx.BV = BV
        return out

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[Any, ...]:
        q, k, v, s_ckpt, gamma = ctx.saved_tensors
        BV = ctx.BV
        B, H, S, D = q.shape
        Nc = S // 64
        do = do.contiguous()

        ds_ckpt = torch.empty_like(s_ckpt)
        ds_final = torch.empty((B, H, D, D), device=q.device, dtype=torch.float32)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dg = torch.empty_like(gamma)

        _ssd_state_scan_bwd_kernel[(D // BV, B * H)](
            q, gamma, do, ds_final, ds_ckpt, ds_final, Nc, D=D, BV=BV, HAS_INIT=False, num_warps=4
        )
        _ssd_chunk_bwd_kernel[(Nc, B * H)](
            q, k, v, s_ckpt, ds_ckpt, gamma, do, dq, dk, dv, dg, Nc, D=D, BV=BV, num_warps=4
        )
        return dq, dk, dv, dg.to(q.dtype if g_is_bf16_sentinel(gamma) else torch.float32)


def g_is_bf16_sentinel(_t: torch.Tensor) -> bool:
    return False


def triton_chunkwise_ssd_gla(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    out = _TritonChunkwiseSSDAutograd.apply(q, k, v, g.float())
    return out


class _TritonChunkwiseGDNAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        B, H, S, D = q.shape
        Nc = S // 64
        BV = 64 if D == 64 else 32
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        beta_f32 = beta.float().contiguous()
        g_f32 = g.float().contiguous()

        u_wy = torch.empty_like(v)
        w_wy = torch.empty_like(k)
        u_corr = torch.empty_like(v)
        gamma = torch.empty_like(g_f32)
        s_ckpt = torch.empty((B, H, Nc, D, D), device=q.device, dtype=torch.bfloat16)
        s_final = torch.empty((B, H, D, D), device=q.device, dtype=torch.float32)
        out = torch.empty_like(v)

        _gdn_wy_prep_fwd_kernel[(Nc, B * H)](
            k, v, beta_f32, g_f32, u_wy, w_wy, gamma, S, D=D, USE_IEEE=False, num_warps=4
        )
        _gdn_state_scan_fwd_kernel[(D // BV, B * H)](
            k, u_wy, w_wy, gamma, s_final, s_ckpt, u_corr, s_final,
            Nc, D=D, DV_STRIDE=D, BV=BV, HAS_INIT=False, num_warps=4
        )
        _gdn_output_fwd_kernel[(Nc, D // BV, B * H)](
            q, k, u_corr, s_ckpt, gamma, out, Nc, D=D, BV=BV, num_warps=4
        )
        ctx.save_for_backward(q, k, v, beta_f32, gamma, u_wy, w_wy, u_corr, s_ckpt)
        ctx.BV = BV
        return out

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[Any, ...]:
        q, k, v, beta_f32, gamma, u_wy, w_wy, u_corr, s_ckpt = ctx.saved_tensors
        BV = ctx.BV
        B, H, S, D = q.shape
        Nc = S // 64
        do = do.contiguous()

        dq = torch.empty_like(q)
        dk_o = torch.empty_like(k)
        du_intra = torch.empty_like(v)
        dgamma_o = torch.empty_like(gamma)
        ds_ckpt = torch.empty_like(s_ckpt)
        du_wy = torch.empty_like(v)
        ds_final = torch.empty((B, H, D, D), device=q.device, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbeta = torch.empty_like(beta_f32)
        dg = torch.empty_like(gamma)

        _gdn_bwd_stage1_kernel[(Nc, B * H)](
            q, k, u_corr, s_ckpt, gamma, do, dq, dk_o, du_intra, dgamma_o, Nc, D=D, BV=BV, num_warps=4
        )
        _gdn_state_scan_bwd_kernel[(D // BV, B * H)](
            q, k, w_wy, gamma, do, du_intra, ds_final, ds_ckpt, du_wy, ds_final,
            Nc, D=D, BV=BV, HAS_INIT=False, num_warps=4
        )
        _gdn_wy_prep_bwd_kernel[(Nc, B * H)](
            k, v, beta_f32, gamma, u_wy, w_wy, u_corr, s_ckpt, ds_ckpt,
            du_wy, dk_o, dgamma_o, dk, dv, dbeta, dg, Nc, D=D, BV=BV, num_warps=4
        )
        return dq, dk, dv, dbeta, dg


def triton_chunkwise_gdn_wy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    return _TritonChunkwiseGDNAutograd.apply(q, k, v, beta.float(), g.float())


# --- 8-GPU Context-Parallel Operators (SSD Parallel Monoid, GDN P2P, GDN Parallel Monoid, Ulysses FA2) ---


class _CPSSDParallelMonoidAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        rank = dist.get_rank()
        w_sz = dist.get_world_size()
        B, H, S_loc, D = q.shape
        Nc = S_loc // 64
        BV = 64 if D == 64 else 32
        q, k, v, g_f32 = q.contiguous(), k.contiguous(), v.contiguous(), g.float().contiguous()

        gamma = torch.empty_like(g_f32)
        s_ckpt = torch.empty((B, H, Nc, D, D), device=q.device, dtype=torch.bfloat16)
        s_zero = torch.empty((B, H, D, D), device=q.device, dtype=torch.float32)

        _ssd_gamma_cumsum_fwd_kernel[(Nc, B * H)](g_f32, gamma, S_loc, num_warps=2)
        _ssd_state_scan_fwd_kernel[(D // BV, B * H)](
            k, v, gamma, s_zero, s_ckpt, s_zero, Nc, D=D, BV=BV, HAS_INIT=False, num_warps=4
        )

        # Total rank log-decay [B, H] = sum of chunk end gammas
        chunk_end_g = gamma.view(B, H, Nc, 64)[:, :, :, 63]
        rank_decay = chunk_end_g.sum(dim=-1)  # [B, H]

        # O(1)-size AllGather of (s_zero, rank_decay) across CP ranks!
        gather_s = torch.empty((w_sz, B, H, D, D), device=q.device, dtype=torch.float32)
        gather_d = torch.empty((w_sz, B, H), device=q.device, dtype=torch.float32)
        dist.all_gather_single(gather_s, s_zero)
        dist.all_gather_single(gather_d, rank_decay)

        if rank > 0:
            s_in = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
            for j in range(rank):
                s_in = torch.exp(gather_d[j])[..., None, None] * s_in + gather_s[j]
            # Add decayed S_in directly to S_ckpt in parallel without re-scanning!
            prefix_c = torch.cumsum(chunk_end_g, dim=-1) - chunk_end_g  # [B, H, Nc]
            s_ckpt = (s_ckpt.float() + torch.exp(prefix_c)[..., None, None] * s_in.unsqueeze(2)).bfloat16()

        out = torch.empty_like(v)
        _gdn_output_fwd_kernel[(Nc, D // BV, B * H)](
            q, k, v, s_ckpt, gamma, out, Nc, D=D, BV=BV, num_warps=4
        )
        ctx.save_for_backward(q, k, v, s_ckpt, gamma, chunk_end_g, gather_d)
        ctx.BV = BV
        return out

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[Any, ...]:
        q, k, v, s_ckpt, gamma, chunk_end_g, gather_d = ctx.saved_tensors
        BV = ctx.BV
        rank = dist.get_rank()
        w_sz = dist.get_world_size()
        B, H, S_loc, D = q.shape
        Nc = S_loc // 64
        do = do.contiguous()

        ds_ckpt = torch.empty_like(s_ckpt)
        ds_zero = torch.empty((B, H, D, D), device=q.device, dtype=torch.float32)
        _ssd_state_scan_bwd_kernel[(D // BV, B * H)](
            q, gamma, do, ds_zero, ds_ckpt, ds_zero, Nc, D=D, BV=BV, HAS_INIT=False, num_warps=4
        )

        gather_ds = torch.empty((w_sz, B, H, D, D), device=q.device, dtype=torch.float32)
        dist.all_gather_single(gather_ds, ds_zero)

        if rank < w_sz - 1:
            ds_out = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
            for j in range(w_sz - 1, rank, -1):
                ds_out = torch.exp(gather_d[j])[..., None, None] * ds_out + gather_ds[j]
            rev_c = torch.flip(chunk_end_g, dims=[-1])
            suf_c = torch.flip(torch.cumsum(rev_c, dim=-1) - rev_c, dims=[-1])
            ds_ckpt = (ds_ckpt.float() + torch.exp(suf_c)[..., None, None] * ds_out.unsqueeze(2)).bfloat16()

        dq, dk, dv, dg = (
            torch.empty_like(q),
            torch.empty_like(k),
            torch.empty_like(v),
            torch.empty_like(gamma),
        )
        _ssd_chunk_bwd_kernel[(Nc, B * H)](
            q, k, v, s_ckpt, ds_ckpt, gamma, do, dq, dk, dv, dg, Nc, D=D, BV=BV, num_warps=4
        )
        return dq, dk, dv, dg


class _CPGDNP2PStatePassAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        rank = dist.get_rank()
        w_sz = dist.get_world_size()
        B, H, S_loc, D = q.shape
        Nc = S_loc // 64
        BV = 64 if D == 64 else 32
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        beta_f32, g_f32 = beta.float().contiguous(), g.float().contiguous()

        u_wy, w_wy, u_corr = torch.empty_like(v), torch.empty_like(k), torch.empty_like(v)
        gamma = torch.empty_like(g_f32)
        s_ckpt = torch.empty((B, H, Nc, D, D), device=q.device, dtype=torch.bfloat16)
        s_in = torch.empty((B, H, D, D), device=q.device, dtype=torch.float32)
        s_out = torch.empty((B, H, D, D), device=q.device, dtype=torch.float32)

        # 1. Parallel WY prep on all 8 GPUs simultaneously!
        _gdn_wy_prep_fwd_kernel[(Nc, B * H)](
            k, v, beta_f32, g_f32, u_wy, w_wy, gamma, S_loc, D=D, USE_IEEE=False, num_warps=4
        )

        # 2. P2P state pass across ranks 0 -> 7
        if rank > 0:
            req_r = dist.P2POp(dist.irecv, s_in, rank - 1)
            for r_op in dist.batch_isend_irecv([req_r]):
                r_op.wait()
        _gdn_state_scan_fwd_kernel[(D // BV, B * H)](
            k, u_wy, w_wy, gamma, s_in, s_ckpt, u_corr, s_out,
            Nc, D=D, DV_STRIDE=D, BV=BV, HAS_INIT=(rank > 0), num_warps=4
        )
        if rank < w_sz - 1:
            req_s = dist.P2POp(dist.isend, s_out, rank + 1)
            for s_op in dist.batch_isend_irecv([req_s]):
                s_op.wait()

        # 3. Parallel chunk output on all 8 GPUs simultaneously!
        out = torch.empty_like(v)
        _gdn_output_fwd_kernel[(Nc, D // BV, B * H)](
            q, k, u_corr, s_ckpt, gamma, out, Nc, D=D, BV=BV, num_warps=4
        )
        ctx.save_for_backward(q, k, v, beta_f32, gamma, u_wy, w_wy, u_corr, s_ckpt)
        ctx.BV = BV
        return out

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[Any, ...]:
        q, k, v, beta_f32, gamma, u_wy, w_wy, u_corr, s_ckpt = ctx.saved_tensors
        BV = ctx.BV
        rank = dist.get_rank()
        w_sz = dist.get_world_size()
        B, H, S_loc, D = q.shape
        Nc = S_loc // 64
        do = do.contiguous()

        dq, dk_o, du_intra = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dgamma_o = torch.empty_like(gamma)
        ds_ckpt = torch.empty_like(s_ckpt)
        du_wy = torch.empty_like(v)
        ds_out = torch.empty((B, H, D, D), device=q.device, dtype=torch.float32)
        ds_in = torch.empty((B, H, D, D), device=q.device, dtype=torch.float32)

        # 1. Parallel Stage-1 backward on all 8 GPUs!
        _gdn_bwd_stage1_kernel[(Nc, B * H)](
            q, k, u_corr, s_ckpt, gamma, do, dq, dk_o, du_intra, dgamma_o, Nc, D=D, BV=BV, num_warps=4
        )

        # 2. Reverse P2P state pass 7 -> 0
        if rank < w_sz - 1:
            req_r = dist.P2POp(dist.irecv, ds_out, rank + 1)
            for r_op in dist.batch_isend_irecv([req_r]):
                r_op.wait()
        _gdn_state_scan_bwd_kernel[(D // BV, B * H)](
            q, k, w_wy, gamma, do, du_intra, ds_out, ds_ckpt, du_wy, ds_in,
            Nc, D=D, BV=BV, HAS_INIT=(rank < w_sz - 1), num_warps=4
        )
        if rank > 0:
            req_s = dist.P2POp(dist.isend, ds_in, rank - 1)
            for s_op in dist.batch_isend_irecv([req_s]):
                s_op.wait()

        # 3. Parallel WY prep backward on all 8 GPUs!
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        dbeta, dg = torch.empty_like(beta_f32), torch.empty_like(gamma)
        _gdn_wy_prep_bwd_kernel[(Nc, B * H)](
            k, v, beta_f32, gamma, u_wy, w_wy, u_corr, s_ckpt, ds_ckpt,
            du_wy, dk_o, dgamma_o, dk, dv, dbeta, dg, Nc, D=D, BV=BV, num_warps=4
        )
        return dq, dk, dv, dbeta, dg


class _CPGDNParallelMonoidAutograd(torch.autograd.Function):
    """Eliminates the O(CP) serial rank-to-rank scan chain via an augmented
    `[S_zero | P]` (`[B, H, D, 2D]`) parallel scan + O(1) `AllGather` + adjoint
    transposition `P_bwd == P^T`."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        rank = dist.get_rank()
        w_sz = dist.get_world_size()
        B, H, S_loc, D = q.shape
        Nc = S_loc // 64
        BV = 64 if D == 64 else 32
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        beta_f32, g_f32 = beta.float().contiguous(), g.float().contiguous()

        u_wy, w_wy = torch.empty_like(v), torch.empty_like(k)
        gamma = torch.empty_like(g_f32)
        _gdn_wy_prep_fwd_kernel[(Nc, B * H)](
            k, v, beta_f32, g_f32, u_wy, w_wy, gamma, S_loc, D=D, USE_IEEE=False, num_warps=4
        )

        # Augmented scan `[U_wy | 0]` starting from `[0 | I_D]` on all 8 GPUs simultaneously!
        u_aug = torch.zeros((B, H, S_loc, 2 * D), device=q.device, dtype=torch.bfloat16)
        u_aug[:, :, :, :D].copy_(u_wy)
        s_init_aug = torch.zeros((B, H, D, 2 * D), device=q.device, dtype=torch.float32)
        eye_d = torch.eye(D, device=q.device, dtype=torch.float32)
        s_init_aug[:, :, :, D:].copy_(eye_d)

        s_ckpt_aug = torch.empty((B, H, Nc, D, 2 * D), device=q.device, dtype=torch.bfloat16)
        u_corr_aug = torch.empty_like(u_aug)
        s_final_aug = torch.empty_like(s_init_aug)

        _gdn_state_scan_fwd_kernel[((2 * D) // BV, B * H)](
            k, u_aug, w_wy, gamma, s_init_aug, s_ckpt_aug, u_corr_aug, s_final_aug,
            Nc, D=D, DV_STRIDE=2 * D, BV=BV, HAS_INIT=True, num_warps=4
        )

        gather_aug = torch.empty((w_sz, B, H, D, 2 * D), device=q.device, dtype=torch.float32)
        dist.all_gather_single(gather_aug, s_final_aug)

        if rank == 0:
            s_ckpt = s_ckpt_aug[:, :, :, :, :D].contiguous()
            u_corr = u_corr_aug[:, :, :, :D].contiguous()
        else:
            # Note: in `u_wy - w_wy @ s`, `s` is right-multiplied on the left by `M_c`:
            # Wait! Check: `s_next = exp(g_end)*s + K_end^T @ (U_wy - W_wy @ s) = (exp(g_end)*I - K_end^T @ W_wy) @ s + K_end^T @ U_wy`!
            # So `P_j` (from `s_0 = I`, `U_wy = 0`) satisfies `S_out_j = P_j @ S_in_j + S_zero_j`!
            s_in = torch.zeros((B * H, D, D), device=q.device, dtype=torch.float32)
            for j in range(rank):
                s_zj = gather_aug[j, :, :, :, :D].reshape(B * H, D, D)
                p_j = gather_aug[j, :, :, :, D:].reshape(B * H, D, D)
                s_in = torch.bmm(p_j, s_in) + s_zj
            s_in_4d = s_in.view(B, H, D, D)
            s_ckpt = torch.empty((B, H, Nc, D, D), device=q.device, dtype=torch.bfloat16)
            u_corr = torch.empty_like(v)
            s_dummy = torch.empty((B, H, D, D), device=q.device, dtype=torch.float32)
            _gdn_state_scan_fwd_kernel[(D // BV, B * H)](
                k, u_wy, w_wy, gamma, s_in_4d, s_ckpt, u_corr, s_dummy,
                Nc, D=D, DV_STRIDE=D, BV=BV, HAS_INIT=True, num_warps=4
            )
        del s_ckpt_aug, u_corr_aug, u_aug

        out = torch.empty_like(v)
        _gdn_output_fwd_kernel[(Nc, D // BV, B * H)](
            q, k, u_corr, s_ckpt, gamma, out, Nc, D=D, BV=BV, num_warps=4
        )
        p_all = gather_aug[:, :, :, :, D:].contiguous()
        ctx.save_for_backward(q, k, v, beta_f32, gamma, u_wy, w_wy, u_corr, s_ckpt, p_all)
        ctx.BV = BV
        return out

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor) -> tuple[Any, ...]:
        q, k, v, beta_f32, gamma, u_wy, w_wy, u_corr, s_ckpt, p_all = ctx.saved_tensors
        BV = ctx.BV
        rank = dist.get_rank()
        w_sz = dist.get_world_size()
        B, H, S_loc, D = q.shape
        Nc = S_loc // 64
        do = do.contiguous()

        dq, dk_o, du_intra = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dgamma_o = torch.empty_like(gamma)
        _gdn_bwd_stage1_kernel[(Nc, B * H)](
            q, k, u_corr, s_ckpt, gamma, do, dq, dk_o, du_intra, dgamma_o, Nc, D=D, BV=BV, num_warps=4
        )

        # Parallel zero-init reverse scan on all 8 GPUs simultaneously!
        ds_ckpt = torch.empty_like(s_ckpt)
        du_wy = torch.empty_like(v)
        ds_zero = torch.empty((B, H, D, D), device=q.device, dtype=torch.float32)
        _gdn_state_scan_bwd_kernel[(D // BV, B * H)](
            q, k, w_wy, gamma, do, du_intra, ds_zero, ds_ckpt, du_wy, ds_zero,
            Nc, D=D, BV=BV, HAS_INIT=False, num_warps=4
        )

        gather_ds = torch.empty((w_sz, B, H, D, D), device=q.device, dtype=torch.float32)
        dist.all_gather_single(gather_ds, ds_zero)

        if rank < w_sz - 1:
            ds_out = torch.zeros((B * H, D, D), device=q.device, dtype=torch.float32)
            for j in range(w_sz - 1, rank, -1):
                p_jt = p_all[j].reshape(B * H, D, D).transpose(1, 2)
                ds_zj = gather_ds[j].reshape(B * H, D, D)
                ds_out = torch.bmm(p_jt, ds_out) + ds_zj
            ds_out_4d = ds_out.view(B, H, D, D)
            ds_dummy = torch.empty_like(ds_zero)
            _gdn_state_scan_bwd_kernel[(D // BV, B * H)](
                q, k, w_wy, gamma, do, du_intra, ds_out_4d, ds_ckpt, du_wy, ds_dummy,
                Nc, D=D, BV=BV, HAS_INIT=True, num_warps=4
            )

        dk, dv = torch.empty_like(k), torch.empty_like(v)
        dbeta, dg = torch.empty_like(beta_f32), torch.empty_like(gamma)
        _gdn_wy_prep_bwd_kernel[(Nc, B * H)](
            k, v, beta_f32, gamma, u_wy, w_wy, u_corr, s_ckpt, ds_ckpt,
            du_wy, dk_o, dgamma_o, dk, dv, dbeta, dg, Nc, D=D, BV=BV, num_warps=4
        )
        return dq, dk, dv, dbeta, dg


class _DiffAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        xc = x.contiguous()
        out = torch.empty_like(xc)
        dist.all_to_all_single(out, xc)
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> torch.Tensor:
        gc = grad_out.contiguous()
        grad_in = torch.empty_like(gc)
        dist.all_to_all_single(grad_in, gc)
        return grad_in


def cp_ulysses_flash_attn2_gqa(
    q_bshd: torch.Tensor, k_bshd: torch.Tensor, v_bshd: torch.Tensor
) -> torch.Tensor:
    """GQA-native DeepSpeed-Ulysses + FlashAttention-2 (`q, k, v` in `[B, S_loc, H, D]`,
    `H_q` and `H_kv` divisible by `world_size`)."""
    w_sz = dist.get_world_size()
    B, S_loc, Hq, D = q_bshd.shape
    Hkv = k_bshd.shape[2]

    def _s2h(t: torch.Tensor, h_tot: int) -> torch.Tensor:
        h_loc = h_tot // w_sz
        # [B, S_loc, w_sz, h_loc, D] -> [w_sz, B, S_loc, h_loc, D] -> all_to_all -> [B, w_sz * S_loc, h_loc, D]
        send = t.view(B, S_loc, w_sz, h_loc, D).permute(2, 0, 1, 3, 4).contiguous()
        recv = _DiffAllToAll.apply(send)
        return recv.permute(1, 0, 2, 3, 4).contiguous().view(B, w_sz * S_loc, h_loc, D)

    def _h2s(t: torch.Tensor, h_tot: int) -> torch.Tensor:
        h_loc = h_tot // w_sz
        send = t.view(B, w_sz, S_loc, h_loc, D).permute(1, 0, 2, 3, 4).contiguous()
        recv = _DiffAllToAll.apply(send)
        return recv.permute(1, 2, 0, 3, 4).contiguous().view(B, S_loc, h_tot, D)

    qf = _s2h(q_bshd, Hq)
    kf = _s2h(k_bshd, Hkv)
    vf = _s2h(v_bshd, Hkv)
    of = flash_attn_func(qf, kf, vf, causal=True)
    return _h2s(of, Hq)


# ============================================================================
# 4. PyTorch Baselines & Telemetry Helpers
# ============================================================================


def pytorch_recurrent_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    B, H, S, D = q.shape
    state = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
    outs = []
    gf = g.float()
    bf = beta.float()
    qf, kf, vf = q.float(), k.float(), v.float()
    for t in range(S):
        st = torch.exp(gf[:, :, t]).unsqueeze(-1).unsqueeze(-1) * state
        kt = kf[:, :, t]
        pred = torch.einsum("bhkd,bhk->bhd", st, kt)
        ut = bf[:, :, t].unsqueeze(-1) * (vf[:, :, t] - pred)
        state = st + torch.einsum("bhk,bhd->bhkd", kt, ut)
        outs.append(torch.einsum("bhkd,bhk->bhd", state, qf[:, :, t]).to(q.dtype))
    return torch.stack(outs, dim=2)


def pytorch_chunkwise_wy_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    B, H, S, D = q.shape
    C = 64
    Nc = S // C
    qc = q.view(B, H, Nc, C, D).float()
    kc = k.view(B, H, Nc, C, D).float()
    vc = v.view(B, H, Nc, C, D).float()
    bc = beta.view(B, H, Nc, C).float()
    gc = g.view(B, H, Nc, C).float()
    gamma = torch.cumsum(gc, dim=-1)
    exp_g = torch.exp(gamma)

    idx = torch.arange(C, device=q.device)
    st_m = idx[:, None] > idx[None, :]
    ca_m = idx[:, None] >= idx[None, :]
    diff = gamma.unsqueeze(-1) - gamma.unsqueeze(-2)
    exp_st = torch.where(st_m, torch.exp(torch.where(st_m, diff, 0.0)), 0.0)
    exp_ca = torch.where(ca_m, torch.exp(torch.where(ca_m, diff, 0.0)), 0.0)

    A = bc.unsqueeze(-1) * exp_st * torch.matmul(kc, kc.transpose(-1, -2))
    eye = torch.eye(C, device=q.device, dtype=torch.float32)
    T = torch.linalg.solve_triangular(
        eye + A, eye.expand_as(A).clone(), upper=False, unitriangular=True
    )
    U_wy = torch.matmul(T, bc.unsqueeze(-1) * vc)
    W_wy = torch.matmul(T, (bc * exp_g).unsqueeze(-1) * kc)

    state = torch.zeros((B, H, D, D), device=q.device, dtype=torch.float32)
    outs = []
    for c in range(Nc):
        uc = U_wy[:, :, c] - torch.matmul(W_wy[:, :, c], state)
        q_dec = exp_g[:, :, c].unsqueeze(-1) * qc[:, :, c]
        m_qk = exp_ca[:, :, c] * torch.matmul(qc[:, :, c], kc[:, :, c].transpose(-1, -2))
        outs.append(torch.matmul(q_dec, state) + torch.matmul(m_qk, uc))
        g_end = gamma[:, :, c, -1:]
        k_end = torch.exp(g_end - gamma[:, :, c]).unsqueeze(-1) * kc[:, :, c]
        state = torch.exp(g_end).unsqueeze(-1) * state + torch.matmul(
            k_end.transpose(-1, -2), uc
        )
    return torch.cat(outs, dim=2).to(q.dtype)


def sample_gpu_clocks(device_idx: int) -> dict[str, float]:
    if not HAS_PYNVML:
        return {"sm_clock_mhz": 0.0, "power_w": 0.0}
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        return {
            "sm_clock_mhz": float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)),
            "power_w": float(pynvml.nvmlDeviceGetPowerUsage(h)) / 1000.0,
        }
    except Exception:
        return {"sm_clock_mhz": 0.0, "power_w": 0.0}


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-8)).item()
    )


def time_fn_with_peak_mem(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    base_mem = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    inc_mib = max(0.0, (torch.cuda.max_memory_allocated(device) - base_mem) / (1024.0**2))

    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iters):
        s_ev.record()
        fn()
        e_ev.record()
        torch.cuda.synchronize(device)
        samples.append(float(s_ev.elapsed_time(e_ev)))
    nv = sample_gpu_clocks(device.index)
    ss = sorted(samples)
    q25 = ss[max(0, (len(ss) - 1) // 4)]
    q75 = ss[min(len(ss) - 1, (3 * (len(ss) - 1)) // 4)]
    return {
        "median_ms": round(statistics.median(ss), 4),
        "iqr_ms": round(q75 - q25, 4),
        "peak_inc_mib": round(inc_mib, 2),
        "sm_clock_after_mhz": nv["sm_clock_mhz"],
        "power_after_w": round(nv["power_w"], 2),
    }


# ============================================================================
# 5. Suite 0: Overflow Audit, FP64 Chain-Rule Refinement, & WY Kernel Audit
# ============================================================================


def run_suite_0_audits(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    # 1. Naive cumulative exp(+gamma) * exp(-gamma) overflow audit vs overflow-free chunkwise
    gen = torch.Generator(device=device)
    gen.manual_seed(11900)
    s_ov = 1024
    q_ov = F.normalize(torch.randn((1, 4, s_ov, 64), generator=gen, device=device, dtype=torch.bfloat16), dim=-1)
    k_ov = F.normalize(torch.randn((1, 4, s_ov, 64), generator=gen, device=device, dtype=torch.bfloat16), dim=-1)
    v_ov = torch.randn((1, 4, s_ov, 64), generator=gen, device=device, dtype=torch.bfloat16)
    g_ov = -F.softplus(torch.randn((1, 4, s_ov), generator=gen, device=device, dtype=torch.float32) - 0.5)

    gam_global = torch.cumsum(g_ov, dim=-1)
    # Naive Q_tilde = Q * exp(gam_global), K_tilde = K * exp(-gam_global) -> exp(-gam_global) overflows FP32 at -gam > 88.7!
    k_tilde = k_ov.float() * torch.exp(-gam_global).unsqueeze(-1)
    q_tilde = q_ov.float() * torch.exp(gam_global).unsqueeze(-1)
    ca = torch.tril(torch.ones((s_ov, s_ov), device=device, dtype=torch.bool))
    naive_scores = torch.where(ca, torch.matmul(q_tilde, k_tilde.transpose(-1, -2)), 0.0)
    naive_out = torch.matmul(naive_scores, v_ov.float())
    naive_has_nan_or_inf = bool((~torch.isfinite(naive_out)).any().item())
    min_global_gamma = float(gam_global.min().item())

    tri_ssd_out = triton_chunkwise_ssd_gla(q_ov, k_ov, v_ov, g_ov)
    triton_ssd_all_finite = bool(torch.isfinite(tri_ssd_out).all().item())

    # 2. FP64 Central-Difference Chain-Rule Verification (h = 1e-4 vs h/2 = 5e-5 -> ~4.00x!)
    B_fd, H_fd, S_fd, D_fd = 1, 2, 128, 64
    gen_fd = torch.Generator(device=device)
    gen_fd.manual_seed(11905)
    q64 = F.normalize(torch.randn((B_fd, H_fd, S_fd, D_fd), generator=gen_fd, device=device, dtype=torch.float64), dim=-1)
    k64 = F.normalize(torch.randn((B_fd, H_fd, S_fd, D_fd), generator=gen_fd, device=device, dtype=torch.float64), dim=-1)
    v64 = torch.randn((B_fd, H_fd, S_fd, D_fd), generator=gen_fd, device=device, dtype=torch.float64)
    b64 = torch.sigmoid(torch.randn((B_fd, H_fd, S_fd), generator=gen_fd, device=device, dtype=torch.float64)) * 1.5
    g64 = -F.softplus(torch.randn((B_fd, H_fd, S_fd), generator=gen_fd, device=device, dtype=torch.float64) - 1.5)
    do64 = torch.randn((B_fd, H_fd, S_fd, D_fd), generator=gen_fd, device=device, dtype=torch.float64)

    def _loss64(qq: torch.Tensor, kk: torch.Tensor, vv: torch.Tensor, bb: torch.Tensor, gg: torch.Tensor) -> torch.Tensor:
        st = torch.zeros((B_fd, H_fd, D_fd, D_fd), device=device, dtype=torch.float64)
        outs = []
        for t in range(S_fd):
            st = torch.exp(gg[:, :, t]).unsqueeze(-1).unsqueeze(-1) * st
            ut = bb[:, :, t].unsqueeze(-1) * (vv[:, :, t] - torch.einsum("bhkd,bhk->bhd", st, kk[:, :, t]))
            st = st + torch.einsum("bhk,bhd->bhkd", kk[:, :, t], ut)
            outs.append(torch.einsum("bhkd,bhk->bhd", st, qq[:, :, t]))
        o_cat = torch.stack(outs, dim=2)
        return (torch.sin(o_cat) * do64).sum()

    # Exact analytical chunkwise WY gradient in FP64
    qr, kr, vr, br, gr = [x.clone().requires_grad_(True) for x in (q64, k64, v64, b64, g64)]
    # Use chunkwise WY in FP64 to prove equivalence
    Nc_fd = S_fd // 64
    qc, kc, vc = [x.view(B_fd, H_fd, Nc_fd, 64, D_fd) for x in (qr, kr, vr)]
    bc, gc = br.view(B_fd, H_fd, Nc_fd, 64), gr.view(B_fd, H_fd, Nc_fd, 64)
    gam_fd = torch.cumsum(gc, dim=-1)
    idx64 = torch.arange(64, device=device)
    st_m = idx64[:, None] > idx64[None, :]
    ca_m = idx64[:, None] >= idx64[None, :]
    diff_fd = gam_fd.unsqueeze(-1) - gam_fd.unsqueeze(-2)
    exp_st = torch.where(st_m, torch.exp(torch.where(st_m, diff_fd, 0.0)), 0.0)
    exp_ca = torch.where(ca_m, torch.exp(torch.where(ca_m, diff_fd, 0.0)), 0.0)
    A_fd = bc.unsqueeze(-1) * exp_st * torch.matmul(kc, kc.transpose(-1, -2))
    eye64 = torch.eye(64, device=device, dtype=torch.float64)
    T_fd = torch.linalg.solve_triangular(eye64 + A_fd, eye64.expand_as(A_fd).clone(), upper=False, unitriangular=True)
    U_fd = torch.matmul(T_fd, bc.unsqueeze(-1) * vc)
    W_fd = torch.matmul(T_fd, (bc * torch.exp(gam_fd)).unsqueeze(-1) * kc)
    s_c = torch.zeros((B_fd, H_fd, D_fd, D_fd), device=device, dtype=torch.float64)
    o_chunks = []
    for c_i in range(Nc_fd):
        uc = U_fd[:, :, c_i] - torch.matmul(W_fd[:, :, c_i], s_c)
        o_chunks.append(
            torch.matmul(torch.exp(gam_fd[:, :, c_i]).unsqueeze(-1) * qc[:, :, c_i], s_c)
            + torch.matmul(exp_ca[:, :, c_i] * torch.matmul(qc[:, :, c_i], kc[:, :, c_i].transpose(-1, -2)), uc)
        )
        ge = gam_fd[:, :, c_i, -1:]
        ke = torch.exp(ge - gam_fd[:, :, c_i]).unsqueeze(-1) * kc[:, :, c_i]
        s_c = torch.exp(ge).unsqueeze(-1) * s_c + torch.matmul(ke.transpose(-1, -2), uc)
    loss_wy = (torch.sin(torch.cat(o_chunks, dim=2)) * do64).sum()
    loss_wy.backward()
    ana_grads = [qr.grad, kr.grad, vr.grad, br.grad, gr.grad]

    # Also verify exact FP64 equivalence between chunkwise WY autograd and recurrent autograd
    q_rec64, k_rec64, v_rec64, b_rec64, g_rec64 = [
        x.clone().requires_grad_(True) for x in (q64, k64, v64, b64, g64)
    ]
    loss_rec64 = _loss64(q_rec64, k_rec64, v_rec64, b_rec64, g_rec64)
    loss_rec64.backward()
    rec64_grads = [q_rec64.grad, k_rec64.grad, v_rec64.grad, b_rec64.grad, g_rec64.grad]
    wy_vs_recurrent_fp64_max_rel_l2 = max(
        float(((ga - grc).norm() / grc.norm().clamp(min=1e-18)).item())
        for ga, grc in zip(ana_grads, rec64_grads)
        if ga is not None and grc is not None
    )

    fd_ratios: dict[str, float] = {}
    base_inputs = [q64, k64, v64, b64, g64]
    for idx_in, (in_name, g_ana) in enumerate(
        zip(["q", "k", "v", "beta", "g"], ana_grads)
    ):
        assert g_ana is not None
        direction = torch.randn_like(base_inputs[idx_in], generator=gen_fd)
        # Unit-RMS perturbation keeps per-element step h * d_i ~ O(h) independent of tensor size!
        direction = direction / direction.pow(2).mean().sqrt().clamp(min=1e-18)
        dir_deriv = float((g_ana * direction).sum().item())
        errs = []
        for h_step in (5e-3, 2.5e-3):
            inp_p = [x.clone() for x in base_inputs]
            inp_m = [x.clone() for x in base_inputs]
            inp_p[idx_in] = inp_p[idx_in] + h_step * direction
            inp_m[idx_in] = inp_m[idx_in] - h_step * direction
            with torch.no_grad():
                fd_est = float((_loss64(*inp_p) - _loss64(*inp_m)).item()) / (2.0 * h_step)
            errs.append(abs(fd_est - dir_deriv))
        fd_ratios[in_name] = round(errs[0] / max(errs[1], 1e-18), 4)

    # 3. BF16 Triton Chunkwise SSD & WY GDN Parity vs Exact FP32 Recurrence (D = 128, S = 512)
    B_p, H_p, S_p, D_p = 2, 4, 512, 128
    qp = F.normalize(torch.randn((B_p, H_p, S_p, D_p), generator=gen, device=device, dtype=torch.bfloat16), dim=-1)
    kp = F.normalize(torch.randn((B_p, H_p, S_p, D_p), generator=gen, device=device, dtype=torch.bfloat16), dim=-1)
    vp = torch.randn((B_p, H_p, S_p, D_p), generator=gen, device=device, dtype=torch.bfloat16)
    bp = torch.sigmoid(torch.randn((B_p, H_p, S_p), generator=gen, device=device, dtype=torch.float32)) * 1.5
    gp = -F.softplus(torch.randn((B_p, H_p, S_p), generator=gen, device=device, dtype=torch.float32) - 2.0)
    dop = torch.randn_like(vp)

    # Reference GDN
    q_r, k_r, v_r = qp.float().requires_grad_(True), kp.float().requires_grad_(True), vp.float().requires_grad_(True)
    b_r, g_r = bp.clone().requires_grad_(True), gp.clone().requires_grad_(True)
    o_r = pytorch_recurrent_gdn(q_r, k_r, v_r, b_r, g_r)
    (o_r * dop.float()).sum().backward()

    q_t, k_t, v_t = qp.clone().requires_grad_(True), kp.clone().requires_grad_(True), vp.clone().requires_grad_(True)
    b_t, g_t = bp.clone().requires_grad_(True), gp.clone().requires_grad_(True)
    o_t = triton_chunkwise_gdn_wy(q_t, k_t, v_t, b_t, g_t)
    (o_t.float() * dop.float()).sum().backward()

    gdn_parity = {
        "out_rel_l2": round(rel_l2(o_t, o_r), 6),
        "dq_rel_l2": round(rel_l2(q_t.grad, q_r.grad), 6),
        "dk_rel_l2": round(rel_l2(k_t.grad, k_r.grad), 6),
        "dv_rel_l2": round(rel_l2(v_t.grad, v_r.grad), 6),
        "dbeta_rel_l2": round(rel_l2(b_t.grad, b_r.grad), 6),
        "dg_rel_l2": round(rel_l2(g_t.grad, g_r.grad), 6),
    }

    # Reference SSD (`beta = 1` without `-k k^T` erasure -> `S_t = exp(g_t) S_{t-1} + k_t v_t^T`)
    q_sr, k_sr, v_sr, g_sr = qp.float().requires_grad_(True), kp.float().requires_grad_(True), vp.float().requires_grad_(True), gp.clone().requires_grad_(True)
    st_s = torch.zeros((B_p, H_p, D_p, D_p), device=device, dtype=torch.float32)
    o_s_list = []
    for t in range(S_p):
        st_s = torch.exp(g_sr[:, :, t]).unsqueeze(-1).unsqueeze(-1) * st_s + torch.einsum("bhk,bhd->bhkd", k_sr[:, :, t], v_sr[:, :, t])
        o_s_list.append(torch.einsum("bhkd,bhk->bhd", st_s, q_sr[:, :, t]))
    o_s_ref = torch.stack(o_s_list, dim=2)
    (o_s_ref * dop.float()).sum().backward()

    q_st, k_st, v_st, g_st = qp.clone().requires_grad_(True), kp.clone().requires_grad_(True), vp.clone().requires_grad_(True), gp.clone().requires_grad_(True)
    o_st = triton_chunkwise_ssd_gla(q_st, k_st, v_st, g_st)
    (o_st.float() * dop.float()).sum().backward()

    ssd_parity = {
        "out_rel_l2": round(rel_l2(o_st, o_s_ref), 6),
        "dq_rel_l2": round(rel_l2(q_st.grad, q_sr.grad), 6),
        "dk_rel_l2": round(rel_l2(k_st.grad, k_sr.grad), 6),
        "dv_rel_l2": round(rel_l2(v_st.grad, v_sr.grad), 6),
        "dg_rel_l2": round(rel_l2(g_st.grad, g_sr.grad), 6),
    }

    # 4. Isolated WY Preparation Audit: PyTorch `solve_triangular` vs Triton IEEE vs Triton BF16 Tensor-Core
    B_w, H_w, S_w, D_w = (1, 8, 2048, 128) if is_smoke else (2, 16, 8192, 128)
    Nc_w = S_w // 64
    warmup_w = 2 if is_smoke else 5
    iters_w = 5 if is_smoke else 15

    kw = F.normalize(torch.randn((B_w, H_w, S_w, D_w), generator=gen, device=device, dtype=torch.bfloat16), dim=-1).contiguous()
    vw = torch.randn((B_w, H_w, S_w, D_w), generator=gen, device=device, dtype=torch.bfloat16).contiguous()
    bw = (torch.sigmoid(torch.randn((B_w, H_w, S_w), generator=gen, device=device, dtype=torch.float32)) * 1.5).contiguous()
    gw = (-F.softplus(torch.randn((B_w, H_w, S_w), generator=gen, device=device, dtype=torch.float32) - 2.0)).contiguous()

    def _pt_wy_prep() -> None:
        kc_w = kw.view(B_w, H_w, Nc_w, 64, D_w).float()
        vc_w = vw.view(B_w, H_w, Nc_w, 64, D_w).float()
        bc_w = bw.view(B_w, H_w, Nc_w, 64)
        gc_w = gw.view(B_w, H_w, Nc_w, 64)
        gam_w = torch.cumsum(gc_w, dim=-1)
        idx = torch.arange(64, device=device)
        st = idx[:, None] > idx[None, :]
        exp_s = torch.where(st, torch.exp(torch.where(st, gam_w.unsqueeze(-1) - gam_w.unsqueeze(-2), 0.0)), 0.0)
        A_w = bc_w.unsqueeze(-1) * exp_s * torch.matmul(kc_w, kc_w.transpose(-1, -2))
        ey = torch.eye(64, device=device)
        T_w = torch.linalg.solve_triangular(ey + A_w, ey.expand_as(A_w).clone(), upper=False, unitriangular=True)
        _ = torch.matmul(T_w, bc_w.unsqueeze(-1) * vc_w).bfloat16()
        _ = torch.matmul(T_w, (bc_w * torch.exp(gam_w)).unsqueeze(-1) * kc_w).bfloat16()

    u_tmp, w_tmp, g_tmp = torch.empty_like(vw), torch.empty_like(kw), torch.empty_like(gw)

    def _tri_ieee_wy() -> None:
        _gdn_wy_prep_fwd_kernel[(Nc_w, B_w * H_w)](
            kw, vw, bw, gw, u_tmp, w_tmp, g_tmp, S_w, D=D_w, USE_IEEE=True, num_warps=4
        )

    def _tri_tc_wy() -> None:
        _gdn_wy_prep_fwd_kernel[(Nc_w, B_w * H_w)](
            kw, vw, bw, gw, u_tmp, w_tmp, g_tmp, S_w, D=D_w, USE_IEEE=False, num_warps=4
        )

    t_pt_wy = time_fn_with_peak_mem(_pt_wy_prep, warmup=warmup_w, iters=iters_w, device=device)
    t_ieee_wy = time_fn_with_peak_mem(_tri_ieee_wy, warmup=warmup_w, iters=iters_w, device=device)
    t_tc_wy = time_fn_with_peak_mem(_tri_tc_wy, warmup=warmup_w, iters=iters_w, device=device)

    return {
        "overflow_audit": {
            "seq_len": s_ov,
            "min_global_cumulative_log_gate": round(min_global_gamma, 2),
            "naive_global_cumsum_has_nan_or_inf": naive_has_nan_or_inf,
            "triton_chunkwise_ssd_all_finite": triton_ssd_all_finite,
        },
        "fp64_chain_rule_refinement": {
            "wy_vs_recurrent_fp64_max_rel_l2": float(f"{wy_vs_recurrent_fp64_max_rel_l2:.3e}"),
            "ratios_h_vs_half_h": fd_ratios,
            "min_refinement_ratio": round(min(fd_ratios.values()), 4),
        },
        "triton_bf16_gdn_parity": gdn_parity,
        "triton_bf16_ssd_parity": ssd_parity,
        "wy_prep_kernel_audit": {
            "num_chunks": B_w * H_w * Nc_w,
            "pytorch_solve_triangular_wy": t_pt_wy,
            "triton_nilpotent_ieee_cuda_core": t_ieee_wy,
            "triton_nilpotent_bf16_tensor_core": t_tc_wy,
            "tensor_core_speedup_vs_pytorch_solve_triangular": round(
                t_pt_wy["median_ms"] / max(t_tc_wy["median_ms"], 1e-6), 4
            ),
            "tensor_core_speedup_vs_ieee_cuda_core": round(
                t_ieee_wy["median_ms"] / max(t_tc_wy["median_ms"], 1e-6), 4
            ),
        },
    }


# ============================================================================
# 6. Suite A: 1-GPU Sequence-Length Scaling (S = 2,048 .. 65,536, Fwd + Bwd)
# ============================================================================


def run_suite_a_seq_scaling(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    seq_lens = [2048, 8192] if is_smoke else [2048, 4096, 8192, 16384, 32768, 65536]
    warmup = 2 if is_smoke else 4
    iters = 4 if is_smoke else 12
    B, H, H_kv, D = 1, 16, 4, 128

    compiled_wy = torch.compile(pytorch_chunkwise_wy_gdn, dynamic=False)
    results: dict[str, Any] = {}

    for S in seq_lens:
        gen = torch.Generator(device=device)
        gen.manual_seed(11920 + S)
        q = F.normalize(torch.randn((B, H, S, D), generator=gen, device=device, dtype=torch.bfloat16), dim=-1)
        k = F.normalize(torch.randn((B, H, S, D), generator=gen, device=device, dtype=torch.bfloat16), dim=-1)
        v = torch.randn((B, H, S, D), generator=gen, device=device, dtype=torch.bfloat16)
        beta = torch.sigmoid(torch.randn((B, H, S), generator=gen, device=device, dtype=torch.float32)) * 1.5
        g = -F.softplus(torch.randn((B, H, S), generator=gen, device=device, dtype=torch.float32) - 2.0)
        do = torch.randn_like(v)

        modes_out: dict[str, Any] = {}

        # 1. Token-by-token PyTorch recurrent GDN (only at S <= 2048 so ~18,000 launches don't burn minutes!)
        if S <= 2048:
            qr, kr, vr = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
            br, gr = beta.clone().requires_grad_(True), g.clone().requires_grad_(True)

            def _run_rec() -> None:
                for t_x in (qr, kr, vr, br, gr):
                    t_x.grad = None
                o = pytorch_recurrent_gdn(qr, kr, vr, br, gr)
                torch.autograd.backward(o, do)

            with profile(activities=[ProfilerActivity.CUDA]) as prof_rec:
                _run_rec()
                torch.cuda.synchronize(device)
            rec_launches = sum(
                int(e.count) for e in prof_rec.key_averages() if e.self_device_time_total > 0
            )
            tm_rec = time_fn_with_peak_mem(_run_rec, warmup=1, iters=3, device=device)
            modes_out["pytorch_recurrent_step_eager"] = {
                **tm_rec,
                "cuda_launches": rec_launches,
            }
            del qr, kr, vr, br, gr
            gc.collect()
            torch.cuda.empty_cache()

        # 2. PyTorch Chunkwise WY (`solve_triangular` + chunk loop) eager & compiled
        # Note: TorchDynamo unrolls `for c in range(Nc)` into `2 * Nc` sequential AOTAutograd
        # stages (~120s compile at S=2048 / Nc=32, >300s at S=8192 / Nc=128!), whereas Triton's
        # `for c in range(0, Nc)` compiles once for any runtime `Nc`. Cap compiled WY at S <= 2048!
        if S <= 16384:
            for m_key, fn_impl in (
                ("pytorch_chunkwise_wy_eager", pytorch_chunkwise_wy_gdn),
                (
                    "pytorch_chunkwise_wy_compiled",
                    compiled_wy if (S <= 2048 and not is_smoke) else None,
                ),
            ):
                if fn_impl is None:
                    continue
                qw, kw, vw = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
                bw, gw = beta.clone().requires_grad_(True), g.clone().requires_grad_(True)

                def _run_wy() -> None:
                    for t_x in (qw, kw, vw, bw, gw):
                        t_x.grad = None
                    o = fn_impl(qw, kw, vw, bw, gw)
                    torch.autograd.backward(o, do)

                t_comp0 = time.perf_counter()
                _run_wy()
                torch.cuda.synchronize(device)
                first_call_s = time.perf_counter() - t_comp0

                tm_wy = time_fn_with_peak_mem(
                    _run_wy, warmup=min(2, warmup), iters=min(5, iters), device=device
                )
                if m_key == "pytorch_chunkwise_wy_compiled":
                    tm_wy["first_call_compile_s"] = round(first_call_s, 2)
                modes_out[m_key] = tm_wy
                del qw, kw, vw, bw, gw
                gc.collect()
                torch.cuda.empty_cache()

        # 3. Triton Chunkwise SSD / GLA (Fwd + Bwd)
        qs, ks, vs, gs = (
            q.clone().requires_grad_(True),
            k.clone().requires_grad_(True),
            v.clone().requires_grad_(True),
            g.clone().requires_grad_(True),
        )

        def _run_ssd() -> None:
            for t_x in (qs, ks, vs, gs):
                t_x.grad = None
            o = triton_chunkwise_ssd_gla(qs, ks, vs, gs)
            torch.autograd.backward(o, do)

        modes_out["triton_chunkwise_ssd_gla"] = time_fn_with_peak_mem(
            _run_ssd, warmup=warmup, iters=iters, device=device
        )
        del qs, ks, vs, gs
        gc.collect()
        torch.cuda.empty_cache()

        # 4. Triton Chunkwise WY Gated DeltaNet (Fwd + Bwd)
        qg, kg, vg = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
        bg, gg = beta.clone().requires_grad_(True), g.clone().requires_grad_(True)

        def _run_gdn() -> None:
            for t_x in (qg, kg, vg, bg, gg):
                t_x.grad = None
            o = triton_chunkwise_gdn_wy(qg, kg, vg, bg, gg)
            torch.autograd.backward(o, do)

        if S == 2048:
            with profile(activities=[ProfilerActivity.CUDA]) as prof_gdn:
                _run_gdn()
                torch.cuda.synchronize(device)
            gdn_launches = sum(
                int(e.count) for e in prof_gdn.key_averages() if e.self_device_time_total > 0
            )
        else:
            gdn_launches = 0

        tm_gdn = time_fn_with_peak_mem(_run_gdn, warmup=warmup, iters=iters, device=device)
        if gdn_launches > 0:
            tm_gdn["cuda_launches"] = gdn_launches
        modes_out["triton_chunkwise_gdn_wy"] = tm_gdn
        del qg, kg, vg, bg, gg
        gc.collect()
        torch.cuda.empty_cache()

        # 5. FlashAttention-2 GQA (16Q / 4KV, D=128) Fwd + Bwd
        q_fa = q.transpose(1, 2).contiguous().detach().requires_grad_(True)
        k_fa = k[:, :H_kv].transpose(1, 2).contiguous().detach().requires_grad_(True)
        v_fa = v[:, :H_kv].transpose(1, 2).contiguous().detach().requires_grad_(True)
        do_fa = do.transpose(1, 2).contiguous()

        def _run_fa2() -> None:
            q_fa.grad = k_fa.grad = v_fa.grad = None
            o_fa = flash_attn_func(q_fa, k_fa, v_fa, causal=True)
            torch.autograd.backward(o_fa, do_fa)

        modes_out["flash_attn2_gqa_16q4kv"] = time_fn_with_peak_mem(
            _run_fa2, warmup=warmup, iters=iters, device=device
        )
        del q_fa, k_fa, v_fa, do_fa
        gc.collect()
        torch.cuda.empty_cache()

        results[str(S)] = {
            "batch": B,
            "heads": H,
            "head_dim": D,
            "seq_len": S,
            "modes": modes_out,
        }
    return results


# ============================================================================
# 7. Suite B: Associative Overwrite Recall (MQAR) & 3:1 Hybrid 4-Layer Stack
# ============================================================================


class HybridMixerLayer(nn.Module):
    def __init__(self, hidden: int, inter_dim: int, heads: int, mixer_kind: str) -> None:
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.mixer_kind = mixer_kind  # "gdn", "ssd", or "fa2_gqa"
        self.kv_heads = max(1, heads // 4)

        self.norm1 = nn.RMSNorm(hidden, eps=1e-5)
        self.wq = nn.Linear(hidden, hidden, bias=False)
        if mixer_kind == "fa2_gqa":
            self.wk = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
            self.wv = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        else:
            self.wk = nn.Linear(hidden, hidden, bias=False)
            self.wv = nn.Linear(hidden, hidden, bias=False)
            self.w_gate = nn.Linear(hidden, heads, bias=True)
            nn.init.constant_(self.w_gate.bias, -4.0)
            if mixer_kind == "gdn":
                self.w_beta = nn.Linear(hidden, heads, bias=True)
                nn.init.constant_(self.w_beta.bias, 0.2)
        self.wo = nn.Linear(hidden, hidden, bias=False)

        self.norm2 = nn.RMSNorm(hidden, eps=1e-5)
        self.ffn_gate = nn.Linear(hidden, inter_dim, bias=False)
        self.ffn_up = nn.Linear(hidden, inter_dim, bias=False)
        self.ffn_down = nn.Linear(inter_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        h = self.norm1(x)
        if self.mixer_kind == "fa2_gqa":
            q = self.wq(h).view(bsz, seqlen, self.heads, self.head_dim)
            k = self.wk(h).view(bsz, seqlen, self.kv_heads, self.head_dim)
            v = self.wv(h).view(bsz, seqlen, self.kv_heads, self.head_dim)
            o = flash_attn_func(q, k, v, causal=True).reshape(bsz, seqlen, self.hidden)
        else:
            q = F.normalize(
                self.wq(h).view(bsz, seqlen, self.heads, self.head_dim).transpose(1, 2),
                dim=-1,
            )
            k = F.normalize(
                self.wk(h).view(bsz, seqlen, self.heads, self.head_dim).transpose(1, 2),
                dim=-1,
            )
            v = self.wv(h).view(bsz, seqlen, self.heads, self.head_dim).transpose(1, 2)
            g = -F.softplus(self.w_gate(h).float()).transpose(1, 2)
            if self.mixer_kind == "gdn":
                beta = (torch.sigmoid(self.w_beta(h).float()) * 1.8).transpose(1, 2)
                o_bh = triton_chunkwise_gdn_wy(q, k, v, beta, g)
            else:
                o_bh = triton_chunkwise_ssd_gla(q, k, v, g)
            o = o_bh.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden)

        x = x + self.wo(o)
        h2 = self.norm2(x)
        x = x + self.ffn_down(F.silu(self.ffn_gate(h2)) * self.ffn_up(h2))
        return x


def _apply_1d_rope(x_bshd: torch.Tensor) -> torch.Tensor:
    bsz, seqlen, heads, dim = x_bshd.shape
    half = dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=x_bshd.device, dtype=torch.float32) / half)
    )
    pos = torch.arange(seqlen, device=x_bshd.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)[None, :, None, :]  # [1, S, 1, D/2]
    cos_f, sin_f = torch.cos(freqs), torch.sin(freqs)
    xf = x_bshd.float()
    x1, x2 = xf[..., :half], xf[..., half:]
    out = torch.cat([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], dim=-1)
    return out.to(x_bshd.dtype)


class SingleLayerOverwriteMQAR(nn.Module):
    """Position-embedding-free 1-layer mixer (`H` heads x `D=64`) with per-QKV causal
    `ShortConv1d(kernel_size=2) + SiLU` and fair input-dependent `beta_t` write gate
    for both `ssd_gla` (`Mamba-2 SSD` `dt_t`) and `gdn_wy` (`Gated DeltaNet`), plus
    `fa2_softmax` with 1D RoPE."""

    def __init__(self, vocab: int, num_vals: int, kind: str, heads: int = 4) -> None:
        super().__init__()
        self.kind = kind
        self.h = heads
        self.dh = 64
        self.d = heads * 64
        self.emb = nn.Embedding(vocab, self.d)
        self.norm = nn.RMSNorm(self.d, eps=1e-5)
        self.wq = nn.Linear(self.d, self.d, bias=False)
        self.wk = nn.Linear(self.d, self.d, bias=False)
        self.wv = nn.Linear(self.d, self.d, bias=False)
        self.cq = nn.Conv1d(self.d, self.d, kernel_size=2, padding=1, groups=self.d)
        self.ck = nn.Conv1d(self.d, self.d, kernel_size=2, padding=1, groups=self.d)
        self.cv = nn.Conv1d(self.d, self.d, kernel_size=2, padding=1, groups=self.d)
        self.w_gate = nn.Linear(self.d, self.h, bias=True)
        nn.init.constant_(self.w_gate.bias, -4.0)  # exp(-softplus(-4.0)) ~ 0.982 initial retention
        self.w_beta = nn.Linear(self.d, self.h, bias=True)
        nn.init.constant_(self.w_beta.bias, 0.2)
        self.wo = nn.Linear(self.d, self.d, bias=False)
        self.head = nn.Linear(self.d, num_vals, bias=False)

    def forward(self, tok_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = tok_ids.shape
        x0 = self.emb(tok_ids)
        h = self.norm(x0)
        q_c = F.silu(self.cq(self.wq(h).transpose(1, 2))[:, :, :seqlen].transpose(1, 2)).contiguous()
        k_c = F.silu(self.ck(self.wk(h).transpose(1, 2))[:, :, :seqlen].transpose(1, 2)).contiguous()
        v_c = F.silu(self.cv(self.wv(h).transpose(1, 2))[:, :, :seqlen].transpose(1, 2)).contiguous()

        if self.kind == "fa2_softmax":
            qf = _apply_1d_rope(q_c.view(bsz, seqlen, self.h, self.dh)).bfloat16()
            kf = _apply_1d_rope(k_c.view(bsz, seqlen, self.h, self.dh)).bfloat16()
            vf = v_c.view(bsz, seqlen, self.h, self.dh).bfloat16()
            o_mix = flash_attn_func(qf, kf, vf, causal=True).float().reshape(bsz, seqlen, self.d)
        else:
            q = F.normalize(
                q_c.view(bsz, seqlen, self.h, self.dh).transpose(1, 2), dim=-1
            ).bfloat16()
            k = F.normalize(
                k_c.view(bsz, seqlen, self.h, self.dh).transpose(1, 2), dim=-1
            ).bfloat16()
            v = v_c.view(bsz, seqlen, self.h, self.dh).transpose(1, 2).bfloat16()
            g = -F.softplus(self.w_gate(h).float()).transpose(1, 2)
            beta = (torch.sigmoid(self.w_beta(h).float()) * 1.8).transpose(1, 2)
            if self.kind == "gdn_wy":
                o_bh = triton_chunkwise_gdn_wy(q, k, v, beta, g)
            else:
                # Fair Mamba-2 SSD (`S_t = exp(g_t) S_{t-1} + beta_t k_t v_t^T`):
                v_gated = (beta.unsqueeze(-1) * v.float()).bfloat16()
                o_bh = triton_chunkwise_ssd_gla(q, k, v_gated, g)
            o_mix = o_bh.float().transpose(1, 2).contiguous().view(bsz, seqlen, self.d)

        return self.head(x0 + self.wo(o_mix))


def run_suite_b_mqar_and_hybrid(device: torch.device, is_smoke: bool) -> dict[str, Any]:
    # Part B.1a: Positive Control — Uncrowded Orthogonal MQAR (`num_keys=16 << H*D=256`, `S=64`, 1x overwrite)
    # Proves `triton_chunkwise_ssd_gla` trains cleanly to ~100% when keys are mutually orthogonal
    # (`k_i^T k_j = 0`) and a single late overwrite out-amplitudes a decayed early write (`alpha^23 > alpha^47`).
    nk_ctrl, nv_ctrl = 16, 32
    voc_ctrl = nk_ctrl + nv_ctrl + 4
    pad_ctrl = nk_ctrl + nv_ctrl
    steps_ctrl = 220 if is_smoke else 360
    bsz_ctrl = 128

    ev8_c = torch.arange(8, device=device) * 2
    ev4_c = 24 + torch.arange(4, device=device) * 2
    qs_c = 48 + torch.arange(8, device=device) * 2

    def _sample_uncrowded_ctrl(
        batch_sz: int, g_rng: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kp = torch.rand((batch_sz, nk_ctrl), generator=g_rng, device=device).argsort(dim=-1)[:, :8]
        v1 = torch.randint(0, nv_ctrl, (batch_sz, 8), generator=g_rng, device=device)
        v2 = torch.randint(0, nv_ctrl, (batch_sz, 4), generator=g_rng, device=device)
        sq = torch.full((batch_sz, 64), pad_ctrl, device=device, dtype=torch.long)
        sq[:, ev8_c] = kp
        sq[:, ev8_c + 1] = nk_ctrl + v1
        sq[:, ev4_c] = kp[:, :4]
        sq[:, ev4_c + 1] = nk_ctrl + v2
        sq[:, qs_c] = kp
        return sq, qs_c.unsqueeze(0).expand(batch_sz, 8), torch.cat([v2, v1[:, 4:]], dim=-1)

    mqar_ctrl_res: dict[str, Any] = {}
    for kind in ("ssd_gla", "gdn_wy", "fa2_softmax"):
        torch.manual_seed(11940)
        m_c = SingleLayerOverwriteMQAR(voc_ctrl, nv_ctrl, kind, heads=4).to(device=device, dtype=torch.float32)
        opt_c = torch.optim.AdamW(m_c.parameters(), lr=1.2e-3, betas=(0.9, 0.98), weight_decay=0.01, fused=True)
        g_c = torch.Generator(device=device)
        g_c.manual_seed(11945)
        ev_s_c, ev_q_c, ev_t_c = _sample_uncrowded_ctrl(256, g_c)
        init_c = 0.0
        for st in range(steps_ctrl):
            ts, tq, tt = _sample_uncrowded_ctrl(bsz_ctrl, g_c)
            opt_c.zero_grad()
            lg = m_c(ts).gather(1, tq.unsqueeze(-1).expand(-1, -1, nv_ctrl))
            loss_c = F.cross_entropy(lg.reshape(-1, nv_ctrl), tt.reshape(-1))
            if st == 0:
                init_c = float(loss_c.item())
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(m_c.parameters(), 1.0, foreach=True)
            opt_c.step()
        with torch.no_grad():
            lge = m_c(ev_s_c).gather(1, ev_q_c.unsqueeze(-1).expand(-1, -1, nv_ctrl))
            ce_c = float(F.cross_entropy(lge.reshape(-1, nv_ctrl), ev_t_c.reshape(-1)).item())
            pr_c = lge.argmax(dim=-1)
            mqar_ctrl_res[kind] = {
                "steps": steps_ctrl,
                "initial_ce_loss": round(init_c, 4),
                "eval_ce_loss": round(ce_c, 4),
                "eval_accuracy_all": round(float((pr_c == ev_t_c).float().mean().item()), 4),
                "eval_accuracy_overwritten_keys": round(float((pr_c[:, :4] == ev_t_c[:, :4]).float().mean().item()), 4),
                "eval_accuracy_untouched_keys": round(float((pr_c[:, 4:] == ev_t_c[:, 4:]).float().mean().item()), 4),
            }
        del m_c, opt_c
        gc.collect()
        torch.cuda.empty_cache()

    # Part B.1b: Crowded State-Reassignment MQAR (`num_keys=96 > D=64` in `H=1, D=64`, `S=128`)
    # 20 distinct KV pairs written (slots 0..39) -> first 4 keys reinforced 3x with OLD value (slots 40..63)
    # -> reassigned 1x to NEW value (slots 64..71) -> 12 distractor overwrites (slots 72..95) -> query 4
    # reassigned + 4 untouched early keys (slots 112..127). Every model gets best-of-LR `{1.2e-3, 3.0e-3}`!
    num_keys = 96
    num_vals = 32
    vocab = num_keys + num_vals + 4
    pad_id = num_keys + num_vals
    steps_mqar = 240 if is_smoke else 400
    bsz_mqar = 128
    seq_mqar = 128  # 2 chunks of 64!

    s20 = torch.arange(20, device=device) * 2
    s_new = 64 + torch.arange(4, device=device) * 2
    s_dist = 72 + torch.arange(12, device=device) * 2
    q_slots = 112 + torch.arange(8, device=device) * 2

    def _sample_overwrite_mqar(
        batch_sz: int, g_rng: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys_p = torch.rand((batch_sz, num_keys), generator=g_rng, device=device).argsort(dim=-1)[:, :20]
        vals_1 = torch.randint(0, num_vals, (batch_sz, 20), generator=g_rng, device=device)
        vals_2 = torch.randint(0, num_vals, (batch_sz, 4), generator=g_rng, device=device)
        v_dist = torch.randint(0, num_vals, (batch_sz, 12), generator=g_rng, device=device)
        seq = torch.full((batch_sz, seq_mqar), pad_id, device=device, dtype=torch.long)
        seq[:, s20] = keys_p
        seq[:, s20 + 1] = num_keys + vals_1
        for r_idx in range(3):
            sr = 40 + r_idx * 8 + torch.arange(4, device=device) * 2
            seq[:, sr] = keys_p[:, :4]
            seq[:, sr + 1] = num_keys + vals_1[:, :4]
        seq[:, s_new] = keys_p[:, :4]
        seq[:, s_new + 1] = num_keys + vals_2
        seq[:, s_dist] = keys_p[:, 8:20]
        seq[:, s_dist + 1] = num_keys + v_dist
        seq[:, q_slots] = keys_p[:, :8]
        q_pos = q_slots.unsqueeze(0).expand(batch_sz, 8)
        targets = torch.cat([vals_2, vals_1[:, 4:8]], dim=-1)
        return seq, q_pos, targets

    mqar_res: dict[str, Any] = {}
    for kind in ("ssd_gla", "gdn_wy", "fa2_softmax"):
        best_entry: dict[str, Any] | None = None
        for lr_cand in (1.2e-3, 3.0e-3):
            torch.manual_seed(11940)
            model_mqar = SingleLayerOverwriteMQAR(vocab, num_vals, kind, heads=1).to(
                device=device, dtype=torch.float32
            )
            params = list(model_mqar.parameters())
            opt = torch.optim.AdamW(
                params, lr=lr_cand, betas=(0.9, 0.98), weight_decay=0.01, fused=True
            )

            g_data = torch.Generator(device=device)
            g_data.manual_seed(11945)
            eval_seq, eval_qp, eval_tgt = _sample_overwrite_mqar(512, g_data)

            init_loss = 0.0
            for st in range(steps_mqar):
                tr_seq, tr_qp, tr_tgt = _sample_overwrite_mqar(bsz_mqar, g_data)
                for p in params:
                    p.grad = None
                logits_all = model_mqar(tr_seq)
                logits = logits_all.gather(1, tr_qp.unsqueeze(-1).expand(-1, -1, num_vals))
                loss = F.cross_entropy(logits.reshape(-1, num_vals), tr_tgt.reshape(-1))
                if st == 0:
                    init_loss = float(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
                opt.step()

            with torch.no_grad():
                lg_all = model_mqar(eval_seq)
                lg_ev = lg_all.gather(1, eval_qp.unsqueeze(-1).expand(-1, -1, num_vals))
                final_loss = float(
                    F.cross_entropy(lg_ev.reshape(-1, num_vals), eval_tgt.reshape(-1)).item()
                )
                preds = lg_ev.argmax(dim=-1)
                acc_all = float((preds == eval_tgt).float().mean().item())
                acc_overwritten = float((preds[:, :4] == eval_tgt[:, :4]).float().mean().item())
                acc_untouched = float((preds[:, 4:] == eval_tgt[:, 4:]).float().mean().item())

            cand_entry = {
                "steps": steps_mqar,
                "best_lr": lr_cand,
                "initial_ce_loss": round(init_loss, 4),
                "eval_ce_loss": round(final_loss, 4),
                "eval_accuracy_all": round(acc_all, 4),
                "eval_accuracy_overwritten_keys": round(acc_overwritten, 4),
                "eval_accuracy_untouched_keys": round(acc_untouched, 4),
            }
            if best_entry is None or cand_entry["eval_ce_loss"] < best_entry["eval_ce_loss"]:
                best_entry = cand_entry
            del model_mqar, params, opt
            gc.collect()
            torch.cuda.empty_cache()
        assert best_entry is not None
        mqar_res[kind] = best_entry

    # Part B.2: 4-Layer H=2048 (16 heads x D=128, FFN=5632) Stack Scaling (`4:0` Pure FA2 vs `3:1` Hybrid GDN+FA2 vs `0:4` Pure GDN)
    stack_seqs = [4096, 16384] if is_smoke else [4096, 16384, 65536]
    warmup_b2 = 2 if is_smoke else 4
    iters_b2 = 4 if is_smoke else 10
    H_mod, I_mod, heads_mod = (1024, 2816, 8) if is_smoke else (2048, 5632, 16)

    hybrid_stack_res: dict[str, Any] = {}
    for s_len in stack_seqs:
        x_in = torch.randn((1, s_len, H_mod), device=device, dtype=torch.bfloat16)
        dy_in = torch.randn_like(x_in)
        seq_entry: dict[str, Any] = {}

        for stack_name, layer_kinds in (
            ("pure_fa2_4layer", ["fa2_gqa", "fa2_gqa", "fa2_gqa", "fa2_gqa"]),
            ("hybrid_3gdn_1fa2_4layer", ["gdn", "gdn", "gdn", "fa2_gqa"]),
            ("pure_gdn_4layer", ["gdn", "gdn", "gdn", "gdn"]),
        ):
            layers = nn.ModuleList(
                [HybridMixerLayer(H_mod, I_mod, heads_mod, lk) for lk in layer_kinds]
            ).to(device=device, dtype=torch.bfloat16)
            x_req = x_in.detach().clone().requires_grad_(True)

            def _step_stack() -> None:
                for p in layers.parameters():
                    p.grad = None
                x_req.grad = None
                h_cur = x_req
                for lyr in layers:
                    h_cur = lyr(h_cur)
                torch.autograd.backward(h_cur, dy_in)

            seq_entry[stack_name] = time_fn_with_peak_mem(
                _step_stack, warmup=warmup_b2, iters=iters_b2, device=device
            )
            del layers, x_req
            gc.collect()
            torch.cuda.empty_cache()

        hybrid_stack_res[str(s_len)] = seq_entry

    return {
        "mqar_uncrowded_positive_control": mqar_ctrl_res,
        "mqar_associative_overwrite": mqar_res,
        "four_layer_hybrid_stack_scaling": hybrid_stack_res,
    }


# ============================================================================
# 8. Suite C: 8-GPU Context Parallelism (S = 16K .. 262K, SSD / GDN / Ulysses FA2)
# ============================================================================


def run_suite_c_cp8(
    rank: int, world_size: int, device: torch.device, is_smoke: bool
) -> dict[str, Any]:
    # 1. Exact 8-GPU vs 1-GPU Parity Verification at S_global = 4096 (512 tokens = 8 chunks per rank)
    B_v, H_v, S_glob_v, D_v = 1, 8, 4096, 128
    S_loc_v = S_glob_v // world_size
    gen_v = torch.Generator(device=device)
    gen_v.manual_seed(11970)

    q_full = F.normalize(torch.randn((B_v, H_v, S_glob_v, D_v), generator=gen_v, device=device, dtype=torch.bfloat16), dim=-1)
    k_full = F.normalize(torch.randn((B_v, H_v, S_glob_v, D_v), generator=gen_v, device=device, dtype=torch.bfloat16), dim=-1)
    v_full = torch.randn((B_v, H_v, S_glob_v, D_v), generator=gen_v, device=device, dtype=torch.bfloat16)
    b_full = torch.sigmoid(torch.randn((B_v, H_v, S_glob_v), generator=gen_v, device=device, dtype=torch.float32)) * 1.5
    g_full = -F.softplus(torch.randn((B_v, H_v, S_glob_v), generator=gen_v, device=device, dtype=torch.float32) - 2.0)
    do_full = torch.randn((B_v, H_v, S_glob_v, D_v), generator=gen_v, device=device, dtype=torch.bfloat16)

    # 1-GPU references
    q1_s, k1_s, v1_s, g1_s = [x.clone().requires_grad_(True) for x in (q_full, k_full, v_full, g_full)]
    o1_s = triton_chunkwise_ssd_gla(q1_s, k1_s, v1_s, g1_s)
    torch.autograd.backward(o1_s, do_full)

    q1_g, k1_g, v1_g, b1_g, g1_g = [x.clone().requires_grad_(True) for x in (q_full, k_full, v_full, b_full, g_full)]
    o1_g = triton_chunkwise_gdn_wy(q1_g, k1_g, v1_g, b1_g, g1_g)
    torch.autograd.backward(o1_g, do_full)

    sl = slice(rank * S_loc_v, (rank + 1) * S_loc_v)
    do_loc = do_full[:, :, sl].contiguous()

    # CP=8 SSD Parallel Monoid
    qc_s = q_full[:, :, sl].contiguous().detach().requires_grad_(True)
    kc_s = k_full[:, :, sl].contiguous().detach().requires_grad_(True)
    vc_s = v_full[:, :, sl].contiguous().detach().requires_grad_(True)
    gc_s = g_full[:, :, sl].contiguous().detach().requires_grad_(True)
    oc_s = _CPSSDParallelMonoidAutograd.apply(qc_s, kc_s, vc_s, gc_s)
    torch.autograd.backward(oc_s, do_loc)

    # CP=8 GDN P2P State Pass
    qc_p = q_full[:, :, sl].contiguous().detach().requires_grad_(True)
    kc_p = k_full[:, :, sl].contiguous().detach().requires_grad_(True)
    vc_p = v_full[:, :, sl].contiguous().detach().requires_grad_(True)
    bc_p = b_full[:, :, sl].contiguous().detach().requires_grad_(True)
    gc_p = g_full[:, :, sl].contiguous().detach().requires_grad_(True)
    oc_p = _CPGDNP2PStatePassAutograd.apply(qc_p, kc_p, vc_p, bc_p, gc_p)
    torch.autograd.backward(oc_p, do_loc)

    # CP=8 GDN Parallel Monoid
    qc_m = q_full[:, :, sl].contiguous().detach().requires_grad_(True)
    kc_m = k_full[:, :, sl].contiguous().detach().requires_grad_(True)
    vc_m = v_full[:, :, sl].contiguous().detach().requires_grad_(True)
    bc_m = b_full[:, :, sl].contiguous().detach().requires_grad_(True)
    gc_m = g_full[:, :, sl].contiguous().detach().requires_grad_(True)
    oc_m = _CPGDNParallelMonoidAutograd.apply(qc_m, kc_m, vc_m, bc_m, gc_m)
    torch.autograd.backward(oc_m, do_loc)

    def _max_rank_rl2(loc_t: torch.Tensor, ref_full: torch.Tensor) -> float:
        val = rel_l2(loc_t, ref_full[:, :, sl])
        t_v = torch.tensor([val], device=device, dtype=torch.float64)
        dist.all_reduce(t_v, op=dist.ReduceOp.MAX)
        return round(float(t_v.item()), 6)

    cp_parity = {
        "cp8_ssd_monoid_max_out_rel_l2": _max_rank_rl2(oc_s, o1_s),
        "cp8_ssd_monoid_max_grad_rel_l2": max(
            _max_rank_rl2(qc_s.grad, q1_s.grad),
            _max_rank_rl2(kc_s.grad, k1_s.grad),
            _max_rank_rl2(vc_s.grad, v1_s.grad),
            _max_rank_rl2(gc_s.grad, g1_s.grad),
        ),
        "cp8_gdn_p2p_max_out_rel_l2": _max_rank_rl2(oc_p, o1_g),
        "cp8_gdn_p2p_max_grad_rel_l2": max(
            _max_rank_rl2(qc_p.grad, q1_g.grad),
            _max_rank_rl2(kc_p.grad, k1_g.grad),
            _max_rank_rl2(vc_p.grad, v1_g.grad),
            _max_rank_rl2(bc_p.grad, b1_g.grad),
            _max_rank_rl2(gc_p.grad, g1_g.grad),
        ),
        "cp8_gdn_monoid_max_out_rel_l2": _max_rank_rl2(oc_m, o1_g),
        "cp8_gdn_monoid_max_grad_rel_l2": max(
            _max_rank_rl2(qc_m.grad, q1_g.grad),
            _max_rank_rl2(kc_m.grad, k1_g.grad),
            _max_rank_rl2(vc_m.grad, v1_g.grad),
            _max_rank_rl2(bc_m.grad, b1_g.grad),
            _max_rank_rl2(gc_m.grad, g1_g.grad),
        ),
    }
    del q_full, k_full, v_full, b_full, g_full, do_full, o1_s, o1_g
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Long-Context 8-GPU CP=8 Scaling Sweep: S_global = 16,384 .. 262,144 (`256K` tokens!)
    cp_seqs = [16384, 65536] if is_smoke else [16384, 65536, 131072, 262144]
    warmup_c = 2 if is_smoke else 4
    iters_c = 4 if is_smoke else 10
    B_c, H_c, H_kv_c, D_c = 1, 16, 8, 128

    cp_scaling: dict[str, Any] = {}
    for S_glob in cp_seqs:
        S_loc = S_glob // world_size
        gen_c = torch.Generator(device=device)
        gen_c.manual_seed(11980 + rank + S_glob)

        ql = F.normalize(torch.randn((B_c, H_c, S_loc, D_c), generator=gen_c, device=device, dtype=torch.bfloat16), dim=-1)
        kl = F.normalize(torch.randn((B_c, H_c, S_loc, D_c), generator=gen_c, device=device, dtype=torch.bfloat16), dim=-1)
        vl = torch.randn((B_c, H_c, S_loc, D_c), generator=gen_c, device=device, dtype=torch.bfloat16)
        bl = torch.sigmoid(torch.randn((B_c, H_c, S_loc), generator=gen_c, device=device, dtype=torch.float32)) * 1.5
        gl = -F.softplus(torch.randn((B_c, H_c, S_loc), generator=gen_c, device=device, dtype=torch.float32) - 2.0)
        dol = torch.randn_like(vl)

        def _time_dist(fn_d: Callable[[], None], it_override: int | None = None) -> dict[str, float]:
            n_it = it_override if it_override is not None else iters_c
            for _ in range(warmup_c):
                fn_d()
            dist.barrier()
            torch.cuda.synchronize(device)
            base_m = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            fn_d()
            torch.cuda.synchronize(device)
            inc_m = max(0.0, (torch.cuda.max_memory_allocated(device) - base_m) / (1024.0**2))
            m_t = torch.tensor([inc_m], device=device, dtype=torch.float64)
            dist.all_reduce(m_t, op=dist.ReduceOp.MAX)

            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            samps: list[float] = []
            for _ in range(n_it):
                dist.barrier()
                s_ev.record()
                fn_d()
                e_ev.record()
                torch.cuda.synchronize(device)
                dt = torch.tensor([float(s_ev.elapsed_time(e_ev))], device=device, dtype=torch.float64)
                dist.all_reduce(dt, op=dist.ReduceOp.MAX)
                samps.append(float(dt.item()))
            ss = sorted(samps)
            med = statistics.median(ss)
            return {
                "step_ms_median": round(med, 4),
                "step_ms_iqr": round(ss[(3 * (len(ss) - 1)) // 4] - ss[(len(ss) - 1) // 4], 4),
                "cluster_ktok_per_s": round((float(B_c * S_glob) / (med * 1e-3)) / 1e3, 2),
                "per_rank_peak_inc_mib": round(float(m_t.item()), 2),
            }

        seq_modes: dict[str, Any] = {}

        # Mode 1: CP=8 SSD Parallel Monoid AllGather
        qs, ks, vs, gs = [x.clone().requires_grad_(True) for x in (ql, kl, vl, gl)]

        def _f_ssd() -> None:
            qs.grad = ks.grad = vs.grad = gs.grad = None
            o = _CPSSDParallelMonoidAutograd.apply(qs, ks, vs, gs)
            torch.autograd.backward(o, dol)

        seq_modes["cp8_ssd_parallel_monoid"] = {
            **_time_dist(_f_ssd),
            "comm_mib_per_rank": round((2 * B_c * H_c * D_c * D_c * 4) / (1024.0**2), 2),
        }
        del qs, ks, vs, gs

        # Mode 2: CP=8 GDN P2P State Pass
        qp, kp, vp, bp, gp = [x.clone().requires_grad_(True) for x in (ql, kl, vl, bl, gl)]

        def _f_p2p() -> None:
            qp.grad = kp.grad = vp.grad = bp.grad = gp.grad = None
            o = _CPGDNP2PStatePassAutograd.apply(qp, kp, vp, bp, gp)
            torch.autograd.backward(o, dol)

        seq_modes["cp8_gdn_p2p_state_pass"] = {
            **_time_dist(_f_p2p),
            "comm_mib_per_rank": round((2 * B_c * H_c * D_c * D_c * 4) / (1024.0**2), 2),
        }
        del qp, kp, vp, bp, gp

        # Mode 3: CP=8 GDN Parallel Monoid AllGather
        qm, km, vm, bm, gm = [x.clone().requires_grad_(True) for x in (ql, kl, vl, bl, gl)]

        def _f_mon() -> None:
            qm.grad = km.grad = vm.grad = bm.grad = gm.grad = None
            o = _CPGDNParallelMonoidAutograd.apply(qm, km, vm, bm, gm)
            torch.autograd.backward(o, dol)

        seq_modes["cp8_gdn_parallel_monoid"] = {
            **_time_dist(_f_mon),
            "comm_mib_per_rank": round((3 * B_c * H_c * D_c * D_c * 4) / (1024.0**2), 2),
        }
        del qm, km, vm, bm, gm

        # Mode 4: CP=8 GQA-Native DeepSpeed-Ulysses + FlashAttention-2 (16Q / 8KV)
        qu = ql.transpose(1, 2).contiguous().detach().requires_grad_(True)
        ku = kl[:, :H_kv_c].transpose(1, 2).contiguous().detach().requires_grad_(True)
        vu = vl[:, :H_kv_c].transpose(1, 2).contiguous().detach().requires_grad_(True)
        dou = dol.transpose(1, 2).contiguous()

        def _f_uly() -> None:
            qu.grad = ku.grad = vu.grad = None
            o = cp_ulysses_flash_attn2_gqa(qu, ku, vu)
            torch.autograd.backward(o, dou)

        it_fa = 4 if S_glob >= 262144 and not is_smoke else None
        # Forward + backward AllToAll moves Q (16), K (8), V (8), O (16) * 2 (fwd+bwd) = 96 head-shards of BF16
        uly_comm_mib = (2 * B_c * S_loc * (2 * H_c + 2 * H_kv_c) * D_c * 2) / (1024.0**2)
        seq_modes["cp8_ulysses_flash_attn2_gqa"] = {
            **_time_dist(_f_uly, it_override=it_fa),
            "comm_mib_per_rank": round(uly_comm_mib, 2),
        }
        del qu, ku, vu, dou

        # Mode 5 (at S = 65536 and 131072): Compare 4-Mixer `3:1` Hybrid (3x CP8 GDN Monoid + 1x CP8 Ulysses FA2)
        # vs 4-Mixer Pure CP8 Ulysses FA2!
        if S_glob in (65536, 131072):
            qh, kh, vh, bh, gh = [x.clone().requires_grad_(True) for x in (ql, kl, vl, bl, gl)]
            qu4 = ql.transpose(1, 2).contiguous().detach().requires_grad_(True)
            ku4 = kl[:, :H_kv_c].transpose(1, 2).contiguous().detach().requires_grad_(True)
            vu4 = vl[:, :H_kv_c].transpose(1, 2).contiguous().detach().requires_grad_(True)
            dou4 = dol.transpose(1, 2).contiguous()

            def _f_hyb4() -> None:
                for tx in (qh, kh, vh, bh, gh, qu4, ku4, vu4):
                    tx.grad = None
                o_lin = (
                    _CPGDNParallelMonoidAutograd.apply(qh, kh, vh, bh, gh)
                    + _CPGDNParallelMonoidAutograd.apply(qh, kh, vh, bh, gh)
                    + _CPGDNParallelMonoidAutograd.apply(qh, kh, vh, bh, gh)
                )
                o_att = cp_ulysses_flash_attn2_gqa(qu4, ku4, vu4)
                torch.autograd.backward([o_lin, o_att], [dol, dou4])

            def _f_fa4() -> None:
                qu4.grad = ku4.grad = vu4.grad = None
                o_att = (
                    cp_ulysses_flash_attn2_gqa(qu4, ku4, vu4)
                    + cp_ulysses_flash_attn2_gqa(qu4, ku4, vu4)
                    + cp_ulysses_flash_attn2_gqa(qu4, ku4, vu4)
                    + cp_ulysses_flash_attn2_gqa(qu4, ku4, vu4)
                )
                torch.autograd.backward(o_att, dou4)

            seq_modes["cp8_4mixer_hybrid_3gdn_1fa2"] = _time_dist(_f_hyb4, it_override=5)
            seq_modes["cp8_4mixer_pure_4fa2"] = _time_dist(_f_fa4, it_override=5)
            del qh, kh, vh, bh, gh, qu4, ku4, vu4, dou4

        del ql, kl, vl, bl, gl, dol
        gc.collect()
        torch.cuda.empty_cache()

        cp_scaling[str(S_glob)] = {
            "global_seq_len": S_glob,
            "local_seq_len": S_loc,
            "heads": H_c,
            "head_dim": D_c,
            "modes": seq_modes,
        }

    return {
        "cp8_vs_1gpu_parity": cp_parity,
        "cp8_long_context_scaling": cp_scaling,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E119: Chunkwise Mamba-2 SSD/GLA & WY Gated DeltaNet Training Kernels + 8-GPU CP"
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(minutes=30), device_id=device
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8, f"Expected 8 GPUs, got {world_size}"

    t0 = time.perf_counter()
    is_smoke = args.mode == "smoke"

    s0_res: dict[str, Any] = {}
    sa_res: dict[str, Any] = {}
    sb_res: dict[str, Any] = {}

    if rank == 0:
        s0_res = run_suite_0_audits(device, is_smoke)
        print(f"[E119] Suite 0 complete at +{time.perf_counter() - t0:.1f}s", flush=True)
    dist.barrier()

    if rank == 0:
        sa_res = run_suite_a_seq_scaling(device, is_smoke)
        print(f"[E119] Suite A complete at +{time.perf_counter() - t0:.1f}s", flush=True)
    dist.barrier()

    if rank == 0:
        sb_res = run_suite_b_mqar_and_hybrid(device, is_smoke)
        print(f"[E119] Suite B complete at +{time.perf_counter() - t0:.1f}s", flush=True)
    dist.barrier()

    sc_res = run_suite_c_cp8(rank, world_size, device, is_smoke)
    dist.barrier()
    elapsed_s = time.perf_counter() - t0

    if rank == 0:
        print(f"[E119] Suite C complete at +{elapsed_s:.1f}s", flush=True)
        payload = {
            "experiment": "E119_linear_attention_gdn_training",
            "mode": args.mode,
            "world_size": world_size,
            "elapsed_s": round(elapsed_s, 2),
            "environment": environment_metadata(),
            "suite_0_audits": s0_res,
            "suite_a_single_gpu_scaling": sa_res,
            "suite_b_mqar_and_hybrid_stack": sb_res,
            "suite_c_cp8_long_context": sc_res,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"Wrote {args.out} in {elapsed_s:.2f}s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
