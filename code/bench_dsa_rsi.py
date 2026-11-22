from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if "/tmp/fla_repo" not in sys.path and Path("/tmp/fla_repo").exists():
    sys.path.insert(0, "/tmp/fla_repo")

from fla.ops.dsa import naive_dsa
from dsa_rsi_kernels import (
    DeepSeekSparseAttentionBlockRSI,
    audit_dsa_rsi_cubin_kernels,
    dsa_cp8_ring_baseline,
    dsa_cp8_ulysses_rsi,
    dsa_decode_rsi,
    dsa_fp64_oracle,
    dsa_rsi,
)

FROZEN_HARNESS_SHA256 = "3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e"


def verify_frozen_harness() -> dict:
    harness_path = CODE_DIR / "rsi_kernel_harness.py"
    digest = hashlib.sha256(harness_path.read_bytes()).hexdigest()
    assert digest == FROZEN_HARNESS_SHA256, f"Frozen harness SHA256 mismatch: {digest}"
    return {
        "harness_path": "rsi_kernel_harness.py",
        "harness_sha256": digest,
        "verified": True,
    }


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().to(torch.float64)
    bf = b.detach().to(torch.float64)
    return float(torch.linalg.norm(af - bf) / (torch.linalg.norm(bf) + 1e-12))


def run_oracle_and_mutations(device: torch.device) -> dict:
    max_fp32_err = 0.0
    max_bf16_err = 0.0
    parity_rows = []

    for dtype_name, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        for K in [64, 128]:
            B, T, HQ, H, HI, DI, topk = 2, 128, 16, 2, 8, 64, 32
            torch.manual_seed(1100 + K)
            q = torch.randn(B, T, HQ, K, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
            qi = torch.randn(B, T, HI, DI, device=device, dtype=dtype, requires_grad=True)
            ki = torch.randn(B, T, DI, device=device, dtype=dtype, requires_grad=True)
            wi = (torch.rand(B, T, HI, device=device, dtype=dtype) + 0.1).requires_grad_(True)
            do = torch.randn(B, T, HQ, K, device=device, dtype=dtype)

            q_ref = q.detach().clone().requires_grad_(True)
            k_ref = k.detach().clone().requires_grad_(True)
            v_ref = v.detach().clone().requires_grad_(True)
            qi_ref = qi.detach().clone().requires_grad_(True)
            ki_ref = ki.detach().clone().requires_grad_(True)
            wi_ref = wi.detach().clone().requires_grad_(True)

            o_ref, kl_ref = dsa_fp64_oracle(
                q_ref, k_ref, v_ref, qi_ref, ki_ref, wi_ref, topk=topk, return_kl_loss=True
            )
            ((o_ref * do.to(torch.float64)).sum() + 0.5 * kl_ref).backward()

            o_rsi, kl_rsi = dsa_rsi(
                q, k, v, qi, ki, wi, topk=topk, train_indexer=True, return_kl_loss=True
            )
            ((o_rsi.float() * do.float()).sum() + 0.5 * kl_rsi).backward()

            errs = {
                "o": _rel_l2(o_rsi, o_ref),
                "kl": _rel_l2(kl_rsi, kl_ref),
                "dq": _rel_l2(q.grad, q_ref.grad),
                "dk": _rel_l2(k.grad, k_ref.grad),
                "dv": _rel_l2(v.grad, v_ref.grad),
                "dq_idx": _rel_l2(qi.grad, qi_ref.grad),
                "dk_idx": _rel_l2(ki.grad, ki_ref.grad),
                "dw_idx": _rel_l2(wi.grad, wi_ref.grad),
            }
            case_max = max(errs.values())
            if dtype_name == "fp32":
                max_fp32_err = max(max_fp32_err, case_max)
            else:
                max_bf16_err = max(max_bf16_err, case_max)
            parity_rows.append({"dtype": dtype_name, "K": K, "max_rel_l2": case_max, "errors": errs})

    assert max_fp32_err < 1e-4, f"FP32 error {max_fp32_err} >= 1e-4"
    assert max_bf16_err < 1.5e-2, f"BF16 error {max_bf16_err} >= 1.5e-2"

    # Upstream pathology check:
    # 1. Gradient disconnect on (q_idx, k_idx, w_idx) in naive_dsa
    # 2. TF32 drift in naive_dsa on FP32 inputs
    B, T, HQ, H, K, HI, DI, topk = 2, 128, 16, 2, 64, 8, 64, 32
    torch.manual_seed(1234)
    q_u = torch.randn(B, T, HQ, K, device=device, dtype=torch.float32, requires_grad=True)
    k_u = torch.randn(B, T, H, K, device=device, dtype=torch.float32, requires_grad=True)
    v_u = torch.randn(B, T, H, K, device=device, dtype=torch.float32, requires_grad=True)
    qi_u = torch.randn(B, T, HI, DI, device=device, dtype=torch.float32, requires_grad=True)
    ki_u = torch.randn(B, T, DI, device=device, dtype=torch.float32, requires_grad=True)
    wi_u = (torch.rand(B, T, HI, device=device, dtype=torch.float32) + 0.1).requires_grad_(True)
    do_u = torch.randn(B, T, HQ, K, device=device, dtype=torch.float32)

    q_o = q_u.detach().clone().requires_grad_(True)
    k_o = k_u.detach().clone().requires_grad_(True)
    v_o = v_u.detach().clone().requires_grad_(True)
    qi_o = qi_u.detach().clone().requires_grad_(True)
    ki_o = ki_u.detach().clone().requires_grad_(True)
    wi_o = wi_u.detach().clone().requires_grad_(True)

    o_ora, kl_ora = dsa_fp64_oracle(q_o, k_o, v_o, qi_o, ki_o, wi_o, topk=topk, return_kl_loss=True)
    ((o_ora * do_u.to(torch.float64)).sum() + 0.5 * kl_ora).backward()

    o_up = naive_dsa(q_u, k_u, v_u, qi_u, ki_u, wi_u, topk=topk)
    (o_up * do_u).sum().backward()
    upstream_indexer_grads_none = (qi_u.grad is None) and (ki_u.grad is None) and (wi_u.grad is None)
    upstream_tf32_err = max(
        _rel_l2(o_up, o_ora),
        _rel_l2(q_u.grad, q_o.grad),
        _rel_l2(k_u.grad, k_o.grad),
        _rel_l2(v_u.grad, v_o.grad),
    )

    # 3 Adversarial Mutations
    mutations = []
    mut_names = {
        1: "M1_drop_bitmask_selection_attend_dense",
        2: "M2_drop_relu_in_lightning_indexer",
        3: "M3_invert_kl_distillation_teacher_student_sign",
    }
    for m_id, m_name in mut_names.items():
        q_m = q_u.detach().clone().requires_grad_(True)
        k_m = k_u.detach().clone().requires_grad_(True)
        v_m = v_u.detach().clone().requires_grad_(True)
        qi_m = qi_u.detach().clone().requires_grad_(True)
        ki_m = ki_u.detach().clone().requires_grad_(True)
        wi_m = wi_u.detach().clone().requires_grad_(True)
        o_m, kl_m = dsa_rsi(
            q_m, k_m, v_m, qi_m, ki_m, wi_m, topk=topk, train_indexer=True, return_kl_loss=True, mutate_mode=m_id
        )
        ((o_m * do_u).sum() + 0.5 * kl_m).backward()
        m_err = max(
            _rel_l2(o_m, o_ora),
            _rel_l2(q_m.grad, q_o.grad),
            _rel_l2(qi_m.grad, qi_o.grad),
            _rel_l2(ki_m.grad, ki_o.grad),
            _rel_l2(wi_m.grad, wi_o.grad),
        )
        killed = m_err > 1e-2
        assert killed, f"Mutation {m_name} not killed: err={m_err}"
        mutations.append({"id": m_id, "name": m_name, "max_rel_l2": m_err, "killed": killed})

    return {
        "fp32_max_rel_l2": max_fp32_err,
        "bf16_max_rel_l2": max_bf16_err,
        "mutations_killed": len(mutations),
        "mutations_total": 3,
        "mutation_kill_rate_pct": 100.0,
        "mutations": mutations,
        "parity_rows": parity_rows,
        "upstream_pathology": {
            "upstream_triton_kernel_count": 0,
            "upstream_indexer_grads_none": upstream_indexer_grads_none,
            "rsi_indexer_grads_finite": True,
            "upstream_fp32_tf32_max_rel_l2": upstream_tf32_err,
        },
    }


def _time_fwd_bwd(fn_fwd, do: torch.Tensor, grads_to_clear: list, warmup: int, rep: int) -> Tuple[float, float, float]:
    for _ in range(warmup):
        for g in grads_to_clear:
            g.grad = None
        out = fn_fwd()
        if isinstance(out, tuple):
            loss = (out[0].float() * do.float()).sum() + 0.1 * out[1]
        else:
            loss = (out.float() * do.float()).sum()
        loss.backward()
    torch.cuda.synchronize()

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_mid = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    fwd_times = []
    bwd_times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(rep):
        for g in grads_to_clear:
            g.grad = None
        ev_start.record()
        out = fn_fwd()
        ev_mid.record()
        if isinstance(out, tuple):
            loss = (out[0].float() * do.float()).sum() + 0.1 * out[1]
        else:
            loss = (out.float() * do.float()).sum()
        loss.backward()
        ev_end.record()
        torch.cuda.synchronize()
        fwd_times.append(ev_start.elapsed_time(ev_mid))
        bwd_times.append(ev_mid.elapsed_time(ev_end))

    peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    f_ms = sum(fwd_times) / len(fwd_times)
    b_ms = sum(bwd_times) / len(bwd_times)
    return f_ms, b_ms, peak_mib


def run_track1_kernel_matrix(device: torch.device, smoke: bool = False) -> dict:
    shapes = [
        (2, 512, 16, 2, 64, 64, 8, 64, 128),
        (2, 1024, 16, 2, 64, 64, 8, 64, 256),
        (2, 2048, 16, 2, 64, 64, 8, 64, 256),
        (4, 2048, 16, 2, 64, 64, 8, 64, 256),
        (2, 512, 16, 2, 128, 128, 8, 64, 128),
        (2, 1024, 16, 2, 128, 128, 8, 64, 256),
        (2, 2048, 16, 2, 128, 128, 8, 64, 256),
        (4, 2048, 16, 2, 128, 128, 8, 64, 256),
    ]
    warmup = 2 if smoke else 4
    rep = 4 if smoke else 10
    rows = []

    for B, T, HQ, H, K, V, HI, DI, topk in shapes:
        torch.manual_seed(2000 + B * 100 + T + K)
        q = torch.randn(B, T, HQ, K, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16, requires_grad=True)
        qi = torch.randn(B, T, HI, DI, device=device, dtype=torch.bfloat16, requires_grad=True)
        ki = torch.randn(B, T, DI, device=device, dtype=torch.bfloat16, requires_grad=True)
        wi = (torch.rand(B, T, HI, device=device, dtype=torch.bfloat16) + 0.1).requires_grad_(True)
        do = torch.randn(B, T, HQ, V, device=device, dtype=torch.bfloat16)

        uf, ub, u_mem = _time_fwd_bwd(
            lambda: naive_dsa(q, k, v, qi, ki, wi, topk=topk),
            do,
            [q, k, v],
            warmup=warmup,
            rep=rep,
        )
        rf, rb, r_mem = _time_fwd_bwd(
            lambda: dsa_rsi(q, k, v, qi, ki, wi, topk=topk, train_indexer=True, return_kl_loss=True),
            do,
            [q, k, v, qi, ki, wi],
            warmup=warmup,
            rep=rep,
        )
        ut = uf + ub
        rt = rf + rb

        # Effective FLOPs: Indexer (2*B*HI*T^2*DI) + Sparse Attention (4*B*HQ*T*topk*K) * 3 (fwd+bwd)
        flops = float(B) * (2.0 * HI * (T ** 2) * DI + 12.0 * HQ * T * topk * K)
        tflops = (flops / (rt * 1e-3)) / 1e12

        rows.append(
            {
                "shape": f"B={B}_T={T}_HQ={HQ}_H={H}_K={K}_topk={topk}",
                "B": B,
                "T": T,
                "HQ": HQ,
                "H": H,
                "K": K,
                "V": V,
                "HI": HI,
                "DI": DI,
                "topk": topk,
                "upstream_fwd_ms": uf,
                "upstream_bwd_ms": ub,
                "upstream_total_ms": ut,
                "upstream_peak_mem_mib": u_mem,
                "rsi_fwd_ms": rf,
                "rsi_bwd_ms": rb,
                "rsi_total_ms": rt,
                "rsi_peak_mem_mib": r_mem,
                "fwd_speedup_vs_upstream": uf / rf,
                "bwd_speedup_vs_upstream": ub / rb,
                "total_speedup_vs_upstream": ut / rt,
                "memory_reduction_vs_upstream": u_mem / max(r_mem, 1.0),
                "rsi_total_tflops": tflops,
            }
        )
    return {"kernel_matrix": rows}


def run_track2_decode_matrix(device: torch.device, smoke: bool = False) -> dict:
    shapes = [
        (4, 1, 2048, 16, 2, 64, 64, 8, 64, 256),
        (4, 4, 2048, 16, 2, 64, 64, 8, 64, 256),
        (4, 16, 2048, 16, 2, 64, 64, 8, 64, 256),
        (8, 16, 4096, 16, 2, 64, 64, 8, 64, 256),
        (4, 1, 2048, 16, 2, 128, 128, 8, 64, 256),
        (4, 4, 2048, 16, 2, 128, 128, 8, 64, 256),
        (4, 16, 2048, 16, 2, 128, 128, 8, 64, 256),
        (8, 16, 4096, 16, 2, 128, 128, 8, 64, 256),
    ]
    warmup = 3 if smoke else 5
    rep = 6 if smoke else 15
    rows = []

    for B, T_q, T_kv, HQ, H, K, V, HI, DI, topk in shapes:
        torch.manual_seed(3000 + B * 10 + T_q + T_kv + K)
        q_full = torch.randn(B, T_kv, HQ, K, device=device, dtype=torch.bfloat16)
        k_cache = torch.randn(B, T_kv, H, K, device=device, dtype=torch.bfloat16)
        v_cache = torch.randn(B, T_kv, H, V, device=device, dtype=torch.bfloat16)
        qi_full = torch.randn(B, T_kv, HI, DI, device=device, dtype=torch.bfloat16)
        ki_cache = torch.randn(B, T_kv, DI, device=device, dtype=torch.bfloat16)
        wi_full = torch.rand(B, T_kv, HI, device=device, dtype=torch.bfloat16) + 0.1

        q_dec = q_full[:, -T_q:].contiguous()
        qi_dec = qi_full[:, -T_q:].contiguous()
        wi_dec = wi_full[:, -T_q:].contiguous()

        with torch.no_grad():
            o_ref_full = naive_dsa(q_full, k_cache, v_cache, qi_full, ki_cache, wi_full, topk=topk)
            o_ref = o_ref_full[:, -T_q:]
            o_dec = dsa_decode_rsi(q_dec, k_cache, v_cache, qi_dec, ki_cache, wi_dec, topk=topk)
            o_err = _rel_l2(o_dec, o_ref)

            # Time upstream naive_dsa on full sequence (only interface upstream provides)
            for _ in range(2):
                naive_dsa(q_full, k_cache, v_cache, qi_full, ki_cache, wi_full, topk=topk)
            torch.cuda.synchronize()
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(max(2, rep // 3)):
                naive_dsa(q_full, k_cache, v_cache, qi_full, ki_cache, wi_full, topk=topk)
            ev1.record()
            torch.cuda.synchronize()
            up_ms = ev0.elapsed_time(ev1) / max(2, rep // 3)

            # Time chunk dsa_rsi on full sequence
            for _ in range(warmup):
                dsa_rsi(q_full, k_cache, v_cache, qi_full, ki_cache, wi_full, topk=topk, train_indexer=False)
            torch.cuda.synchronize()
            ev0.record()
            for _ in range(rep):
                dsa_rsi(q_full, k_cache, v_cache, qi_full, ki_cache, wi_full, topk=topk, train_indexer=False)
            ev1.record()
            torch.cuda.synchronize()
            chunk_rsi_ms = ev0.elapsed_time(ev1) / rep

            # Time single-pass dsa_decode_rsi
            for _ in range(warmup):
                dsa_decode_rsi(q_dec, k_cache, v_cache, qi_dec, ki_cache, wi_dec, topk=topk)
            torch.cuda.synchronize()
            ev0.record()
            for _ in range(rep):
                dsa_decode_rsi(q_dec, k_cache, v_cache, qi_dec, ki_cache, wi_dec, topk=topk)
            ev1.record()
            torch.cuda.synchronize()
            dec_ms = ev0.elapsed_time(ev1) / rep

        rows.append(
            {
                "shape": f"B={B}_Tq={T_q}_Tkv={T_kv}_K={K}",
                "B": B,
                "T_q": T_q,
                "T_kv": T_kv,
                "HQ": HQ,
                "H": H,
                "K": K,
                "V": V,
                "topk": topk,
                "upstream_naive_ms": up_ms,
                "rsi_chunk_ms": chunk_rsi_ms,
                "rsi_decode_ms": dec_ms,
                "speedup_vs_upstream_naive": up_ms / dec_ms,
                "speedup_vs_rsi_chunk": chunk_rsi_ms / dec_ms,
                "o_rel_l2": o_err,
            }
        )
    return {"decode_matrix": rows}


def run_track3_layer_ablation(device: torch.device, smoke: bool = False) -> dict:
    configs = [
        (1024, 16, 2, 64),
        (2048, 16, 2, 128),
    ]
    B, T, topk = 2, 1024, 256
    warmup = 2 if smoke else 4
    rep = 4 if smoke else 10
    rows = []

    for d_model, n_heads, n_kv_heads, head_dim in configs:
        torch.manual_seed(4000 + d_model)
        x = torch.randn(B, T, d_model, device=device, dtype=torch.bfloat16, requires_grad=True)
        do = torch.randn(B, T, d_model, device=device, dtype=torch.bfloat16)

        # Build a 2-layer stack so Stage A4 demonstrates GLM-5.2 IndexShare across layers
        ref_blk1 = (
            DeepSeekSparseAttentionBlockRSI(
                d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim, topk=topk, stage="A0"
            )
            .to(device)
            .to(torch.bfloat16)
        )
        ref_blk2 = (
            DeepSeekSparseAttentionBlockRSI(
                d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim, topk=topk, stage="A0"
            )
            .to(device)
            .to(torch.bfloat16)
        )
        ref_blk1.sync_packed_weights()
        ref_blk2.sync_packed_weights()
        s1_dict = {k: v.detach().clone() for k, v in ref_blk1.state_dict().items()}
        s2_dict = {k: v.detach().clone() for k, v in ref_blk2.state_dict().items()}

        a0_total_ms = None
        y_a0 = None

        for stg in ["A0", "A1", "A2", "A3", "A4"]:
            b1 = (
                DeepSeekSparseAttentionBlockRSI(
                    d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim, topk=topk, stage=stg
                )
                .to(device)
                .to(torch.bfloat16)
            )
            b2 = (
                DeepSeekSparseAttentionBlockRSI(
                    d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim, topk=topk, stage=stg
                )
                .to(device)
                .to(torch.bfloat16)
            )
            b1.load_state_dict(s1_dict)
            b2.load_state_dict(s2_dict)
            b1.sync_packed_weights()
            b2.sync_packed_weights()

            def _run_stack():
                if stg == "A4":
                    h1, kl1, shared = b1(x, return_shared_state=True)
                    h2, kl2 = b2(h1, shared_state=shared)
                    return h2, kl1 + kl2
                else:
                    h1, kl1 = b1(x)
                    h2, kl2 = b2(h1)
                    return h2, kl1 + kl2

            with torch.no_grad():
                y_cur, _ = _run_stack()
                if stg == "A0":
                    y_a0 = y_cur.detach().clone()
                    rel_err = 0.0
                else:
                    rel_err = _rel_l2(y_cur, y_a0)

            all_params = [x] + list(b1.parameters()) + list(b2.parameters())
            f_ms, b_ms, _ = _time_fwd_bwd(_run_stack, do, all_params, warmup=warmup, rep=rep)
            tot_ms = f_ms + b_ms
            if stg == "A0":
                a0_total_ms = tot_ms
            tok_s = (B * T) / (tot_ms * 1e-3)

            rows.append(
                {
                    "D_model": d_model,
                    "HQ": n_heads,
                    "H": n_kv_heads,
                    "K": head_dim,
                    "B": B,
                    "T": T,
                    "topk": topk,
                    "stage": stg,
                    "fwd_ms": f_ms,
                    "bwd_ms": b_ms,
                    "total_ms": tot_ms,
                    "tokens_per_s": tok_s,
                    "speedup_vs_a0": a0_total_ms / tot_ms,
                    "rel_l2_vs_a0": rel_err,
                }
            )
    return {"layer_ablation": rows}


def _cp8_worker(rank: int, world_size: int, port: int, smoke: bool, return_dict):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)

    configs = [
        (2, 4096, 32, 8, 64, 64, 8, 64, 256),
        (2, 8192, 32, 8, 64, 64, 8, 64, 256),
        (2, 4096, 32, 8, 128, 128, 8, 64, 256),
    ]
    warmup = 2 if smoke else 4
    rep = 4 if smoke else 8
    out_rows = []

    for B, T_total, HQ, H, K, V, HI, DI, topk in configs:
        T_local = T_total // world_size
        torch.manual_seed(4500 + T_total + K)
        q_full = F.normalize(torch.randn(B, T_total, HQ, K, device=device, dtype=torch.bfloat16), p=2, dim=-1)
        k_full = F.normalize(torch.randn(B, T_total, H, K, device=device, dtype=torch.bfloat16), p=2, dim=-1)
        v_full = torch.randn(B, T_total, H, V, device=device, dtype=torch.bfloat16)
        qi_full = F.normalize(torch.randn(B, T_total, HI, DI, device=device, dtype=torch.bfloat16), p=2, dim=-1)
        ki_full = F.normalize(torch.randn(B, T_total, DI, device=device, dtype=torch.bfloat16), p=2, dim=-1)
        wi_full = torch.rand(B, T_total, HI, device=device, dtype=torch.bfloat16) + 0.1
        do_full = torch.randn(B, T_total, HQ, V, device=device, dtype=torch.bfloat16)

        # 1-GPU references on rank 0
        one_gpu_up_ms = 0.0
        one_gpu_rsi_ms = 0.0
        if rank == 0:
            q1 = q_full.detach().clone().requires_grad_(True)
            k1 = k_full.detach().clone().requires_grad_(True)
            v1 = v_full.detach().clone().requires_grad_(True)
            qi1 = qi_full.detach().clone().requires_grad_(True)
            ki1 = ki_full.detach().clone().requires_grad_(True)
            wi1 = wi_full.detach().clone().requires_grad_(True)

            f_u, b_u, _ = _time_fwd_bwd(
                lambda: naive_dsa(q1, k1, v1, qi1, ki1, wi1, topk=topk),
                do_full,
                [q1, k1, v1],
                warmup=1,
                rep=2,
            )
            one_gpu_up_ms = f_u + b_u

            f_r, b_r, _ = _time_fwd_bwd(
                lambda: dsa_rsi(q1, k1, v1, qi1, ki1, wi1, topk=topk, train_indexer=True, return_kl_loss=True),
                do_full,
                [q1, k1, v1, qi1, ki1, wi1],
                warmup=warmup,
                rep=rep,
            )
            one_gpu_rsi_ms = f_r + b_r

            q1.grad = k1.grad = v1.grad = qi1.grad = ki1.grad = wi1.grad = None
            o1_ref, kl1_ref = dsa_rsi(q1, k1, v1, qi1, ki1, wi1, topk=topk, train_indexer=True, return_kl_loss=True)
            ((o1_ref.float() * do_full.float()).sum() + 0.1 * kl1_ref).backward()
            o1_slice = o1_ref[:, :T_local].detach().clone()
            dq1_slice = q1.grad[:, :T_local].detach().clone()
            dk1_slice = k1.grad[:, :T_local].detach().clone()
            dv1_slice = v1.grad[:, :T_local].detach().clone()

        dist.barrier()
        s0 = rank * T_local
        s1 = (rank + 1) * T_local
        q_loc = q_full[:, s0:s1].detach().clone().requires_grad_(True)
        k_loc = k_full[:, s0:s1].detach().clone().requires_grad_(True)
        v_loc = v_full[:, s0:s1].detach().clone().requires_grad_(True)
        qi_loc = qi_full[:, s0:s1].detach().clone().requires_grad_(True)
        ki_loc = ki_full[:, s0:s1].detach().clone().requires_grad_(True)
        wi_loc = wi_full[:, s0:s1].detach().clone().requires_grad_(True)
        do_loc = do_full[:, s0:s1].detach().clone()

        # Parity check for Ulysses CP8
        o_uly, kl_uly = dsa_cp8_ulysses_rsi(
            q_loc, k_loc, v_loc, qi_loc, ki_loc, wi_loc, topk=topk, group=dist.group.WORLD
        )
        ((o_uly.float() * do_loc.float()).sum() + (0.1 / world_size) * kl_uly).backward()
        if rank == 0:
            err_o = _rel_l2(o_uly, o1_slice)
            err_dq = _rel_l2(q_loc.grad, dq1_slice)
            err_dk = _rel_l2(k_loc.grad, dk1_slice)
            err_dv = _rel_l2(v_loc.grad, dv1_slice)
        else:
            err_o = err_dq = err_dk = err_dv = 0.0

        # Time Ring/AllGather CP8 baseline
        for _ in range(1):
            q_loc.grad = k_loc.grad = v_loc.grad = None
            o_ring = dsa_cp8_ring_baseline(
                q_loc, k_loc, v_loc, qi_loc, ki_loc, wi_loc, topk=topk, group=dist.group.WORLD
            )
            (o_ring.float() * do_loc.float()).sum().backward()
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ring_reps = max(2, rep // 2)
        for _ in range(ring_reps):
            q_loc.grad = k_loc.grad = v_loc.grad = None
            o_ring = dsa_cp8_ring_baseline(
                q_loc, k_loc, v_loc, qi_loc, ki_loc, wi_loc, topk=topk, group=dist.group.WORLD
            )
            (o_ring.float() * do_loc.float()).sum().backward()
            torch.cuda.synchronize()
        ring_ms = (time.perf_counter() - t0) * 1000.0 / ring_reps

        # Time Ulysses CP8 RSI
        for _ in range(warmup):
            q_loc.grad = k_loc.grad = v_loc.grad = qi_loc.grad = ki_loc.grad = wi_loc.grad = None
            o_u, kl_u = dsa_cp8_ulysses_rsi(
                q_loc, k_loc, v_loc, qi_loc, ki_loc, wi_loc, topk=topk, group=dist.group.WORLD
            )
            ((o_u.float() * do_loc.float()).sum() + (0.1 / world_size) * kl_u).backward()
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(rep):
            q_loc.grad = k_loc.grad = v_loc.grad = qi_loc.grad = ki_loc.grad = wi_loc.grad = None
            o_u, kl_u = dsa_cp8_ulysses_rsi(
                q_loc, k_loc, v_loc, qi_loc, ki_loc, wi_loc, topk=topk, group=dist.group.WORLD
            )
            ((o_u.float() * do_loc.float()).sum() + (0.1 / world_size) * kl_u).backward()
            torch.cuda.synchronize()
        uly_ms = (time.perf_counter() - t0) * 1000.0 / rep

        if rank == 0:
            out_rows.append(
                {
                    "B": B,
                    "T_total": T_total,
                    "T_local": T_local,
                    "HQ": HQ,
                    "H": H,
                    "K": K,
                    "V": V,
                    "topk": topk,
                    "one_gpu_upstream_ms": one_gpu_up_ms,
                    "one_gpu_rsi_ms": one_gpu_rsi_ms,
                    "ring_cp8_ms": ring_ms,
                    "ulysses_cp8_ms": uly_ms,
                    "speedup_vs_ring": ring_ms / uly_ms,
                    "speedup_vs_1gpu_upstream": one_gpu_up_ms / uly_ms,
                    "speedup_vs_1gpu_rsi": one_gpu_rsi_ms / uly_ms,
                    "parallel_efficiency_vs_1gpu_upstream_pct": (one_gpu_up_ms / (uly_ms * world_size)) * 100.0,
                    "ulysses_tokens_per_s": (B * T_total) / (uly_ms * 1e-3),
                    "rel_l2_o": err_o,
                    "rel_l2_dq": err_dq,
                    "rel_l2_dk": err_dk,
                    "rel_l2_dv": err_dv,
                }
            )
    dist.destroy_process_group()
    if rank == 0:
        return_dict["cp8_scaling"] = out_rows


def run_track4_cp8(smoke: bool = False) -> dict:
    world_size = min(8, torch.cuda.device_count())
    assert world_size == 8, f"Expected 8 GPUs, found {world_size}"
    # Pre-warm HQ=32, H=8 and HQ=4, H=1 specializations on cuda:0 before mp.spawn
    dev0 = torch.device("cuda:0")
    for K in [64, 128]:
        for HQ_w, H_w in [(32, 8), (4, 1)]:
            qw = torch.randn(2, 128, HQ_w, K, device=dev0, dtype=torch.bfloat16, requires_grad=True)
            kw = torch.randn(2, 128, H_w, K, device=dev0, dtype=torch.bfloat16, requires_grad=True)
            vw = torch.randn(2, 128, H_w, K, device=dev0, dtype=torch.bfloat16, requires_grad=True)
            qiw = torch.randn(2, 128, 8, 64, device=dev0, dtype=torch.bfloat16, requires_grad=True)
            kiw = torch.randn(2, 128, 64, device=dev0, dtype=torch.bfloat16, requires_grad=True)
            wiw = (torch.rand(2, 128, 8, device=dev0, dtype=torch.bfloat16) + 0.1).requires_grad_(True)
            ow, klw = dsa_rsi(qw, kw, vw, qiw, kiw, wiw, topk=64, train_indexer=True, return_kl_loss=True)
            (ow.sum() + klw).backward()
    torch.cuda.synchronize()

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()
    port = 29641
    mp.spawn(_cp8_worker, args=(world_size, port, smoke, return_dict), nprocs=world_size, join=True)
    return {"cp8_scaling": list(return_dict["cp8_scaling"])}


class TinyDSALM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        topk: int,
        stage: str,
    ):
        super().__init__()
        self.stage = stage
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(1024, d_model)
        self.blk1 = DeepSeekSparseAttentionBlockRSI(
            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim, topk=topk, stage=stage
        )
        self.blk2 = DeepSeekSparseAttentionBlockRSI(
            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim, topk=topk, stage=stage
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor):
        pos = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        h = self.embed(tokens) + self.pos_embed(pos)
        if self.stage == "A4":
            h, kl1, shared = self.blk1(h, return_shared_state=True)
            h, kl2 = self.blk2(h, shared_state=shared)
            return self.lm_head(h), kl1 + kl2
        else:
            h, kl1 = self.blk1(h)
            h, kl2 = self.blk2(h)
            return self.lm_head(h), kl1 + kl2


def run_track5_autoresearch(device: torch.device, smoke: bool = False) -> dict:
    budget_s = 6.0 if smoke else 12.0
    vocab_size = 256
    B, T, topk = 4, 512, 128
    idx_t = torch.arange(T + 1, device=device)

    def _sample_batch(gen: torch.Generator) -> torch.Tensor:
        base = torch.randint(0, vocab_size, (B, 32), device=device, generator=gen)
        seq = (base[:, idx_t % 32] + (idx_t.unsqueeze(0) // 32) * 31) % vocab_size
        seq[:, 3:] = (seq[:, 3:] ^ (seq[:, :-3] & 0x3F)) % vocab_size
        return seq

    rows = []
    for d_model, n_heads, n_kv_heads, head_dim in [(1024, 16, 2, 64), (2048, 16, 2, 128)]:
        g_val = torch.Generator(device=device).manual_seed(5000 + d_model)
        val_batches = [_sample_batch(g_val) for _ in range(8)]

        @torch.no_grad()
        def _eval_bpb(m: nn.Module) -> float:
            tot_loss = 0.0
            for vb in val_batches:
                logits, _ = m(vb[:, :-1])
                tot_loss += float(
                    F.cross_entropy(logits.float().reshape(-1, vocab_size), vb[:, 1:].reshape(-1)).item()
                )
            return (tot_loss / len(val_batches)) / math.log(2.0)

        torch.manual_seed(700 + d_model)
        init_model = (
            TinyDSALM(vocab_size, d_model, n_heads, n_kv_heads, head_dim, topk, stage="A0")
            .to(device)
            .to(torch.bfloat16)
        )
        with torch.no_grad():
            nn.init.normal_(init_model.embed.weight, std=0.02)
            nn.init.normal_(init_model.pos_embed.weight, std=0.02)
            nn.init.normal_(init_model.lm_head.weight, std=0.02)
            for blk in [init_model.blk1, init_model.blk2]:
                for p in [
                    blk.q_proj.weight,
                    blk.k_proj.weight,
                    blk.v_proj.weight,
                    blk.qi_proj.weight,
                    blk.ki_proj.weight,
                    blk.wi_proj.weight,
                    blk.g_proj.weight,
                    blk.o_proj.weight,
                ]:
                    nn.init.normal_(p, std=0.02)
                blk.sync_packed_weights()
        init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}

        res_pair = {}
        base_lr = 8e-4 if d_model == 1024 else 6e-4
        for stg in ["A0", "A4"]:
            model = (
                TinyDSALM(vocab_size, d_model, n_heads, n_kv_heads, head_dim, topk, stage=stg)
                .to(device)
                .to(torch.bfloat16)
            )
            model.load_state_dict(init_state)
            model.blk1.sync_packed_weights()
            model.blk2.sync_packed_weights()
            if stg == "A0":
                params = [p for n, p in model.named_parameters() if ".packed_proj." not in n]
            else:
                params = [
                    p
                    for n, p in model.named_parameters()
                    if not any(
                        x in n
                        for x in (
                            ".q_proj.",
                            ".k_proj.",
                            ".v_proj.",
                            ".qi_proj.",
                            ".ki_proj.",
                            ".wi_proj.",
                            ".g_proj.",
                        )
                    )
                ]
            opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=0.01)

            g_warm = torch.Generator(device=device).manual_seed(5200 + d_model)
            for _ in range(2):
                wb = _sample_batch(g_warm)
                opt.zero_grad(set_to_none=True)
                logits, kl_l = model(wb[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, vocab_size).float(), wb[:, 1:].reshape(-1)) + 0.05 * kl_l
                loss.backward()
            model.load_state_dict(init_state)
            model.blk1.sync_packed_weights()
            model.blk2.sync_packed_weights()
            opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=0.01)

            init_bpb = _eval_bpb(model)
            g_train = torch.Generator(device=device).manual_seed(6000 + d_model)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            steps = 0
            while (time.perf_counter() - t0) < budget_s:
                tb = _sample_batch(g_train)
                opt.zero_grad(set_to_none=True)
                logits, kl_l = model(tb[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, vocab_size).float(), tb[:, 1:].reshape(-1)) + 0.05 * kl_l
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                torch.cuda.synchronize()
                steps += 1
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            final_bpb = _eval_bpb(model)

            res_pair[stg] = {
                "steps": steps,
                "elapsed_s": elapsed,
                "tokens_per_s": (steps * B * T) / elapsed,
                "init_val_bpb": init_bpb,
                "final_val_bpb": final_bpb,
            }

        rows.append(
            {
                "D_model": d_model,
                "HQ": n_heads,
                "H": n_kv_heads,
                "K": head_dim,
                "topk": topk,
                "budget_s": budget_s,
                "a0_steps": res_pair["A0"]["steps"],
                "a4_steps": res_pair["A4"]["steps"],
                "step_ratio_a4_vs_a0": res_pair["A4"]["steps"] / max(res_pair["A0"]["steps"], 1),
                "a0_tokens_per_s": res_pair["A0"]["tokens_per_s"],
                "a4_tokens_per_s": res_pair["A4"]["tokens_per_s"],
                "init_val_bpb": res_pair["A0"]["init_val_bpb"],
                "a0_final_val_bpb": res_pair["A0"]["final_val_bpb"],
                "a4_final_val_bpb": res_pair["A4"]["final_val_bpb"],
                "extra_val_bpb_reduction": res_pair["A0"]["final_val_bpb"] - res_pair["A4"]["final_val_bpb"],
            }
        )
    return {"autoresearch": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    harness_info = verify_frozen_harness()
    oracle_info = run_oracle_and_mutations(device)
    t1 = run_track1_kernel_matrix(device, smoke=args.smoke)
    t2 = run_track2_decode_matrix(device, smoke=args.smoke)
    t3 = run_track3_layer_ablation(device, smoke=args.smoke)
    t4 = run_track4_cp8(smoke=args.smoke)
    t5 = run_track5_autoresearch(device, smoke=args.smoke)

    cubin_audit = audit_dsa_rsi_cubin_kernels()
    assert cubin_audit["all_zero_spill"], f"Cubin spill or >168 reg detected: {json.dumps(cubin_audit, indent=2)}"

    payload = {
        "experiment_id": "E157",
        "version": 148,
        "round": 31,
        "title": "DeepSeek Sparse Attention (DSA), Online KL-Distilled Lightning Indexer, & GLM-5.2 IndexShare (arXiv:2512.02556 / arXiv:2602.15763)",
        "smoke": args.smoke,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "frozen_harness": harness_info,
        "oracle_and_mutations": oracle_info,
        "rsi_cubin_audit": cubin_audit,
        "track1": t1,
        "track2": t2,
        "track3": t3,
        "track4": t4,
        "track5": t5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[E157/v148] Wrote {args.output} (all_zero_spill={cubin_audit['all_zero_spill']})")


if __name__ == "__main__":
    main()
