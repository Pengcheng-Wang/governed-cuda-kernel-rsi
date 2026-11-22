from __future__ import annotations

import argparse
import atexit
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from flash_attn import flash_attn_func, flash_attn_varlen_func
import pynvml
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.profiler import ProfilerActivity, profile

from common import environment_metadata, write_result


class GpuTelemetry:
    def __init__(self) -> None:
        pynvml.nvmlInit()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        nvml_index = int(visible) if visible.isdigit() else 0
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
        self.nvml_index = nvml_index

    def max_clocks(self) -> dict[str, int]:
        return {
            "default_app_sm_clock_mhz": int(
                pynvml.nvmlDeviceGetDefaultApplicationsClock(
                    self.handle,
                    pynvml.NVML_CLOCK_SM,
                )
            ),
            "max_sm_clock_mhz": int(
                pynvml.nvmlDeviceGetMaxClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
            ),
            "max_mem_clock_mhz": int(
                pynvml.nvmlDeviceGetMaxClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            ),
        }

    def lock_sm_clock(self, mhz: int) -> None:
        torch.cuda.synchronize()
        pynvml.nvmlDeviceSetGpuLockedClocks(self.handle, int(mhz), int(mhz))
        time.sleep(0.05)

    def reset_clocks(self) -> None:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(self.handle)
        except Exception:
            pass

    def sample(self) -> dict[str, float | int]:
        return {
            "sm_clock_mhz": int(
                pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
            ),
            "mem_clock_mhz": int(
                pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            ),
            "power_w": round(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0, 2),
            "temp_c": int(
                pynvml.nvmlDeviceGetTemperature(
                    self.handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
            ),
            "pstate": int(pynvml.nvmlDeviceGetPerformanceState(self.handle)),
        }


def benchmark_with_telemetry(
    fn: Callable[[], Any],
    telemetry: GpuTelemetry,
    *,
    warmup: int,
    groups: int,
    iterations_per_group: int,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    pre_warmup = telemetry.sample()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    post_warmup = telemetry.sample()

    samples_ms: list[float] = []
    clocks_before: list[int] = []
    clocks_after: list[int] = []
    powers_w: list[float] = []
    temps_c: list[int] = []

    for _ in range(groups):
        before = telemetry.sample()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations_per_group):
            fn()
        end.record()
        end.synchronize()
        after = telemetry.sample()

        samples_ms.append(start.elapsed_time(end) / iterations_per_group)
        clocks_before.append(int(before["sm_clock_mhz"]))
        clocks_after.append(int(after["sm_clock_mhz"]))
        powers_w.append(0.5 * (float(before["power_w"]) + float(after["power_w"])))
        temps_c.append(max(int(before["temp_c"]), int(after["temp_c"])))

    all_clocks = clocks_before + clocks_after
    return {
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "stdev_ms": statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "groups": groups,
        "iterations_per_group": iterations_per_group,
        "warmup": warmup,
        "group_samples_ms": [round(val, 6) for val in samples_ms],
        "pre_warmup_sm_clock_mhz": int(pre_warmup["sm_clock_mhz"]),
        "post_warmup_sm_clock_mhz": int(post_warmup["sm_clock_mhz"]),
        "group_sm_clocks_before_mhz": clocks_before,
        "group_sm_clocks_after_mhz": clocks_after,
        "median_sm_clock_mhz": float(statistics.median(clocks_after)),
        "min_sm_clock_mhz": min(all_clocks),
        "max_sm_clock_mhz": max(all_clocks),
        "mean_power_w": round(statistics.fmean(powers_w), 2),
        "max_temp_c": max(temps_c),
    }


def merge_two_passes(
    pass_a: dict[str, Any],
    pass_b: dict[str, Any],
) -> dict[str, Any]:
    samples = list(pass_a["group_samples_ms"]) + list(pass_b["group_samples_ms"])
    clocks_before = (
        list(pass_a["group_sm_clocks_before_mhz"])
        + list(pass_b["group_sm_clocks_before_mhz"])
    )
    clocks_after = (
        list(pass_a["group_sm_clocks_after_mhz"])
        + list(pass_b["group_sm_clocks_after_mhz"])
    )
    all_clocks = clocks_before + clocks_after
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "mean_ms": statistics.fmean(samples),
        "stdev_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "groups": int(pass_a["groups"]) + int(pass_b["groups"]),
        "iterations_per_group": int(pass_a["iterations_per_group"]),
        "warmup": int(pass_a["warmup"]) + int(pass_b["warmup"]),
        "group_samples_ms": [round(val, 6) for val in samples],
        "pre_warmup_sm_clock_mhz": min(
            int(pass_a["pre_warmup_sm_clock_mhz"]),
            int(pass_b["pre_warmup_sm_clock_mhz"]),
        ),
        "post_warmup_sm_clock_mhz": min(
            int(pass_a["post_warmup_sm_clock_mhz"]),
            int(pass_b["post_warmup_sm_clock_mhz"]),
        ),
        "group_sm_clocks_before_mhz": clocks_before,
        "group_sm_clocks_after_mhz": clocks_after,
        "median_sm_clock_mhz": float(statistics.median(clocks_after)),
        "min_sm_clock_mhz": min(all_clocks),
        "max_sm_clock_mhz": max(all_clocks),
        "mean_power_w": round(
            0.5 * (float(pass_a["mean_power_w"]) + float(pass_b["mean_power_w"])),
            2,
        ),
        "max_temp_c": max(int(pass_a["max_temp_c"]), int(pass_b["max_temp_c"])),
    }


def measure_attention_memory(
    step_fn: Callable[[], Any],
    params: tuple[torch.Tensor, ...],
) -> dict[str, int]:
    for p in params:
        p.grad = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    pre_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step_fn()
    torch.cuda.synchronize()
    post_bytes = torch.cuda.memory_allocated()
    peak_bytes = torch.cuda.max_memory_allocated()
    for p in params:
        p.grad = None
    return {
        "peak_incremental_memory_bytes": max(0, peak_bytes - pre_bytes),
        "persistent_grad_bytes": max(0, post_bytes - pre_bytes),
        "peak_transient_workspace_bytes": max(0, peak_bytes - post_bytes),
    }


def compute_fp32_attention_reference(
    q_bhsd: torch.Tensor,
    k_bhsd: torch.Tensor,
    v_bhsd: torch.Tensor,
    *,
    mask_2d: torch.Tensor,
    softcap: float = 0.0,
    log_decay_slopes: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Exact FP32 reference evaluated over query chunks to avoid unnecessary FP32 peak memory
    b, h_q, s, d = q_bhsd.shape
    h_kv = k_bhsd.shape[1]
    group_size = h_q // h_kv
    scale = d**-0.5

    qf = q_bhsd.detach().float().requires_grad_(True)
    kf = k_bhsd.detach().float().requires_grad_(True)
    vf = v_bhsd.detach().float().requires_grad_(True)

    k_rep = kf.repeat_interleave(group_size, dim=1)
    v_rep = vf.repeat_interleave(group_size, dim=1)

    idx = torch.arange(s, device=q_bhsd.device)
    q_chunk = 512
    outputs: list[torch.Tensor] = []
    for start in range(0, s, q_chunk):
        end = min(start + q_chunk, s)
        qc = qf[:, :, start:end, :]
        scores = torch.matmul(qc, k_rep.transpose(-2, -1)) * scale
        if softcap > 0.0:
            scores = softcap * torch.tanh(scores / softcap)
        if log_decay_slopes is not None:
            dist = (idx[start:end, None] - idx[None, :]).clamp_min(0).to(torch.float32)
            scores = scores - log_decay_slopes[None, :, None, None] * torch.log2(
                1.0 + dist
            )[None, None, :, :]
        scores = scores.masked_fill(~mask_2d[None, None, start:end, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probs, v_rep))

    out_full = torch.cat(outputs, dim=2)
    loss = out_full.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert qf.grad is not None and kf.grad is not None and vf.grad is not None
    return (
        float(loss.detach().item()),
        out_full.detach().clone(),
        qf.grad.detach().clone(),
        kf.grad.detach().clone(),
        vf.grad.detach().clone(),
    )


def evaluate_attention_correctness(
    forward_bhsd_fn: Callable[[], torch.Tensor],
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grads_to_bhsd_fn: Callable[
        [tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    ref_loss: float,
    ref_out: torch.Tensor,
    ref_dq: torch.Tensor,
    ref_dk: torch.Tensor,
    ref_dv: torch.Tensor,
) -> dict[str, float]:
    for p in params:
        p.grad = None
    out = forward_bhsd_fn()
    loss = out.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()

    loss_val = float(loss.detach().item())
    out_f = out.detach().float()
    dq_f, dk_f, dv_f = grads_to_bhsd_fn(params)

    out_max_abs = float((out_f - ref_out).abs().max().item())
    out_l2_rel = float((out_f - ref_out).norm().item()) / max(
        float(ref_out.norm().item()),
        1e-12,
    )
    dq_rel = float((dq_f - ref_dq).norm().item()) / max(
        float(ref_dq.norm().item()),
        1e-12,
    )
    dk_rel = float((dk_f - ref_dk).norm().item()) / max(
        float(ref_dk.norm().item()),
        1e-12,
    )
    dv_rel = float((dv_f - ref_dv).norm().item()) / max(
        float(ref_dv.norm().item()),
        1e-12,
    )

    # Verify upstream scalar chain-rule scaling (alpha = 1.75)
    scale_alpha = 1.75
    for p in params:
        p.grad = None
    scaled_loss = forward_bhsd_fn().float().square().mean() * scale_alpha
    scaled_loss.backward()
    torch.cuda.synchronize()
    dq_s, dk_s, dv_s = grads_to_bhsd_fn(params)
    chain_err = max(
        float((dq_s - dq_f * scale_alpha).norm().item())
        / max(float((dq_f * scale_alpha).norm().item()), 1e-12),
        float((dk_s - dk_f * scale_alpha).norm().item())
        / max(float((dk_f * scale_alpha).norm().item()), 1e-12),
        float((dv_s - dv_f * scale_alpha).norm().item())
        / max(float((dv_f * scale_alpha).norm().item()), 1e-12),
    )

    for p in params:
        p.grad = None
    del out_f, dq_f, dk_f, dv_f, dq_s, dk_s, dv_s
    return {
        "loss": loss_val,
        "loss_abs_error_vs_fp32": abs(loss_val - ref_loss),
        "out_max_abs_error_vs_fp32": out_max_abs,
        "out_l2_rel_error_vs_fp32": out_l2_rel,
        "dq_l2_rel_error_vs_fp32": dq_rel,
        "dk_l2_rel_error_vs_fp32": dk_rel,
        "dv_l2_rel_error_vs_fp32": dv_rel,
        "scaled_backward_chain_rule_rel_error": chain_err,
    }


def classify_attention_kernel(name: str) -> str:
    lowered = name.lower()
    if "triton_flex_attention" in lowered or "flex_attention" in lowered:
        return "flex_triton_attention"
    if "pytorch_flash" in lowered or "flash_fwd" in lowered or "flash_bwd" in lowered:
        return "flash_cuda_attention"
    if "fmha" in lowered or "cutlass" in lowered:
        return "cutlass_fmha_attention"
    if any(tok in lowered for tok in ("gemm", "nvjet", "cublas", "bmm")):
        return "materialized_gemm"
    return "mask_softmax_pointwise"


def build_doc_packing_regimes(
    profile_name: str,
) -> tuple[tuple[int, int, int, int, int], list[tuple[str, list[int]]]]:
    if profile_name == "smoke":
        shape = (1, 512, 16, 4, 128)
        regimes = [
            ("single_doc_512", [512]),
            ("packed_4docs_mixed", [160, 100, 140, 112]),
        ]
    else:
        shape = (1, 8192, 32, 8, 128)
        regimes = [
            ("single_doc_8192", [8192]),
            ("packed_4docs_long", [2688, 2176, 1920, 1408]),
            (
                "packed_16docs_mixed",
                [
                    700,
                    340,
                    612,
                    420,
                    580,
                    396,
                    650,
                    470,
                    530,
                    382,
                    610,
                    442,
                    560,
                    410,
                    590,
                    500,
                ],
            ),
            ("packed_64docs_sft", [96, 112, 144, 160] * 16),
        ]
    for name, lengths in regimes:
        if sum(lengths) != shape[1]:
            raise ValueError(f"Regime {name} sum {sum(lengths)} != {shape[1]}")
    return shape, regimes


def run_packed_document_gqa_study(
    profile_name: str,
    device: torch.device,
    dtype: torch.dtype,
    telemetry: GpuTelemetry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shape, regimes = build_doc_packing_regimes(profile_name)
    batch, seq_len, h_q, h_kv, head_dim = shape
    group_size = h_q // h_kv
    warmup, groups, iters = (2, 2, 2) if profile_name == "smoke" else (6, 4, 6)
    prof_steps = 2 if profile_name == "smoke" else 6
    profiled_regime = (
        "packed_4docs_mixed" if profile_name == "smoke" else "packed_16docs_mixed"
    )

    boost_mhz = telemetry.max_clocks()["max_sm_clock_mhz"]
    doc_records: list[dict[str, Any]] = []
    mask_build_records: list[dict[str, Any]] = []
    kernel_records: list[dict[str, Any]] = []

    old_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    old_limit = torch._dynamo.config.cache_size_limit
    try:
        telemetry.lock_sm_clock(boost_mhz)
        torch._dynamo.config.cache_size_limit = 64
        with tempfile.TemporaryDirectory(prefix="e113_packed_doc_") as tmpdir:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(tmpdir) / "cache")
            torch._dynamo.reset()

            torch.manual_seed(53)
            q_bhsd = torch.nn.Parameter(
                torch.randn((batch, h_q, seq_len, head_dim), device=device, dtype=dtype)
            )
            k_bhsd = torch.nn.Parameter(
                torch.randn(
                    (batch, h_kv, seq_len, head_dim), device=device, dtype=dtype
                )
            )
            v_bhsd = torch.nn.Parameter(
                torch.randn(
                    (batch, h_kv, seq_len, head_dim), device=device, dtype=dtype
                )
            )

            # Matching [total_tokens, H, D] leaf tensors for flash_attn_varlen_func
            q_thd = torch.nn.Parameter(
                q_bhsd.detach()
                .transpose(1, 2)
                .contiguous()
                .view(batch * seq_len, h_q, head_dim)
            )
            k_thd = torch.nn.Parameter(
                k_bhsd.detach()
                .transpose(1, 2)
                .contiguous()
                .view(batch * seq_len, h_kv, head_dim)
            )
            v_thd = torch.nn.Parameter(
                v_bhsd.detach()
                .transpose(1, 2)
                .contiguous()
                .view(batch * seq_len, h_kv, head_dim)
            )

            compiled_flex_default = torch.compile(
                flex_attention,
                fullgraph=True,
                mode="default",
            )
            compiled_flex_autotune = torch.compile(
                flex_attention,
                fullgraph=True,
                mode="max-autotune-no-cudagraphs",
            )
            compiled_cbm = torch.compile(create_block_mask)

            idx = torch.arange(seq_len, device=device)

            for regime_name, doc_lengths in regimes:
                num_docs = len(doc_lengths)
                max_doc_len = max(doc_lengths)
                mean_doc_len = seq_len / num_docs
                active_causal_pairs = sum(L * (L + 1) // 2 for L in doc_lengths)
                active_pair_fraction = active_causal_pairs / float(seq_len * seq_len)
                # Exact forward+backward useful attention FLOPs:
                # Forward = 4 * B * H_q * D * active_pairs; Backward = 2.5x Forward => 14 * B * H_q * D * active_pairs
                exact_doc_attn_flops = (
                    14.0 * float(batch) * float(h_q) * float(head_dim) * float(active_causal_pairs)
                )

                doc_ids = torch.cat(
                    [
                        torch.full((L,), doc_i, device=device, dtype=torch.int32)
                        for doc_i, L in enumerate(doc_lengths)
                    ]
                )
                cu_list = [0]
                for L in doc_lengths:
                    cu_list.append(cu_list[-1] + L)
                cu_seqlens = torch.tensor(cu_list, device=device, dtype=torch.int32)

                mask_2d = (idx[:, None] >= idx[None, :]) & (
                    doc_ids[:, None] == doc_ids[None, :]
                )
                mask_4d = mask_2d.unsqueeze(0).unsqueeze(0)

                def doc_mask_mod(
                    b_idx: torch.Tensor,
                    h_idx: torch.Tensor,
                    q_idx: torch.Tensor,
                    kv_idx: torch.Tensor,
                    d_ids: torch.Tensor = doc_ids,
                ) -> torch.Tensor:
                    return (q_idx >= kv_idx) & (d_ids[q_idx] == d_ids[kv_idx])

                def doc_score_mod_only(
                    score: torch.Tensor,
                    b_idx: torch.Tensor,
                    h_idx: torch.Tensor,
                    q_idx: torch.Tensor,
                    kv_idx: torch.Tensor,
                    d_ids: torch.Tensor = doc_ids,
                ) -> torch.Tensor:
                    valid = (q_idx >= kv_idx) & (d_ids[q_idx] == d_ids[kv_idx])
                    return torch.where(valid, score, float("-inf"))

                # Benchmark BlockMask construction: eager vs torch.compile(create_block_mask)
                _ = create_block_mask(
                    doc_mask_mod,
                    B=None,
                    H=None,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=device,
                )
                _ = compiled_cbm(
                    doc_mask_mod,
                    B=None,
                    H=None,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=device,
                )
                eager_cbm_t = benchmark_with_telemetry(
                    lambda mm=doc_mask_mod: create_block_mask(
                        mm,
                        B=None,
                        H=None,
                        Q_LEN=seq_len,
                        KV_LEN=seq_len,
                        device=device,
                    ),
                    telemetry,
                    warmup=2,
                    groups=4,
                    iterations_per_group=3,
                )
                compiled_cbm_t = benchmark_with_telemetry(
                    lambda mm=doc_mask_mod: compiled_cbm(
                        mm,
                        B=None,
                        H=None,
                        Q_LEN=seq_len,
                        KV_LEN=seq_len,
                        device=device,
                    ),
                    telemetry,
                    warmup=3,
                    groups=4,
                    iterations_per_group=6,
                )
                bm = compiled_cbm(
                    doc_mask_mod,
                    B=None,
                    H=None,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=device,
                )
                bm_sparsity_pct = float(bm.sparsity())
                bm_active_tile_fraction = 1.0 - (bm_sparsity_pct / 100.0)

                mask_build_records.append(
                    {
                        "experiment": "blockmask_construction_overhead",
                        "regime": regime_name,
                        "shape_b_s_hq_hkv_d": list(shape),
                        "num_documents": num_docs,
                        "mean_doc_len": mean_doc_len,
                        "max_doc_len": max_doc_len,
                        "active_pair_fraction": round(active_pair_fraction, 6),
                        "blockmask_sparsity_percent": round(bm_sparsity_pct, 4),
                        "blockmask_active_tile_fraction": round(
                            bm_active_tile_fraction,
                            6,
                        ),
                        "subtile_valid_density": round(
                            active_pair_fraction / max(bm_active_tile_fraction, 1e-9),
                            6,
                        ),
                        "eager_create_block_mask_ms": eager_cbm_t["median_ms"],
                        "compiled_create_block_mask_ms": compiled_cbm_t["median_ms"],
                        "compile_speedup": eager_cbm_t["median_ms"]
                        / compiled_cbm_t["median_ms"],
                    }
                )

                ref_loss, ref_out, ref_dq, ref_dk, ref_dv = (
                    compute_fp32_attention_reference(
                        q_bhsd,
                        k_bhsd,
                        v_bhsd,
                        mask_2d=mask_2d,
                    )
                )

                bhsd_grad_extract = lambda ps: (
                    ps[0].grad.detach().float(),
                    ps[1].grad.detach().float(),
                    ps[2].grad.detach().float(),
                )
                thd_grad_extract = lambda ps: (
                    ps[0]
                    .grad.detach()
                    .view(batch, seq_len, h_q, head_dim)
                    .transpose(1, 2)
                    .float(),
                    ps[1]
                    .grad.detach()
                    .view(batch, seq_len, h_kv, head_dim)
                    .transpose(1, 2)
                    .float(),
                    ps[2]
                    .grad.detach()
                    .view(batch, seq_len, h_kv, head_dim)
                    .transpose(1, 2)
                    .float(),
                )

                def sdpa_auto_fwd(m4=mask_4d) -> torch.Tensor:
                    return F.scaled_dot_product_attention(
                        q_bhsd,
                        k_bhsd,
                        v_bhsd,
                        attn_mask=m4,
                        enable_gqa=True,
                    )

                def sdpa_eff_expanded_fwd(m4=mask_4d) -> torch.Tensor:
                    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                        kr = k_bhsd.repeat_interleave(group_size, dim=1)
                        vr = v_bhsd.repeat_interleave(group_size, dim=1)
                        return F.scaled_dot_product_attention(
                            q_bhsd,
                            kr,
                            vr,
                            attn_mask=m4,
                        )

                t_no_bm = time.perf_counter()
                _ = compiled_flex_default(
                    q_bhsd,
                    k_bhsd,
                    v_bhsd,
                    score_mod=doc_score_mod_only,
                    enable_gqa=True,
                ).float().square().mean().backward()
                torch.cuda.synchronize()
                setup_no_bm = time.perf_counter() - t_no_bm

                t_def = time.perf_counter()
                _ = compiled_flex_default(
                    q_bhsd,
                    k_bhsd,
                    v_bhsd,
                    block_mask=bm,
                    enable_gqa=True,
                ).float().square().mean().backward()
                torch.cuda.synchronize()
                setup_def = time.perf_counter() - t_def

                t_auto = time.perf_counter()
                _ = compiled_flex_autotune(
                    q_bhsd,
                    k_bhsd,
                    v_bhsd,
                    block_mask=bm,
                    enable_gqa=True,
                ).float().square().mean().backward()
                torch.cuda.synchronize()
                setup_auto = time.perf_counter() - t_auto

                def flex_nobm_fwd(sm=doc_score_mod_only) -> torch.Tensor:
                    return compiled_flex_default(
                        q_bhsd,
                        k_bhsd,
                        v_bhsd,
                        score_mod=sm,
                        enable_gqa=True,
                    )

                def flex_def_fwd(bmask=bm) -> torch.Tensor:
                    return compiled_flex_default(
                        q_bhsd,
                        k_bhsd,
                        v_bhsd,
                        block_mask=bmask,
                        enable_gqa=True,
                    )

                def flex_auto_fwd(bmask=bm) -> torch.Tensor:
                    return compiled_flex_autotune(
                        q_bhsd,
                        k_bhsd,
                        v_bhsd,
                        block_mask=bmask,
                        enable_gqa=True,
                    )

                def varlen_raw_step(
                    cu=cu_seqlens,
                    ml=max_doc_len,
                ) -> torch.Tensor:
                    q_thd.grad = None
                    k_thd.grad = None
                    v_thd.grad = None
                    out_v = flash_attn_varlen_func(
                        q_thd,
                        k_thd,
                        v_thd,
                        cu,
                        cu,
                        ml,
                        ml,
                        causal=True,
                    )
                    loss_v = out_v.float().square().mean()
                    loss_v.backward()
                    return loss_v

                def varlen_bhsd_fwd(
                    cu=cu_seqlens,
                    ml=max_doc_len,
                ) -> torch.Tensor:
                    out_v = flash_attn_varlen_func(
                        q_thd,
                        k_thd,
                        v_thd,
                        cu,
                        cu,
                        ml,
                        ml,
                        causal=True,
                    )
                    return out_v.view(batch, seq_len, h_q, head_dim).transpose(1, 2)

                impl_specs = [
                    (
                        "sdpa_gqa_bool_mask_auto",
                        sdpa_auto_fwd,
                        (q_bhsd, k_bhsd, v_bhsd),
                        bhsd_grad_extract,
                        None,
                        0.0,
                        True,
                    ),
                    (
                        "sdpa_expanded_kv_efficient",
                        sdpa_eff_expanded_fwd,
                        (q_bhsd, k_bhsd, v_bhsd),
                        bhsd_grad_extract,
                        None,
                        0.0,
                        True,
                    ),
                    (
                        "flex_score_mod_no_blockmask",
                        flex_nobm_fwd,
                        (q_bhsd, k_bhsd, v_bhsd),
                        bhsd_grad_extract,
                        None,
                        setup_no_bm,
                        False,
                    ),
                    (
                        "flex_blockmask_default",
                        flex_def_fwd,
                        (q_bhsd, k_bhsd, v_bhsd),
                        bhsd_grad_extract,
                        None,
                        setup_def,
                        True,
                    ),
                    (
                        "flex_blockmask_max_autotune",
                        flex_auto_fwd,
                        (q_bhsd, k_bhsd, v_bhsd),
                        bhsd_grad_extract,
                        None,
                        setup_auto,
                        False,
                    ),
                    (
                        "flash_attn_varlen_cu_seqlens",
                        varlen_bhsd_fwd,
                        (q_thd, k_thd, v_thd),
                        thd_grad_extract,
                        varlen_raw_step,
                        0.0,
                        True,
                    ),
                ]

                candidates: list[dict[str, Any]] = []
                for (
                    impl_name,
                    fwd_bhsd,
                    params_t,
                    grad_ext,
                    override_step,
                    setup_s,
                    prof_flag,
                ) in impl_specs:
                    corr = evaluate_attention_correctness(
                        fwd_bhsd,
                        params_t,
                        grad_ext,
                        ref_loss,
                        ref_out,
                        ref_dq,
                        ref_dk,
                        ref_dv,
                    )
                    if override_step is not None:
                        step_callable = override_step
                    else:
                        def step_callable(
                            f=fwd_bhsd,
                            ps=params_t,
                        ) -> torch.Tensor:
                            ps[0].grad = None
                            ps[1].grad = None
                            ps[2].grad = None
                            loss_t = f().float().square().mean()
                            loss_t.backward()
                            return loss_t

                    candidates.append(
                        {
                            "name": impl_name,
                            "step_fn": step_callable,
                            "params": params_t,
                            "setup_seconds": setup_s,
                            "correctness": corr,
                            "profile_kernels": prof_flag,
                        }
                    )

                del ref_out, ref_dq, ref_dk, ref_dv
                torch.cuda.empty_cache()

                telemetry.lock_sm_clock(boost_mhz)
                pass_a: dict[str, dict[str, Any]] = {}
                pass_b: dict[str, dict[str, Any]] = {}
                for cand in candidates:
                    pass_a[cand["name"]] = benchmark_with_telemetry(
                        cand["step_fn"],
                        telemetry,
                        warmup=warmup,
                        groups=groups,
                        iterations_per_group=iters,
                    )
                for cand in reversed(candidates):
                    pass_b[cand["name"]] = benchmark_with_telemetry(
                        cand["step_fn"],
                        telemetry,
                        warmup=warmup,
                        groups=groups,
                        iterations_per_group=iters,
                    )

                if regime_name == profiled_regime:
                    for cand in candidates:
                        if not cand["profile_kernels"]:
                            continue
                        for _ in range(3):
                            cand["step_fn"]()
                        torch.cuda.synchronize()
                        with profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                        ) as prof:
                            for _ in range(prof_steps):
                                cand["step_fn"]()
                            torch.cuda.synchronize()

                        cat_us: dict[str, float] = {}
                        top_rows: list[dict[str, Any]] = []
                        total_krn = 0.0
                        for evt in prof.key_averages():
                            if str(getattr(evt, "device_type", "")) != "DeviceType.CUDA":
                                continue
                            us = (
                                float(getattr(evt, "self_device_time_total", 0.0))
                                / prof_steps
                            )
                            cnt = float(evt.count) / prof_steps
                            cat = classify_attention_kernel(evt.key)
                            cat_us[cat] = cat_us.get(cat, 0.0) + us
                            total_krn += cnt
                            top_rows.append(
                                {
                                    "kernel": evt.key[:140],
                                    "category": cat,
                                    "calls_per_step": round(cnt, 2),
                                    "cuda_us_per_step": round(us, 3),
                                }
                            )
                        top_rows.sort(
                            key=lambda item: item["cuda_us_per_step"],
                            reverse=True,
                        )
                        tot_us = sum(cat_us.values())
                        kernel_records.append(
                            {
                                "experiment": "cuda_kernel_attribution",
                                "workload": f"packed_doc_{regime_name}",
                                "implementation": cand["name"],
                                "shape_b_s_hq_hkv_d": list(shape),
                                "profiled_steps": prof_steps,
                                "total_cuda_kernel_us_per_step": round(tot_us, 3),
                                "category_us_per_step": {
                                    k: round(v, 3) for k, v in sorted(cat_us.items())
                                },
                                "cuda_kernels_per_step": round(total_krn, 2),
                                "top_cuda_kernels": top_rows[:6],
                            }
                        )

                for cand in candidates:
                    merged = merge_two_passes(pass_a[cand["name"]], pass_b[cand["name"]])
                    mem_stats = measure_attention_memory(
                        cand["step_fn"],
                        cand["params"],
                    )
                    sec = merged["median_ms"] / 1000.0
                    doc_records.append(
                        {
                            "experiment": "packed_document_gqa_training",
                            "protocol": "locked_boost_palindromic_balanced",
                            "clock_policy": f"locked_{boost_mhz}mhz",
                            "regime": regime_name,
                            "num_documents": num_docs,
                            "mean_doc_len": mean_doc_len,
                            "max_doc_len": max_doc_len,
                            "active_pair_fraction": round(active_pair_fraction, 6),
                            "blockmask_sparsity_percent": round(bm_sparsity_pct, 4),
                            "blockmask_active_tile_fraction": round(
                                bm_active_tile_fraction,
                                6,
                            ),
                            "subtile_valid_density": round(
                                active_pair_fraction
                                / max(bm_active_tile_fraction, 1e-9),
                                6,
                            ),
                            "implementation": cand["name"],
                            "shape_b_s_hq_hkv_d": list(shape),
                            "dtype": str(dtype),
                            "setup_seconds": cand["setup_seconds"],
                            **cand["correctness"],
                            **merged,
                            **mem_stats,
                            "tokens_per_second": (batch * seq_len) / sec,
                            "effective_valid_attn_tflop_s": exact_doc_attn_flops
                            / sec
                            / 1e12,
                        }
                    )

                del candidates, mask_2d, mask_4d, bm
                torch.cuda.empty_cache()

            del q_bhsd, k_bhsd, v_bhsd, q_thd, k_thd, v_thd
            torch._dynamo.reset()
            torch.cuda.empty_cache()
    finally:
        torch._dynamo.config.cache_size_limit = old_limit
        telemetry.reset_clocks()
        if old_cache_dir is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = old_cache_dir

    return doc_records, mask_build_records, kernel_records


def run_score_mod_gqa_study(
    profile_name: str,
    device: torch.device,
    dtype: torch.dtype,
    telemetry: GpuTelemetry,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if profile_name == "smoke":
        shape = (1, 256, 16, 4, 128)
        window = 64
        warmup, groups, iters = 2, 2, 2
        prof_steps = 2
    else:
        shape = (2, 4096, 32, 8, 128)
        window = 512
        warmup, groups, iters = 6, 4, 6
        prof_steps = 6

    batch, seq_len, h_q, h_kv, head_dim = shape
    group_size = h_q // h_kv
    scale = head_dim**-0.5
    softcap = 30.0
    boost_mhz = telemetry.max_clocks()["max_sm_clock_mhz"]

    score_records: list[dict[str, Any]] = []
    kernel_records: list[dict[str, Any]] = []

    old_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    old_limit = torch._dynamo.config.cache_size_limit
    try:
        telemetry.lock_sm_clock(boost_mhz)
        torch._dynamo.config.cache_size_limit = 64
        with tempfile.TemporaryDirectory(prefix="e113_score_mod_") as tmpdir:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(Path(tmpdir) / "cache")
            torch._dynamo.reset()

            torch.manual_seed(67)
            q_bhsd = torch.nn.Parameter(
                torch.randn((batch, h_q, seq_len, head_dim), device=device, dtype=dtype)
            )
            k_bhsd = torch.nn.Parameter(
                torch.randn(
                    (batch, h_kv, seq_len, head_dim), device=device, dtype=dtype
                )
            )
            v_bhsd = torch.nn.Parameter(
                torch.randn(
                    (batch, h_kv, seq_len, head_dim), device=device, dtype=dtype
                )
            )

            q_bshd = torch.nn.Parameter(
                q_bhsd.detach().transpose(1, 2).contiguous()
            )
            k_bshd = torch.nn.Parameter(
                k_bhsd.detach().transpose(1, 2).contiguous()
            )
            v_bshd = torch.nn.Parameter(
                v_bhsd.detach().transpose(1, 2).contiguous()
            )

            slopes = torch.linspace(0.05, 0.50, h_q, device=device, dtype=torch.float32)
            idx = torch.arange(seq_len, device=device)
            causal_2d = idx[:, None] >= idx[None, :]
            sw_2d = (idx[:, None] >= idx[None, :]) & (
                (idx[:, None] - idx[None, :]) <= window
            )
            dist_2d = (idx[:, None] - idx[None, :]).clamp_min(0).to(torch.float32)
            log_dist_2d = torch.log2(1.0 + dist_2d)

            compiled_cbm = torch.compile(create_block_mask)
            bm_causal = compiled_cbm(
                lambda b, h, q_i, kv_i: q_i >= kv_i,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
            )
            bm_sw = compiled_cbm(
                lambda b, h, q_i, kv_i, w=window: (q_i >= kv_i) & ((q_i - kv_i) <= w),
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
            )

            compiled_flex_default = torch.compile(
                flex_attention,
                fullgraph=True,
                mode="default",
            )
            compiled_flex_autotune = torch.compile(
                flex_attention,
                fullgraph=True,
                mode="max-autotune-no-cudagraphs",
            )

            def softcap_mod(
                score: torch.Tensor,
                b_i: torch.Tensor,
                h_i: torch.Tensor,
                q_i: torch.Tensor,
                kv_i: torch.Tensor,
            ) -> torch.Tensor:
                return softcap * torch.tanh(score / softcap)

            def log_decay_mod(
                score: torch.Tensor,
                b_i: torch.Tensor,
                h_i: torch.Tensor,
                q_i: torch.Tensor,
                kv_i: torch.Tensor,
                sl: torch.Tensor = slopes,
            ) -> torch.Tensor:
                dist = (q_i - kv_i).clamp_min(0).to(torch.float32)
                return score - sl[h_i] * torch.log2(1.0 + dist)

            law_specs = [
                (
                    "causal_softcap30",
                    causal_2d,
                    bm_causal,
                    softcap,
                    None,
                    softcap_mod,
                    (-1, -1),
                    True,
                ),
                (
                    f"sw{window}_softcap30",
                    sw_2d,
                    bm_sw,
                    softcap,
                    None,
                    softcap_mod,
                    (window, 0),
                    True,
                ),
                (
                    f"sw{window}_log_decay_bias",
                    sw_2d,
                    bm_sw,
                    0.0,
                    slopes,
                    log_decay_mod,
                    (window, 0),
                    False,
                ),
            ]

            bhsd_grad_extract = lambda ps: (
                ps[0].grad.detach().float(),
                ps[1].grad.detach().float(),
                ps[2].grad.detach().float(),
            )
            bshd_grad_extract = lambda ps: (
                ps[0].grad.detach().transpose(1, 2).float(),
                ps[1].grad.detach().transpose(1, 2).float(),
                ps[2].grad.detach().transpose(1, 2).float(),
            )

            for (
                law_name,
                mask_2d,
                bmask,
                cap_val,
                law_slopes,
                score_mod_fn,
                fa_win,
                fa_supported,
            ) in law_specs:
                active_pairs = int(mask_2d.sum().item())
                exact_attn_flops = (
                    14.0
                    * float(batch)
                    * float(h_q)
                    * float(head_dim)
                    * float(active_pairs)
                )
                bm_sparsity_pct = float(bmask.sparsity())

                ref_loss, ref_out, ref_dq, ref_dk, ref_dv = (
                    compute_fp32_attention_reference(
                        q_bhsd,
                        k_bhsd,
                        v_bhsd,
                        mask_2d=mask_2d,
                        softcap=cap_val,
                        log_decay_slopes=law_slopes,
                    )
                )

                if cap_val > 0.0:
                    def mat_fn(
                        q_in: torch.Tensor,
                        k_in: torch.Tensor,
                        v_in: torch.Tensor,
                        m2: torch.Tensor = mask_2d,
                        cp: float = cap_val,
                    ) -> torch.Tensor:
                        kr = k_in.repeat_interleave(group_size, dim=1)
                        vr = v_in.repeat_interleave(group_size, dim=1)
                        sc = torch.matmul(q_in, kr.transpose(-2, -1)).float() * scale
                        sc = cp * torch.tanh(sc / cp)
                        sc = sc.masked_fill(~m2, float("-inf"))
                        pr = torch.softmax(sc, dim=-1).to(q_in.dtype)
                        return torch.matmul(pr, vr)
                else:
                    def mat_fn(
                        q_in: torch.Tensor,
                        k_in: torch.Tensor,
                        v_in: torch.Tensor,
                        m2: torch.Tensor = mask_2d,
                        sl: torch.Tensor = slopes,
                        ld2: torch.Tensor = log_dist_2d,
                    ) -> torch.Tensor:
                        kr = k_in.repeat_interleave(group_size, dim=1)
                        vr = v_in.repeat_interleave(group_size, dim=1)
                        sc = torch.matmul(q_in, kr.transpose(-2, -1)).float() * scale
                        sc = sc - sl[None, :, None, None] * ld2[None, None, :, :]
                        sc = sc.masked_fill(~m2, float("-inf"))
                        pr = torch.softmax(sc, dim=-1).to(q_in.dtype)
                        return torch.matmul(pr, vr)

                t_cm = time.perf_counter()
                compiled_mat_fn = torch.compile(mat_fn, fullgraph=True, mode="default")
                _ = compiled_mat_fn(q_bhsd, k_bhsd, v_bhsd).float().square().mean().backward()
                torch.cuda.synchronize()
                setup_cm = time.perf_counter() - t_cm

                t_fd = time.perf_counter()
                _ = compiled_flex_default(
                    q_bhsd,
                    k_bhsd,
                    v_bhsd,
                    score_mod=score_mod_fn,
                    block_mask=bmask,
                    enable_gqa=True,
                ).float().square().mean().backward()
                torch.cuda.synchronize()
                setup_fd = time.perf_counter() - t_fd

                t_fa = time.perf_counter()
                _ = compiled_flex_autotune(
                    q_bhsd,
                    k_bhsd,
                    v_bhsd,
                    score_mod=score_mod_fn,
                    block_mask=bmask,
                    enable_gqa=True,
                ).float().square().mean().backward()
                torch.cuda.synchronize()
                setup_fa = time.perf_counter() - t_fa

                impl_list: list[tuple[Any, ...]] = [
                    (
                        "eager_materialized_score_mod",
                        lambda mf=mat_fn: mf(q_bhsd, k_bhsd, v_bhsd),
                        (q_bhsd, k_bhsd, v_bhsd),
                        bhsd_grad_extract,
                        None,
                        0.0,
                        True,
                    ),
                    (
                        "compiled_materialized_score_mod",
                        lambda cmf=compiled_mat_fn: cmf(q_bhsd, k_bhsd, v_bhsd),
                        (q_bhsd, k_bhsd, v_bhsd),
                        bhsd_grad_extract,
                        None,
                        setup_cm,
                        True,
                    ),
                    (
                        "flex_attention_default",
                        lambda sm=score_mod_fn, bm_in=bmask: compiled_flex_default(
                            q_bhsd,
                            k_bhsd,
                            v_bhsd,
                            score_mod=sm,
                            block_mask=bm_in,
                            enable_gqa=True,
                        ),
                        (q_bhsd, k_bhsd, v_bhsd),
                        bhsd_grad_extract,
                        None,
                        setup_fd,
                        True,
                    ),
                    (
                        "flex_attention_max_autotune",
                        lambda sm=score_mod_fn, bm_in=bmask: compiled_flex_autotune(
                            q_bhsd,
                            k_bhsd,
                            v_bhsd,
                            score_mod=sm,
                            block_mask=bm_in,
                            enable_gqa=True,
                        ),
                        (q_bhsd, k_bhsd, v_bhsd),
                        bhsd_grad_extract,
                        None,
                        setup_fa,
                        False,
                    ),
                ]

                if fa_supported:
                    def fa_bhsd_fwd(w_cfg=fa_win, cp=cap_val) -> torch.Tensor:
                        return flash_attn_func(
                            q_bshd,
                            k_bshd,
                            v_bshd,
                            causal=True,
                            window_size=w_cfg,
                            softcap=cp,
                        ).transpose(1, 2)

                    def fa_raw_step(w_cfg=fa_win, cp=cap_val) -> torch.Tensor:
                        q_bshd.grad = None
                        k_bshd.grad = None
                        v_bshd.grad = None
                        out_t = flash_attn_func(
                            q_bshd,
                            k_bshd,
                            v_bshd,
                            causal=True,
                            window_size=w_cfg,
                            softcap=cp,
                        )
                        loss_t = out_t.float().square().mean()
                        loss_t.backward()
                        return loss_t

                    impl_list.append(
                        (
                            "flash_attn_native_cuda",
                            fa_bhsd_fwd,
                            (q_bshd, k_bshd, v_bshd),
                            bshd_grad_extract,
                            fa_raw_step,
                            0.0,
                            True,
                        )
                    )

                candidates: list[dict[str, Any]] = []
                for (
                    impl_name,
                    fwd_bhsd,
                    params_t,
                    grad_ext,
                    override_step,
                    setup_s,
                    prof_flag,
                ) in impl_list:
                    corr = evaluate_attention_correctness(
                        fwd_bhsd,
                        params_t,
                        grad_ext,
                        ref_loss,
                        ref_out,
                        ref_dq,
                        ref_dk,
                        ref_dv,
                    )
                    if override_step is not None:
                        step_callable = override_step
                    else:
                        def step_callable(
                            f=fwd_bhsd,
                            ps=params_t,
                        ) -> torch.Tensor:
                            ps[0].grad = None
                            ps[1].grad = None
                            ps[2].grad = None
                            loss_t = f().float().square().mean()
                            loss_t.backward()
                            return loss_t

                    candidates.append(
                        {
                            "name": impl_name,
                            "step_fn": step_callable,
                            "params": params_t,
                            "setup_seconds": setup_s,
                            "correctness": corr,
                            "profile_kernels": prof_flag,
                        }
                    )

                del ref_out, ref_dq, ref_dk, ref_dv
                torch.cuda.empty_cache()

                telemetry.lock_sm_clock(boost_mhz)
                pass_a: dict[str, dict[str, Any]] = {}
                pass_b: dict[str, dict[str, Any]] = {}
                for cand in candidates:
                    pass_a[cand["name"]] = benchmark_with_telemetry(
                        cand["step_fn"],
                        telemetry,
                        warmup=warmup,
                        groups=groups,
                        iterations_per_group=iters,
                    )
                for cand in reversed(candidates):
                    pass_b[cand["name"]] = benchmark_with_telemetry(
                        cand["step_fn"],
                        telemetry,
                        warmup=warmup,
                        groups=groups,
                        iterations_per_group=iters,
                    )

                if law_name == f"sw{window}_softcap30":
                    for cand in candidates:
                        if not cand["profile_kernels"]:
                            continue
                        for _ in range(3):
                            cand["step_fn"]()
                        torch.cuda.synchronize()
                        with profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                        ) as prof:
                            for _ in range(prof_steps):
                                cand["step_fn"]()
                            torch.cuda.synchronize()

                        cat_us: dict[str, float] = {}
                        top_rows: list[dict[str, Any]] = []
                        total_krn = 0.0
                        for evt in prof.key_averages():
                            if str(getattr(evt, "device_type", "")) != "DeviceType.CUDA":
                                continue
                            us = (
                                float(getattr(evt, "self_device_time_total", 0.0))
                                / prof_steps
                            )
                            cnt = float(evt.count) / prof_steps
                            cat = classify_attention_kernel(evt.key)
                            cat_us[cat] = cat_us.get(cat, 0.0) + us
                            total_krn += cnt
                            top_rows.append(
                                {
                                    "kernel": evt.key[:140],
                                    "category": cat,
                                    "calls_per_step": round(cnt, 2),
                                    "cuda_us_per_step": round(us, 3),
                                }
                            )
                        top_rows.sort(
                            key=lambda item: item["cuda_us_per_step"],
                            reverse=True,
                        )
                        tot_us = sum(cat_us.values())
                        kernel_records.append(
                            {
                                "experiment": "cuda_kernel_attribution",
                                "workload": f"score_mod_{law_name}",
                                "implementation": cand["name"],
                                "shape_b_s_hq_hkv_d": list(shape),
                                "profiled_steps": prof_steps,
                                "total_cuda_kernel_us_per_step": round(tot_us, 3),
                                "category_us_per_step": {
                                    k: round(v, 3) for k, v in sorted(cat_us.items())
                                },
                                "cuda_kernels_per_step": round(total_krn, 2),
                                "top_cuda_kernels": top_rows[:6],
                            }
                        )

                for cand in candidates:
                    merged = merge_two_passes(pass_a[cand["name"]], pass_b[cand["name"]])
                    mem_stats = measure_attention_memory(
                        cand["step_fn"],
                        cand["params"],
                    )
                    sec = merged["median_ms"] / 1000.0
                    score_records.append(
                        {
                            "experiment": "score_modified_gqa_training",
                            "protocol": "locked_boost_palindromic_balanced",
                            "clock_policy": f"locked_{boost_mhz}mhz",
                            "score_law": law_name,
                            "flash_attn_native_supported": fa_supported,
                            "blockmask_sparsity_percent": round(bm_sparsity_pct, 4),
                            "active_pair_fraction": round(
                                active_pairs / float(seq_len * seq_len),
                                6,
                            ),
                            "implementation": cand["name"],
                            "shape_b_s_hq_hkv_d": list(shape),
                            "dtype": str(dtype),
                            "setup_seconds": cand["setup_seconds"],
                            **cand["correctness"],
                            **merged,
                            **mem_stats,
                            "tokens_per_second": (batch * seq_len) / sec,
                            "effective_valid_attn_tflop_s": exact_attn_flops
                            / sec
                            / 1e12,
                        }
                    )

                del candidates
                torch.cuda.empty_cache()

            del q_bhsd, k_bhsd, v_bhsd, q_bshd, k_bshd, v_bshd
            torch._dynamo.reset()
            torch.cuda.empty_cache()
    finally:
        torch._dynamo.config.cache_size_limit = old_limit
        telemetry.reset_clocks()
        if old_cache_dir is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = old_cache_dir

    return score_records, kernel_records


def run(profile_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    telemetry = GpuTelemetry()
    atexit.register(telemetry.reset_clocks)
    telemetry.reset_clocks()

    env = environment_metadata()
    if env.get("hostname"):
        env["hostname"] = "node-0"
    env["nvml_max_clocks"] = telemetry.max_clocks()
    env["initial_telemetry"] = telemetry.sample()

    try:
        doc_records, mask_build_records, doc_krn = run_packed_document_gqa_study(
            profile_name,
            device,
            dtype,
            telemetry,
        )
        score_records, score_krn = run_score_mod_gqa_study(
            profile_name,
            device,
            dtype,
            telemetry,
        )
    finally:
        telemetry.reset_clocks()

    time.sleep(0.2)
    env["final_unlocked_telemetry"] = telemetry.sample()

    return {
        "suite": "flex_attention_training",
        "profile": profile_name,
        "environment": env,
        "records": doc_records + score_records,
        "blockmask_construction_records": mask_build_records,
        "kernel_attribution_records": doc_krn + score_krn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_result(args.output, run(args.profile))


if __name__ == "__main__":
    main()
