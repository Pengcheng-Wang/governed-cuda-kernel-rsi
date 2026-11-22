from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

from common import (
    allclose_metrics,
    benchmark_cuda,
    environment_metadata,
    measure_peak_memory,
    write_result,
)


def make_workload(
    base_q: torch.Tensor,
    base_k: torch.Tensor,
    base_v: torch.Tensor,
    implementation: str,
) -> tuple[
    Callable[[], torch.Tensor],
    Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    q = base_q.detach().clone().requires_grad_(True)
    k = base_k.detach().clone().requires_grad_(True)
    v = base_v.detach().clone().requires_grad_(True)

    def forward() -> torch.Tensor:
        if implementation == "torch_sdpa_math":
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=True,
                )
        if implementation == "torch_sdpa_flash":
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=True,
                )
        if implementation == "flash_attn":
            return flash_attn_func(q, k, v, causal=True).transpose(1, 2)
        raise ValueError(f"Unknown implementation: {implementation}")

    def step() -> torch.Tensor:
        q.grad = None
        k.grad = None
        v.grad = None
        output = forward()
        loss = output.float().square().mean()
        loss.backward()
        return loss

    def snapshot() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        step()
        torch.cuda.synchronize()
        output = forward().detach().clone()
        q.grad = None
        k.grad = None
        v.grad = None
        loss = forward().float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        return (
            output,
            q.grad.detach().clone(),
            k.grad.detach().clone(),
            v.grad.detach().clone(),
        )

    return step, snapshot, (q, k, v)


def profiler_summary(
    step: Callable[[], torch.Tensor],
    *,
    warmup: int = 3,
    top_k: int = 20,
) -> dict[str, Any]:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        step()
    torch.cuda.synchronize()

    events: list[dict[str, Any]] = []
    for event in prof.key_averages():
        self_device_us = float(
            getattr(
                event,
                "self_device_time_total",
                getattr(event, "self_cuda_time_total", 0.0),
            )
        )
        device_us = float(
            getattr(
                event,
                "device_time_total",
                getattr(event, "cuda_time_total", 0.0),
            )
        )
        if self_device_us <= 0.0 and device_us <= 0.0:
            continue
        events.append(
            {
                "name": event.key,
                "count": event.count,
                "self_device_time_us": self_device_us,
                "device_time_us": device_us,
                "input_shapes": str(getattr(event, "input_shapes", "")),
            }
        )
    events.sort(key=lambda item: item["self_device_time_us"], reverse=True)
    return {
        "top_cuda_events": events[:top_k],
        "table": prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=top_k,
        ),
    }


def run(profile_name: str, include_profiler: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(37)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    if profile_name == "smoke":
        shapes = [(1, 256, 8, 64)]
        warmup, groups, iterations = 1, 2, 2
    else:
        shapes = [
            (4, 512, 16, 128),
            (2, 2048, 16, 128),
        ]
        warmup, groups, iterations = 5, 5, 5

    records: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for batch, sequence, heads, head_dim in shapes:
        base_q = torch.randn(
            (batch, sequence, heads, head_dim),
            device=device,
            dtype=dtype,
        )
        base_k = torch.randn_like(base_q)
        base_v = torch.randn_like(base_q)

        reference_step, reference_snapshot, reference_leaves = make_workload(
            base_q,
            base_k,
            base_v,
            "torch_sdpa_math",
        )
        reference_output, reference_dq, reference_dk, reference_dv = reference_snapshot()
        implementations = (
            "torch_sdpa_math",
            "torch_sdpa_flash",
            "flash_attn",
        )

        for implementation in implementations:
            if implementation == "torch_sdpa_math":
                step = reference_step
                snapshot = reference_snapshot
                leaves = reference_leaves
                output = reference_output
                dq, dk, dv = reference_dq, reference_dk, reference_dv
            else:
                step, snapshot, leaves = make_workload(
                    base_q,
                    base_k,
                    base_v,
                    implementation,
                )
                output, dq, dk, dv = snapshot()

            timing = benchmark_cuda(
                step,
                warmup=warmup,
                groups=groups,
                iterations_per_group=iterations,
            )
            for leaf in leaves:
                leaf.grad = None
            peak_memory = measure_peak_memory(step)
            seconds = timing["median_ms"] / 1_000.0
            records.append(
                {
                    "experiment": "causal_attention_forward_backward",
                    "implementation": implementation,
                    "shape_bshd": [batch, sequence, heads, head_dim],
                    "dtype": str(dtype),
                    "output_correctness": allclose_metrics(output, reference_output),
                    "dq_correctness": allclose_metrics(dq, reference_dq),
                    "dk_correctness": allclose_metrics(dk, reference_dk),
                    "dv_correctness": allclose_metrics(dv, reference_dv),
                    **timing,
                    "peak_incremental_memory_bytes": peak_memory,
                    "tokens_per_second": batch * sequence / seconds,
                }
            )

            if (
                include_profiler
                and sequence == max(shape[1] for shape in shapes)
                and implementation in {"torch_sdpa_flash", "flash_attn"}
            ):
                profiles.append(
                    {
                        "implementation": implementation,
                        "shape_bshd": [batch, sequence, heads, head_dim],
                        **profiler_summary(step),
                    }
                )

            del output, dq, dk, dv
            torch.cuda.empty_cache()

        del (
            base_q,
            base_k,
            base_v,
            reference_output,
            reference_dq,
            reference_dk,
            reference_dv,
        )
        torch.cuda.empty_cache()

    return {
        "suite": "attention_backward",
        "profile": profile_name,
        "environment": environment_metadata(),
        "records": records,
        "profiler_summaries": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--include-profiler", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_result(args.output, run(args.profile, args.include_profiler))


if __name__ == "__main__":
    main()
