from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from common import benchmark_cuda, environment_metadata, measure_peak_memory, write_result


class TransformerBlock(torch.nn.Module):
    def __init__(self, hidden: int, heads: int, intermediate: int) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm1 = torch.nn.RMSNorm(hidden, eps=1e-6)
        self.qkv = torch.nn.Linear(hidden, 3 * hidden, bias=False)
        self.out_proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.norm2 = torch.nn.RMSNorm(hidden, eps=1e-6)
        self.gate_up = torch.nn.Linear(hidden, 2 * intermediate, bias=False)
        self.down = torch.nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        normalized = self.norm1(x)
        qkv = self.qkv(normalized).view(
            batch,
            sequence,
            3,
            self.heads,
            self.head_dim,
        )
        q, k, v = qkv.unbind(dim=2)
        attention = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        )
        attention = attention.transpose(1, 2).contiguous().view(batch, sequence, self.hidden)
        x = x + self.out_proj(attention)

        gate, up = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


def training_step(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    output = model(x)
    loss = output.float().square().mean()
    loss.backward()
    return loss


def gradient_summary(model: torch.nn.Module) -> dict[str, float]:
    squared_norm = 0.0
    max_abs = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.float()
        squared_norm += grad.square().sum().item()
        max_abs = max(max_abs, grad.abs().max().item())
    return {
        "l2_norm": squared_norm**0.5,
        "max_abs": max_abs,
    }


def run(profile: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(23)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    if profile == "smoke":
        shapes = [(1, 128, 512, 8, 1408)]
        warmup, groups, iterations = 1, 2, 2
    else:
        shapes = [
            (4, 1024, 1024, 16, 2816),
            (1, 4096, 1024, 16, 2816),
        ]
        warmup, groups, iterations = 3, 5, 5

    records: list[dict[str, Any]] = []
    for batch, sequence, hidden, heads, intermediate in shapes:
        eager_model = TransformerBlock(hidden, heads, intermediate).to(
            device=device,
            dtype=dtype,
        )
        compiled_model_base = copy.deepcopy(eager_model)
        x = torch.randn((batch, sequence, hidden), device=device, dtype=dtype)

        eager_loss = training_step(eager_model, x)
        torch.cuda.synchronize()
        eager_gradients = gradient_summary(eager_model)

        compile_started = time.perf_counter()
        compiled_model = torch.compile(
            compiled_model_base,
            fullgraph=True,
            mode="default",
        )
        compiled_loss = training_step(compiled_model, x)
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - compile_started
        compiled_gradients = gradient_summary(compiled_model_base)

        implementations = [
            (
                "torch_eager",
                eager_model,
                0.0,
                eager_loss.detach().item(),
                eager_gradients,
            ),
            (
                "torch_compile",
                compiled_model,
                compile_seconds,
                compiled_loss.detach().item(),
                compiled_gradients,
            ),
        ]
        reference_loss = eager_loss.detach().item()
        reference_grad_norm = eager_gradients["l2_norm"]
        for name, model, setup_seconds, loss_value, gradients in implementations:
            fn = lambda model=model: training_step(model, x)
            timing = benchmark_cuda(
                fn,
                warmup=warmup,
                groups=groups,
                iterations_per_group=iterations,
            )
            peak_memory = measure_peak_memory(fn)
            seconds = timing["median_ms"] / 1_000.0
            records.append(
                {
                    "experiment": "transformer_block_training",
                    "implementation": name,
                    "shape": [batch, sequence, hidden, heads, intermediate],
                    "dtype": str(dtype),
                    "setup_seconds": setup_seconds,
                    "loss": loss_value,
                    "loss_abs_error": abs(loss_value - reference_loss),
                    "gradient_l2_norm": gradients["l2_norm"],
                    "gradient_l2_relative_error": abs(
                        gradients["l2_norm"] - reference_grad_norm
                    )
                    / max(reference_grad_norm, 1e-12),
                    "gradient_max_abs": gradients["max_abs"],
                    **timing,
                    "peak_incremental_memory_bytes": peak_memory,
                    "tokens_per_second": batch * sequence / seconds,
                }
            )

        del eager_model, compiled_model_base, compiled_model, x
        torch.cuda.empty_cache()

    return {
        "suite": "transformer_training",
        "profile": profile,
        "environment": environment_metadata(),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_result(args.output, run(args.profile))


if __name__ == "__main__":
    main()
