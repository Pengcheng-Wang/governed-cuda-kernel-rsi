from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from common import (
    allclose_metrics,
    benchmark_cuda,
    compile_with_timing,
    environment_metadata,
    write_result,
)


class DecodeMLP(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down = torch.nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def capture_graph(
    module: torch.nn.Module,
    sample: torch.Tensor,
) -> tuple[callable, float]:
    static_input = sample.clone()
    for _ in range(3):
        module(static_input)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.cuda.graph(graph):
        static_output = module(static_input)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - started

    def replay() -> torch.Tensor:
        static_input.copy_(sample)
        graph.replay()
        return static_output

    return replay, setup_seconds


def run(profile: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden, intermediate = (1024, 2816) if profile == "smoke" else (4096, 11008)
    batches = [1] if profile == "smoke" else [1, 8, 32]
    warmup, groups, iterations = (3, 3, 3) if profile == "smoke" else (20, 9, 50)

    module = DecodeMLP(hidden, intermediate).to(device=device, dtype=dtype).eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)

    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in batches:
            x = torch.randn((batch, hidden), device=device, dtype=dtype)
            reference = module(x)

            compiled_default, compile_default_seconds = compile_with_timing(
                module,
                (x,),
                mode="default",
            )
            compiled_reduce, compile_reduce_seconds = compile_with_timing(
                module,
                (x,),
                mode="reduce-overhead",
            )
            graph_replay, graph_setup_seconds = capture_graph(module, x)

            implementations = [
                ("torch_eager", lambda: module(x), 0.0),
                ("torch_compile_default", lambda: compiled_default(x), compile_default_seconds),
                (
                    "torch_compile_reduce_overhead",
                    lambda: compiled_reduce(x),
                    compile_reduce_seconds,
                ),
                ("explicit_cuda_graph", graph_replay, graph_setup_seconds),
            ]
            for name, fn, setup_seconds in implementations:
                actual = fn()
                torch.cuda.synchronize()
                correctness = allclose_metrics(actual, reference)
                timing = benchmark_cuda(
                    fn,
                    warmup=warmup,
                    groups=groups,
                    iterations_per_group=iterations,
                )
                seconds = timing["median_ms"] / 1_000.0
                records.append(
                    {
                        "experiment": "decode_mlp_fixed_shape",
                        "implementation": name,
                        "shape": [batch, hidden, intermediate],
                        "dtype": str(dtype),
                        "correctness": correctness,
                        "setup_seconds": setup_seconds,
                        **timing,
                        "tokens_per_second": batch / seconds,
                    }
                )
            del x, reference
            torch.cuda.empty_cache()

    return {
        "suite": "decode_graphs",
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
