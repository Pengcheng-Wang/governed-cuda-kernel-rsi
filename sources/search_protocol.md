# Research protocol

Search date: 2026-11-10.

## Scope

Current CUDA/GPU-kernel optimization for LLM and SLM training and inference,
with emphasis on techniques reproducible on NVIDIA A100 (SM80). Hopper- and
Blackwell-specific techniques are included only as clearly marked context.

## Inclusion priorities

1. Peer-reviewed papers and well-established preprints with released code.
2. Official CUDA, CUTLASS, cuBLAS, NCCL, PyTorch, Triton, and NVIDIA material.
3. Canonical repositories for FlashAttention, vLLM/PagedAttention, quantized
   GEMM, compiler, and distributed-training implementations.
4. Independent benchmark or reproduction evidence where available.

## Exclusions

- Marketing-only performance claims without workload details.
- Benchmarks that change precision, model quality, batch shape, or semantics
  without disclosing the change.
- Hardware-specific claims generalized across GPU generations.
- Stale project versions when a verified successor exists.

## Synthesis fields

For each technique: problem, kernel idea, bottleneck, hardware applicability,
maturity, strongest evidence, limitations, and an A100 reproduction design.

