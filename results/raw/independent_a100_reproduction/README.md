# Independent A100 reproduction artifacts

This directory contains E110's compact independent-allocation reproduction
panel.

- Six smoke JSON files gate the transferred benchmark code.
- Two full repetitions cover fused SwiGLU/RMSNorm, native CUDA SwiGLU,
  educational Triton versus cuBLAS GEMM, causal attention, transformer-block
  training, and single-node 8-GPU NCCL all-reduce.
- Four additional transformer-training repetitions investigate the only
  unstable full-panel result.
- `independent_a100_reproduction_full.json` compares the fresh allocation
  with the original A100 artifacts, including allocation drift, repeat
  spread, correctness, preserved negative controls, and compiler
  variability.
- `SHA256SUMS` records every promoted JSON digest.

The allocation used the same A100 model and software versions as the original
study, so this is allocation-level reproduction rather than a cross-hardware
portability claim. No AML resource identifiers or access metadata are
published.
