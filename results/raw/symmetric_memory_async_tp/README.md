# NVSwitch SymmetricMemory, Sequence Parallelism, and Async-TP artifacts

This directory contains E114's compact 8xA100-SXM4-80GB (NV12 NVSwitch)
measurements across `TP = 2, 4, 8`, evaluating PyTorch 2.13 `SymmetricMemory`
collectives, Megatron-style Tensor and Sequence Parallelism (`TP` vs `TP+SP`),
`DTensor` sequence-axis layouts (`[B, S, H]` `Shard(1)` vs `[S, B, H]`
`Shard(0)`), and TorchInductor `_micro_pipeline_tp` (`Async-TP`) collective-GEMM
overlap in Llama-3-8B GQA Transformer block (`H = 4096, FFN = 14336, 32Q/8KV,
D = 128`) forward+backward training.

- `smoke_symmetric_memory_async_tp.json` gates `TP = 2` and `TP = 8` forward
  outputs, `dx`, sharded weight gradients, `1.75x` upstream scalar backward
  chain-rule scaling, and pre/post-timing profiler kernel assertions before full
  benchmarking.
- `rep1_symmetric_memory_async_tp.json` through
  `rep3_symmetric_memory_async_tp.json` record three independent OS-process
  repetitions at hardware-locked 1410 MHz across:
  1. PyTorch 2.13's `enable_symm_mem_for_group` `FutureWarning` silent no-op and
     persistent TorchInductor FX graph cache poisoning audit;
  2. NVSwitch `SymmetricMemory` one-shot (staged and with-copy) and two-shot
     all-reduce versus NCCL all-reduce from `64 KiB` to `64 MiB`;
  3. Isolated `fused_all_gather_matmul` (dual SwiGLU projections) and
     `fused_matmul_reduce_scatter` versus sequential NCCL + cuBLAS across
     `M_global = 2,048..32,768`; and
  4. Eight end-to-end GQA Transformer block training modes across `1x4096`,
     `2x4096`, `4x4096`, and `8x4096`.
- `symmetric_memory_async_tp_full.json` deterministically aggregates the smoke
  and full repetitions and verifies all nine correctness, compiler-gate,
  all-reduce crossover, wave-quantization crossover, layout/memory, and
  large-batch `Async-TP` speedup gates.
- `SHA256SUMS` records every promoted JSON digest.

All GPU clock locks applied during measurement were reset to factory unlocked
policy before process exit. No AML resource identifiers or access metadata are
published.
