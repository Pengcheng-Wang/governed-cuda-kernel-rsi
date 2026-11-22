# Transformer compile diagnosis artifacts

This directory contains E111's compact A100 hardware diagnosis of the E7 versus
E110 transformer-block compilation speedup gap, compiler-mode frontier, and CUDA
kernel time attribution.

- `smoke_transformer_compile_diagnosis.json` gates the protocol ablation,
  7-implementation compiler/CUDA-Graph frontier, and CUDA kernel profiler on
  small shapes before full benchmarking.
- `rep1_transformer_compile_diagnosis.json` through
  `rep3_transformer_compile_diagnosis.json` record three independent OS-process
  repetitions on NVIDIA A100-SXM4-80GB across unlocked DVFS versus
  hardware-locked 1140 MHz and 1410 MHz SM clocks, cold versus warm
  TorchInductor caches, execution-order reversals, seven execution/compiler
  modes, and `torch.profiler` device-kernel attribution.
- `transformer_compile_diagnosis_full.json` deterministically aggregates the
  smoke and full repetitions against E7 and E110 baselines and checks embedded
  environment, correctness, locked-clock, cache-invariance, Amdahl-closure, and
  regime-crossover validation gates.
- `SHA256SUMS` records every promoted JSON digest.

All GPU clock locks applied during measurement were reset to factory unlocked
policy before process exit and verified afterward. No AML resource identifiers
or access metadata are published.
