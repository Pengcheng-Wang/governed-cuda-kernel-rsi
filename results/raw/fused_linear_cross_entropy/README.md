# Fused linear cross-entropy artifacts

This directory contains E112's compact A100 hardware measurements for
large-vocabulary (`V = 128,256`) LM-head projection plus cross-entropy loss
across SLM (`H = 1024, 2048`) and LLM (`H = 4096`) hidden widths.

- `smoke_fused_linear_cross_entropy.json` gates forward loss, `ignore_index`
  masking, upstream scalar backward chain-rule scaling, and `dx`/`dW` relative
  errors against an exact FP32 reference before full benchmarking.
- `rep1_fused_linear_cross_entropy.json` through
  `rep3_fused_linear_cross_entropy.json` record three independent OS-process
  repetitions on NVIDIA A100-SXM4-80GB comparing `eager_fp32_logits_ce`,
  `eager_bf16_logits_ce`, `torch_compile_full_ce`,
  `checkpointed_chunked_compile_ce`, and custom Triton online-softmax in-place
  gradient kernels (`triton_fused_linear_ce_bf16_dw` and
  `triton_fused_linear_ce_fp32_dw`) across token/width scaling, a 7-point
  chunk-size frontier (`128` to `8192`), and `torch.profiler` CUDA kernel
  attribution.
- `fused_linear_cross_entropy_full.json` deterministically aggregates the smoke
  and full repetitions and verifies correctness, analytic memory closure,
  compiler `mm`-boundary materialization, checkpointed-GEMM crossover, and
  `out_dtype=torch.float32` chunk-count invariance gates.
- `SHA256SUMS` records every promoted JSON digest.

All GPU clock locks applied during measurement were reset to factory unlocked
policy before process exit. No AML resource identifiers or access metadata are
published.
