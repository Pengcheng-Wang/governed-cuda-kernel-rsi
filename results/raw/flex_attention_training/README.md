# Programmable FlexAttention training artifacts

This directory contains E113's compact A100 hardware measurements comparing
PyTorch 2.13 `FlexAttention` (`BlockMask` + `score_mod` + `enable_gqa=True`)
against dense-mask `F.scaled_dot_product_attention`, materialized `torch.compile`,
and hand-specialized `FlashAttention-2` (`flash_attn_varlen_func` and
`flash_attn_func` softcap/sliding-window) in forward+backward training.

- `smoke_flex_attention_training.json` gates forward output, `dQ`/`dK`/`dV`
  gradients, and `1.75x` upstream scalar backward chain-rule scaling against
  exact FP32 chunked attention references before full benchmarking.
- `rep1_flex_attention_training.json` through
  `rep3_flex_attention_training.json` record three independent OS-process
  repetitions on NVIDIA A100-SXM4-80GB across:
  1. Sequence-packed document-masked GQA (`1x8192`, `32Q/8KV`, `D=128`) over
     `1`, `4`, `16`, and `64` packed documents, plus eager versus compiled
     `create_block_mask` overhead;
  2. Score-modified GQA (`2x4096`, `32Q/8KV`, `D=128`) across full-causal
     `tanh` softcapping (`softcap=30`), sliding-window (`W=512`) `tanh`
     softcapping, and sliding-window non-linear logarithmic positional decay
     (`-slope_h * log2(1 + dist)`), which fixed-kernel `flash_attn` cannot
     express; and
  3. `torch.profiler` CUDA kernel attribution.
- `flex_attention_training_full.json` deterministically aggregates the smoke and
  full repetitions and verifies correctness, silent-fallback memory gates,
  `BlockMask` sparsity speedups, sub-tile quantization crossovers versus exact
  `cu_seqlens`, and custom `score_mod` generality gates.
- `SHA256SUMS` records every promoted JSON digest.

All GPU clock locks applied during measurement were reset to factory unlocked
policy before process exit. No AML resource identifiers or access metadata are
published.
