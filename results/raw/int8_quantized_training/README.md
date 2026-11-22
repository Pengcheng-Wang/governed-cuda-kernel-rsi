# E123: Block-Hadamard `INT8` Tensor-Core Training (`SwitchBack` / `Jetfire` / `COAT` / `Halo`), `COAT` Compressed `INT8` Activations, & 8-GPU Quantized-`AllGather` `FSDP2` Training

This directory contains raw and deterministically aggregated 8-GPU `NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch, SM clocks locked at `1,410 MHz`) artifacts for **E123**:

- `smoke_int8_quantized_training.json`: 8-GPU `--mode smoke` gate artifact.
- `rep{1,2,3}_int8_quantized_training.json`: Three independent OS-process 8-GPU `--mode full` repetitions.
- `int8_quantized_training_full.json`: Deterministically aggregated median/IQR summary and 9-gate validation output from `code/analyze_int8_quantized_training.py`.
- `SHA256SUMS`: SHA-256 checksums over the 5 JSON artifacts.

## Suites Covered

1. **Suite 0 (`suite_0_audits`)**:
   - SM80 `INT8` Tensor-Core layout & accumulator-promotion roofline (`bf16_cublas` `312 TFLOP/s` ceiling vs `torch._int_mm` row-major E6 pitfall vs column-major `mma.sync.m16n8k32.row.col.s32.s8.s8.s32`, `Jetfire`/`DeepSeek-V3` `1x128 x 128x128` in-loop SIMT `I2F+FFMA` promotion bottleneck, pure Triton coalesced epilogue-only `INT8`, and `CUTLASS SM80` fused `INT8 -> BF16`).
   - SM80 PTX quantizer ablation (`FP32 atomicCAS` + 10-round Philox + `>255`-reg local-memory spill vs `s32`-bitcast native `RED.MAX.S32` + 3-op SplitMix32 dither + split `64x128` / `128x64` zero-spill tiles).
   - Outlier-spiked `3-GEMM` (`Fwd`, `dX`, `dW`) fidelity (`SwitchBack` unrotated vs `Jetfire` `1x128 x 128x128` vs `1D Block-128` & `2D Kronecker` Tensor-Core Randomized Hadamard) and `64`-step small-signal `RTN-INT8` underflow bias vs unbiased `SR-INT8` convergence.
   - `FP64` Richardson central-difference (`h` vs `h/2` `~4.00x`) analytical chain-rule verification.
2. **Suite A (`suite_a_linear_3gemm_scaling`)**:
   - Single-GPU Llama-3-8B MLP projection (`H = 4,096 -> I = 14,336`) complete `3-GEMM` (`Fwd + dX + dW`) scaling across `M = 2,048 .. 16,384` tokens (`bf16_cublas_eager`, `triton_jetfire_block128_3gemm_int8`, `switchback_hybrid_hadamard_int8`, `triton_pure_halo_3gemm_sr_int8`, `halo_cutlass_3gemm_sr_int8_gacc1`, and weight-stationary `halo_cutlass_3gemm_sr_int8_gacc4` surpassing SM80's `312 TFLOP/s` `BF16` hardware ceiling).
3. **Suite B (`suite_b_coat_swiglu_mlp_stack`)**:
   - 4-layer Llama-3-8B Pre-Norm `SwiGLU` MLP stack (`H = 4,096, I = 14,336`, `~1.05B` MLP params) across `N = 2,048 .. 8,192` tokens comparing `mlp_bf16_eager`, `mlp_bf16_torch_compile`, `mlp_bf16_full_checkpoint`, and `mlp_halo_coat_int8_fused` (`rstd`-invariant Fused `RMSNorm + Hadamard-128 -> INT8` + `COAT` `1 B/elem` `INT8` saved `SwiGLU` activations with zero `4th-GEMM` recomputation).
4. **Suite C (`suite_c_quantized_fsdp2_8gpu`)**:
   - 8-GPU NVSwitch `FSDP2` distributed gradient parity (`halo_coat_quant_fsdp2` vs `fsdp2_bf16` vs `naive_unrotated_rtn_int8_rs_control` against 1-GPU exact `FP32`) and end-to-end training scaling (`6` layers, `H = 4,096, I = 14,336, 32Q/8KV, D = 128`, `1.39B` params, `32,768` global tokens/step) comparing `fsdp2_bf16_eager_reshard`, `fsdp2_bf16_compile_reshard`, `fsdp2_bf16_act_checkpoint`, and shard-local Hadamard `1 B/param` `INT8`-`AllGather` `halo_coat_quant_fsdp2`.
