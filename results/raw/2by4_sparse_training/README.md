# E124: SM80 `2:4` Semi-Structured Sparse Tensor-Core Training (`cuSPARSELt` / `Exact-90 DP` / `KKT-Exact Madow MVUE` / `Grad-SR-STE`), Layout-Aware `Sparse-SwiGLU`, & 8-GPU `9 bits/param` Compressed-`AllGather` `FSDP2` Training

This directory contains raw and deterministically aggregated 8-GPU `NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch, SM clocks locked at `1,410 MHz`) artifacts for **E124**:

- `smoke_2by4_sparse_training.json`: 8-GPU `--mode smoke` gate artifact.
- `rep{1,2,3}_2by4_sparse_training.json`: Three independent OS-process 8-GPU `--mode full` repetitions.
- `2by4_sparse_training_full.json`: Deterministically aggregated median/IQR summary and 9-gate validation output from `code/analyze_2by4_sparse_training.py`.
- `SHA256SUMS`: SHA-256 checksums over the 5 JSON artifacts.

## Suites Covered

1. **Suite 0 (`suite_0_audits`)**:
   - SM80 `2:4` Sparse Tensor-Core roofline (`mma.sp.sync.m16n8k32` `624 TFLOP/s` `BF16` peak: `bf16_cublas_dense` vs PyTorch `SparseSemiStructuredTensorCUTLASS` `0.84x` regression vs `cuSPARSELt` `0.8.1` default `alg_id=0` and searched/tuned `alg_id=3`).
   - PyTorch `2.13` `semi_structured` triple-defect audit: (a) default `prune_dense_static_sort("")` sorting signed values (`74.2%` positive bias, `54.0%` L1 retained vs `70.24%`), (b) `largest_abs_values_greedy` deadlocking `15.8%` of `4x4` bipartite tiles at `nnz = 7 < 8` vs our `0`-spill `Exact-90` factored bipartite DP dual packer (`0.0%` deadlocks, `100.0%` of 90-mask L1 oracle, `1.6x` faster than PyTorch's greedy C++ kernel), and (c) `torch._C._cusparselt.mm_search` returning non-divisible `split_k > 1` plans (`split_k = 3, 5`) that trigger `CUSPARSE_STATUS_INTERNAL_ERROR` under `transpose_result=True`.
   - Neural-gradient `2:4` pruning (`dW`) Monte Carlo convergence (`1,024` steps): permanent systematic bias of `greedy_top2_mse_biased` vs high-variance `uniform_unbiased_2x` vs our `KKT-Exact Madow-Systematic MVUE 2:4` fused transpose+packer.
   - `160`-step `AdamW` Transformer `SwiGLU` pre-training trajectory: mask flip-rate explosion in `naive_ste` and unnormalized `weight_sr_ste` (Zhou et al., ICLR 2021) vs `1/sqrt(v)`-normalized `grad_sr_ste` (Hu et al., ICML 2024) via a single-pass fused Triton `AdamW + 16-bit Bitmask Grad-SR-STE` kernel.
   - `FP64` Richardson central-difference (`h` vs `h/2` `~4.00x`) analytical chain-rule verification.
2. **Suite A (`suite_a_linear_3gemm_scaling`)**:
   - Single-GPU Llama-3-8B MLP projection (`K = 4,096 -> N = 14,336`) complete `3-GEMM` (`Fwd + dX + dW`) scaling across `M = 2,048 .. 16,384` tokens (`bf16_cublas_dense`, `pytorch_cutlass_2by4_2gemm`, `cslt_exact90_2gemm_refresh1`, `cslt_exact90_3gemm_fst_refresh1`, amortized `cslt_exact90_3gemm_fst_refresh10`, and `cslt_exact90_3gemm_fst_gacc4` surpassing SM80's `312 TFLOP/s` dense `BF16` hardware ceiling in pure `BF16`).
3. **Suite B (`suite_b_sparse_swiglu_stack`)**:
   - Isolated layout-aware gated-activation ablation (`col_major_strided_eager_swiglu` `16 KiB`-stride L2 cache thrashing vs `col_major_coalesced_triton_swiglu` and zero-spill `row_major_fused_swiglu_plus_mvue24_pack`).
   - 4-layer Llama-3-8B Pre-Norm `SwiGLU` MLP stack (`H = 4,096, I = 14,336`, `Fwd + Bwd + AdamW / Bitmask Grad-SR-STE` step) across `N = 2,048 .. 8,192` tokens comparing `mlp_bf16_dense_eager`, `mlp_bf16_dense_compile`, `mlp_2by4_pytorch_cutlass_2gemm`, `mlp_2by4_fst_cslt_refresh1`, and `mlp_2by4_fst_cslt_refresh10`.
4. **Suite C (`suite_c_9bit_2by4_fsdp2_8gpu`)**:
   - 8-GPU NVSwitch `9 bits/param` (`8-bit` packed `BF16` values + `1-bit` swizzled `cuSPARSELt` metadata) shard-local `Exact-90` compressed-`AllGather` parity vs 1-GPU global `Exact-90` (`Fwd`, `dX`, `2-GEMM dW`, and multi-step `3-GEMM FST dW`).
   - End-to-end 8-GPU `FSDP2` training scaling (`6` layers, `H = 4,096, I = 14,336, 32Q/8KV, D = 128`, `1.31B` params, `65,536` global tokens/step) comparing `fsdp2_bf16_dense_eager`, `fsdp2_bf16_dense_compile`, `fsdp2_2by4_post_ag_redundant_fst` (`16-bit` `AllGather` + `8x` redundant pruning), `fsdp2_2by4_ffn_hybrid_9bit_ag_refresh1` (`80.8%` FFN `9-bit` `2:4` + Dense GQA), `fsdp2_2by4_9bit_ag_fst_refresh1`, and `fsdp2_2by4_9bit_ag_fst_refresh10` (`4,992.0 -> 2,808.0 MiB/step` `AllGather`, `1.778x` bandwidth reduction, overlapped via block-flat `comm_stream` prefetching + intra-block MLP `ReduceScatter` overlap).
