# E117 — Fine-Grained MoE Training Kernels & 8-GPU Expert-Parallel Overlap

Raw and aggregated JSON artifacts for **E117**:

- `smoke_moe_training_grouped_gemm.json`: 8-GPU smoke and numerical validation gate.
- `rep{1,2,3}_moe_training_grouped_gemm.json`: Three independent OS-process full repetitions on 8x NVIDIA A100-SXM4-80GB (NV12 NVSwitch) at hardware-locked `1410 MHz` SM clocks.
- `moe_training_grouped_gemm_full.json`: Deterministic 3-repetition median/IQR aggregation and 9-gate validation output from `code/analyze_moe_training_grouped_gemm.py`.
- `SHA256SUMS`: SHA-256 checksums for all JSON files in this directory.
