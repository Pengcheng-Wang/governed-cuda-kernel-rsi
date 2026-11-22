# E119 — Chunkwise Mamba-2 SSD/GLA & WY Gated DeltaNet Training Kernels + 8-GPU Context Parallelism

Raw and aggregated JSON artifacts for **E119**:

- `smoke_linear_attention_gdn_training.json`: 8-GPU smoke and numerical/chain-rule validation gate.
- `rep{1,2,3}_linear_attention_gdn_training.json`: Three independent OS-process full repetitions on 8x NVIDIA A100-SXM4-80GB (NV12 NVSwitch) at hardware-locked `1410 MHz` SM clocks.
- `linear_attention_gdn_training_full.json`: Deterministic 3-repetition median/IQR aggregation and 9-gate validation output from `code/analyze_linear_attention_gdn_training.py`.
- `SHA256SUMS`: SHA-256 checksums for all JSON files in this directory.