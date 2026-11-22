# E120 — Multi-Head Latent Attention (MLA) Training Kernels, Weight-Absorption Crossover, & 8-GPU Compressed-Latent Ring CP

Raw and aggregated JSON artifacts for **E120**:

- `smoke_mla_latent_attention_training.json`: 8-GPU smoke, ecosystem compatibility audit, and `FP64` Richardson chain-rule validation gate.
- `rep{1,2,3}_mla_latent_attention_training.json`: Three independent OS-process full repetitions on 8x NVIDIA A100-SXM4-80GB (NV12 NVSwitch) at hardware-locked `1410 MHz` SM clocks.
- `mla_latent_attention_training_full.json`: Deterministic 3-repetition median/IQR aggregation and 9-gate validation output from `code/analyze_mla_latent_attention_training.py`.
- `SHA256SUMS`: SHA-256 checksums for all JSON files in this directory.
