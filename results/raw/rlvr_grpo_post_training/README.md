# E121 — Reasoning Post-Training (RLVR / GRPO / DAPO / DPO) Fused Terminal Kernels, Ragged Group-Prefix-Tree FlashAttention, & 8-GPU FSDP2 Long-CoT Training

Raw and aggregated JSON artifacts for **E121**:

- `smoke_rlvr_grpo_post_training.json`: 8-GPU smoke, `FP32` entropy cancellation audit, `FP64` Richardson chain-rule, and `FSDP2` gradient parity gate.
- `rep{1,2,3}_rlvr_grpo_post_training.json`: Three independent OS-process full repetitions on 8x NVIDIA A100-SXM4-80GB (NV12 NVSwitch) at hardware-locked `1410 MHz` SM clocks.
- `rlvr_grpo_post_training_full.json`: Deterministic 3-repetition median/IQR aggregation and 9-gate validation output from `code/analyze_rlvr_grpo_post_training.py`.
- `SHA256SUMS`: SHA-256 checksums for all JSON files in this directory.
