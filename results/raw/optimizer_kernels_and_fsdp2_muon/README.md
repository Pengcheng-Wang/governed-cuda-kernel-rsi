# E118 — Fused Multi-Tensor Optimizer Kernels, SR-BF16/INT8 AdamW & 8-GPU FSDP2 Owner-Sharded Muon

Raw and aggregated JSON artifacts for **E118**:

- `smoke_optimizer_kernels_and_fsdp2_muon.json`: 8-GPU smoke and numerical validation gate.
- `rep{1,2,3}_optimizer_kernels_and_fsdp2_muon.json`: Three independent OS-process full repetitions on 8x NVIDIA A100-SXM4-80GB (NV12 NVSwitch) at hardware-locked `1410 MHz` SM clocks.
- `optimizer_kernels_and_fsdp2_muon_full.json`: Deterministic 3-repetition median/IQR aggregation and 9-gate validation output from `code/analyze_optimizer_kernels_and_fsdp2_muon.py`.
- `SHA256SUMS`: SHA-256 checksums for all JSON files in this directory.