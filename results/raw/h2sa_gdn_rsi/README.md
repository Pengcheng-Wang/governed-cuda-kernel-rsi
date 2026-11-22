# E159 (`v150`): Round-33 RSI — Hierarchical Dual-Route Sparse-Linear Attention (`H2SA-GDN`) & Governed Loop Optimization (`v3.8`)

- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch)
- **Frozen Kernel Harness SHA256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`
- **Governed Loop Guard (`v3.8`) Manifest SHA256**: `654f7af70709b75c61c1e3d05225ed645aa3b6b5e8d982cf10077287f27176fc`
- **Cubin Spill Audit**: `16` kernels, `83` specializations, `0` spills (`max_regs = 168 <= 168`, `all_zero_spill = True`)
- **FP64 Oracle Parity**: `FP32 max_rel_l2 = 9.24e-07 < 1e-4`, `BF16 max_rel_l2 = 4.72e-03 < 1.5e-2`, `3/3` (`100%`) adversarial mutations killed
- **Governed Loop Optimization (`v3.8`) Verdict**:
  - `check-preconditions`: `GO` (`MDE_80% = 0.0170`, `2^-24 = 5.96e-08`)
  - `audit-splits`: `CLEAN` (`0` row/cluster/time leaks across `Search`, `Confirm`, `Holdout`)
  - `audit-leaks`: `CLEAN`
  - `Round-1 Search Winner`: `C2_SubtileSpan`
  - `Round-2 Crossover Winner`: `C5_H2SA_GDN`
  - `Intersection-Union Confirm (gate-confirm-iut)`: `PROMOTE` across `['baseline', 'dsa_e157', 'moba_e158']`
  - `Atomic Single-Touch Holdout (gate-holdout)`: `FINALIZED` (`holdout_guardrails_passed = True`, `ledger = VERIFIED`, `6` hash-chained entries)
