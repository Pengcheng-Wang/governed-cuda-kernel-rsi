# E160 (`v151`): Round-34 RSI — Adaptive Budget-Split Dual-Route Sparse-Linear Attention (`AdaH2SA-GDN`), Counterfactual Rejected-Block Fact-Flip Identification, & Governed Loop Optimization (`v3.8`) Wave-2 Tournament

- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch)
- **Frozen Kernel Harness SHA256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`
- **Governed Loop Guard (`v3.8`) Wave-2 Manifest SHA256**: `7c64510bbe6009010e1636da9759b5df725e62e583eb27326399cd24c3d829d6`
- **Cubin Spill Audit**: `16` kernels, `89` specializations, `0` spills (`max_regs = 168 <= 168`, `all_zero_spill = True`)
- **FP64 Oracle Parity**: `FP32 max_rel_l2 = 9.01e-07 < 1e-4`, `BF16 max_rel_l2 = 2.99e-03 < 1.5e-2`, `3/3` (`100%`) adversarial mutations killed
- **Counterfactual Rejected-Block Fact-Flip Identification (`Track 1A`)**:
  - `Selected-Only (No GDN)`: `cosine = 0.000000` (`0` sensitivity once routes are frozen)
  - `Shuffled-State Capacity Control`: `cosine = 0.000000` (`0` sensitivity under state permutation)
  - `Uniform-Write H2SA-GDN (E159)`: `retention = 0.2047` (diluted by subsequent filler blocks)
  - `Surprise-Gated AdaH2SA-GDN (E160)`: `cosine = 0.999958`, `retention = 0.5963` (`2.91x` higher retention than uniform write)
- **Governed Loop Optimization (`v3.8`) Wave-2 Verdict**:
  - `check-preconditions`: `GO` (`MDE_80% = 0.0170`, `B10` fresh unsealed Holdout verified)
  - `audit-splits`: `CLEAN` (`0` row/cluster/time leaks across `Search_W2`, `Confirm_W2`, `Holdout_W2`)
  - `audit-leaks`: `CLEAN`
  - `Round-1 Search Winner`: `W2_C3_FixedBudgetSurpGDN`
  - `Round-2 Crossover Winner`: `C5_AdaH2SA_GDN`
  - `Intersection-Union Confirm (gate-confirm-iut)`: `PROMOTE` across `['baseline', 'dsa_e157', 'moba_e158', 'h2sa_gdn_e159']`
  - `Atomic Single-Touch Holdout (gate-holdout)`: `FINALIZED` (`holdout_guardrails_passed = True`, `ledger = VERIFIED`, `7` hash-chained entries)
