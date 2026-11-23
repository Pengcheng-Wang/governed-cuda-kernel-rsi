# E162 (`v153`): LSE-Calibrated Centered-Residual Subtile-WY Sparse-Linear Attention (`CalH2SA-GDN`)

- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`3` independent 8-GPU process repetitions)
- **Frozen Evaluator SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e` (`verified = True`)
- **SM80 Cubin Audit**: `18` kernels, `157` specializations, `0` spills (`max_regs = 168 <= 168`)
- **FP64 Oracle Parity**: `FP32 max_rel_l2 = 8.21e-07`, `BF16 max_rel_l2 = 3.39e-03`, `mutations_killed = 3/3`
- **Selected-Hit vs. Rejected-Miss Duality (`Track 1A`)**:
  - `SelectedHit_Poison`: fidelity `0.996345` vs `0.931919` (`E161`, `+6.44` pp), poison leakage `0.0862` vs `0.5270` (`6.11x` lower poison leakage)
  - `RejectedMiss_Rescue`: cosine `0.998145`, retention `0.097520` vs `0.061882` (`E161`, `1.58x` higher retention)
  - `Dipolar_IntraBlock`: cosine `0.997747` vs `0.000000` (`E160`)
- **Training & Decode Speedups (`Track 1B/1C`)**: `3.03x–17.10x` total training speedup (`15.20x` Fwd, `18.77x` Bwd), `3.23x–167.73x` single-pass decode speedup
- **Wave-4 Governed-Loop v3.8 Tournament (`Track 6`)**: `C5_CalH2SA_GDN` achieved `PROMOTE` across all 6 comparators and `HOLDOUT_PROMOTE` (`0.9846` vs `0.9083` `E161`)
