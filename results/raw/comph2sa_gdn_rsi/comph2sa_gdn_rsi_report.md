# E163 (`v154`): Subspace-Complementary Orthogonal-Query Dual-Route Sparse-Linear Attention (`CompH2SA-GDN`)

- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`3` independent 8-GPU process repetitions)
- **Frozen Evaluator SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e` (`verified = True`)
- **SM80 Cubin Audit**: `19` kernels, `172` specializations, `0` spills (`max_regs = 168 <= 168`)
- **FP64 Oracle Parity**: `FP32 max_rel_l2 = 8.42e-07`, `BF16 max_rel_l2 = 3.44e-03`, `mutations_killed = 3/3`
- **Compositional Hit+Miss Identification (`Track 1A`)**:
  - `Composite_HitAndMiss` rejected-hop cosine `0.999988`, retention `0.134445` vs `0.040058` (`E162`, `3.36x` retention gain), poison suppression `1061.04x` vs `E161`, joint cosine `0.977811` vs `0.973174` (`E162`) & `0.967098` (`E161`)
- **25-Cell Compositional Recall Gain (`Track 3`)**: `+12.73` pp mean gain (up to `+37.25` pp) over `E162` and `+13.25` pp mean gain (up to `+36.37` pp) over `E161`
- **Training & Decode Speedups (`Track 1B/1C`)**: `2.86x–15.81x` total training speedup, `3.43x–188.31x` single-pass decode speedup
- **Wave-5 Governed-Loop v3.8 Tournament (`Track 6`)**: `C5_CompH2SA_GDN` achieved `PROMOTE` across all 7 comparators and `HOLDOUT_PROMOTE` (`0.9891` vs `0.9249` `E162`)
