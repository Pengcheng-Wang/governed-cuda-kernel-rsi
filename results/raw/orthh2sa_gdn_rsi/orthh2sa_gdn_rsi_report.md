# E161 (`v152`): Round-35 RSI — Route-Orthogonalized Subtile-WY Dual-Route Sparse-Linear Attention (`OrthH2SA-GDN`), Dipolar Intra-Block Fact-Disentanglement Identification, & Governed Loop Optimization (`v3.8`) Wave-3 Tournament

- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch)
- **Frozen Kernel Harness SHA256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`
- **Governed Loop Guard (`v3.8`) Wave-3 Manifest SHA256**: `08d2d1d3d170fdb0d53e5ca01949c9a1a6bb706629e46fa8fe50034004a520ed`
- **Cubin Spill Audit**: `16` kernels, `148` specializations, `0` spills (`max_regs = 168 <= 168`, `all_zero_spill = True`)
- **FP64 Oracle Parity**: `FP32 max_rel_l2 = 8.93e-07 < 1e-4`, `BF16 max_rel_l2 = 3.01e-03 < 1.5e-2`, `3/3` (`100%`) adversarial mutations killed
- **Dipolar Intra-Block Fact-Disentanglement Identification (`Track 1A`)**:
  - `Selected-Only (No GDN)`: `cosine = 0.000000` (`0` sensitivity once routes are frozen)
  - `Coarse-Block Uniform-Write H2SA-GDN (E159)`: `cosine = 0.000000`, `retention = 0.000000` (complete intra-block cancellation `bar_v_c = 0`)
  - `Coarse-Block Surprise-Gated AdaH2SA-GDN (E160)`: `cosine = 0.000000`, `retention = 0.000000` (complete intra-block cancellation `bar_v_c = 0`)
  - `Subtile-WY OrthH2SA-GDN (E161)`: `cosine = 0.999613`, `retention = 0.623854` (disentangles opposite-sign facts across subtiles inside the same rejected block!)
- **Governed Loop Optimization (`v3.8`) Wave-3 Verdict**:
  - `check-preconditions`: `GO` (`MDE_80% = 0.0170`, `B10` fresh unsealed Wave-3 Holdout verified)
  - `audit-splits`: `CLEAN` (`0` row/cluster/time leaks across `Search_W3`, `Confirm_W3`, `Holdout_W3`)
  - `audit-leaks`: `CLEAN`
  - `Round-1 Search Winner`: `W3_C2_StaticSubtileWY`
  - `Round-2 Crossover Winner`: `C5_OrthH2SA_GDN`
  - `Intersection-Union Confirm (gate-confirm-iut)`: `PROMOTE` across `['baseline', 'dsa_e157', 'moba_e158', 'h2sa_gdn_e159', 'adah2sa_gdn_e160']`
  - `Atomic Single-Touch Holdout (gate-holdout)`: `HOLDOUT_PROMOTE` (`ledger = VERIFIED`, `8` hash-chained entries)
