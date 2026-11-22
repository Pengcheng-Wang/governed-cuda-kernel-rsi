# E141 (`v132`, Round-15 RSI): RWKV-7 'Goose' & Generalized Diagonal-Plus-Low-Rank (`DPLR`, `arXiv:2503.14456` / `arXiv:2506.02475`)

## 1. Executive Summary (3-Repetition 8xA100-SXM4-80GB Hardware Medians)
- **Frozen Evaluator (`G1`)**: `rsi_kernel_harness.py` `SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`
- **Zero-Spill SM80 Compilation (`G2`)**: `21/21` kernels (`89` specializations) at **`0` cubin spills** vs **`27152` total spills across `22/22` (`100%`) configs** in upstream `fla/ops/generalized_delta_rule/dplr` (`2,840–3,416` spills in `chunk_dplr_fwd_A_kernel_intra_tensorcore`, `1,902–2,312` spills in `chunk_dplr_bwd_kernel_intra_tensorcore`, `72–262` spills in `prepare_wy_repr_bwd_kernel`).
- **Overflow-Free Boundary-Anchored 16x16 Sub-Chunking (`G3`)**: Eliminates **`256` `NaN`s** produced by upstream `chunk_dplr_fwd_A_kernel_intra_tensorcore`'s mid-chunk offset `b_offset = gi[BT//2]` (`0` `NaN`s in Round-15 RSI, `0.0` future-token leakage, `FP32 max_rel_l2 = 2.92e-06`, `BF16 max_rel_l2 = 0.0067`).
- **Suite A Vector-Pareto (`G5` & `G6`)**: Track 1 (`Core Generalized DPLR Layer`) achieves **`29.28x`** geomean speedup over Upstream-Pattern Spill (`25.00x` over `torch.compile`, `28.86x` over Eager Chunk, `774.06x` over Sequential Recurrent); Track 2 (`Full RWKV7GooseBlock`) achieves **`14.40x`** over Baseline (`12.72x` over `torch.compile`).
- **Suite B 8-GPU NVSwitch `DPLR-CP = 8` (`G9`, `T_global = 8,192`)**: `max_loss_rel_diff = 9.06e-08`, `max_param_rel_l2 = 4.69e-06`, `max_grad_rel_l2 = 1.25e-05` (vs `0.5448` error without Dual-Source Channel-Affine Monoid handoff).
- **Suite C Fixed-Wall-Clock `val_bpb` RSI & 5-Arm Scaling (`G7` & `G8`)**: `val_bpb` improved **`1.0401 -> 0.0013` (`-1.0388 val_bpb`)**, single-touch `test_bpb`: **`1.0262 -> 0.0012` (`-1.0249 test_bpb`, recall accuracy `96.56% -> 100.00%`)**.

