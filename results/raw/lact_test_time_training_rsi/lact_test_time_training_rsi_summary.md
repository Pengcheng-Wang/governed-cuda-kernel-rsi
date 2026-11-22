# E131 (`v122`): `LaCT: Test-Time Training Done Right` (`arXiv:2505.23884`, ICLR 2026) Round-5 RSI Summary

- **Immutable Evaluator SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e` (`verified = True`)
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`1410.0 MHz` median SM clock, `89.6 W` peak power)
- **All 9 Validation Gates Passed**: `True`

## 1. Suite 0 — Official `a1600012888/LaCT` Bug Audits, Zero-Spill Cubins, & Exactness Oracles
- **Official Concatenated `[dW0; dW2]` `Muon` Bug (`ttt_operation_fused_kernel.py` L164/268)**: `0.785913` (`78.59%` `rel_l2` spectral corruption) vs **`9.20e-07`** in our independent 3-matrix batched `BatchedCheckpointedMuonNS5Fn`.
- **Official 3-Accumulator Triton Spill & SRAM Overflow Audit**: `4582` (`128x128`) and `4024` (`128x256`) cubin spills + `OutOfResources` at `128x256x64, stages=3`, vs **`0` cubin spills** across all `9` Round-5 RSI Triton kernels.
- **Official `l2norm_triton_kernels.py` `0/0 = NaN` Backward Bug**: `NaN` reproduced on zero-initialized fast-weight rows vs `0` `NaN`s in division-free `FusedL2NormStepFn`; `max_fp32_rel_l2 = 1.60e-06`, `max_bf16_rel_l2 = 0.02226`, `O(h^2)` Richardson `3.9996..4.0002` across `7` derivatives.

## 2. Suite A — Round-5 Vector-Pareto Kernel Ratchets (3-Rep Medians)
- **Track 1 (`Single-Chunk SwiGLU-TTT Apply + Split-H Grad2 + Batched Muon-NS5 + L2Norm`)**: `17.766 -> 7.737 ms` (**`2.30x`** vs Eager).
- **Track 2 (`Multi-Chunk Hybrid LaCT(Muon+Momentum) + SWA + Fused Gate`)**: `66.934 -> 25.797 ms` (**`2.60x`** vs Eager, **`1.53x`** vs `torch.compile`, `5000.3 -> 4207.6 MiB` peak HBM).

## 3. Suite B — 8-GPU NVSwitch Sub-Chunk `CP = 8` + 1-Hop P2P Halo `SWA` (`S_global = 65,536, C_global = 16,384`)
- **Gate G9 (`8-GPU CP = 8` vs `1-GPU` Full-Sequence Parity)**: `max_rel_l2 = 0.004208` across `(out, dW0, dW1, dQ, dK, dmom)` (vs `27.5615` `dW0` inflation when base fast-weight gradients are erroneously AllReduced inside `CP`).
- **8-GPU Long-Context Scaling**: `182.85 -> 35.12 ms` (**`5.21x`** vs `AllGather` Replicated, **`2.12x`** vs RSI `AllGather` Replicated, **`1.57x`** vs `AllGather-SWA`), **`9.71x`** less NVSwitch communication, and **`17940.6 MiB/rank`** HBM saved.

## 4. Suite C — Fixed-`16.0s` `val_bpb` RSI Loop & Matched-Step Beyond-Window Associative Recall Law
- **Fixed-`16.0s` `val_bpb` RSI Loop**: `Gen 0 -> Gen 2` increases steps **`84 -> 228` (`2.71x`)** and improves `val_bpb` `3.6535 -> 3.0547` (**`-0.5988 val_bpb`**, untouched `test_bpb = 3.0608`, untouched beyond-`SWA` recall = **`99.15%`**), auto-reverting `Gen 3` (`L2Norm` removal -> `3.3771`, `REVERT_VAL_BPB`) and `Gen 4` (`C = 32` small-chunk overhead -> `3.7072`, `REVERT_SLOWER`).
- **Matched-Step (`220` Steps) Iso-Param (`~2.60M`) Fast-Weight Law**: `M0_swa_only` (`3.1211`, recall `6.38%`) vs `M2_swiglu_gd_no_muon` (`3.4088`, recall `7.36%`) vs `M1_linear_delta` (`3.0931`, recall `93.03%`) vs **`M3_lact_swiglu_muon_momentum` (`3.0657`, recall `98.83%`)**.
