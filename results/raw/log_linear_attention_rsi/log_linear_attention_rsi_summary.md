# E138 (`v129`, Round-12 RSI): `Log-Linear Attention` (`H-Mamba-2` & `H-GDN`) 3-Repetition Summary

- **Frozen Evaluator `code/rsi_kernel_harness.py` SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`
- **All 9 Falsification Gates Passed**: `True` (`1` smoke + `3` full 8-GPU repetitions)
- **Bitwise XOR `clz` Fenwick Level Identity (`512x512` pairs)**: `0` mismatches
- **Zero-Spill Triton Kernels**: `17/17` kernels at `0` cubin spills (vs `1210` spills for monolithic 8-level negative control)
- **Track 1 `Log-Linear Mamba-2` Geomean Speedup**: `9.299x` vs Upstream `HanGuo97/log-linear-attention`, `1.419x` vs `torch.compile`
- **Track 2 `Log-Linear Gated DeltaNet` Geomean Speedup**: `2.180x` vs Upstream `HanGuo97/log-linear-attention`, `1.082x` vs `torch.compile`
- **8-GPU `Fenwick-CP=8` Parity (`Gate G9`)**: `max_loss_rel_diff = 1.38e-05`, `max_param_rel_l2 = 1.52e-04`, `D64 speedup = 4.158x`, `D128 speedup = 2.774x`
- **Fixed-Wall-Clock 8xA100 DDP `val_bpb` RSI Improvement**: `4.4539 -> 1.2796` (`-3.1646 bpb`), Single-Touch `test_bpb`: `4.4304 -> 1.2812` (`recall_acc = 100.00%`)
