# E137 (`v128`, Round-11 RSI): `nGPT` Hypersphere Representation Learning & Riemannian Retraction Summary

- **Frozen Evaluator `code/rsi_kernel_harness.py` SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`
- **All 9 Falsification Gates Passed**: `True` (`1` smoke + `3` full 8-GPU repetitions)
- **Zero-Spill Triton Kernels**: `19/19` kernels at `0` cubin spills (vs `1682` spills for monolithic `D=128` `dKV` negative control)
- **Track 1 Core Hypersphere Ops Geomean Speedup**: `24.309x` vs Upstream `NVIDIA/ngpt`, `1.192x` vs `torch.compile`
- **Track 2 Full `nGPT` Block + Weight Retraction Geomean Speedup**: `5.164x` vs Upstream `NVIDIA/ngpt` + `normalize_matrices()`, `1.147x` vs `torch.compile`
- **8-GPU `Ring nGPT-CP=8` + ZeRO-1 Shard-Retracted Parity (`Gate G9`)**: `max_loss_rel_diff = 6.25e-08`, `max_param_rel_l2 = 4.73e-07`, `max_grad_rel_l2 = 1.92e-06`, `max_unit_norm_err = 1.79e-07`
- **Fixed-Wall-Clock 8xA100 DDP `val_bpb` RSI Improvement**: `1.4812 -> 0.0963` (`-1.3849 bpb`), Single-Touch `test_bpb`: `1.5008 -> 0.0976` (`recall_acc = 99.8%`)
