# E136 (`v127`, Round-10 RSI): `Differential Transformer V1 & V2` 3-Repetition Summary

- **Frozen Evaluator SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`
- **All 9/9 Falsification Gates Passed**: `True` (`smoke: 9/9`, `rep1: 9/9`, `rep2: 9/9`, `rep3: 9/9`)
- **Suite 0**:
  - `FP32` max relative L2 error: `8.09e-07`
  - `BF16` max relative L2 error: `0.00567`
  - Upstream `multihead_diffattn.py:80,86` GQA `repeat_kv` pre-split bug relative L2 error: `0.9704` (`97.04%`)
  - `FP64` `O(h^2)` Richardson central-difference ratios: `7/7` inside `[3.980, 4.272]`
  - Red-Team exploits rejected: `4/4` (`RT1` V2 GQA `chunk(2)` bug, `RT2` V1 upstream `multihead_diffattn.py` GQA bug, `RT3` `allow_tf32` mutation, `RT4` omitted `dlambda_logit` & `SubLN` radial Jacobian)
  - Compiled Round-10 RSI Triton kernels: `20/20` with `0` cubin register spills (vs `384` spills in monolithic `[64x64x128]` `dKV` kernel)
- **Suite A (Vector-Pareto Ratchets)**:
  - Track 1 (`DiffAttn V2` across `s1..s4`): `1.08x–1.16x` speedup over Official `MultiheadFlashDiffV2` (`a0`) and `1.16x–1.28x` over `a2`
  - Track 2 (`DiffAttn V1 + Fused SubLN` across `m1..m3`): `1.27x–1.59x` speedup over Official `MultiheadFlashDiff2` (`4x-FA2`, `b0`) and `1.17x–1.24x` over `MultiheadFlashDiff1` (`2x-FA2`, `b1`)
- **Suite B (8-GPU NVSwitch Dual-LSE `Ring DiffAttn-CP = 8`, `T_global = 16,384`)**:
  - `Gate G9` 3-step multi-update `FP32` parity (`V2` & `V1`): `max_loss_rel_diff = 1.19e-07`, `max_param_rel_l2 = 8.91e-07`, `max_grad_rel_l2 = 8.79e-07` (vs `727.0%` error from shard-local differential subtraction)
  - Throughput: `17.40 ms` (`1.50x` faster than All-Gather Full-Sequence `26.05 ms`, `1.85x` faster than `2x` Separate Ring Passes `32.13 ms`)
- **Suite C (`12.0s` `8xA100 DDP` `val_bpb` RSI Loop & 5-Arm Multi-Needle Distractor-Noise Cancellation Scaling Law)**:
  - `12.0s` Loop: `c0 = 275` steps (`val_bpb = 1.9274`) -> `c5 = 720` steps (`2.62x`, `val_bpb = 1.3410`, `-0.5864 val_bpb`, auto-reverting `c3` & `c4`)
  - 5-Arm Matched-Step (`150` steps): `D4 (DiffAttn V2 RSI)` (`1.4861` clean / `2.0426` distractor `test_bpb`, `100.0%` needle acc) and `D3 (DiffAttn V1 RSI + SubLN)` (`2.1501` distractor `test_bpb`, `100.0%` needle acc) beat `D0 (Standard GQA Softmax)` (`2.1311`, `98.44%`), `D1 (Wrong GQA chunk(2))` (`2.4696`, `97.40%`), and `D2 (V1 No SubLN)` (`3.4257`, `91.15%`).
