# E150 (`v141`, Round-24 RSI): Wall Attention (`Tilde Research`) Zero-Spill SM80 Suite

## 1. Executive Summary
- **Architecture**: `Wall Attention: Length Generalization With Diagonal Gates` (`Tilde Research` — Timor Averbuch & Dhruv Pai; upstream `fla/ops/wall_attn` & `fla/layers/wall_attn.py`).
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch), `3` independent repetitions.
- **Frozen Cryptographic Evaluator**: `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
- **SM80 Cubin Register Spill Elimination**: Upstream `fla/ops/wall_attn` compiles `961/1,009` (`95.24%`) spilling specializations (`5,266,560 B` total stack spills, `max_regs = 255`, `max_stack = 36,808 B/thread`, `71` SMEM overflows up to `295,424 B`) vs **Round-24 RSI `0` spills across 100% of specializations (`max_regs = 168 <= 168`, `max_smem = 40,960 B`)**.
- **Differentiable `FP64` Oracle & `FP32` Precision**: Fixes upstream broken `naive_wall_attn` (`g.grad = None, g_scalar.grad = None`) and achieves **`317.9x` lower `FP32` output error (`3.23e-6` vs `1.03e-3`)** and **`166.27x` lower `FP32` gate-gradient error (`1.94e-5` vs `3.33e-3`)**.
- **Track 1 Training Kernel Speedup**: Up to **`1.295x` Fwd+Bwd** (`1.712x` Fwd, `1.205x` Bwd; `2.74x` vs unpruned upstream `K=128` spilling cliff).
- **Track 2 Fused KV-Cache Builder & Decode**: **`9.551x` faster `build_wall_kv_cache_rsi`** (`0.0` bitwise diff) and **`4.979x` faster end-to-end cached decode**.
- **Track 3 `WallAttentionBlockRSI` 5-Stage Ablation (`A0 -> A4`)**: Up to **`1.564x` (`D=1024, K=64`)** and **`1.27x` (`D=2048, K=128`)**.
- **Track 4 8-GPU NVSwitch `Wall-CP = 8` (`Ulysses` vs `Ring/AllGather`)**: **`6.08x` (`T=16,384`)** and **`7.154x` (`T=32,768`, `88.28%` parallel scaling efficiency, `0.0` `rel_l2` discrepancy)**.
- **Track 5 Fixed-Wall-Clock (`12.0s`) `val_bpb` Autoresearch**: **`1.471x` (`D=1024`)** and **`1.211x` (`D=2048`)** training throughput, achieving **`-0.992` extra `val_bpb` reduction (`0.1438` vs `1.1358`, `7.90x` lower final `val_bpb`)**.
