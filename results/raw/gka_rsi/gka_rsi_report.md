# Round-30 RSI (`E156` / `v147`): Gated KalmaNet (`GKA`) — A Fading Memory Layer Through Test-Time Ridge Regression (`arXiv:2511.21016`)

## 1. Upstream Pathology & Bug Audit (`fla/ops/gka` & `fla/layers/gka.py`)
- **Severe Register Spilling (`29 / 58` Specializations, `76526` Total Spills, `max_regs = 255`, `max_spills = 10028`/Thread)**: Upstream `chunk_gka_solve_fwd_kernel` (`8/8` spilling, up to `282` spills/thread) and `chunk_gka_bwd_kernel_dk_intra` (`12/12` spilling, up to `10,028` spills/thread and `114,688 B` SMEM) suffer catastrophic local-memory thrashing on `NVIDIA A100-SXM4-80GB` (`SM80`).
- **`SimpleGLABackwardGuard` 16-Bit Backward Crash on `K < 64` or `V < 32`**: Upstream `chunk_gka` delegates its value-readout stage to `chunk_simple_gla`, which crashes in `.backward()` for 16-bit inputs with `K < 64` or `V < 32` (`crashed = True`). Our self-contained `chunk_gka_rsi` succeeds with finite, oracle-verified gradients (`rsi_succeeded = True`).
- **`TF32` Drift in `FP32` Mode**: Upstream `fla/ops/gka` omits `input_precision="ieee"` in `FP32`, causing `max_rel_l2 = 1.99e-03` against the `FP64` reference oracle.
- **Zero-Spill `code/gka_rsi_kernels.py` Cubin Audit**: `26` custom Triton kernels, `225` compiled specializations, **`0` spilling specializations**, **`0` over-168-register specializations**, `max_regs = 168 <= 168`, `max_spills = 0`, `max_smem_bytes = 69632 B`.
- **`FP64` Oracle & Mutation Gate**: `FP32 max_rel_l2 = 5.26e-07 < 1e-3`, `BF16 max_rel_l2 = 1.34e-02 < 1.5e-2`, and `3 / 3` (`100.0%`) adversarial mutations killed.

## 2. Track 1 — 8-Shape Training Kernel Matrix (`chunk_gka_rsi` vs Upstream `chunk_gka`)
- Forward speedup: **`1.11x–2.74x`**
- Backward speedup: **`1.43x–4.74x`**
- Combined `FWD+BWD` speedup: **`1.33x–3.60x`**

## 3. Track 2 — Single-Pass Multi-Warp Online Decoding (`gka_decode_rsi`)
- Speedup vs upstream `chunk_gka`: **`2.49x–9.97x`**
- Speedup vs upstream `fused_recurrent_gka`: **`1.27x–4.90x`**

## 4. Track 3 — 5-Stage `GatedKalmaNetBlockRSI` Layer Periphery Ablation (`A0 -> A4`)
- Stage `A4` speedup vs `A0`: **`1.98x–2.57x`**

## 5. Track 4 — 8-GPU NVSwitch Context Parallelism (`GKA-CP = 8`)
- Ulysses `GKA-CP = 8` speedup vs Ring/AllGather baseline: **`4.56x–16.18x`**
- Ulysses `GKA-CP = 8` speedup vs 1-GPU upstream `chunk_gka`: **`3.36x–15.22x`**

## 6. Track 5 — Fixed-Wall-Clock Autoresearch `val_bpb` Loop (`12 s` Budget)
- `D_model = 1024`: `A4` executes `1535.0` vs `851.0` steps (`1.80x`), reducing `val_bpb` by an extra **`0.4905 bpb`**.
- `D_model = 2048`: `A4` executes `1135.0` vs `490.3` steps (`2.31x`), reducing `val_bpb` by an extra **`0.3242 bpb`**.
