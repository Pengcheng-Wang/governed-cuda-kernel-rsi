# Round-32 RSI (`E158` / `v149`): Mixture of Block Attention (`MoBA`, `arXiv:2502.13189`, Moonshot AI / Kimi Team)

## 1. Upstream Pathology & Bug Audit (`fla/ops/moba/parallel.py` & `fla/layers/moba.py`)
- **Zero Triton Kernels, Two-Stream `_flash_attn_varlen`, 12-Op Eager LSE Merge, Host-Sync `.nonzero()`, & HBM Duplication**: Upstream `fla/ops/moba` provides `0` Triton kernels, duplicates `(topk - 1)x` Q and `2x` KV in HBM, and stalls on `.nonzero()` (`1317.2 MiB` vs `632.4 MiB` = **`2.08x` memory reduction**).
- **Deterministic GQA & `FP32` `RuntimeError` Crashes**: Upstream `MoBA` crashes on GQA (`num_kv_heads < num_heads`, `crashed = True`) and `FP32` inputs (`crashed = True`), and suffers `BF16` mean + `TF32` `einsum` gate top-$k$ boundary flips (`max_rel_l2 = 1.87e-02` vs `2.51e-03`).
- **Zero-Spill `code/moba_rsi_kernels.py` Cubin Audit**: `14` custom Triton kernels, `139` compiled specializations, **`0` spilling specializations**, **`0` over-168-register specializations**, `max_regs = 168 <= 168`, `max_spills = 0`, `max_smem_bytes = 40960 B`.
- **`FP64` Oracle & Mutation Gate**: `FP32 max_rel_l2 = 1.14e-06 < 1e-4`, `BF16 max_rel_l2 = 2.51e-03 < 1.5e-2`, and `3 / 3` (`100.0%`) adversarial mutations killed.

## 2. Track 1 — 8-Shape Training Kernel Matrix (`parallel_moba_rsi` vs Upstream `parallel_moba`)
- Forward speedup: **`2.42x–10.54x`**
- Backward speedup: **`1.64x–4.22x`**
- Combined `FWD+BWD` speedup: **`1.90x–5.71x`**
- Peak HBM reduction: **`1.32x–2.08x`**

## 3. Track 2 — Single-Pass Centroid-Routed Block-Sparse Decoding (`moba_decode_rsi`)
- Speedup vs upstream `parallel_moba`: **`19.97x–32.95x`**
- Speedup vs `parallel_moba_rsi` full-chunk: **`3.95x–11.69x`**

## 4. Track 3 — 5-Stage `MoBABlockRSI` & `BlockShare` Ablation (`A0 -> A4`)
- Stage `A4` speedup vs `A0`: **`3.12x–3.75x`**

## 5. Track 4 — 8-GPU NVSwitch Context Parallelism (`MoBA-CP = 8`)
- Ulysses `MoBA-CP = 8` speedup vs Ring/AllGather baseline: **`5.32x–6.96x`**
- Ulysses `MoBA-CP = 8` speedup vs 1-GPU upstream `parallel_moba`: **`4.29x–6.23x`**

## 6. Track 5 — Fixed-Wall-Clock Autoresearch `val_bpb` Loop (`12 s` Budget)
- `D_model = 1024`: `A4` executes `2103.3` vs `650.0` steps (`3.24x`), reducing `val_bpb` by an extra **`0.1654 bpb`**.
- `D_model = 2048`: `A4` executes `2073.3` vs `643.3` steps (`3.22x`), reducing `val_bpb` by an extra **`0.0954 bpb`**.
