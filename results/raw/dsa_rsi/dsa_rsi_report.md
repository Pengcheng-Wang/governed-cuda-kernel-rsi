# Round-31 RSI (`E157` / `v148`): DeepSeek Sparse Attention (`DSA`), Online KL-Distilled Lightning Indexer, & `GLM-5.2 IndexShare` (`arXiv:2512.02556` / `arXiv:2602.15763`)

## 1. Upstream Pathology & Bug Audit (`fla/ops/dsa/naive.py`)
- **Zero Triton Kernels & `O(B * HI * T^2 + B * HQ * T^2)` Materialized `FP32` Tensor Explosion**: Upstream `fla/ops/dsa` provides `0` Triton kernels (`naive_dsa_indexer` and `naive_dsa`), materializing full `[HI, T, T]` and `[HQ, T, T]` `FP32` matrices and expanding `k, v` by `G = HQ // H = 8` (`2209.4 MiB` vs `387.9 MiB` = **`10.98x` memory reduction**).
- **Complete Gradient Disconnect on `(q_idx, k_idx, w_idx)` (`grad is None`)**: Upstream discards indexer logits after `torch.topk`, leaving `q_idx.grad = None, k_idx.grad = None, w_idx.grad = None` (`upstream_indexer_grads_none = True`). Our online teacher-student KL distillation kernels restore exact, oracle-verified gradients (`rsi_indexer_grads_finite = True`).
- **`TF32` `einsum` Drift in `FP32` Mode**: Upstream `naive_dsa` suffers `max_rel_l2 = 2.73e-07` against the `FP64` reference oracle.
- **Zero-Spill `code/dsa_rsi_kernels.py` Cubin Audit**: `19` custom Triton kernels, `122` compiled specializations, **`0` spilling specializations**, **`0` over-168-register specializations**, `max_regs = 168 <= 168`, `max_spills = 0`, `max_smem_bytes = 25600 B`.
- **`FP64` Oracle & Mutation Gate**: `FP32 max_rel_l2 = 5.97e-07 < 1e-4`, `BF16 max_rel_l2 = 3.19e-03 < 1.5e-2`, and `3 / 3` (`100.0%`) adversarial mutations killed.

## 2. Track 1 — 8-Shape Training Kernel Matrix (`dsa_rsi` vs Upstream `naive_dsa`)
- Forward speedup: **`2.60x–5.22x`**
- Backward speedup: **`1.61x–6.20x`**
- Combined `FWD+BWD` speedup: **`2.32x–5.76x`**
- Peak HBM reduction: **`2.60x–10.98x`**

## 3. Track 2 — Single-Pass Multi-Warp Sparse Decoding (`dsa_decode_rsi`)
- Speedup vs upstream `naive_dsa`: **`89.26x–761.31x`**
- Speedup vs `dsa_rsi` full-chunk: **`11.21x–150.70x`**

## 4. Track 3 — 5-Stage `DeepSeekSparseAttentionBlockRSI` & `GLM-5.2 IndexShare` Ablation (`A0 -> A4`)
- Stage `A4` speedup vs `A0`: **`2.33x–3.04x`**

## 5. Track 4 — 8-GPU NVSwitch Context Parallelism (`DSA-CP = 8`)
- Ulysses `DSA-CP = 8` speedup vs Ring/AllGather baseline: **`13.42x–25.93x`**
- Ulysses `DSA-CP = 8` speedup vs 1-GPU upstream `naive_dsa`: **`13.21x–25.74x`**

## 6. Track 5 — Fixed-Wall-Clock Autoresearch `val_bpb` Loop (`12 s` Budget)
- `D_model = 1024`: `A4` executes `1731.3` vs `545.0` steps (`3.18x`), reducing `val_bpb` by an extra **`2.3737 bpb`**.
- `D_model = 2048`: `A4` executes `1386.0` vs `534.0` steps (`2.60x`), reducing `val_bpb` by an extra **`1.3229 bpb`**.
