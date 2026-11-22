# E130 (`v121`): DeepSeek `Engram` (`Conditional Memory via Scalable Lookup`, `arXiv:2601.07372`) + `mHC` Round-4 RSI Summary

- **Immutable Evaluator SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e` (`verified = True`)
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`1410.0 MHz` median SM clock, `90.5 W` peak power)
- **All 9 Validation Gates Passed**: `True`

## 1. Suite 0 — Numerical Pitfalls, Exactness Oracles, & `JIT-Lazy-Adam` Proof
- **Surjective Tokenizer Compression (`P: V -> V'`)**: `129,280 -> 99,546` (**`23.00%`** vocabulary reduction), **`0`** bitwise `int64` prime hash mismatches vs `engram_demo_v1.py` CPU NumPy, **`0.0`** packed-document cross-doc leakage across both `NgramHash` and `ShortConv` fwd/bwd.
- **Pre-Forward `JIT-Lazy-Adam(lr=5x, wd=0)` vs `SparseAdam`**: **`1.82e-06`** full-table 50-step `rel_l2` vs dense `Adam` (**`361000x`** more accurate than `torch.optim.SparseAdam`'s `0.657` momentum-drop drift).
- **Demo Numerical Pitfalls Uncovered**: `engram_demo_v1.py` `nn.RMSNorm(eps=None)` drifts by `0.0060` in `BF16` (`eps = 0.0078125`), and its forward-only `signed_sqrt` gate creates a `125.0` backward Jacobian cliff at `|s|=1e-6` (`0.0872` `BF16` `dH` jitter) vs Paper `Eq. 6` smooth gate (`0.00167`, `O(h^2)` Richardson `3.9867..4.0247`, `0` cubin spills).

## 2. Suite A — Round-4 Vector-Pareto Kernel Ratchets (3-Rep Medians)
- **Track 1 (`Engram Prime Hash + Gather + Zipfian Bwd + JIT-Lazy-Adam`)**: `4.259 -> 0.791 ms` (**`5.38x`** vs Demo baseline, **`2.18x`** vs GPU-Hash + Dense `Adam`).
- **Track 2 (`M=4 Context-Gate + Dilated Causal ShortConv Fwd+Bwd`)**: `8.279 -> 2.545 ms` (**`3.26x`** vs Eager Demo, **`1.65x`** vs `torch.compile`, **`1.13x`** lower peak HBM).

## 3. Suite B — 8-GPU NVSwitch Sharded `Dedup-AllToAllv` + Layer-2 Async Stream Prefetch (`8.39M` Prime Rows, `S = 2,048`)
- **Gate G9 (`1-GPU` vs `8-GPU` Sharded+Prefetch Parity)**: `max_3step_loss_rel_diff = 1.72e-05`, `reassembled_8shard_3step_master_table_rel_l2 = 0.00e+00`.
- **8-GPU Scaling (`65,536` tok/step)**: `r3` Layer-2 Async Prefetch (`141.69 ms`, `462528 tok/s`) accelerates Replicated Demo (`364.52 ms`) by **`2.57x`**, Naive `AllToAll` (`162.21 ms`) by **`1.14x`**, and Sync `Dedup-AllToAllv` (`142.72 ms`) by **`1.01x`**, cutting NVSwitch Engram payload **`74.3x`** and saving **`16251.6 MiB/rank`**.

## 4. Suite C — Fixed-`12.0s` `val_bpb` RSI Loop & Sparsity-Allocation U-Curve (`arXiv:2601.07372` Sec. 3)
- **Fixed-`12.0s` `val_bpb` Co-Design**: `Gen 0 -> Gen 4` increases steps **`67 -> 168` (`2.51x`)** and improves `val_bpb` `3.6052 -> 3.1677` (**`-0.4374 val_bpb`** vs Eager, **`-0.0242 val_bpb`** vs `torch.compile`, untouched `test_bpb = 3.1768`).
- **Matched-Step Iso-Param / Iso-FLOPs Sparsity Allocation Law**: `rho = 1.00` (`Pure MoE`: `3.2456`) -> **`rho* = 0.75` (`Optimal Hybrid`: `3.1611`)** -> `rho = 0.25` (`Memory-Heavy`: `3.1774`) -> `rho = 0.00` (`Pure Engram`: `3.1838`), beating `Pure MoE` by **`-0.0845 bpb`**, `Pure Engram` by **`-0.0227 bpb`**, `Iso-FLOPs Dense` by **`-0.0927 bpb`**, and `Iso-Param Wider Dense` by **`-0.0839 bpb`**.
