# E129 (`v120`): DeepSeek `mHC` (`Manifold-Constrained Hyper-Connections`, `arXiv:2512.24880`) Round-3 RSI Summary

- **Immutable Evaluator SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e` (`verified = True`)
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`1410.0 MHz` median SM clock, `349.7 W` peak power)
- **All 9 Validation Gates Passed**: `True`

## 1. Suite A — Round-3 Vector-Pareto Kernel Ratchets (3-Rep Medians)
- **Track 1 (`mHC Coeff + Log-Sinkhorn-20`)**: `16.589 -> 1.883 ms` (**`8.81x`** vs Eager, **`1.31x`** vs `torch.compile`, **`3.76x`** lower HBM; `col_max exp` caught as `REJECT_CORRECTNESS`).
- **Track 2 (`mHC Stream Sandwich`)**: `33.480 -> 4.581 ms` (**`7.31x`** vs Eager, **`4.49x`** vs `torch.compile`, **`1.11x`** faster & **`1120 MiB`** leaner than DeepSeek's 2-kernel split).

## 2. Suite B — Selective `L_r^* = 4` Stream Checkpointing & 8-GPU `1.07B` `mHC-FSDP2` (`S = 2,048`, `65,536` tok/step)
- **DeepSeek Eq. 20 Verification**: Closed-form `L_r^* = 4.0` matches empirical minimum `L_r = [4, 4, 4]` across all 3 reps (`9734.5 -> 7949.1 MiB`, symmetric `L_r=2` `8333.1` == `L_r=8` `8333.1 MiB`).
- **8-GPU `1.07B` `mHC-FSDP2` Scaling**: `Rung 3` (`402.7 ms`, `162755 tok/s`) accelerates Eager `mHC` (`839.5 ms`) by **`2.09x`**, cuts `4x`-stream overhead from **`+142.9%`** to **`+10.94%`** (`L_r=1`) / **`+16.52%`** (`L_r^*=4`), and saves **`44037.2 MiB/rank`** (`59.87 -> 16.86 GiB/rank`, leaner than `1-Stream`'s `18.30 GiB/rank`).

## 3. Suite C — Fixed-`12.0s` `val_bpb` Architecture-Kernel Co-Design Loop
- **`Gen 0 -> Gen 4`**: `47 -> 131` updates (**`2.81x`**), `val_bpb` `3.3868 -> 1.7498` (**`-1.6388 val_bpb`** vs Eager, **`-0.3208 val_bpb`** vs `torch.compile`, untouched `test_bpb = 1.7523`), rejecting `Gen 3` symmetric saddle (`cos = 1.0000`) and beating width-matched oracle-LR `1-Stream` at matched steps by **`-1.0048 val_bpb`** with `124,853` fewer parameters.
