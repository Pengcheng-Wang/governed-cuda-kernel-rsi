# Round-29 RSI (`E155`, `v146`) — `Comba: Improving Bilinear RNNs with Closed-loop Control` (`arXiv:2506.02475`)

## 1. Upstream Pathology & SM80 Cubin Audit
- **Inner-Decay L2Norm Erasure Bug (`fla/layers/comba.py:228-250`)**: Upstream computes `p = k * self.decay.sigmoid()` *before* `l2norm_fwd(p)`, erasing the positive scalar $\sigma(\text{decay}_h)$ (`||l2norm(0.9*k) - l2norm(0.1*k)||_rel = 8.32e-07`, `decay.grad max_abs = 9.56e-06` when `use_output_correction=False`). Our `fused_comba_qk_l2norm_corr_rsi` normalizes first (`k_hat = l2norm(k)`), then computes `p = s_h * k_hat` and `q_tilde = q_hat - D_h * p` in 1 launch (`decay.grad max_abs = 6.09e-02`).
- **Upstream SM80 Cubin Spills**: `16 / 53` spilling specializations (`2120` total spills, `max_regs = 255`, `max_spills = 594`, `max_smem = 163856 B`), led by `prepare_wy_repr_bwd_kernel` (`4/4` spilling, `594` max spills).
- **Round-29 RSI SM80 Cubin Audit**: `0 / 212` spilling specializations (`0` spills, `over168 = 0`, `max_regs = 168 <= 168`, `max_smem = 24576 B`) across `25` custom Triton kernels.
- **FP64 Oracle & Mutation Gate**: `FP32 max_rel_l2 = 3.00e-06 < 1e-5` (vs upstream TF32 drift `3.82e-03`), `BF16 max_rel_l2 = 5.73e-03 < 1e-2`, and `3/3` (`100%`) adversarial mutations killed.

## 2. Track 1 — 8-Shape Training Kernel Matrix (3-Rep Mean ± Std)
- **Forward Speedup vs Upstream `chunk_comba`**: `1.55x .. 2.82x` (`8/8` shape wins)
- **Backward Speedup vs Upstream `chunk_comba`**: `1.73x .. 2.12x` (`8/8` shape wins)
- **Total Fwd+Bwd Speedup vs Upstream `chunk_comba`**: `1.67x .. 2.29x` (`8/8` shape wins)

## 3. Track 2 — Single-Pass Multi-Warp Online Decoding (`comba_decode_rsi`)
- **Speedup vs Upstream `fused_recurrent_comba`**: `2.02x .. 2.20x`
- **Speedup vs Upstream `chunk_comba` (`BT=64` padded)**: `16.73x .. 18.49x`

## 4. Track 3 — 5-Stage `CombaBlockRSI` Layer Ablation (`A0 -> A4`)
- **End-to-End Fwd+Bwd Speedup (`A4 vs A0`)**: `1.98x .. 2.07x`

## 5. Track 4 — 8-GPU NVSwitch Context Parallelism (`Comba-CP = 8`)
- **Speedup vs 8-GPU Ring/AllGather Baseline**: `2.59x .. 2.88x`
- **Speedup vs 1-GPU Upstream `chunk_comba`**: `1.24x .. 1.51x`

## 6. Track 5 — Fixed-Wall-Clock `val_bpb` Autoresearch Loop (`12.0s` Budget)
- **Optimizer Step Ratio (`A4 vs A0`)**: `1.75x .. 1.94x`
- **Extra `val_bpb` Reduction (`A0 - A4`)**: `0.4966 .. 1.5705 bpb`
