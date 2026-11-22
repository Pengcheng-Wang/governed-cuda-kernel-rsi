# E135 (`v126`, Round-9 RSI): `Mixture-of-Recursions (MoR)` Dynamic Latent-Depth Training Kernels — Hardware Verification Summary

- **Mode**: `full` (`3` independent OS-process 8-GPU repetitions)
- **Validation Gates**: **`9/9` PASSED** (`all_gates_passed = True`)
- **Immutable Evaluator SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`

## Headline Hardware & Scaling Results (3-Rep Median / IQR)

1. **Suite 0 (Upstream `raymin0223/mixture_of_recursions` Forensics, Cubin Zero-Spill Audit & Exactness)**: `4/4` red-team exploits rejected; `16/16` Round-9 RSI Triton kernels (`37` specializations, max `216` regs) compiled with **`0` cubin spills** (vs `5638` spills in monolithic `[64x128]` `dKV`); `FP32 max_rel_l2 = 3.94e-07`, `BF16 max_rel_l2 = 0.0042`; `7/7` `FP64` `O(h^2)` Richardson checks passed.
2. **Suite A — Track 1 (Position-Indexed Rectangular Causal-Gather `Flash-MoR` `Q_act[K_r] x KV_shared[T]` Fwd+Bwd across `s1..s4`)**: `3.107 -> 1.725 ms` (**`1.80x`** geomean speedup over Official `kv_sharing` full-sequence `FA2`+gather and **`2.45x`** over 4D-masked SDPA, saving **`117.9 MiB`** peak HBM at `T=8192`).
3. **Suite A — Track 2 (End-to-End `Middle-Cycle` `N_r = 3` Recursive `MoR` Stack Fwd+Bwd across `m1..m3`)**: `14.336 -> 9.441 ms` (**`1.52x`** over Official `pad_sequence`+4D-mask loop, saving **`85.2 MiB`** peak HBM, and **`1.28x`** over Official `kv_sharing` full-sequence clone).
4. **Suite B (`8-GPU` NVSwitch Recursion-Pinned Tied-Weight `FSDP2` + Whole-Document `MoR-LPT AllToAllv` at `B_global = 32, T = 2048`)**: Cut tied-weight `FSDP2` `AllGather` collectives **`6 -> 1`** (**`6.0x`** reduction), collapsed rank active-token FLOP skew **`4.00x -> 2.00x`**, verified 3-step `Gate G9` parity (`max_param_rel_l2 = 6.79e-07` vs `0.8631` rank-local balancing router gradient bias), and achieved **`3.61x`** over Naive `FSDP2`+Pad4D, **`3.32x`** over Naive `FSDP2`+Official `KV-Share`, and **`2.77x`** pure `MoR-LPT` NVSwitch gain.
5. **Suite C (`12.0s` Strict-Causal `8xA100 DDP` `val_bpb` Autonomous RSI Loop)**: Verified `strict_causal_prefix_max_abs_diff = 0.0e+00` (vs `0.0312` batch top-k causal leakage); `gen2_rsi_round9_rect_flash_mor_kv_share` ran **`1.38x`** more updates (`353 -> 487`), improving `val_bpb` by **`-0.0967`** and untouched single-touch `test_bpb` by **`-0.0948`** (`3-Hop Acc = 100.00%`).

## Matched-Step Parameter-Sharing & Multi-Hop Latent-Depth Reasoning Scaling Law

| Arm | Wall (s) | `val_bpb` | Clean `test_bpb` | `1-Hop` Acc | `3-Hop` Acc | `E[r | 1-Hop]` | `E[r | 3-Hop]` | Depth Gap `Δr` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M0_shallow_vanilla_iso_param_nr1` | `2.16s` | `2.0641` | `2.0729` | `92.47%` | `35.60%` | `1.00` | `1.00` | `+-0.00` |
| `M1_fixed_recursive_middle_cycle_nr3` | `3.15s` | `1.6927` | `1.7023` | `100.00%` | `100.00%` | `3.00` | `3.00` | `+0.00` |
| `M2_mor_token_choice_rec_cache_nr3` | `3.42s` | `2.0382` | `2.0678` | `100.00%` | `8.91%` | `2.34` | `2.16` | `+-0.18` |
| `M3_mor_expert_aux_bce_rec_cache_nr3` | `3.54s` | `2.0477` | `2.0725` | `100.00%` | `9.58%` | `2.00` | `3.00` | `+1.00` |
| `M4_mor_causal_kv_share_flash_mor_rsi_nr3` | `3.34s` | `1.6662` | `1.6787` | `100.00%` | `100.00%` | `2.00` | `3.00` | `+1.00` |
