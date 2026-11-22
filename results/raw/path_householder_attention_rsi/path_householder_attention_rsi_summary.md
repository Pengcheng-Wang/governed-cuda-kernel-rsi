# E134 (`v125`, Round-8 RSI): `PaTH Attention` & `PaTH-FoX` Householder Transformations — Hardware Verification Summary

- **Mode**: `full` (`3` independent OS-process 8-GPU repetitions)
- **Validation Gates**: **`9/9` PASSED** (`all_gates_passed = True`)
- **Immutable Evaluator SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`

## Headline Hardware & Scaling Results (3-Rep Median / IQR)

1. **Suite 0 (Forensics, Cubin Zero-Spill Audit & Exactness)**: `4/4` red-team exploits rejected; `22/22` Round-8 RSI Triton kernels (`40` specializations, max `255` regs) compiled with **`0` cubin spills** (vs **`3732` spills** in official `fla-org` `chunk_cumprod_householder_bwd_kernel` at `K=128`); `FP32 max_rel_l2 = 9.95e-07`, `BF16 max_rel_l2 = 0.0079`; `8/8` `FP64` `O(h^2)` Richardson checks passed.
2. **Suite A — Track 1 (Core `PaTH-FoX` Householder Attention Fwd+Bwd across `s1..s4`)**: `41.024 -> 3.554 ms` (**`11.54x`** geomean speedup over Eager, **`10.97x`** over `torch.compile`, and **`2.52x`** geomean / up to **`5.13x`** at `K=128` over Official `fla-org/flash-linear-attention` `parallel_path_attn`, saving **`10.8 MiB`** peak HBM at `T=4096`).
3. **Suite A — Track 2 (End-to-End `PaTH-FoX` Block: `ShortConv3+SiLU+L2Norm+BetaSigmoid Frontend + PaTH-FoX Attention`)**: `50.920 -> 5.064 ms` (**`10.05x`** over Eager, **`2.88x`** over Eager Frontend + Official `fla-org`, and **`2.82x`** over `torch.compile` Frontend + Official `fla-org`).
4. **Suite B (`8-GPU` NVSwitch Distributed Context Parallelism `Ring PaTH-CP = 8` at `T_global = 8192, K = 128`)**: `Gate G9` 3-step `FP32` parity `max_param_rel_l2 = 1.12e-06` (vs `0.3105` shard-local no-handoff bug); delivers **`55.56x`** over All-Gather Eager, **`8.33x`** over All-Gather Official `fla-org` (cutting peak HBM **`2.30x`**, saving **`370.9 MiB/rank`**), and **`3.49x`** over Dense-Matrix Ring Handoff.
5. **Suite C (`15.0s` `8xA100 DDP` `PaTH-FoX` `val_bpb` Autonomous RSI Loop)**: `gen2_rsi_round8_ut_duality_path_fox` runs **`2.45x`** more updates (`415 -> 1016`, vs `794` for Official `fla-org`), improving `val_bpb` by **`-0.1682`** and untouched single-touch `test_bpb` by **`-0.1632`** (`State-Tracking Acc = 99.46%`).

## Matched-Step Position-Encoding & 2x Length-Extrapolation Scaling Law

| Arm | `val_bpb` | Clean `test_bpb` | `2x` Extrap `test_bpb` | Extrap `Δbpb` | Clean State Acc | `2x` Extrap State Acc |
|---|---:|---:|---:|---:|---:|---:|
| `P0_nope_causal_attention` | `2.7135` | `2.7371` | `2.9532` | `+0.2161` | `54.93%` | `43.76%` |
| `P1_rope_fixed_rotary_attention` | `2.1111` | `2.1158` | `2.8288` | `+0.7130` | `58.73%` | `45.86%` |
| `P2_alibi_static_decay_attention` | `2.1782` | `2.1851` | `2.3820` | `+0.1970` | `61.26%` | `53.26%` |
| `P3_fox_adaptive_forgetting_only` | `2.1934` | `2.1867` | `2.3145` | `+0.1279` | `60.60%` | `51.89%` |
| `P4_path_fox_rsi_householder_attention` | `2.0357` | `2.0285` | `2.1936` | `+0.1651` | `91.66%` | `89.96%` |
