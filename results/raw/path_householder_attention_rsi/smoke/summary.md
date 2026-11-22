# E134 (`v125`, Round-8 RSI): `PaTH Attention` & `PaTH-FoX` Householder Transformations — Hardware Verification Summary

- **Mode**: `smoke` (`1` independent OS-process 8-GPU repetitions)
- **Validation Gates**: **`9/9` PASSED** (`all_gates_passed = True`)
- **Immutable Evaluator SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`

## Headline Hardware & Scaling Results (3-Rep Median / IQR)

1. **Suite 0 (Forensics, Cubin Zero-Spill Audit & Exactness)**: `4/4` red-team exploits rejected; `22/22` Round-8 RSI Triton kernels (`40` specializations, max `255` regs) compiled with **`0` cubin spills** (vs **`3732` spills** in official `fla-org` `chunk_cumprod_householder_bwd_kernel` at `K=128`); `FP32 max_rel_l2 = 9.95e-07`, `BF16 max_rel_l2 = 0.0079`; `8/8` `FP64` `O(h^2)` Richardson checks passed.
2. **Suite A — Track 1 (Core `PaTH-FoX` Householder Attention Fwd+Bwd across `s1..s4`)**: `40.930 -> 3.547 ms` (**`11.54x`** geomean speedup over Eager, **`10.90x`** over `torch.compile`, and **`2.54x`** geomean / up to **`5.15x`** at `K=128` over Official `fla-org/flash-linear-attention` `parallel_path_attn`, saving **`10.8 MiB`** peak HBM at `T=4096`).
3. **Suite A — Track 2 (End-to-End `PaTH-FoX` Block: `ShortConv3+SiLU+L2Norm+BetaSigmoid Frontend + PaTH-FoX Attention`)**: `50.626 -> 5.074 ms` (**`9.98x`** over Eager, **`2.87x`** over Eager Frontend + Official `fla-org`, and **`2.81x`** over `torch.compile` Frontend + Official `fla-org`).
4. **Suite B (`8-GPU` NVSwitch Distributed Context Parallelism `Ring PaTH-CP = 8` at `T_global = 4096, K = 128`)**: `Gate G9` 3-step `FP32` parity `max_param_rel_l2 = 1.12e-06` (vs `0.3105` shard-local no-handoff bug); delivers **`12.95x`** over All-Gather Eager, **`3.34x`** over All-Gather Official `fla-org` (cutting peak HBM **`1.81x`**, saving **`122.0 MiB/rank`**), and **`2.48x`** over Dense-Matrix Ring Handoff.
5. **Suite C (`6.0s` `8xA100 DDP` `PaTH-FoX` `val_bpb` Autonomous RSI Loop)**: `gen2_rsi_round8_ut_duality_path_fox` runs **`2.45x`** more updates (`168 -> 411`, vs `330` for Official `fla-org`), improving `val_bpb` by **`-0.3482`** and untouched single-touch `test_bpb` by **`-0.3424`** (`State-Tracking Acc = 95.90%`).

## Matched-Step Position-Encoding & 2x Length-Extrapolation Scaling Law

| Arm | `val_bpb` | Clean `test_bpb` | `2x` Extrap `test_bpb` | Extrap `Δbpb` | Clean State Acc | `2x` Extrap State Acc |
|---|---:|---:|---:|---:|---:|---:|
| `P0_nope_causal_attention` | `2.7260` | `2.7528` | `2.9528` | `+0.2000` | `51.66%` | `42.99%` |
| `P1_rope_fixed_rotary_attention` | `2.1825` | `2.1791` | `2.7659` | `+0.5868` | `59.27%` | `49.26%` |
| `P2_alibi_static_decay_attention` | `2.2141` | `2.2069` | `2.3233` | `+0.1163` | `62.27%` | `55.44%` |
| `P3_fox_adaptive_forgetting_only` | `2.1728` | `2.1726` | `2.2789` | `+0.1063` | `62.59%` | `54.85%` |
| `P4_path_fox_rsi_householder_attention` | `2.0444` | `2.0349` | `2.1454` | `+0.1104` | `92.68%` | `90.81%` |
