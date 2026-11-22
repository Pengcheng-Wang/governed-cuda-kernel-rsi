# E133 (`v124`): `H-Net: Dynamic Chunking for End-to-End Hierarchical Sequence Modeling` (`arXiv:2507.07955`) Round-7 RSI Summary

- **Validation Gates**: **`9/9` PASSED** (`all_gates_passed = True`)
- **Repetitions**: `3` independent `8x NVIDIA A100-SXM4-80GB` (`1,410 MHz` locked) runs
- **Immutable Evaluator SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`

## Headline Hardware & Scaling Results (3-Rep Median / IQR)

1. **Suite 0 (Forensics & Exactness)**: `4/4` red-team exploits rejected; `17/17` Round-7 RSI Triton kernels (`33` specializations, max `32` regs) compiled with **`0` cubin spills** (vs `72` spills for naive 1-pass `[8, BD=4096], nw=16`); `FP32 max_rel_l2 = 5.90e-07`, `BF16 max_rel_l2 = 0.0043`; `7/7` `FP64` `O(h^2)` Richardson checks passed.
2. **Suite A — Track 1 (Isolated Dynamic Chunking Operators: `Router + Pack + EMA + Upsample*STE + Residual`)**: `23.654 -> 3.060 ms` (**`6.31x`** over Official `goombalab/hnet` `dc.py`, **`2.63x`** faster than `torch.compile(dynamic=True)`, saving **`1861.8 MiB`** peak HBM).
3. **Suite A — Track 2 (End-to-End 1-Stage Hierarchical `H-Net` Block with `Varlen FlashAttention-2`)**: `45.586 -> 19.276 ms` (**`2.36x`** over Official `goombalab/hnet`, **`1.23x`** over `torch.compile(dynamic=True)`, saving **`1363.9 MiB`** vs `compile`).
4. **Suite B (`8-GPU` NVSwitch Whole-Document Cost-LPT `AllToAllv` + Global Token-Weighted Ratio Loss)**: `Gate G9` 3-step `FP32` parity `max_param_rel_l2 = 2.64e-06` (vs `0.7626` local-ratio bias); reduces Main-Network straggler cost imbalance **`2.96x -> 1.01x`**, cuts redistribution payload **`3.20x`** (`384.0 -> 120.2 MiB/rank`), and delivers **`1.80x`** over Official Batched DDP (`1.84x` pure Cost-LPT NVSwitch gain).
5. **Suite C (`12.0s` `8xA100 DDP` Byte-Level `val_bpb` RSI Loop)**: `gen2_rsi_round7_superfused_dc` runs **`1.41x`** more updates (`145 -> 204`), improving `val_bpb` by **`-0.0891`** and untouched single-touch `test_bpb` by **`-0.0871`** (`KV recall = 5.60%`).

## Matched-Step Byte-Level Chunking & 1-Byte Phase-Shift / Character-Noise Robustness Law

| Arm | `val_bpb` | Clean `test_bpb` | Shifted+Noisy `test_bpb` | Noise `Δbpb` | Clean KV Acc | Noisy KV Acc | Boundary `F1` | Bytes/Chunk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `H0_isotropic_byte_swa` | `2.1448` | `2.1438` | `2.6147` | `+0.4709` | `1.37%` | `1.29%` | `22.16%` | `1.00` |
| `H1_static_stride4_hourglass` | `1.5004` | `1.4997` | `2.1253` | `+0.6289` | `4.50%` | `3.97%` | `16.63%` | `4.00` |
| `H2_spacebyte_delimiter_heuristic` | `1.3719` | `1.3711` | `2.1069` | `+0.7387` | `2.93%` | `2.95%` | `31.13%` | `4.24` |
| `H3_hnet_dc_no_ema_no_ste` | `1.7482` | `1.7480` | `2.2984` | `+0.5650` | `5.97%` | `5.49%` | `22.40%` | `3.21` |
| `H4_hnet_rsi_dynamic_chunking` | `1.3367` | `1.3355` | `2.0520` | `+0.7187` | `6.37%` | `5.86%` | `43.82%` | `3.06` |
