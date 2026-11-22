# E132 Summary — Round-6 RSI Study: Moonshot Kimi `Attention Residuals` (`arXiv:2603.15031`)

- **Validation Gates**: **`9/9` PASSED** (`all_gates_passed = True`)
- **Repetitions**: `3` independent `8x NVIDIA A100-SXM4-80GB` (`1,410 MHz` locked) runs
- **Immutable Evaluator SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`

## Headline Hardware & Scaling Results (3-Rep Median / IQR)

1. **Suite 0 (Verification & Forensics)**: `4/4` red-team exploits rejected; `14/14` Round-6 RSI Triton kernels compiled with **`0` cubin spills** (vs `28` spills for naive 1-pass `BD=4096, nw=16`); `FP32 max_rel_l2 = 2.64e-06`, `BF16 max_rel_l2 = 0.0137`; `7/7` `FP64` `O(h^2)` Richardson checks in `[3.90, 4.10]`.
2. **Suite A — Track 1 (Batched Two-Phase `Block AttnRes` + Superfused Pre-RMSNorm)**: **`6.38x`** speedup over Moonshot `Fig. 2` Eager (`IQR = 0.005`), **`1.50x`** faster than `fullgraph=True` `torch.compile`, and **`-2542.1 MiB`** lower peak HBM than `torch.compile`.
3. **Suite A — Track 2 (Multi-Block Sequential Transformer Stack, `N_B=8` + Final Readout)**: **`5.26x`** over Eager (`1.49x` over `torch.compile`), saving **`-23748.2 MiB`** via `RSIPhase1UnbindInterBlockFn` + zero-`O1_bar` retention.
4. **Suite B (`8-GPU` NVSwitch Interleaved `1F1B` `P=8, V=4` Cross-Stage Block Caching, `Sec. 4.1 Eq. 7-8`)**: **`4.72x`** speedup & **`9.93x`** P2P traffic reduction over `Full AttnRes` PP, **`6.88x`** over Eager Naive `Block AttnRes` PP, and **`1.18x`** pure cross-stage caching gain (`2.60x` comm reduction), with exact `Gate G9` 1-GPU parity.
5. **Suite C (`15.0s` `8xA100 DDP` `val_bpb` RSI Loop + 5-Arm Deep-24-Layer Law)**: `gen2_rsi_superfused_two_phase` executes **`1.71x`** more steps, improving `val_bpb` by **`-0.6946`** and untouched single-touch `test_bpb` by **`-0.6944`**.

## Matched-Step Deep-24-Layer (`48`-Sublayer) Residual-Topology & PreNorm-Dilution Law

| Arm | `val_bpb` (med) | `test_bpb` (med) | Trigger Acc | Shallow Acc | Mid Acc | Deep Acc | Norm Growth `||h_L||/||h_1||` | Grad Norm `CV` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `R0_standard_prenorm` | `1.7315` | `1.7341` | `8.63%` | `8.25%` | `8.45%` | `9.38%` | `430.611x` | `0.495` |
| `R1_denseformer_static` | `1.7915` | `1.7933` | `11.15%` | `10.60%` | `11.47%` | `11.46%` | `7.756x` | `0.143` |
| `R2_highway_sigmoid_gate` | `2.1802` | `2.1807` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `46.997x` | `6.523` |
| `R3_block_attnres_n8` | `1.6990` | `1.6996` | `16.53%` | `16.70%` | `17.53%` | `14.97%` | `49.157x` | `0.261` |
| `R4_full_attnres_n48` | `1.7581` | `1.7590` | `7.39%` | `7.08%` | `6.98%` | `8.33%` | `18.203x` | `0.675` |
