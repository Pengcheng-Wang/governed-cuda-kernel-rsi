# E148 (`v139`, Round-22 RSI): Cyclic Flow Attention (`CyFA`) Zero-Spill SM80 Kernel & Layer Suite

## 1. Executive Summary
- **Architecture**: `Cyclic Flow Attention (CyFA): Linear Sequence Modeling with Relative-Time-Partitioned Memory` (`arXiv:2609.36259`, Chen et al.; upstream `fla.ops.cyfa` & `fla.layers.cyfa`).
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch), `3` independent repetitions.
- **Frozen Cryptographic Evaluator**: `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
- **SM80 Cubin Register Spill Elimination**: Upstream `fla.ops.cyfa` + `fla.ops.common` compiles `19/91` spilling specializations (`max_regs = 255`, `max_spills = 306`, `total_spills = 874`, `max_smem = 147456` B) vs **Round-22 RSI `0/132` spilling specializations across all `19/19` kernels (`max_regs = 168 <= 168`, `total_spills = 0`, `max_smem = 40960` B)**.
- **Exact `FP32` / `BF16` Recurrent Oracle Parity**: `max_fp32_rel_l2 = 7.78e-06` (`869.9x` more accurate than upstream `chunk_cyfa` `6.77e-03` `TF32` truncation drift), `max_bf16_rel_l2 = 8.05e-03`, `3/3` adversarial mutations killed.
- **Track 3A Kernel Speedup (12 Shapes)**: Peak **`1.926x` Fwd+Bwd** (`1.958x` Bwd, `1.874x` Fwd; mean `1.79x` Fwd+Bwd).
- **Track 3B End-to-End Block Ablation (`A0 -> A4`)**: Peak **`1.597x` Fwd+Bwd** (`mean = 1.587x`).
- **Track 4 Fixed-Wall-Clock `val_bpb` Autoresearch (`30.0s`)**: **`1.441x` step rate** (`1289.0` vs `895.0` steps), driving `val_bpb` from `8.2405` to **`1.3391`** vs `1.516` (`-0.1769` extra `val_bpb` gain).
- **Track 5 8-GPU NVSwitch `CyFA-CP = 8` (`Ulysses` vs `Ring`)**: Peak **`3.31x` Fwd+Bwd** (`mean = 2.739x`).

## 2. Track 3A: 12-Shape Kernel Matrix (`chunk_cyfa` vs `chunk_cyfa_rsi`, 3-Rep Median)
| B | T | H | K | V | M | Upstream Fwd (ms) | RSI Fwd (ms) | Fwd Speedup | Upstream Bwd (ms) | RSI Bwd (ms) | Bwd Speedup | Upstream Fwd+Bwd (ms) | RSI Fwd+Bwd (ms) | Fwd+Bwd Speedup |
|---|---|---|---|---|---|-------------------|--------------|-------------|-------------------|--------------|-------------|-----------------------|------------------|-----------------|
| 4 | 1024 | 4 | 128 | 128 | 64 | 0.836 | 0.446 | **1.87x** | 2.019 | 1.037 | **1.95x** | 2.855 | 1.482 | **1.93x** |
| 2 | 2048 | 4 | 128 | 128 | 64 | 0.817 | 0.446 | **1.83x** | 2.012 | 1.042 | **1.93x** | 2.829 | 1.488 | **1.90x** |
| 2 | 4096 | 4 | 128 | 128 | 64 | 0.832 | 0.461 | **1.80x** | 2.024 | 1.046 | **1.94x** | 2.857 | 1.507 | **1.90x** |
| 1 | 8192 | 4 | 128 | 128 | 64 | 0.845 | 0.526 | **1.60x** | 2.027 | 1.049 | **1.93x** | 2.872 | 1.575 | **1.82x** |
| 4 | 1024 | 4 | 128 | 128 | 128 | 0.833 | 0.453 | **1.84x** | 2.028 | 1.036 | **1.96x** | 2.861 | 1.487 | **1.92x** |
| 2 | 2048 | 4 | 128 | 128 | 128 | 0.832 | 0.449 | **1.85x** | 2.030 | 1.052 | **1.93x** | 2.862 | 1.501 | **1.91x** |
| 2 | 4096 | 4 | 128 | 128 | 128 | 0.840 | 0.476 | **1.77x** | 2.035 | 1.051 | **1.94x** | 2.875 | 1.527 | **1.88x** |
| 1 | 8192 | 4 | 128 | 128 | 128 | 0.840 | 0.638 | **1.32x** | 2.064 | 1.079 | **1.91x** | 2.882 | 1.717 | **1.68x** |
| 2 | 2048 | 4 | 128 | 256 | 64 | 0.829 | 0.455 | **1.82x** | 2.017 | 1.049 | **1.92x** | 2.846 | 1.505 | **1.89x** |
| 2 | 2048 | 4 | 256 | 256 | 128 | 0.828 | 0.459 | **1.81x** | 2.015 | 1.044 | **1.93x** | 2.844 | 1.502 | **1.89x** |
| 2 | 4096 | 4 | 256 | 256 | 128 | 0.835 | 0.544 | **1.53x** | 2.027 | 1.413 | **1.44x** | 2.859 | 1.957 | **1.46x** |
| 1 | 8192 | 4 | 256 | 256 | 128 | 0.838 | 0.746 | **1.12x** | 2.038 | 1.474 | **1.38x** | 2.876 | 2.220 | **1.29x** |

## 3. Track 3B: 5-Stage `CyclicFlowAttentionBlockRSI` Progressive Ablation (`A0 -> A4`)
| Config | B | T | D | K | V | M | A0 (ms) | A1 (ms) | A2 (ms) | A3 (ms) | A4 (ms) | A4 vs A0 | A4 vs A1 |
|--------|---|---|---|---|---|---|---------|---------|---------|---------|---------|----------|----------|
| `CyFA-512-M64-K128-V128` | 2 | 2048 | 512 | 128 | 128 | 64 | 7.158 | 5.736 | 4.786 | 4.551 | 4.503 | **1.59x** | **1.27x** |
| `CyFA-512-M128-K128-V128` | 2 | 2048 | 512 | 128 | 128 | 128 | 7.140 | 5.735 | 4.790 | 4.579 | 4.505 | **1.58x** | **1.27x** |
| `CyFA-1K-M64-K128-V256` | 2 | 2048 | 1024 | 128 | 256 | 64 | 7.268 | 5.744 | 4.823 | 4.632 | 4.551 | **1.60x** | **1.26x** |
| `CyFA-1K-M128-K256-V256` | 2 | 2048 | 1024 | 256 | 256 | 128 | 7.173 | 5.715 | 4.832 | 4.627 | 4.548 | **1.58x** | **1.26x** |

## 4. Track 5: 8-GPU NVSwitch `CyFA-CP = 8` Scaling (`Ulysses` vs `Ring`)
| T_total | T_local | H | K | V | M | Ring CP=8 (ms) | Ulysses CP=8 (ms) | Ulysses vs Ring |
|---------|---------|---|---|---|---|----------------|-------------------|-----------------|
| 16384 | 2048 | 16 | 128 | 128 | 64 | 8.292 | 4.063 | **2.04x** |
| 16384 | 2048 | 16 | 128 | 128 | 128 | 12.443 | 4.096 | **3.04x** |
| 32768 | 4096 | 16 | 128 | 128 | 64 | 13.509 | 5.609 | **2.41x** |
| 32768 | 4096 | 16 | 128 | 128 | 128 | 21.088 | 6.371 | **3.31x** |
| 65536 | 8192 | 16 | 128 | 128 | 64 | 24.128 | 10.224 | **2.36x** |
| 65536 | 8192 | 16 | 128 | 128 | 128 | 38.472 | 11.739 | **3.28x** |
