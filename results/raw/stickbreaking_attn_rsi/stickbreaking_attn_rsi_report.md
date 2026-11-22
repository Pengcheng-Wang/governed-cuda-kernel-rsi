# E152 (`v143`, Round-26 RSI): Stick-Breaking Attention (`arXiv:2410.17980`, ICLR 2025)

## Headline Summary (`3` Independent Full 8-GPU Repetitions on `8x NVIDIA A100-SXM4-80GB`)
- **Frozen Cryptographic Evaluator (`code/rsi_kernel_harness.py`)**: `SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e` (`3/3` adversarial mutations killed).
- **SM80 Cubin Register/Spill Audit**: All **`7/7`** Round-26 RSI Triton kernels (**`44`** specializations) compile with **`0` register spills (`max_regs = 168 <= 168`, `max_smem = 45312` B)** vs upstream `fla/ops/stickbreaking_attn`'s **`29/108` (`26.85%`)** spilling specializations (**`1294`** total spills, `max_regs = 255`, `max_spills = 154`, `max_smem = 73728` B).
- **Exact `FP64` Oracle Parity**: `FP32 max_rel_l2 = 3.16e-06` (**`466.2x`** more accurate than upstream `TF32` `rel_l2 = 1.47e-03`), `BF16 max_rel_l2 = 4.16e-03`, and resolves upstream `StickBreakingAttention.forward`'s `FP32` `TypeError` crash.
- **Training Kernel Matrix (`8` Shapes)**: Up to **`1.29x`** Forward, **`1.51x`** Backward (via Single-Pass Reverse-Stick Subtraction + Coalesced `[B*HQ, NS, T]` Snapshot Layout), and **`1.43x`** Total speedup.
- **Tensor-Core GQA-Grouped Decoding (`Track 2`)**: Up to **`32.99x`** faster than upstream scalar SIMT `stickbreaking_attn_decoding`.
- **5-Stage `StickBreakingAttentionBlockRSI` Ablation (`Track 3`)**: **`1.53x`** (`D=1024, K=64`) and **`1.41x`** (`D=2048, K=128`) end-to-end layer speedup (`A0 -> A4`).
- **8-GPU NVSwitch `StickBreaking-CP = 8` (`Track 4`)**: Up to **`7.21x`** faster than Sequence-AllGather/Ring (`7.70x` scaling vs 1 GPU).
- **`12.0s` Fixed-Wall-Clock Autoresearch (`Track 5`)**: **`1.49x`** (`D=1024`, `-0.0754 bpb`) and **`1.37x`** (`D=2048`, `-0.8320 bpb`).

## Track 1C: 8-Shape Training Kernel Latency Matrix (`B=2`, `BF16`)
| Shape | Upstream Fwd (ms) | RSI Fwd (ms) | Fwd Speedup | Upstream Bwd (ms) | RSI Bwd (ms) | Bwd Speedup | Upstream Tot (ms) | RSI Tot (ms) | Total Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B2_T1024_HQ16_H16_K64` | 0.137 | 0.106 | **1.29x** | 0.678 | 0.532 | **1.27x** | 0.815 | 0.638 | **1.28x** |
| `B2_T1024_HQ16_H4_K64` | 0.137 | 0.106 | **1.29x** | 0.787 | 0.540 | **1.46x** | 0.924 | 0.646 | **1.43x** |
| `B2_T2048_HQ16_H16_K64` | 0.383 | 0.366 | **1.05x** | 2.395 | 1.827 | **1.31x** | 2.778 | 2.193 | **1.27x** |
| `B2_T2048_HQ16_H4_K64` | 0.382 | 0.362 | **1.05x** | 2.769 | 1.837 | **1.51x** | 3.151 | 2.200 | **1.43x** |
| `B2_T4096_HQ16_H4_K64` | 1.426 | 1.371 | **1.04x** | 9.702 | 6.703 | **1.45x** | 11.128 | 8.074 | **1.38x** |
| `B2_T1024_HQ16_H4_K128` | 0.170 | 0.163 | **1.04x** | 1.023 | 0.824 | **1.24x** | 1.193 | 0.987 | **1.21x** |
| `B2_T2048_HQ16_H16_K128` | 0.593 | 0.597 | **0.99x** | 3.160 | 2.886 | **1.09x** | 3.753 | 3.483 | **1.08x** |
| `B2_T2048_HQ16_H4_K128` | 0.580 | 0.593 | **0.98x** | 3.520 | 2.904 | **1.21x** | 4.100 | 3.496 | **1.17x** |

## Track 2: Tensor-Core GQA-Grouped Decoding vs Upstream Scalar SIMT Decoding (`B=4, HQ=16, H=2`)
| `(T_q, T_kv, K)` | Upstream Decode (ms) | RSI Tensor-Core GQA Decode (ms) | Speedup | `o` `rel_l2` | `rem` `rel_l2` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `(1, 2048, 64)` | 0.136 | 0.071 | **1.91x** | `0.00e+00` | `0.00e+00` |
| `(16, 2048, 64)` | 0.494 | 0.084 | **5.90x** | `1.24e-05` | `0.00e+00` |
| `(64, 2048, 64)` | 1.591 | 0.107 | **14.80x** | `1.93e-05` | `0.00e+00` |
| `(1, 4096, 128)` | 0.422 | 0.218 | **1.94x** | `0.00e+00` | `0.00e+00` |
| `(16, 4096, 128)` | 2.766 | 0.242 | **11.45x** | `8.76e-06` | `0.00e+00` |
| `(64, 4096, 128)` | 10.204 | 0.309 | **32.99x** | `2.74e-05` | `0.00e+00` |

## Track 3: 5-Stage `StickBreakingAttentionBlockRSI` Periphery Ablation (`B=2, T=2048, HQ=16, H=4`)
| Config | Stage | Fwd (ms) | Bwd (ms) | Total (ms) | Tokens/s | Speedup vs `A0` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `D1024_K64` | `A0` | 0.781 | 3.642 | 4.424 | 925,943 | **1.00x** |
| `D1024_K64` | `A1` | 0.789 | 2.672 | 3.461 | 1,183,380 | **1.28x** |
| `D1024_K64` | `A2` | 0.745 | 2.615 | 3.359 | 1,219,255 | **1.32x** |
| `D1024_K64` | `A3` | 0.619 | 2.225 | 2.844 | 1,440,396 | **1.56x** |
| `D1024_K64` | `A4` | 0.667 | 2.226 | 2.893 | 1,415,853 | **1.53x** |
| `D2048_K128` | `A0` | 1.426 | 5.456 | 6.882 | 595,165 | **1.00x** |
| `D2048_K128` | `A1` | 1.442 | 4.844 | 6.286 | 651,651 | **1.09x** |
| `D2048_K128` | `A2` | 1.408 | 4.691 | 6.099 | 671,636 | **1.13x** |
| `D2048_K128` | `A3` | 1.131 | 3.922 | 5.054 | 810,499 | **1.36x** |
| `D2048_K128` | `A4` | 1.061 | 3.812 | 4.873 | 840,476 | **1.41x** |

## Track 4: 8-GPU NVSwitch Context Parallelism (`StickBreaking-CP = 8`, `HQ=32, H=8, K=64`)
| `T_total` (`B`) | 1-GPU RSI (ms) | CP=8 Ring (ms) | CP=8 Ulysses RSI (ms) | Ulysses vs Ring | Scaling vs 1 GPU | `o` `rel_l2` | `dq` `rel_l2` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `16384` (`B=2`) | 264.14 | 252.93 | 35.09 | **7.21x** | **7.53x** | `0.00e+00` | `0.00e+00` |
| `32768` (`B=1`) | 526.85 | 490.16 | 68.40 | **7.17x** | **7.70x** | `0.00e+00` | `0.00e+00` |

## Track 5: `12.0s` Fixed-Wall-Clock `val_bpb` Autoresearch Loop (`4` Layers, `B=2, T=2048`)
| Config | Stage | Steps (`12.0s`) | Tokens/s | Final `val_bpb` | Step Ratio (`A4/A0`) | `val_bpb` Delta (`A4 - A0`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `D1024_K64` | `A0` | 631.7 | 215,342 | 2.0809 | **1.49x** | **-0.0754** |
| `D1024_K64` | `A4` | 942.7 | 321,260 | 2.0055 | **1.49x** | **-0.0754** |
| `D2048_K128` | `A0` | 398.3 | 135,210 | 8.7232 | **1.37x** | **-0.8320** |
| `D2048_K128` | `A4` | 544.7 | 184,824 | 7.8912 | **1.37x** | **-0.8320** |
