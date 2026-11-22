# Round-27 RSI (`E153`, `v144`): Test-Time Training (`TTT-Linear`, `arXiv:2407.04620`, ICML 2024 / JMLR 2025)

## 1. Executive Summary & Headline Metrics (3-Rep Mean ± Std on `8x NVIDIA A100-SXM4-80GB`)
- **Paper**: *Learning to (Learn at Test Time): RNNs with Expressive Hidden States* (Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin; ICML 2024 / JMLR 2025, `arXiv:2407.04620`).
- **Frozen Cryptographic Evaluator (`code/rsi_kernel_harness.py`)**: `SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e` (`verified = True`).
- **Track 1A (`FP64` Oracle & Mutations)**: `FP32 max_rel_l2 = 4.24e-07` (**`3153x` more accurate** than upstream backward `TF32` truncation `1.34e-03` while fixing upstream's `T % 16 != 0` tail-padding bug `1.08e+00` on `final_state_bias`), `BF16 max_rel_l2 = 5.39e-03`, **`3/3` (`100%`) adversarial mutations killed**.
- **Track 1B (SM80 Cubin Register & Spill Audit)**: Upstream `fla.ops.ttt` (`chunk` + `fused_chunk`) spills across **`78/119` (`65.55%`) specializations (`85,682` total spills, `max_regs = 255`, `max_spills = 4614`, `max_smem = 151,808 B`, plus `3` `OutOfResources` crashes)** -> Round-27 RSI (`9/9` kernels): **`0` register spills across `100%` of specializations (`max_regs <= 168`)**.
- **Track 1C (8-Shape Training Kernel Matrix)**: Wins `8/8` shapes in Fwd, Bwd, and Total, reaching up to **`4.14x` Fwd, `2.16x` Bwd, and `2.47x` Total speedup** vs `chunk_ttt_linear` (and up to **`6.58x`** vs `fused_chunk_ttt_linear`).
- **Track 2 (Single-Pass Online Test-Time Adaptation Decoding)**: **`11.00x–11.43x` faster** than `chunk_ttt_linear` (`9.05x–9.33x` vs `fused_chunk_ttt_linear`).
- **Track 3 (`TTTLinearBlockRSI` `A0 -> A4`)**: **`1.51x`** (`D=1024, K=64`) and **`1.73x`** (`D=2048, K=128`) end-to-end training block speedup.
- **Track 4 (`TTT-CP = 8` on 8x A100 NVSwitch)**: Up to **`2.70x`** faster than 8-GPU `Ring/AllGather` baseline with **`15.9%` 8-GPU parallel efficiency** and `0.00e+00` `rel_l2` vs single-GPU execution.
- **Track 5 (`12.0s` Fixed-Wall-Clock Autoresearch)**: **`1.56x`** more steps (`-0.2465 bpb` at `D=1024`) and **`1.53x`** more steps (`-0.1855 bpb` at `D=2048`).

## 2. Track 1C: 8-Shape Training Kernel Matrix (`BF16`, 3-Rep Mean ± Std)
| Shape | Up `chunk` Fwd (ms) | Up `chunk` Bwd (ms) | Up `chunk` Tot (ms) | Up `fused` Tot (ms) | RSI Fwd (ms) | RSI Bwd (ms) | RSI Tot (ms) | Fwd Speedup | Bwd Speedup | Tot vs `chunk` | Tot vs `fused` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B=2,T=512,H=16,K=64,V=64` | 0.528±0.005 | 1.322±0.063 | 1.851±0.059 | 1.576 | 0.128±0.000 | 0.621±0.049 | 0.749±0.049 | **4.14x** | **2.13x** | **2.47x** | **2.11x** |
| `B=2,T=1024,H=16,K=64,V=64` | 0.530±0.010 | 1.334±0.073 | 1.865±0.070 | 1.985 | 0.209±0.000 | 0.660±0.004 | 0.869±0.004 | **2.54x** | **2.02x** | **2.15x** | **2.29x** |
| `B=2,T=2048,H=16,K=64,V=64` | 0.521±0.005 | 2.081±0.010 | 2.602±0.007 | 3.835 | 0.421±0.001 | 1.283±0.002 | 1.704±0.002 | **1.24x** | **1.62x** | **1.53x** | **2.25x** |
| `B=4,T=2048,H=16,K=64,V=64` | 0.533±0.004 | 2.332±0.012 | 2.865±0.009 | 3.955 | 0.474±0.000 | 1.588±0.002 | 2.062±0.002 | **1.12x** | **1.47x** | **1.39x** | **1.92x** |
| `B=2,T=512,H=16,K=128,V=128` | 0.526±0.004 | 1.734±0.005 | 2.261±0.004 | 7.567 | 0.294±0.000 | 0.901±0.002 | 1.195±0.002 | **1.79x** | **1.93x** | **1.89x** | **6.33x** |
| `B=2,T=1024,H=16,K=128,V=128` | 0.818±0.000 | 3.555±0.011 | 4.373±0.011 | 15.063 | 0.576±0.001 | 1.749±0.001 | 2.325±0.002 | **1.42x** | **2.03x** | **1.88x** | **6.48x** |
| `B=2,T=2048,H=16,K=128,V=128` | 1.616±0.002 | 7.021±0.002 | 8.637±0.002 | 30.091 | 1.137±0.000 | 3.439±0.002 | 4.575±0.002 | **1.42x** | **2.04x** | **1.89x** | **6.58x** |
| `B=4,T=1024,H=16,K=128,V=128` | 0.909±0.001 | 4.611±0.003 | 5.520±0.003 | 18.035 | 0.662±0.000 | 2.139±0.002 | 2.801±0.002 | **1.37x** | **2.16x** | **1.97x** | **6.44x** |

## 3. Track 2: Single-Pass Online Test-Time Adaptation Decoding (`_ttt_decode_adapt_rsi_kernel`)
| Shape | Up `chunk` (ms) | Up `fused` (ms) | RSI Decode (ms) | Speedup vs `chunk` | Speedup vs `fused` | RSI `hbt` `rel_l2` | Up `hbt` `rel_l2` (Tail Bug) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `B=4,Tq=1,H=16,K=64` | 0.530±0.003 | 0.434±0.005 | 0.047±0.001 | **11.23x** | **9.19x** | `1.91e-04` | `1.10e-01` |
| `B=4,Tq=4,H=16,K=64` | 0.528±0.007 | 0.433±0.001 | 0.047±0.001 | **11.18x** | **9.16x** | `2.90e-04` | `6.22e-02` |
| `B=4,Tq=16,H=16,K=64` | 0.529±0.002 | 0.431±0.002 | 0.046±0.002 | **11.43x** | **9.33x** | `3.37e-04` | `5.36e-03` |
| `B=8,Tq=16,H=16,K=64` | 0.527±0.002 | 0.434±0.001 | 0.048±0.001 | **11.00x** | **9.05x** | `3.28e-04` | `5.58e-03` |
| `B=4,Tq=1,H=16,K=128` | 0.527±0.002 | 0.434±0.002 | 0.048±0.002 | **11.08x** | **9.14x** | `1.03e-04` | `1.32e-01` |
| `B=4,Tq=4,H=16,K=128` | 0.529±0.001 | 0.436±0.001 | 0.048±0.002 | **11.04x** | **9.10x** | `1.54e-04` | `9.22e-02` |
| `B=4,Tq=16,H=16,K=128` | 0.531±0.002 | 0.435±0.002 | 0.048±0.001 | **11.15x** | **9.13x** | `2.14e-04` | `5.16e-03` |
| `B=8,Tq=16,H=16,K=128` | 0.530±0.001 | 0.438±0.003 | 0.048±0.002 | **11.18x** | **9.23x** | `2.09e-04` | `5.06e-03` |

## 4. Track 3: 5-Stage Progressive Layer Periphery Ablation (`TTTLinearBlockRSI` `A0 -> A4`)
| Config | Stage | Fwd (ms) | Bwd (ms) | Total (ms) | Speedup vs `A0` | Tokens/s |
|---|---|---:|---:|---:|---:|---:|
| `D=1024,H=16,K=64` | `A0` (A0_upstream_ttt_block_baseline) | 1.133±0.010 | 2.674±0.008 | 3.808±0.015 | **1.00x** | 1,075,718 |
| `D=1024,H=16,K=64` | `A1` (A1_zero_spill_ttt_kernel_no_bwd_replay) | 1.085±0.003 | 2.491±0.007 | 3.576±0.010 | **1.06x** | 1,145,412 |
| `D=1024,H=16,K=64` | `A2` (A2_plus_fused_in_kernel_output_ln_residual) | 0.841±0.001 | 2.076±0.007 | 2.917±0.008 | **1.30x** | 1,404,033 |
| `D=1024,H=16,K=64` | `A3` (A3_plus_fused_qkve_single_gemm) | 0.841±0.002 | 1.960±0.003 | 2.801±0.005 | **1.36x** | 1,462,212 |
| `D=1024,H=16,K=64` | `A4` (A4_plus_persistent_2d_triton_rmsnorm) | 0.747±0.002 | 1.770±0.002 | 2.518±0.004 | **1.51x** | 1,627,016 |
| `D=2048,H=16,K=128` | `A0` (A0_upstream_ttt_block_baseline) | 2.562±0.004 | 9.232±0.006 | 11.793±0.007 | **1.00x** | 347,309 |
| `D=2048,H=16,K=128` | `A1` (A1_zero_spill_ttt_kernel_no_bwd_replay) | 2.550±0.004 | 6.338±0.006 | 8.888±0.009 | **1.33x** | 460,824 |
| `D=2048,H=16,K=128` | `A2` (A2_plus_fused_in_kernel_output_ln_residual) | 2.125±0.002 | 5.615±0.010 | 7.740±0.012 | **1.52x** | 529,229 |
| `D=2048,H=16,K=128` | `A3` (A3_plus_fused_qkve_single_gemm) | 2.171±0.002 | 5.287±0.001 | 7.458±0.003 | **1.58x** | 549,215 |
| `D=2048,H=16,K=128` | `A4` (A4_plus_persistent_2d_triton_rmsnorm) | 1.966±0.001 | 4.848±0.003 | 6.814±0.002 | **1.73x** | 601,118 |

## 5. Track 4: 8-GPU NVSwitch Context Parallelism (`TTT-CP = 8`)
| Shape | 1-GPU Up (ms) | 1-GPU RSI (ms) | 8-GPU Ring (ms) | 8-GPU Ulysses RSI (ms) | vs Ring | vs 1-GPU RSI | Parallel Eff. | Max `rel_l2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B=2,T=16384,H=16,K=64` | 20.29 | 13.42 | 23.87±0.01 | 10.96±0.03 | **2.18x** | **1.22x** | **15.3%** | `0.00e+00` |
| `B=2,T=32768,H=16,K=64` | 40.57 | 26.91 | 47.36±0.04 | 21.63±0.01 | **2.19x** | **1.25x** | **15.6%** | `0.00e+00` |
| `B=2,T=16384,H=16,K=128` | 71.94 | 37.10 | 78.76±0.72 | 29.20±0.04 | **2.70x** | **1.27x** | **15.9%** | `0.00e+00` |

## 6. Track 5: Fixed-Wall-Clock (`12.0s`) `val_bpb` Autoresearch Loop
| Config | Base Steps | RSI Steps | Step Ratio | Base `val_bpb` | RSI `val_bpb` | `val_bpb` Gain |
|---|---:|---:|---:|---:|---:|---:|
| `D=1024,H=16,K=64` | 1373.3±7.6 | 2141.7±46.5 | **1.56x** | 3.4737±0.0533 | 3.2272±0.1846 | **-0.2465 bpb** |
| `D=2048,H=16,K=128` | 1056.7±2.9 | 1620.0±5.0 | **1.53x** | 3.7686±0.0031 | 3.5831±0.0902 | **-0.1855 bpb** |
