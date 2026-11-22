# Round-28 RSI (`E154`, `v145`): Titans (`arXiv:2501.00663`, Google Research — Learning to Memorize at Test Time)

## 1. Executive Summary & Headline Metrics (3-Rep Mean ± Std on `8x NVIDIA A100-SXM4-80GB`)
- **Paper**: *Titans: Learning to Memorize at Test Time* (Ali Behrouz, Peilin Zhong, Vahab Mirrokni — Google Research, `arXiv:2501.00663`; upstream `fla/ops/titans`).
- **Frozen Cryptographic Evaluator (`code/rsi_kernel_harness.py`)**: `SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e` (`verified = True`).
- **Track 1A (`FP64` Oracle, Upstream Bug Audit, & Mutations)**: `FP32 max_rel_l2 = 7.77e-07`, `BF16 max_rel_l2 = 7.68e-03`, **`3/3` (`100%`) adversarial mutations killed** (`M1_drop_momentum_coupling_vector_f`, `M2_drop_intra_chunk_surprise_matrix_G`, `M3_drop_fused_output_ln_residual`), while fixing upstream `fla/ops/titans`'s `@pytest.mark.skipif(True, reason='FIXME')` tail-padding bug at $T = 37$ (`40/296` = `13.51%` `NaN`s in `dalpha`, `8/296` = `2.7%` `NaN`s in `deta`, and `1.22e-01` error on `final_state`) and `cal_G_log` `FP64 -> FP32` silent truncation.
- **Track 1B (SM80 Cubin Register & Spill Audit)**: Upstream `fla/ops/titans` has **`0` Triton kernels** (`288` sequential Python loops per 16-token chunk in `log_impl.py`) -> Round-28 RSI (`12/12` Triton kernels, `137` specializations): **`0` register spills across `100%` of specializations (`max_regs = 128 <= 168`, `max_smem = 18,432 B`)**.
- **Track 1C (8-Shape Training Kernel Matrix)**: Wins `8/8` shapes in Fwd, Bwd, and Total, reaching up to **`121.63x` Fwd, `216.78x` Bwd, and `181.98x` Total speedup** vs `vectorized_pytorch_chunk_titans` (and up to **`3,080.7x` faster** than upstream `chunk_titans_linear_ref`).
- **Track 2 (Single-Pass Online Test-Time Memorization Decoding)**: **`37.66x–43.58x` faster** than `vectorized_pytorch_chunk_titans` (`5.17x–5.43x` vs `chunk_titans_linear_rsi`).
- **Track 3 (`TitansBlockRSI` `A0 -> A4`)**: **`129.30x`** (`D=1024, K=64`) and **`53.97x`** (`D=2048, K=128`) end-to-end training block speedup.
- **Track 4 (`Titans-CP = 8` on 8x A100 NVSwitch)**: Up to **`1.60x`** faster than 8-GPU `Ring/AllGather` baseline with **`16.4%` 8-GPU parallel efficiency** and `0.00e+00` `rel_l2` vs single-GPU execution.
- **Track 5 (`12.0s` Fixed-Wall-Clock Autoresearch)**: **`35.42x`** more steps (`-3.2291 bpb` at `D=1024`) and **`25.33x`** more steps (`-0.7225 bpb` at `D=2048`).

## 2. Track 1C: 8-Shape Training Kernel Matrix (`BF16`, 3-Rep Mean ± Std)
| Shape | Vec PyTorch Fwd (ms) | Vec PyTorch Bwd (ms) | Vec PyTorch Tot (ms) | RSI Fwd (ms) | RSI Bwd (ms) | RSI Tot (ms) | Fwd Speedup | Bwd Speedup | Tot vs Vec | Tot vs Upstream `ref` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B=2,T=512,H=16,K=64,V=64` | 39.423±0.191 | 120.219±2.265 | 159.642±2.242 | 0.338±0.000 | 0.734±0.066 | 1.073±0.066 | **116.51x** | **163.74x** | **148.84x** | **3,080.7x** (3,304.3 ms) |
| `B=2,T=1024,H=16,K=64,V=64` | 78.691±0.841 | 242.113±2.412 | 320.804±3.251 | 0.653±0.001 | 1.135±0.001 | 1.788±0.001 | **120.56x** | **213.23x** | **179.40x** | — |
| `B=2,T=2048,H=16,K=64,V=64` | 155.870±0.350 | 481.876±3.693 | 637.746±4.009 | 1.282±0.001 | 2.223±0.001 | 3.504±0.001 | **121.63x** | **216.78x** | **181.98x** | — |
| `B=4,T=2048,H=16,K=64,V=64` | 155.499±0.406 | 482.903±5.894 | 638.401±6.242 | 1.533±0.001 | 2.802±0.002 | 4.335±0.001 | **101.44x** | **172.35x** | **147.28x** | — |
| `B=2,T=512,H=16,K=128,V=128` | 39.559±0.131 | 122.913±1.755 | 162.472±1.866 | 0.767±0.001 | 1.504±0.007 | 2.272±0.006 | **51.56x** | **81.70x** | **71.52x** | **1,473.2x** (3,346.4 ms) |
| `B=2,T=1024,H=16,K=128,V=128` | 78.453±0.240 | 243.530±2.033 | 321.983±2.265 | 1.513±0.018 | 2.857±0.002 | 4.371±0.018 | **51.84x** | **85.23x** | **73.67x** | — |
| `B=2,T=2048,H=16,K=128,V=128` | 155.882±0.385 | 480.616±5.392 | 636.497±5.692 | 2.995±0.023 | 5.663±0.025 | 8.659±0.040 | **52.04x** | **84.86x** | **73.51x** | — |
| `B=4,T=1024,H=16,K=128,V=128` | 79.312±0.161 | 245.426±1.366 | 324.738±1.443 | 1.845±0.011 | 3.774±0.006 | 5.619±0.015 | **42.98x** | **65.04x** | **57.80x** | — |

## 3. Track 2: Single-Pass Online Test-Time Memorization Decoding (`_titans_decode_memorize_rsi_kernel`)
| Shape | Vec PyTorch (ms) | `chunk_rsi` (ms) | RSI Decode (ms) | Speedup vs Vec | Speedup vs `chunk_rsi` | RSI `mt` `rel_l2` | RSI `st` `rel_l2` | Up `mt` `rel_l2` (Tail Bug) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B=4,Tq=1,H=16,K=64` | 2.165±0.019 | 0.265±0.001 | 0.051±0.001 | **42.20x** | **5.17x** | `1.99e-04` | `3.97e-04` | `5.60e-02` |
| `B=4,Tq=4,H=16,K=64` | 2.169±0.011 | 0.266±0.002 | 0.050±0.001 | **43.58x** | **5.35x** | `5.63e-04` | `2.10e-03` | `5.34e-02` |
| `B=4,Tq=16,H=16,K=64` | 1.978±0.008 | 0.272±0.001 | 0.050±0.000 | **39.33x** | **5.41x** | `1.59e-03` | `2.56e-03` | `1.51e-02` |
| `B=8,Tq=16,H=16,K=64` | 1.950±0.007 | 0.269±0.002 | 0.050±0.000 | **39.42x** | **5.43x** | `1.64e-03` | `2.59e-03` | `1.54e-02` |
| `B=4,Tq=1,H=16,K=128` | 2.194±0.008 | 0.272±0.002 | 0.051±0.001 | **43.06x** | **5.33x** | `1.71e-04` | `3.75e-04` | `5.49e-02` |
| `B=4,Tq=4,H=16,K=128` | 2.167±0.012 | 0.268±0.003 | 0.051±0.001 | **42.53x** | **5.25x** | `3.87e-04` | `1.78e-03` | `5.44e-02` |
| `B=4,Tq=16,H=16,K=128` | 1.961±0.020 | 0.266±0.006 | 0.050±0.002 | **39.58x** | **5.36x** | `1.34e-03` | `2.57e-03` | `1.80e-02` |
| `B=8,Tq=16,H=16,K=128` | 1.954±0.004 | 0.270±0.004 | 0.052±0.001 | **37.66x** | **5.20x** | `1.32e-03` | `2.60e-03` | `1.80e-02` |

## 4. Track 3: 5-Stage Progressive Layer Periphery Ablation (`TitansBlockRSI` `A0 -> A4`)
| Config | Stage | Fwd (ms) | Bwd (ms) | Total (ms) | Speedup vs `A0` | Tokens/s |
|---|---|---:|---:|---:|---:|---:|
| `D=1024,H=16,K=64` | `A0` (A0_vectorized_pytorch_titans_block_baseline) | 156.069±0.952 | 507.498±16.206 | 663.567±17.134 | **1.00x** | 6,175 |
| `D=1024,H=16,K=64` | `A1` (A1_zero_spill_titans_kernel_bi_triangular_decay) | 2.098±0.000 | 4.202±0.001 | 6.300±0.002 | **105.33x** | 650,200 |
| `D=1024,H=16,K=64` | `A2` (A2_plus_fused_in_kernel_output_ln_residual) | 1.907±0.001 | 3.804±0.004 | 5.711±0.003 | **116.19x** | 717,238 |
| `D=1024,H=16,K=64` | `A3` (A3_plus_fused_qkv_gates_single_gemm) | 1.851±0.003 | 3.562±0.003 | 5.413±0.000 | **122.59x** | 756,695 |
| `D=1024,H=16,K=64` | `A4` (A4_plus_persistent_2d_triton_rmsnorm) | 1.759±0.001 | 3.373±0.000 | 5.132±0.001 | **129.30x** | 798,124 |
| `D=2048,H=16,K=128` | `A0` (A0_vectorized_pytorch_titans_block_baseline) | 155.396±0.877 | 511.190±3.990 | 666.586±3.678 | **1.00x** | 6,145 |
| `D=2048,H=16,K=128` | `A1` (A1_zero_spill_titans_kernel_bi_triangular_decay) | 4.679±0.002 | 10.050±0.003 | 14.729±0.003 | **45.26x** | 278,088 |
| `D=2048,H=16,K=128` | `A2` (A2_plus_fused_in_kernel_output_ln_residual) | 4.283±0.001 | 9.319±0.002 | 13.602±0.003 | **49.01x** | 301,132 |
| `D=2048,H=16,K=128` | `A3` (A3_plus_fused_qkv_gates_single_gemm) | 4.245±0.001 | 8.784±0.025 | 13.030±0.026 | **51.16x** | 314,362 |
| `D=2048,H=16,K=128` | `A4` (A4_plus_persistent_2d_triton_rmsnorm) | 4.040±0.002 | 8.310±0.003 | 12.350±0.001 | **53.97x** | 331,655 |

## 5. Track 4: 8-GPU NVSwitch Context Parallelism (`Titans-CP = 8`)
| Shape | 1-GPU Vec (ms) | 1-GPU RSI (ms) | 8-GPU Ring (ms) | 8-GPU Ulysses RSI (ms) | vs Ring | vs 1-GPU RSI | Parallel Eff. | Max `rel_l2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `B=2,T=16384,H=16,K=64` | 6220.30 | 27.60 | 35.70±0.03 | 22.38±0.08 | **1.59x** | **1.23x** | **15.4%** | `0.00e+00` |
| `B=2,T=32768,H=16,K=64` | 11516.08 | 55.27 | 70.38±0.03 | 44.25±0.03 | **1.59x** | **1.25x** | **15.6%** | `0.00e+00` |
| `B=2,T=16384,H=16,K=128` | 5450.10 | 70.40 | 85.60±0.23 | 53.48±0.10 | **1.60x** | **1.32x** | **16.4%** | `0.00e+00` |

## 6. Track 5: Fixed-Wall-Clock (`12.0s`) `val_bpb` Autoresearch Loop
| Config | Base Steps | RSI Steps | Step Ratio | Base `val_bpb` | RSI `val_bpb` | `val_bpb` Gain |
|---|---:|---:|---:|---:|---:|---:|
| `D=1024,H=16,K=64` | 40.0±0.0 | 1416.7±18.9 | **35.42x** | 8.4814±0.0000 | 5.2523±0.1849 | **-3.2291 bpb** |
| `D=2048,H=16,K=128` | 40.0±0.0 | 1013.3±2.9 | **25.33x** | 9.4567±0.0000 | 8.7342±0.6131 | **-0.7225 bpb** |
