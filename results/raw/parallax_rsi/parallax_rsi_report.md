# E149 (`v140`, Round-23 RSI): Parallax Parameterized Local Linear Attention Zero-Spill SM80 Suite

## 1. Executive Summary
- **Architecture**: `Parallax: Parameterized Local Linear Attention` (`arXiv:2605.29157`, Yifei Zuo et al.; upstream `fla.ops.parallax` & `fla.layers.parallax`).
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch), `3` independent repetitions.
- **Frozen Cryptographic Evaluator**: `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
- **SM80 Cubin Register Spill Elimination**: Upstream `fla.ops.parallax` compiles `22/42` spilling specializations (`34/42` with `regs > 168`, `max_regs = 255`, `max_spills = 26`, `total_spills = 190`, `max_smem = 91392` B) vs **Round-23 RSI `0/115` spilling specializations across all `10/10` kernels (`max_regs = 168 <= 168`, `total_spills = 0`, `max_smem = 49152` B)**.
- **Exact `FP32` / `BF16` `naive_parallax` Oracle Parity**: `max_fp32_rel_l2 = 8.46e-06` (whereas upstream `parallel_parallax` crashes with `TypeError: parallel_parallax requires bf16 or fp16 inputs, got q.dtype=torch.float32` on `FP32`), `max_bf16_rel_l2 = 3.08e-03`, `3/3` adversarial mutations killed.
- **Track 3A Kernel Speedup (12 Shapes)**: Peak **`2.222x`** (`1.544x` Train Fwd+Bwd, `1.665x` Bwd, `2.222x` Decode/Fwd; mean `1.389x`).
- **Track 3B End-to-End Block Ablation (`A0 -> A4`)**: Peak **`2.207x` Fwd+Bwd** (`mean = 1.512x`).
- **Track 4 Fixed-Wall-Clock `val_bpb` Autoresearch (`30.0s`)**: **`2.166x` step rate** (`3629.0` vs `1676.0` steps), driving `val_bpb` from `8.2497` to **`0.9557`** vs `0.985` (`-0.0293` extra `val_bpb` gain).
- **Track 5 8-GPU NVSwitch `Parallax-CP = 8` (`Ulysses` vs `5-Tuple Ring`)**: Peak **`2.596x`** (`mean = 2.266x`, `0.0` max abs diff vs single-GPU).

## 2. Track 3A: 12-Shape Kernel Matrix (`parallel_parallax` / `parallax_decode` vs Round-23 RSI, 3-Rep Median)
| Kind | B | T | Sq | HQ | H | D | W | Upstream Fwd (ms) | RSI Fwd (ms) | Fwd Speedup | Upstream Bwd (ms) | RSI Bwd (ms) | Bwd Speedup | Upstream Total (ms) | RSI Total (ms) | Total Speedup |
|------|---|---|----|----|---|---|---|-------------------|--------------|-------------|-------------------|--------------|-------------|---------------------|----------------|---------------|
| `train_fwdbwd` | 2 | 2048 | 2048 | 16 | 4 | 64 | 512 | 0.254 | 0.223 | **1.14x** | 0.829 | 0.512 | **1.62x** | 1.082 | 0.735 | **1.47x** |
| `train_fwdbwd` | 2 | 2048 | 2048 | 16 | 4 | 64 | -1 | 0.467 | 0.459 | **1.02x** | 1.651 | 1.022 | **1.62x** | 2.118 | 1.481 | **1.43x** |
| `train_fwdbwd` | 2 | 2048 | 2048 | 16 | 16 | 64 | 512 | 0.256 | 0.208 | **1.24x** | 0.877 | 0.527 | **1.67x** | 1.133 | 0.734 | **1.54x** |
| `train_fwdbwd` | 2 | 4096 | 4096 | 16 | 4 | 64 | 512 | 0.511 | 0.459 | **1.11x** | 1.652 | 1.022 | **1.62x** | 2.162 | 1.481 | **1.46x** |
| `train_fwdbwd` | 2 | 2048 | 2048 | 16 | 4 | 128 | 512 | 0.487 | 0.352 | **1.38x** | 1.084 | 0.932 | **1.16x** | 1.571 | 1.284 | **1.22x** |
| `train_fwdbwd` | 2 | 2048 | 2048 | 16 | 4 | 128 | -1 | 0.954 | 0.724 | **1.32x** | 1.987 | 1.855 | **1.07x** | 2.941 | 2.577 | **1.14x** |
| `train_fwdbwd` | 2 | 2048 | 2048 | 16 | 16 | 128 | 512 | 0.501 | 0.353 | **1.42x** | 1.107 | 0.953 | **1.16x** | 1.608 | 1.305 | **1.23x** |
| `train_fwdbwd` | 2 | 4096 | 4096 | 16 | 4 | 128 | 512 | 0.990 | 0.719 | **1.38x** | 2.173 | 1.895 | **1.15x** | 3.163 | 2.613 | **1.21x** |
| `train_fwdbwd` | 2 | 4096 | 4096 | 16 | 4 | 128 | -1 | 3.425 | 2.639 | **1.30x** | 6.827 | 6.576 | **1.04x** | 10.252 | 9.214 | **1.11x** |
| `train_fwdbwd` | 1 | 8192 | 8192 | 16 | 4 | 128 | 1024 | 1.774 | 1.302 | **1.36x** | 3.637 | 3.353 | **1.08x** | 5.412 | 4.656 | **1.16x** |
| `chunk_decode` | 4 | 8192 | 256 | 16 | 4 | 128 | -1 | 2.029 | 1.398 | **1.45x** | 0.000 | 0.000 | **1.45x** | 2.029 | 1.398 | **1.45x** |
| `one_step_decode` | 16 | 4096 | 1 | 16 | 4 | 128 | -1 | 0.354 | 0.159 | **2.22x** | 0.000 | 0.000 | **2.22x** | 0.354 | 0.159 | **2.22x** |

## 3. Track 3B: 5-Stage `ParallaxBlockRSI` Progressive Ablation (`A0 -> A4`)
| Config | B | T | Hidden | HQ | H | D | W | A0 (ms) | A1 (ms) | A2 (ms) | A3 (ms) | A4 (ms) | A4 vs A0 | A4 vs A1 |
|--------|---|---|--------|----|---|---|---|---------|---------|---------|---------|---------|----------|----------|
| `Parallax-1K-HQ16-H4-D64-W512` | 2 | 2048 | 1024 | 16 | 4 | 64 | 512 | 3.939 | 3.744 | 2.919 | 2.775 | 1.785 | **2.21x** | **2.10x** |
| `Parallax-1K-HQ16-H4-D64-Full` | 2 | 2048 | 1024 | 16 | 4 | 64 | -1 | 3.960 | 3.732 | 2.895 | 2.834 | 2.509 | **1.58x** | **1.49x** |
| `Parallax-2K-HQ16-H4-D128-W512` | 2 | 2048 | 2048 | 16 | 4 | 128 | 512 | 4.335 | 4.089 | 3.625 | 3.727 | 3.650 | **1.19x** | **1.12x** |
| `Parallax-2K-HQ16-H4-D128-Full` | 2 | 2048 | 2048 | 16 | 4 | 128 | -1 | 5.295 | 4.981 | 4.909 | 4.973 | 4.930 | **1.07x** | **1.01x** |

## 4. Track 5: 8-GPU NVSwitch `Parallax-CP = 8` Scaling (`Ulysses` vs `5-Tuple Ring`)
| T_global | T_local | HQ | H | D | W | 5-Tuple Ring CP=8 (ms) | Ulysses CP=8 (ms) | Ulysses vs Ring |
|----------|---------|----|---|---|---|------------------------|-------------------|-----------------|
| 16384 | 2048 | 16 | 8 | 128 | 1024 | 1.877 | 0.812 | **2.31x** |
| 16384 | 2048 | 16 | 8 | 128 | -1 | 7.360 | 3.237 | **2.27x** |
| 32768 | 4096 | 16 | 8 | 128 | 1024 | 3.513 | 1.476 | **2.38x** |
| 32768 | 4096 | 16 | 8 | 128 | -1 | 23.078 | 11.206 | **2.06x** |
| 65536 | 8192 | 16 | 8 | 128 | 1024 | 7.185 | 2.768 | **2.60x** |
| 65536 | 8192 | 16 | 8 | 128 | -1 | 82.918 | 41.965 | **1.98x** |
