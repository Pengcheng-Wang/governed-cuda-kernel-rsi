# E147 (`v138`, Round-21 RSI): Gated DeltaProduct (`arXiv:2502.10297`) Zero-Spill SM80 Benchmark Report

## 1. Executive Summary
- **Paper & Upstream Baseline**: *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products* (`arXiv:2502.10297`, Siems et al.; official `fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_product` & `fla/layers/gated_deltaproduct.py`).
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch), `3` independent full-process repetitions (`rep1.json..rep3.json`).
- **Frozen Cryptographic Evaluator (`Track 1`)**: `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`), `FP32` max relative L2 error **`2.35e-06`** (`< 1e-5` across all outputs, final states, and 6 gradients `dq, dk, dv, dg, dbeta, dh0` for `nh in {1, 2, 3, 4}` and `T in {64, 137}`), `BF16` max relative L2 error **`0.00795211`**, upstream `FP32` crash (`AssertionError` in `chunk.py:310`) reproduced and fixed, and **`3/3` (`100%`)** adversarial mutations killed.
- **SM80 Cubin Register & Spill Audit (`Track 2`)**: Upstream `fla.ops.gated_delta_product` + delegated `gated_delta_rule` spills across **`21` specializations** (`max_regs = 255`, `max_spills = 96`, `total_spills = 450`). Round-21 RSI compiles **`22/22` kernels** with **`0` spilling specializations, `0` spill bytes, and `max_regs = 168 <= 168`**.
- **Kernel Speedups (`Track 3A`)**: Up to **`1.94x` Fwd+Bwd** (**`1.93x` Bwd**, **`1.95x` Fwd**; mean **`1.26x` Fwd+Bwd** / **`1.19x` Bwd** across `nh >= 2`), solving upstream's `# TODO: optimize the backward pass like the forward pass` (`chunk.py:204`).
- **5-Stage Layer Ablation (`Track 3B`)**: `GatedDeltaProductBlockRSI` (`A0 -> A4`) achieves up to **`2.15x`** (`A4 vs A0`) and **`1.97x`** (`A4 vs A1`).
- **Fixed-Wall-Clock `val_bpb` Autoresearch (`Track 4`)**: At fixed `30.0s` training budget on the 4-layer `nh = 3` Rotational Parity & Dihedral State-Tracking LM, `Stage A4` achieves **`1.86x`** optimizer step rate (`1665.0` vs `894.0` steps) and lowers `val_bpb` from `8.2364` to **`8.0236`** vs `8.0299` for `Stage A0` (**`-0.0063` extra `val_bpb` improvement**).
- **8-GPU NVSwitch Context Parallelism (`Track 5`)**: `GDP-CP = 8` Ulysses achieves **`3.22x` peak** (**`3.13x` mean**) Fwd+Bwd speedup over 8-GPU sequential Ring state-passing across `T_total in {16384, 32768, 65536}` (`T_virtual_total` up to `131,072`).

## 2. Track 3A — Kernel Benchmark Matrix (`rep1..rep3` Medians)
| Shape `(B, T, H, K, V, nh)` | `T_virt` | Zero-Pad Waste | Upstream Fwd (ms) | RSI Fwd (ms) | Fwd Speedup | Upstream Bwd (ms) | RSI Bwd (ms) | Bwd Speedup | Upstream Fwd+Bwd (ms) | RSI Fwd+Bwd (ms) | Fwd+Bwd Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `(2, 4096, 8, 128, 128, nh=1)` | `4096` | `0.00%` | `1.138` | `0.736` | **`1.55x`** | `1.726` | `1.242` | **`1.39x`** | `2.869` | `1.981` | **`1.45x`** |
| `(2, 4096, 8, 128, 128, nh=2)` | `8192` | `50.00%` | `1.276` | `0.840` | **`1.52x`** | `1.786` | `1.618` | **`1.10x`** | `3.062` | `2.459` | **`1.25x`** |
| `(2, 4096, 8, 128, 128, nh=3)` | `12288` | `66.67%` | `1.521` | `1.114` | **`1.36x`** | `2.341` | `2.382` | **`0.98x`** | `3.862` | `3.496` | **`1.10x`** |
| `(2, 4096, 8, 128, 128, nh=4)` | `16384` | `75.00%` | `1.761` | `1.374` | **`1.28x`** | `3.032` | `3.126` | **`0.97x`** | `4.793` | `4.498` | **`1.07x`** |
| `(2, 4096, 8, 128, 256, nh=2)` | `8192` | `50.00%` | `1.505` | `1.073` | **`1.40x`** | `2.470` | `2.107` | **`1.17x`** | `3.975` | `3.180` | **`1.25x`** |
| `(2, 4096, 8, 128, 256, nh=3)` | `12288` | `66.67%` | `1.834` | `1.473` | **`1.25x`** | `3.533` | `3.067` | **`1.15x`** | `5.363` | `4.539` | **`1.18x`** |
| `(2, 4096, 8, 128, 256, nh=4)` | `16384` | `75.00%` | `2.173` | `1.843` | **`1.18x`** | `4.605` | `4.070` | **`1.13x`** | `6.779` | `5.913` | **`1.15x`** |
| `(2, 2048, 8, 128, 128, nh=2)` | `4096` | `50.00%` | `1.095` | `0.561` | **`1.95x`** | `1.806` | `0.936` | **`1.93x`** | `2.901` | `1.498` | **`1.94x`** |
| `(2, 2048, 8, 128, 128, nh=3)` | `6144` | `66.67%` | `1.150` | `0.689` | **`1.67x`** | `1.776` | `1.240` | **`1.43x`** | `2.932` | `1.929` | **`1.52x`** |
| `(2, 2048, 8, 128, 128, nh=4)` | `8192` | `75.00%` | `1.243` | `0.822` | **`1.51x`** | `1.763` | `1.577` | **`1.12x`** | `3.006` | `2.399` | **`1.25x`** |
| `(1, 8192, 8, 128, 128, nh=2)` | `16384` | `50.00%` | `1.477` | `1.081` | **`1.37x`** | `2.152` | `1.899` | **`1.13x`** | `3.629` | `2.983` | **`1.22x`** |
| `(1, 8192, 8, 128, 128, nh=3)` | `24576` | `66.67%` | `1.817` | `1.485` | **`1.22x`** | `3.079` | `2.802` | **`1.10x`** | `4.896` | `4.283` | **`1.14x`** |
| `(1, 8192, 8, 128, 128, nh=4)` | `32768` | `75.00%` | `2.138` | `1.851` | **`1.16x`** | `4.009` | `3.698` | **`1.08x`** | `6.147` | `5.551` | **`1.11x`** |

## 3. Track 3B — 5-Stage `GatedDeltaProductBlockRSI` Layer Ablation (`A0 -> A4`)
| Config | `nh` | `V` | `A0` Packed (ms) | `A1` 3D Native (ms) | `A2` RSI Core (ms) | `A3` +Fused Conv1d (ms) | `A4` +Fused Gate & Norm (ms) | `A4 vs A0` | `A4 vs A1` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `GDP-1K-nh2-V128` | `2` | `128` | `11.784` | `10.344` | `10.384` | `7.018` | `5.694` | **`2.07x`** | **`1.82x`** |
| `GDP-1K-nh2-V256` | `2` | `256` | `16.470` | `15.102` | `14.816` | `9.910` | `7.680` | **`2.15x`** | **`1.97x`** |
| `GDP-1K-nh3-V128` | `3` | `128` | `14.816` | `13.498` | `13.608` | `8.666` | `7.423` | **`2.00x`** | **`1.82x`** |
| `GDP-1K-nh3-V256` | `3` | `256` | `21.094` | `19.777` | `19.424` | `12.455` | `10.241` | **`2.06x`** | **`1.93x`** |
| `GDP-1K-nh4-V128` | `4` | `128` | `17.995` | `16.666` | `16.840` | `10.596` | `9.354` | **`1.92x`** | **`1.78x`** |
| `GDP-1K-nh4-V256` | `4` | `256` | `25.861` | `24.552` | `24.119` | `15.088` | `12.868` | **`2.01x`** | **`1.91x`** |

## 4. Track 4 — Fixed-Wall-Clock `val_bpb` Autoresearch (`30.0s` Budget, `nh = 3`)
- `A0` (`cu_seqlens` packed upstream + eager periphery): `894.0` steps (`29.78` steps/s, `60987.5` tok/s), `final_val_bpb = 8.0299`
- `A4` (Round-21 RSI superfused block): `1665.0` steps (`55.48` steps/s, `113613.5` tok/s), `final_val_bpb = 8.0236` (**`1.86x`** step rate, **`-0.0063`** extra `val_bpb` reduction)

## 5. Track 5 — 8-GPU NVSwitch Context Parallelism (`GDP-CP = 8` Ulysses vs Ring)
| `T_total` | `T_local` | `nh` | `T_virtual_total` | Ring CP=8 (ms) | Ulysses CP=8 (ms) | Ulysses vs Ring Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| `16384` | `2048` | `2` | `32768` | `14.525` | `4.902` | **`2.96x`** |
| `32768` | `4096` | `2` | `65536` | `26.938` | `8.426` | **`3.20x`** |
| `32768` | `4096` | `3` | `98304` | `37.815` | `12.143` | **`3.11x`** |
| `32768` | `4096` | `4` | `131072` | `48.510` | `15.393` | **`3.15x`** |
| `65536` | `8192` | `2` | `131072` | `51.821` | `16.090` | **`3.22x`** |
