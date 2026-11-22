# E139 (`v130`, Round-13 RSI): `MesaNet` Locally Optimal Test-Time Training Summary

- **Paper**: arXiv:2506.05233v2 (von Oswald, Scherrer, Kobayashi, Versari, Yang et al., Google Paradigms of Intelligence / Google DeepMind / MIT CSAIL, 2025)
- **Official Repository**: `fla-org/flash-linear-attention/tree/main/fla/ops/mesa_net`
- **Frozen Evaluator SHA-256**: `3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`
- **All 9 Gates Passed Across Smoke + 3 Repetitions**: `True`

## 1. Suite 0: Spectral Convergence, Numerical Parity, Richardson O(h^2) & Zero-Spill Audit
- **Upstream 30-Step Unpreconditioned CG (`x0=0, +1e-5`) Rel L2 Error**: `1.294e-03`
- **RSI Warm-Start Jacobi-PCG (`x0=D^-1 q, 1e-30`) Step-6 Rel L2 Error**: `4.851e-08` (`26671.6x` more accurate in `5.0x` fewer steps)
- **FP32 Max Rel L2 Error vs `torch.linalg.solve` Oracle**: `1.288e-06`
- **BF16 Max Rel L2 Error vs `torch.linalg.solve` Oracle**: `0.0045`
- **FP64 Richardson O(h^2) Checks Passed**: `7/7`
- **SM80 Cubin Spills**: `0` across all `15/15` Round-13 RSI kernels (`49` specializations) vs `1942` spills in monolithic `[64x128]` negative control

## 2. Suite A: Single-GPU Vector-Pareto Speedups (3-Rep Medians)
- **Track 1 (`Core MesaNet Layer Fwd+Bwd`)**: **`53.51x`** geomean speedup vs Upstream 30-Step CG (`A0`), **`31.91x`** vs `torch.compile` (`A2`), and up to **`2452.82x`** vs sequential `torch.linalg.solve` (`A1`).
- **Track 2 (`Full MesaNet Block Fwd+Bwd`)**: **`19.40x`** geomean speedup vs Upstream `MesaNetBlock` (`B0`) and **`12.23x`** vs `torch.compile` (`B2`).

## 3. Suite B: 8-GPU NVSwitch Dual-State Prefix-Scan `Mesa-CP = 8` (`T_global = 8,192`)
- **Gate G9 3-Step FP32 Parity**: `max_loss_rel_diff = 1.188e-07`, `max_param_rel_l2 = 9.390e-07`, `max_grad_rel_l2 = 2.125e-06` (vs `1.1446` error when omitting prefix states)
- **D=64 Speedup vs All-Gather Full-Sequence**: **`55.38x`** (`169.73 -> 3.06 ms`, `13.08x` lower peak HBM)
- **D=128 Speedup vs All-Gather Full-Sequence**: **`73.17x`** (`225.45 -> 3.08 ms`, `18.22x` lower peak HBM)

## 4. Suite C: Fixed-2.1s `8xA100 DDP` `val_bpb` Loop & 5-Arm Scaling Law
- **Fixed-2.1s `val_bpb` Loop**: `0.2698 -> 0.0466` (`-0.2231 val_bpb`), single-touch `test_bpb`: `0.2924 -> 0.0549` (`-0.2374 test_bpb`), recall accuracy: `92.80% -> 99.02%`.
