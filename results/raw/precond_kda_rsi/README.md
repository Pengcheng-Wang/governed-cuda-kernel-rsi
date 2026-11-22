# E142 (`v133`, Round-16 RSI): Preconditioned DeltaNet (`PrecondKDA`) & Adaptive Diagonal Curvature Preconditioning (`ATK`)

- **Paper / Source**: Moonshot AI / Kimi Team & FLA Team, *"Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences"* (`arXiv:2604.21100`); official reference `fla-org/flash-linear-attention/tree/main/fla/ops/precond_kda` & `fla/ops/atk`.
- **Frozen Evaluator**: `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch, `node-0`).
- **Artifacts**:
  - `precond_kda_rsi_smoke.json`, `precond_kda_rsi_rep{1,2,3}.json` (`9/9` gates passed across all runs)
  - `precond_kda_rsi_summary.json`, `precond_kda_rsi_report.md`
  - `suiteA_track1_core_pkda.csv`, `suiteA_track2_full_block.csv`, `suiteB_cp8_scaling.csv`, `suiteC_rsi_candidates.csv`, `suiteC_scaling_law_5arm.csv`
  - `SHA256SUMS.txt` (`10/10` verified entries)
