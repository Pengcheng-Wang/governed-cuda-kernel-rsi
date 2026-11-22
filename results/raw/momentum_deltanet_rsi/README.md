# E144 (`v135`, Round-18 RSI): Momentum DeltaNet Coupled Dual-State `[S, M]` (`arXiv:2605.05838`)

- **Paper / Source**: FLA Team, *"Momentum DeltaNet: Coupled Dual-State Heavy-Ball Momentum Test-Time Gradient Descent"* (`arXiv:2605.05838`); official reference `fla-org/flash-linear-attention/tree/main/fla/ops/momentum_delta_rule` & `fla/layers/momentum_deltanet.py`.
- **Frozen Evaluator**: `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch, `node-0`).
- **Artifacts**:
  - `smoke.json`, `rep{1,2,3}.json`
  - `momentum_deltanet_rsi_summary.json`, `momentum_deltanet_rsi_report.md`
  - `track3_kernel_sweep.csv`, `track3_block_stage_ladder.csv`, `track4_autoresearch.csv`, `track5_cp8_scaling.csv`
  - `SHA256SUMS.txt` (`10/10` verified entries)
