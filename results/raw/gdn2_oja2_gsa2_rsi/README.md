# E146 (`v137`, Round-20 RSI): Gated DeltaNet-2 (`GDN-2`), Gated Oja Rule 2 (`Oja-2`), & Gated Slot Attention 2 (`GSA-2`) Zero-Spill SM80 Triton Kernels

- **Sources**: `NVlabs/GatedDeltaNet-2` (`fla/ops/gdn2`, `fla/layers/gdn2.py`) and `fla-org/flash-linear-attention` (`fla/ops/gated_oja_rule`, `fla/ops/gated_oja_rule2`, `fla/layers/gsa2.py`).
- **Frozen Evaluator**: `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch, `node-0`).
- **Artifacts**:
  - `smoke.json`, `rep{1,2,3}.json`
  - `gdn2_oja2_gsa2_rsi_summary.json`, `gdn2_oja2_gsa2_rsi_report.md`
  - `track3_kernel_matrix.csv`, `track3_layer_ablations.csv`, `track4_autoresearch.csv`, `track5_cp8_scaling.csv`
  - `SHA256SUMS.txt` (`10/10` verified entries)
