# E145 (`v136`, Round-19 RSI): Gated KalmaNet (`GKA`) Zero-Spill SM80 Triton Kernels (`arXiv:2511.21016`)

- **Paper / Source**: Songlin Yang, Yu Zhang, Zhiyuan Li, *"Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression"* (`arXiv:2511.21016`); official reference `fla-org/flash-linear-attention/tree/main/fla/ops/gka` & `fla/layers/gka.py`.
- **Frozen Evaluator**: `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch, `node-0`).
- **Artifacts**:
  - `smoke.json`, `rep{1,2,3}.json`
  - `gated_kalmanet_rsi_summary.json`, `gated_kalmanet_rsi_report.md`
  - `track3_kernel_matrix.csv`, `track3_layer_ablations.csv`, `track4_autoresearch.csv`, `track5_cp8_scaling.csv`
  - `SHA256SUMS.txt` (`10/10` verified entries)
