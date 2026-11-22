# E150 (`v141`, Round-24 RSI) Benchmark Artifact Directory

- **Experiment**: `E150` (`v141`, Round-24 RSI) — `Wall Attention: Length Generalization With Diagonal Gates` (`Tilde Research`)
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch)
- **Files**:
  - `smoke.json`, `rep1.json`, `rep2.json`, `rep3.json`: Raw smoke and 3-rep benchmark outputs from `code/bench_wall_attn_rsi.py`.
  - `wall_attn_rsi_summary.json`: Aggregated 3-rep statistics and compiler/oracle audits.
  - `wall_attn_rsi_report.md`: Markdown summary report.
  - `wall_attn_kernel_matrix.csv`, `wall_attn_layer_ablation.csv`, `wall_attn_autoresearch.csv`, `wall_attn_cp8_scaling.csv`: CSV tables.
  - `SHA256SUMS.txt`: Cryptographic checksums of all artifacts.
