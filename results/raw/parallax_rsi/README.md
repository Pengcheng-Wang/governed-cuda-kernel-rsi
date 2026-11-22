# E149 (`v140`, Round-23 RSI) Benchmark Artifact Directory

- **Experiment**: `E149` (`v140`, Round-23 RSI) — `Parallax: Parameterized Local Linear Attention` (`arXiv:2605.29157`)
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch)
- **Files**:
  - `smoke.json`, `rep1.json`, `rep2.json`, `rep3.json`: Raw smoke and 3-rep benchmark outputs from `code/bench_parallax_rsi.py`.
  - `parallax_rsi_summary.json`: Aggregated 3-rep median/IQR/CoV statistics and cubin/parity/mutation audits.
  - `parallax_rsi_report.md`: Markdown summary report.
  - `parallax_kernel_matrix.csv`, `parallax_layer_ablation.csv`, `parallax_autoresearch.csv`, `parallax_cp8_scaling.csv`: CSV tables.
  - `SHA256SUMS.txt`: Cryptographic checksums of all artifacts.
