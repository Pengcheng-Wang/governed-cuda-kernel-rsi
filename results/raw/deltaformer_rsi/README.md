# E151 (`v142`, Round-25 RSI) Benchmark Artifact Directory

- **Experiment**: `E151` (`v142`, Round-25 RSI) — `DeltaFormer: Understanding Transformer from the Perspective of Associative Memory` (`arXiv:2505.19488`)
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch)
- **Files**:
  - `smoke.json`, `rep1.json`, `rep2.json`, `rep3.json`: Raw smoke and 3-rep benchmark outputs from `code/bench_deltaformer_rsi.py`.
  - `deltaformer_rsi_summary.json`: Aggregated 3-rep statistics and compiler/oracle audits.
  - `deltaformer_rsi_report.md`: Markdown summary report.
  - `deltaformer_kernel_matrix.csv`, `deltaformer_layer_ablation.csv`, `deltaformer_autoresearch.csv`, `deltaformer_cp8_scaling.csv`: CSV tables.
  - `SHA256SUMS.txt`: Cryptographic manifest of all artifacts in this directory.
