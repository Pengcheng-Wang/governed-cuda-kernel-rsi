# E148 (`v139`, Round-22 RSI) Benchmark Artifact Directory

- **Experiment**: `E148` (`v139`, Round-22 RSI) — `Cyclic Flow Attention (CyFA): Linear Sequence Modeling with Relative-Time-Partitioned Memory` (`arXiv:2609.36259`)
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch)
- **Files**:
  - `rep1.json`, `rep2.json`, `rep3.json`: Raw 3-rep benchmark outputs from `code/bench_cyfa_rsi.py --mode full`.
  - `cyfa_rsi_summary.json`: Aggregated 3-rep median/IQR/CoV statistics and cubin/parity/mutation audits.
  - `cyfa_rsi_report.md`: Markdown summary report.
  - `cyfa_kernel_matrix.csv`, `cyfa_layer_ablation.csv`, `cyfa_autoresearch.csv`, `cyfa_cp8_scaling.csv`: CSV tables.
  - `SHA256SUMS.txt`: Cryptographic checksums of all artifacts.
