# E143 (`v134`, Round-17 RSI): Triadic Gated DeltaNet (`Triadic-GDN`) & Triadic Kimi Delta Attention (`Triadic-KDA`)

- **Paper / Source**: FLA Team, *"Triadic Gated DeltaNet & Triadic Kimi Delta Attention: Kronecker Tensor-Product State Expansion & Hadamard-Factored UT/WY Linear Recurrences"* (`arXiv:2609.36529`); official reference `fla-org/flash-linear-attention/tree/main/fla/ops/triadic_gdn` & `fla/ops/triadic_kda`.
- **Frozen Evaluator**: `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch, `node-0`).
- **Artifacts**:
  - `triadic_gdn_rsi_smoke.json`, `triadic_gdn_rsi_rep{1,2,3}.json`
  - `triadic_gdn_rsi_summary.json`, `triadic_gdn_rsi_report.md`
  - `track3_kernel_sweep.csv`, `track3_block_stage_ladder.csv`, `track4_autoresearch.csv`, `track5_cp8_scaling.csv`
  - `SHA256SUMS.txt` (`9/9` verified entries)
