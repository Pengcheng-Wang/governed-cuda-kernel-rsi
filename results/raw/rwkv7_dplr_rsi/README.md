# E141 (`v132`, Round-15 RSI): RWKV-7 "Goose" & Generalized Diagonal-Plus-Low-Rank (`DPLR`) Delta Rule

- **Paper / Source**: Peng, Zhang, Goldstein et al., *"RWKV-7 'Goose' with Expressive Dynamic State Evolution"* (`arXiv:2503.14456`) & Hu et al., *"Comba: Improving Bilinear RNNs with Closed-loop Control"* (`arXiv:2506.02475`); official reference `fla-org/flash-linear-attention/tree/main/fla/ops/generalized_delta_rule/dplr` & `fla/ops/rwkv7`.
- **Frozen Evaluator**: `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch, `node-0`).
- **Artifacts**:
  - `rwkv7_dplr_rsi_smoke.json`, `rwkv7_dplr_rsi_rep{1,2,3}.json` (`9/9` gates passed across all runs)
  - `rwkv7_dplr_rsi_summary.json`, `rwkv7_dplr_rsi_report.md`
  - `suiteA_track1_core_dplr.csv`, `suiteA_track2_full_block.csv`, `suiteB_cp8_scaling.csv`, `suiteC_rsi_candidates.csv`, `suiteC_scaling_law_5arm.csv`
  - `SHA256SUMS.txt` (`10/10` verified entries)
