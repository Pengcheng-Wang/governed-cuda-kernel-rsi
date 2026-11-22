# E136 (`v127`): `Differential Transformer V1 & V2` Common-Mode Attention-Noise Cancellation Training Kernels — Raw Hardware Artifacts

- **Paper**: Tianchi Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang & Furu Wei, *Differential Transformer*, `arXiv:2410.05258` (ICLR 2025) + January 2026 *Differential Transformer V2* (`DIFF V2`); official repo `microsoft/unilm/tree/master/Diff-Transformer` & `Diff-Transformer/Diff-Transformer-V2`.
- **Hardware**: `1x8 NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch, SM clock locked at `1,410 MHz`).
- **Evaluator**: Frozen `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
- **Contents**:
  - `diff_transformer_rsi_summary.json`: 3-repetition aggregated 9-gate verification summary (`all_nine_gates_passed = true`).
  - `diff_transformer_rsi_summary.md`: Markdown summary of 3-rep median metrics across Suite 0, Suite A, Suite B, and Suite C.
  - `diff_transformer_rsi_rep1.json`, `diff_transformer_rsi_rep2.json`, `diff_transformer_rsi_rep3.json`: Raw JSON results for all 3 independent OS-process repetitions.
  - `diff_transformer_rsi_smoke.json`: 8-GPU smoke verification pass (`9/9` gates passed).
  - `suite0_redteam_ledger.jsonl`, `suite_a_ledger_rep1.jsonl`, `suite_c_ledger_rep1.jsonl`: Immutable RSI ratchet & red-team audit ledgers.
  - `SHA256SUMS`: Standardized 7-entry SHA-256 manifest.
