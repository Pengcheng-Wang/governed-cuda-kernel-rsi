# E137 (`v128`): `nGPT: Normalized Transformer with Representation Learning on the Hypersphere` Training & Riemannian Retraction Kernels — Raw Hardware Artifacts

- **Paper**: Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun & Boris Ginsburg (NVIDIA), *nGPT: Normalized Transformer with Representation Learning on the Hypersphere*, `arXiv:2410.01131v2` (ICLR 2025); official repo `NVIDIA/ngpt`.
- **Hardware**: `1x8 NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch, `1410 MHz` locked SM clocks).
- **Evaluator**: Frozen `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
- **Contents**:
  - `ngpt_hypersphere_rsi_summary.json`: 3-repetition aggregated 9-gate verification summary (`all_nine_gates_passed = true`).
  - `ngpt_hypersphere_rsi_summary.md`: Markdown summary of 3-rep median metrics across Suite 0, Suite A, Suite B, and Suite C.
  - `ngpt_hypersphere_rsi_rep1.json`, `ngpt_hypersphere_rsi_rep2.json`, `ngpt_hypersphere_rsi_rep3.json`: Raw JSON results for all 3 independent OS-process repetitions.
  - `ngpt_hypersphere_rsi_smoke.json`: 8-GPU smoke verification pass (`9/9` gates passed).
  - `suite0_redteam_ledger.jsonl`, `suite_a_ledger_rep1.jsonl`, `suite_c_ledger_rep1.jsonl`: Immutable RSI ratchet & red-team audit ledgers.
  - `SHA256SUMS`: Standardized 7-entry SHA-256 manifest.
