# E135 (`v126`): `Mixture-of-Recursions (MoR)` Dynamic Latent-Depth Training Kernels — Raw Hardware Artifacts

- **Paper**: Sangmin Bae et al., *Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation*, `arXiv:2507.10524v2` (NeurIPS 2025); official repo `raymin0223/mixture_of_recursions` (`commit 82c97f29cd140efa01f5cc4fe0de73a4c3b6ffd6`).
- **Hardware**: `1x8 NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch), locked at `1,410 MHz` SM clock.
- **Evaluator**: Frozen `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f07831179f4665fb83fee3c2b11e3269ccf106c3f412336bf77a4925904e`).
- **Contents**:
  - `mor_recursive_depth_rsi_full.json`: 3-repetition aggregated 9-gate verification summary (`all_gates_passed = True`).
  - `mor_recursive_depth_rsi_summary.md`: Markdown summary of 3-rep median / IQR metrics.
  - `rep1/`, `rep2/`, `rep3/`: Raw JSON and immutable RSI/autoresearch TSV ledgers for all 3 independent OS-process repetitions.
  - `smoke/`: 8-GPU smoke verification pass and ledgers.
