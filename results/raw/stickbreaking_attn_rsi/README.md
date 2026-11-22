# Round-26 RSI (`E152` / `v143`): Stick-Breaking Attention (`arXiv:2410.17980`, ICLR 2025)

Hardware-verified 5-track benchmark artifacts on `8x NVIDIA A100-SXM4-80GB` (`node-0`, SM clock locked at `1410 MHz`).

## Artifacts
- `smoke.json`: 5-track smoke verification gate.
- `rep1.json`, `rep2.json`, `rep3.json`: 3 independent full 8-GPU repetitions.
- `stickbreaking_attn_rsi_summary.json`: Aggregated mean/median/std/CV% summary.
- `stickbreaking_attn_rsi_report.md`: Detailed Markdown report.
- `stickbreaking_kernel_matrix.csv`: 8-shape training + 6-shape decoding matrix.
- `stickbreaking_layer_ablation.csv`: 5-stage `StickBreakingAttentionBlockRSI` (`A0..A4`) ablation.
- `stickbreaking_cp8_scaling.csv`: 8-GPU NVSwitch `StickBreaking-CP = 8` scaling.
- `stickbreaking_autoresearch.csv`: Fixed-`12.0s` wall-clock `val_bpb` autoresearch results.
- `SHA256SUMS.txt`: Cryptographic SHA-256 manifest of all artifacts in this directory.
