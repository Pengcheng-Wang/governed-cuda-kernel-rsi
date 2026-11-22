# E138 (`v129`, Round-12 RSI): `Log-Linear Attention` (`H-Mamba-2` & `H-GDN`) Artifacts

- **Hardware**: `8x NVIDIA A100-SXM4-80GB` (`NV12` NVSwitch, SM clock locked at `1410 MHz`)
- **Source Paper & Official Repository Audited**: Han Guo, Songlin Yang, Tarushii Goel, Eric P. Xing, Tri Dao & Yoon Kim, *Log-Linear Attention* (`arXiv:2506.04761v3`, Mar 2026, `HanGuo97/log-linear-attention`)
- **Verification**: 1 smoke + 3 independent 8-GPU `--mode full` repetitions (`all_nine_gates_passed = True`), `17/17` zero-spill Triton kernels (`0` cubin spills).
