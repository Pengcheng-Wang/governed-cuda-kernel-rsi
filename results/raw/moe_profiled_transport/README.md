# Profiled fixed-split MoE transport

This experiment removes per-batch count exchange and CPU split extraction
without sending every uniform static slot. A source-destination capacity
matrix is calibrated before the measured step and reused as fixed
`all_to_all_single` input/output split lists.

Four profiles are compared for balanced, bias-1, and bias-4 routing:

1. Uniform exact: the evaluation matrix global maximum in every cell.
2. Calibrated 1.125x: elementwise maximum across eight disjoint calibration
   batches with multiplicative headroom.
3. Calibrated +16: the same calibration maximum plus sixteen rows per cell.
4. Oracle matrix: the held-out evaluation count matrix, a zero-padding upper
   bound rather than a deployable profile.

The dynamic control retains the same highest router weights under each matrix
and performs per-batch count exchange. The fixed path packs into the profiled
matrix and reuses static split lists. Payload-only and full route-to-output
critical paths are measured separately.

The matrix includes a 100-token two-GPU gate, repeated 8- and 24-GPU runs, two
8-GPU crossovers, and five 24-GPU crossovers including reversed profile order.
All static outputs must match the dynamic control and independently masked
global-expert reference bit-for-bit.