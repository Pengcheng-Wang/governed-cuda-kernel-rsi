# Pacing coalescing order accuracy-frontier artifacts

This directory contains the compact E88 local-only cost-estimator accuracy frontier built from E83–E87.

- `pacing_coalescing_order_accuracy_frontier_full.json` records exact policy crossover points under symmetric radius-1/radius-2 and one-sided under/over-estimation kernels.
- `SHA256SUMS` records the promoted raw artifact digest.

The exact cost-classification probability is swept from zero to one. Non-exact mass follows a declared ordinal offset kernel and is clamped at grid boundaries. Expected regression, acceleration, positive-delay burden, savings, and support maximum delay are computed from the full E83 cell counts. Because expected counts are affine in accuracy, all pairwise crossover points are solved analytically and verified against a 101-point grid.

The source allocation remained canceled, so this is not new GPU evidence and was not promoted to durable AML storage.
