# Pacing coalescing order prevalence artifacts

This directory contains the compact E89 local-only cost-prevalence robustness audit built from E83 and E88.

- `pacing_coalescing_order_prevalence_full.json` records exhaustive 10%-simplex winner counts and distribution-free maximum per-cost regret across all E88 estimator kernels and 101 accuracy points.
- `SHA256SUMS` records the promoted raw artifact digest.

The six actual cost-cell prevalences are enumerated as all integer compositions of ten, yielding 3,003 normalized profiles including all six simplex vertices. For each profile, policies are compared by expected regression, delay burden, configuration segments, acceleration, and deterministic name. A separate minimax-regret policy minimizes the maximum regression regret over individual cost cells.

The source allocation remained canceled, so this is not new GPU evidence and was not promoted to durable AML storage.
