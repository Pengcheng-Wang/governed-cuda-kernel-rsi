# Pacing coalescing order global-Chernoff artifacts

This directory contains the compact E94 local-only global-minimizer Chernoff
audit built from E83, E92, and E93.

- `pacing_coalescing_order_global_chernoff_full.json` records 192
  representative cases spanning four estimator kernels, four accuracies, four
  regret tolerances, and easy/median/hard E91 sample-requirement profiles.
- `SHA256SUMS` records the promoted raw artifact digest.

For each true over-tolerance policy, E94 bounds the event that it has no higher
empirical regression than every tested policy. A multivariate exponential
tilt enforces all pairwise inequalities simultaneously. The exact pairwise
true-winner tilt warm-starts the optimizer and remains a valid fallback, so
the resulting sample requirement cannot exceed E93's.

The source allocation remained canceled, so this is not new GPU evidence and
was not promoted to durable AML storage.
