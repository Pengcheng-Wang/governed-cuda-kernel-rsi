# Pacing coalescing order Chernoff-certification artifacts

This directory contains the compact E93 local-only policy-specific
finite-sample certification audit built from E83, E89, E91, and E92.

- `pacing_coalescing_order_chernoff_certification_full.json` records
  categorical Chernoff rates and union-bound sample summaries across all 3,003
  prevalence profiles, four estimator kernels, 101 accuracy points, and
  0/1/5/10/25/50 excess regressions per million path comparisons.
- `SHA256SUMS` records the promoted raw artifact digest.

For each true over-tolerance competitor, empirical selection requires its
sample mean regression difference from the true winner to be nonpositive.
E93 computes the exact categorical Chernoff exponent for that event and sums
only the bad-policy bounds. The output compares this finite-sample certificate
with E91's distribution-free L1 bound and E92's representative empirical
calibration.

The source allocation remained canceled, so this is not new GPU evidence and
was not promoted to durable AML storage.
