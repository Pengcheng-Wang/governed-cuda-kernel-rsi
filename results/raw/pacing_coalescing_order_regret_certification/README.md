# Pacing coalescing order regret-certification artifacts

This directory contains the compact E91 local-only regret-tolerant prevalence
certification audit built from E83, E89, and E90.

- `pacing_coalescing_order_regret_certification_full.json` records pairwise-safe
  L1 radii and multinomial sample-size summaries across all 3,003 prevalence
  profiles, four estimator kernels, 101 accuracy points, and
  0/1/5/10/25/50 excess regressions per million path comparisons.
- `SHA256SUMS` records the promoted raw artifact digest.

For each true prevalence profile, an empirical policy is acceptable when its
true weighted regression exceeds the true winner by no more than the declared
tolerance. The safe radius is the minimum L1 distance at which any
over-tolerance policy can tie the true winner on empirical regression. A
six-category distribution-free concentration inequality converts that radius
into 95% and 99% confidence sample requirements.

The source allocation remained canceled, so this is not new GPU evidence and
was not promoted to durable AML storage.
