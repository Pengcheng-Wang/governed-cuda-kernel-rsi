# Pacing coalescing order path-allocation artifacts

This directory contains the compact E98 local-only future-path allocation
study built from E83, E89, E96, and E97.

- `pacing_coalescing_order_path_allocation_full.json` compares uniform and
  mean-absolute-policy-contrast allocation of additional path comparisons.
- Each path allocation is crossed with equal and E97 mean-contrast
  familywise-alpha allocation.
- Geometry weights use only policy mappings, declared estimator kernels, and
  the accuracy grid. They do not use observed slower-event outcomes.
- Projected future successes retain E83 rates and round to integer counts.
  This is a planning sensitivity model; actual certification must use newly
  measured outcomes after preregistering the allocation.
- `SHA256SUMS` records the promoted raw artifact digest.

Cellwise binomial and rectangular-confidence limitations from E96 remain.
The source AML allocation remained canceled, so this is not new GPU evidence
and was not promoted to durable AML storage.
