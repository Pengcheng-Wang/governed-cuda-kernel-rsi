# Pacing coalescing order risk-alpha-allocation artifacts

This directory contains the compact E97 local-only familywise-alpha
allocation audit built from E83, E89, and E96.

- `pacing_coalescing_order_risk_alpha_allocation_full.json` compares equal,
  mean-absolute-contrast, RMS-contrast, and maximum-contrast allocations over
  72 lower and 72 upper E83 risk bounds.
- Each geometry-weighted allocation reserves 10% of alpha equally and assigns
  90% from policy mappings, declared estimator kernels, and the accuracy grid;
  it does not use observed slower-event outcomes.
- A selection-safe best-of-four divides family alpha across all four
  candidates before choosing profilewise.
- `SHA256SUMS` records the promoted raw artifact digest.

The evidence multipliers are unchanged-rate planning sensitivity, not new
simulated paths. Cellwise binomial and rectangular-confidence limitations
from E96 remain.

The source allocation remained canceled, so this is not new GPU evidence and
was not promoted to durable AML storage.
