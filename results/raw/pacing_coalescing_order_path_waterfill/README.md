# Pacing coalescing order path-waterfill artifacts

This directory contains the compact E99 local-only water-filled future-path
allocation study built from E83, E89, E97, and E98.

- `pacing_coalescing_order_path_waterfill_full.json` replaces E98's linear
  additional-path weights with an analytic lower-bounded final-count design.
- At 95% family confidence with E97 mean-contrast alpha weights, the design
  minimizes the outcome-independent Hoeffding-width surrogate
  `sum_i c_i sqrt(log(2 / (alpha w_i)) / (2 n_i))`.
- KKT conditions assign active final cell counts in proportion to the
  two-thirds power of confidence-adjusted policy contrast while preserving
  every existing E83 comparison and the exact total budget.
- Projected future successes retain E83 rates and round to integer counts.
  This is planning sensitivity; actual certification must use newly measured
  outcomes under a preregistered allocation.
- `SHA256SUMS` records the promoted raw artifact digest.

Cellwise binomial and rectangular-confidence limitations from E96 remain.
The source AML allocation remained canceled, so this is not new GPU evidence
and was not promoted to durable AML storage.
