# Pacing coalescing order path-minimax artifacts

This directory contains the compact E101 local-only all-pair minimax
path-allocation study built from E83, E89, E98, and E99.

- `pacing_coalescing_order_path_minimax_full.json` minimizes the maximum
  confidence-adjusted symmetric width over every policy pair, estimator
  kernel, 1% accuracy point, and actual-cost vertex.
- The 87,264-constraint convex problem is solved by a deterministic
  cutting-plane epigraph method. The active problem is a lower bound and the
  full separation pass certifies feasibility and optimality to tolerance.
- The design uses policy geometry only; it deliberately does not use
  observed slower-event outcomes or measured policy winners.
- The artifact compares the minimax surrogate and exact Clopper-Pearson
  robust regret with E98 linear geometry allocation and E99 mean water
  filling.
- Projected future successes retain E83 rates and round to integer counts.
  This is planning sensitivity; actual certification must use newly measured
  outcomes under a preregistered allocation.
- `SHA256SUMS` records the promoted raw artifact digest.

Cellwise binomial and rectangular-confidence limitations from E96 remain.
The source AML allocation remained canceled, so this is not new GPU evidence
and was not promoted to durable AML storage.
