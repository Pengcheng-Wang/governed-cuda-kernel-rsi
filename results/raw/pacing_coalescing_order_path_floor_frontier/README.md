# Pacing coalescing order path-floor-frontier artifacts

This directory contains the compact E100 local-only floor-regularized
water-fill study built from E83, E89, E98, and E99.

- `pacing_coalescing_order_path_floor_frontier_full.json` reserves
  0/10/25/50/75/100% of each additional budget using E98 linear
  mean-contrast weights, then water-fills the remaining budget with E99's
  mean-width objective.
- A zero floor exactly reproduces E99; a full floor exactly reproduces E98.
- The artifact reports exact global and per-kernel p90/maximum-regret
  frontiers, local Pareto membership, endpoint dominance, sampling
  concentration, and coverage requirements.
- Each floor is a preregistered alternative, not a post-hoc selectable
  confidence family.
- Projected future successes retain E83 rates and round to integer counts.
  This is planning sensitivity; actual certification must use newly measured
  outcomes under a preregistered allocation.
- `SHA256SUMS` records the promoted raw artifact digest.

Cellwise binomial and rectangular-confidence limitations from E96 remain.
The source AML allocation remained canceled, so this is not new GPU evidence
and was not promoted to durable AML storage.
