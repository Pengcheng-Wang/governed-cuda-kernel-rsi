# Pacing coalescing order stage-weight outcome-sensitivity artifacts

This directory contains the compact E107 local-only outcome-sensitivity
study built from E83 and E106.

- `pacing_coalescing_order_stage_weight_outcome_sensitivity_full.json`
  compares multipliers 1, `sqrt(2)`, 1.428317, 1.5, and 1.6.
- Four deterministic future-count projections cover floor, nearest,
  aggregate-balanced, and ceiling integer rounding.
- Fixed-seed point-rate binomial and Jeffreys posterior-predictive scenarios
  test whether E106's small continuous-weight advantage survives plausible
  future outcome variation.
- Candidate-relative regret, winner counts, family-specific summaries, and
  direct 1.428317-versus-`sqrt(2)` comparisons are reported.
- The simulations test planning stability, not frequentist coverage. Every
  candidate remains post-hoc for E83 and may only be preregistered for a
  future independent pilot.
- `SHA256SUMS` records the promoted raw artifact digest.

The posterior treats E83 risk cells as independent and inherits earlier
binomial/rectangular limitations. The source AML allocation remained
canceled, so this is not new GPU evidence and was not promoted to durable
AML storage.
