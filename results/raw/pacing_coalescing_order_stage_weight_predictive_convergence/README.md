# Pacing coalescing order predictive-convergence artifacts

This directory contains the compact E108 local-only posterior-predictive
convergence study built from E83 and E107.

- `pacing_coalescing_order_stage_weight_predictive_convergence_full.json`
  compares only `sqrt(2)` and E106's 1.428317 multiplier.
- Four independent deterministic seed blocks provide 128 Jeffreys
  posterior-predictive future datasets.
- Scenario-level minimax wins, direct condition comparisons, exact
  Clopper-Pearson simulation-frequency intervals, seed-block stability, and
  prefix convergence are reported.
- Batched interval inversion reproduces E107's first 32-scenario block and a
  single-scenario evaluator control.
- The simulation quantifies Monte Carlo stability, not frequentist coverage.
  Both candidates remain post-hoc for E83 and may only be preregistered for a
  future independent pilot.
- `SHA256SUMS` records the promoted raw artifact digest.

The posterior treats E83 risk cells as independent and inherits earlier
binomial/rectangular limitations. The source AML allocation remained
canceled, so this is not new GPU evidence and was not promoted to durable
AML storage.
