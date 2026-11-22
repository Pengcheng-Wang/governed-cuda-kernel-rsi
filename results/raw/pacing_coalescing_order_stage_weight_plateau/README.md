# Pacing coalescing order stage-weight-plateau artifacts

This directory contains the compact E105 local-only dense Stouffer
stage-weight study built from E83 and E104.

- `pacing_coalescing_order_stage_weight_plateau_full.json` evaluates 21
  fixed full-alpha future-stage multipliers: a coarse 0.5–3.0 grid, a fine
  1.0–2.0 grid, and the exact `1/sqrt(2)` and `sqrt(2)` anchors.
- Every candidate is a valid two-stage alternative only if selected before
  observing the pilot.
- The artifact reports per-budget p90/maximum optima, 0.25/0.5/1% tested
  plateaus, local winner counts, candidate regret relative to local optima,
  and one multiplier minimizing worst global relative regret across budgets
  and confidences.
- The robust multiplier and dense-grid optima are planning recommendations
  for a future independent pilot. They do not authorize retuning the existing
  E83 pilot at full alpha.
- Exact small adaptive-design enumeration, E104 anchor reproduction, root
  validation, and monotonicity checks are included.
- Projected future successes retain E83 rates and round to integer counts.
  This is planning sensitivity, not new measured paths.
- `SHA256SUMS` records the promoted raw artifact digest.

The finite grid does not prove a continuous optimum. The source AML
allocation remained canceled, so this is not new GPU evidence and was not
promoted to durable AML storage.
