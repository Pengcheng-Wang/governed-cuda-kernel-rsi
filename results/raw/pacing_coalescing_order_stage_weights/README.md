# Pacing coalescing order stage-weight artifacts

This directory contains the compact E104 local-only Stouffer stage-weight
study built from E83 and E103.

- `pacing_coalescing_order_stage_weights_full.json` compares fixed future
  weights equal to `multiplier * sqrt(global_future_factor)` for multipliers
  `1/sqrt(2)`, `1`, `sqrt(2)`, `2`, and `3`.
- Each fixed multiplier is a valid alternative only if selected before
  observing the pilot. A selection-safe best-of-five divides family alpha
  across every candidate before profilewise choice.
- Exact small adaptive-design enumeration verifies type-I control for fixed
  multipliers.
- A separate exhaustive stress grid demonstrates that realized per-cell
  sample-size weights depending on pilot outcomes can exceed nominal type-I
  error.
- The artifact reports individual and selection-safe regret frontiers,
  multiplier winners, coverage requirements, root checks, and the adaptive
  weight counterexample.
- Projected future successes retain E83 rates and round to integer counts.
  This is planning sensitivity; actual certification must use newly measured
  outcomes under a preregistered rule.
- `SHA256SUMS` records the promoted raw artifact digest.

The method covers one pilot and one future stage, not repeated peeking.
Cellwise binomial and rectangular-confidence limitations from E96 remain.
The source AML allocation remained canceled, so this is not new GPU evidence
and was not promoted to durable AML storage.
