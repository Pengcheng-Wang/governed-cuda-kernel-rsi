# Pacing coalescing order continuous stage-weight artifacts

This directory contains the compact E106 local-only numerical continuous
Stouffer stage-weight study built from E83 and E105.

- `pacing_coalescing_order_stage_weight_continuous_full.json` uses vectorized
  exact binomial-tail bisection, nested meshes, local parabolic candidate
  refinement, and direct plateau-root solving over multipliers 0.5-3.0.
- The study estimates condition-specific continuous optima, one minimax
  robust multiplier, and common 0.25/0.5/1% relative-regret intervals.
- E105's 21 candidates and selected scalar interval roots are reproduced as
  validation controls.
- Every fixed candidate remains valid only if selected before observing the
  pilot. The numerical optimum is a planning recommendation for a future
  independent pilot, not permission to retune E83 at full alpha.
- Projected future successes retain E83 rates and integer rounding. This is
  planning sensitivity, not new measured paths.
- `SHA256SUMS` records the promoted raw artifact digest.

The numerical refinement is not a symbolic proof of a unique optimum. The
source AML allocation remained canceled, so this is not new GPU evidence and
was not promoted to durable AML storage.
