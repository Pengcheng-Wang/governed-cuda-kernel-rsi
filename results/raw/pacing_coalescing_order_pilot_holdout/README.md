# Pacing coalescing order pilot-holdout artifacts

This directory contains the compact E102 local-only pilot/holdout
path-allocation study built from E83, E89, E97, and E99.

- `pacing_coalescing_order_pilot_holdout_full.json` treats E83 as a fixed
  pilot that selects deployment policies and future path allocation.
- Final simultaneous Clopper-Pearson intervals use projected future holdout
  counts only; no E83 pilot count enters the certificate.
- The study compares uniform holdout sampling, outcome-independent geometry,
  pilot-selected-policy/all-competitor geometry, and current-worst-only pilot
  geometry. Water-filled methods reserve 10% of each future budget equally
  across all 72 cells.
- The artifact reports safe outcome-aware gains, evidence requirements,
  score concentration, the failure of current-worst-only overfocus, and the
  statistical price of discarding pilot counts rather than reusing them.
- Projected future successes retain E83 rates and round to integer counts.
  This is planning sensitivity; actual certification must use newly measured
  holdout outcomes under the fixed pilot design.
- `SHA256SUMS` records the promoted raw artifact digest.

Cellwise binomial and rectangular-confidence limitations from E96 remain.
The source AML allocation remained canceled, so this is not new GPU evidence
and was not promoted to durable AML storage.
