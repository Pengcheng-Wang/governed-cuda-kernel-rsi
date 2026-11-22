# Pacing coalescing order adaptive-reuse artifacts

This directory contains the compact E103 local-only two-stage adaptive-reuse
study built from E83 and E102.

- `pacing_coalescing_order_adaptive_reuse_full.json` combines exact one-sided
  pilot and conditional holdout binomial p-values.
- It compares future-only Clopper-Pearson, exact Fisher product, a
  future-budget-weighted Stouffer combination, and an explicitly invalid
  pooled fixed-sample Clopper-Pearson oracle.
- Stouffer weights depend only on the preregistered global future budget:
  pilot weight 1 and future weight `sqrt(future_factor)`. They never depend
  on pilot outcomes or pilot-adaptive per-cell sample counts.
- Exact small adaptive-design enumeration verifies one-sided type-I error;
  interval roots, family alpha, and E102 reproduction are checked directly.
- The pooled fixed-sample oracle is shown only to quantify remaining
  conservatism. It is not valid after pilot-adaptive sample allocation.
- Projected future successes retain E83 rates and round to integer counts.
  This is planning sensitivity; actual certification must use newly measured
  outcomes under the declared two-stage design.
- `SHA256SUMS` records the promoted raw artifact digest.

The method is valid for one pilot and one future stage, not arbitrary
repeated peeking. Cellwise binomial and rectangular-confidence limitations
from E96 remain. The source AML allocation remained canceled, so this is not
new GPU evidence and was not promoted to durable AML storage.
