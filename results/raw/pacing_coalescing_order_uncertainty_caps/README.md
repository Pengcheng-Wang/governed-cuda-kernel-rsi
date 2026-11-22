# Pacing coalescing order uncertainty-cap artifacts

This directory contains the compact E87 local-only joint cost-uncertainty and delay-cap audit built from E83 and E86.

- `pacing_coalescing_order_uncertainty_caps_full.json` records radius-1/radius-2 uncertainty-aware member-order policies over every feasible observed hard delay cap.
- `SHA256SUMS` records the promoted raw artifact digest.

For each estimated cost and breach metric, a candidate order must satisfy the delay cap for every actual cost in the uncertainty set. Selection minimizes worst-cell regression, then delay, aggregate regression, acceleration, and deterministic order rank. Results report all-pair and adversarial envelopes, first caps beating fixed newest-first, the cap that minimizes aggregate risk, and the first cap restoring the fixed worst-cell envelope.

The source allocation remained canceled, so this is not new GPU evidence and was not promoted to durable AML storage.
