# Pacing coalescing order compression artifacts

This directory contains the compact E85 local-only policy-compression audit built from E83 and E84.

- `pacing_coalescing_order_compression_full.json` records exact contiguous marginal-cost bucket policies for every feasible observed delay cap and configuration-key budget.
- `SHA256SUMS` records the promoted raw artifact digest.

The audit enumerates all `4^6` order mappings for each drift metric and delay cap, compresses identical adjacent orders into configuration segments, and combines metrics under an explicit lexicographic objective: regression count, realized maximum delay, configuration keys, acceleration count, and deterministic order rank. It verifies monotonicity across key budgets and delay caps and reproduces the E84 metric-cost oracle.

The source allocation remained canceled, so this is not new GPU evidence and was not promoted to durable AML storage.
