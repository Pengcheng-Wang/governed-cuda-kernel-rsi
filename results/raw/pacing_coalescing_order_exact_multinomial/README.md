# Pacing coalescing order exact-multinomial artifacts

This directory contains the compact E95 local-only exact finite-sample audit
built from E83, E92, and E94.

- `pacing_coalescing_order_exact_multinomial_full.json` records 192
  representative cases. It scans every sample size through the E94 bound when
  the cumulative multinomial count space has at most one million states and
  point-checks E92/E93/E94 sample sizes when each has at most one million
  states.
- `SHA256SUMS` records the promoted raw artifact digest.

Each count vector is evaluated with the full
regression/delay/complexity/faster/name lexicographic selector, and its exact
multinomial probability is summed. Because lattice probabilities can be
nonmonotone in sample size, the reported exact certificate is one plus the
last sample size whose failure probability exceeds 5%; E94 guarantees all
larger sample sizes after its bound.

The source allocation remained canceled, so this is not new GPU evidence and
was not promoted to durable AML storage.
