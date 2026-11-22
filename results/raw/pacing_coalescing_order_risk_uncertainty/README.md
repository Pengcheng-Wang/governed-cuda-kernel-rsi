# Pacing coalescing order risk-uncertainty artifacts

This directory contains the compact E96 local-only finite-path uncertainty
audit built from E83 and E89.

- `pacing_coalescing_order_risk_uncertainty_full.json` records simultaneous
  95%/99% Bonferroni Clopper-Pearson envelopes for 72 E83 slower-event cells
  and their exact rectangular propagation through all policies, four
  estimator kernels, 101 accuracies, and 3,003 prevalence profiles.
- It also sweeps unchanged-rate integer evidence multipliers from 1x to 256x.
- `SHA256SUMS` records the promoted raw artifact digest.

The replication multipliers scale observed successes and comparisons at fixed
empirical rates. They are planning sensitivity, not new simulated paths.
Clopper-Pearson treats each aggregate cell as binomial; shared paths and
heterogeneous simulation strata are listed as model limitations.

The source allocation remained canceled, so this is not new GPU evidence and
was not promoted to durable AML storage.
