# Pacing coalescing order certificate-calibration artifacts

This directory contains the compact E92 local-only finite-sample calibration
of E91's regret-tolerant prevalence certificates.

- `pacing_coalescing_order_certificate_calibration_full.json` records 192
  representative cases spanning four estimator kernels, four accuracies, four
  regret tolerances, and easy/median/hard E91 sample-requirement profiles.
- Each case uses 20,000 deterministic multinomial replicates at seven tested
  fractions of the E91 distribution-free sample bound.
- `SHA256SUMS` records the promoted raw artifact digest.

The simulation runs the full regression/delay/complexity/faster/name
lexicographic policy selector and reports actual over-tolerance selection,
one-sided 95% Clopper-Pearson bounds, L1-radius exceedance, and the first
tested sample fraction whose Monte Carlo upper bound is below 5%.

The source allocation remained canceled, so this is not new GPU evidence and
was not promoted to durable AML storage.
