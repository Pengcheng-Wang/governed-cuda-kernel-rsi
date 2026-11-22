# Pacing coalescing order prevalence-certification artifacts

This directory contains the compact E90 local-only finite-sample prevalence certification audit built from E83 and E89.

- `pacing_coalescing_order_prevalence_certification_full.json` records exact L1 stability radii and multinomial sample-size summaries across all 3,003 prevalence profiles, four estimator kernels, and 101 accuracy points.
- `SHA256SUMS` records the promoted raw artifact digest.

For each prevalence profile, the selected policy's radius is the exact minimum L1 distance to a competitor regression boundary. A six-category distribution-free concentration inequality converts that radius into 95% and 99% confidence sample requirements. Zero-radius profiles are policy-identity boundaries and cannot be certified by more prevalence samples; infinite-radius profiles have no lower-regression competitor anywhere on the tested simplex.

The source allocation remained canceled, so this is not new GPU evidence and was not promoted to durable AML storage.
