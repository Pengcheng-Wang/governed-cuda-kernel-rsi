# Pacing coalescing order dependence-sensitivity artifacts

This directory contains the compact E109 local-only correlated
posterior-predictive stage-weight study built from E83 and E108.

- `pacing_coalescing_order_stage_weight_dependence_sensitivity_full.json`
  compares `sqrt(2)` with 1.428317.
- The E108 independent first block is joined by global-, cost-, order-, and
  metric-correlated Gaussian-copula Jeffreys profiles plus a mixed
  75%-shared-variance profile.
- Each profile has 32 future datasets. Correlated profiles use antithetic
  latent draws and independent conditional binomial outcomes.
- Per-profile scenario wins, direct condition comparisons, exact
  simulation-frequency intervals, and candidate-relative regret are
  reported.
- E108's independent block is reproduced exactly and beta-CDF roundtrips
  validate the copula marginal transform.
- The profiles are dependence stress tests, not an estimated production
  traffic model or a policy-coverage proof.
- `SHA256SUMS` records the promoted raw artifact digest.

Only positive Gaussian-copula dependence is tested. Both candidates remain
post-hoc for E83 and may only be preregistered for a future independent
pilot. The source AML allocation remained canceled, so this is not new GPU
evidence and was not promoted to durable AML storage.
