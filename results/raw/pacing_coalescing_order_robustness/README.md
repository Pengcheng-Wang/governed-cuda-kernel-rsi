# Pacing coalescing order robustness artifacts

This directory contains the compact E84 local-only policy-selection audit built from E83's exact-batch member-order frontier.

- `pacing_coalescing_order_robustness_full.json` records per-cost lexicographic oracles, robust fixed-order selections, domain-aware and drift-adaptive policies, exact faster/regression counts, metric-cost oracle regret, and the multi-objective Pareto front.
- `SHA256SUMS` records the promoted raw artifact digest.

The audit compares fixed, domain-aware, cost-aware, metric-aware, and metric+cost-aware member-order policies without rerunning Bernoulli paths. All policies are outcome-independent. Selection uses explicit lexicographic objectives rather than arbitrary utility weights, and zero-cost ties preserve chronological `oldest_first` order.

The source allocation remained canceled, so this is not new GPU evidence and was not promoted to durable AML storage.
