# Causal batched-decode attribution

This directory contains a local measured-service audit of the batch-1 decode
clock used by the E57 tenant study.

An optional observer records every batched decode step without changing the
scheduler. The fitted step is exactly additive:

- Each active request receives one fitted batch coefficient.
- Context-token cost is assigned to that request's current context.
- Attention intercept and batch-dependent MLP service are shared equally.

This causal allocation is compared with:

1. E57 standalone batch-1 decode service for each prompt/output class.
2. Simple equal sharing of each observed decode step.

Prompt-fair, gain-4 tenant, gain-8 tenant, and weighted-reservation policies
run at 0.3/0.4 times decode-only capacity under no-cache and tenant-correlated
prefix reuse. Both measured decode replicates and three Poisson seeds produce
16 aggregate records and 96 scenarios.

Every scenario verifies exact accepted/rejected accounting, prefill/decode
tokens and causal pairs, finite completion only for accepted requests, batch
size at most 64, KV within 32 GiB, the hard prefill-slice budget, and the
unchanged admission threshold. Exact and equal allocations each sum to every
decode step within floating-point tolerance, the observer total equals
`decode_busy_ms`, all 96 E57 policy scenarios reproduce with zero metric
drift, and v40 defaults remain byte-identical.

Key results:

- Standalone batch-1 service overcharged causal batched decode by
  12.7–15.1x at no-cache 0.3x, 19.4–21.0x at no-cache 0.4x, and
  20.1–22.0x at cached 0.4x.
- Total prefill-plus-decode service remained overestimated by about
  6.9–11.2x.
- The error was structured. At no-cache 0.4x gain-8 tenant fairness,
  standalone accounting assigned generation-heavy traffic 55.9% of accepted
  total service versus 36.2% under causal attribution. Total-share L1 error
  was 20.3 percentage points.
- Under the weighted-reservation policy at no-cache 0.4x, standalone
  generation-heavy share was 52.3% versus 34.6% causally; interactive share
  was 31.2% versus 31.7%.
- Equal step sharing was much closer than standalone accounting. Its total
  tenant-share L1 error stayed around 1.7–3.8 percentage points across the
  reported matrix.
- Mean observed batches grew from roughly 16–21 at 0.3x to 27–32 at no-cache
  0.4x and 31–38 with prefix reuse, explaining the larger standalone
  overcharge at higher throughput.
- Interactive, long-context, and generation-heavy standalone/causal decode
  ratios differed because output length and completion time changed the batch
  occupancy each tenant experienced.

Batch-1 service remains a monotone offered-work proxy, but it is not causal
GPU accounting. The attribution is exact only for the fitted affine attention
model plus batch-dependent MLP term; live kernels require direct step
instrumentation.
