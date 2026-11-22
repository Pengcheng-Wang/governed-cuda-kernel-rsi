# Stall-capped prefill admission

This directory contains a local measured-service simulation that separates
the two mechanisms mixed by the v41 adaptive controller:

1. A hard non-escalating 100 ms predicted prefill-slice budget.
2. Optional admission when queued prefill GPU service plus the arriving
   request exceeds two or four seconds.

Fixed-1024 and adaptive-100 controls reproduce v41 exactly. No-cache and
mixed-prefix workloads run at 0.1/0.2/0.3/0.4 times decode-only capacity.
Five policies, both measured decode replicates, and three Poisson seeds
produce 40 aggregate records and 240 scenarios.

Every scenario verifies exact accepted/rejected classification, exact accepted
prefill/decode tokens and causal pairs, finite completion only for accepted
requests, batch size at most 64, KV within 32 GiB, and zero hard-slice budget
violations. All 96 matching v41 scenario comparisons have zero metric drift.

Key results:

- Every hard-policy scenario held p99 maximum token gap below 120 ms.
- At no-cache 0.3x decode-only load, hard slicing without admission reached
  41.8-second p99 TTFT. The two-second guard accepted 94.5% of requests,
  reduced p99 TTFT to 12.3 seconds, and raised completed output throughput
  from 861 to 950 tokens/s.
- Accepted predicted prefill backlog p99 stayed at 1.98 seconds, but actual
  p99 TTFT was 6.2x larger because decode work and future arrivals were not
  modeled by the admission predictor.
- At no-cache 0.4x load, the two-second guard accepted 85.9%, reduced p99
  TTFT from 99.1 to 15.2 seconds, and held maximum-gap p99 at 86 ms.
- Rejected prompts were 3.11x longer than accepted prompts at 0.3x and 3.46x
  longer at 0.4x. Output lengths were similar, so admission selected prefill
  cost.
- Prefix reuse raised 0.4x acceptance to 94.8% but retained prompt-length
  bias.

The predictor sums queued prefill GPU service only. Its two/four-second
thresholds are controls rather than end-to-end TTFT guarantees; decode
occupancy, future arrivals, CPU overhead, and tenant fairness are excluded.
