# Weighted-fair prefill admission

This directory contains a local measured-service simulation that makes the
E52 prompt-fairness objective explicit instead of assigning one static
prompt-length credit.

All policies keep the hard non-escalating 100 ms prefill-slice controller and
the 8x safety-calibrated admission baseline. The online controller tracks
smoothed acceptance for 512/2048/6144-token prompts. A class below the
equal-class mean receives a bounded exponential credit; an overrepresented
class receives a tighter limit. Gains 2/4/8 are compared with the exact E52
8x static and square-root-credit controls.

No-cache and mixed-prefix workloads run at 0.1/0.2/0.3/0.4 times decode-only
capacity. Five policies, both measured decode replicates, and three Poisson
seeds produce 40 aggregate records and 240 scenarios.

Every scenario verifies exact accepted/rejected accounting, accepted
prefill/decode tokens and causal pairs, finite completion only for accepted
requests, batch size at most 64, KV within 32 GiB, the hard prefill-slice
budget, and the dynamic safety-adjusted admission limit. Both static controls
reproduce all 96 matching E52 scenarios with zero metric drift.

Key results:

- At no-cache 0.3x load, gain 2/4/8 raised prompt-class Jain fairness from
  0.666 under static 8x admission to 0.946/0.979/0.993. P99 TTFT was
  462/510/548 ms and output throughput was 797/761/728 tokens/s.
- The gain-4 controller outperformed square-root credit at that load:
  1.139x output throughput, 0.494x p99 TTFT, and higher Jain fairness
  (0.979 versus 0.956).
- At no-cache 0.4x load, gain 4 accepted 64.6%, delivered 878 tokens/s,
  reached 547 ms p99 TTFT, and achieved 0.969 Jain fairness. Square-root
  credit accepted 46.6%, delivered 634 tokens/s, reached 1.12 s TTFT, and
  achieved 0.892 Jain fairness.
- Prefix reuse improved the frontier. At cached 0.4x load, gain 4 accepted
  83.8%, delivered 1,151 tokens/s, reached 462 ms p99 TTFT, and achieved
  0.991 Jain fairness.
- At no-cache 0.3x, gain 4 assigned mean credits of 0.675/0.884/1.692 to
  512/2048/6144-token prompts. No evaluated policy saturated the configured
  0.5x/4x credit bounds.
- Static square-root credit remained competitive at light load. At no-cache
  0.1x it accepted 94.6% and delivered 348 tokens/s with 0.999 Jain fairness;
  gain 4 accepted 89.9% and delivered 332 tokens/s, but reduced p99 TTFT from
  754 to 445 ms.

The controller optimizes equal prompt-class acceptance, not tenant, token, or
utility-weighted fairness. Cumulative feedback assumes a stationary class
mixture and may adapt too slowly after abrupt workload shifts. Rejected
requests are not retried or routed to another replica.
