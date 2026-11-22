# Adaptive prefill time slicing

This directory contains a local measured-service simulation that extends the
v40 prefill/decode scheduler.

Fixed monolithic, 2048-, 1024-, and 512-token controls are reproduced from
v40. Two adaptive policies choose the largest candidate chunk whose predicted
32-layer prefill service fits a 50 or 100 ms budget while decode work is
ready. The budget increases as the oldest prefill request waits:

- 50 ms base, then 100 ms after 0.5 s and 150 ms after 2 s.
- 100 ms base, then 150 ms after 1 s and 300 ms after 4 s.

When no decode work is ready, the remaining prompt runs monolithically.
No-cache and mixed-prefix workloads run at 0.1/0.2/0.3/0.4 times decode-only
capacity. Six policies, both measured decode replicates, and three Poisson
seeds produce 48 aggregate records and 288 scenarios.

All 192 matching fixed-policy scenario comparisons reproduce v40 exactly.
Every adaptive scenario inherits exact prefill/decode token, causal-pair,
batch, and KV gates from v40 and additionally verifies valid chunk selection
and zero service-budget violations.

Key results:

- The adaptive 100 ms controller retained 98.9% of no-cache monolithic
  request capacity, versus 93.4% for fixed 1024 and 80.5% for fixed 512.
- At no-cache 0.2x decode-only load, adaptive 100 ms produced 1.72-second p99
  TTFT and 84.6 ms p99 TPOT, between fixed 2048 and fixed 1024.
- Only 5.9% queue-age escalation pushed p99 maximum token gap to 143 ms,
  nearly the fixed-2048 result. Rare long slices dominated the tail.
- With prefix reuse at 0.2x load, adaptive 100 ms slightly improved fixed-1024
  TTFT and TPOT while lowering system load, but 0.26% escalation raised
  maximum-gap p99 from 82 to 136 ms.
- At no-cache 0.3x load, 85.6% of adaptive-100 slices escalated and the policy
  converged toward large chunks to drain the queue.

The controller uses exact queue age and fitted kernel service. It excludes
prediction noise, CPU overhead, tenants, priorities, and admission. Its
hand-designed thresholds are a systems control, not a production-tuned policy.
