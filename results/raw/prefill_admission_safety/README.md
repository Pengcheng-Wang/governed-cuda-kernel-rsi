# Prefill admission safety and fairness

This directory contains a local measured-service simulation that extends the
v42 hard-stall controller with structural admission margins and an explicit
prompt-length fairness control.

All policies keep the non-escalating 100 ms predicted prefill-slice budget.
The admission controls compare queued prefill GPU service with a nominal
two-second limit:

1. Raw 1x prediction.
2. 2x, 4x, and 8x safety factors.
3. An 8x control whose effective limit grows with
   `sqrt(prompt_tokens / 512)`.

No-cache and mixed-prefix workloads run at 0.1/0.2/0.3/0.4 times decode-only
capacity. Six policies, both measured decode replicates, and three Poisson
seeds produce 48 aggregate records and 288 scenarios.

Every scenario verifies exact accepted/rejected accounting, accepted
prefill/decode tokens and causal pairs, finite completion only for accepted
requests, batch size at most 64, KV within 32 GiB, and zero hard-slice budget
violations. The hard/no-admission and raw-two-second controls reproduce all 96
matching v42 scenarios with zero metric drift.

Key results:

- At no-cache 0.3x load, raw admission accepted 94.5% and reached 12.25-second
  p99 TTFT. A 4x margin accepted 90.5% and cut p99 TTFT to 636 ms.
- An 8x margin reached 293 ms p99 TTFT at 0.3x and 318 ms at 0.4x, but
  accepted no 6144-token prompts. Prompt-class Jain fairness was about 0.666.
- At 0.3x, square-root prompt credit restored 85.8% long-prompt acceptance and
  raised Jain fairness to 0.956, but accepted only 64.6% overall, lowered
  output throughput from 884 to 668 tokens/s, and raised p99 TTFT to 1.03 s.
- At 0.4x, the credit policy overcompensated: long prompts were accepted
  2.541x as often as short prompts while aggregate acceptance fell to 46.6%.
- Prefix reuse softened but did not remove the frontier. Cached 0.4x 8x safety
  accepted 91.0% with 317 ms p99 TTFT, while long/short prompt acceptance
  remained 0.505x.

The safety factors operate on an intentionally incomplete prefill-only
predictor. Decode occupancy, future arrivals, CPU overhead, retry behavior,
and tenant-level contracts are excluded. The square-root credit is a
diagnostic control, not a recommended production fairness function.
