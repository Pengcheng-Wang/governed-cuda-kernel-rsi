# Nonstationary weighted-fair admission

This directory contains a local measured-service simulation that tests whether
the E53 cumulative fairness controller carries stale state across abrupt
prompt-mix changes.

The workload has four exact 500-request phases:

1. 80% short, 15% medium, 5% long prompts.
2. 15% short, 20% medium, 65% long prompts.
3. 15% short, 70% medium, 15% long prompts.
4. 80% short, 15% medium, 5% long prompts again.

The hard non-escalating 100 ms prefill slice, 8x safety baseline, gain 4,
32-request/85%-acceptance prior, and 0.5x/4x credit bounds are unchanged.
Cumulative feedback is compared with 128/256/512-request rolling histories,
plus static 8x and square-root controls.

Stationary controls run at 0.1/0.2/0.3/0.4 times decode-only capacity.
Nonstationary phases run at 0.3/0.4 times capacity. Both prefix modes, both
measured decode replicates, and three Poisson seeds produce 48 aggregate
records and 288 scenarios.

Every scenario verifies exact accepted/rejected accounting, prefill/decode
tokens and causal pairs, finite completion only for accepted requests, batch
size at most 64, KV within 32 GiB, exact phase class counts, the hard
prefill-slice budget, and the dynamic safety-adjusted admission limit. The
three stationary controls reproduce all 144 matching E53 scenarios with zero
metric drift.

Key results:

- Aggregate fairness can hide phase-local exclusion. At no-cache 0.4x load,
  square-root credit reported 0.956 overall Jain fairness but only 0.841 mean
  phase fairness and 0.660 worst-phase fairness. With prefix reuse, its
  0.990 aggregate Jain score hid a 0.555 worst phase.
- At no-cache 0.3x, cumulative feedback achieved 0.981 mean and 0.954 worst
  phase Jain fairness, 656 output tokens/s, and 721 ms p99 TTFT. It retained
  hysteresis: the short-return dominant-class credit was 0.207 lower than in
  the initial short-heavy phase.
- Rolling windows traded stability for forgetting. At 0.3x, 128/256/512
  requests delivered mean phase Jain 0.903/0.941/0.970, throughput
  724/698/675 tokens/s, p99 TTFT 577/598/642 ms, and short-return credit
  deltas +0.016/-0.018/-0.126.
- At no-cache 0.4x, the 512-request window improved cumulative mean/worst
  phase Jain from 0.942/0.862 to 0.958/0.922, raised throughput from 750 to
  803 tokens/s, and reduced p99 TTFT from 783 to 678 ms.
- With prefix reuse at 0.4x, the 512-request window improved cumulative
  mean/worst phase Jain from 0.981/0.954 to 0.983/0.966, raised throughput
  from 1,047 to 1,094 tokens/s, and reduced p99 TTFT from 1.14 s to 670 ms.
- A 128-request window nearly eliminated return-phase credit hysteresis and
  maximized throughput, but its no-cache 0.4x worst-phase Jain score was only
  0.809. The shortest window was responsive but noisy.

No window is universally best. A request-count history also changes its
effective time horizon with arrival rate and ignores prompt cost. A live
controller should test wall-time or service-cost decay and explicit
tenant-level objectives.
