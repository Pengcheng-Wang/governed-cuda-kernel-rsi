# Service-clock calibration robustness

This directory contains a local measured-service simulation that isolates
fairness-clock error from admission and execution.

The E55 gain-4, 40-second predicted-service decay controller is perturbed by:

1. Uniform 0.75x and 1.25x service clocks.
2. 0.75x and 1.25x clock cost only for 6144-token prompts.
3. A cache-blind clock that charges no-cache prefill service even when prefix
   reuse reduces actual candidate work.

Admission backlog prediction remains correct in every policy. A 512-request
controller and exact service-40 controller are retained as controls.

The rotating short-heavy/long-heavy/medium-heavy/short-return workload runs at
0.3/0.4 times decode-only capacity. Seven policies, both prefix modes, both
measured decode replicates, and three Poisson seeds produce 28 aggregate
records and 168 scenarios.

Every scenario verifies exact accepted/rejected accounting, prefill/decode
tokens and causal pairs, finite completion only for accepted requests, batch
size at most 64, KV within 32 GiB, exact phase class counts, monotonic history
ordering under global clock scale, the hard prefill-slice budget, and the
unchanged admission threshold. The two controls reproduce all 48 matching E55
scenarios with zero metric drift. Cache-blind no-cache execution reproduces
exact service-40 across 12 additional scenarios.

Key results:

- Uniform scale error mostly retuned the half-life. At no-cache 0.4x, 0.75x
  service retained 402.6 effective requests and raised worst-phase Jain
  fairness from 0.927 to 0.944, but reduced throughput from 799 to
  767 tokens/s. A 1.25x clock retained 285.9 requests, lowered fairness to
  0.911, and reduced p99 TTFT from 687 to 646 ms.
- Long-only ±25% error changed the mean clock to only 0.936x/1.064x.
  At no-cache 0.4x, undercounting long service raised worst-phase Jain to
  0.937 and delivered 783 tokens/s; overcounting reached 0.924 and
  801 tokens/s.
- At no-cache 0.3x, uniform undercounting improved worst-phase Jain from
  0.964 to 0.971 with nearly unchanged throughput, while uniform
  overcounting lowered it to 0.951.
- Cache blindness was the largest structured error. At cached 0.4x it charged
  1.973x mean and 2.394x long-prompt service, reducing effective history from
  440.2 to 333.1 requests and worst-phase Jain from 0.979 to 0.971.
- Cached throughput and TTFT changed little under cache blindness
  (1,073→1,076 tokens/s and 859→841 ms) because admission itself continued
  using the correct cached cost.

A global service scale can be absorbed by half-life tuning. Class-correlated
error cannot: missed prefix reuse or tenant-specific cost bias silently gives
different groups different memory horizons even when aggregate calibration is
good.
