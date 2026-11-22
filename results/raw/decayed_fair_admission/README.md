# Time- and service-decayed fairness state

This directory contains a local measured-service simulation that replaces the
E54 hard request-count history with exponential fairness-state decay.

Observed per-class offered and accepted counts decay before each admission
decision with a 20- or 40-second half-life. Two controller clocks are tested:

1. Scheduler wall time between arrivals.
2. Predicted candidate prefill GPU service offered by each arrival.

Gain 4, the hard non-escalating 100 ms prefill slice, 8x safety baseline,
32-request/85%-acceptance prior, and 0.5x/4x credit bounds are unchanged.
Cumulative and 512-request controls reproduce E54.

Stationary cumulative controls run at 0.1/0.2/0.3/0.4 times decode-only
capacity. The rotating short-heavy/long-heavy/medium-heavy/short-return phases
run at 0.3/0.4 times capacity. Both prefix modes, both measured decode
replicates, and three Poisson seeds produce 32 aggregate records and
192 scenarios.

Every scenario verifies exact accepted/rejected accounting, prefill/decode
tokens and causal pairs, finite completion only for accepted requests, batch
size at most 64, KV within 32 GiB, exact phase class counts, nonnegative
decayed state with accepted weight no greater than offered weight, the hard
prefill-slice budget, and the dynamic safety-adjusted admission limit. The
published cumulative stationary and cumulative/512 rotating controls reproduce
all 96 matching E54 scenarios with zero metric drift.

Key results:

- Service-clock memory was load-invariant for the same no-cache prompt
  sequence. The 20/40-second policies retained 204.0/333.3 effective requests
  at both 0.3x and 0.4x load. Wall-time histories grew with arrival rate:
  220.9→282.5 and 386.8→469.2 effective requests.
- At no-cache 0.3x, service-40 improved window-512 mean/worst phase Jain
  fairness from 0.970/0.940 to 0.981/0.964, while throughput changed
  675→678 tokens/s and p99 TTFT remained about 642 ms.
- At no-cache 0.4x, service-20 maximized throughput at 825 tokens/s and reduced
  p99 TTFT to 627 ms, but worst-phase Jain fell to 0.869. Service-40 retained
  0.964/0.927 mean/worst fairness with 799 tokens/s and 687 ms TTFT.
- Wall-40 reached the strongest no-cache 0.4x worst-phase fairness among
  decayed policies, 0.936, but delivered only 761 tokens/s with 726 ms TTFT.
- Prefix reuse reduced candidate service, so service-clock memory expanded in
  request units. At cached 0.4x, service-40 retained 440.2 effective requests
  and achieved 0.989/0.980 mean/worst phase fairness, 1,074 tokens/s, and
  859 ms p99 TTFT.
- At cached 0.4x, service-20 was the lower-latency compromise:
  0.985/0.966 phase fairness, 1,086 tokens/s, and 740 ms TTFT.

Predicted-service decay is more invariant to arrival rate than wall-time or
request-count memory, but it advances on predicted candidate prefill work,
not measured end-to-end GPU occupancy. The half-life remains an explicit
fairness/throughput policy choice.
