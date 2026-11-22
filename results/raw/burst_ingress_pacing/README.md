# Burst ingress pacing

This directory contains E61, a measured-service simulation of explicit burst
backpressure after E60 showed that pending service commitments cannot repair
burst queueing.

Four- and sixteen-request bursts are spread uniformly across 0%, 10%, 25%,
50%, or 90% of their natural inter-burst period while preserving request
count and mean arrival rate. The equal-share E59 gain-16 actual-service
controller is compared with E60's batch-32 pending-service commitment.

Scheduling uses the paced arrival. End-to-end queue latency, TTFT, E2E, and
throughput elapsed time remain anchored to the original simultaneous client
arrival, so pacing gains cannot hide intentional ingress delay. Server-only
tail latency and accepted/offered ingress-delay distributions are reported
separately.

Reproduce from the repository root:

```powershell
python code\simulate_burst_ingress_pacing.py `
  --root . `
  --output results\raw\burst_ingress_pacing\burst_ingress_pacing_full.json
```

The full run checks exact zero-pacing E60 equivalence, original-arrival
latency accounting, monotonic pacing, and the inherited commitment, token,
causal-pair, batch, KV, observer, slice, and admission invariants. This
remains a fitted-service simulation rather than a production ingress queue.

Key results:

- The full matrix contains 40 aggregate records and 240 scenarios. All 48
  matching zero-pacing E60 scenarios reproduced with zero metric drift.
- For no-cache burst 16, 10% pacing raised acceptance 29.2%→38.4% and
  throughput 382→510 output tokens/s. P99 client TTFT rose 549→622 ms while
  server-only TTFT improved slightly to 545 ms.
- At 25% pacing, throughput reached 686 tokens/s and acceptance 48.2%, with
  772 ms p99 client TTFT and 366 ms p99 ingress delay.
- At 90% pacing, throughput reached 1,144 tokens/s and acceptance 74.6%, but
  p99 ingress delay was 1.32 seconds and client TTFT 1.68 seconds.
- For no-cache burst 4, 10% pacing raised throughput 924→1,016 tokens/s for
  only 37 ms p99 ingress delay and 24 ms more client TTFT.
- Prefix reuse raised the frontier: cached burst-16 10% pacing raised
  throughput 559→739 tokens/s and acceptance 42.4%→56.6%.
- Batch-32 commitments added little after shaping. At no-cache burst 16 and
  25% pacing, actual-only/batch-32 throughput was 686/665 tokens/s and target
  error 3.0/2.9 points.
- Heavy pacing recovers throughput by waiting. Original-arrival latency must
  be retained or ingress delay is misreported as scheduler efficiency.
