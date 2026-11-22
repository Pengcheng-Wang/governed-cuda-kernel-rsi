# SLO-budgeted absolute ingress pacing

This directory contains E62, a measured-service simulation that converts the
period-relative E61 pacing frontier into production-style absolute delay
budgets.

Four- and sixteen-request bursts run at 0.3x and 0.4x decode-only load with
requested pacing windows of 0, 50, 100, 150, 200, 300, 400, and 600 ms.
Windows are capped at 90% of the current inter-burst period to preserve
monotonic arrival order. End-to-end latency and throughput elapsed time remain
anchored to the original simultaneous client arrival.

For 650, 800, and 1000 ms client-TTFT SLOs, a candidate is feasible only when
the maximum per-scenario p99 TTFT across both measured service replicates and
all three seeds is within the SLO. The feasible window with maximum median
output-token throughput is selected.

Reproduce from the repository root:

```powershell
python code\simulate_slo_ingress_pacing.py `
  --root . `
  --output results\raw\slo_ingress_pacing\slo_ingress_pacing_full.json
```

The full run checks exact zero-window E61 equivalence, robust SLO selection,
window capping, original-arrival latency accounting, and the inherited
commitment, token, causal-pair, batch, KV, observer, slice, and admission
invariants. This remains a finite-grid fitted-service simulation rather than
a production token-bucket optimizer.

Key results:

- The full matrix contains 64 aggregate records and 384 scenarios. All 24
  matching zero-window E61 scenarios reproduced with zero drift.
- At no-cache 0.4x under a 650 ms SLO, burst 4 selected 100 ms and delivered
  1,087 tokens/s; burst 16 selected 50 ms and delivered 443 tokens/s.
- At cached 0.4x under the same SLO, both burst sizes selected 200 ms.
  Burst-16 throughput reached 795 tokens/s, 1.42x unpaced, with 620 ms
  worst-scenario p99 TTFT.
- At no-cache 0.3x burst 16, unpaced maximum p99 was already 635 ms, so the
  650 ms optimizer selected zero delay. At 0.4x it selected 50 ms.
- Under an 800 ms SLO, no-cache/cached burst 16 at 0.4x both selected 300 ms,
  reaching 628/890 tokens/s and 1.64x/1.59x unpaced throughput.
- Four-request traffic saturated the period cap. At no-cache 0.4x, a
  requested 400 ms window became about 330 ms; the 800 and 1000 ms SLOs
  selected the same point.
- Cached burst 16 under a 1000 ms SLO selected 600 ms, delivered
  1,076 tokens/s—1.93x unpaced—and kept maximum p99 at 963 ms.
- Absolute pacing should be selected from burst size, offered load,
  cache/service regime, and the client-visible latency contract rather than
  configured as one universal delay.
