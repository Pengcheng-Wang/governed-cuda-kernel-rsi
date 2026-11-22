# Robust pacing-policy compression

This directory contains E63, a deterministic analysis of how much state a
deployable pacing policy needs.

The E62 robust oracle has eight states:

- Prefix mode: no cache or tenant-correlated prefix reuse.
- Decode-only load: 0.3x or 0.4x.
- Burst size: 4 or 16 requests.

Candidate policy families share one pacing window universally or key it by
burst, load, cache, burst+load, burst+cache, load+cache, or the full state.
For each 650/800/1000 ms client-TTFT SLO:

1. **Robust selection** requires maximum p99 TTFT across all six E62
   scenarios in every policy key to satisfy the SLO.
2. **Median selection** tunes only against median p99, then exposes hidden
   violations across the same six scenarios.

Each policy reports key count, mean/geometric/worst throughput retention
versus the exact robust full-state oracle, acceptance regret, scenario and
state SLO violations, maximum excess, selected windows, and per-state
outcomes.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_policy_compression.py `
  --root . `
  --output results\raw\pacing_policy_compression\pacing_policy_compression_full.json
```

The analysis reproduces all 24 E62 robust oracle windows exactly. It inherits
E62's finite window grid and synthetic state matrix; policy keys are exact
categories rather than noisy online estimates.

Key results:

- At 650 ms, universal robust control retained 84.0% mean and 70.3%
  worst-state oracle throughput. Cache-only two-key control reached
  91.3%/78.2%; burst+cache four-key control reached 95.8%/86.2%.
- Load+cache reached a slightly higher 96.3% mean at 650 ms but only 78.2%
  worst-state retention, exposing the mean-versus-tail feature tradeoff.
- At 800 ms, burst+load retained 98.5% mean/88.3% worst-state throughput;
  burst+cache retained 98.0%/90.4%. Universal control retained only
  90.9%/81.6%.
- At 1000 ms, cache-only two-key control retained 99.8% mean and 98.9%
  worst-state oracle throughput. Burst+cache reproduced the oracle exactly.
- Median full-state tuning at 800 ms appeared to reach 105.0% of the robust
  oracle but violated 16.7% of scenarios across five states, with up to
  797 ms SLO excess.
- Median burst+cache tuning exceeded the 650 ms SLO by up to 397 ms; median
  full-state tuning exceeded the 1000 ms SLO by up to 597 ms.
- More state does not make median tuning safe. Robust maximum-p99 selection,
  not lookup-table size, provides the SLO guarantee.
