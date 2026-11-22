# Confidence-set pacing lookup

This directory contains E65, a deterministic analysis of confidence-aware
fallback for the robust pacing policies from E63 and E64.

Each keyed burst/load/cache feature has a binary low-confidence flag. A
flagged feature may take either value; an unflagged feature is fixed to its
estimate. For every estimated key and flag mask, the policy selects the
maximum-throughput pacing window that satisfies the client-TTFT SLO for every
state in that confidence set.

The study evaluates joint 5%, 10%, and 20% state-estimation error with:

- Oracle flags: 100% mistake recall and 0% false alarms.
- High-recall clean flags: 95% recall and 5% false alarms.
- High-recall noisy flags: 95% recall and 20% false alarms.
- Moderate clean flags: 80% recall and 5% false alarms.
- Weak noisy flags: 60% recall and 20% false alarms.
- Always flag: the all-state-safe universal fallback.

Each record reports expected mean/geometric/worst-state throughput versus the
E62 oracle, scenario SLO violation probability, unsafe-path probability,
confidence-miss probability, false-alarm probability, expected confidence-set
size, selected-window distribution, and deltas from E64's naive and static
Hamming-1 baselines.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_confidence_sets.py `
  --root . `
  --output results\raw\pacing_confidence_sets\pacing_confidence_sets_full.json
```

The full analysis contains 432 records and reproduces all 144 matching E64
naive/Hamming-1 baselines exactly. Empty flag sets reproduce the E63 robust
mapping, while all-feature flag sets reproduce the universal robust mapping.
Analytic and enumerated confidence probabilities agree within floating-point
tolerance.

Key results:

- At 800 ms with 10% joint error, oracle confidence retained 99.3% mean
  oracle throughput with zero violations. The static Hamming-1 baseline
  retained 94.9% with 0.13% scenario violation probability.
- A 95%-recall/5%-false-alarm full-state signal retained 99.0% throughput,
  reduced the naive violation probability from 3.36% to 0.17%, and improved
  throughput by 4.16 percentage points over static Hamming-1.
- Raising false alarms from 5% to 20% at the same 95% mistake recall reduced
  full-state throughput from 99.0% to 98.0%. It barely changed residual risk:
  missed mistakes, not false alarms, control the safety floor.
- At 800 ms/10% error, the 95%/5% confidence-set policy used 1.48 keys on
  average rather than the static Hamming-1 neighborhood, while its confidence
  miss probability was 1.49%.
- Burst+cache retained 97.2% throughput with 0.04% scenario violation
  probability; cache-only retained 95.7% with 0.03%. Feature-specific
  confidence avoids paying for dimensions that a compact policy does not use.
- At 650 ms, the 95%/5% full policy retained 98.1% throughput versus 89.7%
  for static Hamming-1, with similar residual violation probability
  (0.26% versus 0.24%).
- At 800 ms, full-policy violation probability for the 95%/5% signal scaled
  from 0.09% to 0.17% to 0.33% as joint estimator error rose from 5% to 10%
  to 20%.
- Always flag was deterministically safe but collapsed to universal control:
  only 90.9% mean oracle throughput at 800 ms.

The safety invariant is explicit: scenario violation probability is no
greater than unsafe-path probability, which is no greater than the
probability that at least one mistaken keyed feature was not flagged. A
confidence set cannot protect an unflagged mistake, so confidence calibration
and miss monitoring are part of the serving policy rather than an offline
model-quality detail.
