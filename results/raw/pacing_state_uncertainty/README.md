# Pacing state-estimation uncertainty

This directory contains E64, a deterministic safety analysis of robust pacing
lookups when burst size, offered load, or cache regime is misclassified.

The study starts from E63's robust lookup families and E62's underlying
scenario results. State errors are modeled as independent binary flips under:

- Perfect classification.
- Joint 5%, 10%, and 20% error on burst, load, and cache.
- Isolated 10% burst, load, or cache error.

Two strategies are compared:

1. **Naive:** use the robust E63 window for the estimated key.
2. **Hamming-1 safe:** for each estimated key, select the
   maximum-throughput window that is SLO-feasible for every true state whose
   keyed features differ by at most one bit.

Each record reports expected mean/geometric/worst-state throughput versus the
exact E62 robust oracle, acceptance regret, expected scenario SLO violation
probability, unsafe-path probability, maximum excess, selected-window
expectation, key-error rate, and every true/estimated-state path.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_state_uncertainty.py `
  --root . `
  --output results\raw\pacing_state_uncertainty\pacing_state_uncertainty_full.json
```

The analysis reproduces all 24 E63 robust zero-error family/SLO metrics
exactly. Hamming-1-safe policies have zero violations under every isolated
single-feature error profile; residual joint-error risk comes only from
simultaneous mistakes in two or more keyed features.

Key results:

- At 800 ms with 10% joint error, naive full-state lookup retained an
  apparent 100.4% of oracle throughput but had 3.36% expected scenario
  violation probability and 9.04% unsafe-path probability.
- The full-state Hamming-1 envelope retained 94.9% throughput and reduced
  violations to 0.13% and unsafe paths to 0.58%. Perfect-classification
  retention was 95.0%, the direct safety cost.
- Every isolated 10% burst, load, or cache error had zero envelope
  violations. Residual risk under joint errors came from simultaneous
  multi-feature mistakes.
- At 800 ms and 10% joint error, burst+cache envelope retained 92.2%
  throughput with 0.02% violation probability. Cache-only envelope retained
  90.9% with zero violations.
- Naive full-state isolated-error violations at 800 ms were 1.87% for load,
  1.25% for burst, and 0.63% for cache.
- At 650 ms, naive full-state violation probability rose
  2.71%→5.23%→9.65% for 5/10/20% joint error; the envelope reduced this to
  0.06%→0.24%→0.90%.
- Rare simultaneous errors can retain a large maximum SLO excess even when
  expected violation probability is small. Deterministic guarantees require
  an all-state safe window or rejection/defer behavior.
