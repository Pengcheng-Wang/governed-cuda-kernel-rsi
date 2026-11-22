# Confidence-detector operating frontier

This directory contains E66, a deterministic operating-surface analysis for
the confidence-set pacing controller from E65.

The study independently sweeps:

- Mistake-flag recall from 50% to 100%.
- Correct-feature false-alarm rate from 0% to 100%.
- Joint burst/load/cache estimator error at 5%, 10%, and 20%.
- 650, 800, and 1000 ms client-TTFT SLOs.
- All eight E63 policy families.

For each of 72 SLO/family/error surfaces, 99 confidence operating points are
evaluated. The maximum-throughput point is selected for expected scenario
SLO-violation budgets of 0%, 0.05%, 0.1%, 0.25%, 0.5%, 1%, 2%, and 5%.
Selections are reported three ways:

1. **Unconstrained surface:** an idealized upper bound that may not lie on a
   real detector ROC curve.
2. **Fixed false-alarm slices:** the recall requirement and throughput at a
   specified false-alarm rate.
3. **Deployable named profiles:** E65's 95%/5%, 95%/20%, 80%/5%, 60%/20%,
   and always-flag choices; oracle confidence is excluded from deployment
   selection.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_confidence_frontier.py `
  --root . `
  --output results\raw\pacing_confidence_frontier\pacing_confidence_frontier_full.json
```

The full analysis contains 7,128 grid points and reproduces all 432 matching
E65 profiles exactly.

Key results:

- For the full policy at 800 ms, 10% joint state error, and a 5%
  false-alarm rate, a 0.1% violation budget required at least 97.5% mistake
  recall and retained 99.01% oracle throughput. A 0.25% budget required 95%
  recall and retained 99.04%; a 0.5% budget required 90% recall and retained
  99.09%.
- The SLO changes the detector requirement. At 10% error, 5% false alarms,
  and a 0.1% risk budget, the 650/800/1000 ms contracts required
  99%/97.5%/92.5% recall, respectively.
- More state error requires more recall. At 800 ms, 5% false alarms, and a
  0.25% risk budget, 5%/10%/20% joint estimator error required
  90%/95%/97.5% recall.
- False alarms spend throughput. Under zero expected violations at 800 ms,
  10% state error, throughput fell from 99.32% at 0% false alarms to 98.99%
  at 5% and 97.90% at 20%.
- Higher false alarms can slightly reduce the recall needed for a risk
  budget, but not enough to recover their throughput tax. At the 0.25%
  budget, 20% false alarms allowed 92.5% recall versus 95% at 5% false
  alarms, yet retained only 97.98% rather than 99.04% throughput.
- Named profiles expose a policy cliff. At 800 ms/10% error, the full policy
  needed always-flag universal control for a 0.1% budget, retained 90.9%
  throughput. Relaxing to 0.25% enabled the 95%/5% profile and jumped to
  99.0%; a 1% budget enabled the 80%/5% profile at 99.2%.
- Compact policies crossed the same budget sooner. At 800 ms/10% error and a
  0.1% budget, cache-only and burst+cache could use the 95%/5% profile at
  95.7% and 97.2% throughput, while the full policy still required
  always-flag.

The primary systems lesson is that confidence calibration is an SLO control,
not only a model-quality metric. Required mistake recall depends jointly on
the service SLO, state-error rate, policy dimensionality, and tolerated risk.
A deployment should select on a measured ROC curve, not the unconstrained
surface, and should fall back when online recall confidence no longer meets
the chosen risk budget.
