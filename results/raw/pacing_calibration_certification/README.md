# Finite-sample confidence calibration

This directory contains E67, a deterministic sample-size audit for the
confidence operating points selected by E66.

The analysis asks how many labeled requests are needed before a deployment
can certify all three assumptions behind a fixed-false-alarm E66 point:

1. Per-feature state-estimation error is below its 5%, 10%, or 20% cap.
2. Confidence mistake recall is above the E66-required recall.
3. Correct-feature false alarms are below the 5% cap.

One-sided Wilson score bounds are used with a Bonferroni correction across
the three bounds for every keyed feature. The result is a family-wide joint
95% or 99% confidence certificate. Sample sizes are solved on the
conservative expected-count trajectory and therefore represent nominal
monitoring plans, not exact binomial power calculations.

Calibration scenarios cover:

- Balanced margin: 80% of the state-error cap, 99% recall, 50% of the
  false-alarm cap.
- Very high recall: the same margins with 99.5% recall.
- Tight recall: the same margins with 97.5% recall.
- Near error boundary: 95% of the state-error cap.
- Near false-alarm boundary: 90% of the false-alarm cap.
- Perfect recall: 100% recall, retaining the other balanced margins.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_calibration_certification.py `
  --root . `
  --output results\raw\pacing_calibration_certification\pacing_calibration_certification_full.json
```

The full analysis contains 4,320 records across three SLOs, eight policy
families, three state-error caps, five E66 risk budgets, six calibration
scenarios, and two joint-confidence levels.

Key results:

- At 800 ms/10% state error, the full-policy 0.1% violation budget required
  97.5% recall. With actual 99% recall and balanced margins, joint-95%
  certification required 8,731 labeled requests; joint-99% required 12,670.
- Relaxing the budget to 0.25% lowered required recall to 95% and the
  joint-95% sample to 2,393 requests. At 0.5% and 1% budgets, state-error
  certification became the limiter at 1,451 requests.
- The strict 650 ms/0.1% point required 99% recall. Actual 99% recall cannot
  certify a strict lower bound of 99% from finite data; even 99.5% recall
  required 31,915 labeled requests at joint 95% confidence.
- The 1000 ms/0.1% point required only 92.5% recall, so the balanced scenario
  was limited by state-error certification at 1,451 requests.
- Policy dimensionality affected both the required recall and Bonferroni
  correction. At 800 ms/10% error/0.1% risk, cache-only required 1,019
  requests, burst+cache 1,290, and full state 8,731.
- Small calibration margins dominate. For the 800 ms full policy at the
  0.25% budget, near-error-boundary and near-false-alarm-boundary scenarios
  required 23,211 and 13,316 requests, versus 2,393 for balanced margins.
- Increasing joint confidence from 95% to 99% raised every finite
  certification sample by about 45% in the highlighted full-policy cases.
- Higher state-error prevalence produces more labeled mistakes but also
  requires stricter recall. At 800 ms/0.25% risk, balanced 5%/10%/20% error
  profiles required 3,063/2,393/4,366 requests.
- Most non-universal zero-risk points require recall one and are
  uncertifiable from finite data. A few loose-1000-ms burst/load policies
  remain zero-risk at lower recall because their remaining wrong-key paths
  are already SLO-safe.

The operational lesson is to separate policy feasibility from monitoring
certifiability. A detector can have a nominally feasible E66 operating point
yet lack enough labeled mistakes to prove its recall. Strict service
contracts need either large continuously refreshed calibration sets,
time-uniform sequential bounds, or a structural safe fallback while the
certificate is unavailable.

Method reference: Wilson, E. B. (1927), “Probable Inference, the Law of
Succession, and Statistical Inference,” *Journal of the American Statistical
Association*, https://doi.org/10.1080/01621459.1927.10502953.
