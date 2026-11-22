# Anytime-valid confidence monitoring

This directory contains E68, a conservative time-uniform monitoring audit for
the confidence operating points selected by E66 and planned with one-shot
Wilson intervals in E67.

Repeatedly checking a fixed 95% or 99% interval does not preserve its nominal
error rate. E68 assigns each metric-feature stream a Bonferroni share of the
joint error budget and spends it over effective trial count `m` as:

```text
alpha_m = alpha_stream * 6 / (pi^2 * m^2)
radius_m = sqrt(log(1 / alpha_m) / (2 * m))
```

Hoeffding bounds using this summable schedule are valid simultaneously over
all integer monitoring times by a union bound. The construction is
transparent but deliberately conservative; mixture and empirical-Bernstein
confidence sequences can be much tighter.

The audit contains:

- 2,880 anytime-certification records across four E67 calibration scenarios.
- 2,160 fresh post-change drift-detection records.
- Joint 95% and 99% confidence.
- State-error, mistake-recall, and false-alarm breaches.
- Exact links to the matching E67 one-shot Wilson plans.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_anytime_monitoring.py `
  --root . `
  --output results\raw\pacing_anytime_monitoring\pacing_anytime_monitoring_full.json
```

Key results:

- Among 1,918 finite matching certifications, the conservative anytime plan
  required a median 41.65x the E67 one-shot sample, 75.41x at p90, and up to
  282.21x.
- At 800 ms/10% state error, the full-policy 0.1% risk point increased from
  8,731 one-shot labels to 770,825 anytime labels. The 0.25% point increased
  from 2,393 to 91,775; the 0.5% point from 1,451 to 33,135.
- The strict 650 ms/0.1% point with actual 99.5% recall increased from
  31,915 to 8,114,375 labels. The loose 1000 ms point was state-error limited
  at 33,135 rather than recall limited.
- Policy dimensionality still mattered, but the tight recall margin dominated
  the full policy. At 800 ms/10% error/0.1% risk, cache-only, burst+cache, and
  full-state plans required 31,646, 32,586, and 770,825 labels.
- Near the state-error or false-alarm cap, the 800 ms full-policy 0.25% point
  needed 649,150 or 705,598 labels.
- A fresh post-change audit detected a 10%→12.5% state-error breach after
  20,433 requests and a 5%→7.5% false-alarm breach after 22,210 requests at
  joint 95% confidence.
- Detecting a 97.5%→95% recall breach required 255,413 requests because only
  the expected 20,433 state mistakes carried recall labels.
- Naively repeating independent 95% checks 24 times would have a 70.8%
  chance of at least one false alarm; 168 checks raise the illustration to
  99.98%. Even nominal 99% checks reach 81.5% after 168 independent looks.

Drift detection starts a fresh post-change audit. A production system that
mixes historical and post-change samples needs rolling windows, change-point
logic, or time-uniform martingale methods that discount stale evidence.

References:

- Hoeffding, W. (1963), “Probability Inequalities for Sums of Bounded Random
  Variables,” *JASA*, https://doi.org/10.1080/01621459.1963.10500830.
- Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. (2021),
  “Time-uniform, nonparametric, nonasymptotic confidence sequences,”
  *Annals of Statistics*, https://doi.org/10.1214/20-AOS1991.
