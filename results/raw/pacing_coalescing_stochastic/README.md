# Stochastic monitor-coalescing audit

This directory contains E79, a realized-path follow-up to E78's
expected-rate latest-pending monitor model.

The simulation samples cumulative Bernoulli sufficient statistics at the
original recursive E70 checkpoints:

- certification jointly samples state errors, conditional mistake recall,
  and conditional false alarms;
- drift samples random eligible labels and post-change metric outcomes;
- FIFO evaluates every checkpoint;
- latest-pending coalescing evaluates only its deterministic,
  outcome-independent subset.

The preregistered matrix uses 24 stratified certification profiles, 18 drift
profiles, 4,096 paths per profile and cadence, all four geometric cadences,
one/two/four/eight workers, and service costs 10/100/500.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_coalescing_stochastic.py `
  --root . `
  --output results\raw\pacing_coalescing_stochastic\pacing_coalescing_stochastic_full.json `
  --paths 4096
```

Key results:

- The vectorized direct-KL crossing predicate matched the scalar bound
  inversion in 28,224 checks. All 48 queue profiles processed the final look
  with exact accounting, and all 864 null coalescing policies were pathwise
  subsets of FIFO alerts.
- Certification had zero observed regressions across 4,718,592 path/config
  comparisons and accelerated 7.05%.
- Drift accelerated 22.25% of 3,538,944 comparisons, but 1,160 paths
  (0.0328%) were slower because a skipped crossing reversed before the next
  processed look. No eventual action was missed.
- Regression was concentrated in false-alarm drift (0.0633%) and state-error
  drift (0.0347%); mistake-recall drift was 0.00034%.
- The worst aggregate configuration was eight workers, service 500, cadence
  1.50x: 4.30% faster and 0.137% slower. Only 95.38% of dropped first
  crossings persisted at the next processed look.
- The worst individual profile was the low-margin 95% false-alarm drift
  case with one worker, service 100, cadence 2.00x. It regressed on 0.830%
  of paths; positive delay had median 1,947 and p90/maximum 3,995 requests.
- The largest simulated FIFO false-alert rate was 0.439% against a 1.667%
  stream-alpha budget. Coalescing never created an alert absent from FIFO.

The operational lesson is to distinguish expected-rate dominance from
realized-path dominance. Latest-pending coalescing preserves optional-stopping
validity because look selection is outcome-independent, and it usually
accelerates action, but empirical reversals create a small drift-delay tail.
Deployments needing strict pathwise latency guarantees require a retained-look
or confirmation policy rather than the single latest-pending queue alone.
