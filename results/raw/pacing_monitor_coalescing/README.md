# Latest-pending monitor coalescing

This directory contains E78, a freshness-oriented follow-up to E77's
asynchronous FIFO monitor-worker model.

Running jobs finish nonpreemptively. When all workers are busy, each new
checkpoint replaces the older queued checkpoint, so the pending queue contains
at most the newest look. Selection uses only look arrival and worker state;
skipped looks leave their alpha unspent.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_monitor_coalescing.py `
  --root . `
  --output results\raw\pacing_monitor_coalescing\pacing_monitor_coalescing_full.json
```

Key results:

- The confidence-threshold gate passed 558,138 certification and 183,330
  drift metric-grid checks under E68's fixed expected empirical rates.
- Every exact-scan per-schedule action was no later than FIFO for one, two,
  four, and eight workers at each integer service cost from 0 through 500.
  All 7,631,232 per-case oracle comparisons were equal or faster.
- At service 500, one-worker drift median/p90 action delay fell from E77's
  2,999/3,928 requests to 546/852; minimax changed from 2.00x to 1.25x and
  the maximum saving was 4,000 requests.
- Certification winner changes across the exact scan numbered 11/7/0/0 for
  one/two/four/eight workers. Drift numbered 53/19/10/1.
- The oscillations were not lexical tie artifacts. Forty-nine of the 53
  one-worker drift transitions were decided by maximum regret and four by
  p90 regret.
- Fine selected cadences commonly dropped 20-30% of jobs near service 500.
  Integer look arrivals, nonpreemptive completion, and dropped-crossing replay
  make cadence selection a discrete lattice policy rather than a monotone
  service-cost frontier.

The validity claim is limited to outcome-independent coalescing and the fixed
expected-rate threshold gate. Variable service, preemption, fluctuating
empirical estimates, and data-dependent look selection require separate
analysis.
