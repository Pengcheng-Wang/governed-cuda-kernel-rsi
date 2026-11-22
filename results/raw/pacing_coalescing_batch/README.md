# Exact batched look replay

This directory contains E81, an exact-statistics alternative to dropping
monitoring looks.

When all workers are busy, every pending declared checkpoint is merged into
one nonpreemptive monitor job. The batch preserves every sufficient-statistic
snapshot and completes all member looks together. A one-look batch costs the
same as FIFO; each additional look costs a configurable fraction of a
standalone check.

The canonical matrix uses marginal fractions
0/0.01/0.05/0.1/0.2/0.5/1/2/5/10/25/50/100%, the same 24 certification and
18 drift profiles, 4,096 paths, four cadences, one/two/four/eight workers, and
service costs 10/100/500 used by E79-E80.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_coalescing_batch.py `
  --root . `
  --output results\raw\pacing_coalescing_batch\pacing_coalescing_batch_full.json `
  --paths 4096
```

Key results:

- All 624 queue profiles assigned every look exactly once and processed the
  final look. All 26,208 alternative comparisons retained every eventual
  action, and all 11,232 null comparisons had exactly the same alert
  membership as FIFO.
- With zero marginal cost, the batch schedule reproduced E78's 48 launch
  completions while evaluating every look. It saved 13.73% of launches and
  20.86% of modeled service work, accelerated 22.27% of drift actions, and
  produced zero certification or drift regressions.
- Any positive marginal cost lost strict no-later timing. At 0.01% marginal
  cost, aggregate drift regression was 0.0361%, but weighted mean positive
  delay was only 1.16 requests and the maximum was 3.
- A 0.2% marginal cost was the largest tested value below a 0.1% aggregate
  drift-regression budget. It accelerated 22.19%, regressed 0.0985%, and had
  weighted mean/max positive delay 6.84/45 requests. Its worst profile
  regressed 10.42%.
- Worst-profile risk rose much faster than aggregate risk. A 0.1% marginal
  cost already reached 6.67% worst-profile regression; only zero marginal
  cost met 0.1% and 1% worst-profile caps.
- Certification had zero observed regressions through 50% marginal cost.
  At 100%, total service work equaled FIFO but batch completion delayed 2.59%
  of certification and 8.21% of drift comparisons.
- At 50% marginal cost, batching still saved 16.76% of launches and 12.31%
  of modeled service work, but drift regression reached 0.286% and maximum
  positive delay reached 11,250 requests.

Exact batching is statistically stronger than dropping: it preserves every
declared look and every alert decision. It is also latency-dominant when extra
snapshots are free. With positive marginal work, it becomes a batch-service
frontier: small costs trade a modest frequency of tiny delays for launch/work
savings, while large costs create severe batch-fence latency.
