# Bounded monitor-coalescing depth frontier

This directory contains E80, a bounded freshest-K queue follow-up to E79.

Running monitor jobs finish nonpreemptively. The pending queue retains the
newest K checkpoints, drops its oldest entry when full, and processes retained
jobs in checkpoint order. Pending depths 1/2/4/8/16/32/64 are replayed on the
same 24 certification and 18 drift profiles, 4,096 paths, four cadences,
one/two/four/eight workers, and service costs 10/100/500 used by E79.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_coalescing_depth.py `
  --root . `
  --output results\raw\pacing_coalescing_depth\pacing_coalescing_depth_full.json `
  --paths 4096
```

Key results:

- Depth 1 exactly reproduced E78 in 48 queue checks. A pending depth equal to
  the number of looks exactly reproduced FIFO in 48 checks. All 336 queue
  profiles processed the final look with exact accounting, and processed work
  was nondecreasing across 288 adjacent-depth checks.
- Certification had zero observed regressions at depths 1/8/16/32/64. Depths
  2 and 4 each produced one near-boundary slowdown in 4,718,592 repeated
  comparisons: 0.000021%, with 21,980 requests of delay.
- Drift regression fell from 0.0328% at depth 1 to 0.0268%/0.0211%/0.0137%/
  0.00746%/0.00314% at depths 2/4/8/16/32.
- The work and benefit costs were substantial. Processed-look fraction rose
  from 86.27% at depth 1 to 94.83% at depth 16 and 98.03% at depth 32, while
  the faster-action fraction fell from 22.25% to 14.47% and 8.63%.
- Depth 16 was the smallest tested policy below a 0.01% aggregate drift
  regression target. It still had 0.708% worst-profile regression.
- Increasing depth did not control severity. At depth 32, the worst profile
  regressed 0.684% and positive delay had median 15,750 and maximum 19,500
  requests.
- Depth 64 processed every look in the tested grids, exactly reducing to FIFO.
  It was the only tested depth with zero aggregate and worst-profile
  regression, and therefore offered no coalescing acceleration.
- Every tested depth that dropped at least one look admits an adversarial path
  where only a dropped look crosses. Strict pathwise FIFO dominance requires
  a no-drop policy; bounded depth only tunes empirical risk.

The operational choice is explicit: depth 1 maximizes savings under a loose
aggregate risk budget; depth 16 meets the tested 0.01% aggregate target; no
bounded dropping depth met a 0.1% worst-profile target. Systems requiring a
strict or worst-profile guarantee must process every look or use a different
policy that changes the reference contract.
