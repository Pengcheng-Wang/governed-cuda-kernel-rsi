# Streaming exact batched look replay

This directory contains E82, a partial-result follow-up to E81's exact
batched replay.

The queue and total service work are identical to E81. Pending declared looks
are merged into one batch, but member results are emitted chronologically as
each incremental snapshot finishes rather than waiting for the batch fence.
Every declared look and alert decision remains exact.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_coalescing_streaming.py `
  --root . `
  --output results\raw\pacing_coalescing_streaming\pacing_coalescing_streaming_full.json `
  --paths 4096
```

Key results:

- All 768 queue profiles assigned every look, preserved E81 launch/work
  accounting, and completed every member no later than its batch fence.
  All 32,256 alternative comparisons retained every eventual FIFO action and
  were no later than E81; all 13,824 null comparisons matched FIFO alerts.
- Zero marginal cost reproduced E81 exactly. Streaming became strictly better
  once marginal work was positive.
- FIFO regressions remained zero through 0.2% marginal cost. At that point,
  streaming saved 13.84% of launches and 20.88% of modeled service work,
  accelerated 22.22% of drift actions, and improved 2.31% of actions versus
  the E81 batch fence.
- The first observed regression appeared at 0.25% marginal cost: 0.0339%
  aggregate and 5.13% worst-profile, with weighted mean/max positive delay
  only 1.05/2 requests.
- A 0.4% marginal cost remained below a 0.1% aggregate regression budget:
  0.0734% aggregate, 15.06% worst-profile, and 1.84/7-request mean/max delay.
- Streaming dominated the batch fence at every marginal cost. At 50%, it
  accelerated 7.46% of actions versus E81, reduced aggregate drift regression
  from 0.286% to 0.202%, and reduced maximum positive delay from 11,250 to
  2,992 requests.
- At 100%, total work equaled FIFO. Streaming still improved 10.77% of actions
  versus the fence, reducing drift regression from 8.21% to 0.264% and maximum
  positive delay from 22,500 to 4,999 requests.

Chronological partial release is therefore a strict improvement over a batch
fence under the same work model. It preserves exact monitoring semantics and
substantially widens the usable marginal-cost region, but positive incremental
work can still lose FIFO timing on some paths.
