# Exact batch member-order frontier

This directory contains E83, an outcome-independent member-order follow-up to
E82's streaming exact batches.

Batch formation, launches, total work, worker assignment, and exact alert
membership are fixed. Four orders are compared:

- oldest-first: chronological E82 baseline;
- newest-first: newest checkpoint to oldest;
- oldest-newest: alternate oldest and newest;
- newest-oldest: alternate newest and oldest.

The compact result uses the full E79 path matrix at marginal costs
0/0.25/1/10/50/100%, but persists aggregate, configuration, breach-metric,
worst-profile, and order-versus-chronological summaries rather than every
per-profile record.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_coalescing_order.py `
  --root . `
  --output results\raw\pacing_coalescing_order\pacing_coalescing_order_full.json `
  --paths 4096
```

Key results:

- All 1,152 queue/order profiles assigned every look and preserved identical
  work. Oldest-first reproduced E82 in 288 checks; all four orders were
  identical at zero marginal cost. All 48,384 alternative comparisons
  retained every action, and all 20,736 null comparisons had equal alerts.
- At 0.25% marginal cost, newest-oldest reduced drift regression from
  oldest-first's 0.0339% to 0.0138% and worst-profile regression from 5.13%
  to 0.732%. Maximum positive delay increased from 2 to 9 requests.
- At 1%, newest-first minimized aggregate regression at 0.0151%; newest-oldest
  was close at 0.0162% but limited maximum delay to 64 rather than 225
  requests. Oldest-first reached 0.143% aggregate and 34.0% worst-profile.
- At 10%, newest-oldest was best on aggregate regression (0.0186%) and kept
  worst-profile regression at 0.732%. Its maximum delay was 848 requests,
  versus 2,250 for newest-first and 339 for oldest-first.
- At 50%, newest-first minimized regression at 0.0259%; newest-oldest traded
  a small increase to 0.0282% for a 3,726-request maximum delay instead of
  11,250. Oldest-first reached 0.202% aggregate and 34.0% worst-profile.
- At 100%, newest-first remained the risk-minimizing order at 0.0440%
  aggregate and 0.952% worst-profile, but maximum delay reached 22,500
  requests. Oldest-first had 0.264% aggregate, 34.0% worst-profile, and
  4,999-request maximum delay.
- Certification had zero FIFO regressions for every order through 50%.
  At 100%, oldest-first minimized aggregate regression (0.0070%), while
  newest-first minimized worst-profile regression (1.29% versus 2.91%).

Member order is therefore a frequency/severity control. Newest-first best
protects drift risk, but can create very large delay on the rare paths it
hurts. Newest-oldest is the balanced default across low-to-moderate marginal
cost because it retains most of the risk reduction while bounding the delay
tail more tightly.
