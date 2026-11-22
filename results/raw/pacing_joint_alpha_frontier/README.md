# Joint certification/drift alpha frontier

This directory contains E72, a deployable global-allocation audit for the
metric-aware Bernoulli-KL checkpoint monitor.

E71 selected alpha weights to reduce certification labels, then measured the
resulting drift delay. E72 instead evaluates every one of E71's 172 positive
allocations as a single preregistered controller configuration across all
1,918 finite certification cases and 1,890 active drift checks per cadence.
It reports the certification/drift Pareto frontier and selects allocations
under explicit maximum drift-slowdown caps.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_joint_alpha_frontier.py `
  --root . `
  --output results\raw\pacing_joint_alpha_frontier\pacing_joint_alpha_frontier_full.json
```

Key results:

- Equal alpha reproduced all 7,560 finite E70 drift crossings exactly.
- At 1.10x, equal alpha was the only allocation with no certification or
  drift regression and the only allocation below a strict 1.10x maximum
  drift cap. E71's certification-first 60% state / 10% recall / 30% false
  alarm split reduced certification geometric mean by 1.61%, but worsened
  some certification cases by 1.1002x and some drift checks by 1.1058x.
- At 1.25x, 35%/35%/30% was certification-nonworsening and reduced its
  certification geometric mean by 0.99%. Its drift geometric mean rose only
  0.11%, but its worst drift check consumed the full 1.25x cap.
- At 1.50x, the same near-equal split had slightly better certification and
  drift geometric means, but one integer-rounded drift checkpoint reached
  1.5002x. Equal alpha therefore remained the strict 1.50x-cap selection.
- At 2.00x, 35%/35%/30% became jointly nonworsening: certification geometric
  mean fell 2.00%, drift geometric mean fell 0.04%, and neither domain had a
  ratio above one. It improved 2.92% of certification cases and 0.053% of
  drift checks without slowing any.
- Four 2.00x allocations were jointly nonworsening. Two changed only drift;
  35%/35%/30% was the only one that also reduced certification cost.
- E71's 80%/15%/5% 2.00x certification optimum reduced certification
  geometric mean 9.43%, but increased drift geometric mean 14.16% and could
  delay a drift signal by a full 2.00x checkpoint.
- A symmetric geometric-mean objective still selected allocations with large
  tail regressions. Robust deployment needs an explicit worst-drift cap, not
  only an average joint score.

The operational rule is schedule-specific. Equal alpha remains the robust
global default at fine and medium checkpoint cadences. Coarse 2.00x
checkpoint quantization creates a real near-equal allocation that is
nonworsening in both domains, but it should be preregistered and revalidated
if traffic weighting, checkpoint cadence, or detector assumptions change.
