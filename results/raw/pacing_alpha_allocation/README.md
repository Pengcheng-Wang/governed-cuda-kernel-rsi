# Metric-aware alpha allocation

This directory contains E71, a weighted-Bonferroni audit for the
Bernoulli-KL checkpoint monitor from E70.

E70 divides joint alpha equally across three metric families—state error,
mistake recall, and false alarms—and then equally across every keyed
dimension. E71 keeps the same total alpha, checkpoints, and KL bounds but
sweeps 171 positive 5%-grid allocations plus the exact equal split.

For each calibration scenario and checkpoint schedule, E71 selects the
preregistered allocation requiring the fewest labels. It also:

- Reproduces the exact E70 equal-split result.
- Evaluates one global median-optimized allocation per schedule.
- Evaluates a global minimax allocation that minimizes the worst case.
- Applies the balanced-scenario certification allocation to fresh drift
  detection, exposing the cost of de-emphasized metrics.
- Directionally inverts lower/upper Bernoulli-KL drift bounds and reproduces
  all 7,560 finite equal-alpha E70 drift crossings exactly.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_alpha_allocation.py `
  --root . `
  --output results\raw\pacing_alpha_allocation\pacing_alpha_allocation_full.json
```

The full analysis includes 11,520 certification records and 8,640 drift
records across the four E69 schedules.

Key results:

- At 1.10x, scenario-conditioned allocation reduced the median certification
  sample to 90.9% of equal-alpha E70 and never increased it. It selected a
  non-equal split in 1,369 of 1,918 finite cases.
- At 1.25x/1.50x/2.00x, checkpoint quantization made the median unchanged,
  but selected allocations improved 703/321/275 cases. Best-case reductions
  were 20%/33.3%/50%.
- For the 800 ms full-policy 0.1% point at 1.10x, shifting alpha to
  15% state error / 55% recall / 30% false alarms reduced labels from 32,895
  to 29,904.
- At the 0.25% point, a 30%/40%/30% split reduced labels from 7,872 to 7,156.
- The strict 650 ms/0.1% very-high-recall case used 5%/90%/5% at 1.50x and
  fell from 138,254 to 92,169 labels.
- Near the state-error cap, 50%/20%/30% at 1.10x reduced 113,577 to 103,251.
  Near the false-alarm cap, 5%/5%/90% at 1.25x reduced 61,694 to 49,355.
- No single non-equal 1.10x split was globally non-worsening. The
  median-optimized 60%/10%/30% split improved 25.9% of cases but worsened
  8.9% by one checkpoint, up to 10%.
- A near-equal 35%/35%/30% split was globally non-worsening at 1.25x and
  1.50x, improving 4.43% and 0.16% of cases. At 2.00x, a
  80%/15%/5% state-heavy split improved 14.29% without worsening any case.
- Certification-optimized alpha can delay drift. At 1.10x, 20.24% of active
  drift checks became slower; p90 was 1.10x equal-alpha and the worst case
  1.21x. For the highlighted full-policy split, state-error detection slowed
  from 4,441 to 4,886 requests while recall and false-alarm detection were
  unchanged.
- At coarser schedules, fewer drift cases changed but rare delay could be one
  entire checkpoint: worst-case 1.25x/1.50x/2.00x slowdown at the matching
  schedules.

The operational lesson is that alpha allocation is part of the controller
configuration and must be preregistered. Per-scenario oracle allocation
cannot be learned from the same monitoring data without invalidating the
guarantee. Equal allocation remains the robust 1.10x minimax choice, while
schedule-specific global allocations can safely improve selected coarser
checkpoint regimes. Certification gains must be reviewed together with drift
latency for every de-emphasized metric.
