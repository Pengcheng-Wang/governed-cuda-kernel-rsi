# Exhaustive cross-grid alpha robustness

This directory contains E74, the exhaustive version of E73's checkpoint-grid
robustness audit.

E73 retested six selected allocations. E74 evaluates all 172 positive E71
5%-grid allocations across all 22 unique 2.00x checkpoint sequences produced
by eight logarithmic phases and ceil/nearest/floor integer rounding.

For each allocation, E74 reports:

- Certification and drift geometric means across grids.
- Worst certification and drift ratios across all grids and cases.
- Number of grids on which neither domain is worse than equal alpha.
- Cross-grid minimax, Pareto, and explicit drift-cap selections.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_cross_grid_alpha.py `
  --root . `
  --output results\raw\pacing_cross_grid_alpha\pacing_cross_grid_alpha_full.json
```

Key results:

- The phase-0 power-of-two grid reproduced all 172 E72 2.00x certification
  and drift summaries exactly.
- Equal alpha was the only allocation jointly nonworsening on all 22 grids
  and was the unique robust minimax selection.
- 160/172 allocations were jointly nonworsening on no grid. Only 12 passed
  even one grid. The best non-equal survival count was 11/22, reached by
  30%/35%/35% and 35%/35%/30%.
- Under cross-grid maximum drift caps of 1.00x, 1.10x, 1.25x, and 1.50x,
  equal alpha was the only eligible allocation. At 2.00x, 30%/35%/35%
  became drift-eligible but its worst certification ratio was 2.00010x, so
  equal remained the only certification-nonworsening choice.
- Eleven non-equal allocations improved both cross-grid geometric means, but
  every one retained a full-checkpoint tail regression. The best balanced
  average was 55% state / 20% recall / 25% false alarms:
  0.98448x certification and 0.99740x drift geometric means, but
  2.00041x/2.01220x worst certification/drift and zero nonworsening grids.
- The robust Pareto front contains 13 allocations. Every non-equal member
  trades a small mean improvement for an approximately 2x tail regression.
- The exact-grid 35%/35%/30% candidate remained nonworsening on 11/22 grids,
  with cross-grid means 1.00067x/0.99907x and worst ratios
  2.00003x/2.00045x.

The exhaustive conclusion is that equal alpha is not merely the best of
E73's selected controls. It is the only allocation on the full positive 5%
simplex that is robust to the tested checkpoint phase and rounding choices.
Mean improvements are not deployable evidence when a neighboring grid can
double activation or fallback delay.
