# Checkpoint cadence robustness

This directory contains E75, a schedule-selection audit after E74 fixed equal
metric alpha as the only robust allocation.

E70 compared one recursive-ceil checkpoint sequence for each cadence. E75
adds direct logarithmic schedules over eight phases and
ceil/nearest/floor integer rounding. For each cadence, robust cost is the
maximum request count over every unique valid grid.

The four cadences are:

- 1.10x: 25 unique grids.
- 1.25x: 25 unique grids.
- 1.50x: 25 unique grids.
- 2.00x: 22 unique grids.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_cadence_robustness.py `
  --root . `
  --output results\raw\pacing_cadence_robustness\pacing_cadence_robustness_full.json
```

Key results:

- Every original E70 grid reproduced exactly: 7,672 finite certification and
  7,560 finite drift crossings.
- Nominal E70 winners were dominated by 1.50x certification and 2.00x drift.
  Under worst-over-grid cost, 1.10x became the unique winner for
  1,773/1,918 certification cases and 1,663/1,890 drift cases. 1.25x won the
  remainder except for 11 drift ties.
- The nominal winner survived robust selection in only 5.53% of
  certification cases and 9.21% of drift cases.
- 1.10x was the global robust minimax cadence for both domains. Relative to
  the per-case robust cadence oracle, its worst regret was only 1.00882x for
  certification and 1.01796x for drift; p90 regret was 1.00000x and
  1.00339x.
- 1.25x was the nearest alternative, with 1.03699x/1.08029x worst
  certification/drift regret. 1.50x reached 1.15114x/1.16058x and 2.00x
  reached 1.41075x/1.44689x.
- Phase/rounding uncertainty tax grew with cadence. Median robust/base cost
  was 1.048x/1.050x for 1.10x certification/drift,
  1.105x/1.105x for 1.25x, 1.253x/1.253x for 1.50x, and
  1.189x/1.415x for 2.00x.
- The robust per-case oracle itself cost a median 1.202x certification and
  1.232x drift versus the nominal exact-grid oracle. Maximum taxes were
  1.447x and 1.455x.
- The reversal was broad across all 650/800/1000 ms SLOs and all policy
  families/breach metrics. Coarse-grid nominal wins were usually lost after
  accounting for checkpoint construction.

The operational conclusion is to use the 1.10x cadence when exact checkpoint
implementation is not frozen. The nominal label savings of coarser schedules
were mostly lattice-dependent. Cadence, phase, rounding, and alpha spending
must be versioned as one monitoring certificate.
