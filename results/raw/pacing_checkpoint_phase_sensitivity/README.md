# Checkpoint phase and rounding sensitivity

This directory contains E73, a robustness audit for E72's 2.00x checkpoint
alpha frontier.

E72 found three non-equal allocations that were jointly nonworsening on the
exact power-of-two grid. E73 tests whether that property survives one full
logarithmic phase cycle and integer implementation choices. It evaluates:

- Eight phase offsets from 0/8 through 7/8.
- `ceil`, nearest-integer, and `floor` request rounding.
- The three non-equal E72 jointly nonworsening allocations.
- Equal alpha, the E71 certification optimum, and the E72 joint-geomean
  negative control.

The 24 requested configurations collapse to 22 unique checkpoint sequences.
Every sequence retains look-index alpha spending and is valid only when
preregistered.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_checkpoint_phase_sensitivity.py `
  --root . `
  --output results\raw\pacing_checkpoint_phase_sensitivity\pacing_checkpoint_phase_sensitivity_full.json
```

Key results:

- The phase-0 power-of-two grid reproduced every candidate's E72
  certification and drift summary exactly.
- Equal alpha was jointly nonworsening on all 22 unique grids and remained
  the robust minimax candidate.
- The E72 35% state / 35% recall / 30% false-alarm allocation was jointly
  nonworsening on only 11/22 grids. Across all grids its certification
  geometric mean was 1.00067x equal and drift geometric mean was 0.99907x.
  Its worst certification ratio was 2.00003x and worst drift ratio was
  2.00045x.
- The 30%/35%/35% allocation also passed only 11/22 grids and had a 2.00010x
  worst certification ratio.
- The 30%/40%/30% allocation passed only 4/22 grids. Its worst certification
  and drift ratios were 2.00010x and 2.00045x.
- The E71 80%/15%/5% certification optimum and E72 55%/15%/30% joint-average
  split passed no grid. Their worst drift ratios reached 2.01220x.
- The failure is a checkpoint-lattice effect. A small phase or rounding
  change can move equal and weighted alpha to adjacent 2.00x looks, turning
  a tie or improvement into a full-checkpoint regression.

The operational conclusion is stronger than E72's exact-grid result:
checkpoint positions, rounding, cadence, and alpha weights form one
preregistered certificate. The 35%/35%/30% allocation is not robust to
ordinary grid implementation changes. Equal alpha remains the safe global
default unless the exact checkpoint sequence is frozen and independently
validated.
