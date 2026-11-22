# Causal tenant-service fairness

This directory contains E59, a measured-service simulation that closes the
tenant admission loop with causal GPU-service accounting.

The simulator reuses the A100 paged-GQA, CUDA-Graph MLP, FlashAttention
prefill, and cuBLAS measurements used by E46-E58. Prefill slices are charged
directly to their request. Each batched decode step is allocated with the
exact additive E58 fitted model: per-request batch and context terms plus
equal shares of the attention intercept and batch-dependent MLP term.

The admission ledger decays only when observed GPU service executes, using a
40,000 ms service half-life and a 4,000 ms target-shaped prior. Equal and
weighted 1.0/0.85/0.70 service-share targets are tested at feedback gains
4, 8, and 16. The published E57 equal gain-8 and weighted controls are rerun
with the same observers so proxy-clock and causal-service control can be
compared on identical workloads.

Reproduce from the repository root:

```powershell
python code\simulate_causal_service_fairness.py `
  --root . `
  --output results\raw\causal_service_fairness\causal_service_fairness_full.json
```

The full run checks exact E57 control equivalence, hard prefill-slice and
admission bounds, KV/batch/token accounting, observer coverage of prefill and
decode busy time, exact per-step decode allocation, deterministic tenant
composition, and causal-ledger accounting. This remains a fitted-service
simulation rather than a live production-serving or billing measurement.

Key results:

- The full matrix contains 32 aggregate records and 192 scenarios. All 48
  selected E57 control comparisons reproduced with zero metric drift.
- At no-cache 0.4x load, the E57 equal gain-8 proxy control had 0.990
  tenant-acceptance Jain but only 0.969 causal-service Jain, with
  interactive/long/generation service shares 25.2%/38.4%/36.2%.
- Equal causal gains 4/8/16 reached service Jain 0.948/0.984/0.995 and
  target L1 error 10.0/5.3/2.9 percentage points. Gain 16 rebalanced shares
  to 33.1%/30.5%/36.2%.
- Gain 16 delivered 1,030 output tokens/s and 478 ms p99 TTFT. It retained
  98.6% of the E57 proxy-control throughput while improving acceptance and
  TTFT, but lost throughput versus weaker causal gain 4.
- Weighted gains 4/8/16 reached reservation-normalized causal Jain
  0.917/0.966/0.989 with target L1 error 12.1/7.8/4.4 points at no-cache
  0.4x. Only gain 16 clearly improved the actual reservation contract over
  the proxy control.
- With tenant-correlated prefix reuse, weighted gain 16 reached 0.969
  normalized Jain and 7.7-point target error. Causal accounting still needs
  sufficient feedback authority to overcome correlated work and cache mix.
