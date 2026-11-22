# Admission-time causal service commitments

This directory contains E60, a measured-service simulation of feedback lag
inside a serving scheduler's admission batches.

E59 updates tenant fairness only after GPU service executes. If several
arrivals are admitted at the same scheduler timestamp, every decision sees
the same actual-service ledger. E60 preserves E59 execution and compares:

1. Actual observed service only.
2. Pending prefill service predicted by the 100 ms hard-slice model.
3. Pending prefill plus decode service assuming batch occupancy 32.
4. Pending prefill plus deliberately pessimistic batch-1 decode service.

Each accepted commitment is added before the next admission decision at the
same timestamp. Pending prefill and decode are removed as their predicted
steps execute; prediction residuals reconcile when each stage completes.
Actual prefill and exact fitted-model batched decode still update the
40,000 ms service-decayed ledger.

Poisson arrivals are compared with deterministic groups of 4 and 16 requests
at the same timestamp while preserving mean request rate. Equal tenant shares
and weighted 1.0/0.85/0.70 reservations use the E59 gain-16 controller.

Reproduce from the repository root:

```powershell
python code\simulate_causal_service_commitments.py `
  --root . `
  --output results\raw\causal_service_commitments\causal_service_commitments_full.json
```

The full run checks exact E59 actual-only equivalence, commitment drainage,
prefill prediction reconciliation, fixed-batch decode commitment accounting,
hard prefill-slice and admission bounds, observer coverage, and the inherited
token, causal-pair, batch, and KV invariants. It reports both aggregate
service-share fairness and the actual target error of work accepted together
inside each multi-request admission batch. This remains a fitted-service
simulation rather than a production serving or billing measurement.

Key results:

- The full matrix contains 48 aggregate records and 288 scenarios. All 24
  matching E59 actual-only Poisson scenarios reproduced with zero drift.
- Actual-only feedback left 100% of consecutive decisions at one scheduler
  timestamp on an unchanged ledger. Batch-32 commitments reduced this to
  36.1% for no-cache Poisson arrivals and 70.2% for 16-request bursts.
- The local outcome moved much less: no-cache burst-16 mean target error for
  actual service accepted in one multi-request batch improved only
  44.8→43.4 points with batch-32 commitments.
- Batch-32 predicted 0.95x actual no-cache Poisson service with 1.2-point
  tenant-share error, but only 0.35x actual service under burst 16. A static
  occupancy assumption did not transfer across arrival regimes.
- Batch-1 commitments reduced burst-16 aggregate target error 2.8→1.9 points
  and local batch error 44.8→36.2 points, but reserved 3.3x actual service
  and raised p99 TTFT 549→723 ms.
- Under no-cache Poisson traffic, batch-1 reserved 7.2x actual service,
  reduced throughput 1,030→844 output tokens/s, raised TTFT 478→806 ms, and
  worsened target error 2.9→9.4 points.
- At identical mean no-cache load, actual-only throughput fell
  1,030→924→382 output tokens/s for Poisson, burst-4, and burst-16 arrivals;
  acceptance fell 69.4%→62.7%→29.2%.
- Prefix reuse raised throughput but preserved the conclusion: commitments
  address feedback lag, while burst queueing requires explicit backpressure.
