# Persistent-fit queue latency simulation

This directory turns the real v35 adaptive/additive depth-8 service
measurements into an end-to-end single-server queue model.

- Arrivals are Poisson and offered load is normalized to the measured scalar
  service capacity.
- A full queue of eight uses the measured vector-controller batch time.
- If the oldest request reaches the configured timeout before eight requests
  are ready, it executes through the measured scalar path.
- Two completion bounds are reported. `streaming` spaces results through the
  measured sequential batch execution; `fenced` releases all eight at batch
  completion.
- Each record aggregates both measured service replicates and three arrival
  seeds over 30,000 requests. Immediate scalar results are validated against
  the M/D/1 mean-response formula using dedicated 200,000-request traces.

The simulation exposes a load-dependent policy:

- At 50% scalar utilization, a 0.5 ms timeout adds roughly 0.5 ms to p99 and
  forms almost no vector batches. Immediate execution is preferable.
- At 75%, streaming completion improves but the conservative fenced p99 is
  still 6-7% worse than immediate execution.
- At 90%, a 0.5 ms timeout improves fenced p99 by 40.6% on 8 GPUs and 33.2%
  on 24 GPUs, even though throughput is still arrival-limited.
- At 110%, adaptive batching improves throughput by about 11.5% and bounds
  fenced p99 at 73.7/90.0 ms instead of 4.0/6.5 seconds on 8/24 GPUs.
- At 120%, the queues are near or beyond effective capacity. Tail latency
  returns to seconds even though throughput remains higher.

These are measured-service simulations, not new GPU executions. They exclude
the rest of the model, scheduler fairness, request-length variation, ingress,
and cancellation.
