# Static decode graph bucketing

This directory contains a local measured-service simulation derived from the
existing A100 FlashInfer paged-GQA and CUDA-Graph MLP results.

The study compares:

1. One ragged dynamic paged-decode group.
2. One global static context bucket with power-of-two batch rounding.
3. Sequential two-, four-, or eight-context static buckets with power-of-two
   batch rounding.
4. A four-context exact-batch ablation that removes batch padding but creates
   many more graph signatures.

Homogeneous and mixed prompt/output workloads run at 0.25/0.5/0.8/1.0 times
ragged closed-loop capacity. Both measured attention process replicates and
three Poisson arrival seeds produce 48 aggregate records and 288 scenarios.
Every scenario verifies exact generated-token accounting, finite completion,
batch size at most 64, upward-only bucket rounding, and KV reservations within
the 32 GiB budget.

The main model grants no graph replay speedup. Measured evidence is reported
separately: the batch-2/context-257 paged kernel improved 2.022x, while
batch-64/context-2048 changed by only 1.001x. An intentionally optimistic
capacity bound grants every static attention group the full tiny-shape saving.

Key results:

- Ragged mixed-workload capacity was 3669 output tokens/s.
- The best static exact-batch four-context policy reached 1696 tokens/s,
  46.2% of ragged capacity. The optimistic graph bound reached 50.4%.
- Power-of-two batch rounding lowered four-context capacity by 1.176x but
  reduced observed graph signatures by 3.38x.
- Eight context buckets reduced context work from 2.22x to 1.83x versus four
  power-of-two buckets, but extra sequential groups lowered capacity by 7.1%.
- Fine homogeneous buckets retained 79.5% of ragged capacity, or 83.2% under
  the optimistic bound.
- At 0.5x ragged mixed-workload load, the best static policy was already
  overloaded and reached 12.96-second p99 decode-stage TTFT; ragged execution
  remained at 23.5 ms.

This is a fixed-context-padding control, not a claim that CUDA Graphs are
incompatible with ragged attention. Production engines can capture paged or
ragged kernels with fixed metadata buffers and avoid this context padding.
