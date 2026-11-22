# Chunked prefill and prefix reuse

This directory contains a local measured-service simulation derived from the
existing A100 FlashAttention, cuBLAS, paged-GQA, and CUDA-Graph MLP results.

The 32-layer prefill service model:

- Fits FlashAttention latency to causal attention pairs across the measured
  512/2048/8192 sequence points, then scales work from 16 to 32 query heads.
- Fits cuBLAS launch cost and throughput from 4096-square M=512/4096 GEMMs and
  validates the model on the measured 4096-to-11008 and 11008-to-4096
  projections.
- Charges Q/K/V/O and gate/up/down dense work for each uncached prompt chunk.

The scheduler compares monolithic prefill with 2048/1024/512/256-token chunks.
The 512- and 256-token controls run two or four decode steps per prefill chunk.
The prefix mode gives half the requests a deterministic 75%-cached prompt.
Four loads, six policies, two prefix modes, both measured decode replicates,
and three Poisson seeds produce 48 aggregate records and 288 scenarios.

Every scenario verifies exact uncached prefill and output-token accounting,
exact causal attention-pair conservation across chunks, finite completion,
batch size at most 64, and KV reservations within a 32 GiB budget. The
FlashAttention fit has 4.33% maximum error; the dense validation error is
0.65%.

Key results:

- No-cache monolithic capacity was 7.36 requests/s. 2048/1024/512/256-token
  chunking added 2.5%/9.0%/21.8%/52.2% prefill service.
- At 0.2x decode-only load, 1024-token chunks retained 93.4% of monolithic
  capacity and cut p99 maximum token gap from 348 to 86 ms. P99 TPOT improved
  from 108 to 70 ms, while p99 TTFT worsened from 1.20 to 2.23 seconds.
- A 512-token chunk reduced maximum-gap p99 to 55 ms but raised p99 TTFT to
  4.96 seconds near 0.92 system load.
- A 2:1 decode-priority policy reached 28.7 ms p99 TPOT but exceeded its own
  capacity and grew p99 TTFT to 21 seconds. The 4:1 policy was overloaded at
  only 0.1x decode-only load.
- The mixed prefix mode removed 33.7% of monolithic prefill service and raised
  monolithic capacity by 30.0%, from 7.36 to 9.57 requests/s.

The simulation executes prefill and decode sequentially. It does not model a
fused mixed batch, tensor parallelism, cache lookup/eviction, shared prefix
memory, KV append kernels, CPU scheduling cost, or GPU jitter.
