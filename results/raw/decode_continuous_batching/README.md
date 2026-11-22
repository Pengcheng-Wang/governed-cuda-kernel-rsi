# Heterogeneous continuous decode

This directory contains a local discrete-event simulation driven by the
existing A100 decode measurements:

- Two process replicates of FlashInfer tensor-core paged GQA at batch
  1/16/64, context 2048/8192, and one heterogeneous batch-64 shape.
- The measured explicit-CUDA-Graph 4096-to-11008 decode MLP at batch 1/8/32.

The simulator composes those per-layer service curves across 32 layers. It
models a batch-64 continuous decode scheduler, reserves full prompt-plus-output
KV against a 32 GiB budget, and compares:

1. Ragged summed-context paged work with unbounded FCFS.
2. A padded-to-longest negative control.
3. Ragged work with a 250 ms predicted decode-stage TTFT admission guard.

The full matrix covers homogeneous and mixed prompt/output lengths, five
offered loads, both measured attention replicates, and three Poisson arrival
seeds. It contains 30 aggregate records and 180 scenarios. Every scenario
asserts exact request classification and generated-token accounting, batch
size at most 64, and KV reservations within budget. The attention fit's worst
relative error is 12.4%.

Key mixed-workload findings:

- Padded work inflation was about 3.26x. At 0.8x ragged capacity, ragged
  execution sustained about 1.92x the padded control's output-token
  throughput.
- At 1.1x capacity, unbounded ragged FCFS reached 10.1-second p99
  decode-stage TTFT while p99 TPOT remained about 19.5 ms.
- The admission guard accepted 88.5% of arrivals, retained 98.1% of output
  throughput, and held p99 decode-stage TTFT at 248 ms.

These are measured-kernel simulations, not new GPU executions or end-to-end
serving measurements. TTFT starts at the decode scheduler and excludes
ingress, tokenization, prefill, prefix-cache lookup, and sampling.
