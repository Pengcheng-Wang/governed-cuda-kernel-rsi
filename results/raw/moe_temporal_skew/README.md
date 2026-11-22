# Integrated temporal MoE skew artifacts

This directory preserves a two-GPU correctness smoke and two full A100
process launches on 8 and 24 GPUs. Each temporal cycle contains one
microbatch per rank. Rotating schedules bias destination `step % world_size`,
so every rank is hot once and aggregate receiver totals return to near
balance; persistent schedules bias rank 0 in every microbatch.

The full router, permutation, variable dispatch, local fused-expert, return,
and weighted-combine path is measured for steady balanced, rotating bias 1/2,
and persistent bias 1/2 schedules. Results include cycle critical path,
per-step p50/p95/p99/max latency, aggregate and instantaneous receiver
imbalance, route/output drift, and global-expert correctness. Files are copied
serially by node 0.
