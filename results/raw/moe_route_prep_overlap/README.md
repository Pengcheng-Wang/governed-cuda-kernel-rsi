# Online MoE route/count preparation overlap

This experiment moves router projection, fused top-k, token permutation,
metadata packing, count exchange, and CPU split extraction into the measured
expert-parallel path.

It compares:

1. A precomputed payload-pipeline lower bound matching the v23 timing scope.
2. Online staged preparation followed by the same payload pipeline.
3. Online double-buffered preparation using a dedicated NCCL count
   communicator, a pinned nonblocking send-count copy, blocking count
   exchange/receive-count extraction, and a separate router CUDA stream.

BF16 and 16-byte-aligned INT8 packet returns are both measured. The online
overlap schedule prepares future microbatches while the current variable
dispatch, local experts, return, and weighted combine remain active.

The receive split must still cross to Python because
`all_to_all_single` requires host split lists. NCCL did not reliably inherit
the dependency from the external router stream when count exchange was
launched before the route event at scale. The benchmark therefore overlaps
device router/packing work, then launches count exchange and extracts receive
splits after the route event. That synchronization is part of the measured
limitation rather than being hidden outside timing.

BF16 output must match an independently evaluated global-expert reference
exactly. Online preparation must not change BF16 or packet output, and packet
quality must remain within the existing v23 envelope.
