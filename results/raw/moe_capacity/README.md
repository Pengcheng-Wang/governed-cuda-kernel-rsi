# Integrated MoE capacity artifacts

This directory preserves a two-GPU correctness smoke and two full A100
process launches on 8 and 24 GPUs. The workload reuses the integrated
global-biased router, token permutation, variable dispatch, local fused
experts, return all-to-all, and weighted combine path.

For each source-destination segment, capacity factors 1.25, 2, 4, and 8 keep
the highest router weights up to
`ceil(capacity_factor * local_assignments / world_size)`. Dropped weights are
not renormalized. Every distributed result is checked against an
independently masked global-expert reference; output quality is separately
reported against the uncapped route. Separate stage timings reuse prepared
tensors and are diagnostic rather than additive. Files are copied serially
by node 0.
