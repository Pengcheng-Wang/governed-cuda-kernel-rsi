# Fixed-profile MoE route packer

This experiment optimizes the lossless calibrated fixed-profile transport path.
It compares the v28 stable weight-sort plus destination-sort baseline with a
lossless destination-only stable sort and a fused Triton atomic histogram/
scatter kernel.

The Triton kernel assigns each top-k route a destination-local slot using an
atomic counter, gathers the token hidden row, and writes hidden state, local
expert ID, router weight, and return token index in one program. Fixed profile
capacities and offsets are precomputed outside the measured step.

Packing-only timing reuses router outputs. Route timing includes projection and
fused top-k. End-to-end timing adds fixed dispatch, local experts, return, and
weighted combine. Every optimized path must have zero overflow, retain the
same assignment set, and match the dynamic and masked outputs exactly.