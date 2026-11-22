# GPU-compacted static MoE padding

This experiment keeps the v26 fixed-shape equal-size dispatch and return
collectives, but marks zero-weight padding rows with expert ID -1. vLLM then
ignores those invalid assignments during its GPU expert-alignment pass, so no
receive count or split list returns to Python.

Three expert paths are compared on the same static transport:

1. Dense padding: padding is expert 0 with weight zero, so all rows enter the
   aligned expert kernels.
2. Mapped sparse: invalid IDs plus an identity expert map; vLLM compacts the
   assignments and zero-initializes its output cache.
3. Row-cleanup sparse: invalid IDs without an expert map, followed by a custom
   Triton kernel that clears only invalid output rows.

Exact and 12.5%-headroom lossless capacities are measured for balanced,
bias-1, and bias-4 routes. The matrix includes a two-GPU gate, repeated 8- and
24-GPU runs, two 8-GPU crossovers, and three 24-GPU crossovers. All methods
must select identical assignments, match the masked global-expert reference
bit-for-bit, and produce exact zeros for invalid rows.

The result distinguishes global padding from critical-rank padding. Under
receiver skew, most padding can reside on noncritical ranks, so a large
aggregate aligned-row reduction does not imply a shorter distributed critical
path.