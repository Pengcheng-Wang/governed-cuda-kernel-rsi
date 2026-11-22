# Exact static-capacity MoE frontier

This experiment refines the coarse static-capacity sweep by deriving the exact
lossless source-destination slot size from each deterministic router output.
The threshold is the global maximum assignment count across every source and
destination pair.

Balanced and moderate-skew routes are measured at 75%, 87.5%, one row below,
exactly at, 112.5%, and 125% of that threshold. Every point compares the same
retained assignments in the host-split variable path and fixed-shape padded
path. The threshold-minus-one point must drop work, while the exact threshold
and all larger capacities must be lossless.

The matrix includes a two-GPU correctness gate, repeated 8- and 24-GPU runs,
position-balanced crossovers, and descending-order 24-GPU controls. Static and
variable outputs must match the masked global-expert reference bit-for-bit;
quality against the uncapped output is reported separately for clipped points.