# Static-capacity padded MoE dispatch

This experiment tests an A100-compatible way to remove the variable-split
host boundary: assign every source-destination pair a static capacity and use
equal-size all-to-all for dispatch and return.

The variable control exchanges counts and materializes Python split lists.
The padded path:

1. Selects the same highest router weights per source-destination segment.
2. Packs accepted assignments into fixed destination slots entirely on GPU.
3. Sends equal-size hidden and metadata tensors without count exchange.
4. Runs dense experts on valid and zero-weight padding rows.
5. Returns a fixed-size tensor and combines all slots; padding contributes
   exact zeros.

Capacity factors 1/2/4/8/24 trade host synchronization for padding, dense
expert work, assignment drops, and output quality. Full runs cover balanced,
bias-1, and bias-4 routes at 512 tokens/rank. Position-balanced crossovers
pair a dropped candidate with a topology-specific lossless capacity for every
route.

The padded and variable paths must select identical assignments and match
bit-for-bit. Quality versus the uncapped global-expert output is reported
separately when capacity drops assignments.
