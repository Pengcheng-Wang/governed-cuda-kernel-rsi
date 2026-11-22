# Packed MoE dispatch artifacts

This directory preserves a two-GPU correctness smoke, two full process
launches on 8 and 24 GPUs, four focused order-sensitivity launches, and two
24-GPU Latin-square crossover launches.

The split baseline sends hidden BF16, local expert int32, and router-weight
BF16 with three variable payload all-to-alls after count exchange. The
all-in-one path appends ID and weight as BF16 columns and uses one payload
collective, but must materialize contiguous hidden activations before fused
experts. The metadata-only path keeps hidden activations contiguous and packs
ID plus weight into one metadata collective, reducing payload collectives
from three to two.

All variants are checked against each other and the global-expert BF16
reference before timing. The crossover rotates split, all-in-one, and
metadata-only methods through first, middle, and last positions over 30
synchronized cycles. Files are copied serially by node 0.
