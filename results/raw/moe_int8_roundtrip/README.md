# INT8 MoE round-trip artifacts

This directory preserves a two-GPU correctness smoke, two full process
launches on 8 and 24 GPUs, and four 24-GPU Latin-square crossover launches.

The BF16 baseline uses metadata-packed BF16 forward dispatch and one BF16
return all-to-all. The forward-only path uses fused Triton INT8 dispatch but
keeps the BF16 return. The round-trip path additionally quantizes every
expert-output row to INT8 with one BF16 scale, sends payload and scale in two
return collectives, and dequantizes before weighted combine.

Every run decomposes forward-only, isolated return-only, and complete
round-trip numerical drift. Crossovers rotate BF16, forward-only INT8, and
round-trip INT8 through all end-to-end timing positions and separately compare
BF16 versus compressed return. Files are copied serially by node 0.
