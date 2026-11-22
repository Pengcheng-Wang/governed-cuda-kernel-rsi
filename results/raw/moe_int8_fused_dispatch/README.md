# Fused INT8 MoE dispatch artifacts

This directory preserves a two-GPU correctness smoke, two full process
launches on 8 and 24 GPUs, and four 24-GPU Latin-square crossover launches.

The Triton send kernel fuses per-token absolute-max reduction, BF16 scale
rounding, INT8 quantization, and expert-ID/router-weight/scale metadata
packing. The receive kernel fuses INT8 dequantization with metadata unpacking
into contiguous BF16 hidden states, int32 expert IDs, and BF16 router weights.

BF16 metadata-packed, eager PyTorch INT8, and fused Triton INT8 paths share
the same routing, payload collectives, fused experts, BF16 return
communication, and weighted combine. Every run checks metadata and numerical
drift against BF16 and eager INT8 paths. Crossovers rotate all three methods
through timing positions. Files are copied serially by node 0.
