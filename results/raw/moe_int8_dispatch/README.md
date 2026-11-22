# INT8 MoE dispatch artifacts

This directory preserves a two-GPU quality smoke, two full process launches
on 8 and 24 GPUs, and two 24-GPU alternating-order crossover launches.

The BF16 baseline keeps hidden activations contiguous and packs expert ID plus
router weight into one BF16 metadata collective. The compressed path
symmetrically quantizes each routed hidden row to INT8, transmits one BF16
per-token scale with the metadata, and dequantizes back to BF16 before the
same fused expert kernel. The BF16 return all-to-all is unchanged.

Every run verifies expert IDs and router weights, measures hidden and final
output drift against the BF16 path, and records quantize/dequantize overhead.
The crossover alternates BF16 and INT8 timing order for balanced, bias-1, and
bias-4 routing at 2048 tokens per rank. Files are copied serially by node 0.
