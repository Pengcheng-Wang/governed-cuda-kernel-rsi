# Aligned INT8 return overlap

This experiment integrates the 16-byte aligned single-collective INT8 return
format into the fixed-work expert-parallel pipeline.

Every microbatch performs:

1. Equal-split BF16 dispatch all-to-all.
2. vLLM fused expert computation.
3. One of three expert-output return paths:
   - BF16 in one collective.
   - INT8 values and one BF16 scale in two collectives.
   - INT8 values plus the BF16 scale in one 16-byte-aligned `uint8` packet.

Each return path is measured sequentially and in the original three-stage
pipeline, which overlaps return of chunk `i-1` and dispatch of chunk `i+1`
with expert compute and return quantization for chunk `i`. Dequantization
runs only after its asynchronous return completes.

The full profiles hold total work fixed at 1024 tokens per peer with four
chunks and 4096 tokens per peer with eight chunks. Routing is precomputed;
router work, permutation, variable counts, and capacity are outside this
isolated overlap benchmark.

Pipelined and sequential outputs must agree exactly within each quantized
format. The aligned packet and two-collective INT8 outputs must also match
bit-for-bit. Numerical drift versus BF16 is reported separately.
