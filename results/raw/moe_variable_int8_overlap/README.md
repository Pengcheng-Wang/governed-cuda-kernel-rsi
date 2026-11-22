# Variable-split aligned INT8 overlap

This experiment extends aligned INT8 expert-output return overlap to
router-derived variable expert-parallel traffic.

Each microbatch uses the real BF16 router projection, vLLM fused top-k,
token duplication, and destination sorting to determine send counts. Router
work, count exchange, and CPU count extraction are prepared outside timing;
the measured path includes:

1. Variable-split BF16 hidden and packed ID/weight dispatch.
2. Local vLLM fused experts.
3. BF16, two-collective INT8, or 16-byte-aligned packet return.
4. Dequantization and weighted inverse permutation.

Every return format is measured sequentially and in a four-microbatch
pipeline. The pipeline overlaps return of chunk `i-1` and dispatch of chunk
`i+1` with expert compute and return quantization for chunk `i`.

The full profile holds work fixed at 2048 tokens per rank, split into four
chunks, and compares balanced, bias-1, and bias-4 global routing. BF16 output
must match a global-expert reference exactly. Packet and two-collective INT8
must match bit-for-bit, and pipelined output must match sequential output
within each format.
