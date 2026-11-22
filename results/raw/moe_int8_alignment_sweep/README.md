# INT8 return packet alignment sweep

This transport-only experiment generalizes the packed MoE return result
without allocating router or expert weights.

Each source rank creates a deterministic assignment-count row for balanced,
moderately hot, or extremely hot receiver traffic. The expert rank generates
BF16 output rows grouped by source, then returns them with:

1. One BF16 all-to-all.
2. INT8 values and one BF16 scale in two all-to-alls.
3. One `uint8` packet all-to-all with row sizes rounded to 2, 4, 8, 16, 32,
   or 64 bytes.

The Triton packet kernel writes INT8 values, the raw BF16 scale, and zero
padding in one launch. Packet outputs must match the two-collective quantized
return bit-for-bit.

The width matrix pairs standard transformer widths with `H-2` controls:
510/512, 1022/1024, and 4094/4096. The smaller width makes the raw
`H + 2`-byte packet naturally aligned, while the standard width requires
padding. This isolates alignment from collective count and nearly all payload
size effects.

Full profiles report conversion, prebuilt transport, and
quantize+transport+dequantize timings. Crossover profiles rotate BF16,
two-collective INT8, and every packet alignment through timing positions.
Only node 0 writes the durable copy.
