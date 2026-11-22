# Packed INT8 MoE return

This directory compares three expert-output return paths:

1. BF16 payload in one variable-split all-to-all.
2. INT8 payload plus one BF16 scale per assignment in two all-to-alls.
3. The same INT8 values and BF16 scale byte-packed into one `uint8` row and
   sent in one all-to-all.
4. A 16-byte-aligned packed control that pads the 1026-byte row to 1040
   bytes, retaining one all-to-all and 49.2% byte reduction versus BF16.

For hidden size 1024, both INT8 layouts send 1026 bytes per assignment versus
2048 bytes for BF16. The packed layout changes collective count, not the
quantization format.

The packed packet is a contiguous `[assignments, hidden + 2]` `uint8` tensor.
Its first `hidden` bytes are viewed as strided `int8`; its final two bytes are
viewed as one strided BF16 scale. A Triton kernel writes both typed views in
one quantization launch, and a second Triton kernel dequantizes the received
packet. No separate scale-pack or unpack copy is used.

The unaligned row intentionally tests the minimal byte count. The aligned
control zero-fills 14 padding bytes in the quantization kernel so rows and
destination segments begin on 16-byte boundaries. This distinguishes
collective-count savings from GPU-memory and NCCL alignment effects.

Quality gates compare BF16, forward-only INT8, two-collective round-trip INT8,
and packed round-trip INT8. The packed and two-collective paths must match
exactly because they use the same per-row symmetric quantization. The aligned
and unaligned packets must also match exactly.

Stage timings reuse prepared tensors and are diagnostic. Full-layer timings
include routing, forward dispatch, local fused experts, return communication,
and weighted combine. Crossover profiles rotate methods through timing
positions; the integrated critical path is authoritative.
