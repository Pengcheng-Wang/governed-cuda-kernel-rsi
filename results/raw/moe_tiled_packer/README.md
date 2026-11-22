# Tiled fixed-profile MoE packer scaling

This experiment scales the lossless fixed-profile route packer across hidden
widths and assignment counts. It compares the two-sort baseline, the v29
one-program-per-assignment atomic kernel where supported, and a two-pass
Triton design.

The tiled design first atomically reserves one destination-local slot per
assignment, then scatters the hidden row in BLOCK_SIZE-wide tiles while tile
zero writes expert ID, router weight, and return token index. This avoids one
program spanning a full production-width hidden row.

The matrix covers hidden widths 512/1024/4096, token counts 512/2048/8192,
balanced/moderate/extreme routing, and 2/8/24 GPUs. Every packed result is
sorted by token/expert key and checked for exact hidden, weight, token, zero
padding, and zero-overflow equality.