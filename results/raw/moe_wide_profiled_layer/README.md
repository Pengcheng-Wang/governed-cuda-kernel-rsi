# Wide fixed-profile MoE layer

These results integrate the tile-512 fixed-profile packer into a
hidden-4096/intermediate-4096 expert-parallel MoE layer.

- Dynamic control: online variable split lists.
- Fixed controls: two-sort packing and two-pass tiled atomic packing.
- Routing: top-2 over two experts per rank, with balanced, bias-1, and bias-4
  held-out routes.
- Calibration: maximum source-destination counts over eight disjoint batches,
  followed by the smallest additive margin that covers every held-out cell.
- Lossless margins: +18 rows per cell on 8 GPUs and +16 on 24 GPUs.
- Repetition: two order-balanced 8-GPU crossovers and five 24-GPU crossovers,
  plus two full runs on each topology.
- Timing: route-plus-pack, prebuilt payload
  (dispatch→experts→return→combine), and complete end-to-end execution.
- Correctness: zero overflow, identical assignment sets, exact zero padding,
  and bit-exact agreement among dynamic, fixed, and independently masked
  global-expert outputs.

The tile-512 kernel remains faster than dynamic route packing, but the
fixed-profile layer is not generally faster end to end. Per-cell headroom
pads 30–37% of rows on 24 GPUs, so padded communication and dense expert work
erase the routing gain for balanced and moderate skew.

`moe_wide_profiled_capacity_24gpu.json` is a calibration-only diagnostic. It
allocates no expert weights and records the exact held-out additive margin for
each route and token count.
