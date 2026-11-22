# Integrated MoE router-skew sweep artifacts

This directory preserves a two-GPU correctness smoke and two full A100
process launches on 8 and 24 GPUs. The sweep adds a common FP32 bias of
0, 0.5, 1, 1.5, 2, 3, or 4 to rank 0's experts before fused top-2 while
holding inputs, gate weights, and expert weights fixed.

Each run covers router projection, token duplication and sorting,
variable-split hidden/id/weight dispatch, local fused experts, return
all-to-all, and weighted combine. Results include rank-0 assignment share,
receive/expert imbalance, effective destinations, route-set agreement and
output cosine versus bias 0, stage timing, and integrated latency. Every
distributed output is checked against the corresponding global-expert
reference. Files are copied serially by node 0.
