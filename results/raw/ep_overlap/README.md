# Expert-parallel dispatch/compute overlap artifacts

This directory preserves correctness-gated A100 measurements for a four-microbatch pipeline that overlaps equal-split BF16 NCCL all-to-all dispatch of microbatch i+1 with vLLM fused-expert computation for microbatch i. Full 8-GPU and 24-GPU benchmarks have two process-level repetitions. Routing, permutation, combine communication, and router computation are outside this isolated study. Files are copied serially by node 0 from node-local scratch.
