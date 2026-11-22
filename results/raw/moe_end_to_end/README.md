# Integrated expert-parallel MoE artifacts

This directory preserves a two-GPU correctness smoke and two full A100
process launches on 8 and 24 GPUs. The synthetic layer includes BF16 router
projection, vLLM fused top-2, token duplication and destination sorting,
count exchange, variable-split hidden/id/weight dispatch, two local experts
per rank, return all-to-all, and FP32 weighted index-add combine.

The balanced, globally biased, and locality-biased routes use identical
expert weights and inputs across repetitions. Every promoted output is
checked against a local global-expert reference. Separate router, dispatch,
expert, and return timings reuse prepared tensors and are diagnostic rather
than additive; the integrated end-to-end critical path is authoritative.
Files are copied serially by node 0.
