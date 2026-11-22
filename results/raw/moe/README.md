# Mixture-of-Experts benchmark artifacts

This directory preserves correctness-gated A100 results for the local vLLM fused-expert study and NCCL expert-dispatch all-to-all study. Files are copied serially by node 0 from node-local scratch. Full benchmarks have process-level repetitions where available; smoke files are retained as execution checks.
