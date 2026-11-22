# Expert-parallel overlap chunk-count sweep

This directory preserves a fixed-total-work A100 study of one, two, four, and eight microbatches for dispatch/compute overlap. The 24-GPU full runs hold total tokens per peer at 1024 or 4096 while varying per-chunk payload. They compare communication-only, fused-expert-only, sequential, and pipelined schedules with exact sequential-versus-pipelined output validation. Two full process-level repetitions are retained. Files are copied serially by node 0 from node-local scratch.
