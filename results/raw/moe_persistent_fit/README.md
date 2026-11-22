# Persistent batched profile-fit controller

This directory tests whether a serving scheduler can amortize the global MoE
profile-fit decision across several independent ready microbatches.

- `scalar_host` packs, all-reduces one overflow flag, synchronizes to the CPU,
  and branches independently for every microbatch.
- `vector_host` retains all speculative fixed routes, all-reduces one vector
  of overflow flags, performs one CPU synchronization, then executes fixed
  all-to-all on hits and the original dynamic path on misses.
- `ideal_precomputed` is an upper bound with known fit decisions.
- Queue depths 1/2/4/8 use natural trained OLMoE hit and miss cases at layers
  0, 8, and 15. Focused depth-8 controls use 20 balanced timing cycles.
- Additive and sliding adaptive profiles use the same calibration policies as
  the preceding trained-router experiments.

The controller-only cost amortizes almost linearly. At depth 8, vector
decisions cost about 18.9 microseconds per case on 8 GPUs and 22.5-24.0
microseconds on 24 GPUs, versus 123-126 and 156-157 microseconds for scalar
decisions.

End-to-end benefit also requires enough profile hits:

- Additive depth-8 mixed queues reached 1.111x dynamic on 8 GPUs but only
  0.954x on 24 GPUs. Their observed hit rates were 68.5% and 33.3%.
- Adaptive profiles raised hit rates to 88.9% and 63.9%, reaching
  1.234x/1.088x dynamic on 8/24 GPUs.
- The vector controller was 1.24x/1.16x faster than per-case scalar control
  for the adaptive 8-/24-GPU queues.
- Retaining eight speculative routes increased peak allocated memory by about
  36 MiB on 8 GPUs and 46 MiB on 24 GPUs.

All global decisions matched the precomputed fit labels. Maximum BF16
difference was 0.0009765625 and minimum cosine similarity was 0.99999976.
