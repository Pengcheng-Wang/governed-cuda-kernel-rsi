# Device-side profile-fit and conditional fallback

This directory evaluates a production-style profile-fit decision on trained
OLMoE router traces.

- The tiled fixed-profile packer produces a device-resident overflow scalar.
- A functional NCCL all-reduce makes the fit decision identical on every rank.
- `torch.cond` consumes that scalar without `Tensor.item()` or a Python branch.
- Profile hits use fixed asymmetric all-to-all.
- Misses use an exact fixed-signature fallback: all-gather each source token
  once, filter local-expert assignments on device, and reduce-scatter weighted
  token contributions back to their source ranks.
- The fallback needs no Python split extraction. Its hidden-row traffic is
  `world_size / top_k` times dynamic assignment traffic: 1x on 8 ranks and 3x
  on 24 ranks for top-8 routing.

The two-GPU smoke contains synthetic hit and miss cases. The 8- and 24-GPU
crossovers select real additive-profile hits and misses at layers 0, 8, and
15, with normal and reversed method order. BF16 fallback accumulation stayed
within a maximum absolute output difference of 0.015625 and minimum cosine
similarity above 0.999993.

The result is negative but informative. Device-conditional execution reached
about 0.582x dynamic speed on 8 GPUs and 0.616x on 24 GPUs after combining the
order-balanced runs. The scalar fit collective, speculative pack on misses,
and structured-control-flow overhead cost more than the saved host barrier.
Each distinct profile/case also incurred about 0.36/0.70 seconds of one-time
conditional graph compilation on 8/24 GPUs.
