# Profile-hit fallback and grouped P2P

These results replay trained OLMoE router-count distributions and compare:

- fully dynamic metadata-packed all-to-all;
- tile-512 fixed all-to-all on profile hits with full dynamic fallback on
  misses;
- grouped fixed-size NCCL point-to-point dispatch/return on hits with the same
  full dynamic fallback.

The additive policy uses the prior +18/+16 per-cell margins. The adaptive
policy uses a two-domain sliding window with a topology-specific uniform row
budget: 16% on 8 ranks and 48% on 24 ranks. Router inference, profile
construction, and hit decisions are outside timing.

An explicit two-rank semantic gate posted eight-row receives for three- and
five-row sends. It timed out after 90 seconds; this PyTorch/NCCL path therefore
requires exact receive tensor sizes. Grouped P2P is only used for fixed-size
profile hits.

Two position-balanced crossovers were completed on 8 and 24 A100 GPUs. Fixed
all-to-all with full dynamic fallback is the positive result. Grouped P2P is
the negative control because per-peer operations are substantially slower than
the all-to-all collectives at this fan-out.

All completed methods preserve exact assignments. Maximum BF16 output error is
0.0009765625 and minimum cosine similarity is 0.99999976.
