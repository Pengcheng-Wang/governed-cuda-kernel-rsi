# CUDA kernel optimization lab

This lab is a reproducible learning project for CUDA and GPU-kernel
optimization in modern LLM/SLM training and inference. It combines a
source-verified state-of-the-art review with measured experiments on NVIDIA
A100 GPUs.

## Boundaries

- All new work belongs in this directory. Prior labs are read-only references.
- Resource identifiers, connection settings, operational logs, and
  `progress.md` are local-only and ignored.
- The user-owned allocation is never cancelled or released by this project.
- Node-local storage is ephemeral and not shared. Only node 0 writes the
  project namespace on durable shared storage.
- Heavy shared-storage writes are serialized. Benchmarks run from node-local
  storage and copy only compact results to durable storage.
- Performance claims require correctness checks, warm-up, synchronized timing,
  workload and software metadata, and repeated measurements.
- Faster is not automatically better: comparisons must preserve numerical
  intent, precision, model shape, and training/inference semantics.

## Study tracks

1. **Execution fundamentals**: launch overhead, coalescing, occupancy, memory
   bandwidth, arithmetic intensity, and roofline reasoning.
2. **Fused elementwise, reduction, optimizer, and terminal-loss kernels**:
   SwiGLU and RMSNorm in eager PyTorch, `torch.compile`, Triton, and a native
   CUDA extension, single-launch pointer-table multi-tensor Triton `AdamW`
   kernels (`FP32` master + `BF16` `grad_dtype`, 128-block `INT8` `m` + `UINT8`
   `sqrt(v)` 8-bit states, and unbiased Philox stochastic-rounding `SR-BF16`
   master weights) with in-register global `clip_grad_norm_` scaling and
   shape-bucketed batched Tensor-Core `Muon` (`E118`), large-vocabulary
   (`V = 128,256`) token-tiled Linear + Cross-Entropy combining cuBLAS Tensor
   Core GEMMs with an in-place Triton online-softmax gradient kernel across
   1B/3B SLM and 8B LLM shapes (`E112`), and Reasoning Post-Training (`RLVR` /
   `GRPO` / `DAPO` / `DPO`) terminal kernels combining shift-invariant centered
   online `(m, d, e)` vocabulary entropy (`3`-GEMM `FusedLinearGRPO/DAPO`) and
   `4`-GEMM Two-Pass `Saved-LSE` `FusedLinearDPO` (`E121`), plus SLM Knowledge
   Distillation (`Forward KL` / `MiniLLM Reverse KL` / `GKD JSD_0.5`) 6-statistic
   coupled centered `(m_S, d_S, e_S, m_T, d_T, c_ST)` monoid kernels and
   DeepSeek-V3 Multi-Token Prediction (`MTP`) Fused Dual-`RMSNorm` + shared-head
   batched `FP32-dW` kernels (`E122`), plus zero-gather / zero-atomic
   block-segmented Triton `DiT` `adaLN-Zero` + `Gate-Residual-adaLN` (`4.90x` vs
   eager, `2.62x` vs `torch.compile`, `E127`), Round-2 Adaptive Direct-`dc6` /
   Split-`32` 3-Point & Unified `adaLN-Zero` (`10.42x` / `17.43x` over Round-1,
   `85.91x` over eager, `1,243.6 GB/s` physical HBM, `E128`), and
   Active-Mask-Compacted `3-GEMM` `FusedLinearMaskedCE` (`V = 128,256`) with
   complementary-antithetic `1/t` NELBO masking (`E127`).
3. **Tensor Core GEMM**: cuBLAS/PyTorch versus an educational Triton matmul
   across representative transformer shapes.
4. **Attention and sub-quadratic sequence mixers**: math SDPA, Flash SDPA,
   FlashAttention-2 (`flash_attn_func` and `flash_attn_varlen_func`), PyTorch
   `FlexAttention` (`BlockMask` + `score_mod` + `enable_gqa=True`) across dense
   causal MHA, sequence-packed document-masked GQA (`S = 8192`), Gemma-2
   sliding-window `tanh` logit soft-capping, and custom non-linear logarithmic
   positional decay, plus overflow-free chunkwise `Mamba-2 SSD / GLA` and
   SM80 Tensor-Core `4 × 16` nilpotent WY `Gated DeltaNet` training kernels
   (`Fwd + Analytical Bwd` up to `S = 65,536` on 1 GPU and `S = 262,144` on
   8-GPU Context Parallelism, including `3:1` Hybrid stacks and crowded
   associative state-reassignment MQAR, `E119`), and DeepSeek-V2/V3/Kimi-K2
   Multi-Head Latent Attention (`MLA`: Split-Dim `128+64 -> 128` Triton
   `Flash-MLA` vs SM80 SRAM-safe `4 × 128` strip-streamed `512+64 -> 512`
   Weight-Absorbed MLA, `S_q = 1..4096` decode-to-prefill crossover, and
   zero-attention-recompute `Latent-SAC`, `E120`), and Two-Slice Ragged
   Group-Prefix-Tree `FlashAttention` (`32Q/8KV, D=128`) for shared-prompt
   `GRPO` rollout trees (`G = 4, 8, 16`, `E121`), and GQA-Packed Two-Region
   `EAGLE-3` Training-Time Test (`TTT`) attention (`4 anchors x 4 steps x 4 GQA
   heads = 64` rows/CTA, `E122`), and Natively Trainable Dynamic Sparse
   Attention (`DeepSeek NSA` Folded-Staircase `FA2` compression + `[Subchunk-4 x
   G=16]` selection + 100% Lock-Free Split-`256` `CSC` backward & `Moonshot Kimi
   MoBA` GQA-group varlen LSE routing + Zero-Selection-Recompute `NSA-SAC`,
   `S = 4K..64K`, `E125`), and Native-Resolution Multimodal VLM & Omni-Modal
   Training Kernels (`Window-Morton-2x2` zero-permutation ragged ViT + `19.5x`
   tubelet pack, single-launch `3D M-RoPE`, & Surgical 2-Pass Disjoint
   Reversed-Upper `FlashAttention-2` + Zero-Atomic Triton `LSE` hybrid
   bidirectional-vision / causal-text `Omni-Attention`, `P = 4K..32K` &
   `S = 4K..64K`, `E126`), and Block-Diffusion `BD3-LM` Dual-Stream `[x_t ||
   x_0]` `2L x 2L` Training Kernels (`W = 4..256`, `L = 2K..32K` / `2L =
   4K..64K`: Adaptive Zero-Spill Split-Stream GQA Microblocks + Shift-Isomorphic
   Surgical `FA2` + Exact Terminal Clean-Branch Pruning, `E127`; Round-2
   `K_TILE=16` Disjoint-Lifetime `0`-Cubin-Spill GQA Microblocks + Guarded
   Zero-Copy `OBC` + Direct Global-`(Out_t, LSE_t)` `FA2` Backward `84.5x` `dlse`
   error reduction, `E128`).
5. **Decode and KV cache**: eager execution, CUDA Graphs, contiguous/paged
   MHA/GQA/MQA, and FlashInfer CUDA-core/tensor-core dispatch.
6. **Quantization & low-precision training**: generic W8A8 as an inference
   negative control, production vLLM Marlin W4A16 across decode-to-prefill
   projection shapes, and full `3-GEMM` (`Fwd + dX + dW`) Block-Hadamard `INT8`
   Tensor-Core training (`SwitchBack` / `Jetfire` / `COAT` / `Halo`) surpassing
   SM80's `312 TFLOP/s` `BF16` hardware ceiling with `s32`-bitcast `RED.MAX.S32`
   + SplitMix32 stochastic-rounded Hadamard quantizers, `rstd`-invariant Fused
   `RMSNorm -> INT8`, and `COAT` `1 B/elem` `INT8` saved `SwiGLU` activations
   (`E123`), plus SM80 `2:4` Semi-Structured Sparse `BF16` Tensor-Core `3-GEMM`
   pre-training (`cuSPARSELt` `448 eff TFLOP/s`, `Exact-90` bipartite DP dual
   packer, `KKT-Exact Madow MVUE 2:4` `dW`, `16-bit Bitmask Grad-SR-STE`, and
   layout-aware `Sparse-SwiGLU`, `E124`).
7. **Distributed training**: NCCL all-reduce, DDP weak scaling, FSDP2
   throughput/memory/collective tradeoffs on 1/8/24 GPUs, intra-node NVSwitch
   `SymmetricMemory` one-shot/two-shot collectives, Megatron Tensor and
   Sequence Parallelism (`[B, S, H]` `Shard(1)` versus `[S, B, H]`
   `Shard(0)`), TorchInductor `Async-TP` (`_micro_pipeline_tp`)
   collective-GEMM overlap across `TP = 2, 4, 8`, long-context Context
   Parallelism (`CP = 2, 4, 8` across `S = 8K..128K`) comparing Contiguous vs
   Zig-Zag Ring FlashAttention-2 (with a fused Triton online-softmax LSE
   merger and double-buffered P2P overlap), GQA-native vs KV-expanded
   DeepSpeed-Ulysses, Hybrid 2D `USP (Ulysses x Zig-Zag Ring)` across the
   `CP > H_kv` sparse-GQA wall, and PyTorch 2.13 `context_parallel`, plus
   16-layer Llama-3-8B-width Pipeline Parallelism (`PP = 4, 8`) comparing
   `GPipe`, `1F1B`, `Interleaved 1F1B`, Zero-Bubble (`ZB1P` and `ZB-V`), and
   DeepSeek-V3 `DualPipeV` with a custom zero-copy `SplitDWLinear` +
   in-place cuBLAS `addmm_` `dw_builder`, sequence-length `t_F : t_Bx : t_Bw`
   skew analysis (`S = 1K..8K`), and Selective Activation Checkpointing (`SAC`),
   plus 8-GPU `FSDP2` distributed optimizer scaling (`3.49B` isolated and
   `0.97B` end-to-end training steps) comparing `AdamW` (`foreach`/`fused`),
   Triton `INT8`-moment and `SR-BF16` `AdamW`, PyTorch 2.13 `DTensor` `Muon`,
   and Moonshot-style 2-stream pipelined owner-sharded `AllToAll` + batched
   `BF16` `bmm` Tensor-Core `Muon` (E118), plus 8-GPU long-context linear-time
   Context Parallelism (`CP = 8`, `S = 16K..262K`) comparing serial P2P
   state-passing, an `O(1)` `AllGather` augmented `[S_zero | P]` parallel
   affine-monoid scan (`WY GDN` & `SSD`), and GQA-native DeepSpeed-Ulysses
   `FlashAttention-2` (E119), plus 8-GPU Multi-Head Latent Attention Context
   Parallelism (`CP = 8`, `S = 16K..131K`) comparing `14.33x`-smaller
   `Compressed-Latent Zig-Zag Ring` (`[c_KV | k_R]` `576` BF16/tok with
   `ncclStream`-safe ping-pong buffers), `Decompressed Ring`, `FA2 V-padded
   Ulysses`, and `Split-Dim Ulysses + Shared-K_rope AllGather` (E120), plus
   8-GPU `FSDP2` Long-CoT `RLVR` training auditing rank-local token-mean
   normalization bias vs global `DAPO` `world_size / N_resp,global` weighting
   and Adaptive Tree Split + Quadratic-Cost LPT Prefix-Tree `GQA` (E121), plus
   8-GPU NVSwitch Hybrid `FSDP2`-Backbone / `TP = 8` Vocabulary-Parallel
   (`V_shard = 16,032`) Coupled-Monoid Teacher-Student SLM Distillation (`24
   B/tok` `[N, 6]` collective + `FP32` `dH_S` `ReduceScatter`, E122), plus
   8-GPU NVSwitch Shard-Local Hadamard `1 B/param` `INT8` Quantized-`AllGather`
   + `COAT` `INT8`-Activation `FSDP2` Training (`1.39B` params, `65,536` global
   tok/step, E123), 8-GPU NVSwitch Shard-Local `Exact-90` `9 bits/param`
   Compressed-`AllGather` `2:4` Sparse `BF16` `FSDP2` Training (`1.31B` params,
   `65,536` global tok/step, `4,992 -> 2,808 MiB/step` `AllGather`, E124), and
   8-GPU NVSwitch Group-Preserving `2D USP-NSA (Ulysses-4 x
   Causal-Asymmetric-ZigZag-2)` Dynamic Sparse Attention Context Parallelism
   (`CP = 8 > H_kv = 4`, `S = 16K..131K`, E125), and 8-GPU NVSwitch Two-Stage
   Modality-Decoupled `AllToAllv` (`4x`-Compressed Post-Merger Return) +
   `Quad-Cost-LPT` `FSDP2` Multimodal Dual-Tower Training (`0.93B` `ViT-1280 +
   Merger + Omni-LLM-3584`, `B = 16, 32`, `50,176` global LLM tokens, E126),    and 8-GPU NVSwitch `1.12B` Block-Diffusion `BD3-LM` (`DiT` `adaLN-Zero` + `V =
   128,256`) `FSDP2` 4-Rung RSI Training (`B = 16, 32` up to `131,072` `[x_t ||
   x_0]` dual tokens/step, saving `15.92 GiB/rank` with `M = 0` empty-mask
   collective symmetry, E127; Round-2 Merged `QKV-Split-RoPE` + `FusedSwiGLU` +
   `Split-32 AdaLN` + 2-Block NVSwitch Prefetch breaking E127's `1.087x` ceiling
   to `463,119 2L tok/s` = `1.76x` over Round-1 / `2.21x` over Literature at `W =
   64` and `1.64x` / `1.70x` at `W = 16`, plus a 5-generation fixed-`12.0s`
   `val_bpb` `autoresearch` loop reaching `2.30x` updates and `-2.75 val_bpb`,
   E128) on 8xA100-SXM4-80GB.
8. **Mixture of Experts**: end-to-end differentiable fine-grained MoE training
   (`E = 8, 32, 64`, `top-2/4/8`, `Fwd + dX + dW + dRouter + Switch/z-loss`)
   with device-resident L2-swizzled Triton Grouped GEMMs, fused token
   permute/unpermute + `d_probs` backward, dynamic-routing CUDA Graphs, and
   8-GPU `EP = 8` `Split-dW` backward `AllToAll` overlap (E117), alongside
   inference fused expert scheduling, top-k, routing skew,
   assignment padding, profiler evidence, 8-/24-GPU all-to-all dispatch, and
   fixed-total microbatch dispatch/compute overlap, granularity tuning, and
   the full dispatch→expert→combine round trip, variable splits, locality,
   hot spots, capacity clipping, fused router top-k, and assignment
   preparation, plus an integrated router→permutation→variable-dispatch→
   local-expert→weighted-combine layer and its destination-capacity versus
   output-quality frontier, a continuous router-skew onset curve, and
   temporal rotating-hotspot cycles, plus split versus packed dispatch
   metadata and activation-compressed INT8 dispatch with order-balanced
   24-GPU crossovers, including fused Triton conversion kernels and
   bidirectional activation compression on the expert-output return path,
   plus single-collective scale packing, packet-alignment controls, and a
   paired-width 2/4/8/16/32/64-byte alignment sweep, then integration of the
   aligned packet into the fixed-work three-stage expert pipeline and a
   router-derived variable-split follow-up that exposes when BF16
   communication hides better than compressed conversion, followed by
   router/count preparation inside the measured online pipeline and a
   static-capacity equal-split alternative that trades host synchronization
   for padding, dense expert work, and capacity loss, plus an exact
   router-derived capacity frontier that finds the minimum lossless slot
   count and nearby fused-expert shape cliffs, followed by invalid-expert GPU
   compaction that separates aggregate padding from critical-rank work, then
   fixed-profile asymmetric transport that removes online count exchange
   without transmitting every uniform static slot, followed by a fused
   Triton atomic packer that removes both general-purpose sorts and the
   intermediate gather/scatter chain, then a tiled reservation/scatter
   design that scales to hidden 4096 and tunes 128/256/512-wide row tiles,
   followed by hidden-4096 layer integration with topology-specific lossless
   profile calibration and route/payload/end-to-end decomposition, then
   trained OLMoE router traces, bounded-row adaptive profiles, and an exact
   variable spill side channel that exposes the extra-collective boundary,
   followed by profile-hit/full-dynamic fallback and grouped P2P controls,
   then a device-resident overflow predicate with `torch.cond` and an exact
   count-free all-gather/reduce-scatter miss path, followed by a persistent
   scheduler control that batches fit flags across ready microbatches and
   trades queued-route memory for fewer global decisions, followed by a
   measured-service Poisson queue study that maps throughput gains to
   p50/p95/p99 response latency and timeout policy. A complementary serving
   study composes measured paged-GQA and decode-MLP kernels into a 32-layer
   continuous-batching model with heterogeneous lengths, ragged versus padded
   work, KV limits, decode-stage TTFT/TPOT, and an SLO-aware admission guard,
   followed by a static graph-shape bucket frontier that separates context
   padding, batch padding, sequential bucket launches, and graph-cache size,
   then a nine-class admission audit that separates request-count SLO gains
   from output-token fairness, followed by a measured FlashAttention/cuBLAS
   chunked-prefill model with prefix reuse and decode-priority controls, then
   an adaptive kernel-time-slice controller with queue-age escalation,
   followed by hard stall caps, explicit prefill-backlog admission, and
   structural safety-factor versus prompt-fairness controls, then an online
   equal-class feedback controller that exposes the latency/throughput cost
   of explicit fairness, followed by rotating prompt-mix phases that compare
   cumulative state with bounded rolling histories, then exponential
   wall-time versus predicted-service decay, calibration-error robustness,
   tenant/reservation-class objectives, and causal batched-decode service
   attribution, followed by a closed-loop controller whose fairness ledger
   consumes observed prefill slices and exactly allocated batched decode
   service, then admission-time pending-service commitments under Poisson
   and fixed-size request bursts, followed by ingress pacing that keeps
   latency anchored to the original client arrival, then absolute pacing
   budgets selected against robust client-TTFT SLOs, followed by compression
   of the robust pacing oracle into deployable lookup policies and a
   state-estimation uncertainty safety audit, then confidence-set fallback
   driven by per-feature mistake recall and false-alarm rates, followed by
   explicit risk-budgeted detector operating frontiers and finite-sample
   monitoring certificates for those operating assumptions, then
   anytime-valid repeated-monitoring and drift-detection costs, followed by
   geometric checkpoint schedules that trade label efficiency for action
   latency and Bernoulli-KL Chernoff intervals that remove most of the
   remaining distribution-free bound overhead, followed by metric-aware
   weighted alpha allocation and a global certification/drift Pareto audit
   with explicit worst-fallback-delay caps, then checkpoint phase/rounding
   sensitivity and an exhaustive full-simplex audit to distinguish robust
   choices from lattice artifacts, followed by robust checkpoint-cadence
   selection under alternative valid grid constructions and a per-look
   operational-cost frontier, followed by asynchronous monitor-worker
   concurrency and queueing, then outcome-independent latest-pending
   coalescing that replaces stale queued looks without preempting running
   checks, followed by realized Bernoulli-path auditing that separates
   optional-stopping validity from strict pathwise action-latency dominance,
   then a freshest-K pending-depth frontier that quantifies aggregate risk,
   worst-profile tails, processed work, and the FIFO-equivalent limit,
   followed by exact batched replay that preserves every declared look and
   prices the marginal cost of extra sufficient-statistic snapshots, then
   chronological partial-result streaming that removes the batch fence under
   identical launch and work accounting, followed by outcome-independent
   member-order selection that trades regression frequency against delay
   severity without changing work, then a robust policy-selection audit that
   compares fixed, domain-aware, cost-aware, metric-aware, and metric+cost-aware
   orders with explicit lexicographic objectives and a complexity-aware Pareto
   frontier, followed by exact adjacent-cost-bucket compression under observed
   hard delay caps to separate rule-table complexity from tail-latency cost,
   then an ordinal cost-estimation uncertainty audit that tests nominal,
   compressed, fixed, and minimax mappings under adjacent bucket errors,
   followed by a joint uncertainty-plus-delay-cap frontier that exposes
   separate aggregate, adversarial, and worst-cell deployment gates, then
   an analytic estimator-accuracy frontier that derives exact policy
   crossovers under symmetric and directionally biased cost errors, followed
   by exhaustive cost-prevalence sensitivity and distribution-free minimax
   per-cost regret, then exact prevalence-boundary geometry and conservative
   finite-sample policy-certification budgets, followed by regret-tolerant
   certification that converts small outcome slack into practical data
   requirements, then deterministic multinomial calibration that measures
   the gap between distribution-free guarantees and realized policy failure,
   followed by policy-specific categorical Chernoff certificates that recover
   most of that gap with a finite-sample analytic guarantee, then multivariate
   global-minimizer tilts that remove pairwise-event slack, followed by exact
   multinomial lattice enumeration for tractable representative cases, then
   simultaneous finite-path uncertainty propagation for the measured E83
   regression surface itself, followed by outcome-independent policy-geometry
   allocation of the familywise confidence budget and the future path-
   comparison budget, then a KKT-derived water-filled refinement for scarce
   confirmatory evidence and a floor-regularization audit that exposes an
   endpoint-only p90/minimax frontier, followed by an all-pair minimax
   negative control that demonstrates the cost of over-broad robustness,
   then a statistically valid pilot/holdout design for task-aware future
   sampling and an exact two-stage p-value combination that safely reuses
   pilot evidence, followed by a stage-weight audit that separates
   preregistered gains from post-hoc and pilot-dependent invalidity, and a
   dense fixed-weight plateau showing that `sqrt(2)` is within 0.154% of the
   condition-specific global optimum in the worst tested case, followed by
   numerical continuous-limit refinement that places the minimax multiplier
   at 1.428317 with 0.1292% worst relative-regret excess, then future-outcome
   sensitivity and 128-dataset convergence showing that the simpler
   `sqrt(2)` rule is more stable, including under structured correlated-risk
   stress tests.
9. **Independent A100 reproduction and compiler-clock diagnosis**: a second
   8xA100 allocation that reproduced representative fusion, attention,
   collective, correctness, and GEMM-ordering claims while exposing
   process-level compiler variability, followed by an NVML-instrumented and
   hardware-clock-locked diagnosis that isolated the E7/E110 discrepancy to
   unlocked 1140-to-1410 MHz DVFS clock transitions, ruled out Inductor cache
   drift (<0.11%), closed Amdahl's law with `torch.profiler` device-kernel
   shares, and mapped the crossover from full-step CUDA Graphs on launch-bound
   microbatches (3.36x) to `max-autotune-no-cudagraphs` fused Triton GEMM
   epilogues on 4,096-token steps (1.12-1.14x).
10. **Model-level synthesis & autonomous RSI kernel loops**: transformer
   training workloads that tie kernel-level findings to end-to-end throughput,
   memory, and held-out `val_bpb`, governed by `PROGRAM.md` and immutable
   SHA-256-verified `code/rsi_kernel_harness.py` (`3cf0f078...`) Vector-Pareto
   Ratchets (`rsi_ledger.tsv`, `valbpb_autoresearch_ledger.tsv`) across
   Block-Diffusion `BD3-LM` / `DiT` `adaLN-Zero` (`E127`, `E128`) and DeepSeek
   `mHC` (`Manifold-Constrained Hyper-Connections`, `arXiv:2512.24880`)
   multi-stream Birkhoff residual highways (`8.81x` Single-Scan `Norm-After-Matmul`
   + Division-Free On-Chip `Log-Sinkhorn-20`, `7.31x` Superfused Sublayer
   Sandwich beating DeepSeek's 2-kernel split, `L_r^* = 4` selective stream
   checkpointing compressing `1.07B` `n = 4` `mHC-FSDP2` overhead from `+142.95%`
   to `+10.94%–16.52%` while saving `43.01 GiB/rank`, and `-1.639 val_bpb` /
   `-1.005` matched-step `val_bpb` over larger `1-Stream`, `E129`), and DeepSeek
   `Engram: Conditional Memory via Scalable Lookup` (`arXiv:2601.07372`) paired
   with `M = 4` `mHC` (`5.38x` Zero-Sync Tile-Fused Segmented Backward +
   Pre-Forward `JIT-Lazy-Adam` `361,000x` closer to dense `Adam` than
   `SparseAdam`, `3.26x` Zero-Transpose `M = 4` `Eq. 6` Context-Gate + `Split-64`
   `ShortConv`, `2.57x` `8.39M`-row 8-GPU NVSwitch `Dedup-AllToAllv` + Layer-2
   Async Stream Prefetch cutting comm `74.3x` and saving `15.87 GiB/rank`,
   `-0.437 val_bpb` inside `12.0s`, and strict 4-point convex Sparsity-Allocation
   U-Curve `rho* = 0.75` beating `Pure MoE`, `Pure Engram`, `Iso-FLOPs Dense`,
   and `2.9x`-FLOPs `Wider Dense`, `E130`), and online nonlinear fast-weight
   Test-Time Training (`LaCT: Test-Time Training Done Right`, `arXiv:2505.23884`,
   ICLR 2026: fixing `a1600012888/LaCT`'s `78.59%` concatenated `[dW0; dW2]`
   `Muon` corruption, `4,582`-spill 3-accumulator SRAM overflow, and `0/0 = NaN`
   `L2Norm` backward via `9` zero-spill Triton kernels + `out_dtype=float32`
   Tensor-Core epilogues + `[3*B_h, R, K]` `FP32` `baddbmm` `Muon-NS5`, `2.30x`
   single-chunk & `2.60x` multi-chunk speedups beating `torch.compile` by `1.53x`
   with `-793 MiB` HBM, `5.21x` `64K` 8-GPU NVSwitch Within-Chunk `CP = 8` + P2P
   Halo `SWA` cutting comm `9.71x` and saving `17.52 GiB/rank`, and `99.15%`
   beyond-window associative recall inside `16.0s` while proving `Muon-NS5`
   vaults multi-chunk nonlinear recall from `7.36%` to `98.83%`, `E131`), and
   Moonshot Kimi depth-wise selective residual aggregation (`Attention Residuals`
   & `Block AttnRes`, `arXiv:2603.15031`: replacing community `Hydra-2P`'s
   slower-than-eager `(N*M)`-way `tl.atomic_add` backward and `28`-spill naive
   `BD=4096` tiles with `14/14` zero-spill Split-`M` lock-free kernels via
   `q_eff = w * gamma_key` + closed-form radial `RMSNorm` VJP + zero-`O1_bar`
   `RSIPhase1UnbindInterBlockFn` + superfused `Phase 2 + Pre-RMSNorm`, `6.38x`
   batched & `5.26x` 8-block sequential speedups beating `fullgraph=True`
   `torch.compile` by `1.50x` / `1.49x` while saving `2.48 GiB` / `23.19 GiB`
   HBM, `4.72x` 8-GPU NVSwitch Interleaved `1F1B` `P = 8, V = 4` Cross-Stage
   Virtual-Rank Block Caching cutting P2P comm `9.93x` vs `Full AttnRes` and
   `2.60x` vs Naive `Block` PP at `7.7e-7` `FP32` `Gate G9` parity, `-0.695
   val_bpb` inside `15.0s`, and matched-step Deep-`24`-Layer `Block AttnRes
   (N=8)` beating `Standard PreNorm`, `Full AttnRes`, `DenseFormer`, and
   `Highway` while suppressing `430.6x -> 49.2x` `PreNorm` dilution norm
   explosion, `E132`), and tokenizer-free end-to-end hierarchical dynamic
   chunking (`H-Net`, `arXiv:2507.07955`, `goombalab/hnet`: replacing upstream
   `dc.py`'s boundary-flipping `BF16` `F.normalize` router, `int(num_tokens.max())`
   graph break + dual `argsort`, and `BF16` `dt=-log(1-p)` `mamba_ssm` SSD
   surrogate with `17/17` zero-spill `max 32 regs` Round-7 RSI Triton kernels on
   unbounded `gridDim.x`, `6.31x` isolated DC & `2.36x` `Varlen FA2` stage
   speedups beating `torch.compile(dynamic=True)` by `2.63x` / `1.23x` while
   saving `1.82 GiB` / `1.33 GiB` HBM, `1.80x` 8-GPU NVSwitch Whole-Document
   Cost-LPT `AllToAllv` + Global Ratio Loss collapsing `2.96x -> 1.01x`
   straggler skew at `3.20x` less comm than raw bytes and `6.4e-8` 3-step `Gate
   G9` loss parity, `1.41x` steps + `-0.0891 val_bpb` / `-0.0871 test_bpb` inside
   a compression-governed `12.0s` loop catching `alpha = 0` dense-collapse
   reward hacking, and matched-step byte-level `H-Net` beating `SpaceByte`,
   `Static Stride-4`, `Unsmoothed DC`, and `Isotropic Byte SWA` under 1-byte
   phase shifts + character noise, `E133`), and data-dependent non-commutative
   Householder reflections + forgetting-gated softmax attention (`PaTH
   Attention`, `arXiv:2505.16381v2`, `fla-org/flash-linear-attention` paired
   with `FoX`, `arXiv:2503.02130`: replacing official `fla-org`
   `chunk_cumprod_householder_bwd_kernel`'s `3,732`-byte cubin register spills
   at `K = 128` and `@input_guard` `4x` `.contiguous()` copies with `22/22`
   zero-spill Round-8 RSI Triton kernels via `[BK=64, 64]` Block-Recurrent
   `UT`-Transform WY prep + Two-Phase Split-`K` `64+64` inter-chunk backward +
   superfused `ShortConv3+SiLU+L2Norm+BetaSigmoid` frontend, `11.54x` core &
   `10.05x` full-block speedups over Eager beating Official `fla-org` by `2.52x`
   geomean / up to `5.13x` at `K = 128` (`Track 1`) and `2.88x` (`Track 2`),
   `55.56x` vs All-Gather Eager / `8.33x` vs All-Gather Official `fla-org` /
   `3.49x` vs Dense-Matrix Ring on 8-GPU NVSwitch Compact WY `Ring PaTH-CP = 8`
   at `T_global = 8,192, K = 128` with `2.30x` lower HBM and `1.12e-6` `Gate G9`
   parity, `2.45x` steps + `-0.1682 val_bpb` / `-0.1632 test_bpb` inside `15.0s`,
   and matched-step `PaTH-FoX RSI` achieving `2.0285` clean / `2.1936` `2x`
   extrap `test_bpb` and `91.66%` / `89.96%` cyclic `S_4` state-tracking
   accuracy vs `43.8%–61.3%` for `RoPE`/`ALiBi`/`FoX`/`NoPE`, `E134`), and
   dynamic recursive latent-depth Transformers (`Mixture-of-Recursions`,
   `arXiv:2507.10524v2`, NeurIPS 2025, `raymin0223/mixture_of_recursions`:
   diagnosing `kv_sharing` `2.0x` full-sequence clone waste, `512 MiB` 4D masks,
   `logsumexp(dim=-1)` `[bs,k,1]` z-loss collapse, prefix-slice RoPE corruption,
   and `flash-attn` `v2.8.3` GQA empty-doc crashes, compiling `16/16` zero-spill
   Round-9 RSI Triton kernels vs `5,638` spills in monolithic `[64x128]` `dKV`
   for `1.80x` `Track 1` Rectangular Causal-Gather `Flash-MoR` and `1.52x`
   `Track 2` `Middle-Cycle` `MoR` stack speedups, cutting tied-weight `FSDP2`
   `AllGather` collectives `6 -> 1` (`6.0x`) with `MoR-LPT AllToAllv` across 8x
   A100 NVSwitch for `3.61x` over Naive `FSDP2`+Pad4D and `2.77x` pure `MoR-LPT`
   gain at `6.79e-7` `Gate G9` parity, `1.38x` steps + `-0.0967 val_bpb` /
   `-0.0948 test_bpb` inside a strict-causal `12.0s` loop, and matched-step
   `M4 (MoR Causal Aux-BCE + KV-Share + Flash-MoR RSI)` achieving `1.6787
   test_bpb`, `100.00%` `3-Hop` accuracy, and adaptive depth gap `Delta r =
   +1.00` vs `8.9%–9.6%` `3-Hop` accuracy under recursion-wise caching, `E135`),
   and common-mode attention-noise cancellation (`Differential Transformer V1 &
   V2`, `arXiv:2410.05258`, ICLR 2025 + Jan 2026 `DIFF V2`,
   `microsoft/unilm/tree/master/Diff-Transformer`: uncovering an undocumented
   `97.04%` relative L2 error GQA `repeat_kv` pre-split bug in
   `multihead_diffattn.py:80,86`, compiling `20/20` zero-spill Round-10 RSI
   Triton kernels via two-slice `64+64` staging and 1D `(r, m_idx)` loop
   flattening vs `384` spills in monolithic `[64x64x128]` `dKV` for `1.08x–1.16x`
   `Track 1` (`DiffAttn V2`) and `1.27x–1.59x` `Track 2` (`DiffAttn V1 + Fused
   SubLN`) speedups over official `unilm`, scaling `T_global = 16,384` across
   8x A100 NVSwitch via Dual-LSE `Ring DiffAttn-CP = 8` for `1.50x` over
   All-Gather Full-Sequence and `1.85x` over `2x` Separate Ring Passes at
   `1.19e-7` 3-step `FP32` `Gate G9` parity vs `727.0%` shard-local subtraction
   error, `2.62x` steps + `-0.5864 val_bpb` inside a `12.0s` loop, and
   matched-step `D4 (DiffAttn V2 RSI)` achieving `2.0426` distractor `test_bpb`
   and `100.00%` multi-needle retrieval accuracy vs `2.1311` / `98.44%` for
   `Standard GQA Softmax`, `2.4696` for `Wrong GQA chunk(2)`, and `3.4257` for
   `V1 No SubLN`, `E136`), and hypersphere representation learning & Riemannian
   tangent-retracted training (`nGPT: Normalized Transformer with Representation
   Learning on the Hypersphere`, `arXiv:2410.01131v2`, ICLR 2025, `NVIDIA/ngpt`:
   uncovering an upstream `model.py:40-58` `sinusoidal_pos.chunk(2, dim=-1)`
   non-orthogonal RoPE bug that breaks relative-position shift invariance by
   `max |Delta logit| = 1.1523` plus `20` unfused Chordal LERP kernels/block and
   `94%–98%` Euclidean AdamW radial step distortion, compiling `19/19` zero-spill
   Round-11 RSI Triton kernels vs `1,682` spills in monolithic `[64x64x128]`
   `dKV` for `24.31x` `Track 1` (`1.19x` vs `torch.compile`) and `5.16x`
   `Track 2` (`1.15x` vs `torch.compile`) geomean speedups over upstream
   `NVIDIA/ngpt`, scaling `T_global = 8,192` across 8x A100 NVSwitch via Cosine
   `Ring nGPT-CP = 8` + `ZeRO-1` row-sharded Riemannian weight retraction for
   `1.73x–2.60x` speedup at `6.25e-8` 3-step `FP32` `Gate G9` loss parity and
   `1.79e-7` unit-norm accuracy vs `1.8284` split-reduction-axis norm error,
   `1.75x` steps + `-1.3849 val_bpb` / `-1.4031 test_bpb` inside a `2.1s` loop,
   and matched-step `N4 (nGPT Pair-Isotropic SO(2) RoPE + Riemannian Tangent
   Retraction)` achieving `0.1112 test_bpb` and `99.65%` associative recall vs
   `3.8328` / `1.76%` for `Standard Pre-RMSNorm GPT`, `0.6170` / `26.52%` for
   `Upstream Broken chunk(2) RoPE`, and `0.1840` for `Euclidean Retraction`,
   `E137`; and Round-12 `PROGRAM.md` Sec. 15 extending the same frozen RSI
   evaluator to **`Log-Linear Attention` (`Log-Linear Mamba-2` & `Log-Linear
   Gated DeltaNet`, `arXiv:2506.04761v3`, Mar 2026, Guo, Yang, Goel, Xing, Dao
   & Kim; official repo `HanGuo97/log-linear-attention`)**, proving the
   3-instruction IEEE-754 Bitwise-XOR Fenwick Level Identity (`0` mismatches
   across `512x512` pairs, `0` bytes of `llut` in HBM vs upstream's hardcoded
   `/export/share/...` paths and $O(T^2)$ `[T,T]` `int64` `llut` allocation),
   eliminating `4,096` shared-memory `tl.atomic_add` ops/CTA onto `dl` and
   5D `gamma` allocations across `17/17` zero-spill Round-12 RSI Triton kernels
   vs `1,210` spills in a monolithic 8-level register-resident kernel for
   `9.36x` `Track 1` (`1.42x` vs `torch.compile`, `30.35x` vs quadratic
   materialized `hattention` at `T=4,096` with `13.85x` lower HBM) and `2.18x`
   `Track 2` (`1.08x` vs `torch.compile`) geomean speedups over upstream
   `hattention`, scaling `T_global = 8,192` across 8x A100 NVSwitch via the
   Hierarchical Fenwick-Tree Shard-Decoupling Theorem (`Fenwick-CP = 8`, `11`
   fine levels rank-local + `3` coarse inter-rank levels) for `4.21x` (`D=64`,
   `3.07x` less HBM) and `2.77x` (`D=128`, `1.85x` less HBM) speedup at
   `1.38e-5` 3-step `FP32` `Gate G9` loss parity vs `11.69%` error without
   coarse levels, `-3.1646 val_bpb` / `-3.1401 test_bpb` (`100.00%` recall)
   inside a `2.1s` loop, and matched-step `L4 (Log-Linear GDN RSI)` achieving
   `1.2027 test_bpb` and `100.00%` recall vs `6.1229` / `1.71%` for `Mamba-2
   SSD Fixed-State`, `5.6164` / `41.70%` for `H-GDN Wrong Aw Inverse`, and
   `4.7307` for `GDN Fixed-State`, `E138`; and Round-13 `PROGRAM.md` Sec. 16
   extending the same frozen RSI evaluator to **`MesaNet: Sequence Modeling by
   Locally Optimal Test-Time Training` (`arXiv:2506.05233v2`, von Oswald,
   Scherrer, Kobayashi, Versari, Yang et al., Google Paradigms of Intelligence /
   Google DeepMind / MIT CSAIL; official repo
   `fla-org/flash-linear-attention/tree/main/fla/ops/mesa_net`)**, diagnosing
   upstream `chunk_cg_solver_fwd.py`/`bwd.py`'s $x^{(0)} = 0$ cold start,
   commented-out Jacobi preconditioner (`naive.py:111`), and `+ 1e-5` biased CG
   denominators that stall 30-step CG at `1.294e-3` relative L2 error, plus
   `1,942` cubin spills at $K = V = 128$, and replacing them with `15/15`
   zero-spill (`49` specializations) Causal Query-Subtiled (`BM=16`) Warm-Start
   Jacobi-PCG kernels ($x^{(0)} = D_t^{-1} q_t$, `4.85e-8` error in 6 steps =
   `26,672x` higher accuracy in `5.0x` fewer iterations) and the Unified
   Dual-Gram Adjoint ($C_c = dO_c V_c^\top - d\tilde{Q}_c K_c^\top$) for
   `53.51x` `Track 1` (`31.91x` vs `torch.compile`, up to `2,482.58x` vs
   sequential `torch.linalg.solve`, `12.24x` lower HBM) and `19.40x` `Track 2`
   (`12.23x` vs `torch.compile`) geomean speedups over upstream `fla/ops/mesa_net`,
   scaling `T_global = 8,192` across 8x A100 NVSwitch via Dual-State Prefix-Scan
   `Mesa-CP = 8` for `55.38x` (`D=64`, `13.08x` less HBM) and `73.17x` (`D=128`,
   `18.22x` less HBM) speedup at `1.19e-7` 3-step `FP32` `Gate G9` loss parity
   vs `114.5%` error without prefix states, `-0.2231 val_bpb` / `-0.2374 test_bpb`
   (`99.02%` recall) inside a `2.1s` loop, and matched-step `M4 (MesaNet
   Anisotropic ShortConv RSI PCG-6)` achieving `0.0859 test_bpb` and `97.58%`
   recall vs `0.3350` / `89.94%` for `Scalar Lambda PCG`, `5.3852` / `8.69%` for
   `Online 1-Step DeltaNet`, and `5.9654` / `3.47%` for `Hebbian GLA 0-CG`,
   `E139`; and Round-14 `PROGRAM.md` Sec. 17 extending the same frozen RSI
   evaluator to **`Kimi Linear: An Expressive, Efficient Attention Architecture
   — Kimi Delta Attention (KDA)` (`arXiv:2510.26692`, Moonshot AI / Kimi Team,
   Zhang, Lin, Yao, Hu, Meng et al.; official repo
   `fla-org/flash-linear-attention/tree/main/fla/ops/kda`)**, diagnosing
   upstream `chunk_fwd.py:77`/`chunk_bwd.py:786`'s omitted `lower_bound`
   argument that causes `safe_gate=True` to produce `32,768 / 524,288` (`6.25%`)
   `NaN`s at $g_{t,d} = -7.0$, `arXiv:2510.26692` Listing 8(b)'s raw-$v_i$
   inter-chunk update bug (`18.06%` error), and upstream
   `chunk_kda_bwd_kernel_wy_dqkg_fused`'s `156–3,648` SM80 register spills across
   `100%` (`36/36`) of its autotuned specializations (`48,438` total spills),
   and replacing them with `19/19` zero-spill (`94` specializations) Round-14
   RSI Triton kernels via overflow-free non-positive causal sub-chunking (`0`
   `NaN`s) and split-`K`/`V` WY backward (`BK=32, BV=32` + transposed
   inverse-triangular adjoint $-A^\top dA_{\text{raw}} A^\top$) for `23.45x`
   `Track 1` (`22.93x` vs `torch.compile`, `528.86x` vs Recurrent, `18.19x`
   lower HBM) and `12.04x` `Track 2` (`10.90x` vs `torch.compile`, `14.00x`
   lower HBM) geomean speedups, scaling `T_global = 8,192` across 8x A100
   NVSwitch via Channel-Affine Monoid `KDA-CP = 8` for `31.93x` (`D=64`,
   `68.58x` less HBM) and `35.12x` (`D=128`, `97.91x` less HBM) speedup at
   `1.19e-7` 3-step `FP32` `Gate G9` loss parity vs `29.35%` error without
   handoff, `-4.4440 val_bpb` / `-4.4416 test_bpb` (`100.00%` recall) inside a
   `1.4s` loop, and matched-step `K4 (Kimi Linear 3:1 Hybrid KDA + MLA RSI)`
   achieving `0.0047 test_bpb` and `100.00%` recall vs `0.1795` / `99.77%` for
   `Pure Channel KDA RSI`, `3.7865` / `65.70%` for `Coarse Scalar GDN`,
   `5.0138` / `3.28%` for `Paper Listing 8(b) Bug`, and `5.0162` / `2.89%` for
   `GLA No Delta Rule`, `E140`), and **Round-15 `RWKV-7 "Goose" & Generalized
   Diagonal-Plus-Low-Rank (DPLR) Delta Rule` (`arXiv:2503.14456` / `2506.02475`,
   `fla-org/flash-linear-attention/tree/main/fla/ops/generalized_delta_rule/dplr`
   & `fla/ops/rwkv7`) RSI** (proving upstream `chunk_A_fwd`'s mid-chunk
   `b_offset = gi[BT//2]` produces `256` `NaN`s at $w = -2.85$ nats and that
   `100%` (`22/22`) of upstream's autotuned configs across `chunk_A_fwd`,
   `chunk_A_bwd`, and `wy_fast_bwd` spill `72–3,416` registers = `27,152` total
   spills, and replacing them with `21/21` zero-spill (`89` specializations)
   Round-15 RSI Triton kernels via boundary-anchored $16 \times 16$ sub-chunking
   (`0` `NaN`s), positive inverse-triangular adjoint $+\operatorname{StrictTril}
   (M^\top dM_{\text{raw}} M^\top)$, Euler homogeneous `dgk`, and superfused
   `TokenShift-5Lerp` / `kk + k_tilde + a + b` / `GroupNorm + <r, k_tilde*r_k>v +
   Gate` for `29.28x` `Track 1` (`25.00x` vs `torch.compile`, `774.06x` vs
   Recurrent, `19.72x` lower HBM) and `14.40x` `Track 2` (`12.72x` vs
   `torch.compile`, `14.32x` lower HBM) geomean speedups, scaling `T_global =
   8,192` across 8x A100 NVSwitch via Dual-Source Channel-Affine Monoid
   `DPLR-CP = 8` for `30.64x` (`D=64`, `51.65x` less HBM) and `32.56x` (`D=128`,
   `81.40x` less HBM) speedup at `9.06e-8` 3-step `FP32` `Gate G9` loss parity vs
   `54.48%` error without handoff, `-1.0388 val_bpb` / `-1.0249 test_bpb`
   (`100.00%` recall) inside a `1.4s` loop, and matched-step `R4 (RWKV-7 Goose
   Full Block RSI)` achieving `0.0023 test_bpb` and `100.00%` recall vs `0.0048` /
   `100.00%` for `Pure Channel DPLR RSI`, `2.2082` / `62.11%` for `Coarse Scalar
   DeltaNet`, `4.9269` / `7.42%` for `RWKV-6 Diagonal Decay No DPLR`, and `5.0437`
   / `3.36%` for `Conflated Inclusive ge Bug`, `E141`), and **Round-16
   `Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear
   Recurrences` (`PrecondKDA` & `ATK`, `arXiv:2604.21100`,
   `fla-org/flash-linear-attention/tree/main/fla/ops/precond_kda` &
   `fla/ops/atk`) RSI** (proving upstream `SAFE_GATE` sub-chunk anchor shift
   `b_gn = g[i_ti] - 7.5*GATE_SHIFT` produces `401,408` `NaN`s at $g = -7.2$ nats,
   5 upstream kernels spill `50,812` total registers across `70` configs, and
   `chunk.py:540-545` raises `NotImplementedError` for Context Parallelism, and
   replacing them with `24/24` zero-spill (`108` specializations, `max_regs = 168
   <= 168`) Round-16 RSI Triton kernels via 4-launch $16 \times 16$ sub-chunked
   `ATK`, pre-masked non-positive causal sub-chunking (`0` `NaN`s), row-subtiled
   `BM=16, BK=32` asymmetric WY backward, outer `strictTril` mask on
   $\mathrm{d}A_{kk}$, asymmetric Euler `dg`, and superfused `ShortConv1d+SiLU` /
   `QK-L2Norm+4-Gates` / `FusedRMSNorm+SigmoidGate` for `19.27x` `Track 1`
   (`15.21x` vs `torch.compile`, `630.47x` vs Recurrent, `15.94x` lower HBM) and
   `12.11x` `Track 2` (`10.56x` vs `torch.compile`, `11.93x` lower HBM) geomean
   speedups, scaling `T_global = 8,192` across 8x A100 NVSwitch via Coupled
   Curvature-Matrix `PKDA-CP = 8` for `31.24x` (`D=64`, `40.68x` less HBM) and
   `32.17x` (`D=128`, `63.03x` less HBM) speedup at `1.19e-7` 3-step `FP32` `Gate
   G9` loss parity vs `44.67%` error without handoff, `-4.0029 val_bpb` /
   `-4.0967 test_bpb` (`97.34%` recall) inside a `1.4s` loop, and matched-step
   `R4 (PrecondKDA Full Block RSI)` achieving `0.0786 test_bpb` and `98.28%`
   recall vs `2.0110` / `73.59%` for `Pure PrecondKDA + ATK RSI`, `4.5417` /
   `25.31%` for `Static Scalar RMS Scaling`, `5.0146` / `2.50%` for
   `Un-Preconditioned KDA`, and `5.0300` / `2.89%` for `Symmetric Akk Bug`,
   `E142`).
30. **Round-17 `Triadic Gated DeltaNet & Triadic Kimi Delta Attention: Kronecker
    Tensor-Product State Expansion & Hadamard-Factored UT/WY Linear Recurrences`
    (`Triadic-GDN` & `Triadic-KDA`, `arXiv:2609.36529`, FLA Team;
    `fla-org/flash-linear-attention/tree/main/fla/ops/triadic_gdn` &
    `fla/ops/triadic_kda`) RSI** (proving upstream `chunk_triadic_gdn` omits
    Dual-Sphere `L2Norm` so $\mathbb{E}[\|\kappa_t\|_2^2] = 252.11$ explodes
    $A_i = (I+A)^{-1}$ to `65,536` `NaN`s in `o` and `163,660` `NaN`s in
    gradients, spills `1,684` SM80 registers across `19/37` specializations,
    crashes `SM80` shared memory on `(K=128, V=128, FP32)` (`229,632 > 166,912`
    B), truncates `FP32` to `TF32` (`2.23e-3` error), and provides no Triton
    kernels for `Triadic-KDA`, and replacing them with `25/25` zero-spill (`78`
    specializations, `max_regs = 168 <= 168`) Round-17 RSI Triton kernels via
    fused Dual-Sphere `L2Norm` ($S^{E-1} \times S^{K-1}$), in-register
    `[K=64, BV=16, E=4]` state recurrence + parallel-over-chunks readout,
    $16 \times 16$ sub-chunked Kronecker-sum `Triadic-KDA` (`30.89x` speedup,
    `95.260 -> 3.081 ms`), and superfused `TriadicGDNBlockRSI` (`29.58x`
    speedup, `104.206 -> 3.491 ms`) for `1.77x` (`T=2048`) and `1.26x`
    (`T=4096`) `Triadic-GDN` speedup over upstream `fla` (`64.81x–75.09x` over
    eager chunk), `8,167.8x` `o` / `8,013.4x` `dq` `FP32` precision improvement
    (`max_fp32_rel_l2 = 6.38e-7`), `6.67x` more optimizer steps (`398 -> 2,678`)
    and `-0.1189 val_bpb` in a `25s` autoresearch loop, and `1.73x` vs 1x A100
    (`1.79x–1.81x` vs 8-GPU ring, `3,960.7 kTok/s` at $T_{\text{global}} =
    65,536$) with `0.000e+00` parity via 8-GPU NVSwitch Ulysses-WY
    `Triadic-CP = 8`, `E143`; and Round-18 `Momentum DeltaNet` (`arXiv:2605.05838`,
    FLA Team; official repo `fla-org/flash-linear-attention/tree/main/fla/ops/momentum_delta_rule`
    & `fla/layers/momentum_deltanet.py`), proving that upstream `chunk_momentum_delta_rule`
    has `0` CUDA Triton kernels and `0` backward Triton kernels (falling back to
    `naive.py` + `torch.autograd.grad` full forward replay at `188.14–636.80 ms`),
    spills `648–4,624` registers on its Ascend forward kernels while crashing
    `SM80` with `OutOfResources` at `K=128, V=128` (`219,672 > 166,912` B SMEM)
    and `K=256, V=256` (`416,280 > 166,912` B SMEM), raises `AssertionError` on
    `FP32`, and injects `beta=0` `(beta_p+1e-6).log()` momentum drift (`2.79e-3`),
    and replacing it with `26/26` zero-spill (`190` specializations, `max_regs =
    168 <= 168`) Round-18 RSI Triton kernels via 3-RHS coupled WY forward,
    adjoint factorization `dX = Ai^T dV_new`, $O(B_T)$ 1D vector-recurrence gate
    adjoint, and superfused `MomentumDeltaNetBlockRSI` (`51.69x` speedup vs
    upstream layer, `200.30 -> 3.966 ms`, `38.27x` vs `A0`) for `84.76x`
    (`T=2048`), `78.91x` (`T=4096`), and `75.27x` (`T=8192`) kernel Fwd+Bwd
    speedup over upstream `fla` (`27.82x` at `K=128`, `14.75x` at `K=256`),
    `12.33x` more optimizer steps (`188 -> 2,318`) and `-1.1277 val_bpb` in a
    `25s` autoresearch loop, and `5.15x` vs 1x A100 (`5.32x` vs 8-GPU ring,
    `738.9 kTok/s` at $T_{\text{global}} = 32,768, H = 128$, `64.4%` scaling
    efficiency) with `0.000e+00` parity via zero-copy `B=1` 8-GPU NVSwitch
    Ulysses-WY `Momentum-CP = 8`, `E144`; and Round-19 `Gated KalmaNet (GKA)`
    (`arXiv:2511.21016`, Songlin Yang, Yu Zhang, Zhiyuan Li; official repo
    `fla-org/flash-linear-attention/tree/main/fla/ops/gka` & `fla/layers/gka.py`),
    proving that upstream `fla/ops/gka` spills in `193/219` (`88.1%`) compiled
    specializations (`1,519,648` B total stack, `ptxas` `regs=32` allocator
    collapse up to `47,864` B stack/thread, `262,144 > 166,912` B SMEM overflow,
    and `82.3–200.1 s` cold compile from unrolling `NUM_ITER=30` into a `1.7 MB`
    PTX file) and divides `0.0 / 0.0 = NaN` when $\|H_t\|_F = 0$ so a single
    $k_0 = 0$ token contaminates `32,768 / 65,536` (`50%`) of `dk` to `NaN`, and
    replacing it with `21/21` zero-spill (`121` specializations, `max_regs = 168
    <= 168`) Round-19 RSI Triton kernels via query-subtiled (`BM=32/16`)
    runtime-loop Chebyshev solver, decoupled `FP32` Frobenius norm (`3,689.5x`
    `FP32` output accuracy improvement, `0` `NaN`s), slab-streamed implicit
    ridge backward (`3.32x` backward speedup at `K=128`), and superfused
    `GatedKalmaNetBlockRSI` (`2.41x` at `K=64`, `2.19x` at `K=128`) for `1.98x`
    (`K=64, T=2048`) to `2.61x` (`K=128, T=2048`) `chunk_gka_rsi` Fwd+Bwd speedup,
    `2.23x` more optimizer steps (`437 -> 973`) and `-1.0849 val_bpb` in a `15s`
    autoresearch loop, and `3.43x` vs 8-GPU ring (`3.28x` vs 1x A100, `4,505.6
    kTok/s` at $T_{\text{total}} = 32,768, K = 128$) with `0.000e+00` parity via
    zero-copy `B=1` 8-GPU NVSwitch Ulysses `GKA-CP = 8`, `E145`; and Round-20
    `Gated DeltaNet-2 (GDN-2), Gated Oja Rule 2 (Oja-2), & Gated Slot Attention
    2 (GSA-2)` (`NVlabs/GatedDeltaNet-2` `fla/ops/gdn2`, `fla/layers/gdn2.py`
    and `fla-org/flash-linear-attention` `fla/ops/gated_oja_rule`,
    `fla/ops/gated_oja_rule2`, `fla/layers/gsa2.py`), proving that upstream
    `chunk_gdn2`, `chunk_gated_oja_rule`, and `chunk_gated_oja_rule2` spill
    across `22` compiled specializations (`max_regs = 255, max_spills = 366` in
    `chunk_gdn2_bwd_kernel_wy_dqkg_fused`, `140` spills in `oja2_bwd_kernel_wy`,
    `86` spills in `chunk_gdn2_fwd_kernel_inter_solve_fused`) and launch 6
    separate WY matrix kernels in `Oja-2`, and replacing them with `37/37`
    zero-spill (`76` specializations, `max_regs = 162 <= 168`) Round-20 RSI
    Triton kernels via 3-pass split WY backward, $16 \times 16$ block-subtiled
    coupled $(A_{bv}, W, U)$ `Oja-2` WY solve, 16-token sub-sliced `dhu`
    backward, and superfused `GDN2BlockRSI` (`1.194x` speedup, `15.212 -> 12.736
    ms`) & `GSA2BlockRSI` (`1.226x` at $M=64$, `1.263x` at $M=128$, `1.315x` at
    $M=256$ `41.547 -> 31.604 ms`) for up to `1.497x` `chunk_gdn2_rsi` Fwd+Bwd
    (`1.613x` Bwd) and `1.469x` `chunk_gated_oja_rule2_rsi` Fwd+Bwd (`18.583 ->
    12.646 ms` at $M=256, T=8192$) speedups, `1.770x` more optimizer steps
    (`474 -> 838`, `32.3 -> 57.2 kTok/s`) and `-0.0180` extra `val_bpb` gain in a
    `30s` autoresearch loop, and `2.860x–3.318x` speedup over 8-GPU dual-state
    ring across $T_{\text{total}} = 16,384..65,536$ with `0.0` max absolute
    difference via zero-copy 8-GPU NVSwitch Ulysses `GSA2-CP = 8`, `E146`; and
    Round-21 `Gated DeltaProduct (GDP): Multi-Householder Product Linear
    Recurrences for Rotational & Parity State Tracking` (`arXiv:2502.10297`,
    Siems et al.; official repo `fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_product`
    & `fla/layers/gated_deltaproduct.py`), proving that upstream
    `chunk_gated_delta_product` leaves `# TODO: optimize the backward pass like
    the forward pass` (`chunk.py:204`) unresolved by allocating zero-padded
    length-$T \cdot N_h$ `q_new, do_new, g_new` tensors in HBM (`50%` waste at
    $N_h=2$, `66.7%` at $N_h=3$, `75%` at $N_h=4$), spills across `21` compiled
    specializations (`450` total spills, `max_regs = 255, max_spills = 96`),
    raises `AssertionError` on `FP32`, and injects `cu_seqlens` varlen indexing
    across dense 3D batches, and replacing it with `22/22` zero-spill (`90`
    specializations, `max_regs = 168 <= 168`) Round-21 RSI Triton kernels via
    Universal `BT = 64` virtual chunking, rectangular `[BQ_PAD, 64]` readout &
    native stride-$N_h$ backward (`_gdp_fwd_o_rect_kernel`,
    `_gdp_bwd_dv_local_rect_kernel`, `_gdp_bwd_dhu_rect_kernel`,
    `_gdp_bwd_state_dqkw_pass1_rect_kernel`), 1-matrix WY adjoint identity in
    `_gdp_bwd_wy_dA_pass2c_kernel`, Euler repeated squaring $(I+N)(I+N^2)(I+N^4)(I+N^8)$
    in `_gdp_fwd_solve_tril_16x16_kernel`, and superfused
    `GatedDeltaProductBlockRSI` (`Stage A4`: `2.07x` at $N_h=2, V=128$, `2.15x`
    at $N_h=2, V=256$, `2.06x` at $N_h=3, V=256$, `2.01x` at $N_h=4, V=256$) for
    up to `1.94x` `chunk_gated_delta_product_rsi` Fwd+Bwd speedup (`1.95x` Fwd,
    `1.93x` Bwd), `1.86x` more optimizer steps (`894 -> 1,665`, `61.0 -> 113.6
    kTok/s`) and `-0.0063` extra `val_bpb` reduction in a `30s` autoresearch
    loop, and `2.96x–3.22x` (`3.13x` mean) speedup over 8-GPU sequential ring
    state-passing across $T_{\text{total}} = 16,384..65,536$ ($T_{\text{virt}}$
    up to `131,072`) with `0.0` max absolute difference via zero-copy 8-GPU
    NVSwitch Ulysses `GDP-CP = 8`, `E147`; and Round-22 `Cyclic Flow Attention
    (CyFA): Linear Sequence Modeling with Relative-Time-Partitioned Memory`
    (`arXiv:2609.36259`, Chen et al.; official repo
    `fla-org/flash-linear-attention/tree/main/fla/ops/cyfa` &
    `fla/layers/cyfa.py`), proving that upstream `chunk_cyfa` spills up to
    `4,932` registers across `69` compiled specializations (`58,354` total
    spills, `max_regs = 255`, `max_smem = 147,456` B), uses global
    `tl.atomic_add` on `dg_force`, recomputes `chunk_fwd_h` 4 times per step
    plus `qk_rmsnorm_fwd` and `_point_write` twice in backward even at
    `checkpoint_level == 0`, and truncates `FP32` inputs to `TF32`
    (`max_rel_l2 = 6.77e-3`), and replacing it with `19/19` zero-spill (`132`
    specializations, `max_regs = 168 <= 168`, `max_smem = 40,960` B) Round-22
    RSI Triton kernels (`max_fp32_rel_l2 = 7.78e-6` = `869.9x` more accurate)
    via strip-tiled `BC = 64/32` 2-pass modal readout, `BR = 32` sequential
    `dq -> dk` live-range decoupling with atomic-free `dg_part` reduction,
    saved forward states `(q_norm, k_norm, write, h_k, h_v)`, fused `BT = 256`
    `(4, 64)` clock/decay scans, and superfused `CyclicFlowAttentionBlockRSI`
    (`Stage A4`: `1.58x–1.60x` speedup over `Stage A0`) for up to `1.93x`
    `chunk_cyfa_rsi` Fwd+Bwd speedup (`1.96x` Bwd, `1.87x` Fwd; `1.79x` mean
    Fwd+Bwd across 12 shapes), `1.441x` more optimizer steps (`895 -> 1,289`,
    `60.9 -> 87.7 kTok/s`) and `-0.1769` extra `val_bpb` reduction (`8.2405 ->
    1.3391` vs `1.5160`) in a `30s` autoresearch loop, and `2.04x–3.31x`
    (`2.74x` mean) speedup over 8-GPU 3-state `(h_k, h_v, lambda_t)` ring
    across $T_{\text{total}} = 16,384..65,536$ with `0.0` max absolute
    difference via zero-copy 8-GPU NVSwitch Ulysses `CyFA-CP = 8`, `E148`; and
    Round-23 `Parallax: Parameterized Local Linear Attention` (`arXiv:2605.29157`,
    Yifei Zuo et al.; official repo `fla-org/flash-linear-attention/tree/main/fla/ops/parallax`
    & `fla/layers/parallax.py`), proving that upstream `fla/ops/parallax` spills
    across `22/42` compiled specializations (`34/42` with `regs > 168`, `max_regs =
    255, max_spills = 26, total_spills = 190, max_smem = 91,392` B), raises
    `TypeError` on `FP32` inputs, reloads `K, V` tiles $G\times$ redundantly
    under GQA while launching a separate `bwd_preprocess` kernel and two
    `einops.reduce` kernels, and calls `r, _ = self.rotary(r, r)` in
    `fla/layers/parallax.py:134` (`14` periphery kernel launches per step), and
    replacing it with `10/10` zero-spill (`68` specializations, `max_regs = 168
    <= 168`, `max_smem = 49,152` B) Round-23 RSI Triton kernels
    (`max_fp32_rel_l2 = 8.46e-6`, `max_bf16_rel_l2 = 3.08e-3`) via GQA-packed
    `[BM = 16, D]` query tiles, single-comparison `uint32` causal/window mask
    `(row - col).to(uint32) < W_BOUND`, live-range instruction scheduling, fused
    `(delta_t, delta_b)` preprocess in `bwd_dqr`, single-pass GQA `dKV`
    reduction, and superfused `ParallaxBlockRSI` (`Stage A4`: up to `2.21x`
    speedup over `Stage A0`, `1.51x` mean) for `12/12` kernel shape wins up to
    `2.22x` (`1.54x` Train Fwd+Bwd, `1.67x` Bwd, `1.42x` Train Fwd, `1.39x`
    mean), `2.166x` more optimizer steps (`1,676 -> 3,629`, `114.3 -> 247.6
    kTok/s`) and `-0.0293` extra `val_bpb` reduction (`8.2497 -> 0.9557` vs
    `0.9850`) in a `30s` autoresearch loop, and `1.98x–2.60x` (`2.27x` mean)
    speedup over 8-GPU coupled 5-tuple online-softmax `(m, d1, d2, O1, O2)` ring
    across $T_{\text{global}} = 16,384..65,536$ with `0.0` max absolute
    difference via zero-copy 8-GPU NVSwitch Ulysses `Parallax-CP = 8`, `E149`;
    and Round-24 `Wall Attention: Length Generalization With Diagonal Gates`
    (`Tilde Research`, Timor Averbuch & Dhruv Pai; official repo
    `fla-org/flash-linear-attention/tree/main/fla/ops/wall_attn` &
    `fla/layers/wall_attn.py`), proving that upstream `fla/ops/wall_attn` spills
    across `961 / 1,009` (`95.24%`) compiled specializations (`5,266,560 B` =
    `5.27 MB` total stack spills, `max_regs = 255, max_stack = 36,808 B/thread`,
    `71` SMEM overflows up to `295,424 B`), returns `g.grad = None, g_scalar.grad
    = None` in `naive_wall_attn` while truncating `FP32` `tl.dot` to `TF32`
    (`rel_l2_o = 1.03e-3, rel_l2_dg = 3.33e-3`), spends `86%` of decode time inside
    7-op eager `build_wall_kv_cache`, and launches 6 separate input GEMMs +
    unfused periphery in `WallAttention.forward`, and replacing it with `0`-spill
    (`max_regs = 168 <= 168`, `max_smem = 40,960 B`) Round-24 RSI Triton kernels
    (`FP32 rel_l2_o = 3.23e-6` = `317.9x` more accurate, `rel_l2_dg = 1.94e-5` =
    `166.3x` more accurate) via Fixed-Anchor ($R_q = P_{i_t \cdot BT}$ & $R_k =
    P_{i_t \cdot BT}$) single-loop accumulation, Softmax Shift-Invariance
    ($\text{dc}_{q, t} = \sum_s dS_{t, s} \equiv 0$), single-launch
    `_wall_build_kv_cache_rsi_kernel` (`8.25x–9.55x` faster KV-cache build,
    `3.67x–4.98x` faster end-to-end decode, `0.0` bitwise diff), and superfused
    `WallAttentionBlockRSI` (`1.56x` at $D=1024, K=64$ & `1.27x` at $D=2048,
    K=128$) for `4/4` training shape wins (`1.71x` Fwd, `1.21x` Bwd, `1.29x`
    Total, `2.74x` vs unpruned $K=128$ spilling cliff), `6.08x` ($16\text{K}$) &
    `7.15x` ($32\text{K}$, `88.3%` parallel scaling efficiency over 1 GPU, `0.0`
    `rel_l2` discrepancy) via fused 8-GPU NVSwitch Ulysses `Wall-CP = 8`, and
    `1.47x` ($D=1024$) / `1.21x` ($D=2048$) more optimizer steps with `-0.9920`
    extra `val_bpb` reduction (`0.1438` vs `1.1358`, `7.90x` lower final `val_bpb`)
    in a `12s` autoresearch loop, `E150`), and **Round-25 RSI (`v142`):
    `DeltaFormer: Understanding Transformer from the Perspective of Associative
    Memory` (`arXiv:2505.19488`, Jusen Du, Weigao Sun, Disen Lan, Jiaxi Hu, Yu
    Cheng; `fla-org/flash-linear-attention/tree/main/fla/ops/deltaformer` &
    `fla/layers/deltaformer.py`)** (eliminating upstream `torch.linalg.solve_triangular`'s
    `68.8%` GPU-time bottleneck + `160` `.transpose(0,1).contiguous()` copies,
    `24 / 104` (`23.08%`) spilling specializations with `1,606` total spills /
    `max_regs = 255, max_spills = 274` in `parallel_deltaformer_bwd_kernel_qk`,
    unmasked OOB `tl.load(u_blk_ptr)`, `FP32` `RuntimeError` crash, GQA
    `AssertionError` crash, and `past_key_values` `NotImplementedError` via
    `13/13` `0`-spill (`195` specializations, `max_regs = 168 <= 168`, `max_smem
    = 18,432 B`) Round-25 RSI Triton kernels with in-place Tensor-Core Euler-16
    unit-lower-triangular solve `_df_tril_solve_fwd/bwd_rsi_kernel` (`12.7x`
    faster than `solve_triangular`), split `dq`/`dk` backward kernels, and the
    exact Diagonal-LSE Stage-2 Collapse identity $P^{(2)}_{t, s} = (1 - \alpha_t)
    P^{(1)}_{t, s}$, achieving `216.9x` lower `FP32` error (`8.36e-6` vs
    `1.81e-3`), `8/8` training shape wins up to `4.84x` Fwd, `2.76x` Bwd, `3.48x`
    Total, `4.08x–7.28x` faster single-pass cached decode (`deltaformer_decode_rsi`),
    `2.81x` ($D=1024, K=64$) & `2.14x` ($D=2048, K=128$) `DeltaFormerBlockRSI`
    speedup, `3.71x` ($16\text{K}$) & `2.56x` ($32\text{K}$) 8-GPU NVSwitch
    Ulysses `DeltaFormer-CP = 8` speedup, and `2.70x` ($D=1024$) / `2.05x`
    ($D=2048$) more optimizer steps with `-6.7788` ($14.55\times$ lower) &
    `-2.7225` ($7.85\times$ lower) final `val_bpb` in a `12s` autoresearch loop,
    `E151`), and **Round-26 RSI (`v143`): `Stick-Breaking Attention`
    (`arXiv:2410.17980`, ICLR 2025 — Shawn Tan, Yikang Shen, Songlin Yang,
    Aaron Courville, Rameswar Panda;
    `fla-org/flash-linear-attention/tree/main/fla/ops/stickbreaking_attn` &
    `fla/layers/stickbreaking_attn.py`)** (eliminating upstream
    `parallel_stickbreaking_attn_bwd_kernel_dq`'s redundant 2-pass causal key
    scan + 32-way uncoalesced `[B, T, HQ, NS]` HBM scatter/gather + 2-tuple
    `tl.associative_scan`, `29 / 108` (`26.85%`; `36 / 114` = `31.58%` full
    grid) spilling specializations with `1,294..2,440` total spills / `max_regs =
    255, max_spills = 154..314`, scalar SIMT `stickbreaking_attn_decoding`
    reloading the KV cache $T_Q \times G$ times per KV head, and `FP32`
    `TypeError` crash in `StickBreakingAttention.forward` via `7/7` `0`-spill
    (`44` specializations, `max_regs = 168 <= 168`, `max_smem = 45,312 B`)
    Round-26 RSI Triton kernels using the Reverse-Stick Subtraction Identity
    $\text{log\_rem}(i, i_k) = L_i - \sum_{m=0}^{i_k} B_{i, m}$, coalesced
    `[B*HQ, NS, T]` snapshot layout, predicated inline PTX `_neg_softplus2_ptx`
    (`-29` regs/site), and 2-subblock `BS = 16` `dKV` decomposition at $K = 128$,
    achieving `466.2x` lower `FP32` error (`3.16e-6` vs `1.47e-3`), `8/8`
    training shape wins in Bwd & Total up to `1.29x` Fwd, `1.51x` Bwd, `1.43x`
    Total, `1.91x–32.99x` faster Tensor-Core GQA decoding
    (`stickbreaking_attn_decoding_rsi`), `1.53x–1.56x` ($D=1024, K=64$) & `1.41x`
    ($D=2048, K=128$) `StickBreakingAttentionBlockRSI` speedup, `7.21x` ($16\text{K}$)
    & `7.17x` ($32\text{K}$, `7.70x` scaling vs 1 GPU = `96.2%` parallel
    efficiency, `0.00e+00` `rel_l2`) 8-GPU NVSwitch Ulysses `StickBreaking-CP = 8`
    speedup, and `1.49x` ($D=1024$, `-0.0754 bpb`) / `1.37x` ($D=2048$,
    `-0.8320 bpb`) in a `12s` autoresearch loop, `E152`), and **Round-27 RSI
    (`v144`): `Test-Time Training (TTT-Linear): Learning to (Learn at Test Time):
    RNNs with Expressive Hidden States` (`arXiv:2407.04620`, ICML 2024 / JMLR
    2025 — Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan
    Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori
    Hashimoto, Carlos Guestrin;
    `fla-org/flash-linear-attention/tree/main/fla/ops/ttt`)** (eliminating
    upstream `fla/ops/ttt`'s `78 / 119` (`65.55%`) spilling specializations with
    `85,682` total register spills / `max_regs = 255, max_spills = 4,614` +
    `FP32 K=128` `295,176 > 166,912 B` `OutOfResources` crash, redundant full
    forward recurrence replay `chunk_ttt_linear_bwd_h` in backward + `3`
    `.clone()` calls in `norm_residual`, uncoalesced `[BV, BK]` state loads,
    `1.34e-3` `FP32` `TF32` backward truncation, and `1.08e+00` tail-padding bug
    on `final_state_bias` (`hbt`) when $T \bmod 16 \ne 0$ via `9/9` `0`-spill
    (`62` specializations, `max_regs = 168 <= 168`, `max_smem = 73,728 B`)
    Round-27 RSI Triton kernels using SM80 `MMA <-> DotOperand0` transposed state
    congruence `b_ht = h^T [V, K]`, the Self-Adjoint LayerNorm Identity
    $\langle \text{dv\_new}, v_{\text{new}} \rangle \equiv \langle \text{dy}, y
    \rangle$, factored intra-chunk dual attention, and in-kernel output
    `LayerNorm + residual`, achieving `3,153x` lower `FP32` backward error
    (`4.24e-7` vs `1.34e-3`), `8/8` training shape wins in Fwd, Bwd & Total up
    to `4.14x` Fwd, `2.16x` Bwd, `2.47x` Total vs `chunk_ttt_linear` (`6.58x` vs
    `fused_chunk_ttt_linear`), `11.00x–11.43x` faster single-pass online
    test-time adaptation decoding (`ttt_decode_adapt_rsi`), `1.51x` ($D=1024,
    K=64$) & `1.73x` ($D=2048, K=128$) `TTTLinearBlockRSI` speedup, `2.18x–2.70x`
    8-GPU NVSwitch Ulysses `TTT-CP = 8` speedup over Ring (`0.00e+00` `rel_l2`),
    and `1.56x` ($D=1024$, `-0.2465 bpb`) / `1.53x` ($D=2048$, `-0.1855 bpb`) in
    a `12s` autoresearch loop, `E153`), and **Round-28 RSI (`v145`): `Titans:
    Learning to Memorize at Test Time` (`arXiv:2501.00663`, Google Research —
    Ali Behrouz, Peilin Zhong, Vahab Mirrokni;
    `fla-org/flash-linear-attention/tree/main/fla/ops/titans`)** (replacing
    upstream `fla/ops/titans`'s `0` Triton kernels and `288` sequential Python
    loop steps per 16-token chunk taking `3,304.3 ms` at $B=2, T=512, K=64$ and
    fixing its `@pytest.mark.skipif(True, reason='FIXME')` tail-padding bug at
    $T = 37$ that causes `40/296` (`13.51%`) `NaN`s in `dalpha`, `8/296`
    (`2.70%`) `NaN`s in `deta`, and `1.22e-1` error on `final_state` plus
    `cal_G_log` `FP64 -> FP32` truncation via `12/12` `0`-spill (`137`
    specializations, `max_regs = 128 <= 168`, `max_smem = 18,432 B`) Round-28
    RSI Triton kernels using the Bi-Triangular Decay Factoring Theorem
    $G = (L^{(\beta)} L^{(\eta)}) \odot \theta^\top, f = L^{(\beta)} m$,
    transposed dual-state `(M^T, S^T)` `BK=16` slab streaming, precomputed
    `y_grad` in `bwd_dv_local`, split `bwd_dq`/`bwd_dk`, and in-kernel output
    `LayerNorm + residual`, achieving `FP32 max_rel_l2 = 7.77e-7`, `8/8`
    training shape wins in Fwd, Bwd & Total up to `121.63x` Fwd, `216.78x` Bwd,
    `181.98x` Total vs `vectorized_pytorch_chunk_titans` (`3,080.7x` vs upstream
    `chunk_titans_linear_ref`), `37.66x–43.58x` faster single-pass online
    test-time memorization decoding (`titans_decode_memorize_rsi`, `5.17x–5.43x`
    vs `chunk_titans_linear_rsi`), `129.30x` ($D=1024, K=64$) & `53.97x`
    ($D=2048, K=128$)     `TitansBlockRSI` speedup, `1.59x–1.60x` 8-GPU NVSwitch Ulysses `Titans-CP =
    8` speedup over Ring (`0.00e+00` `rel_l2`), and `35.42x` ($D=1024$,
    `-3.2291 bpb`) / `25.33x` ($D=2048$, `-0.7225 bpb`) in a `12s` autoresearch
    loop, `E154`), and **Round-29 RSI (`v146`): `Comba: Improving Bilinear RNNs
    with Closed-loop Control` (`arXiv:2506.02475` — Jiaxi Hu, Yongqi Pan, Jusen
    Du, Disen Lan, Xiaozhe Ren, Yibo Wang, Xuezhi Cao, Yunsheng Wu, Yu Zhang,
    Songlin Yang; `fla-org/flash-linear-attention/tree/main/fla/ops/comba` &
    `fla/layers/comba.py`)** (fixing upstream `Comba.forward`'s Inner-Decay
    `L2Norm` Erasure Bug where `l2norm_fwd(k * self.decay.sigmoid())` erases
    $\sigma(\text{decay}_h) > 0$ with `rel_diff = 8.32e-7` and `decay.grad
    max_abs = 9.56e-6` when `use_output_correction=False`, and eliminating
    upstream `fla/ops/comba`'s `16 / 53` (`30.19%`) spilling specializations
    with `2,120` total register spills / `max_regs = 255, max_spills = 594` in
    `prepare_wy_repr_bwd_kernel` / `max_smem = 163,856 B`, redundant
    $64\times 64\times K$ `b_A` recomputation, `3.82e-3` `FP32` `TF32` drift, and
    redundant backward forward-replay via `25/25` `0`-spill (`212`
    specializations, `max_regs = 168 <= 168`, `max_smem = 24,576 B`) Round-29
    RSI Triton kernels using the Euler Homogeneous Degree-1 WY & Score Adjoint
    Identities $\sum_j \mathrm{d}M_{ij} M_{ij} = \langle \mathrm{d}p_i^{(M)},
    p_i \rangle, -\sum_i \mathrm{d}M_{ij} M_{ij} = -\langle \mathrm{d}k_j^{(M)},
    k_j \rangle$, combined exclusive/inclusive reverse cumsum, and unified
    `p_sub=0..7` `bwd_dhu`, achieving `1,272x` lower `FP32` error (`3.00e-6` vs
    `3.82e-3`), `8/8` training shape wins in Fwd, Bwd & Total up to `2.82x` Fwd,
    `2.12x` Bwd, `2.30x` Total vs upstream `chunk_comba`, `2.02x–2.20x` faster
    online decoding vs `fused_recurrent_comba` (`16.74x–18.49x` vs
    `chunk_comba`), `2.07x` ($D=1024, K=64$) & `1.98x` ($D=2048, K=128$)
    `CombaBlockRSI` speedup, `2.59x–2.88x` 8-GPU NVSwitch Ulysses `Comba-CP = 8`
    speedup over Ring (`0.00e+00` `rel_l2`), and `1.94x` ($D=1024$, `-1.5705
    bpb`) / `1.75x` ($D=2048$, `-0.4966 bpb`) in a `12s` autoresearch loop,
    `E155`), and **Round-30 RSI (`v147`): `Gated KalmaNet (GKA): A Fading
    Memory Layer Through Test-Time Ridge Regression` (`arXiv:2511.21016`,
    `fla-org/flash-linear-attention/tree/main/fla/ops/gka` &
    `fla/layers/gka.py`)** (fixing upstream `chunk_gka`'s 16-bit
    `SimpleGLABackwardGuard` `.backward()` crash on `K < 64` or `V < 32`,
    eliminating upstream `fla/ops/gka`'s `29 / 58` (`50.0%`) spilling
    specializations with `76,526` total register spills / `max_regs = 255,
    max_spills = 10,028` in `gka_bwd_dk_intra` / `max_smem = 114,688 B` and
    `1.99e-3` `FP32` `TF32` drift via `26/26` `0`-spill (`225` specializations,
    `max_regs = 168 <= 168`, `max_smem = 69,632 B`) Round-30 RSI Triton kernels
    using Sub-Chunk Strip-State Folding `BS = 32` at $K \le 64$, post-loop
    live-range decoupled `_gka_bwd_dqkg_gla_rsi_kernel` (`4.24x` faster),
    `BS = 16` strip backward, 3-Quadrant Symmetric Gramian online decoding
    (`b_h00, b_h01, b_h11`), and the exact Chebyshev IFT Adjoint `FP64` oracle,
    achieving `3,785x` lower `FP32` error (`5.26e-7` vs `1.99e-3`), `8/8`
    training shape wins in Fwd, Bwd & Total up to `2.74x` Fwd, `4.74x` Bwd,
    `3.60x` Total vs upstream `chunk_gka`, `1.28x–4.90x` faster online decoding
    vs `fused_recurrent_gka` (`2.49x–9.97x` vs `chunk_gka`), `1.98x` ($D=1024,
    K=64$) & `2.57x` ($D=2048, K=128$) `GatedKalmaNetBlockRSI` speedup,
    `4.56x–16.18x` 8-GPU NVSwitch Ulysses `GKA-CP = 8` speedup over Ring
    (`3.36x–15.22x` vs 1-GPU upstream, `0.00e+00` `rel_l2`), and `1.80x`
    ($D=1024$, `-0.4905 bpb`) / `2.31x` ($D=2048$, `-0.3242 bpb`) in a `12s`
    autoresearch loop, `E156`), and **Round-31 RSI (`v148`): `DeepSeek Sparse
    Attention (DSA), Online KL-Distilled Lightning Indexer, & GLM-5.2
    IndexShare` (`arXiv:2512.02556` / `arXiv:2602.15763`,
    `fla-org/flash-linear-attention/tree/main/fla/ops/dsa`)** (resolving
    upstream `fla/ops/dsa/naive.py`'s `0`-Triton-kernel `2,209.4 MiB` dense
    `FP32` tensor explosion and complete gradient disconnect `q_idx.grad =
    k_idx.grad = w_idx.grad = None` via `19/19` `0`-spill (`122`
    specializations, `max_regs = 168 <= 168`, `max_smem = 25,600 B`) Round-31
    RSI Triton kernels featuring a Flattened-`(BM * HI)` Register-Resident MQA
    Lightning Indexer, a 64-Bit Bitmask Table (`bitmask [B, T, T // 64]`
    `uint64`) + `tile_active` GQA-Packed `Flash-DSA`, Online Teacher-Student KL
    Distillation (`FP32 max_rel_l2 = 5.97e-7 < 1e-4`), Single-Pass Multi-Warp
    Sparse Decoding (`dsa_decode_rsi`), and `GLM-5.2 IndexShare` cross-layer
    index reuse, achieving `8/8` training shape wins in Fwd, Bwd, Total & Peak
    HBM up to `5.22x` Fwd, `6.20x` Bwd, `5.76x` Total & `10.98x` HBM reduction
    vs upstream `naive_dsa`, `89.26x–761.31x` faster single-pass sparse
    decoding vs `naive_dsa` (`11.21x–150.70x` vs `dsa_rsi` chunk), `3.04x`
    ($D=1024, K=64$) & `2.33x` ($D=2048, K=128$) 2-layer
    `DeepSeekSparseAttentionBlockRSI` + `GLM-5.2 IndexShare` speedup,
    `13.42x–25.93x` 8-GPU NVSwitch Ulysses `DSA-CP = 8` speedup over Ring
    (`13.21x–25.74x` vs 1-GPU upstream, `0.00e+00` `rel_l2`), and `3.18x`
    ($D=1024$, `-2.3737 bpb`) / `2.60x` ($D=2048$, `-1.3229 bpb`) in a `12s`
    autoresearch loop, `E157`), and **Round-32 RSI (`v149`): `MoBA: Mixture of
    Block Attention for Long-Context LLMs` (`arXiv:2502.13189`, Moonshot AI /
    Kimi Team — `fla-org/flash-linear-attention/tree/main/fla/ops/moba` &
    `fla/layers/moba.py`)** (resolving upstream `fla/ops/moba/parallel.py` and
    `fla/layers/moba.py`'s `0`-Triton-kernel two-stream `_flash_attn_varlen` +
    12-op eager `index_reduce`/`index_add_` LSE merge + `.nonzero()` CPU-GPU
    stalls + `(topk - 1)x` Q / `2x` KV HBM duplication, deterministic GQA
    `RuntimeError` crash when `num_kv_heads < num_heads`, deterministic `FP32`
    `RuntimeError` crash, and `1.87e-2` `BF16`/`TF32` block-gate boundary drift
    via `14/14` `0`-spill (`139` specializations, `max_regs = 168 <= 168`,
    `max_smem = 40,960 B`) Round-32 RSI Triton kernels featuring the Unified
    Historical + Local Causal Block-Bitmask Identity (`b_qmask | (1 << c_q)`),
    Single-Pass Tensor-Core `Flash-MoBA` (`parallel_moba_rsi`), Single-Pass
    Centroid-Routed Block-Sparse Decoding (`moba_decode_rsi`), Fused
    `QK-RMSNorm+RoPE`, and cross-layer `BlockShare`, achieving `8/8` training
    shape wins in Fwd, Bwd, Total & Peak HBM up to `10.54x` Fwd, `4.22x` Bwd,
    `5.71x` Total & `2.08x` HBM reduction vs upstream `parallel_moba`,
    `19.97x–32.95x` faster single-pass block-sparse decoding vs `parallel_moba`
    (`3.95x–11.69x` vs `parallel_moba_rsi` chunk), `3.75x` ($D=1024, K=64$) &
    `3.12x` ($D=2048, K=128$) 2-layer `MoBABlockRSI` + `BlockShare` speedup,
    `5.32x–6.96x` 8-GPU NVSwitch Ulysses `MoBA-CP = 8` speedup over Ring
    (`4.29x–6.23x` vs 1-GPU upstream, `0.00e+00` `rel_l2`), and `3.24x`
    ($D=1024$, `-0.1654 bpb`) / `3.22x` ($D=2048$, `-0.0954 bpb`) in a `12s`
    autoresearch loop, `E158`), and **Round-33 RSI (`v150`): `Hierarchical
    Dual-Route Sparse-Linear Attention (H2SA-GDN) & Governed Loop Optimization
    v3.8 Tournament`** (co-designed and peer-reviewed with M365 Copilot
    `Cowork` `GPT 6.2 Astra, Max` under the exact dense-vs-selected residual
    decomposition $\mathbf{o}_{\text{dense}} - \mathbf{o}_{\text{sel}} =
    m(\mathbf{r} - \mathbf{s})$ — *"a routing miss need not be a memory miss"* —
    resolving coarse block routing's $1/B_C$ needle dilution (`MoBA E158`
    Holdout `0.6437`) and token-only routing's long-horizon state drop (`DSA
    E157` Holdout `0.7775`) via `16/16` `0`-spill (`83` specializations,
    `max_regs = 168 <= 168`, `max_smem = 40,960 B`) Round-33 RSI Triton kernels
    combining Route 1 coarse block centroids, Route 2 rejected-block subtile
    outlier rescue $\tilde{\mathbf{k}}_u = \boldsymbol{\mu}_u +
    \lambda(\mathbf{k}_{s^*(u)} - \boldsymbol{\mu}_c)$, and Route 3 `BV=32`
    V-tiled / `BK=64` K-split causal block-delta Gated-DeltaNet fast-weight
    memory $\mathbf{S}_c$, achieving `8/8` training shape wins up to `8.03x`
    Fwd, `4.97x` Total & `4.55x` HBM reduction, `24.06x–108.41x` faster 3-launch
    single-pass decode (`0.139–0.151 ms`), `3.51x` ($D=1024, K=64$) & `3.24x`
    ($D=2048, K=128$) `H2SAGDNBlockRSI` + `DualRouteShare` stack speedup,
    `2.67x–5.41x` 8-GPU NVSwitch Ulysses `H2SA-GDN-CP = 8` speedup over Ring,
    `3.13x–3.18x` more steps in `12s` autoresearch, and `governed_loop_guard.py
    v3.8` `PROMOTE` (`0.9578` Confirm IUT across all 3 comparators) +
    `HOLDOUT_PROMOTE` (`0.9535` Single-Touch Holdout, `+0.3098` over `MoBA
    E158` and `+0.1760` over `DSA E157` across `RULER_Needle`,
    `BABILong_StateTrack`, `LongBench_DocQA`, `RepoBench_Code`, and
    `Crowded_MQAR`), `E159`).

See `EXPERIMENTS.md` for the preregistered matrix and
`results/benchmark_report.md` for measured findings once available.
