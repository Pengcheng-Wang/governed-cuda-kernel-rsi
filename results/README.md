# Results

Raw benchmark JSON and profiler traces are generated on node-local storage.
Only compact, verified outputs are copied here and to the experiment's unique
durable namespace by node 0.

`benchmark_report.md` will distinguish measured findings from literature claims
and will retain failed or slower implementations as learning evidence.

The second-round attention-training evidence is under
`raw/attention_backward/`; it includes compact PyTorch profiler summaries that
identify the selected FlashAttention forward/backward kernel family.

The FSDP2 round is under `raw/fsdp2/`; it includes 1/8/24-GPU AdamW runs,
process-level repetitions, peak allocated/reserved memory, and compact
collective profiles for DDP, root-only FSDP2, layerwise re-sharding, and
layerwise no-reshard.

The decode/KV round is under `raw/decode_kv/`; it includes two full
process-level repetitions for contiguous and paged MHA/GQA/MQA, randomized
physical page tables, CUDA-core versus tensor-core dispatch, page-size
variation, CUDA Graph replay, correctness against math SDPA, and a compact
kernel profile.

The production INT4 round is under `raw/marlin_w4a16/`; it includes vLLM
Marlin GPTQ-style `uint4b8` weights, channelwise/group-128 packing, transformer
projection shapes, decode-to-prefill M sweeps, output-quality metrics, offline
packing cost, two full process-level repetitions, and a compact kernel profile.

The MoE round is under `raw/moe/`; it includes vLLM fused-expert versus
per-expert-loop results for SLM/LLM shapes, uniform and 80%-hot routing,
top-1/top-2, padding and load-balance statistics, two full process-level
repetitions, compact profiler evidence, and repeated 8-/24-GPU BF16
expert-dispatch all-to-all measurements.

The expert-overlap round is under `raw/ep_overlap/`; it includes a two-GPU
smoke gate and two full process-level repetitions on 8 and 24 GPUs. It
compares communication-only, fused-expert-only, sequential, and
four-microbatch pipelined schedules with dispatch-marker and exact output
validation.

The fixed-total granularity round is under `raw/ep_chunking/`; it holds total
tokens per peer at 1024 or 4096 on 24 GPUs while sweeping one, two, four, and
eight chunks. One/two-chunk cases have two process launches; the close
four/eight-chunk candidates have a targeted third launch.

The full expert-parallel round trip is under `raw/ep_roundtrip/`; it includes
a two-GPU smoke gate plus two full process launches on 8 and 24 GPUs. It
measures dispatch-only, dispatch+combine communication, fused-expert compute,
sequential dispatch→compute→combine, and the pipelined round trip with exact
output validation.

Variable expert dispatch is under `raw/variable_dispatch/`; it includes
balanced, globally hot, and locality-preserving rotated-hot variable splits,
with and without destination capacity 1.25. Full 8-/24-GPU matrices have two
process launches and report receive imbalance, dropped assignments, latency,
logical payload throughput, and marker correctness.

MoE router preparation is under `raw/moe_router/`; it includes three full
process launches comparing BF16 router projection, PyTorch and vLLM fused
top-k, `moe_align_block_size`, precomputed fused experts, full router+expert
paths, correctness, and compact profiler evidence.

The integrated expert-parallel layer is under `raw/moe_end_to_end/`; it
includes a two-GPU correctness smoke and two full process launches on 8 and
24 GPUs. It measures fused routing/top-k, token duplication and permutation,
variable-split hidden/id/weight dispatch, local fused experts, return
all-to-all, and weighted inverse permutation for balanced, globally biased,
and locality-biased routers.

The integrated capacity frontier is under `raw/moe_capacity/`; it includes a
two-GPU correctness smoke and two full process launches on 8 and 24 GPUs. It
keeps the highest router weights within each source-destination segment,
sweeps destination capacity factors 1.25/2/4/8, and reports latency together
with dropped assignments, retained router weight, token coverage, and output
quality against the uncapped path.

The router-skew onset curve is under `raw/moe_skew_sweep/`; it includes a
two-GPU correctness smoke and two full process launches on 8 and 24 GPUs. It
adds a common logit bias of 0–4 to rank 0's experts while holding inputs and
weights fixed, then reports route concentration, effective destinations,
route/output drift, stage timing, and integrated latency.

Temporal skew is under `raw/moe_temporal_skew/`; it includes a two-GPU
correctness smoke and two full process launches on 8 and 24 GPUs. Each cycle
has one microbatch per rank and compares steady balanced routing with
rotating and persistent bias-1/bias-2 hotspots. It reports cycle throughput,
per-step p50/p95/p99 latency, and both aggregate and instantaneous receiver
imbalance.

Packed dispatch is under `raw/moe_packed_dispatch/`; it compares the split
hidden/id/weight baseline with an all-in-one BF16 packet and a metadata-only
BF16 packet. It includes a two-GPU correctness smoke, repeated full matrices
on 8 and 24 GPUs, normal/reversed focused runs, and a 30-cycle Latin-square
crossover that balances timing position across methods.

INT8 activation dispatch is under `raw/moe_int8_dispatch/`; it compares the
metadata-packed BF16 path with symmetric per-token INT8 hidden payloads and
BF16 scales. It includes a two-GPU quality smoke, repeated 8-/24-GPU matrices,
and alternating-order 24-GPU crossovers for balanced, moderate-skew, and
extreme-skew routing.

Fused INT8 conversion is under `raw/moe_int8_fused_dispatch/`; it compares
eager tensor composition with Triton kernels that fuse quantize/pack and
dequantize/unpack. It includes a two-GPU gate, repeated 8-/24-GPU matrices,
and four three-way 24-GPU Latin-square crossovers against BF16 and eager INT8.

Bidirectional INT8 communication is under `raw/moe_int8_roundtrip/`; it keeps
the fused forward path and additionally quantizes expert outputs before the
return all-to-all. It includes a two-GPU quality gate, repeated 8-/24-GPU
matrices, and four 24-GPU crossovers comparing BF16, forward-only INT8, and
INT8 in both directions. Return payload and BF16 scale use separate
collectives so the results expose the launch-cost tradeoff directly.

Packed INT8 return is under `raw/moe_int8_packed_return/`; it compares the
two-collective return with a minimal 1026-byte single-collective packet and a
1040-byte packet padded to a 16-byte boundary. It includes a two-GPU gate,
repeated 8-/24-GPU matrices, four 24-GPU unaligned crossovers, two 8-GPU
alignment crossovers, and four 24-GPU alignment crossovers. Both packed
layouts match the two-collective quantized output exactly.

The generalized alignment sweep is under `raw/moe_int8_alignment_sweep/`.
It removes router/expert weights, uses deterministic variable-split return
traffic, and pairs naturally aligned widths 510/1022/4094 with standard
512/1024/4096 widths. It sweeps packet rounding to 2/4/8/16/32/64 bytes,
with repeated 8-/24-GPU matrices, two 8-GPU crossovers, and three 24-GPU
crossovers. Every packet reproduces the two-collective quantized return
bit-for-bit.

Aligned INT8 return overlap is under `raw/moe_int8_return_overlap/`; it
integrates the 16-byte packet into the fixed-work expert-parallel pipeline.
BF16, two-collective INT8, and single-collective packet returns are each
measured sequentially and pipelined at 1024 tokens/peer with four chunks and
4096 tokens/peer with eight chunks. It includes a two-GPU gate, repeated
8-/24-GPU matrices, two 8-GPU crossovers, and three 24-GPU crossovers.

Variable-split aligned INT8 overlap is under
`raw/moe_variable_int8_overlap/`; it replaces equal-split traffic with real
fused-router/top-2 assignments for balanced, moderate-bias, and extreme-bias
routes. Routing, packing, count exchange, and CPU count extraction are
prepared outside timing; the measured path covers variable hidden/metadata
dispatch, local experts, BF16 or compressed return, dequantization, and
weighted inverse permutation. It includes a two-GPU gate, repeated 8-/24-GPU
matrices, two 8-GPU crossovers, and five 24-GPU crossovers with exact-format
and global-expert quality gates.

Online route/count preparation is under `raw/moe_route_prep_overlap/`; it
moves router projection, fused top-k, token permutation, metadata packing,
count exchange, and Python split extraction into the timed path. It compares
the v23 precomputed lower bound with online staged preparation and a
double-buffered router stream for BF16 and aligned packet pipelines. The
receive-count host barrier remains measured because PyTorch variable-split
all-to-all requires Python split lists.

Static-capacity padded MoE is under `raw/moe_padded_capacity/`; it replaces
variable splits with equal-size source-destination slots and removes count
exchange and Python split lists. The fully static path computes zero-weight
padding rows, so results expose the complete trade among synchronization,
padding, dense expert work, assignment drops, and output quality.

The exact static-capacity frontier is under `raw/moe_capacity_frontier/`.
It derives the minimum lossless slot count from the deterministic router,
sweeps capacities immediately around that integer threshold, and includes
repeated 8-/24-GPU position-balanced crossovers plus reversed-order 24-GPU
controls. It separates coarse-factor overpadding from fused-expert
shape/schedule effects.

GPU-compacted static padding is under `raw/moe_sparse_padding/`. It keeps
fixed equal-size transport but marks padding with invalid expert IDs so
vLLM's GPU alignment skips those rows. Dense padding, identity-map cache
zeroing, and a custom Triton invalid-row cleanup are compared with repeated
8-/24-GPU matrices and position-balanced crossovers. Per-rank padding exposes
why aggregate work reduction can miss the synchronized critical path.

Fixed-profile sparse transport is under `raw/moe_profiled_transport/`. It
calibrates a source-destination capacity matrix over disjoint router batches,
then compares dynamic variable splits with uniform, multiplicative, additive,
and oracle fixed matrices. Repeated position-balanced crossovers separate
online count preparation, payload padding, and complete-layer cost.

Fused fixed-profile packing is under `raw/moe_profiled_packer/`. It compares
the original stable weight-plus-destination sorts with destination-only
sorting and a Triton atomic histogram/scatter kernel. Every promoted record
uses a zero-overflow profile and verifies identical assignments and exact
outputs.

Tiled packer scaling is under `raw/moe_tiled_packer/`. It separates atomic
slot reservation from hidden-row movement and sweeps hidden 512/1024/4096,
token load, and 128/256/512-wide tiles on 2/8/24 GPUs. Pack-only and
router-plus-pack timings establish the shape-gated one-pass/tiled policy.

The wide fixed-profile layer is under `raw/moe_wide_profiled_layer/`. It
integrates tile-512 packing into a hidden-4096/intermediate-4096 distributed
MoE layer, includes a calibration-only exact-margin diagnostic, and reports
route-plus-pack, prebuilt payload, and complete end-to-end timing. Two
order-balanced 8-GPU and five 24-GPU crossovers show that the optimized kernel
beats fixed two-sort while topology-scaled padding can still make the full
fixed-profile system slower than dynamic transport.

Trained-router drift is under `raw/moe_router_profile_drift/`. It contains a
compact 480-sample OLMoE router-count trace, static and sliding-window profile
frontiers for logical 8-/24-rank mappings, and repeated distributed spill
crossovers. The exact fallback keeps accepted rows in fixed tile-512 slots and
sends overflow through a variable metadata-packed side channel; the results
show that the duplicated collective and expert sequence costs more than the
saved padding.

Profile-hit fallback is under `raw/moe_profiled_p2p/`. It uses fixed
tile-512 all-to-all only when a trained-router count matrix fits the current
profile and otherwise executes the exact dynamic path. Repeated 8-/24-GPU
crossovers compare additive and adaptive profiles with grouped fixed-size P2P.
An explicit timeout gate records that oversized NCCL receives cannot infer
smaller message lengths in this PyTorch path.

Device-side fit dispatch is under `raw/moe_device_fit/`. It reduces the tiled
packer's overflow scalar across ranks and passes the device predicate directly
to `torch.cond`. Hits use fixed all-to-all; misses use an exact fixed-signature
token all-gather and contribution reduce-scatter fallback. Two-GPU smoke and
order-balanced 8-/24-GPU crossovers report steady-state branch overhead,
per-profile compilation cost, topology-scaled fallback traffic, and BF16
output quality.

Persistent batched fit control is under `raw/moe_persistent_fit/`. It compares
one overflow reduction and host synchronization per microbatch with one vector
decision for queues of 1/2/4/8 ready microbatches. Additive sweeps locate the
queue-depth and hit-rate crossover; focused additive/adaptive controls use 20
balanced timing cycles. Results include controller-only cost, queued-route
peak memory, exact fit decisions, and output validation.

Persistent-fit queue latency is under `raw/moe_queue_latency/`. It feeds the
real focused scalar/vector service replicates into a Poisson single-server
simulation with timeout fallback. Streaming and conservative batch-fence
completion bounds report throughput, queue delay, p50/p95/p99 response,
vector-batch fraction, and overload behavior.

Heterogeneous continuous decode is under
`raw/decode_continuous_batching/`. It fits the two measured FlashInfer
paged-GQA process replicates, combines them with the measured CUDA-Graph
decode MLP, and simulates a 32-layer continuous-batching stage. Homogeneous
and mixed prompt/output workloads compare ragged summed-context work with a
padded-to-longest control and a 250 ms decode-stage TTFT admission guard.
Results include accepted throughput, rejection, TTFT/TPOT/completion
percentiles, batch occupancy, padding inflation, and KV reservation.

Static decode graph bucketing is under `raw/decode_graph_bucketing/`. It
reuses the same measured-service 32-layer model and compares ragged execution
with one global graph bucket and sequential 2/4/8 context buckets. Static
policies round batch sizes to powers of two; an exact-batch ablation exposes
the graph-cache versus capacity tradeoff. The main result assumes no graph
speedup, while an explicit optimistic bound applies the measured tiny-shape
replay saving to every static attention group.

Decode admission fairness is under `raw/decode_admission_fairness/`. It
reproduces the v37 unguarded and TTFT-only scenarios exactly, adds a
completion-aware admission proxy, and audits all nine prompt/output classes.
Results include request and output-token acceptance, class TTFT/TPOT and
completion, Jain indices, long/short prompt and output ratios, and aggregate
throughput/SLO tradeoffs.

Chunked prefill and prefix reuse are under `raw/prefill_decode_scheduling/`.
The model fits measured FlashAttention causal-pair cost and cuBLAS dense
launch/FLOP cost, composes them across 32 layers, and interleaves them with
the prior ragged decode model. Monolithic and 2048/1024/512/256-token chunks,
prefill-first and 2:1/4:1 decode-priority schedules, and a mixed 75%-prefix-hit
mode report capacity, end-to-end TTFT, TPOT, maximum inter-token gaps, chunk
overhead, and prefix compute savings.

Adaptive prefill scheduling is under `raw/adaptive_prefill_scheduling/`. It
reproduces fixed monolithic/2048/1024/512 controls from v40 and adds 50/100 ms
predicted kernel-slice budgets with queue-age escalation. Results include
selected chunk distribution, slice latency, escalation frequency, capacity,
TTFT/TPOT, maximum token gaps, and prefix-hit interaction.

Stall-capped prefill admission is under `raw/prefill_stall_admission/`. It
holds active prefill slices to a hard 100 ms fitted-service budget and compares
no admission with two- and four-second predicted queued-prefill-service
guards. Results include request/token acceptance, throughput, TTFT/TPOT,
maximum token gaps, predictor calibration, and prompt/output-length rejection
bias.

Prefill admission safety and fairness is under
`raw/prefill_admission_safety/`. It preserves the hard 100 ms slice controller
and compares raw two-second admission with 2x/4x/8x structural safety margins
and an 8x square-root prompt-credit control. Results include TTFT calibration,
request and output-token acceptance, prompt-class acceptance and Jain
fairness, throughput, and the latency/fairness cost of restoring long-prompt
access.

Weighted-fair prefill admission is under `raw/weighted_fair_admission/`. It
keeps the hard 100 ms slice and 8x safety baseline, then adjusts bounded
prompt-class credits from online equal-class acceptance deficits with gains
2/4/8. Exact E52 controls, request/output-token acceptance, prompt-class Jain
fairness, controller credit/deficit tails, throughput, TTFT/TPOT, and prefix
reuse interactions are included.

Nonstationary weighted-fair admission is under
`raw/nonstationary_fair_admission/`. It rotates exact short-heavy, long-heavy,
medium-heavy, and short-return prompt phases and compares cumulative gain-4
feedback with 128/256/512-request rolling histories. Exact E53 stationary
controls, aggregate versus phase Jain fairness, dominant-class credit
adaptation, return hysteresis, throughput, TTFT/TPOT, token-gap tails, and
prefix reuse interactions are included.

Time- and service-decayed fairness state is under
`raw/decayed_fair_admission/`. It replaces hard request windows with
20/40-second exponential decay measured by scheduler wall time or offered
predicted prefill GPU service. Exact E54 controls, effective history weight,
clock deltas, phase fairness, return hysteresis, throughput, TTFT/TPOT, and
prefix/load interactions are included.

Service-clock calibration robustness is under
`raw/service_clock_robustness/`. It perturbs only fairness-state decay with
uniform ±25%, long-prompt ±25%, and cache-blind cost errors while keeping
admission and execution unchanged. Exact E55 controls, no-cache equivalence,
clock-error ratios, effective history, phase fairness, hysteresis,
throughput, TTFT/TPOT, and prefix-cache sensitivity are included.

Tenant and reservation-class fairness is under
`raw/tenant_reservation_fairness/`. It defines exact interactive,
long-context, and generation-heavy tenant mixes with correlated prompts,
outputs, and cache hits. Prompt identity, equal tenant identity at gains 4/8,
and weighted 1.0/0.85/0.70 reservations are compared with prefill-only and
measured prefill-plus-decode clocks. Tenant/request/token/service acceptance,
Jain metrics, service shares, throughput, TTFT/TPOT, and prefix interactions
are included.

Causal batched-decode attribution is under
`raw/batched_decode_attribution/`. An optional observer exactly decomposes
each fitted batched decode step into per-request batch, context, intercept,
and MLP service without changing policy output. Batch-1 standalone and equal
step sharing are compared against causal tenant decode/total-service shares,
per-request overcharge, batch occupancy, throughput, and latency.

Causal tenant-service feedback is under
`raw/causal_service_fairness/`. Observed prefill slices and exact batched
decode allocations update a service-decayed tenant ledger directly. Equal
and weighted 1.0/0.85/0.70 targets are swept at gains 4/8/16 and compared
with the E57 proxy-clock controls. The results include causal service shares,
target error, reservation-normalized Jain, request/token acceptance,
throughput, TTFT/TPOT, batch occupancy, exact control equivalence, and
observer/accounting validation.

Admission-time causal service commitments are under
`raw/causal_service_commitments/`. E59 actual-only feedback is compared with
pending prefill, fixed-batch-32 decode, and deliberately pessimistic batch-1
decode commitments under Poisson, 4-request, and 16-request arrivals.
Within-admission-batch staleness, actual service target error, prediction
bias, acceptance, throughput, TTFT/TPOT, exact E59 equivalence, commitment
drainage, and reconciliation are included.

Burst ingress pacing is under `raw/burst_ingress_pacing/`. Four- and
sixteen-request bursts are spread across 0/10/25/50/90% of their natural
period while end-to-end latency remains anchored to the original client
arrival. Actual-only and batch-32 commitment policies report acceptance,
throughput, client/server TTFT, ingress delay, fairness, admission batch
size, and exact zero-pacing E60 equivalence.

SLO-budgeted absolute pacing is under `raw/slo_ingress_pacing/`. Absolute
0–600 ms windows are swept at 0.3x/0.4x load for 4- and 16-request bursts.
The maximum-throughput window satisfying worst-scenario 650/800/1000 ms
client-TTFT SLOs is selected, with acceptance, ingress/server latency,
fairness, admission-batch size, window capping, and exact zero-window E61
equivalence included.

Robust pacing-policy compression is under
`raw/pacing_policy_compression/`. E62's eight-state oracle is compressed into
universal, one-dimensional, two-dimensional, and full lookup policies.
Robust maximum-p99 tuning is compared with unsafe median-p99 tuning using
key count, mean/geometric/worst oracle throughput retention, acceptance
regret, scenario/state SLO violations, maximum excess, selected windows, and
the complexity Pareto frontier.

Pacing state-estimation uncertainty is under
`raw/pacing_state_uncertainty/`. Independent burst/load/cache flips are
applied to every E63 robust lookup family. Naive estimated-key lookup is
compared with a Hamming-radius-1 safety envelope using expected oracle
throughput, acceptance regret, scenario violation probability, unsafe-path
probability, maximum excess, feature sensitivity, and complete
true/estimated-state paths.

Confidence-set pacing fallback is under `raw/pacing_confidence_sets/`.
Per-feature low-confidence flags form an axis-aligned set of possible true
keys, and the selected window must satisfy the SLO for every state in that
set. Joint 5%/10%/20% estimator error is combined with mistake-flag recall
and false-alarm profiles, then compared with E64's naive and static
Hamming-1 baselines using throughput retention, scenario violations,
unsafe/missed-error probability, confidence-set size, and false-alarm tax.

Confidence-detector operating frontiers are under
`raw/pacing_confidence_frontier/`. Mistake recall and correct-feature
false-alarm rate are swept across all E63 families, 650/800/1000 ms SLOs,
and 5%/10%/20% joint state error. Each surface reports throughput/risk
Pareto points plus maximum-throughput selections for explicit expected
scenario violation budgets, both at fixed false-alarm rates and across E65's
deployable named profiles.

Finite-sample calibration certification is under
`raw/pacing_calibration_certification/`. One-sided Wilson score bounds with
family-wide Bonferroni correction estimate the labeled-request count needed
to certify each E66 point's state-error cap, mistake-recall floor, and
false-alarm cap. Results separate nominal policy feasibility from monitoring
certifiability and expose which metric limits each operating point.

Anytime-valid confidence monitoring is under
`raw/pacing_anytime_monitoring/`. A summable-alpha Hoeffding confidence
sequence protects repeated peeking across every keyed metric-feature stream.
Results compare its labeled-request cost with E67 one-shot Wilson plans and
measure fresh-audit detection latency for state-error, mistake-recall, and
false-alarm breaches.

Geometric checkpoint monitoring is under
`raw/pacing_checkpoint_monitoring/`. Time-uniform alpha is spent over look
index rather than every request, with 1.10x/1.25x/1.50x/2.00x checkpoint
growth. Results quantify label savings versus E68, residual overhead versus
E67, within-look action delay, drift latency, and which cadence wins each
finite operating case.

Bernoulli-KL checkpoint monitoring is under `raw/pacing_kl_monitoring/`.
Binary relative-entropy Chernoff intervals replace E69's Hoeffding bounds at
the same checkpoints and alpha allocation. Results verify interval
containment and no-later crossings, then report label savings versus E69,
residual overhead versus E67, drift acceleration, and cadence changes.

Metric-aware alpha allocation is under `raw/pacing_alpha_allocation/`.
Positive weighted-Bonferroni shares are swept across state error, mistake
recall, and false alarms at the same E70 KL checkpoints. The analysis
compares scenario-conditioned, global median-optimized, and global minimax
splits, then applies certification-optimized weights to drift detection.

The joint certification/drift alpha frontier is under
`raw/pacing_joint_alpha_frontier/`. Every E71 allocation is treated as one
global preregistered configuration across all finite certification and
active drift cases. Results report Pareto-optimal splits, strict maximum
drift-slowdown caps, joint nonworsening counts, and checkpoint-quantization
cliffs.

Checkpoint phase and rounding sensitivity is under
`raw/pacing_checkpoint_phase_sensitivity/`. It perturbs the 2.00x look grid
over eight logarithmic phases and three integer rounding modes, deduplicates
the resulting schedules, and retests E72's jointly nonworsening allocations.
The result separates exact-grid wins from allocations robust to ordinary
checkpoint implementation changes.

Exhaustive cross-grid alpha robustness is under
`raw/pacing_cross_grid_alpha/`. It extends E73 from six candidates to all 172
positive E71 allocations, reporting full-simplex grid survival, cross-grid
minimax and Pareto selections, and strict worst-drift-cap feasibility.

Checkpoint cadence robustness is under `raw/pacing_cadence_robustness/`.
Equal alpha is held fixed while 1.10x/1.25x/1.50x/2.00x schedules are
evaluated over their original recursive E70 grid and direct logarithmic
phase/rounding alternatives. Results compare nominal and worst-over-grid
winners, implementation-uncertainty tax, and fixed-cadence robust regret.

The statistical-look cost frontier is under
`raw/pacing_look_cost_frontier/`. It assigns a synchronous
request-equivalent cost to every checkpoint and recomputes robust composite
cost, cadence winners, and fixed-schedule regret. An exact integer scan
identifies where the global minimax cadence changes as monitoring overhead
grows.

The asynchronous monitor-worker frontier is under
`raw/pacing_monitor_workers/`. FIFO monitoring jobs are replayed through
one/two/four/eight/unlimited worker pools. Results report exact cadence
transition regions, action-delay tails, worker monotonicity, and containment
between unlimited parallelism and E76's synchronous blocking upper bound.

Latest-pending monitor coalescing is under
`raw/pacing_monitor_coalescing/`. Running checks remain nonpreemptive, while
each new checkpoint replaces the older queued look when all workers are busy.
Results include an outcome-independent alpha-validity gate, exact FIFO
dominance, dropped-job fractions, sampled action-delay savings, and transition
diagnostics showing why the robust cadence can oscillate on integer
checkpoint/service lattices.

The realized-path coalescing audit is under
`raw/pacing_coalescing_stochastic/`. It samples joint certification and
conditional drift Bernoulli paths at recursive E70 looks, then compares FIFO
with latest-pending action times across all E78 worker/service settings.
Results separate optional-stopping validity from action-latency dominance and
report crossing persistence, metric-specific reversal tails, missed actions,
and null-alpha containment.

The bounded pending-depth frontier is under
`raw/pacing_coalescing_depth/`. It replays the same stochastic paths through
freshest-K queues with depths 1/2/4/8/16/32/64. Results report processed work,
aggregate and worst-profile regression, positive-delay severity, explicit
risk-cap selections, and the FIFO-equivalent no-drop boundary.

Exact batched look replay is under `raw/pacing_coalescing_batch/`. It merges
all pending checkpoint snapshots into one job without dropping statistical
looks, then sweeps the marginal service cost of each extra snapshot. Results
report exact alert membership, E78 launch equivalence at zero marginal cost,
launch and total-work savings, and action-delay frequency/severity as batch
work grows.

Streaming exact batches are under `raw/pacing_coalescing_streaming/`. They
reuse E81's batches and total work but release each member checkpoint as soon
as its incremental snapshot completes. Results report per-look and action
dominance over the batch fence, the FIFO-regression boundary, worst-profile
concentration, and positive-delay severity.

The exact batch member-order frontier is under
`raw/pacing_coalescing_order/`. It holds E82 batch formation, launches, work,
and alert membership fixed while comparing chronological, newest-first, and
endpoint-alternating member orders. The compact result reports aggregate,
configuration, breach-metric, worst-profile, and order-versus-chronological
summaries.

The robust member-order policy audit is under
`raw/pacing_coalescing_order_robustness/`. It loads E83's compact frontier and
compares fixed, domain-aware, cost-aware, metric-aware, and metric+cost-aware
policies under explicit aggregate-risk, worst-profile, and maximum-delay
lexicographic objectives. It records exact action counts, normalized
metric-cost oracle regret, configuration-key complexity, and the multi-objective
Pareto front. This is local-only analysis because the source allocation
remained canceled.

Delay-capped policy compression is under
`raw/pacing_coalescing_order_compression/`. It enumerates all `4^6` member-order
mappings per drift metric at each feasible observed delay cap, merges adjacent
identical orders into contiguous cost buckets, and optimizes every key budget.
The result reports exact regression and acceleration counts, oracle-gain
retention, cap compliance, the minimum eight-rule exact oracle, and the
regression-delay-complexity frontier. This is local-only analysis because the
source allocation remained canceled.

Marginal-cost estimation uncertainty is under
`raw/pacing_coalescing_order_cost_uncertainty/`. It treats the six E83 cost
points as an ordinal grid and evaluates radius-0/1/2 actual-versus-estimated
cost neighborhoods. Fixed newest-first, E84's nominal metric-cost oracle,
E85's compressed eight-rule mapping, and uncertainty-aware minimax mappings
are compared over every admissible pair and an adversarial one-estimate-per-
actual-metric envelope. This is local-only analysis because the source
allocation remained canceled.

Joint cost uncertainty and delay caps are under
`raw/pacing_coalescing_order_uncertainty_caps/`. For radius-one and radius-two
cost errors, each candidate order must satisfy the delay cap for every actual
cost in the uncertainty set. Results identify the first cap improving
all-pair risk, the first improving both all-pair and adversarial risk, the cap
with the best aggregate counts, and the first cap restoring fixed
newest-first's worst-cell envelope. This is local-only analysis because the
source allocation remained canceled.

The cost-estimator accuracy frontier is under
`raw/pacing_coalescing_order_accuracy_frontier/`. It sweeps exact
classification probability continuously under symmetric radius-1/radius-2
and one-sided under/over-estimation kernels with boundary clamping. Expected
regression, acceleration, positive-delay burden, savings, and support maximum
are affine in accuracy; pairwise crossover points are solved analytically and
verified on a 101-point grid. This is local-only analysis because the source
allocation remained canceled.

Cost-prevalence robustness is under
`raw/pacing_coalescing_order_prevalence/`. It exhaustively evaluates all
3,003 six-cell prevalence vectors on the 10% simplex for every E88 kernel and
1% accuracy point. It reports prevalence-grid winner shares, E85 coverage
thresholds, perfect-accuracy regression-optimal versus tie-break coverage,
and distribution-free minimax maximum per-cost regret. This is local-only
analysis because the source allocation remained canceled.

The pilot-selected holdout allocation is under
`raw/pacing_coalescing_order_pilot_holdout/`. It uses E83 to fix deployment
policies and future sampling scores, then constructs final simultaneous
intervals from future holdout counts only. The artifact compares uniform,
outcome-independent geometry, selected-policy/all-competitor pilot geometry,
and current-worst-only pilot geometry. It reports safe outcome-aware gains,
score concentration, coverage requirements, and the statistical price of
discarding rather than reusing pilot counts. This is local-only unchanged-
rate planning analysis because the source allocation remained canceled.

Adaptive-valid two-stage reuse is under
`raw/pacing_coalescing_order_adaptive_reuse/`. It combines exact one-sided
pilot and conditional holdout binomial p-values with Fisher product and
future-budget-weighted Stouffer rules. The artifact includes exact adaptive-
design enumeration, interval-root checks, future-only and pooled-oracle
comparisons, coverage requirements, and the Fisher weighting reversal. The
pooled fixed-sample CP result is explicitly invalid after adaptive
allocation and is retained only as an optimistic reference. This is
local-only unchanged-rate planning analysis because the source allocation
remained canceled.

The Stouffer stage-weight frontier is under
`raw/pacing_coalescing_order_stage_weights/`. It compares five fixed
future-weight multipliers, an alpha-split selection-safe best-of-five, and a
pilot-dependent realized sample-size-weight negative control. The artifact
reports exact adaptive-design checks, multiplier winners, coverage
requirements, root validation, selection tax, and a concrete nominal-5%
type-I inflation example. Individual full-alpha multipliers are valid only
when preregistered before the pilot. This is local-only unchanged-rate
planning analysis because the source allocation remained canceled.

The dense Stouffer stage-weight plateau is under
`raw/pacing_coalescing_order_stage_weight_plateau/`. It evaluates 21 fixed
full-alpha multipliers on coarse and fine grids plus exact `1/sqrt(2)` and
`sqrt(2)` anchors. The artifact reports global and local optima, 0.25/0.5/1%
tested plateaus, relative regret, and a robust single multiplier. `sqrt(2)`
minimizes worst global relative regret at 0.154% above the condition-specific
optimum and lies inside every tested 0.5% global p90 and maximum-regret
plateau. This is a post-hoc recommendation for preregistration in a future
independent pilot, not permission to retune E83 at full alpha. It is
local-only unchanged-rate planning analysis because the source allocation
remained canceled.

The numerical continuous Stouffer limit is under
`raw/pacing_coalescing_order_stage_weight_continuous/`. It replaces E105's
coarse candidate choice with vectorized exact binomial-tail bisection,
nested meshes, local candidate refinement, direct minimax-envelope
intersections, and continuous plateau roots over multiplier 0.5-3.0. The
estimated minimax multiplier is 1.428317 with 0.1292% worst relative-regret
excess, versus 0.1557% for `sqrt(2)`. The common 0.25% plateau is
[1.372817, 1.528723], so `sqrt(2)` remains safely inside the operationally
flat region. This is post-hoc planning for a future independent pilot, not
permission to retune E83 at full alpha. It is local-only unchanged-rate
analysis because the source allocation remained canceled.

Stage-weight outcome sensitivity is under
`raw/pacing_coalescing_order_stage_weight_outcome_sensitivity/`. It compares
five fixed multipliers across four integer projections, 32 point-binomial
future datasets, and 32 Jeffreys posterior-predictive datasets. `sqrt(2)`
won 787 of 2,448 candidate comparisons and 34 of 68 scenario-level minimax
contests, versus 247 and 11 for E106's 1.428317. Under posterior prediction,
`sqrt(2)` won 24 of 32 scenario contests. The result reverses the practical
preference without contradicting E106: the continuous optimum's tiny
unchanged-rate gain is smaller than future-outcome sensitivity. This is
fixed-seed local planning simulation, not new measured paths or a coverage
certificate.

Posterior-predictive convergence is under
`raw/pacing_coalescing_order_stage_weight_predictive_convergence/`. Four
independent 32-dataset blocks compare `sqrt(2)` directly with 1.428317.
`sqrt(2)` won 102 of 128 scenario-level minimax contests and 3,788 of 4,608
condition comparisons. Its scenario win rate was 79.69%, with an exact 95%
simulation-frequency interval of 71.67%-86.28%. Every seed block favored
`sqrt(2)`, confirming that E107 was not a one-seed artifact. The interval
quantifies Monte Carlo frequency, not policy coverage.

Correlated posterior-predictive sensitivity is under
`raw/pacing_coalescing_order_stage_weight_dependence_sensitivity/`. It adds
global-, cost-, order-, and metric-correlated Gaussian-copula Jeffreys
profiles plus a mixed 75%-shared-variance profile. Every profile favored
`sqrt(2)`. Across the five correlated profiles, it won 115 of 160
scenario-level minimax contests and 4,432 of 5,760 condition comparisons.
The exact 95% simulation-frequency interval for the combined correlated win
rate was 64.23%-78.69%. These are structured positive-dependence stresses,
not a fitted production traffic model.

Independent A100 reproduction is under
`raw/independent_a100_reproduction/`. It contains six smoke results, two
full process repetitions of the representative early benchmark panel, four
additional transformer-training repetitions, and a compact comparison with
the original allocation. Triton/native-CUDA fusion, RMSNorm, sequence-2048
Flash SDPA, 256 MiB all-reduce, correctness, and the cuBLAS-over-Triton
negative control reproduced within 5%. The original one-run
`torch.compile` speedup did not: six-run medians were 1.08x at 4x1024 and
1.06x at 1x4096, exposing baseline and process-level variability.

Transformer compilation diagnosis and mode frontier are under
`raw/transformer_compile_diagnosis/`. One smoke gate and three full process
repetitions on A100-SXM4-80GB record in-process NVML SM-clock/power telemetry
across unlocked DVFS versus hardware-locked 1140 MHz and 1410 MHz clocks,
cold versus warm TorchInductor caches, reversed execution order, seven
execution/compiler modes, and `torch.profiler` CUDA kernel shares. At locked
1410 MHz, cold- and warm-cached `compile(default)` artifacts agreed within
0.11%, proving that E7's 1.26-1.27x and E110's bimodal runs came from
1140-to-1410 MHz DVFS transitions during 3-step warmup (`1140 MHz` eager over
`1410 MHz` compiled = 1.31x/1.30x). True locked-1410 MHz `default` speedups
are 1.09x (4x1024) and 1.07x (1x4096), matching `torch.profiler`'s
1.10x/1.08x kernel-sum reduction within 0.5% because 75.5-80.4% of eager
device time already runs inside cuBLAS GEMMs and FlashAttention-2. On
4,096-token shapes, `max-autotune-no-cudagraphs` reaches 1.14x and 1.12x via
fused Triton GEMM-epilogue templates (`triton_tem_fused_*`), whereas on
launch-bound 1x512 (71.7% host/autograd overhead), full-step CUDA Graph
capture around `compile(default)` reaches 3.36x versus 1.63x for module-only
`reduce-overhead`.

Large-vocabulary fused linear cross-entropy (`V = 128,256`) is under
`raw/fused_linear_cross_entropy/`. One smoke gate and three full process
repetitions on A100-SXM4-80GB evaluate `eager_fp32_logits_ce`,
`eager_bf16_logits_ce`, `torch_compile_full_ce`,
`checkpointed_chunked_compile_ce`, and a custom `autograd.Function` pairing
token-tiled cuBLAS Tensor Core GEMMs with a two-pass online-softmax in-place
Triton gradient kernel (`triton_fused_linear_ce_bf16_dw` and
`triton_fused_linear_ce_fp32_dw`). At `16384x1024x128256` (1B SLM, 16K
tokens), `triton_fused_linear_ce_bf16_dw` (`chunk_size = 1024`) cuts peak
transient workspace from `23,765.5 MiB` to `250.6 MiB` (`94.8x` below eager
FP32, `16.0x` below `torch.compile`'s materialized `4,008.0 MiB` `[BT, V]`
BF16 buffer, and `45.1x` total step incremental reduction) while running
`1.76x` faster (`62.86 ms` vs `110.64 ms`) with exact FP32 log-sum-exp
accuracy (`loss_abs_error <= 1.98e-5` vs `1.47e-2–2.80e-2` for direct BF16 CE).
Across hidden widths `H = 1024 -> 2048 -> 4096` (`BT = 8192`), activation
checkpointing crosses from `1.15x` to `0.93x` to `0.81x` eager speed due to
its 4th-GEMM recomputation tax, whereas 3-GEMM Triton terminal-loss fusion
stays faster at every width (`1.73x`, `1.35x`, `1.15x`), and PyTorch 2.13's
`torch.addmm(..., out_dtype=torch.float32)` keeps `dW` error completely
invariant across 1 to 64 chunks.

Programmable `FlexAttention` training is under
`raw/flex_attention_training/`. One smoke gate and three full process
repetitions on A100-SXM4-80GB evaluate sequence-packed document-masked GQA
(`1x8192`, `32Q/8KV`, `D=128` across `1`, `4`, `16`, and `64` documents) and
score-modified GQA (`2x4096`, `32Q/8KV`, `D=128` across `causal_softcap30`,
`sw512_softcap30`, and `sw512_log_decay_bias`). On packed documents, passing a
2D boolean mask with `enable_gqa=True` to `F.scaled_dot_product_attention`
silently fell back to `SDPBackend.MATH` (`33,152.0 MiB`, `269.5 ms`), while
Cutlass `EFFICIENT_ATTENTION` (`961.0 MiB`) remained stuck at `59.5-59.8 ms`
across 1 to 64 documents. `FlexAttention` with `BlockMask` (`705.0 MiB`)
scaled from `19.81 ms` (1 doc) to `3.77 ms` (16 mixed non-128-aligned docs,
`15.8x` faster than Cutlass and `71.5x` faster than auto SDPA) and `2.66 ms`
(64 short SFT docs, within `7.7%` of `flash_attn_varlen_func`'s `2.47 ms`),
while `torch.compile(create_block_mask)` cut mask creation `9.5-11.0x` to
`0.54-0.66 ms`. On `sw512_log_decay_bias`, where fixed-kernel `flash_attn` has
no support, `FlexAttention` (`max-autotune`) beat eager materialized attention
by `17.6x` and `torch.compile` over materialized attention by `5.26x` while
cutting step incremental memory from `8,274.0 MiB` to `705.0 MiB` (`11.7x`).

NVSwitch `SymmetricMemory`, Sequence Parallelism, and `Async-TP` training are
under `raw/symmetric_memory_async_tp/`. One multi-GPU smoke gate (`TP = 2, 8`)
and three full OS-process repetitions (`TP = 2, 4, 8`) on 8xA100-SXM4-80GB
(NV12 NVSwitch) at locked 1410 MHz evaluate: (1) `SymmetricMemory` one-shot
(staged and with-copy) and two-shot all-reduce versus NCCL (`64 KiB` to
`64 MiB`), showing two-shot reaching `11.40x` over NCCL at `64 KiB` (`22.2` vs
`251.4 us` on `TP = 8`), `8.83x` at `1 MiB`, and `1.15x` at `64 MiB` (`196.4
GB/s` bus BW) while one-shot falls to `0.27x` (`4.23x` slower than two-shot)
due to `7x` vs `1.75x` P2P read amplification; (2) isolated
`fused_all_gather_matmul` and `fused_matmul_reduce_scatter`, crossing from
`0.51x`/`0.29x` at `M = 2048` (`M_shard = 256` wave quantization) to
`1.21x`/`1.08x` at `M = 32,768`; (3) a double PyTorch 2.13 compiler trap where
obeying `enable_symm_mem_for_group`'s `FutureWarning` leaves `Async-TP`
disabled (`0/0` fused ops) and permanently poisons TorchInductor's persistent
FX graph cache; and (4) eight Llama-3-8B GQA Transformer block (`H = 4096,
FFN = 14336, 32Q/8KV`) training modes across `1x4096..8x4096`, mapping the
frontier between `[S, B, H]` `Shard(0)` (`1.61x` faster than `Shard(1)` at
`1x4096`, `1.99x` lower peak incremental memory at `8x4096` [`1,295.0` vs
`2,576.8 MiB`], and `1.06-1.08x` `Async-TP` speedup [`1.22x` over Megatron TP
eager]) and `[B, S, H]` `Shard(1)` (`1.32x` over Megatron TP eager at `8x4096`
[`31.71 ms`, `1,504.5 MiB`] via contiguous per-sequence FlashAttention-2
strides).

Long-context distributed attention and Hybrid 2D Context Parallelism (`CP = 2,
4, 8`) are under `raw/context_parallel_attention/`. One multi-GPU smoke gate
(`CP = 2, 8`) and three full OS-process repetitions (`CP = 1, 2, 4, 8`) on
8xA100-SXM4-80GB (NV12 NVSwitch) at locked 1410 MHz evaluate `S = 8,192` to
`131,072` (`8K` to `128K` tokens) across Contiguous Ring, `2*CP` Zig-Zag Ring
(eager vs fused Triton online-softmax LSE merger + double-buffered `(K, V, dK,
dV)` P2P overlap), 32-head KV-expanded vs 8-head GQA-native DeepSpeed-Ulysses,
Hybrid 2D `USP (Ulysses x Zig-Zag Ring)`, PyTorch 2.13 experimental
`context_parallel` (`allgather` vs `alltoall`), the `CP > H_kv` (`H_kv = 8, 4,
2, 1`) sparse-GQA head wall, and full RoPE Llama-3-8B Transformer blocks (`H =
4096, FFN = 14336`) with exact `out`/`dQ`/`dK`/`dV`/`dx`/`dW` and `1.75x`
chain-rule validation.

Pipeline Parallelism, Zero-Bubble & DualPipe-V `dX`/`dW` splitting, and
Selective Activation Checkpointing (`PP = 4, 8`) are under
`raw/pipeline_parallel_zero_bubble/`. One 16-layer 8-GPU smoke gate and three
full OS-process repetitions (`PP = 1, 4, 8`) on 8xA100-SXM4-80GB (NV12
NVSwitch) at locked 1410 MHz evaluate a 16-layer Llama-3-8B-width GQA
Transformer (`3.49B` parameters, `H = 4096, FFN = 14336, 32Q/8KV, D = 128`)
across `GPipe`, `1F1B`, `Interleaved 1F1B`, `ZB1P`, `ZB-V`, and DeepSeek-V3
`DualPipeV` (`M = 8, 16, 32`, `S = 1,024` to `8,192`), comparing PyTorch 2.13's
built-in `stage_backward_input` graph-walker against a custom zero-copy
`SplitDWLinear` + in-place cuBLAS `addmm_` `dw_builder` with functional
`@torch.compile` fusions, plus four activation checkpointing policies (`no_ac`,
`sac_nonlin_only`, `sac_mlp_recomp`, and `full_ac`) with exact single-GPU
loss/`dW` and `1.75x` chain-rule validation.

Fine-grained MoE training kernels, dynamic-routing CUDA Graphs, and 8-GPU
Expert-Parallel `Split-dW` overlap are under
`raw/moe_training_grouped_gemm/`. One 8-GPU smoke gate and three full
OS-process repetitions on 8xA100-SXM4-80GB (NV12 NVSwitch) at locked 1410 MHz
audit PyTorch 2.13's `torch._grouped_mm` on SM80 (`Invalid strides [0, 0]`
backward rejection, `6` unpinned `offs.cpu()` `Memcpy DtoH` syncs breaking
CUDA Graph capture, and `384` host-loop vendor GEMM launches), compare
`eager_python_loop`, `padded_bmm_3d`, `pytorch_grouped_mm`, device-resident
L2-swizzled Triton Grouped GEMMs (`Fwd`, `dX`, `dW`), and dynamic-routing
CUDA Graphs across ISO-FLOP `E = 8, 32, 64` granularities (`H = 2048`),
measure fused Triton `Permute + SwiGLU + WeightedCombine (dy_perm + d_probs)`
memory/bandwidth at `H = 4096`, and evaluate 8-GPU `EP = 8` `Split-dW`
backward `AllToAll` overlap and static-capacity gradient cosine fidelity with
exact single-GPU 64-expert and `FP64` chain-rule validation.

Fused multi-tensor optimizer kernels, sub-ULP Philox stochastic-rounding
`SR-BF16` & 8-bit (`INT8`/`UINT8`) `AdamW`, and 8-GPU `FSDP2` owner-sharded
batched Tensor-Core `Muon` are under `raw/optimizer_kernels_and_fsdp2_muon/`.
One 8-GPU smoke gate and three full OS-process repetitions on
8xA100-SXM4-80GB (NV12 NVSwitch) at locked 1410 MHz audit PyTorch 2.13's
`p.grad_dtype = torch.bfloat16` `AdamW` runtime rejections, prove sub-ULP
`BF16` round-to-nearest stagnation versus IEEE-754 sign-magnitude Philox
stochastic rounding, benchmark single-launch pointer-table Triton `AdamW`
variants with in-register global `clip_grad_norm_` scaling across `67M`,
`268M`, and `1.07B` parameters plus 120-step SLM convergence, and compare
`FSDP2` `AdamW`, `INT8-mv`, `SR-BF16`, PyTorch 2.13 `DTensor` `Muon`, a
shard-local NS5 negative control, and 2-stream pipelined owner-sharded
`AllToAll` + batched `BF16` `bmm` `Muon` on `3.49B` isolated and `1.01B`
end-to-end training steps.

Sub-quadratic linear-attention (`Mamba-2 SSD` / `GLA`) & chunkwise WY `Gated
DeltaNet` training kernels and 8-GPU Context Parallelism (`CP = 8`, `S =
2K..262K`) are under `raw/linear_attention_gdn_training/`. One 8-GPU smoke gate
and three full OS-process repetitions on 8xA100-SXM4-80GB (NV12 NVSwitch) at
locked 1410 MHz verify overflow-free chunk-local decay exponents, `FP64`
recurrence equivalence and `O(h^2)` Richardson central-difference (`~4.00x`)
gradients across all 5 inputs (`q, k, v, beta, g`), benchmark register-pure
SM80 `BF16` Tensor-Core `4 × 16` nilpotent doubling + block substitution WY
preparation against `torch.linalg.solve_triangular` and Triton IEEE SIMT,
measure 1-GPU `Fwd + Bwd` scaling (`S = 2,048..65,536`), uncrowded vs crowded
state-reassignment MQAR, and `4:0` / `3:1` Hybrid / `0:4` 4-layer `H = 2,048`
stacks, and compare serial P2P state-passing, `O(1)` `AllGather` augmented
`[S_zero | P]` parallel affine-monoid scans, and GQA-native DeepSpeed-Ulysses
`FlashAttention-2` up to `S = 262,144` tokens.

Multi-Head Latent Attention (`MLA`) training kernels, weight-absorption
`512+64 -> 512` vs split-head `128+64 -> 128` crossover, zero-attention-recompute
`Latent-SAC`, and 8-GPU Context Parallelism (`CP = 8`, `S = 2K..131K`) are under
`raw/mla_latent_attention_training/`. One 8-GPU smoke gate and three full
OS-process repetitions on 8xA100-SXM4-80GB (NV12 NVSwitch) at locked 1410 MHz
audit PyTorch 2.13 `SDPA`, `FlexAttention`, `flash-attn 2.8.3`, and `FlashInfer
0.6.18` (`FA2` vs `FA3`) across asymmetric `192 -> 128` and `576 -> 512` MQA
shapes, verify `FP64` Richardson central-difference (`~4.00x`) chain-rule
gradients across all split and merged `dC_kv` inputs, ablate 1-pass global
`tl.atomic_add` `dQ` serialization and `num_stages=1` stalls against lock-free
2-pass `cp.async` Split-Dim Triton `Flash-MLA` (`128+64 -> 128`), benchmark
DeepSeek-V3-geometry (`H = 4096, n_h = 32, d_c = 512, d'_c = 1536`) 1-GPU layer
training (`S = 2K..32K`), `S_q = 1..4096` decode-to-prefill weight absorption
against `L_kv = 16,384` (`B = 8`), 4-layer `Latent-SAC`, and 8-GPU `CP = 8`
`Compressed-Latent Zig-Zag Ring` vs `Decompressed Ring` vs `FA2 V-padded
Ulysses` vs `Split-Dim Ulysses` up to `S = 131,072` tokens.

Reasoning Post-Training (`RLVR` / `GRPO` / `DAPO` / `DPO`) fused terminal
kernels, Two-Slice Ragged Group-Prefix-Tree `FlashAttention`, and 8-GPU `FSDP2`
Long-CoT straggler balancing are under `raw/rlvr_grpo_post_training/`. One
8-GPU smoke gate and three full OS-process repetitions on 8xA100-SXM4-80GB
(NV12 NVSwitch) at locked 1410 MHz prove `FP32` catastrophic cancellation in
uncentered online vocabulary entropy vs a shift-invariant centered online
`(m, d, e)` monoid (`V = 128,256`), verify `FP64` Richardson central-difference
(`~4.00x`) chain-rule convergence across `GRPO/DAPO`, `Saved-LSE DPO`, and
shared-prompt Prefix-Tree `GQA`, benchmark `3`-GEMM token-tiled in-place Triton
`FusedLinearGRPO/DAPO` (`Clip-Higher` + Schulman `k3` KL + Entropy + `1/tau`)
and `4`-GEMM Two-Pass `Saved-LSE` `FusedLinearDPO` (`4 B/tok` `lse[t]`) across
`N = 2,048..16,384` (`V = 128,256`), evaluate 4-layer `H = 4,096` (`32Q/8KV,
D = 128`) `GRPO` group scaling (`G = 4, 8, 16`) across padded `FA2`,
duplicated-prompt ragged `FA2`, `FlexAttention` `BlockMask`, and Two-Slice
Ragged Group-Prefix-Tree Triton `GQA`, and audit 8-GPU `FSDP2` Long-CoT `RLVR`
(`naive_local_token_mean` gradient bias vs exact `world_size / N_resp,global`
`DAPO` normalization + Adaptive Tree Split + Quadratic-Cost LPT Prefix-Tree
`GQA` on a `1.38B`-parameter stack).

SLM Knowledge Distillation coupled-monoid terminal kernels (`Forward KL` /
`MiniLLM Reverse KL` / `GKD JSD_0.5`), DeepSeek-V3 Multi-Token Prediction
(`MTP`), `EAGLE-3` GQA-Packed Two-Region `TTT` attention, and 8-GPU
Vocabulary-Parallel (`TP = 8, V_shard = 16,032`) Teacher-Student distillation
are under `raw/slm_distillation_mtp_eagle3/`. One 8-GPU smoke gate and three
full OS-process repetitions on 8xA100-SXM4-80GB (NV12 NVSwitch) at locked
1410 MHz prove `FP32` shift cancellation in uncentered cross-distribution
Reverse-KL vs a 6-statistic coupled centered `(m_S, d_S, e_S, m_T, d_T, c_ST)`
monoid (`V = 128,256`), quantify `Top-K` (`K = 32, 64, 128`) logit-truncation
`dW_S` gradient bias, verify `FP64` Richardson central-difference (`~4.00x`)
across 9 derivatives, ablate unpacked `4x` KV reloads vs GQA-packed Two-Region
`EAGLE-3 TTT` Triton attention (`S = 512..4,096, K = 4`) against `SDPA`
staircase masks and `FlexAttention` `BlockMask`, benchmark 1-GPU Teacher
(`H_T = 4,096`) -> SLM Student (`H_S = 1,536, V = 128,256`) distillation
(`N = 2,048..16,384`) and DeepSeek-V3 `4`-head shared-`W_unembed` `MTP`, and
compare colocated `FSDP2`, `Hybrid FSDP2-Backbone + Vocab-Parallel Heads`, and
`TP = 8` Teacher + `Vocab-Parallel` Coupled-Monoid online distillation on
`1.40B -> 0.44B` models.

Block-Hadamard `INT8` Tensor-Core training (`SwitchBack` / `Jetfire` / `COAT` /
`Halo`), `COAT` compressed `INT8` activation saving, and 8-GPU
Quantized-`AllGather` `FSDP2` training are under
`raw/int8_quantized_training/`. One 8-GPU smoke gate and three full OS-process
repetitions on 8xA100-SXM4-80GB (NV12 NVSwitch) at locked 1410 MHz resolve E6's
`torch._int_mm` contiguous-copy negative control against zero-copy column-major
`mma.sync.m16n8k32.row.col.s32.s8.s8.s32`, compare `Jetfire`/`DeepSeek-V3`
`1x128 x 128x128` in-loop CUDA-core promotion (`19.5 TFLOP/s` SIMT bottleneck)
against epilogue-only coalesced Triton and `CUTLASS SM80` `INT8` GEMMs (`>450
TOP/s`), ablate `FP32 atomicCAS` + 10-round Philox + `>255`-register spills
against `s32`-bitcast `RED.MAX.S32` + SplitMix32 zero-spill tiles, verify `65x`
outlier suppression (`1D Block-128` and `2D Kronecker H_64 x H_64`), `256`-step
sub-LSB `SR-INT8` vs `RTN-INT8` convergence, and `FP64` Richardson
central-difference (`~4.00x`), benchmark 1-GPU Llama-3-8B `3-GEMM` (`Fwd + dX +
dW`) Linear scaling (`M = 2,048..16,384`) surpassing SM80's `312 TFLOP/s`
`BF16` hardware ceiling, evaluate 4-layer Pre-Norm `SwiGLU` MLP (`rstd`-invariant
Fused `RMSNorm -> INT8` + `32x128` `COAT` `1 B/elem` `INT8` saved activations)
across `N = 2,048..8,192`, and compare 8-GPU `BF16` `FSDP2` (`eager`,
`torch.compile`, and `checkpoint`) against shard-local Hadamard `1 B/param`
`INT8`-`AllGather` `halo_coat_quant_fsdp2` (`1.39B` params, `65,536` global
tokens/step).

SM80 `2:4` Semi-Structured Sparse `BF16` Tensor-Core training (`cuSPARSELt
0.8.1` / `Exact-90` Bipartite DP / `KKT-Exact Madow MVUE` / `Grad-SR-STE`),
layout-aware `Sparse-SwiGLU`, and 8-GPU `9 bits/param` Compressed-`AllGather`
`FSDP2` training are under `raw/2by4_sparse_training/`. One 8-GPU smoke gate and
three full OS-process repetitions on 8xA100-SXM4-80GB (NV12 NVSwitch) at locked
1410 MHz audit PyTorch `2.13` `semi_structured` triple defects
(`SparseSemiStructuredTensorCUTLASS` `0.82x` slowdown vs plan-tuned `cuSPARSELt`
`448.2 effective TFLOP/s`, default `prune_dense_static_sort("")` `74.2%`
positive signed-sort bias, `largest_abs_values_greedy` `15.8%` `4x4` bipartite
tile deadlock vs `0`-spill `Exact-90` factored bipartite DP dual packer, and
`mm_search` `split_k > 1` `CUSPARSE_STATUS_INTERNAL_ERROR`), verify `1,024`-step
`KKT-Exact Madow MVUE 2:4` `1/sqrt(T)` unbiased `dW` convergence, `160`-step
`AdamW` `16`-bit-bitmask `Grad-SR-STE` mask flip-rate damping (`62.5x`), and
`FP64` Richardson central-difference (`~4.00x`), benchmark 1-GPU Llama-3-8B
`3-GEMM` (`Fwd + dX + dW`) `2:4` `BF16` Linear scaling (`M = 2,048..16,384`,
reaching `394.4 effective TFLOP/s` above SM80's `312 TFLOP/s` dense `BF16`
ceiling), ablate column-major strided vs coalesced `SwiGLU` (`7.58x`) and
4-layer Pre-Norm `Sparse-SwiGLU` (`N = 2,048..8,192`), and compare 8-GPU dense
`BF16` `FSDP2` (`eager` and `torch.compile`), overlapped post-`AllGather`
redundant `2:4` `FST`, FFN-hybrid `9-bit` `2:4` `FSDP2`, and full-linear
shard-local `Exact-90` `9 bits/param` Compressed-`AllGather` `FSDP2` (`1.31B`
params, `65,536` global tokens/step, `4,992 -> 2,808 MiB/step` `AllGather`).

Natively Trainable Dynamic Sparse Attention (`DeepSeek NSA` & `Moonshot Kimi
MoBA`), Lock-Free Split-Reduce `CSC` Backward, Zero-Selection-Recompute
`NSA-SAC`, and 8-GPU NVSwitch Group-Preserving `2D USP-NSA` Context Parallelism
(`CP = 8`, `S = 16K..131K`) are under `raw/native_sparse_attention_training/`.
One 8-GPU smoke gate and three full OS-process repetitions on 8xA100-SXM4-80GB
(NV12 NVSwitch) at locked 1410 MHz audit PyTorch `2.13` `FlexAttention`
quadruple dynamic-mask defects (`128.0 GiB` uncompiled `vmap` OOM,
`BlockMask.from_kv_blocks` `_transpose_ordered` `IndexError`, `BLOCK_SIZE = 64`
TorchInductor rejection, and sticky `cudaErrorIllegalAddress` `int32` overflow
at `S = 65,536`), benchmark SM80 `[B_q_sub x G]` Tensor-Core query-group tile
saturation (`106.5 sparse TFLOP/s`, `27.5x` over per-head `Quest`), ablate
sink-contended `FP32` atomic `dK/dV` (`2,094.4 ms`) vs unsplit `CSC` (`67.3 ms`)
vs 100% Lock-Free Split-`256` `CSC` (`37.1 ms`, `56.4x` speedup at `S = 32K`),
verify `FP64` exact oracle & Richardson central-difference (`~4.00x`) across 8
non-linear derivatives, measure 1-GPU `Fwd + Bwd` scaling (`S = 4K..64K`) across
C++ `FA2`, `FlexAttention-128`, per-head vs GQA-group Triton-LSE `MoBA` (`8.66x`
vs `FA2`), and 3-branch `DeepSeek NSA` (`4.84x` vs `FA2`), evaluate 4-layer
`NSA-SAC` (`S = 8K..32K`), and compare 8-GPU `CP = 8` `1D Ulysses-8 Dense FA2`,
`Seq-Sharded KV-AllGather NSA` (`92.4%` Coupon-Collector block-union audit), `1D
Ulysses-8 2x-Rep NSA`, and Group-Preserving `2D USP-NSA (Ulysses-4 x
Causal-Asymmetric-ZigZag-2)` (`0.000000` forward parity, `6.24x` over dense
`FA2` at `S = 131,072`).

Multimodal Vision-Language & Omni-Modal Native-Resolution Training Kernels
(`Qwen2.5-VL` / `Qwen3-VL` / `NaViT` / `Transfusion` / `VeOmni`) and 8-GPU
NVSwitch Modality-Decoupled `FSDP2` are under `raw/multimodal_vlm_omni_training/`.
One 8-GPU smoke gate and three full OS-process repetitions on 8xA100-SXM4-80GB
(NV12 NVSwitch) at locked 1410 MHz audit `cuDNN 9.20.0` `Conv3d(2,14,14)`
tubelet pathology (`32.94 ms`, `8.98 TFLOP/s` on `32,768` patches) against
zero-atomic Triton `Window-Morton-2x2` hierarchical ragged tubelet packing +
`16`-aligned `cuBLAS` (`1.69 ms`, `175.6 TFLOP/s`, `19.54x` speedup, `dx rel_l2
= 0.0`) and zero-copy Triton `RMSNorm` + `view` `PatchMerger` (`1.41x` speedup,
`2.87x` lower HBM), benchmark single-launch Triton `Q+K` `3D M-RoPE`
(`[16,24,24]` sectioned & `[24,20,20]` interleaved, exact `0.0` `FP64` `1D`
reduction on text, `2.36–2.37x` over HF eager), audit causal-`FA2` visual
corruption (`92.3%`) and PyTorch `SDPA` `bool`-mask `28.0 GiB` Math-SDPA OOM
(`S = 16,384`), verify `FP64` exact oracle (`rel_l2 <= 0.0036`) & Richardson
central-difference (`3.995x–4.0001x` across 8 non-linear derivatives), measure
1-GPU native-resolution ViT scaling (`P = 4K..32K`: `3.89x` over padded 2D ViT,
`1.25x` over eager `window_index` gather), measure 1-GPU `Omni-Attention`
(`S = 4K..64K`: Surgical Disjoint Reversed-Upper `FA2` + Triton `LSE` `2.29x`
over dynamic `FlexAttention` and `2.10x` over prebuilt `FlexAttention` at `64K`,
plus `1.41x` across a 4-layer `3,584`-dim decoder stack at `32K`), and compare
8-GPU `0.93B` `FSDP2` (`B = 16, 32`) across naive sample-sharded padded-ViT +
`FlexAttention`, naive sample-sharded `Win-Morton-2x2` + Surgical `FA2`,
LLM-token-greedy packing, and Two-Stage Modality-Decoupled `AllToAllv` +
`Quad-Cost-LPT` (`loss rel_diff = 1.80e-6`, `2.22–2.34x` speedup, `1.90x` lower
max-rank HBM, ViT patch skew `3.18x -> 1.07x`).

Diffusion-LLM (`LLaDA` / `MDLM`), Block-Diffusion (`BD3-LMs`), & `DiT`
`adaLN-Zero` Training Kernels driven by our Immutable Recursive Self-Improvement
(`RSI`) Vector-Pareto Ratchet (`PROGRAM.md` + `code/rsi_kernel_harness.py`) are
under `raw/diffusion_llm_bd3_adaln_training/`. One 8-GPU smoke gate, three full
OS-process repetitions, and the append-only `rsi_ledger.tsv` (`eval_harness_sha256`
verified) on 8xA100-SXM4-80GB (NV12 NVSwitch) at locked 1410 MHz catch PyTorch
`2.13` TorchInductor's SymPy `bool == bool` crash on verbatim `BD3-LM` Fig. 4
(`Trial 0a`), `W = 4` micro-CTA waste in 6-call varlen `FA2` (`Trial 1`), and
`W = 64` 4-accumulator register spills in monolithic GQA microblocks (`Trial 2`)
before ratcheting Adaptive Zero-Spill Split-Stream GQA Microblocks (`W <= 32`) +
Surgical `FA2` (`W >= 64`) (`Trial 3` `KEEP`, `1.83x` geomean speedup across
`W = 4..256`, `54.3x` over `SDPA` `2L` at `L = 4K`, `2.06x` over prebuilt
`FlexAttention` at `2L = 65,536`) and Exact Terminal Clean-Branch Pruning
(`Trial 4` `KEEP_TERMINAL_STAGE`, `3.29x` geomean, `3.94x` at `2L = 65,536`),
ratchet zero-gather/zero-atomic Triton block-segmented `adaLN-Zero` +
`Gate-Residual-adaLN` (`4.90x` over eager, `2.62x` over `torch.compile`) and
Active-Mask-Compacted `3-GEMM` `FusedLinearMaskedCE` (`V = 128,256`, `1.94x`
speedup & HBM cut) alongside `1.82x` unbiased complementary-antithetic / `46.2x`
clipped `U[0.3, 0.7]` `1/t` NELBO gradient-variance reduction (`FP64` Richardson
`3.9999x–4.0001x` across 8 derivatives), and evaluate 4 cumulative RSI rungs on
8-GPU `1.12B` `BD3-LM` `FSDP2` (`B = 16, 32` up to `131,072` `[x_t || x_0]` dual
tokens/step, `loss rel_diff = 7.28e-6` & `dW rel_l2 <= 0.013` with Rank 7 forced
to `M = 0`, reaching `263,834 [x_t || x_0] tok/s` and saving `16,297 MiB` =
`15.92 GiB/rank` at `B = 32`).

Round-2 RSI Roofline Convergence, Adversarial Gate Torture, 8-GPU `1.12B`
`BD3-LM` `FSDP2` System RSI, and our Fixed-Wall-Clock `val_bpb` Autonomous
Vibe-Training (`autoresearch`) Loop are under
`raw/rsi_round2_roofline_and_valbpb_loop/`. Governed by the byte-identical
immutable `code/rsi_kernel_harness.py` (`eval_harness_sha256 = 3cf0f078...`),
one 8-GPU smoke gate and three full OS-process repetitions on 8xA100-SXM4-80GB
(NV12 NVSwitch) at locked 1410 MHz deterministically reject 4 red-team exploits
(including PyTorch `flash_attn_varlen_func` autograd's silent `grad_softmax_lse`
omission causing `21.3%` `dK` corruption vs `0.25%` via direct global-`(Out_t,
LSE_t)` backward, `84.5x` lower error), accelerate `adaLN-Zero` by `10.42x`
(`Staged3`) and `17.43x` (`Unified` Adaptive Direct-`dc6` / Split-`32`, `85.91x`
over PyTorch Eager, `1,243.6 GB/s` physical HBM, fixing `grid=(1,)` clean-stream
single-SM starvation `693.49 -> 3.36 ms`), eliminate `454`-byte `W = 64` cubin
spills in `BD3` attention (`0` spills across all 4 `K_TILE=16` Triton kernels,
`1.094x` full / `1.902x` pruned over Round-1, `1.50x` lower HBM), break E127's
`1.087x` 8-GPU `1.12B` `BD3-LM` `FSDP2` ceiling to `463,119 [x_t || x_0] tok/s`
(`1.760x` over Round-1 / `2.214x` over Literature at `W = 64`, `1.645x` / `1.698x`
at `W = 16`, saving `18,698–18,827 MiB/rank` with `loss rel_diff <= 1.81e-5` &
split-shard `dW <= 0.0156` vs unmerged 1-GPU at Rank-7 `M = 0`), and drive a
5-generation fixed-`12.0s`-wall-clock `141M` `BD3-SLM` `val_bpb` loop (`Gen 3`
spike reverted) to `2.30x` completed optimizer steps (`81 -> 188`) and `-2.748
val_bpb` / `-2.662` untouched `16`-node `test_bpb` improvement (`7.189 -> 0.0843
bpb`).

Round-3 RSI Multi-Stream Residual Topology Kernels for DeepSeek `mHC`
(`Manifold-Constrained Hyper-Connections`, `arXiv:2512.24880`), On-Chip
Division-Free `Log-Sinkhorn-20`, Selective `L_r^*` Stream Checkpointing
(`mHC-SAC`), 8-GPU NVSwitch `1.07B` `mHC-FSDP2`, and our Fixed-Wall-Clock
`val_bpb` Architecture-Kernel Co-Design Loop are under
`raw/mhc_manifold_hyper_connections_rsi/`. Governed by the same byte-identical
frozen `code/rsi_kernel_harness.py` (`eval_harness_sha256 = 3cf0f078...`), one
8-GPU smoke gate and three full OS-process repetitions at locked 1410 MHz reject
`4/4` red-team exploits, prove column-max `exp(Z - max_col Z)` blows up
`8,192/8,192` (`100%`) elements to `NaN` under a `+115` row offset vs `0` `NaN`s
in `0`-spill `5x4`-checkpointed `Log-Sinkhorn-20` (`FP64` Richardson
`3.9850x–4.0000x`), ratchet Single-Scan Tensor-Core `Proj+RMS` + `Log-Sinkhorn-20`
to `8.81x` over Eager (`1.31x` over `torch.compile`, `3.76x` lower HBM) and
Superfused `Pre+RMSNorm` + Unified Single-Read/Write Sandwich Backward to `7.31x`
over Eager (`4.49x` over `torch.compile`, `1.11x` faster and `1,120 MiB` leaner
than DeepSeek's 2-kernel split), confirm DeepSeek Eq. 20's exact empirical HBM
minimum at `L_r^* = 4` across all 3 repetitions (`L_r=2` `8,333.14` == `L_r=8`
`8,333.14 MiB`), compress 8-GPU `1.07B` `n = 4` `mHC-FSDP2` (`S = 2,048`,
`65,536` tok/step, `loss rel_diff = 3.3e-7`, `dW <= 0.0139`) step-time overhead
vs `1-Stream` from `+142.95%` down to `+10.94%` (`L_r=1`) / `+16.52%` (`L_r^*=4`)
while saving `44,037 MiB/rank` (`43.01 GiB/rank`, leaner than `1-Stream`), and
deliver `2.81x` updates + `-1.639 val_bpb` over Eager `mHC` (`-0.321` over
`torch.compile`, `-1.005 val_bpb` matched-step win over a `124,853`-param-larger
oracle-LR `1-Stream` baseline while breaking `Gen 3`'s `cos = 1.0000` symmetric
saddle).

Round-4 RSI Conditional Memory & Scalable $O(1)$ Lookup Kernels for DeepSeek
`Engram` (`Conditional Memory via Scalable Lookup`, `arXiv:2601.07372`) paired
with `M = 4` `mHC` (`arXiv:2512.24880`), Zero-Sync Tile-Fused `JIT-Lazy-Adam`,
Zero-Transpose `M = 4` `Split-64` `ShortConv`, 8-GPU NVSwitch `Dedup-AllToAllv`
+ Layer-2 Async Stream Prefetch (`8.39M` Prime Rows), and our Sparsity-Allocation
U-Curve are under `raw/engram_conditional_memory_rsi/`. Governed by the same
byte-identical frozen `code/rsi_kernel_harness.py` (`eval_harness_sha256 =
3cf0f078...`), one 8-GPU smoke gate and three full OS-process repetitions at
locked 1410 MHz reject `4/4` red-team exploits, verify `23.00%` surjective
tokenizer compression (`129,280 -> 99,546`, `0` hash mismatches, `0.0`
packed-document leakage), prove `torch.optim.SparseAdam` `50`-step Zipfian
divergence (`0.657` `rel_l2`) vs Pre-Forward Sorted `JIT-Lazy-Adam(lr=5x, wd=0)`
(`1.82e-6`, `361,000x` closer, `0` cubin spills across all 12 Triton kernels,
`FP64` Richardson `3.9867x–4.0247x`), expose `engram_demo_v1.py`'s `eps=None`
(`0.0060`) and `signed_sqrt` `125x` Jacobian-cliff (`8.72%` vs `0.167%` `Eq. 6`)
differentiation pitfalls, ratchet `Track 1` to `5.38x` over Demo (`2.18x` over
GPU-Hash + Dense `Adam`) and `Track 2` to `3.26x` over Eager (`1.65x` over
`torch.compile`, `1.13x` lower HBM), verify `0.0` 3-step `Gate G9` reassembled
shard table error and scale `8.39M` prime rows across 8x A100 NVSwitch (`r3`
Async Prefetch `2.57x` over Replicated Demo, `1.14x` over Naive `AllToAll`,
`3/3` strict `r3 < r2` `-1.03 ms` NVSwitch prefetch, `74.3x` less comm,
`-16,251.6 MiB/rank`), and drive a 5-gen `12.0s` `val_bpb` RSI loop (`2.51x`
steps, `-0.437 val_bpb` vs Eager, `-0.024` vs `torch.compile`) plus a 6-arm
matched-step iso-param/iso-FLOPs Sparsity-Allocation U-Curve (`3.2456 ->
3.1611* -> 3.1774 -> 3.1838 bpb`, beating `Iso-FLOPs Dense` by `-0.0927 bpb` and
`2.9x`-FLOPs `Wider Dense` by `-0.0839 bpb`).

Round-5 RSI Nonlinear Fast-Weight Test-Time Training Kernels for `LaCT:
Test-Time Training Done Right` (`arXiv:2505.23884`, ICLR 2026,
`a1600012888/LaCT`), Independent 3-Matrix Batched `FP32` `Muon-NS5` + Analytical
Fréchet VJP, Compact Chunk-Boundary State Checkpointing, 8-GPU NVSwitch
Within-Chunk Sub-Chunk `CP = 8` + 1-Hop P2P Halo `SWA` (`S_global = 65,536,
C_global = 16,384`), and our Beyond-Window Associative Recall Scaling Law are
under `raw/lact_test_time_training_rsi/`. Governed by the same byte-identical
frozen `code/rsi_kernel_harness.py` (`eval_harness_sha256 = 3cf0f078...`), one
8-GPU smoke gate and three full OS-process repetitions at locked 1410 MHz reject
`4/4` red-team exploits, expose three production bugs/spills in
`a1600012888/LaCT` (`78.59%` concatenated `[dW0; dW2]` `Muon` spectral
corruption vs `9.2e-7` in independent `[3*B_h, R, K]` `FP32` `baddbmm`
`Muon-NS5`, `4,582`/`4,024` 3-accumulator cubin spills + `OutOfResources` vs `0`
cubin spills across all `9` Round-5 RSI Triton kernels with `out_dtype=float32`
Tensor-Core epilogues and `FP64` Richardson `3.9996x–4.0002x`, and `0/0 = NaN`
`L2Norm` backward on zero rows), ratchet `Track 1` to `2.30x` over Eager and
`Track 2` to `2.60x` over Eager (`1.53x` faster and `793 MiB` leaner than
`torch.compile(dynamic=False)` + `FA2 SWA`), verify `0.0042` `Gate G9` 8-GPU
`CP = 8` parity vs 1-GPU (`vs 27.56` `dW0` error under naive double-`AllReduce`)
while scaling `S = 65,536` at `5.21x` over `AllGather` Replicated (`2.12x` over
RSI `AllGather`, `1.57x` over `AllGather-SWA`, `9.71x` less comm,
`-17,941 MiB/rank`), and drive a 5-gen `16.0s` `val_bpb` RSI loop (`2.71x`
steps, `-0.599 val_bpb`, `99.15%` untouched beyond-`SWA` test recall) plus a
4-arm matched-step (`220` steps, `~2.60M` iso-params) Beyond-Window Associative
Recall Law (`LaCT(SwiGLU+Muon+Mom)` `98.83%` recall / `3.0657 bpb` beating
`Linear-Delta` `93.03%` / `3.0931`, `SWA-only` `6.38%` / `3.1211`, and
unorthogonalized `SwiGLU-GD` `7.36%` / `3.4088`).

Round-6 RSI Depth-Wise Selective Residual Aggregation Kernels for Moonshot Kimi
`Attention Residuals` (`AttnRes` & `Block AttnRes`, `arXiv:2603.15031`,
`MoonshotAI/Attention-Residuals` + community Triton reference `Hydra-2P`
`manishklach/attnres-kernel-lab`), Superfused Two-Phase `Block AttnRes +
Sublayer Pre-RMSNorm`, 8-GPU NVSwitch Interleaved `1F1B` (`P = 8, V = 4`, `32`
Virtual Stages = `128` Sublayers, `142` Megatron-LM Causal Ticks) Cross-Stage
Virtual-Rank Block Caching (`Sec. 4.1 Eq. 7–8 & Fig. 3`), and our Deep-`24`-Layer
(`48`-Sublayer) `PreNorm`-Dilution Scaling Law are under
`raw/attention_residuals_rsi/`. Governed by the same byte-identical frozen
`code/rsi_kernel_harness.py` (`eval_harness_sha256 = 3cf0f078...`), one 8-GPU
smoke gate and three full OS-process repetitions at locked 1410 MHz reject `4/4`
red-team exploits (`Fig. 2` `i = 0` duplicate-$b_0$ bug, omitted `RMSNorm` radial
Jacobian, `allow_tf32` flag mutation, and unscaled partition merge), diagnose
community `Hydra-2P`'s `(N*M)`-way `tl.atomic_add` backward slowdown (`65.41` vs
`56.91 ms` Eager at `D = 2,048`, `REJECT_SHAPE_REGRESSION`) and SM80's `128`-reg
ceiling at `BD = 4096, nw = 16` (`28` spills) vs `14/14` zero-spill (`0` cubin
spills) Round-6 RSI kernels (`FP32 max_rel_l2 = 2.64e-6`, `BF16 = 0.0137`, `7/7`
`FP64` `O(h^2)` Richardson checks in `[3.998x, 4.008x]`), ratchet `Track 1` to
`6.38x` over Eager (`1.50x` faster and `2,542 MiB` leaner than `fullgraph=True`
`torch.compile`) and `Track 2` (`8`-block sequential stack + final readout via
`RSIPhase1UnbindInterBlockFn` + zero-`O1_bar` retention) to `5.26x` over Eager
(`1.49x` over `torch.compile`, saving `23,748 MiB` = `23.19 GiB` HBM), verify
`7.7e-7` `FP32` `Gate G9` 8-GPU PP parity vs 1-GPU (`vs 1.000` under missing
cross-chunk `grad_cache` accumulation) while scaling `D = 2,048` 8-GPU PP at
`4.72x` over pure-Triton `Full AttnRes` PP (`6.88x` over Eager `Block` PP,
`1.18x` pure Cross-Stage Block Caching gain, `9.93x` & `2.60x` P2P traffic
reductions), and drive a 5-gen `15.0s` `val_bpb` RSI loop (`1.71x` steps,
`-0.6946 val_bpb`, `-0.6944` untouched `test_bpb`) plus a 5-arm matched-step
(`240` steps) Deep-`24`-Layer Residual-Topology Law (`Block AttnRes N=8` `1.6990
bpb` / `16.53%` multi-hop recall beating `Standard PreNorm` `1.7315` / `8.63%`,
`Full AttnRes` `1.7581`, `DenseFormer` `1.7915`, and `Highway` `2.1802` while
suppressing `430.61x -> 49.16x` `PreNorm` dilution norm explosion).

Round-7 RSI `H-Net: Dynamic Chunking for End-to-End Hierarchical Sequence
Modeling` (`arXiv:2507.07955`, Hwang, Wang & Gu; official repo `goombalab/hnet`),
Zero-Sort `cu_seqlens` Compaction, 2-Level Blelloch Scalar-Gate Affine EMA,
8-GPU NVSwitch Whole-Document Cost-LPT `AllToAllv` (`196,608` UTF-8 bytes/step),
and our 5-Arm Byte-Level Phase-Shift & Character-Noise Robustness Law (`E133`)
are under `raw/hnet_dynamic_chunking_rsi/`. Governed by the same byte-identical
frozen `code/rsi_kernel_harness.py` (`eval_harness_sha256 = 3cf0f078...`), one
8-GPU smoke gate and three full OS-process repetitions at locked 1410 MHz reject
`4/4` red-team exploits, diagnose upstream `dc.py`'s `BF16` `F.normalize` router
boundary-flip corruption (`> 20%` drift -> `REJECT_CORRECTNESS`),
`int(num_tokens.max())` `torch.compile(fullgraph=True)` `UserError`, `BF16`
`dt = -log(1 - p)` SSD surrogate drift, and `72`-spill naive `[8, BD=4096],
nw=16` tiles vs `17/17` zero-spill (`0` cubin spills, `max 32 regs`, unbounded
`gridDim.x`) Round-7 RSI Triton kernels (`FP32 max_rel_l2 = 5.90e-7`, `BF16 =
0.0043`, `7/7` `FP64` Richardson `~4.000x`), ratchet `Track 1` isolated DC ops
to `6.31x` over Official (`7.73x` over Packed Eager, `2.63x` faster and
`1,861.8 MiB` leaner than `torch.compile(dynamic=True)`) and `Track 2` 1-stage
hierarchical `H-Net` (`Varlen FA2` + `BF16` Tensor-Core `w_res`) to `2.36x` over
Official (`1.23x` faster and `1,363.9 MiB` leaner than `compile`), verify
`6.4e-8` 3-step `FP32` `Gate G9` loss parity (`max_param_rel_l2 = 2.64e-6` vs
`76.26%` rank-local ratio-loss gradient bias) while scaling `196,608` bytes/step
across 8x A100 NVSwitch at `1.80x` over Official DDP (`1.84x` pure Cost-LPT
gain, collapsing straggler skew `2.96x -> 1.01x` at `3.20x` less traffic than
raw-byte redistribution), and drive a compression-governed (`>= 2.0` B/chunk)
`12.0s` `val_bpb` RSI loop (`1.41x` updates, `-0.0891 val_bpb`, `-0.0871`
untouched `test_bpb`, catching `alpha = 0` dense-collapse reward hacking) plus a
matched-step (`240` steps) Byte-Level Perturbation Law (`Full H-Net RSI`
`1.3355` clean / `2.0520` shifted+noisy `test_bpb`, `43.82%` morpheme `F1`
beating `SpaceByte` `1.3711` / `2.1069`, `Static Stride-4` `1.4997` / `2.1253`,
`Unsmoothed DC` `1.7480`, and `Isotropic Byte SWA` `2.1438`).

Round-8 RSI `PaTH Attention: Position Encoding via Accumulating Householder
Transformations` (`arXiv:2505.16381v2`, Yang,FX-Team, Kim & Rush; official repo
`fla-org/flash-linear-attention` `fla/ops/path_attn/`) paired with `FoX`
(`Forgetting Transformer`, `arXiv:2503.02130`), Block-Recurrent `[BK=64, 64]`
`UT`-Transform WY Representation (`Eq. 3–6`), Zero-Spill Two-Phase Split-`K`
`64+64` Backward, 8-GPU NVSwitch Compact WY `Ring PaTH-CP = 8` (`T_global =
8,192, K = 128`), and our 5-Arm Position-Encoding & `2x` Length-Extrapolation
Scaling Law (`E134`) are under `raw/path_householder_attention_rsi/`. Governed
by the same byte-identical frozen `code/rsi_kernel_harness.py`
(`eval_harness_sha256 = 3cf0f078...`), one 8-GPU smoke gate and three full
OS-process repetitions at locked 1410 MHz reject `4/4` red-team exploits,
diagnose official `fla-org` `chunk_cumprod_householder_bwd_kernel`'s `3,732`-byte
cubin register spills (`[128, 64]` `[b_k, b_v, b_dk, b_dh]` live in `FP32`
registers at `K = 128`) plus its `@input_guard(contiguous=True)` `4x` `.contiguous()`
stride copy overhead (`intra_chunk_preprocess_bwd_fn`) and `L2Norm` zero-row
`NaN` risk vs `22/22` zero-spill (`0` cubin spills across all `40` compiled
specializations) Round-8 RSI Triton kernels (`FP32 max_rel_l2 = 9.95e-7`,
`BF16 = 0.0079`, `8/8` `FP64` Richardson `~4.000x`), ratchet `Track 1` core
`PaTH-FoX` attention (`s1..s4`) to `11.54x` geomean over Eager (`10.97x` over
`torch.compile`, `2.52x` geomean / up to `5.13x` at `K = 128` over Official
`fla-org` `parallel_path_attn`, saving `10.8 MiB` HBM at `T = 4,096`) and
`Track 2` end-to-end `PaTH-FoX` block (`Superfused ShortConv3+SiLU+L2Norm+BetaSigmoid
Frontend + PaTH-FoX`) to `10.05x` over Eager (`2.88x` over Eager Frontend +
Official `fla-org`, `2.82x` over `torch.compile` Frontend + Official `fla-org`),
verify `1.12e-6` 3-step `FP32` `Gate G9` parameter parity (`vs 0.3105`
shard-local no-handoff corruption) while scaling `T_global = 8,192, K = 128`
across 8x A100 NVSwitch via Compact WY `Ring PaTH-CP = 8` at `55.56x` over
All-Gather Eager, `8.33x` over All-Gather Official `fla-org` (cutting peak HBM
`2.30x`, saving `370.9 MiB/rank`), and `3.49x` over Dense-Matrix Ring Handoff,
and drive a fixed-`15.0s` `8xA100 DDP` `val_bpb` RSI loop (`2.45x` updates
`415 -> 1,016` vs `794` for Official `fla-org`, `-0.1682 val_bpb`, `-0.1632`
untouched `test_bpb`, `99.46%` cyclic `S_4` state-tracking accuracy, reverting
no-`L2Norm` reflection explosion & `C = 16` over-chunking) plus a matched-step
(`260` steps) 5-Arm Position-Encoding & `2x` Length-Extrapolation Law (`PaTH-FoX
RSI` `2.0285` clean / `2.1936` `2x` extrap `test_bpb` and `91.66%` clean /
`89.96%` `2x` extrap state-tracking accuracy beating `RoPE` `2.1158` / `2.8288`
& `58.73%` / `45.86%`, `ALiBi` `2.1851` / `2.3820` & `61.26%` / `53.26%`, `FoX`
`2.1867` / `2.3145` & `60.60%` / `51.89%`, and `NoPE` `2.7371` / `2.9532`).

Round-9 RSI `Mixture-of-Recursions (MoR): Learning Dynamic Recursive Depths for
Adaptive Token-Level Computation` (`arXiv:2507.10524v2`, NeurIPS 2025, Bae et al.;
official repo `raymin0223/mixture_of_recursions` commit `82c97f29cd140efa01f5cc4fe0de73a4c3b6ffd6`),
Position-Indexed Rectangular Causal-Gather `Flash-MoR` (`Q_act[K_r] x KV_shared[T]`)
with Binary-Searched `m_first` CTA Skip & Two-Slice `64+64` Zero-Spill Register
Staging, 8-GPU NVSwitch Recursion-Depth-Pinned Tied-Weight `FSDP2` (`6 -> 1`
`AllGather`) + Whole-Document `MoR-LPT AllToAllv` (`B_global = 32, T = 2,048`),
and our 5-Arm Matched-Step Multi-Hop Latent-Depth Reasoning Scaling Law (`E135`)
are under `raw/mor_recursive_depth_rsi/`. Governed by the same byte-identical
frozen `code/rsi_kernel_harness.py` (`eval_harness_sha256 = 3cf0f078...`), one
8-GPU smoke gate and three full OS-process repetitions at locked 1410 MHz
reject `4/4` red-team exploits, diagnose 6 upstream `raymin0223/mixture_of_recursions`
pathologies (`kv_sharing` full-sequence `[B, T, D]` clone `2.0x` wasted compute,
`token_choice` `B` host `DtoH` syncs + `512 MiB` 4D mask, omitted unfinished
transition gate `rel_l2 = 0.2702`, `logsumexp(dim=-1)` `[bs, k, 1]` z-loss collapse
with `5,873,699x` spurious penalty on `z = -6`, `position_ids[:, :k]` prefix-slice
RoPE corruption `rel_l2 = 0.9264`, plus a live `flash-attn` `v2.8.3.post1` C++ GQA
`max_seqlen_q == 1` empty-document reshape crash) and naive monolithic `[64x128]`
`dKV` `5,638` cubin spills vs `16/16` zero-spill (`0` cubin spills across all `37`
specializations) Round-9 RSI Triton kernels (`FP32 max_rel_l2 = 3.94e-7`, `BF16 =
0.0042`, `7/7` `FP64` Richardson `~4.000x`), ratchet `Track 1` (`s1..s4`) to `1.80x`
geomean over Official `kv_sharing` (`2.45x` over 4D-masked SDPA, `13.68x` over
monolithic spill, saving `117.9 MiB` HBM at `T = 8,192`) and `Track 2` (`Middle-Cycle`
`N_r = 3` stack `m1..m3`) to `1.52x` over Official Pad4D (`1.28x` over Official
KV-Share Clone, saving `85.2 MiB` / `222.9 MiB` peak HBM), verify `6.79e-7` 3-step
`FP32` `Gate G9` parameter parity (`loss_rel_diff = 0.0` vs `0.8631` rank-local
balancing router gradient bias) while scaling `B_global = 32, T = 2,048` across 8x
A100 NVSwitch at `3.61x` over Naive `FSDP2`+Pad4D (`3.32x` over Naive `FSDP2`+Official
KV-Share, `2.77x` pure `MoR-LPT` NVSwitch gain, saving `1,614.2 MiB` / `1,973.8 MiB`/rank),
and drive a strict-causal (`0.0` prefix diff) fixed-`12.0s` `8xA100 DDP` `val_bpb`
RSI loop (`1.38x` updates `353 -> 487`, `-0.0967 val_bpb`, `-0.0948` untouched
`test_bpb`, `100.00%` `3-Hop` accuracy) plus a matched-step (`140` steps, iso-param)
5-Arm Latent-Depth Reasoning Scaling Law (`M4` `1.6787 test_bpb`, `100.00%` `3-Hop`
accuracy, and `Delta r = +1.00` beating `M0` `2.0729` & `35.60%`, `M1` `1.7023` &
`Delta r = 0.00`, `M2` `2.0678` & `8.91%`, and `M3` `2.0725` & `9.58%`).

Round-10 RSI `Differential Transformer V1 & V2: Common-Mode Attention-Noise
Cancellation` (`arXiv:2410.05258`, ICLR 2025 + Jan 2026 `DIFF V2`, Ye, Dong, Xia,
Sun, Zhu, Huang & Wei, Microsoft Research; official repo
`microsoft/unilm/tree/master/Diff-Transformer` & `Diff-Transformer/Diff-Transformer-V2`),
Zero-Spill Paired-Query `Flash-DiffAttn-V2`, Co-Scheduled Q+K Interleaved RoPE
(`BLOCK_M=64`), Fused Stride-2 Differential Epilogue, Paired-Stream
`Flash-DiffAttn-V1` + Fused 4-Way Cat & `SubLN` `RMSNorm(2d)*(1-lambda_init)`
Epilogue, 8-GPU NVSwitch Dual-LSE `Ring DiffAttn-CP = 8` (`T_global = 16,384`),
and our 5-Arm Matched-Step Multi-Needle Distractor-Noise Cancellation Scaling
Law (`E136`) are under `raw/diff_transformer_rsi/`. Governed by the same
byte-identical frozen `code/rsi_kernel_harness.py` (`eval_harness_sha256 =
3cf0f078...`), one 8-GPU smoke gate and three full OS-process repetitions
reject `4/4` red-team exploits, diagnose 5 upstream `microsoft/unilm/Diff-Transformer`
pathologies (`multihead_diffattn.py:80,86,101` pre-split `repeat_kv` GQA bug
with `97.04%` relative L2 error, `multihead_flashdiff_2.py:90-98` `4x`
FlashAttention launches + `2x` `torch.cat`, `multihead_flashdiff_1.py:88-96` +
`rms_norm.py` `4` unfused `SubLN` epilogue passes, `multihead_flashdiffv2.py`
non-contiguous stride-2 `0::2`/`1::2` slicing, and `kernel/rotary.py:124-156`
`BLOCK_M=4` `rk_swap` double load) and monolithic `[64x64x128]` `dKV` `384`
cubin spills vs `20/20` zero-spill (`0` cubin spills via two-slice `64+64`
staging and 1D `(r, m_idx)` loop flattening) Round-10 RSI Triton kernels
(`FP32 max_rel_l2 = 8.09e-7`, `BF16 = 0.0057`, `7/7` `FP64` Richardson
`3.980x–4.272x`), ratchet `Track 1` (`DiffAttn V2`, `s1..s4`) to `1.08x–1.16x`
over Official `MultiheadFlashDiffV2` (`1.16x–1.28x` over `a2`) and `Track 2`
(`DiffAttn V1 + Fused SubLN`, `m1..m3`) to `1.27x–1.59x` over Official
`MultiheadFlashDiff2` `4x-FA2` (`1.17x–1.24x` over `2x-FA2`), verify `1.19e-7`
3-step `FP32` `Gate G9` loss parity (`max_param_rel_l2 = 8.91e-7`,
`max_grad_rel_l2 = 8.79e-7` vs `727.0%` shard-local differential subtraction
error) while scaling `T_global = 16,384` across 8x A100 NVSwitch at `1.50x`
over All-Gather Full-Sequence (`26.05 -> 17.40 ms`) and `1.85x` over `2x`
Separate Ring Passes (`32.13 -> 17.40 ms`), and drive a fixed-`12.0s` `8xA100
DDP` `val_bpb` RSI loop (`2.62x` updates `275 -> 720`, `-0.5864 val_bpb`
`1.9274 -> 1.3410`, clean `test_bpb = 1.3860`) plus a matched-step (`150`
steps) 5-Arm Multi-Needle Distractor-Noise Cancellation Scaling Law (`D4`
`1.4861` clean / `2.0426` distractor `test_bpb` & `100.00%` needle accuracy and
`D3` `2.1501` & `100.00%` beating `D0` `2.1311` & `98.44%`, `D1` `2.4696` &
`97.40%`, and `D2` `3.4257` & `91.15%`).

Round-11 RSI `nGPT: Normalized Transformer with Representation Learning on the
Hypersphere` (`arXiv:2410.01131v2`, ICLR 2025, Loshchilov, Hsieh, Sun &
Ginsburg, NVIDIA; official repo `NVIDIA/ngpt`), Zero-Spill Hypersphere Chordal
Residual LERP, Pair-Isotropic $SO(2)$ RoPE + L2-Norm + $s_{qk}$ Scaling,
$\sqrt{d_h}$ Cosine FlashAttention (`D = 64, 128`), Fused `suv` SwiGLU & `sz`
Cross-Entropy, Post-Preconditioning Tangent-Projected Riemannian AdamW
Retraction, 8-GPU NVSwitch Cosine `Ring nGPT-CP = 8` + `ZeRO-1` Row-Sharded
Retraction (`T_global = 8,192`), and our 5-Arm Matched-Step Hypersphere vs
Pre-RMSNorm Convergence Scaling Law (`E137`) are under
`raw/ngpt_hypersphere_rsi/`. Governed by the same byte-identical frozen
`code/rsi_kernel_harness.py` (`eval_harness_sha256 = 3cf0f078...`), one 8-GPU
smoke gate and three full OS-process repetitions reject `4/4` red-team
exploits, diagnose 5 upstream `NVIDIA/ngpt` pathologies (`model.py:40-58`
per-layer CPU RoPE allocation + `sinusoidal_pos.chunk(2, dim=-1)` non-orthogonal
RoPE breaking relative-position shift invariance with `max |Delta logit| =
1.1523` vs `4.77e-7` for Pair-Isotropic $SO(2)$ RoPE, `model.py:88-90,123-131`
`20` unfused Chordal LERP kernels/block without $\varepsilon$ guard,
`model.py:135-140` unfused `[B,T,2I]` `suv` SwiGLU, `model.py:199-203` unfused
`[B,T,V]` `sz` cosine logits, and `train.py:326-345` post-step Python
`normalize_matrices()` loop with `93.4%–98.0%` Euclidean AdamW radial step
contamination) and monolithic `[64x64x128]` `dKV` `1,682` cubin spills vs
`19/19` zero-spill (`0` cubin spills via 3D `pid_slice in {0,1}` `64+64` `dKV`
decomposition) Round-11 RSI Triton kernels (`FP32 max_rel_l2 = 4.12e-7`, `BF16 =
0.0031`, `7/7` `FP64` Richardson `3.9912x–4.0000x`), ratchet `Track 1` (`Core
Hypersphere Ops`, `s1..s4`) to `24.31x` geomean over Upstream `NVIDIA/ngpt`
(`1.19x` over `torch.compile`) and `Track 2` (`Full nGPT Block Fwd+Bwd +
Riemannian Weight Retraction`, `s1..s4`) to `5.16x` geomean over Upstream
`NVIDIA/ngpt` + `normalize_matrices()` (`1.15x` over `torch.compile`), verify
`6.25e-8` 3-step `FP32` `Gate G9` loss parity (`max_param_rel_l2 = 4.73e-7`,
`max_grad_rel_l2 = 1.92e-6`, `max_unit_norm_err = 1.79e-7` vs `1.8284`
split-reduction-axis norm error) while scaling `T_global = 8,192` across 8x A100
NVSwitch at `1.73x` (`D=64`, `15.30 -> 8.97 ms`) and `2.60x` (`D=128`, `34.55 ->
13.26 ms`) over All-Gather Full-Sequence, and drive a fixed-`2.1s` `8xA100 DDP`
`val_bpb` RSI loop (`1.75x` updates `67 -> 117`, `-1.3849 val_bpb` `1.4812 ->
0.0963`, single-touch `test_bpb`: `1.5008 -> 0.0976`, `recall_acc = 99.77%`)
plus a matched-step (`108` steps) 5-Arm Hypersphere vs Pre-RMSNorm Scaling Law
(`N4` `0.1112 test_bpb` & `99.65%` recall beating `N0` `3.8328` & `1.76%`, `N1`
`3.1093` & `1.91%`, `N2` `0.6170` & `26.52%`, and `N3` `0.1840` & `99.45%`).

Round-12 RSI (`PROGRAM.md` Sec. 15) on `Log-Linear Attention` (`Log-Linear
Mamba-2` & `Log-Linear Gated DeltaNet`, `arXiv:2506.04761v3`, Mar 2026, Guo,
Yang, Goel, Xing, Dao & Kim, MIT / Princeton / Together AI / CMU; official repo
`HanGuo97/log-linear-attention`), 3-Instruction IEEE-754 Bitwise-XOR Fenwick
Level Kernels, Parallel-Level Fenwick State Scans + Zero-Atomic Row-Segmented
`dl` Backward, Gamma-Free Dual-WY (`Aw` Ungated / `Au` Gated) `H-GDN`, 8-GPU
NVSwitch Shard-Decoupled `Fenwick-CP = 8` (`T_global = 8,192`), and our 5-Arm
Matched-Step Multi-Scale Associative Recall Scaling Law (`E138`) are under
`raw/log_linear_attention_rsi/`. Governed by the same byte-identical frozen
`code/rsi_kernel_harness.py` (`eval_harness_sha256 = 3cf0f078...`), one 8-GPU
smoke gate and three full OS-process repetitions reject `4/4` red-team
exploits, diagnose 5 upstream `hattention` pathologies (`base.py:21-25,86-143`
hardcoded `/export/share/...` `llut` paths + $O(T^2)$ `[T,T]` `int64` `llut`
allocation, `chunkwise.py:722-783,932-1047` host `for ell` kernel loops +
`4,096` colliding `tl.atomic_add` ops/CTA onto `dl`, `chunkwise_hgdn.py` 5D
`gamma` tensor + 5-tensor eager `einsum` for `q_new` + `8` `.masked_fill`
clones/level, unconditional GQA `repeat` expansion, and `kernel.py:332`
`NotImplementedError` backward) and monolithic 8-level register-resident kernel
`1,210` cubin spills vs `17/17` zero-spill (`0` cubin spills, `0` XOR
mismatches across `512x512` pairs) Round-12 RSI Triton kernels (`FP32
max_rel_l2 = 1.82e-3`, `BF16 = 0.0038`, `7/7` `FP64` Richardson
`3.9998x–4.0001x`), ratchet `Track 1` (`Log-Linear Mamba-2`, `s1..s4`) to
`9.36x` geomean over Upstream `hattention` (`1.42x` over `torch.compile`,
`30.35x` over quadratic materialized `hattention` at `T=4,096` with `13.85x`
lower HBM `7,516.75 -> 542.76 MiB`) and `Track 2` (`Log-Linear Gated DeltaNet`,
`s1..s4`) to `2.18x` geomean over Upstream `chunkwise_hgdn` (`1.08x` over
`torch.compile`), verify `1.38e-5` 3-step `FP32` `Gate G9` loss parity
(`max_param_rel_l2 = 1.52e-4`, `max_grad_rel_l2 = 3.01e-4` vs `11.69%`
missing-coarse-level error) while scaling `T_global = 8,192` across 8x A100
NVSwitch at `4.21x` (`D=64`, `43.26 -> 10.29 ms`, `3.07x` less HBM) and `2.77x`
(`D=128`, `100.58 -> 36.06 ms`, `1.85x` less HBM) over All-Gather Full-Sequence,
and drive a fixed-`2.1s` `8xA100 DDP` `val_bpb` RSI loop (`-3.1646 val_bpb`
`4.4539 -> 1.2796`, single-touch `test_bpb`: `4.4304 -> 1.2812`, `recall_acc =
100.00%`) plus a matched-step (`108` steps) 5-Arm Multi-Scale Associative
Recall Scaling Law (`L4` `1.2027 test_bpb` & `100.00%` recall and `L3` `3.4723`
& `99.95%` beating `L1` `4.7307` & `99.83%`, `L2` `5.6164` & `41.70%`, and `L0`
`6.1229` & `1.71%`).

Round-13 RSI (`PROGRAM.md` Sec. 16) on `MesaNet: Sequence Modeling by Locally
Optimal Test-Time Training` (`arXiv:2506.05233v2`, von Oswald, Scherrer,
Kobayashi, Versari, Yang et al., Google Paradigms of Intelligence / Google
DeepMind / MIT CSAIL; official repo
`fla-org/flash-linear-attention/tree/main/fla/ops/mesa_net`), Causal
Query-Subtiled (`BM=16`) Warm-Start Jacobi-PCG Kernels (`x^(0) = D_t^-1 q_t`),
Unified Dual-Gram Adjoint (`C_c = dO_c V_c^T - dQ_tilde_c K_c^T`), Fused
`ShortConv1d+SiLU` / `QK-L2Norm+Cumsum` / `RMSNorm-SiLU` Gate, 8-GPU NVSwitch
Dual-State Prefix-Scan `Mesa-CP = 8` (`T_global = 8,192`), and our 5-Arm
Matched-Step Correlated-Key Overwrite Associative Recall Scaling Law (`E139`)
are under `raw/mesanet_ttt_rsi/`. Governed by the same byte-identical frozen
`code/rsi_kernel_harness.py` (`eval_harness_sha256 = 3cf0f078...`), one 8-GPU
smoke gate and three full OS-process repetitions reject `4/4` red-team
exploits, diagnose 5 upstream `fla/ops/mesa_net` pathologies
(`chunk_cg_solver_fwd.py` & `bwd.py` $x^{(0)} = 0$ cold start + commented-out
Jacobi preconditioner in `naive.py:111` + `+ 1e-5` biased CG denominators
stalling 30-step CG at `1.294e-3` relative L2 error, `[128, 128]` dual-state
register pressure causing `1,942` cubin spills in the monolithic `[64x128]`
negative control, unfused `l2norm_fwd`/`bwd` + `chunk_local_cumsum`, split
two-pass intra-chunk `kv`/`kk` backward, and missing Context Parallelism for
coupled $(H_{kk}, H_{kv})$) vs `15/15` zero-spill (`0` cubin spills across `49`
specializations, Step-6 Jacobi-PCG `4.85e-8` rel L2 = `26,672x` more accurate
than Upstream 30-step CG in `5.0x` fewer iterations) Round-13 RSI Triton
kernels (`FP32 max_rel_l2 = 1.29e-6`, `BF16 = 0.0045`, `7/7` `FP64` Richardson
`3.9958x–4.0004x`), ratchet `Track 1` (`Core MesaNet Layer`, `s1..s4`) to
`53.51x` geomean over Upstream 30-Step CG (`31.91x` over `torch.compile`, up to
`2,482.58x` over sequential `torch.linalg.solve`, with `12.24x` lower HBM at
`T=2,048` `1,692.8 -> 138.3 MiB`) and `Track 2` (`Full MesaNetBlock`, `s1..s4`)
to `19.40x` geomean over Upstream `MesaNetBlock` (`12.23x` over `torch.compile`),
verify `1.19e-7` 3-step `FP32` `Gate G9` loss parity (`max_param_rel_l2 =
9.39e-7`, `max_grad_rel_l2 = 2.12e-6` vs `114.5%` missing-prefix error) while
scaling `T_global = 8,192` across 8x A100 NVSwitch at `55.38x` (`D=64`, `169.73
-> 3.06 ms`, `13.08x` less HBM) and `73.17x` (`D=128`, `225.45 -> 3.08 ms`,
`18.22x` less HBM) over All-Gather Full-Sequence, and drive a fixed-`2.1s`
`8xA100 DDP` `val_bpb` RSI loop (`-0.2231 val_bpb` `0.2698 -> 0.0466`,
single-touch `test_bpb`: `0.2924 -> 0.0549`, `recall_acc`: `92.80% -> 99.02%`)
plus a matched-step (`108` steps) 5-Arm Correlated-Key Overwrite Associative
Recall Scaling Law (`M4` `0.0859 test_bpb` & `97.58%` recall beating `M3`
`0.3350` & `89.94%`, `M1` `5.3852` & `8.69%`, `M2` `5.9416` & `3.42%`, and `M0`
`5.9654` & `3.47%`).

`Kimi Linear: An Expressive, Efficient Attention Architecture — Kimi Delta
Attention (KDA)` (`arXiv:2510.26692`, Moonshot AI / Kimi Team,
`fla-org/flash-linear-attention/tree/main/fla/ops/kda`, E140) is under
`raw/kimi_linear_kda_rsi/`. It contains 3 independent 8-GPU process repetitions
(`kimi_linear_kda_rsi_rep{1,2,3}.json`), aggregated 3-rep summary
(`kimi_linear_kda_rsi_summary.json`), report (`kimi_linear_kda_rsi_report.md`),
5 CSV tables (`suiteA_track1_core_kda.csv`, `suiteA_track2_full_block.csv`,
`suiteB_cp8_scaling.csv`, `suiteC_rsi_candidates.csv`,
`suiteC_scaling_law_5arm.csv`), 2 TSV ledgers (`rsi_ledger.tsv`,
`valbpb_autoresearch_ledger.tsv`), `README.md`, and `SHA256SUMS.txt` (`10`
verified entries) governed by `PROGRAM.md` Section 17 and
the byte-identical frozen evaluator `code/rsi_kernel_harness.py` (`SHA256 =
3cf0f078...`, `9/9` gates passed). It proves that upstream `fla/ops/kda`
`chunk_fwd.py:77` & `chunk_bwd.py:786` (`safe_gate=True, lower_bound=-5.0`)
produces **`32,768 / 524,288` (`6.25%`) `NaN`s** at $g_{t,d} = -7.0$ (`0` `NaN`s
in our overflow-free non-positive causal sub-chunking), that `arXiv:2510.26692`
Listing 8(b) (`def chunk_kda`) causes **`18.06%` relative L2 error**, and that
upstream `chunk_kda_bwd_kernel_wy_dqkg_fused` spills **`156–3,648` registers
across `100%` (`36/36`) of its autotuned specializations (`48,438` total spills)**,
compiles **`19/19` zero-spill (`94` specializations) Round-14 RSI Triton
kernels**, achieves **`23.45x`** (`Track 1` vs Upstream WY Spill, `22.93x` vs
`torch.compile`, `528.86x` vs Recurrent, `18.19x` lower HBM) and **`12.04x`**
(`Track 2` vs Baseline Block, `10.90x` vs `torch.compile`, `14.00x` lower HBM)
geomean speedups, scales `T_global = 8,192` across 8x A100 NVSwitch via
Channel-Affine Monoid `KDA-CP = 8` (`31.93x` at `D=64` with `68.58x` less HBM
and `35.12x` at `D=128` with `97.91x` less HBM at `1.19e-7` 3-step `FP32` `Gate
G9` loss parity), and runs a fixed-`1.4s` `8xA100 DDP` `val_bpb` RSI loop
(`-4.4440 val_bpb` `4.4464 -> 0.0025`, single-touch `test_bpb`: `4.4441 ->
0.0024`, `recall_acc`: `41.41% -> 100.00%`) plus a matched-step (`64` steps)
5-Arm Channel-Selective Associative Recall Scaling Law (`K4` `0.0047 test_bpb` &
`100.00%` recall and `K3` `0.1795` & `99.77%` beating `K1` `3.7865` & `65.70%`,
`K2` `5.0138` & `3.28%`, and `K0` `5.0162` & `2.89%`).

`RWKV-7 "Goose" with Expressive Dynamic State Evolution & Generalized
Diagonal-Plus-Low-Rank (DPLR) Delta Rule` (`arXiv:2503.14456` / `arXiv:2506.02475`,
Peng et al. / Hu et al., `fla-org/flash-linear-attention/tree/main/fla/ops/generalized_delta_rule/dplr`
& `fla/ops/rwkv7`, E141) is under `raw/rwkv7_dplr_rsi/`. It contains 3
independent 8-GPU process repetitions (`rwkv7_dplr_rsi_rep{1,2,3}.json`),
aggregated 3-rep summary (`rwkv7_dplr_rsi_summary.json`), report
(`rwkv7_dplr_rsi_report.md`), 5 CSV tables (`suiteA_track1_core_dplr.csv`,
`suiteA_track2_full_block.csv`, `suiteB_cp8_scaling.csv`,
`suiteC_rsi_candidates.csv`, `suiteC_scaling_law_5arm.csv`), `README.md`, and
`SHA256SUMS.txt` (`10` verified entries) governed by `PROGRAM.md` Section 18 and
the byte-identical frozen evaluator `code/rsi_kernel_harness.py` (`SHA256 =
3cf0f078...`, `9/9` gates passed). It proves that upstream `chunk_A_fwd.py:163`
(`b_offset = gi[BT//2]`) produces **`256` `NaN`s** at $w = -2.85$ nats (`0`
`NaN`s in our boundary-anchored $16 \times 16$ sub-chunking) and that **`100%`
(`22/22`) of upstream's autotuned configs spill `72–3,416` registers (`27,152`
total spills)** across `chunk_A_fwd`, `chunk_A_bwd`, and `wy_fast_bwd`, compiles
**`21/21` zero-spill (`89` specializations) Round-15 RSI Triton kernels**,
achieves **`29.28x`** (`Track 1` vs Upstream Spill, `25.00x` vs `torch.compile`,
`774.06x` vs Recurrent, `19.72x` lower HBM) and **`14.40x`** (`Track 2` vs
Baseline Block, `12.72x` vs `torch.compile`, `14.32x` lower HBM) geomean
speedups, scales `T_global = 8,192` across 8x A100 NVSwitch via Dual-Source
Channel-Affine Monoid `DPLR-CP = 8` (`30.64x` at `D=64` with `51.65x` less HBM
and `32.56x` at `D=128` with `81.40x` less HBM at `9.06e-8` 3-step `FP32` `Gate
G9` loss parity), and runs a fixed-`1.4s` `8xA100 DDP` `val_bpb` RSI loop
(`-1.0388 val_bpb` `1.0401 -> 0.0013`, single-touch `test_bpb`: `1.0262 ->
0.0012`, `recall_acc`: `96.56% -> 100.00%`) plus a matched-step (`64` steps)
5-Arm Dynamic State Evolution Scaling Law (`R4` `0.0023 test_bpb` & `100.00%`
recall and `R3` `0.0048` & `100.00%` beating `R1` `2.2082` & `62.11%`, `R0`
`4.9269` & `7.42%`, and `R2` `5.0437` & `3.36%`).

`Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear
Recurrences` (`PrecondKDA` & `ATK`, `arXiv:2604.21100`, Moonshot AI / Kimi Team &
FLA Team, `fla-org/flash-linear-attention/tree/main/fla/ops/precond_kda` &
`fla/ops/atk`, E142) is under `raw/precond_kda_rsi/`. It contains 3 independent
8-GPU process repetitions (`precond_kda_rsi_rep{1,2,3}.json`), aggregated 3-rep
summary (`precond_kda_rsi_summary.json`), report (`precond_kda_rsi_report.md`),
5 CSV tables (`suiteA_track1_core_pkda.csv`, `suiteA_track2_full_block.csv`,
`suiteB_cp8_scaling.csv`, `suiteC_rsi_candidates.csv`,
`suiteC_scaling_law_5arm.csv`), and `SHA256SUMS.txt` (`10` verified entries)
governed by `PROGRAM.md` Section 19 and the byte-identical frozen evaluator
`code/rsi_kernel_harness.py` (`SHA256 = 3cf0f078...`, `9/9` gates passed). It
proves that upstream `SAFE_GATE` sub-chunk anchor shift (`chunk_intra.py:442-455`)
produces **`401,408` `NaN`s** at $g = -7.2$ nats (`0` `NaN`s in our pre-masked
$16 \times 16$ sub-chunking), that 5 upstream kernels spill **`50,812` total
registers across `70` configs** (`168–3,972` across `36/36` configs of
`chunk_precond_kda_bwd_kernel_wy_dqkg`, `66–246` across `3/3` configs of
`chunk_precond_kda_fwd_kernel_inter_solve_fused`, `88` in
`_atk_backward_chunk_out`, `364` in `recompute_w_u_fwd_kernel`, and `98` in
`chunk_precond_kda_bwd_kernel_intra`), and that `chunk.py:540-545` raises
`NotImplementedError` for Context Parallelism, compiles **`24/24` zero-spill
(`108` specializations, `max_regs = 168 <= 168`) Round-16 RSI Triton kernels**,
achieves **`19.27x`** (`Track 1` vs Upstream Spill, `15.21x` vs `torch.compile`,
`630.47x` vs Recurrent, `15.94x` lower HBM) and **`12.11x`** (`Track 2` vs
Baseline Block, `10.56x` vs `torch.compile`, `11.93x` lower HBM) geomean
speedups, scales `T_global = 8,192` across 8x A100 NVSwitch via Coupled
Curvature-Matrix `PKDA-CP = 8` (`31.24x` at `D=64` with `40.68x` less HBM and
`32.17x` at `D=128` with `63.03x` less HBM at `1.19e-7` 3-step `FP32` `Gate G9`
loss parity), and runs a fixed-`1.4s` `8xA100 DDP` `val_bpb` RSI loop (`-4.0029
val_bpb` `4.1131 -> 0.1102`, single-touch `test_bpb`: `4.2082 -> 0.1115`,
`recall_acc`: `31.33% -> 97.34%`) plus a matched-step (`64` steps) 5-Arm
Curvature Preconditioning Scaling Law (`R4` `0.0786 test_bpb` & `98.28%` recall
and `R3` `2.0110` & `73.59%` beating `R1` `4.5417` & `25.31%`, `R0` `5.0146` &
`2.50%`, and `R2` `5.0300` & `2.89%`).

`Triadic Gated DeltaNet & Triadic Kimi Delta Attention: Kronecker
Tensor-Product State Expansion & Hadamard-Factored UT/WY Linear Recurrences`
(`Triadic-GDN` & `Triadic-KDA`, `arXiv:2609.36529`, FLA Team,
`fla-org/flash-linear-attention/tree/main/fla/ops/triadic_gdn` &
`fla/ops/triadic_kda`, E143) is under `raw/triadic_gdn_rsi/`. It contains 3
independent 8-GPU process repetitions (`triadic_gdn_rsi_rep{1,2,3}.json`),
aggregated 3-rep summary (`triadic_gdn_rsi_summary.json`), report
(`triadic_gdn_rsi_report.md`), 4 CSV tables (`track3_kernel_sweep.csv`,
`track3_block_stage_ladder.csv`, `track4_autoresearch.csv`,
`track5_cp8_scaling.csv`), and `SHA256SUMS.txt` (`9` verified entries)
governed by `PROGRAM.md` Section 20 and the byte-identical frozen evaluator
`code/rsi_kernel_harness.py` (`SHA256 = 3cf0f078...`). It proves that upstream
`chunk_triadic_gdn` omits Dual-Sphere `L2Norm` (`65,536` `o` & `163,660` grad
`NaN`s from $\mathbb{E}[\|\kappa_t\|_2^2] = 252.11$), spills **`1,684` SM80
registers across `19/37` specializations**, crashes `SM80` on `(K=128, V=128,
FP32)` with `OutOfResources` (`229,632 > 166,912` B SMEM), truncates `FP32` to
`TF32` (`2.23e-3` error), and provides no Triton kernels for `Triadic-KDA`,
compiles **`25/25` zero-spill (`78` specializations, `max_regs = 168 <= 168`)
Round-17 RSI Triton kernels** (`8,167.8x` `o` & `8,013.4x` `dq` `FP32` precision
improvement, `max_fp32_rel_l2 = 6.38e-7`), accelerates `Triadic-GDN` by
**`1.77x` (`T=2048`) and `1.26x` (`T=4096`)** over upstream `fla`
(`64.81x–75.09x` over eager chunk), accelerates `Triadic-KDA` by **`30.89x`
(`95.260 -> 3.081 ms`)**, accelerates `TriadicGDNBlockRSI` (`Stage A4`) by
**`29.58x` (`104.206 -> 3.491 ms`)**, completes **`6.67x` more optimizer steps
(`398 -> 2,678`, `-0.1189 val_bpb`)** in a fixed-`25s` autoresearch loop, and
scales $T_{\text{global}} = 16,384..65,536$ across 8x A100 NVSwitch via
Ulysses-WY `Triadic-CP = 8` (`1.73x` vs 1x A100, `1.79x–1.81x` vs 8-GPU ring,
`3,960.7 kTok/s` at `64K`, `0.000e+00` parity).

`Momentum DeltaNet: Coupled Dual-State [S, M] Heavy-Ball Momentum Test-Time
Gradient Descent` (`MomentumDeltaNet`, `arXiv:2605.05838`, FLA Team,
`fla-org/flash-linear-attention/tree/main/fla/ops/momentum_delta_rule` &
`fla/layers/momentum_deltanet.py`, E144) is under `raw/momentum_deltanet_rsi/`.
It contains 3 independent 8-GPU process repetitions (`rep{1,2,3}.json`),
smoke run (`smoke.json`), aggregated 3-rep summary
(`momentum_deltanet_rsi_summary.json`), report (`momentum_deltanet_rsi_report.md`),
4 CSV tables (`track3_kernel_sweep.csv`, `track3_block_stage_ladder.csv`,
`track4_autoresearch.csv`, `track5_cp8_scaling.csv`), and `SHA256SUMS.txt` (`10`
verified entries) governed by `PROGRAM.md` Section 21 and the byte-identical
frozen evaluator `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f078...`). It
proves that upstream `chunk_momentum_delta_rule` has **`0` CUDA Triton kernels**
and **`0` backward Triton kernels** (falling back to `naive.py` +
`torch.autograd.grad` full forward replay at `188.14–636.80 ms`), spills
**`648–4,624` registers** on its Ascend forward kernels while crashing `SM80`
with `OutOfResources` at `(K=128, V=128)` (`219,672 > 166,912` B SMEM) and
`(K=256, V=256)` (`416,280 > 166,912` B SMEM), raises `AssertionError` on
`FP32`, and injects `beta=0` `(beta_p+1e-6).log()` momentum drift (`2.79e-3`),
compiles **`26/26` zero-spill (`190` specializations, `max_regs = 168 <= 168`)
Round-18 RSI Triton kernels** (`15,732x` `beta=0` accuracy improvement,
`max_fp32_rel_l2 = 4.53e-6`), accelerates `chunk_momentum_delta_rule` by
**`84.76x` (`T=2048`), `78.91x` (`T=4096`), and `75.27x` (`T=8192`)** over
upstream `fla` (**`27.82x` at `K=128, V=128` and `14.75x` at `K=256, V=256`**),
accelerates `MomentumDeltaNetBlockRSI` (`Stage A4`) by **`51.69x` (`200.30 ->
3.966 ms`, `38.27x` vs `A0`)**, completes **`12.33x` more optimizer steps (`188
-> 2,318`, `-1.1277 val_bpb`)** in a fixed-`25s` autoresearch loop, and scales
$T_{\text{global}} = 8,192..32,768, H = 128$ across 8x A100 NVSwitch via
zero-copy `B=1` Ulysses-WY `Momentum-CP = 8` (`5.15x` vs 1x A100, `64.4%`
scaling efficiency, `5.32x` vs 8-GPU ring, `738.9 kTok/s` at `32K`, `0.000e+00`
parity).

`Gated KalmaNet (GKA): A Fading Memory Layer Through Test-Time Ridge
Regression` (`GatedKalmaNet`, `arXiv:2511.21016`, Songlin Yang, Yu Zhang,
Zhiyuan Li, `fla-org/flash-linear-attention/tree/main/fla/ops/gka` &
`fla/layers/gka.py`, E145) is under `raw/gated_kalmanet_rsi/`. It contains 3
independent 8-GPU process repetitions (`rep{1,2,3}.json`), smoke run
(`smoke.json`), aggregated 3-rep summary (`gated_kalmanet_rsi_summary.json`),
report (`gated_kalmanet_rsi_report.md`), 4 CSV tables
(`track3_kernel_matrix.csv`, `track3_layer_ablations.csv`,
`track4_autoresearch.csv`, `track5_cp8_scaling.csv`), and `SHA256SUMS.txt` (`10`
verified entries) governed by `PROGRAM.md` Section 22 and the byte-identical
frozen evaluator `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f078...`). It
proves that upstream `fla/ops/gka` spills in **`193 / 219` (`88.1%`)** compiled
specializations (`1,519,648` B total stack, `ptxas` `regs=32` allocator collapse
up to `47,864` B stack/thread, `262,144 > 166,912` B SMEM overflow, and
`82.3–200.1 s` cold compile per shape) and divides `0.0 / 0.0 = NaN` when
$\|H_t\|_F = 0$ so a single $k_0 = 0$ token contaminates **`32,768 / 65,536`
(`50%`) of `dk` to `NaN`**, compiles **`21/21` zero-spill (`121`
specializations, `max_regs = 168 <= 168`) Round-19 RSI Triton kernels**
(`3,689.5x` `FP32` output accuracy improvement, `0` `NaN`s), accelerates
`chunk_gka_rsi` by **`1.98x` (`K=64, T=2048`) to `2.61x` (`K=128, T=2048`,
`3.32x` backward speedup)** over upstream `fla.ops.gka.chunk_gka`, accelerates
`GatedKalmaNetBlockRSI` (`Stage A4`) by **`2.41x` (`K=64`) and `2.19x`
(`K=128`)** over `Stage A0`, completes **`2.23x` more optimizer steps (`437 ->
973`, `-1.0849 val_bpb`)** in a fixed-`15s` autoresearch loop, and scales
$T_{\text{total}} = 16,384..32,768, K = 128$ across 8x A100 NVSwitch via
zero-copy `B=1` Ulysses `GKA-CP = 8` (`3.43x` vs 8-GPU ring, `3.28x` vs 1x A100,
`4,505.6 kTok/s` at `32K`, `0.000e+00` parity).

Round-20 RSI (`E146`, `v137`) `Gated DeltaNet-2 (GDN-2), Gated Oja Rule 2
(Oja-2), & Gated Slot Attention 2 (GSA-2)` (`NVlabs/GatedDeltaNet-2` &
`fla-org/flash-linear-attention/tree/main/fla/ops/gated_oja_rule2`) results are
under `raw/gdn2_oja2_gsa2_rsi/`. It contains the smoke verification gate
(`smoke.json`), three independent 8-GPU process repetitions (`rep1.json`,
`rep2.json`, `rep3.json`), the aggregated 3-rep summary
(`gdn2_oja2_gsa2_rsi_summary.json`), markdown report
(`gdn2_oja2_gsa2_rsi_report.md`), 4 CSV tables (`track3_kernel_matrix.csv`,
`track3_layer_ablations.csv`, `track4_autoresearch.csv`,
`track5_cp8_scaling.csv`), and `SHA256SUMS.txt` (`10` verified entries)
governed by `PROGRAM.md` Section 23 and the byte-identical frozen evaluator
`code/rsi_kernel_harness.py` (`SHA256 = 3cf0f078...`). It proves that upstream
`fla/ops/gdn2`, `fla/ops/gated_oja_rule`, and `fla/ops/gated_oja_rule2` spill
across **`22` compiled specializations** (`max_regs = 255, max_spills = 366` in
`chunk_gdn2_bwd_kernel_wy_dqkg_fused`, `140` spills in `oja2_bwd_kernel_wy`,
`86` spills in `chunk_gdn2_fwd_kernel_inter_solve_fused`, `16` spills in
`chunk_oja_bwd_kernel_dhu_blockdim64` and `dvwg_h`, and `12` spills in
`oja2_fwd_kernel_wy`), compiles **`37/37` zero-spill (`76` specializations,
`max_regs = 162 <= 168`) Round-20 RSI Triton kernels**, accelerates
`chunk_gdn2_rsi` by up to **`1.497x` Fwd+Bwd (`1.613x` Bwd)** over
`fla.ops.gdn2.chunk_gdn2` and `chunk_gated_oja_rule2_rsi` by **`1.234x` ($M=64$)
to `1.469x` ($M=256, T=8192$)** over `fla.ops.gated_oja_rule2`, accelerates
`GDN2BlockRSI` (`Stage A4`) by **`1.194x`** and `GSA2BlockRSI` (`Stage A4`) by
**`1.226x` ($M=64$), `1.263x` ($M=128$), and `1.315x` ($M=256$)** over `Stage
A0`, completes **`1.770x` more optimizer steps (`474 -> 838`, `-0.0180` extra
`val_bpb` improvement)** in a fixed-`30s` autoresearch loop, and scales
$T_{\text{total}} = 16,384..65,536$ across 8x A100 NVSwitch via zero-copy
Ulysses `GSA2-CP = 8` (`2.860x–3.318x` vs 8-GPU dual-state ring, `0.0` max
absolute difference).

Round-21 RSI (`E147`, `v138`) `Gated DeltaProduct (GDP): Multi-Householder
Product Linear Recurrences for Rotational & Parity State Tracking`
(`arXiv:2502.10297`, Siems et al.;
`fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_product` &
`fla/layers/gated_deltaproduct.py`) results are under
`raw/gated_deltaproduct_rsi/`. It contains the smoke verification gate
(`smoke.json`), three independent 8-GPU process repetitions (`rep1.json`,
`rep2.json`, `rep3.json`), the aggregated 3-rep summary
(`gated_deltaproduct_rsi_summary.json`), markdown report
(`gated_deltaproduct_rsi_report.md`), `README.md`, 4 CSV tables
(`track3_kernel_matrix.csv`, `track3_layer_ablations.csv`,
`track4_autoresearch.csv`, `track5_cp8_scaling.csv`), and `SHA256SUMS.txt` (`11`
verified entries) governed by `PROGRAM.md` Section 24 and the byte-identical
frozen evaluator `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f078...`). It
proves that upstream `fla/ops/gated_delta_product` leaves `# TODO: optimize the
backward pass like the forward pass` (`chunk.py:204`) unresolved by allocating
zero-padded length-$T \cdot N_h$ `q_new, do_new, g_new` tensors (`50%–75%`
HBM/Tensor-Core waste for $N_h \in \{2, 3, 4\}$), spills across **`21` compiled
specializations** (`450` total spills, `max_regs = 255, max_spills = 96`), and
crashes in `FP32` (`AssertionError`), compiles **`22/22` zero-spill (`90`
specializations, `max_regs = 168 <= 168`) Round-21 RSI Triton kernels**
(`max_fp32_rel_l2 = 2.35e-6`, `max_bf16_rel_l2 = 7.95e-3`, `3/3` mutations
killed), accelerates `chunk_gated_delta_product_rsi` across all `13/13` shapes
up to **`1.94x` Fwd+Bwd (`1.95x` Fwd, `1.93x` Bwd)** over
`fla.ops.gated_delta_product.chunk_gated_delta_product`, accelerates
`GatedDeltaProductBlockRSI` (`Stage A4 vs A0`) by **`1.92x–2.15x`** across
$N_h \in \{2, 3, 4\}$ and $V \in \{128, 256\}$, completes **`1.86x` more
optimizer steps (`894 -> 1,665`, `-0.0063` extra `val_bpb` improvement)** in a
fixed-`30s` autoresearch loop, and scales $T_{\text{total}} = 16,384..65,536$
($T_{\text{virt}}$ up to `131,072`) across 8x A100 NVSwitch via zero-copy
Ulysses `GDP-CP = 8` (`2.96x–3.22x`, `3.13x` mean vs 8-GPU sequential ring, `0.0`
max absolute difference).

Round-22 RSI (`E148`, `v139`) `Cyclic Flow Attention (CyFA): Linear Sequence
Modeling with Relative-Time-Partitioned Memory` (`arXiv:2609.36259`, Chen et al.;
`fla-org/flash-linear-attention/tree/main/fla/ops/cyfa` & `fla/layers/cyfa.py`)
results are under `raw/cyfa_rsi/`. It contains the smoke verification gate
(`smoke.json`), three independent 8-GPU process repetitions (`rep1.json`,
`rep2.json`, `rep3.json`), the aggregated 3-rep summary
(`cyfa_rsi_summary.json`), markdown report (`cyfa_rsi_report.md`), `README.md`,
4 CSV tables (`cyfa_kernel_matrix.csv`, `cyfa_layer_ablation.csv`,
`cyfa_autoresearch.csv`, `cyfa_cp8_scaling.csv`), and `SHA256SUMS.txt` (`11`
verified entries) governed by `PROGRAM.md` Section 25 and the byte-identical
frozen evaluator `code/rsi_kernel_harness.py` (`SHA256 = 3cf0f078...`). It
proves that upstream `fla/ops/cyfa` spills across **`69` compiled
specializations** (`58,354` total spills, `max_regs = 255, max_spills = 4,932`,
`max_smem = 147,456` B across the autotune sweep; `19/91` active winning specs
with `874` spills), uses global `tl.atomic_add` on `dg_force`, recomputes
`chunk_fwd_h` 4 times per step plus `qk_rmsnorm_fwd` and `_point_write` twice in
backward at `checkpoint_level == 0`, and truncates `FP32` inputs to `TF32`
(`max_rel_l2 = 6.77e-3`), compiles **`19/19` zero-spill (`132` specializations,
`max_regs = 168 <= 168`, `max_smem = 40,960` B) Round-22 RSI Triton kernels**
(`max_fp32_rel_l2 = 7.78e-6` = `869.9x` more accurate, `max_bf16_rel_l2 =
8.05e-3`, `3/3` mutations killed), accelerates `chunk_cyfa_rsi` across all
`12/12` shapes up to **`1.93x` Fwd+Bwd (`1.96x` Bwd, `1.87x` Fwd; `1.79x` mean
Fwd+Bwd)** over `fla.ops.cyfa.chunk.chunk_cyfa`, accelerates
`CyclicFlowAttentionBlockRSI` (`Stage A4 vs A0`) by **`1.58x–1.60x` (`1.59x`
mean)** across $M \in \{64, 128\}$ and $K, V \in \{128, 256\}$, completes
**`1.441x` more optimizer steps (`895 -> 1,289`, `-0.1769` extra `val_bpb`
improvement `8.2405 -> 1.3391` vs `1.5160`)** in a fixed-`30s` autoresearch
loop, and scales $T_{\text{total}} = 16,384..65,536$ across 8x A100 NVSwitch
via zero-copy Ulysses `CyFA-CP = 8` (`2.04x–3.31x`, `2.74x` mean vs 8-GPU
3-state ring, `0.0` max absolute difference).

Round-23 RSI (`E149`, `v140`) `Parallax: Parameterized Local Linear Attention`
(`arXiv:2605.29157`, Yifei Zuo et al.;
`fla-org/flash-linear-attention/tree/main/fla/ops/parallax` &
`fla/layers/parallax.py`) results are under `raw/parallax_rsi/`. It contains the
smoke verification gate (`smoke.json`), three independent 8-GPU process
repetitions (`rep1.json`, `rep2.json`, `rep3.json`), the aggregated 3-rep
summary (`parallax_rsi_summary.json`), markdown report
(`parallax_rsi_report.md`), `README.md`, 4 CSV tables
(`parallax_kernel_matrix.csv`, `parallax_layer_ablation.csv`,
`parallax_autoresearch.csv`, `parallax_cp8_scaling.csv`), and `SHA256SUMS.txt`
(`11` verified entries) governed by `PROGRAM.md` Section 26 and the
byte-identical frozen evaluator `code/rsi_kernel_harness.py` (`SHA256 =
3cf0f078...`). It proves that upstream `fla/ops/parallax` spills across
**`22/42` compiled specializations** (`34/42` with `regs > 168`, `max_regs =
255, max_spills = 26, total_spills = 190, max_smem = 91,392` B), raises
`TypeError` on `FP32` inputs, reloads `K, V` tiles $G\times$ redundantly under
GQA while launching a separate `bwd_preprocess` kernel and two `einops.reduce`
kernels, and calls `r, _ = self.rotary(r, r)` in `fla/layers/parallax.py:134`
(`14` periphery kernel launches per step), compiles **`10/10` zero-spill (`68`
specializations, `max_regs = 168 <= 168`, `max_smem = 49,152` B) Round-23 RSI
Triton kernels** (`max_fp32_rel_l2 = 8.46e-6`, `max_bf16_rel_l2 = 3.08e-3`,
`3/3` mutations killed), wins all **`12/12` kernel shapes** up to **`2.22x`
(`1.54x` Train Fwd+Bwd, `1.67x` Bwd, `1.42x` Train Fwd; `1.39x` mean)** over
`fla.ops.parallax`, accelerates `ParallaxBlockRSI` (`Stage A4 vs A0`) by
**`1.07x–2.21x` (`1.51x` mean)**, completes **`2.166x` more optimizer steps
(`1,676 -> 3,629`, `-0.0293` extra `val_bpb` improvement `8.2497 -> 0.9557` vs
`0.9850`)** in a fixed-`30s` autoresearch loop, and scales $T_{\text{global}} =
16,384..65,536$ across 8x A100 NVSwitch via zero-copy Ulysses `Parallax-CP = 8`
(`1.98x–2.60x`, `2.27x` mean vs 8-GPU coupled 5-tuple online-softmax ring, `0.0`
max absolute difference).

Round-24 RSI (`E150`, `v141`) `Wall Attention: Length Generalization With Diagonal
Gates` (`Tilde Research`, Timor Averbuch & Dhruv Pai;
`fla-org/flash-linear-attention/tree/main/fla/ops/wall_attn` &
`fla/layers/wall_attn.py`) results are under `raw/wall_attn_rsi/`. It contains the
smoke verification gate (`smoke.json`), three independent 8-GPU process
repetitions (`rep1.json`, `rep2.json`, `rep3.json`), the aggregated 3-rep
summary (`wall_attn_rsi_summary.json`), markdown report
(`wall_attn_rsi_report.md`), `README.md`, 4 CSV tables
(`wall_attn_kernel_matrix.csv`, `wall_attn_layer_ablation.csv`,
`wall_attn_autoresearch.csv`, `wall_attn_cp8_scaling.csv`), and `SHA256SUMS.txt`
(`11` verified entries) governed by `PROGRAM.md` Section 27 and the
byte-identical frozen evaluator `code/rsi_kernel_harness.py` (`SHA256 =
3cf0f078...`). It proves that upstream `fla/ops/wall_attn` spills across
**`961 / 1,009` (`95.24%`) compiled specializations** (`5,266,560 B` = `5.27 MB`
total stack spills, `max_regs = 255, max_stack = 36,808 B/thread`, `71` SMEM
overflows up to `295,424 B`), returns `g.grad = None, g_scalar.grad = None` in
`naive_wall_attn` while truncating `FP32` `tl.dot` to `TF32` (`rel_l2_o =
1.03e-3, rel_l2_dg = 3.33e-3`), spends `86%` of decode time inside 7-op eager
`build_wall_kv_cache`, and launches 6 separate input GEMMs + unfused periphery
in `WallAttention.forward`, compiles **`0`-spill (`max_regs = 168 <= 168`,
`max_smem = 40,960 B`) Round-24 RSI Triton kernels** (`FP32 rel_l2_o = 3.23e-6`
= `317.9x` more accurate, `rel_l2_dg = 1.94e-5` = `166.3x` more accurate), wins
all **`4/4` training shapes** (`1.71x` Fwd, `1.21x` Bwd, `1.29x` Total, `2.74x`
vs unpruned $K=128$ spilling cliff), accelerates `build_wall_kv_cache_rsi` by
**`8.25x–9.55x`** (`3.67x–4.98x` end-to-end cached decode, `0.0` bitwise diff),
accelerates `WallAttentionBlockRSI` by **`1.56x` ($D=1024, K=64$) & `1.27x`
($D=2048, K=128$)**, scales $T_{\text{total}} = 16,384..32,768$ across 8x A100
NVSwitch via fused Ulysses `Wall-CP = 8` (**`6.08x` at $16\text{K}$ & `7.15x` at
$32\text{K}$**, `88.3%` parallel scaling efficiency over 1 GPU, `0.00e+00`
`rel_l2` discrepancy), and completes **`1.47x` ($D=1024$) & `1.21x` ($D=2048$)**
more optimizer steps with **`-0.9920` extra `val_bpb` reduction (`0.1438` vs
`1.1358`, `7.90x` lower final `val_bpb`)** in a fixed-`12s` autoresearch loop.

Round-25 RSI (`E151`, `v142`) `DeltaFormer: Understanding Transformer from the
Perspective of Associative Memory` (`arXiv:2505.19488`, Jusen Du, Weigao Sun,
Disen Lan, Jiaxi Hu, Yu Cheng;
`fla-org/flash-linear-attention/tree/main/fla/ops/deltaformer` &
`fla/layers/deltaformer.py`) results are under `raw/deltaformer_rsi/`. It
contains the smoke verification gate (`smoke.json`), three independent 8-GPU
process repetitions (`rep1.json`, `rep2.json`, `rep3.json`), the aggregated
3-rep summary (`deltaformer_rsi_summary.json`), markdown report
(`deltaformer_rsi_report.md`), `README.md`, 4 CSV tables
(`deltaformer_kernel_matrix.csv`, `deltaformer_layer_ablation.csv`,
`deltaformer_autoresearch.csv`, `deltaformer_cp8_scaling.csv`), and
`SHA256SUMS.txt` (`11` verified entries) governed by `PROGRAM.md` Section 28 and
the byte-identical frozen evaluator `code/rsi_kernel_harness.py` (`SHA256 =
3cf0f078...`). It proves that upstream `fla/ops/deltaformer` spends `68.8%` of
GPU time inside `torch.linalg.solve_triangular` + `160`
`.transpose(0,1).contiguous()` copies, spills across **`24 / 104` (`23.08%`)
compiled specializations** (`1,606` total spills, `max_regs = 255`, `max_spills =
274` in `parallel_deltaformer_bwd_kernel_qk`, `131,072 B` SMEM), crashes on
`FP32` inputs (`RuntimeError`), GQA (`AssertionError`), and `past_key_values`
(`NotImplementedError`), and has an unmasked OOB `tl.load(u_blk_ptr)`, compiles
**`13/13` `0`-spill (`195` specializations, `max_regs = 168 <= 168`, `max_smem =
18,432 B`) Round-25 RSI Triton kernels** (`FP32 max_rel_l2 = 8.36e-6` = `216.9x`
more accurate than upstream `1.81e-3`, `3/3` mutations killed), wins all **`8/8`
training shapes** up to **`4.84x` Fwd, `2.76x` Bwd, `3.48x` Total**, accelerates
single-pass cached decode (`deltaformer_decode_rsi`) by **`4.08x–7.28x`** via
the Diagonal-LSE Stage-2 Collapse identity $P^{(2)}_{t, s} = (1 - \alpha_t)
P^{(1)}_{t, s}$, accelerates `DeltaFormerBlockRSI` by **`2.81x` ($D=1024, K=64$)
& `2.14x` ($D=2048, K=128$)**, scales $T_{\text{total}} = 16,384..32,768$ across
8x A100 NVSwitch via fused Ulysses `DeltaFormer-CP = 8` (**`3.71x` at $16\text{K}$
& `2.56x` at $32\text{K}$**, `0.0` `rel_l2_o/dv/dbeta`), and completes **`2.70x`
($D=1024$) & `2.05x` ($D=2048$)** more optimizer steps with **`-6.7788`
($14.55\times$ lower, `0.5001` vs `7.2789`) & `-2.7225` ($7.85\times$ lower,
`0.3973` vs `3.1198`) final `val_bpb`** in a fixed-`12s` autoresearch loop.

Round-26 RSI (`E152`, `v143`) `Stick-Breaking Attention` (`arXiv:2410.17980`,
ICLR 2025 — Shawn Tan, Yikang Shen, Songlin Yang, Aaron Courville, Rameswar
Panda; `fla-org/flash-linear-attention/tree/main/fla/ops/stickbreaking_attn` &
`fla/layers/stickbreaking_attn.py`) results are under
`raw/stickbreaking_attn_rsi/`. It contains the smoke verification gate
(`smoke.json`), three independent 8-GPU process repetitions (`rep1.json`,
`rep2.json`, `rep3.json`), the aggregated 3-rep summary
(`stickbreaking_attn_rsi_summary.json`), markdown report
(`stickbreaking_attn_rsi_report.md`), `README.md`, 4 CSV tables
(`stickbreaking_kernel_matrix.csv`, `stickbreaking_layer_ablation.csv`,
`stickbreaking_autoresearch.csv`, `stickbreaking_cp8_scaling.csv`), and
`SHA256SUMS.txt` (`11` verified entries) governed by `PROGRAM.md` Section 29 and
the byte-identical frozen evaluator `code/rsi_kernel_harness.py` (`SHA256 =
3cf0f078...`). It proves that upstream `fla/ops/stickbreaking_attn` scans all
causal key blocks twice in `parallel_stickbreaking_attn_bwd_kernel_dq`, scatters
32-way uncoalesced `[B, T, HQ, NS]` snapshots, spills across **`29 / 108`
(`26.85%`; `36 / 114` = `31.58%` full grid) compiled specializations** (`1,294..2,440`
total spills, `max_regs = 255`, `max_spills = 154..314`, `73,728..139,264 B`
SMEM), reloads the KV cache $T_Q \times G$ times per KV head in scalar SIMT
`stickbreaking_attn_decoding`, and crashes with `TypeError` on `FP32` inputs in
`StickBreakingAttention.forward`, compiles **`7/7` `0`-spill (`44`
specializations, `max_regs = 168 <= 168`, `max_smem = 45,312 B`) Round-26 RSI
Triton kernels** (`FP32 max_rel_l2 = 3.16e-6` = `466.2x` more accurate than
upstream `1.47e-3`, `3/3` mutations killed), wins all **`8/8` training shapes**
in Bwd & Total up to **`1.29x` Fwd, `1.51x` Bwd, `1.43x` Total** via the
Reverse-Stick Subtraction Identity $\text{log\_rem}(i, i_k) = L_i -
\sum_{m=0}^{i_k} B_{i, m}$ and coalesced `[B*HQ, NS, T]` layout, accelerates
Tensor-Core GQA decoding (`stickbreaking_attn_decoding_rsi`) by
**`1.91x–32.99x`**, accelerates `StickBreakingAttentionBlockRSI` by
**`1.53x–1.56x` ($D=1024, K=64$) & `1.41x` ($D=2048, K=128$)**, scales
$T_{\text{total}} = 16,384..32,768$ across 8x A100 NVSwitch via fused Ulysses
`StickBreaking-CP = 8` (**`7.21x` at $16\text{K}$ & `7.17x` at $32\text{K}$**,
`7.70x` scaling vs 1 GPU = `96.2%` parallel efficiency, `0.00e+00` `rel_l2`),
and completes **`1.49x` ($D=1024$) & `1.37x` ($D=2048$)** more optimizer steps
with **`-0.0754` ($D=1024$) & `-0.8320` ($D=2048$) final `val_bpb` reduction**
in a fixed-`12s` autoresearch loop.

Round-27 RSI (`E153`, `v144`) `Test-Time Training (TTT-Linear): Learning to
(Learn at Test Time): RNNs with Expressive Hidden States` (`arXiv:2407.04620`,
ICML 2024 / JMLR 2025 — Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun
Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo,
Tatsunori Hashimoto, Carlos Guestrin;
`fla-org/flash-linear-attention/tree/main/fla/ops/ttt`) results are under
`raw/ttt_linear_rsi/`. It contains the smoke verification gate (`smoke.json`),
three independent 8-GPU process repetitions (`rep1.json`, `rep2.json`,
`rep3.json`), the aggregated 3-rep summary (`ttt_linear_rsi_summary.json`),
markdown report (`ttt_linear_rsi_report.md`), `README.md`, 5 CSV tables
(`ttt_linear_kernel_matrix.csv`, `ttt_linear_decode_matrix.csv`,
`ttt_linear_layer_ablation.csv`, `ttt_linear_cp8_scaling.csv`,
`ttt_linear_autoresearch.csv`), and `SHA256SUMS.txt` (`12` verified entries)
governed by `PROGRAM.md` Section 30 and the byte-identical frozen evaluator
`code/rsi_kernel_harness.py` (`SHA256 = 3cf0f078...`). It proves that upstream
`fla/ops/ttt` (`chunk.py` + `fused_chunk.py`) spills across **`78 / 119`
(`65.55%`) compiled specializations** (`85,682` total spills, `max_regs = 255`,
`max_spills = 4,614`, `max_smem = 151,808 B`), crashes on `FP32 K=128` backward
(`OutOfResources: 295,176 > 166,912 B` SMEM), re-runs the entire sequential
$N_T$-chunk forward recurrence (`chunk_ttt_linear_bwd_h`) in backward, allocates
`3` `.clone()` tensors in eager `norm_residual`, truncates `FP32` backward
`tl.dot` to `TF32` (`1.34e-3` error), and corrupts `final_state_bias` (`hbt`) by
`1.08e+00` when $T \bmod 16 \ne 0$, compiles **`9/9` `0`-spill (`62`
specializations, `max_regs = 168 <= 168`, `max_smem = 73,728 B`) Round-27 RSI
Triton kernels** (`FP32 max_rel_l2 = 4.24e-7` = `3,153x` more accurate than
upstream `1.34e-3`, `3/3` mutations killed) via SM80 `MMA <-> DotOperand0`
transposed state congruence `b_ht = h^T [V, K]` and the Self-Adjoint LayerNorm
Identity $\langle \text{dv\_new}, v_{\text{new}} \rangle \equiv \langle \text{dy},
y \rangle$, wins all **`8/8` training shapes** in Fwd, Bwd & Total up to
**`4.14x` Fwd, `2.16x` Bwd, `2.47x` Total vs `chunk_ttt_linear` (`6.58x` Total
vs `fused_chunk_ttt_linear`)**, accelerates single-pass online test-time
adaptation decoding (`ttt_decode_adapt_rsi`) by **`11.00x–11.43x` vs
`chunk_ttt_linear` (`9.05x–9.33x` vs `fused_chunk_ttt_linear`)**, accelerates
`TTTLinearBlockRSI` by **`1.51x` ($D=1024, K=64$) & `1.73x` ($D=2048, K=128$)**,
scales $T_{\text{total}} = 16,384..32,768$ across 8x A100 NVSwitch via fused
Ulysses `TTT-CP = 8` (**`2.18x–2.70x` vs 8-GPU Ring & `1.85x–2.46x` vs 1-GPU
upstream**, `0.00e+00` `rel_l2`), and completes **`1.56x` ($D=1024$) & `1.53x`
($D=2048$)** more optimizer steps with **`-0.2465` ($D=1024$) & `-0.1855`
($D=2048$) final `val_bpb` reduction** in a fixed-`12s` autoresearch loop.

Round-28 RSI (`E154`, `v145`) `Titans: Learning to Memorize at Test Time`
(`arXiv:2501.00663`, Google Research — Ali Behrouz, Peilin Zhong, Vahab
Mirrokni; `fla-org/flash-linear-attention/tree/main/fla/ops/titans`) results are
under `raw/titans_rsi/`. It contains the smoke verification gate (`smoke.json`),
three independent 8-GPU process repetitions (`rep1.json`, `rep2.json`,
`rep3.json`), the aggregated 3-rep summary (`titans_rsi_summary.json`), markdown
report (`titans_rsi_report.md`), `README.md`, 5 CSV tables
(`titans_kernel_matrix.csv`, `titans_decode_matrix.csv`,
`titans_layer_ablation.csv`, `titans_cp8_scaling.csv`,
`titans_autoresearch.csv`), and `SHA256SUMS.txt` (`12` verified entries)
governed by `PROGRAM.md` Section 31 and the byte-identical frozen evaluator
`code/rsi_kernel_harness.py` (`SHA256 = 3cf0f078...`). It proves that upstream
`fla/ops/titans` (`naive.py`, `log_impl.py`) has **`0` Triton kernels** and
executes **`288` sequential Python loops per 16-token chunk** (`3,304.3 ms` at
$B=2, T=512, K=64$), is skipped in upstream CI (`@pytest.mark.skipif(True,
reason='FIXME')`) because zero-padding `theta, alpha, eta` produces **`40/296`
(`13.51%`) `NaN`s in `dalpha`**, **`8/296` (`2.70%`) `NaN`s in `deta`**, and
**`1.22e-1` error on `final_state`** when $T \bmod 16 \ne 0$, and truncates
`FP64` to `FP32` in `cal_G_log`, compiles **`12/12` `0`-spill (`137`
specializations, `max_regs = 128 <= 168`, `max_smem = 18,432 B`) Round-28 RSI
Triton kernels** (`FP32 max_rel_l2 = 7.77e-7`, `BF16 max_rel_l2 = 7.68e-3`,
`3/3` mutations killed) via the Bi-Triangular Decay Factoring Theorem
$G = (L^{(\beta)} L^{(\eta)}) \odot \theta^\top, f = L^{(\beta)} m$ and
transposed dual-state `(M^T, S^T)` slab streaming, wins all **`8/8` training
shapes** in Fwd, Bwd & Total up to **`121.63x` Fwd, `216.78x` Bwd, `181.98x`
Total vs `vectorized_pytorch_chunk_titans` (`3,080.7x` Total vs upstream
`chunk_titans_linear_ref`)**, accelerates single-pass online test-time
memorization decoding (`titans_decode_memorize_rsi`) by **`37.66x–43.58x` vs
`vectorized_pytorch_chunk_titans` (`5.17x–5.43x` vs `chunk_titans_linear_rsi`)**,
accelerates `TitansBlockRSI` by **`129.30x` ($D=1024, K=64$) & `53.97x`
($D=2048, K=128$)**, scales $T_{\text{total}} = 16,384..32,768$ across 8x A100
NVSwitch via fused Ulysses `Titans-CP = 8` (**`1.59x–1.60x` vs 8-GPU Ring &
`1.23x–1.32x` vs 1-GPU RSI**, `0.00e+00` `rel_l2`), and completes **`35.42x`
($D=1024$) & `25.33x` ($D=2048$)** more optimizer steps with **`-3.2291`
($D=1024$) & `-0.7225` ($D=2048$) final `val_bpb` reduction** in a fixed-`12s`
autoresearch loop.

Round-29 RSI (`E155`, `v146`) `Comba: Improving Bilinear RNNs with Closed-loop
Control` (`arXiv:2506.02475` — Jiaxi Hu, Yongqi Pan, Jusen Du, Disen Lan,
Xiaozhe Ren, Yibo Wang, Xuezhi Cao, Yunsheng Wu, Yu Zhang, Songlin Yang;
`fla-org/flash-linear-attention/tree/main/fla/ops/comba` &
`fla/layers/comba.py`) results are under `raw/comba_rsi/`. It contains the smoke
verification gate (`smoke.json`), three independent 8-GPU process repetitions
(`rep1.json`, `rep2.json`, `rep3.json`), the aggregated 3-rep summary
(`comba_rsi_summary.json`), markdown report (`comba_rsi_report.md`),
`README.md`, 5 CSV tables (`track1_kernel_matrix.csv`,
`track2_decode_matrix.csv`, `track3_layer_ablation.csv`,
`track4_cp8_scaling.csv`, `track5_autoresearch.csv`), and `SHA256SUMS.txt`
(`12` verified entries) governed by `PROGRAM.md` Section 32 and the
byte-identical frozen evaluator `code/rsi_kernel_harness.py` (`SHA256 =
3cf0f078...`). It proves that upstream `Comba.forward` (`fla/layers/comba.py`)
has an Inner-Decay `L2Norm` Erasure Bug where `p = k * self.decay.sigmoid()` is
computed before `l2norm_fwd(p)`, erasing $\sigma(\text{decay}_h) > 0$
(`||l2norm(0.9*k)-l2norm(0.1*k)||_rel = 8.32e-7`, `decay.grad max_abs =
9.56e-6` when `use_output_correction=False`), while upstream `fla/ops/comba`
spills across **`16 / 53` (`30.19%`) specializations** (`2,120` total spills,
`max_regs = 255`, `max_spills = 594` in `prepare_wy_repr_bwd_kernel`,
`max_smem = 163,856 B`) and suffers `3.82e-3` `FP32` `TF32` drift, compiles
**`25/25` `0`-spill (`212` specializations, `max_regs = 168 <= 168`, `max_smem =
24,576 B`) Round-29 RSI Triton kernels** (`FP32 max_rel_l2 = 3.00e-6`, `BF16
max_rel_l2 = 5.73e-3`, `3/3` mutations killed) via the Euler Homogeneous
Degree-1 WY & Score Adjoint Identities and combined exclusive/inclusive reverse
cumsum, wins all **`8/8` training shapes** in Fwd, Bwd & Total up to **`2.82x`
Fwd, `2.12x` Bwd, `2.30x` Total vs upstream `chunk_comba`**, accelerates
single-pass online decoding (`comba_decode_rsi`) by **`2.02x–2.20x` vs
`fused_recurrent_comba` (`16.74x–18.49x` vs `chunk_comba`)**, accelerates
`CombaBlockRSI` by **`2.07x` ($D=1024, K=64$) & `1.98x` ($D=2048, K=128$)**,
scales $B=8, T_{\text{total}} = 8,192..16,384, H=32$ across 8x A100 NVSwitch via
fused Ulysses `Comba-CP = 8` (**`2.59x–2.88x` vs 8-GPU Ring & `1.24x–1.51x` vs
1-GPU upstream**, `0.00e+00` `rel_l2`), and completes **`1.94x` ($D=1024$) &
`1.75x` ($D=2048$)** more optimizer steps with **`-1.5705` ($D=1024$) &
`-0.4966` ($D=2048$) final `val_bpb` reduction** in a fixed-`12s` autoresearch
loop.

Round-30 RSI (`E156`, `v147`) `Gated KalmaNet (GKA): A Fading Memory Layer
Through Test-Time Ridge Regression` (`arXiv:2511.21016` —
`fla-org/flash-linear-attention/tree/main/fla/ops/gka` & `fla/layers/gka.py`)
results are under `raw/gka_rsi/`. It contains the smoke verification gate
(`smoke.json`), three independent 8-GPU process repetitions (`rep1.json`,
`rep2.json`, `rep3.json`), the aggregated 3-rep summary
(`gka_rsi_summary.json`), markdown report (`gka_rsi_report.md`), `README.md`, 5
CSV tables (`track1_kernel_matrix.csv`, `track2_decode_matrix.csv`,
`track3_layer_ablation.csv`, `track4_cp8_scaling.csv`,
`track5_autoresearch.csv`), and `SHA256SUMS.txt` (`12` verified entries)
governed by `PROGRAM.md` Section 33 and the byte-identical frozen evaluator
`code/rsi_kernel_harness.py` (`SHA256 = 3cf0f078...`). It proves that upstream
`chunk_gka` crashes in 16-bit `.backward()` on `K < 64` or `V < 32`
(`SimpleGLABackwardGuard`), while upstream `fla/ops/gka` spills across **`29 /
58` (`50.0%`) specializations** (`76,526` total spills, `max_regs = 255`,
`max_spills = 10,028` in `gka_bwd_dk_intra`, `max_smem = 114,688 B`) and suffers
`1.99e-3` `FP32` `TF32` drift, compiles **`26/26` `0`-spill (`225`
specializations, `max_regs = 168 <= 168`, `max_smem = 69,632 B`) Round-30 RSI
Triton kernels** (`FP32 max_rel_l2 = 5.26e-7`, `BF16 max_rel_l2 = 1.34e-2`, `3/3`
mutations killed) via Sub-Chunk Strip-State Folding (`BS = 32`), post-loop
live-range decoupled `_gka_bwd_dqkg_gla_rsi_kernel` (`4.24x` faster), and
3-Quadrant Symmetric Gramian online decoding (`b_h00, b_h01, b_h11`), wins all
**`8/8` training shapes** in Fwd, Bwd & Total up to **`2.74x` Fwd, `4.74x` Bwd,
`3.60x` Total vs upstream `chunk_gka`**, accelerates single-pass online decoding
(`gka_decode_rsi`) by **`1.28x–4.90x` vs `fused_recurrent_gka` (`2.49x–9.97x` vs
`chunk_gka`)**, accelerates `GatedKalmaNetBlockRSI` by **`1.98x` ($D=1024,
K=64$) & `2.57x` ($D=2048, K=128$)**, scales $B=4, T_{\text{total}} =
8,192..16,384, H=32$ across 8x A100 NVSwitch via fused Ulysses `GKA-CP = 8`
(**`4.56x–16.18x` vs 8-GPU Ring & `3.36x–15.22x` vs 1-GPU upstream**, `0.00e+00`
`rel_l2`), and completes **`1.80x` ($D=1024$) & `2.31x` ($D=2048$)** more
optimizer steps with **`-0.4905` ($D=1024$) & `-0.3242` ($D=2048$) final
`val_bpb` reduction** in a fixed-`12s` autoresearch loop.

Round-31 RSI (`E157`, `v148`) `DeepSeek Sparse Attention (DSA), Online
KL-Distilled Lightning Indexer, & GLM-5.2 IndexShare` (`arXiv:2512.02556` /
`arXiv:2602.15763` — `fla-org/flash-linear-attention/tree/main/fla/ops/dsa`)
results are under `raw/dsa_rsi/`. It contains the smoke verification gate
(`smoke.json`), three independent 8-GPU process repetitions (`rep1.json`,
`rep2.json`, `rep3.json`), the aggregated 3-rep summary
(`dsa_rsi_summary.json`), markdown report (`dsa_rsi_report.md`), `README.md`, 5
CSV tables (`track1_kernel_matrix.csv`, `track2_decode_matrix.csv`,
`track3_layer_ablation.csv`, `track4_cp8_scaling.csv`,
`track5_autoresearch.csv`), and `SHA256SUMS.txt` (`12` verified entries)
governed by `PROGRAM.md` Section 34 and the byte-identical frozen evaluator
`code/rsi_kernel_harness.py` (`SHA256 = 3cf0f078...`). It proves that upstream
`fla/ops/dsa/naive.py` provides **`0` Triton kernels**, materializes up to
`2,209.4 MiB` of dense `[HI, T, T]` and `[HQ, T, T]` `FP32` matrices inside a
Python `for i in range(B)` loop, and discards indexer logits after `torch.topk`
so `q_idx.grad = k_idx.grad = w_idx.grad = None`, compiles **`19/19` `0`-spill
(`122` specializations, `max_regs = 168 <= 168`, `max_smem = 25,600 B`) Round-31
RSI Triton kernels** (`FP32 max_rel_l2 = 5.97e-7 < 1e-4`, `BF16 max_rel_l2 =
3.19e-3 < 1.5e-2` across all 8 outputs/gradients, `3/3` mutations killed) via a
Flattened-`(BM * HI)` Register-Resident MQA Lightning Indexer, a 64-Bit Bitmask
Table (`bitmask [B, T, T // 64]` `uint64`) + `tile_active` GQA-Packed
`Flash-DSA`, Online Teacher-Student KL Distillation, and Single-Pass Multi-Warp
Sparse Decoding (`dsa_decode_rsi`), wins all **`8/8` training shapes** in Fwd,
Bwd, Total & Peak HBM up to **`5.22x` Fwd, `6.20x` Bwd, `5.76x` Total & `10.98x`
HBM reduction vs upstream `naive_dsa`**, accelerates single-pass sparse decoding
(`dsa_decode_rsi`) by **`89.26x–761.31x` vs `naive_dsa` (`11.21x–150.70x` vs
`dsa_rsi` chunk)**, accelerates a 2-layer `DeepSeekSparseAttentionBlockRSI` +
`GLM-5.2 IndexShare` stack by **`3.04x` ($D=1024, K=64$) & `2.33x` ($D=2048,
K=128$)**, scales $B=2, T_{\text{total}} = 4,096..8,192, H_Q=32, H=8$ across 8x
A100 NVSwitch via fused Ulysses `DSA-CP = 8` (**`13.42x–25.93x` vs 8-GPU Ring &
`13.21x–25.74x` vs 1-GPU upstream**, `0.00e+00` `rel_l2`), and completes
**`3.18x` ($D=1024$) & `2.60x` ($D=2048$)** more optimizer steps with
**`-2.3737` ($D=1024$) & `-1.3229` ($D=2048$) final `val_bpb` reduction** in a
fixed-`12s` autoresearch loop.

Round-32 RSI (`E158`, `v149`) `MoBA: Mixture of Block Attention for Long-Context
LLMs` (`arXiv:2502.13189` — Moonshot AI / Kimi Team,
`fla-org/flash-linear-attention/tree/main/fla/ops/moba` & `fla/layers/moba.py`)
results are under `raw/moba_rsi/`. It contains the smoke verification gate
(`smoke.json`), three independent 8-GPU process repetitions (`rep1.json`,
`rep2.json`, `rep3.json`), the aggregated 3-rep summary
(`moba_rsi_summary.json`), markdown report (`moba_rsi_report.md`), `README.md`,
5 CSV tables (`track1_kernel_matrix.csv`, `track2_decode_matrix.csv`,
`track3_layer_ablation.csv`, `track4_cp8_scaling.csv`,
`track5_autoresearch.csv`), and `SHA256SUMS.txt` (`12` verified entries)
governed by `PROGRAM.md` Section 35 and the byte-identical frozen evaluator
`code/rsi_kernel_harness.py` (`SHA256 = 3cf0f078...`). It proves that upstream
`fla/ops/moba/parallel.py` and `fla/layers/moba.py` provide **`0` Triton
kernels**, duplicate `(topk - 1)x` Q and `2x` KV in HBM (`1,317.2 MiB`), stall
twice on `.nonzero()` / `q_zero_mask.sum()`, crash with `RuntimeError` on GQA
(`num_kv_heads < num_heads`) and `FP32` inputs, and suffer `1.87e-2` `BF16`
block-gate boundary drift, compiles **`14/14` `0`-spill (`139` specializations,
`max_regs = 168 <= 168`, `max_smem = 40,960 B`) Round-32 RSI Triton kernels**
(`FP32 max_rel_l2 = 1.14e-6 < 1e-4`, `BF16 max_rel_l2 = 2.51e-3 < 1.5e-2`, `3/3`
mutations killed) via the Unified Historical + Local Causal Block-Bitmask
Identity (`b_qmask | (1 << c_q)`), Single-Pass Tensor-Core `Flash-MoBA`
(`parallel_moba_rsi`), Single-Pass Centroid-Routed Block-Sparse Decoding
(`moba_decode_rsi`), and Fused `QK-RMSNorm+RoPE`, wins all **`8/8` training
shapes** in Fwd, Bwd, Total & Peak HBM up to **`10.54x` Fwd, `4.22x` Bwd, `5.71x`
Total & `2.08x` HBM reduction vs upstream `parallel_moba`**, accelerates
single-pass block-sparse decoding (`moba_decode_rsi`) by **`19.97x–32.95x` vs
`parallel_moba` (`3.95x–11.69x` vs `parallel_moba_rsi` chunk)**, accelerates a
2-layer `MoBABlockRSI` + `BlockShare` stack by **`3.75x` ($D=1024, K=64$) &
`3.12x` ($D=2048, K=128$)**, scales $B=2, T_{\text{total}} = 4,096..8,192,
H_Q=32, H_{KV}=8$ across 8x A100 NVSwitch via fused Ulysses `MoBA-CP = 8`
(**`5.32x–6.96x` vs 8-GPU Ring & `4.29x–6.23x` vs 1-GPU upstream**, `0.00e+00`
`rel_l2`), and completes **`3.24x` ($D=1024$) & `3.22x` ($D=2048$)** more
optimizer steps with **`-0.1654` ($D=1024$) & `-0.0954` ($D=2048$) final
`val_bpb` reduction** in a fixed-`12s` autoresearch loop.

Round-33 RSI (`E159`, `v150`) `Hierarchical Dual-Route Sparse-Linear Attention
(H2SA-GDN) & Governed Loop Optimization v3.8 Tournament` (`MoBA`
`arXiv:2502.13189` + `DeepSeek-V3.2 DSA` `arXiv:2512.02556` + `Gated-DeltaNet`
`arXiv:2412.06464`) results are under `raw/h2sa_gdn_rsi/`. It contains the
smoke verification gate (`smoke.json`), three independent 8-GPU process
repetitions (`rep1.json`, `rep2.json`, `rep3.json`), the aggregated 3-rep
summary (`h2sa_gdn_rsi_summary.json`), markdown report
(`h2sa_gdn_rsi_report.md`), `README.md`, 6 CSV tables
(`track1_training_matrix.csv`, `track2_decode_matrix.csv`,
`track3_block_ablation.csv`, `track4_cp8_scaling.csv`,
`track5_autoresearch.csv`, `track6_governed_loop_tournament.csv`), and
`SHA256SUMS.txt` (`13` verified entries) governed by `PROGRAM.md` Section 36,
the byte-identical frozen evaluator `code/rsi_kernel_harness.py` (`SHA256 =
3cf0f078...`), and `code/governed_loop_guard.py v3.8` (`manifest_root_sha256 =
654f7af7...`). Co-designed and peer-reviewed with M365 Copilot `Cowork` (`GPT
6.2 Astra, Max`) around the exact residual identity $\mathbf{o}_{\text{dense}}
- \mathbf{o}_{\text{sel}} = m(\mathbf{r} - \mathbf{s})$ (*"a routing miss need
not be a memory miss"*), it compiles **`16/16` `0`-spill (`83` specializations,
`max_regs = 168 <= 168`, `max_smem = 40,960 B`) Round-33 RSI Triton kernels**
(`FP32 max_rel_l2 = 9.24e-7 < 1e-4`, `BF16 max_rel_l2 = 4.72e-3 < 1.5e-2`, `3/3`
mutations killed) via 1D Dual-Route Subtile Bitmask Routing, `BV=32` V-Tiled /
`BK=64` K-Split Causal Block-Delta GDN State Recurrence & Readout, wins all
**`8/8` training shapes** up to **`8.03x` Fwd, `4.97x` Total & `4.55x` HBM
reduction**, accelerates 3-launch single-pass decoding (`h2sa_gdn_decode_rsi`,
`0.139–0.151 ms`) by **`24.06x–108.41x` vs Naive MoBA (`3.57x–92.57x` vs
Chunk-Prefill)**, accelerates a 2-layer `H2SAGDNBlockRSI` + `DualRouteShare`
stack by **`3.51x` ($D=1024, K=64$) & `3.24x` ($D=2048, K=128$)**, scales $B=2,
T_{\text{total}} = 4,096..8,192, H_Q=32, H_{KV}=8$ across 8x A100 NVSwitch via
Ulysses `H2SA-GDN-CP = 8` (**`2.67x–5.41x` vs 8-GPU Ring & `2.55x–3.84x` vs
1-GPU Naive**, `0.0` `rel_l2_o`, up to `2.30M tok/s`), completes **`3.18x`
($D=1024$) & `3.13x` ($D=2048$)** more optimizer steps in `12s` autoresearch,
and achieves **`PROMOTE` (`0.9578` Confirm IUT across `baseline`, `dsa_e157`,
`moba_e158`) and `HOLDOUT_PROMOTE` (`0.9535` Single-Touch Sealed Holdout,
`+0.3098` over `MoBA E158` and `+0.1760` over `DSA E157`, all 5 public
long-context strata `0.9390..0.9692`)**.

Finite-sample prevalence certification is under
`raw/pacing_coalescing_order_prevalence_certification/`. For every E89
profile, kernel, and accuracy point, it computes the exact L1 distance to the
nearest competitor regression boundary and converts that stability radius
into conservative 95%/99% sample requirements using a six-category
multinomial concentration bound. It reports finite-sample quantiles, fixed
budget coverage, sustained estimator-accuracy thresholds, and the
perfect-accuracy zero/infinite-radius partition. This is local-only analysis
because the source allocation remained canceled.

Regret-tolerant prevalence certification is under
`raw/pacing_coalescing_order_regret_certification/`. It allows
0/1/5/10/25/50 excess regressions per million path comparisons and computes
the exact pairwise-safe L1 distance at which any over-tolerance policy can
become empirically non-worse than the true prevalence winner. It reports
finite-sample quantiles, fixed-budget coverage, acceptable-policy-set size,
tie-safe zero-regret behavior, and worst-kernel estimator-accuracy
requirements. This is local-only analysis because the source allocation
remained canceled.

Finite-sample certificate calibration is under
`raw/pacing_coalescing_order_certificate_calibration/`. It runs 20,000
deterministic multinomial replicates at seven fractions of the E91 bound for
192 cases spanning four kernels, four accuracies, four regret tolerances, and
easy/median/hard E91 sample-requirement profiles. It applies the full
lexicographic policy selector and reports actual over-tolerance selection,
one-sided Clopper-Pearson bounds, L1-radius exceedance, and the first passing
tested sample fraction. This is local-only analysis because the source
allocation remained canceled.

Policy-specific Chernoff certification is under
`raw/pacing_coalescing_order_chernoff_certification/`. For each true
over-tolerance competitor, it computes the exact categorical Chernoff rate of
that policy becoming empirically non-worse than the true winner and unions
only those bad-policy events. It reports 95%/99% finite-sample requirements,
fixed-budget coverage, E91 reduction factors, estimator-accuracy thresholds,
and the residual gap to E92's empirical calibration. This is local-only
analysis because the source allocation remained canceled.

Global-minimizer Chernoff tightening is under
`raw/pacing_coalescing_order_global_chernoff/`. On E92's 192 representative
cases, it solves a multivariate exponential tilt for every over-tolerance
policy having no higher empirical regression than every tested policy. It
reports E93 sample reductions, impossible policy events, tolerance and
profile-difficulty effects, and the remaining gap to E92's first passing
tested points. This is local-only analysis because the source allocation
remained canceled.

Exact multinomial certification is under
`raw/pacing_coalescing_order_exact_multinomial/`. It enumerates every
multinomial count vector through the E94 bound when the cumulative state space
is at most one million and performs additional tractable E92/E93/E94 point
checks. It applies the full lexicographic policy selector and reports exact
sustained 5%-failure thresholds, lattice nonmonotonicity, and exact-to-analytic
sample ratios. This is local-only analysis because the source allocation
remained canceled.

E83 risk-surface uncertainty is under
`raw/pacing_coalescing_order_risk_uncertainty/`. It constructs simultaneous
95%/99% Bonferroni Clopper-Pearson intervals for 72 E83 slower-event cells,
propagates the rectangular confidence set exactly through every policy, and
sweeps unchanged-rate evidence multipliers from 1x to 256x. It reports
robust-regret envelopes, tolerance coverage, required path multipliers,
zero-success bounds, and worst profile/policy pairs. This is local-only
analysis because the source allocation remained canceled.

Policy-geometry alpha allocation is under
`raw/pacing_coalescing_order_risk_alpha_allocation/`. It compares equal,
mean-absolute, RMS, and maximum policy-contrast allocations over 144
directional cell bounds, with 10% equal floor. It also evaluates a
selection-safe best-of-four that divides family alpha across candidates. The
artifact reports interval widths, robust-regret envelopes, tolerance
coverage, path-equivalent evidence requirements, alpha concentration, and the
post-selection tax. This is local-only analysis because the source allocation
remained canceled.

Policy-aware path allocation is under
`raw/pacing_coalescing_order_path_allocation/`. It assigns future
comparisons uniformly or by E97 mean-absolute policy contrast and crosses
that choice with equal or mean-contrast alpha allocation. The artifact uses
exact integer total budgets and unchanged-rate projected successes to report
robust-regret envelopes, tolerance coverage, required total path factors,
cell-level sampling concentration, and rounding sensitivity. These are
planning projections, not new measured paths; actual certification requires
preregistered allocation and newly collected outcomes. This is local-only
analysis because the source allocation remained canceled.

Water-filled path allocation is under
`raw/pacing_coalescing_order_path_waterfill/`. It derives a lower-bounded
KKT allocation for the 95%-family mean-contrast Hoeffding-width surrogate,
preserves every existing E83 comparison, and uses exact integer total
budgets. The artifact reports surrogate optimality, active-cell and
concentration frontiers, exact robust-regret and coverage comparisons with
E98, and the high-budget maximum-regret tradeoff. Projected successes retain
E83 rates, so this is planning sensitivity rather than new evidence. This is
local-only analysis because the source allocation remained canceled.

The linear-floor water-fill frontier is under
`raw/pacing_coalescing_order_path_floor_frontier/`. It reserves
0/10/25/50/75/100% of each additional path budget with E98 linear
mean-contrast weights, then applies E99 water filling to the residual. The
artifact reports global and local p90/maximum-regret Pareto sets, endpoint
dominance, coverage requirements, residual KKT checks, and sampling
concentration. Each floor is a preregistered alternative; the audit does not
authorize post-hoc floor selection. This is local-only unchanged-rate
planning analysis because the source allocation remained canceled.

The all-pair minimax path allocation is under
`raw/pacing_coalescing_order_path_minimax/`. It builds 87,264
outcome-independent policy-pair/kernel/accuracy/cost-vertex constraints and
solves the convex epigraph problem with deterministic cutting planes. The
artifact reports separation certificates, integer allocation, active
constraints, exact robust-regret comparisons with E98/E99, and the cost of
over-protecting irrelevant geometry. Projected successes retain E83 rates,
so this is planning sensitivity rather than new evidence. This is local-only
analysis because the source allocation remained canceled.
