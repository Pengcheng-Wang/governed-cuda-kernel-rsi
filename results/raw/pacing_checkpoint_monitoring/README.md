# Geometric checkpoint monitoring

This directory contains E69, a scheduled-look alternative to E68's
every-request anytime monitoring.

Each metric-feature stream keeps the same Bonferroni share of joint alpha,
but monitoring occurs only at geometric request checkpoints. At look index
`k`, the stream spends:

```text
alpha_k = alpha_stream * 6 / (pi^2 * k^2)
```

Hoeffding bounds are valid simultaneously at the declared checkpoints. The
controller must not inspect or act on those bounds between checkpoints.

Four schedules are compared through one billion requests:

- 1.10x growth: 200 looks.
- 1.25x growth: 90 looks.
- 1.50x growth: 51 looks.
- 2.00x growth: 31 looks.

The full analysis includes 11,520 certification records and 8,640 fresh
post-change drift records. Each record reports the selected checkpoint,
look index, alpha spent, pure within-look quantization delay, and ratios
against both E67 one-shot Wilson plans and E68 every-request anytime plans.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_checkpoint_monitoring.py `
  --root . `
  --output results\raw\pacing_checkpoint_monitoring\pacing_checkpoint_monitoring_full.json
```

Key results:

- The 1.10x/1.25x/1.50x/2.00x schedules used median
  53.3%/51.3%/51.5%/47.1% of E68's certification labels.
- Median certification quantization delay was
  4,705/11,345/24,373/13,662 requests; p90 delay was
  36,681/70,589/137,413/244,155. Coarser schedules had better alpha
  concentration but worse tail action latency.
- Against E67 one-shot Wilson plans, median overhead remained
  22.18x/21.76x/21.85x/22.23x. Checkpointing removed about half of E68's
  conservative cost, but did not close the gap to one-shot monitoring.
- At 800 ms/10% error/0.1% risk, the full policy required
  431,323/459,668/466,608/524,288 labels instead of E68's 770,825. The 1.10x
  schedule was best for this tight-recall case.
- At the 0.25% budget, the same policy required
  58,280/61,694/61,446/65,536 labels instead of 91,775.
- The strict 650 ms/0.1% point required
  4,248,472/4,281,004/3,543,305/4,194,304 labels. Here 1.50x won because its
  tighter per-look alpha outweighed checkpoint delay.
- For drift detection, median request ratios versus E68 were
  60.6%/58.6%/56.2%/64.1%. The 1.50x schedule won the most drift cases;
  2.00x was hurt by checkpoint delay.
- For the full 800 ms policy, the 97.5%→95% recall breach required
  151,172/150,623/138,254/131,072 requests versus E68's 255,413. The 2.00x
  schedule was statistically cheapest, but its pure checkpoint delay was
  16,347 requests.
- Schedule winners varied by workload. 2.00x won 843 of 1,918 finite
  certification cases, while 1.50x won 780 of 1,890 drift cases. No single
  cadence dominated label cost and response latency.

The operational lesson is to declare monitoring times as part of the safety
protocol. A 1.10x schedule is a reasonable low-delay baseline; 1.25–1.50x
is a stronger label-efficiency region when delayed policy activation or
fallback is acceptable. A schedule is valid only if implementations do not
peek between declared checkpoints.
