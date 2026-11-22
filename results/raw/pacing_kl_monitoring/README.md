# Bernoulli-KL checkpoint monitoring

This directory contains E70, a tighter fixed-look confidence-bound audit for
the geometric checkpoint schedules from E69.

At every preregistered checkpoint and allocated one-sided alpha, E70 replaces
Hoeffding's distribution-free radius with a Bernoulli-KL Chernoff interval.
For empirical rate `p_hat`, trial count `n`, and candidate endpoint `q`, the
endpoint is found by inverting:

```text
n * kl(p_hat || q) <= log(1 / alpha)
```

where `kl` is binary relative entropy. Chernoff's Bernoulli tail bound
controls each fixed-look failure by the allocated alpha. E69's summable
look-index schedule and per-feature Bonferroni split then retain the same
joint scheduled-look validity.

The full analysis includes 11,520 certification records and 8,640 fresh
drift records across the same 1.10x/1.25x/1.50x/2.00x checkpoints.
Every finite KL interval is contained inside its matching Hoeffding interval,
and every KL crossing occurs no later than E69.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_kl_monitoring.py `
  --root . `
  --output results\raw\pacing_kl_monitoring\pacing_kl_monitoring_full.json
```

Key results:

- Median certification labels were only 18.0%/21.0%/19.8%/25.0% of E69
  Hoeffding for 1.10x/1.25x/1.50x/2.00x schedules.
- Median overhead versus E67 one-shot Wilson plans fell from about 22x in
  E69 to 4.35x/4.17x/4.17x/4.75x.
- Median KL certification cost was only
  9.29%/9.12%/8.88%/10.50% of E68's every-request Hoeffding plan.
- At 800 ms/10% error/0.1% risk, the full policy required
  32,895/31,587/27,309/32,768 labels versus E69's
  431,323/459,668/466,608/524,288 and E67's 8,731.
- At the 0.25% budget, the same policy required
  7,872/6,623/8,091/8,192 labels versus E69's
  58,280/61,694/61,446/65,536.
- The strict 650 ms/0.1% point required
  124,935/120,498/138,254/131,072 labels, down from E69's multi-million
  counts but still 3.78–4.33x the 31,915 one-shot Wilson plan.
- Near the 800 ms state-error cap, 1.50x required 92,169 labels versus
  649,150 in E69. Near the false-alarm cap it required 61,446 versus 705,598.
- KL also accelerated drift detection. At 1.10x, the full-policy state-error,
  recall, and false-alarm breaches required 4,441/18,566/2,756 requests
  versus E69's 12,680/151,172/12,680.
- The highlighted recall breach was cheapest at 2.00x with 16,384 requests,
  versus 131,072 in E69 and 255,413 in E68.
- 1.50x won 1,146 of 1,918 finite certification cases. 2.00x won 781 of
  1,890 drift cases. Tighter bounds changed the best cadence relative to E69.

The operational lesson is that statistical-bound choice dominates cadence
tuning. Geometric checkpoints solved about half of E68's overhead; the
Bernoulli-KL interval removed most of what remained while preserving the same
alpha protocol. Exact binomial inversion and optimized mixture confidence
sequences remain possible next steps.

Reference: Garivier, A. and Cappé, O. (2011), “The KL-UCB Algorithm for
Bounded Stochastic Bandits and Beyond,” COLT 2011,
https://arxiv.org/abs/1102.2490.
