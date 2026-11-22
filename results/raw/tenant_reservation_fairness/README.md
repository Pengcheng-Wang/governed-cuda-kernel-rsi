# Tenant and reservation-class fairness

This directory contains a local measured-service simulation that moves the
fairness identity from prompt length to tenant and reservation class.

The exact 2,000-request workload contains:

1. **Interactive**: 50% of requests, mostly 512-token prompts and 32-token
   outputs, with a 70% prefix-cache hit rate.
2. **Long-context**: 25%, mostly 6144-token prompts and short/medium outputs,
   with a 50% cache hit rate.
3. **Generation-heavy**: 25%, mostly 2048-token prompts and 512-token outputs,
   with a 10% cache hit rate.

Policies compare prompt-class fairness, equal tenant fairness with gain 4 and
gain 8, and weighted tenant reservations with acceptance multipliers
1.0/0.85/0.70. Fairness clocks use either predicted prefill service with a
40-second half-life or predicted prefill plus measured batch-1 decode service
with a 320-second half-life. Admission remains prefill-backlog based.

The tenant workload runs at 0.3/0.4 times decode-only capacity under no-cache
and tenant-correlated 75%-prefix reuse. An E55 rotating service-40 control is
also reproduced. The full matrix contains 24 aggregate records and
144 scenarios.

Every scenario verifies exact accepted/rejected accounting, prefill/decode
tokens and causal pairs, finite completion only for accepted requests, batch
size at most 64, KV within 32 GiB, exact tenant prompt/output/cache counts,
positive decode service for all nine prompt/output classes, decayed-state
accounting, the hard prefill-slice budget, and the unchanged admission
threshold. All 24 matching E55 controls reproduce with zero metric drift.

Key results:

- Prompt fairness was not tenant fairness. At no-cache 0.4x, prompt control
  reported 0.940 prompt Jain and 0.979 tenant Jain, yet
  interactive/long/generation acceptance was 65.2%/45.5%/59.4%.
- Gain-4 tenant identity improved throughput from 944 to 1,097 tokens/s and
  p99 TTFT from 550 to 479 ms, but tenant Jain was only 0.973 because the
  prefill-cost admission boundary still penalized long-context work.
- Gain 8 raised no-cache 0.4x tenant/service Jain to 0.990/0.987, with
  acceptance 70.2%/54.6%/64.6%, 1,044 tokens/s, and 515 ms p99 TTFT.
- Weighted 1.0/0.85/0.70 reservations reached 0.978 normalized Jain with
  acceptance 82.6%/50.5%/59.9%, 1,029 tokens/s, and 477 ms TTFT.
- The weighted policy raised interactive accepted-service share
  26.9%→31.2% and reduced generation-heavy share 56.0%→52.3%.
- Prefix reuse was tenant-correlated: exact hit counts were 700/250/50.
  At cached 0.4x, gain-8 equal fairness reached 0.996 tenant Jain,
  1,321 tokens/s, and 488 ms TTFT.
- Cached weighted reservations reached 0.989 normalized Jain,
  1,311 tokens/s, and 438 ms TTFT.
- End-to-end service clocks retained about 314–322 effective requests versus
  372–482 for prefill-only clocks. They included mean candidate decode
  service near 1.17 seconds and total service near 1.25–1.30 seconds.

Tenant identity, controller clock, admission cost, and reservation target are
separate controls. The decode clock uses standalone batch-1 service and does
not causally allocate shared batched decode cost.
