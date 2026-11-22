# Decode admission fairness

This directory contains a local measured-service simulation using the same
mixed-length 32-layer decode model and seeds as the v37 continuous-batching
study.

The policies are:

1. Unguarded FCFS.
2. A 250 ms predicted decode-stage TTFT guard.
3. A dual guard that also requires a conservative predicted completion below
   10 seconds.

The mixed workload has three prompt lengths and three output lengths, producing
nine request classes. Four offered loads, two measured A100 attention
replicates, and three Poisson seeds produce 12 aggregate records and 72
scenarios. Every scenario reports aggregate request/token acceptance and
class-level TTFT, TPOT, completion, accepted-token share, Jain fairness, and
long/short prompt and output ratios.

The unguarded and TTFT-only paths reproduce all matching v37 scenario metrics
exactly. Every scenario also verifies exact request classification and
generated-token accounting, finite completion only for accepted requests,
batch size at most 64, KV within the 32 GiB budget, and nonempty class samples.

Key results at 1.1x ragged capacity:

- TTFT-only acceptance was 88.5% with Jain index 0.9996. Long-output
  acceptance was 99.3% of short-output acceptance.
- The dual guard accepted 90.3% of requests by favoring shorter generations.
  Accepted output-token fraction fell from 88.5% to 87.8%, mean accepted
  output length fell from 133.6 to 130.9 tokens, and long-output acceptance
  fell to 95.1% of short-output acceptance.
- Output throughput retained 99.8% of TTFT-only. P99 completion improved only
  from 9.49 to 9.37 seconds.
- At 1.25x load, the dual guard accepted 81.0% rather than 78.7% of requests,
  while long-output acceptance fell to 94.1% of short-output acceptance.

The completion estimate is a policy proxy, not a calibrated end-to-end
predictor. It holds current active contexts fixed and charges the arriving
request at its mean generation context. TTFT excludes prefill, prefix reuse,
ingress, tokenization, and sampling.
