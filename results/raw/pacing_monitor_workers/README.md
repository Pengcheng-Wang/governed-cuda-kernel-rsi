# Asynchronous monitor-worker frontier

This directory contains E77, an asynchronous counterpart to E76's
synchronous statistical-look cost model.

Every declared checkpoint creates one FIFO monitoring job with a fixed
request-equivalent service time. Alert action occurs when the first
statistically crossing look finishes processing. E77 compares one, two,
four, eight, and unlimited monitor workers across E75's valid checkpoint
grids.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_monitor_workers.py `
  --root . `
  --output results\raw\pacing_monitor_workers\pacing_monitor_workers_full.json
```

Key results:

- Zero service time reproduced all E75 robust winner counts for every worker
  policy. Unlimited workers added exactly one service time to each per-case
  request-only action oracle.
- Every asynchronous action was no later than E76's synchronous blocking
  upper bound. Median delay was nondecreasing as worker count fell.
- Unlimited workers retained the 1.10x minimax cadence for both domains
  throughout the exact 0–500 scan and sampled service costs through 10,000.
- Certification exact transitions:
  - One worker: 1.10x at 0–41, 1.25x at 42–100, 1.50x at 101–259,
    2.00x at 260–500.
  - Two workers: 1.10x at 0–82, 1.25x at 83–207, 1.50x at 208–500.
  - Four workers: 1.10x at 0–165, 1.25x at 166–418,
    1.50x at 419–500.
  - Eight workers: 1.10x at 0–352, 1.25x at 353–500.
- Drift exact transitions occurred much earlier:
  - One worker: 1.10x at 0–3, 1.25x at 4–7, 1.50x at 8–18,
    2.00x at 19–500.
  - Two workers: boundaries 8, 16, and 39.
  - Four workers: boundaries 15, 35, and 84.
  - Eight workers: boundaries 31, 72, and 334.
- At service cost 10, certification remained 1.10x for every worker count.
  Drift selected 1.50x with one worker, 1.25x with two, and 1.10x with
  four or more.
- At service cost 100, certification used 1.25x with one/two workers and
  1.10x with four or more. Drift used 2.00x with one through four workers,
  1.50x with eight, and 1.10x with unlimited workers.
- At service cost 500, certification selected 2.00x/1.50x/1.50x/1.25x/1.10x
  for one/two/four/eight/unlimited workers. Every finite drift worker pool
  selected 2.00x; unlimited workers retained 1.10x.
- Queueing was tail-heavy. At service 500, certification median/p90 action
  delay was 500/3,241 requests with one worker and 500/517 with eight.
  Drift was 2,999/3,928 with one and 513/562 with eight, versus exactly 500
  under unlimited parallelism.

The operational conclusion is that monitor concurrency and look service time
must be selected together with cadence. Asynchronous workers avoid E76's
fully synchronous tax, but a small FIFO pool can still accumulate large tail
delay on sparse-label drift checks. Four or eight workers substantially
extend the fine-cadence region; unlimited concurrency is the lower bound.
