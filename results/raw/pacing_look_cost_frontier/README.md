# Statistical-look cost frontier

This directory contains E76, an operational-cost extension of E75's robust
checkpoint cadence audit.

E75 counted labeled requests but treated each statistical look as free. E76
adds a synchronous request-equivalent cost to every look:

`composite cost = labeled requests + overhead_per_look * look_index`

For each case and cadence, the reported cost is the maximum composite cost
over E75's valid recursive/direct phase and integer-rounding grids.

The request-equivalent unit is deployment-specific. If a synchronous check
takes `t` seconds while traffic arrives at `r` requests/second, a first-order
mapping is `t * r` request-equivalents per look. Asynchronous or batched
checks can cost much less on the request path.

Reproduce from the repository root:

```powershell
python code\analyze_pacing_look_cost_frontier.py `
  --root . `
  --output results\raw\pacing_look_cost_frontier\pacing_look_cost_frontier_full.json
```

Key results:

- Zero overhead reproduced all 7,672 E75 certification schedule costs and
  7,560 drift schedule costs, including the exact robust winner counts.
- The exact integer scan from 0–500 request-equivalents/look found these
  global minimax regions:
  - Certification: 1.10x at 0–2, 1.25x at 3–44, 1.50x at 45–344, and
    2.00x at 345–500.
  - Drift: 1.10x only at 0, 1.25x at 1–3, 1.50x at 4–43, and
    2.00x at 44–500.
- Sampled 1,000/2,000/5,000/10,000 overheads remained in the 2.00x region
  for both domains.
- A one-request-equivalent look cost already moved drift minimax from 1.10x
  to 1.25x. Certification retained 1.10x through cost two and moved at cost
  three.
- At overhead 10, certification used 1.25x and drift used 1.50x. Median
  per-case oracle cost was 1.0185x/1.1117x the request-only oracle.
- At overhead 100, certification used 1.50x and drift used 2.00x. Median
  oracle cost rose to 1.1176x/1.6724x.
- At overhead 500, both used 2.00x and median oracle cost reached
  1.3924x/3.2470x request-only.
- Fixed global minimax and per-case winner plurality are different
  objectives. At some boundaries a cadence wins more cases, while another
  has lower worst-case regret and is therefore selected as global minimax.

The operational conclusion is conditional:

- Use 1.10x when statistical looks are effectively free or asynchronous.
- Move to 1.25x/1.50x as synchronous look cost grows.
- Use 2.00x only when checkpoint processing dominates request-count delay.

Production must measure look cost and map it into this frontier. A cadence
cannot be selected from request count alone.
