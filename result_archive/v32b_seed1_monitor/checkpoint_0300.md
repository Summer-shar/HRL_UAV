# v32b seed-1 checkpoint-300 monitor

Status: **MID-RUN WARNING; NOT A FINAL VERDICT**

This record was frozen while the 1000-episode seed-1 run was still active.
The formal protocol permits an early stop only for an engineering failure;
finite but poor policy performance is not a stop condition.

## Fixed-evaluation checkpoints

| Episode | J | Weighted task delay (ms) | Power mean (W) | Power UCB95 (W) | Feasible | U2G distance (m) | Lambda after training episode |
|---:|---:|---:|---:|---:|:---:|---:|---:|
| 50 | 0.7067 | 965.71 | 833.87 | 835.26 | yes | 0.73 | 0.000 |
| 100 | 1.1153 | 1470.19 | 847.77 | 852.56 | no | 70.00 | 0.168 |
| 150 | 3.2223 | 7581.99 | 833.56 | 835.91 | yes | 116.14 | 0.713 |
| 200 | 3.4017 | 9986.15 | 837.99 | 841.68 | no | 140.17 | 1.042 |
| 250 | 3.7796 | 12564.83 | 860.95 | 863.32 | no | 149.92 | 1.491 |
| 300 | 4.4939 | 17210.18 | 864.23 | 866.83 | no | 154.43 | 3.149 |

## Current interpretation

1. The load-switch pipeline remains operational and all values are finite.
2. Lambda responds in the correct episode-level direction when the recent
   power window exceeds the budget, so the dual update is not silently off.
3. The fixed-evaluation policy exhibits a clear late-drift warning: U2G
   distance and delay rise together after episode 50.
4. The dual mechanism has not restored fixed-evaluation feasibility by episode
   300. This does not yet prove that no later feasible checkpoint exists.
5. Episode 50 remains a feasible checkpoint candidate under the frozen
   feasibility-first selection rule. Its matched P0 replay has not yet been
   run, so no positioning-benefit claim can be made.

## Required continuation

- Continue to episode 1000 without changing the algorithm or selecting a
  result-dependent stopping point.
- Inspect episodes 400--1000 for recovery or recurrence after the 12000-entry
  replay has replaced its early data (about 400 episodes at 30 macro steps).
- Make the final decision only from all saved 50-episode candidates, matched
  P0/P1-prime replay, and the frozen hierarchical inference rule.
