# P1-prime validation analysis

## Frozen decision

The single pre-registered 20-seed CRN validation returns
`POWER_INFEASIBLE`. The selected parameters must not be changed and rerun as
the same experiment.

- Residual offset target: `40 m`
- Residual slew cap: `10 m` per `10 s` macro period
- Macro-average approach rate: `1.0 m/s`
- Physical deployment speed for a full residual leg: `10 m/s` during the
  `1 s` deployment interval
- CRN trace hashes: all matched

## Measured result

| Metric | P0 | P1-prime | Paired effect |
|---|---:|---:|---:|
| Weighted task delay | 1091.183 ms | 1066.687 ms | -24.496 ms |
| Average system power | 831.957 W | 840.390 W | +8.433 W |
| Propulsion power | 594.270 W | 602.306 W | +8.036 W |
| CPU power | 235.460 W | 235.896 W | +0.436 W |
| Communication power | 2.227 W | 2.189 W | -0.039 W |

The paired delay effect is `-24.496 ms`, with 95% CI
`[-31.031, -17.962] ms` and paired effect size `d_z=-1.755`. The delay gate
therefore passes.

P1-prime's one-sided power UCB95 is `841.797 W`, which exceeds the frozen
`840 W` budget by `1.797 W`. The power gate therefore fails even though its
mean power exceeds the budget by only `0.390 W`.

## Why the analytic screen was optimistic

The pre-screen predicted a P1-prime UCB95 of `838.395 W`. It modeled the
residual maneuver as independent legs relative to hover. In the executed
environment, however, each UAV simultaneously follows a moving business
centroid. The deployment transient is charged from the squared magnitude of
the combined displacement, so centroid motion and residual slew contain a
cross term and cannot be added as independent power components.

This omission is visible in the replay decomposition:

- predicted maneuver increment: `3.978 W`;
- measured propulsion increment: `8.036 W`;
- measured transient-propulsion increment: `8.036 W`;
- measured CPU and communication net increment: `0.397 W`.

The screen therefore remains useful as a kinematic ranking tool but is not a
valid feasibility certificate for dynamic-centroid trajectories. A future
screen would need either paired trajectory replay or a bound on the combined
centroid-plus-residual displacement before declaring a candidate feasible.

## Research interpretation

P1-prime is not a failed position policy. It provides strong causal evidence
that load-aware support positioning has material value: all 20 paired delay
deltas are beneficial and the mean gain is about `24.5 ms`. It nevertheless
cannot be reported as a feasible baseline under the strict UCB95 power rule.

The correct role of P1-prime is therefore an infeasible, causal heuristic that
exposes the available delay benefit but lacks adaptive power control. The next
learned-policy experiment should retain the `840 W` operating point and test
whether the dual-controlled HRL policy can capture part of this benefit while
remaining feasible. This result does not justify the universal claim that no
feasible hand-written policy exists; it only rejects the two pre-registered
support rules that were actually tested.

