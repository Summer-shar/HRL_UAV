# v32a clean online n-step terminal protocol

> Status: FROZEN before execution. Date: 2026-08-10.
>
> This is the final bounded test of online delay-Critic target horizon. No
> checkpoint selection, extension, parameter rescue, or additional n-step arm
> is permitted after observing the result.

## 1. Question

Does changing only the macro delay-Critic target from one-step to five-step
recover the fixed-probe ranking error and improve deployed delay under the
v24 system-dual architecture?

The v24 mainline learned trajectory loses 12.954 ms relative to its
same-checkpoint dynamic-centroid reference. The 22.701 ms value belongs to the
failed v25 coverage arm and is not used as the v32a anchor.

## 2. Frozen factors

- Fixed-load v24 environment: `[12.5, 5.5, 3.5]` Mbps, dynamic users, 10 GHz,
  840 W system-average budget, pure Lagrangian system dual.
- Historical U2U numerical representation and routing-goal calibration.
- No neutral coverage, eta rescaling, `K_prev`, min-Actor, Advisor, BC,
  distance penalty, or reward change.
- Both arms reuse and freeze the same v24 seed-1 micro SAC checkpoint. The
  lower policy is deterministic during joint training. No micro update or
  replay restoration is allowed.
- Macro training seeds: `1, 42, 123`; 100 episodes each; final checkpoint is
  episode 100.

| Arm | Delay target | Common replay sequence span | Energy target |
|---|---:|---:|---:|
| control | 1 step | 5 | 1 step |
| treatment | 5 steps | 5 | 1 step |

## 3. Engineering gate

All items must pass before interpreting performance:

1. On a fixed replay batch, the sequence-sampler n=1 target is exactly equal
   to the historical one-step formula.
2. Macro replay uses a dedicated RNG and is unaffected by global NumPy draws.
3. For each training seed and episode, n=1/n=5 replay-start hashes and
   target-noise hashes match. A mismatch makes that seed `PIPELINE_INVALID`.
4. Both arms use identical network initialization, optimizer settings,
   exploration schedule, external episode seeds, and frozen micro checkpoint.

Historical v24 terminal delay is descriptive only. It is not an equivalence
gate because a common sequence sampler intentionally changes the replay-start
support by excluding starts without an available continuation.

## 4. Fixed-probe mechanism test

Use the frozen v28 states, Actor/zero actions, and G30 labels. Evaluate Critic
checkpoints at episodes 60, 70, 80, 90, and 100. Six fork points and five
checkpoints are aggregated within each evaluation episode seed before any
cross-seed statistic.

For every macro training seed, report the centroid-region Q1 ranking agreement
difference `n5 - n1` over the 20 fixed evaluation seeds. The primary mechanism
summary is a two-level bootstrap over macro training seed and evaluation seed.

Mechanism PASS requires all three seed-level point estimates to be positive
and the hierarchical 95% interval lower bound to exceed zero.

## 5. Final CRN evaluation

Replay every episode-100 checkpoint on eval seeds `1000..1019`. Within each
macro training seed, n=1/n=5 must share user, arrival, channel, and centroid
traces. Report delay, power, centroid distance, path length, and flow metrics.

- Delay PASS: hierarchical 95% interval upper bound of `D_n5 - D_n1` is below
  zero, and no training-seed point estimate is positive.
- Constraint PASS: n=5 power UCB95 is at most 840 W for every training seed.

## 6. Frozen decision

| Result | Verdict | v32b choice |
|---|---|---|
| Engineering gate fails | `PIPELINE_INVALID` | No algorithm conclusion |
| Mechanism, delay, and constraint pass | `FULL_PASS` | Use n=5 |
| Mechanism passes but delay does not | `MECHANISM_ONLY` | Use n=1 |
| Mechanism fails | `NSTEP_REJECTED` | Use n=1 |
| Any n=5 constraint failure | `CONSTRAINT_FAIL` | Use n=1 |

Any verdict other than `FULL_PASS` closes the n-step line permanently for this
paper. v32b must then use one-step targets.

## 7. Outputs

```
result_archive/v32a_clean_nstep/
|-- protocol_snapshot.json
|-- pipeline_gate.json
|-- replay_hash_audit.csv
|-- fixed_probe/
|-- final_crn/
|-- summary.json
`-- verdict.md
```
