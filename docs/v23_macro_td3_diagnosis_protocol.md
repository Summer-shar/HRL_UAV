# v23 Macro TD3 Diagnosis Protocol

Status: diagnostic protocol, not the final algorithm specification  
Date frozen: 2026-07-30  
Baseline config: `configs/tcom_dynamic_u2u_sparse_v22_long1000.json`  
Baseline branch: `codex/u2u-model`

## Purpose

This protocol isolates why the centralized macro TD3 learns a trajectory that
is worse than deterministic dynamic-centroid tracking. It freezes the current
physical scenario and lower-layer behavior while testing one macro-learning
mechanism at a time.

The protocol does not assume that target smoothing, action coverage, replay
capacity, exploration persistence, or disk projection is the final root cause.

## Established Evidence

- The full test suite passed before v23 changes: 80 tests.
- The lower SAC is healthy under a centroid-tracking macro policy.
- Common-random-number replay ranks the centroid policy above the final TD3
  policy in realized delay and power.
- The final delay Critic ranks the Actor action above the centroid action on
  the inspected states, despite the opposite realized policy-level ranking.
- `h_flying_penalty` is diagnostic-only and does not enter the macro replay
  reward.
- Hard disk projection and replay eviction may amplify drift, but their timing
  does not support either one as the initial cause.

## FROZEN: Current Diagnostic Stage

The following remain unchanged in the first v23 experiment:

- 3 UAVs, 39 users, fixed region-to-UAV association.
- Zone loads `[12.5, 5.5, 3.5]` Mbps with load scale 1.15.
- 10 GHz maximum CPU frequency per UAV.
- Active-link sharing of the 3 MHz total U2U bandwidth.
- Gauss-Markov user mobility without the historical circular zone boundary.
- 30 macro periods per episode; each period is 10 s.
- The first 1 s is deployment: arrivals continue and all service is paused.
- The remaining 9 s contains 18 resource frames of 0.5 s.
- `priced_maxweight` CPU allocation.
- 840 W system average-power budget.
- `constrained_delay` implementation with `energy_dual_lr=0`.
- Fixed macro power-violation penalty weight 20.
- `centroid_waypoint` macro action with 200 m normalized waypoint radius.
- 20 m maximum physical movement per macro period.
- Episode-persistent post-warmup exploration.
- Macro replay capacity 12,000.
- Hard unit-disk action projection.
- Existing lower-SAC training settings and all physical parameters.

## NEXT: First Controlled Experiment

### Instrumentation

Instrumentation must not consume random numbers, alter network modes, update
parameters, or write diagnostic samples into the training replay.

Record:

- raw Actor action norm: mean, p95, maximum, and fraction at or above 1;
- executed macro action norm;
- replay action-norm histogram and fractions below 0.1 and 0.25;
- delay-Critic Q1/Q2 disagreement on Actor actions and centroid actions;
- Q1 and minimum-twin preference of Actor actions over centroid actions.

Terminology: Critic "preference" is not accuracy. The Actor is optimized to
seek high predicted Q, so preference becomes evidence of error only when it
conflicts with paired realized returns.

### Single Mechanism Change

Create a new config derived from the v22 baseline and change only:

```text
td3_policy_noise: 0.20 -> 0.05
td3_noise_clip:    0.50 -> 0.10
```

This is treated as one target-smoothing mechanism change. All other training
settings remain frozen.

### Evaluation

- Run seed 1 for up to 1,000 episodes.
- Use deterministic evaluation and common random seeds.
- Compare the TD3 policy against deterministic dynamic-centroid tracking.
- Judge delay using paired differences, not an arbitrary absolute J threshold.
- Report mean power and its UCB against the actual 840 W budget.
- Inspect the whole training trajectory; checkpoints are diagnostic snapshots,
  not substitutes for convergence.

## Counterfactual Critic Diagnostic

Policy-level centroid rollouts establish realized return and action coverage,
but they cannot directly calibrate `Q^pi(s, 0)` because their continuation
policy differs from the TD3 Actor.

For a valid same-state counterfactual:

1. Follow the deterministic Actor to obtain the base trajectory.
2. At macro indices `0, 5, 10, 15, 20, 25`, clone the complete environment and
   RNG state.
3. Branch A reuses the base Actor trajectory.
4. Branch B executes centroid action `a=0` for the first macro decision.
5. Both branches follow the same deterministic Actor thereafter.
6. Verify identical user mobility, task arrivals, and other exogenous traces.
7. Reject a branch pair if the exogenous trace hashes differ.
8. Compare Critic predictions with realized discounted returns using the same
   replay reward definition and discount factor.

Sampling:

- intermediate monitoring: 5 independent evaluation episodes;
- final diagnostic: 20 independent evaluation episodes;
- six branch points are aggregated within each episode;
- the episode, not the branch point, is the independent statistical unit.

No counterfactual or centroid transition enters the training replay.

## Dynamic-Centroid Position Scan

Run this independently from training with a frozen lower SAC.

- Other UAVs use deterministic `a=0` centroid tracking.
- `a=0` sets the waypoint to the current centroid; it does not teleport the UAV
  and remains subject to the 20 m movement limit.
- Scan radii `[0, 20, 40, 60, 80]` m.
- Scan eight directions relative to the current inter-centroid geometry.
- Deduplicate the radius-zero point, giving 33 candidates per UAV.
- Use 3-5 common seeds for search.
- Retain the top three candidates per UAV and evaluate 27 joint combinations.
- Validate the final four policy/placement choices on 20 held-out seeds.
- Select feasible candidates first, then compare end-to-end delay.
- Record U2G, U2U, and flow metrics only as explanatory variables.

The existing static absolute-position search is not equivalent to this dynamic
relative-centroid scan.

## CONDITIONAL: Only After Supporting Evidence

- If target-noise reduction is insufficient and low-norm actions are sparse,
  test neutral-action coverage as a separate intervention.
- If neutral coverage is tested, use a seed-1 2x2 diagnostic:
  target noise high/low x neutral coverage off/on.
- If early correct samples disappear after replay saturation, test replay
  capacity 30,000 separately.
- If exploration persistence remains suspect, test five-macro-period block
  persistence rather than independent noise every macro step.
- Test soft projection only after measuring meaningful saturation or projection
  effects. A soft mapping changes the reachable waypoint radius and requires
  training from scratch.

Only the winning minimal configuration proceeds to 1,000 episodes and three
formal seeds.

## CONDITIONAL: Final Dynamic-Dual Version

This stage starts only after macro value estimation is stable.

- Keep the `priced_maxweight` CPU subproblem if computation allocation remains
  part of the paper's contribution.
- Add normalized energy price/lambda to both macro and micro observations.
- Remove the fixed linear violation penalty from replay rewards, unless a
  formally defined augmented-Lagrangian term replaces it.
- Keep delay and energy value estimates distinguishable so replay collected
  under different lambda values remains interpretable.
- During a pilot, calibrate lambda and energy critics before Actor updates.
- Use a minimum calibration period and a pre-registered maximum; failure to
  stabilize is a failed pilot, not a reason for unlimited training.
- Use the pilot only to choose one fixed calibration length for formal runs.
- Restart every formal seed from clean networks, optimizers, replay, RNG, and
  lambda state. Pilot checkpoints are never inherited.

If formal macro algorithms share a lower policy for a controlled comparison,
that lower checkpoint must be produced independently for each formal seed, not
reused from the pilot.

## OPEN

- Whether target smoothing is the main value-estimation failure mechanism.
- The actual low-norm action density in the macro replay.
- Whether neutral-action coverage is necessary after target-noise correction.
- Whether five-period exploration persistence improves coverage without
  destroying coherent waypoint relocation.
- The dynamic-centroid position-value upper bound in the current communication
  model.
- The fixed formal lambda-calibration length and final constrained objective.

## Change Control

Any behavioral change outside `NEXT` requires a new protocol version. Formal
experiments begin only after the diagnostic winner, constrained objective,
calibration length, comparison methods, seeds, and evaluation rules are frozen.
