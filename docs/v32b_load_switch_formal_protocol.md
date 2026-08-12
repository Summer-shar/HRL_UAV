# v32b load-switch HRL formal protocol

> Status: FROZEN before v32b training. Date: 2026-08-11.
> v32a selected one-step TD targets. P1 and P1-prime remain power-infeasible
> diagnostic policies; no P1-double-prime tuning is permitted.

## 1. Scientific question

Can the dual-controlled hierarchical policy capture the causal delay value of
load-aware positioning while satisfying the strict system-average `840 W`
constraint? P0 centroid tracking is the feasible positioning baseline. P1 and
P1-prime establish position value but are excluded from feasible-policy
ranking.

## 2. Frozen environment

- Three phases cycle in this order:
  `[12.5,5.5,3.5]`, `[3.5,12.5,5.5]`, `[5.5,3.5,12.5]` Mbps.
- Each episode starts in phase 1. Each phase dwell is sampled independently
  and uniformly from 8 through 12 macro periods using a dedicated RNG derived
  from the episode seed. The RNG does not consume agent, arrival, mobility, or
  channel random streams.
- A phase change is applied before `get_macro_state()` and before the macro
  action. No future phase, countdown, or dwell realization enters either RL
  state.
- Episode length is 30 macro periods; macro period is 10 s, including one 1 s
  deployment interval and eighteen 0.5 s service frames.
- Dynamic users, `load_scale=1.15`, CPU 10 GHz, system budget 840 W, and the
  reference communication model remain unchanged.
- Macro load, realized flow, and capacity state blocks are enabled.

## 3. Frozen main method

- Main method: dual-controlled TD3-SAC HRL with both levels trainable.
- Pure Lagrangian constrained-delay objective; fixed violation penalty is 0.
- System multiplier is present in macro and micro states.
- Micro SAC pretraining: 300 episodes from a clean initialization under the
  same load-switch process and centroid-waypoint macro reference.
- Joint training: 1000 episodes from clean macro networks; micro SAC continues
  with the frozen two-timescale learning-rate protocol.
- TD3 delay target: one step. Actor aggregation: Q1. Neutral coverage: off.
- Historical nominal-rate U2U representation is retained. K-prev remains an
  untested open question and is not introduced in v32b.

## 4. Repetitions and common randomness

- Matched training seeds: `1, 7, 42, 99, 123`.
- Fixed evaluation seeds per training seed: `1000..1019`.
- All methods use the same schedule function and episode-seed mapping.
- Twenty evaluation seeds are repeated scenarios, not substitutes for the
  five independent training seeds.

## 5. Checkpoint and primary decision

- Evaluate every 50 training episodes.
- For each training seed, first retain checkpoints whose one-sided power
  UCB95 is at most 840 W, then select the retained checkpoint with minimum
  weighted task delay. If none is feasible, that training seed is infeasible.
- Re-evaluate the selected checkpoint with its own micro SAC and multiplier.
  P0 must be replayed using the same micro SAC, multiplier, and 20 CRN seeds;
  historical P0/P1-prime numbers may not be substituted.
- Primary delay claim requires all three conditions:
  1. every TD3 training seed has power UCB95 at most 840 W;
  2. TD3-P0 paired mean delay is beneficial for every training seed;
  3. a hierarchical resampling analysis over training seeds and evaluation
     seeds gives a 95% interval excluding zero.
- Descriptive capture is the fraction of the matched P1-prime-minus-P0 delay
  reference recovered by TD3. P1-prime remains marked infeasible and does not
  participate in feasible-policy ranking.

## 6. Monitoring and stopping discipline

- Episode-level logs must include average power, dual update power, lambda,
  movement, and the realized switch steps/dwells.
- Because lambda updates once per episode, intra-episode switch-window
  oscillation is not a meaningful lambda diagnostic. Phase-conditioned macro
  power is inspected by frozen-policy replay; lambda tracking is episode-level.
- The first seed may be stopped only for an engineering failure: non-finite
  values, a phase/state timing mismatch, missing dual state, failed checkpoint
  restore, or a broken lambda-to-CPU response chain. Poor delay or an
  unfavorable but finite learning curve is not an engineering stop.
- No hyperparameter, budget, phase, dwell, seed, or checkpoint rule may be
  changed after observing v32b results.

## 7. Formal comparison matrix

After the first monitored seed passes engineering checks, run the matched
five-seed matrix: TD3-SAC, PPO-SAC, PPO-SCA, and No-Coop. P0 is a paired replay
baseline. P1 and P1-prime are diagnostic infeasible references only.

