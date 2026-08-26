# v52 Ultra-Imbalance HRL Protocol

Status: FROZEN for the seed-1 screening run.

## 1. Purpose

v52 tests whether independently trained hierarchical policies can learn useful
UAV movement after the physical operating point has been shown to contain a
causal position benefit. It does not force, imitate, or reward the 20 m support
trajectory used by the pre-training gate.

## 2. Frozen Scene

Only the distribution of the fixed total offered load changes from v51:

```
[18.5, 2.0, 1.0] -> [1.0, 18.5, 2.0] -> [2.0, 1.0, 18.5] Mbps
```

The total is still 21.5 Mbps before `load_scale=1.15`. CPU remains 10 GHz per
UAV, U2U bandwidth remains 3 MHz, the macro period remains 10 s, the resource
frame remains 0.5 s, RPGM mobility is unchanged, and the objective remains
`0.9 * delay + 0.1 * energy`.

At each phase the offered loads after scaling are 21.275, 2.300, and 1.150
Mbps. The corresponding 10 s arrivals are 212.75, 23.00, and 11.50 Mbit,
while each UAV can serve 90 Mbit during the 9 s service interval. One zone is
locally infeasible, both support zones have spare capacity, and aggregate
capacity remains feasible with a 22.75 Mbit margin per macro period.

## 3. Pre-Training Physical Gates

The following read-only audits justify this scene selection:

- Holding a micro action for 1 s instead of refreshing it every 0.5 s changes
  weighted end-to-end delay by +0.596 ms, 95% CI [-0.049, 1.242]. The 0.5 s
  micro decision interval is therefore not the primary bottleneck.
- In the v51 load distribution, signed support offsets change geometry and
  U2U traffic but all nonzero offsets worsen the weighted objective. This
  explains the nearly overlapping learned trajectories in v51.
- In the frozen v52 load distribution, a causal 20 m support offset was
  selected on five search seeds and then evaluated once on independent seeds
  8200-8219. Relative to dynamic-centroid tracking it produced:
  - weighted delay: -6.090 ms, 95% CI [-7.911, -4.269], 20/20 seeds improved;
  - transition-window delay: -8.247 ms;
  - mean queue: -0.311 Mbit;
  - terminal queue: -0.433 Mbit;
  - support-heavy distance: -19.666 m;
  - nominal support-link rate: +0.0925 Mbps;
  - heavy-to-support flow: +4.475 Mbit;
  - U2U active time: -8.558 s;
  - average power: +7.275 W.

The 20 m rule is a diagnostic benchmark only. It is not an Advisor, behavior
cloning target, reward term, action constraint, initialization, or training
sample source.

## 4. Independent Training Matrix

All trainable methods start from their own random initialization. Every SAC
method receives its own 200-episode micro pretraining run, replay buffer,
optimizer state, and checkpoint. It then performs its own 800-episode run.
No micro checkpoint, replay transition, optimizer, or network is shared across
methods.

1. `joint_ppo_gated_sac`: proposed PPO macro plus autonomous-gated SAC micro.
2. `cppo_sca`: PPO macro plus pure SCA micro.
3. `joint_ppo_sac`: PPO macro plus original residual SAC micro.
4. `centroid_gated_sac`: dynamic-centroid macro plus gated SAC micro.
5. `centroid_sac`: dynamic-centroid macro plus original SAC micro.
6. `fixed_sac`: fixed deployment plus original SAC micro.
7. `joint_td3_sac`: TD3 macro plus original SAC micro.
8. `no_coop`: no U2U cooperation ablation.

The final checkpoint is always used. Training curves or intermediate
checkpoints cannot be used to select a favorable result.

## 5. Execution and Evaluation

1. Run the complete eight-method smoke matrix for seed 1.
2. Run the complete seed-1 matrix with 200 micro-pretraining episodes and 800
   subsequent episodes.
3. Evaluate final checkpoints on common, training-independent seeds 8300-8319.
4. Only if the seed-1 screen passes may the unchanged configuration expand to
   training seeds 1, 7, 42, 99, and 123.

The seed-1 screen passes only if all conditions hold:

- the complete method has lower paired weighted delay than both `cppo_sca`
  and `centroid_gated_sac`, with the 95% CI upper bound below zero;
- mean and terminal queues remain stable and completion does not degrade;
- learned PPO movement responds causally to the current heavy zone and yields
  the same direction of distance-rate-flow-queue-delay effects established by
  the physical gate;
- PPO and SAC Actor/Critic update counters are nonzero.

Power and the realized ms/W exchange rate remain descriptive outputs under the
soft weighted objective; there is no hard 840 W gate.

## 6. Stop Rule

If seed 1 fails, v52 is reported as a failed learning result. Do not extend
training, select an intermediate checkpoint, alter the load phases, add reward
shaping, or change the action/state definition to rescue it. Any later method
change requires a separately named and preregistered version.
