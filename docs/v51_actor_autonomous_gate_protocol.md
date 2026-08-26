# v51 Actor-Autonomous Residual Gate Protocol

Status: FROZEN for the seed-1 screening run.

## 1. Method

The macro controller is PPO. The micro controller is a standard
tanh-squashed Gaussian SAC policy with a 13-dimensional latent action:

```
a = [z_g, u_1, ..., u_12] in [-1, 1]^13
g = (z_g + 1) / 2
u_effective = g * u
```

`u_effective` is passed to the existing support-preserving SCA residual
mapping. The environment routing action remains 12-dimensional; only the SAC
latent action, replay action, and Critic action are 13-dimensional.

The Critic estimates `Q(s, [z_g, u])`. Replay stores the same sampled latent
action that produced the executed transition. The existing tanh-Gaussian
log-probability implementation therefore remains exact. The Critic never
defines or filters an action.

Interpretation:

- `z_g` near -1: the effective residual approaches zero, reproducing SCA.
- `z_g` near +1: the full learned residual is applied.
- intermediate values continuously interpolate residual strength.

The gate is an Actor output, not a constraint and not a reward term.

## 2. Frozen Invariants

- Physical environment, RPGM mobility, load switching, communication model,
  CPU model, reward, and 0.9/0.1 delay-energy weights remain v49-identical.
- No distance shaping, U2U-flow reward, Advisor, hard gate, or extra Critic.
- The legacy 12-dimensional SAC path is unchanged when
  `micro_autonomous_residual_gate=0`.
- All formal methods use independent networks, replay, optimizers, and
  micro pretraining.
- Final checkpoints are evaluated; no checkpoint selection is used to rescue
  a failed method.

## 3. Seed-1 Matrix

1. `cppo_sca`: PPO macro plus pure SCA micro baseline.
2. `joint_ppo_sac`: PPO macro plus the original 12-D residual SAC.
3. `centroid_gated_sac`: dynamic-centroid macro plus 13-D gated residual SAC.
4. `joint_ppo_gated_sac`: PPO macro plus 13-D gated residual SAC.

Each SAC method receives its own 200-episode micro pretraining run followed by
800 joint-training episodes. Twenty common evaluation seeds (8000-8019) are
used only to estimate conditional environment variability for each trained
checkpoint. A seed-1 result is a screening result, not cross-training-seed
evidence.

## 4. Component Diagnostics

In addition to independently trained methods, use the final complete-method
checkpoint for read-only paired replays:

- same gated micro policy, learned PPO macro versus dynamic centroid;
- same learned PPO macro, gated micro action versus forced `g=0` SCA.

These replays isolate component behavior but are diagnostics, not separately
trained formal methods.

## 5. Screening Criteria

For paired delay differences `d = A - B`, lower is better. Superiority requires
the 95% CI upper bound to be below zero.

- Complete method versus fresh PPO-SCA: CI upper bound < 0.
- Complete method versus fresh dynamic-centroid gated SAC: CI upper bound
  < -3 ms.
- Complete method versus fresh original residual SAC: CI upper bound < 0.
- Queue trajectories must not show sustained terminal growth or overflow.
- PPO and SAC Actor/Critic optimizer update counts must all be positive.
- Load-switch replays must report whether support movement points toward the
  newly heavy zone; distance from centroid alone is not an acceptance metric.

Gate mean, standard deviation, and quantiles are descriptive. Convergence near
zero or one is not itself a failure if performance criteria are satisfied.

Passing seed 1 permits, but does not establish, a five-training-seed run.
