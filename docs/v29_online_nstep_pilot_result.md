# v29 online n-step pilot result

> Status: FINAL PRE-REGISTERED VERDICT
>
> Date: 2026-08-06
>
> Verdict: **CONTROL_NOT_REPRODUCED**. The v29 run cannot be used as the
> registered causal verdict on online five-step targets. Do not extend the run,
> select an intermediate checkpoint, or start formal training from this variant.

## 1. Registered comparison

The pilot compared two freshly trained TD3 macro controllers with the same
environment, seed, 100-episode budget, micro SAC checkpoint, replay sequence
sampler, power dual, and optimizer settings. The sole treatment difference was
the delay-Critic target horizon:

| Arm | Delay target | Replay sequence span | Energy target |
|---|---:|---:|---:|
| Control | 1 step | 5 steps | 1 step |
| Treatment | 5 steps | 5 steps | 1 step |

The protocol required the new one-step control to reproduce the historical v24
control within a pre-registered +/-3 ms equivalence band before interpreting
the treatment effect.

## 2. Fixed-probe mechanism result

Fixed v28 test states (episode seeds 64000--64019) were evaluated at checkpoints
60, 70, 80, 90, and 100. Six fork points and five checkpoints were averaged
inside each episode seed; the 20 episode seeds were the statistical units.

| State region | Q1 agreement, n=1 | Q1 agreement, n=5 | Paired n5-n1 difference, 95% CI |
|---|---:|---:|---:|
| Actor trajectory | 55.00% | 54.67% | -0.33 [-2.15, 1.48] percentage points |
| Centroid trajectory | 38.33% | 38.00% | -0.33 [-2.29, 1.62] percentage points |

The pre-registered mechanism criterion required the lower confidence bound of
the centroid-state improvement to exceed zero. It failed. Five-step targets did
not prevent the 60--100 episode probe-ranking degradation.

## 3. Final common-random-number evaluation

The final episode-100 checkpoints were replayed on seeds 1000--1019. External
user-motion and traffic traces matched by hash for every paired comparison.

| Metric | Historical v24 | New n=1 control | New n=5 treatment |
|---|---:|---:|---:|
| Weighted task delay | 773.440 ms | 812.922 ms | 835.585 ms |
| Mean power | 837.948 W | 831.116 W | 831.335 W |
| Mean centroid distance | 55.413 m | 58.338 m | 71.340 m |
| Total path length | 696.482 m | 708.670 m | 727.470 m |

The new one-step control was **39.483 ms** slower than historical v24, with a
95% paired CI of **[36.702, 42.264] ms**. This lies far outside the registered
equivalence band, so the frozen verdict is `CONTROL_NOT_REPRODUCED`.

Within the two new-code arms, n=5 was **22.663 ms** slower than n=1, with a 95%
paired CI of **[19.712, 25.614] ms**. Both arms satisfied the power constraint
(UCB95 833.033 W and 833.357 W). These numbers are descriptive only under the
frozen decision table because the historical control gate failed.

## 4. Post-hoc control-path diagnosis

The configuration audit found no unintended environment or optimizer change.
Relative to v24, the n=1 run added a five-transition sequence sampler, replay
logging, and more frequent evaluation. Training data and rewards were exactly
identical through episode 50, and the first TD3 update at episode 50 produced
exactly the same delay-Critic loss in both runs (0.012613901868462563).

The first divergence appeared during episode 51. TD3 updates occur after every
macro step, while the current episode does not yet contain all future members
of a five-step sequence. The new sampler therefore rejected and resampled
incomplete starts: 7 resamples in episode 51 and 13 in episode 52. Historical
v24 sampled individual transitions uniformly and had no such rejection.

This changes two things at once:

1. the replay-start distribution excludes temporarily incomplete recent
   transitions;
2. variable resampling consumes the global NumPy random stream, which also
   drives later exploration.

The new n=1 arm is therefore not the historical v24 one-step training path,
even though its target formula is mathematically one-step. The failure is a
control-pipeline mismatch, not evidence that the one-step target itself became
worse.

## 5. Scientific conclusion and next boundary

v28a established that longer frozen-policy target views improved offline
ranking. v29 did **not** establish that the same change improves online TD3:

- the fixed-probe mechanism benefit was absent;
- the n=5 endpoint was worse than the same-sampler n=1 endpoint;
- the registered historical-control equivalence prerequisite failed.

No formal run should use v29. Any follow-up must be a newly registered protocol.
A cleaner follow-up should separate replay sampling from exploration with a
dedicated RNG and replace end-to-end historical trajectory equivalence with an
exact one-step implementation-parity test on fixed replay batches. Historical
performance can remain a descriptive reference rather than a gate that a
sequence-based sampler cannot be expected to reproduce exactly from one
training seed.
