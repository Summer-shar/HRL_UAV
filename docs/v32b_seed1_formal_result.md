# v32b training-seed 1 formal result

> Status: RESULT. The v32b environment, algorithm, checkpoint rule, and
> reference-policy rules remain frozen. This report records the first complete
> training seed and does not authorize result-dependent tuning.

## 1. Engineering validity

- Micro SAC pretraining completed `300/300` episodes and joint TD3-SAC
  training completed `1000/1000` episodes without non-finite values or a
  checkpoint-restore failure.
- The full regression suite passes (`149/149`).
- The selected checkpoint is `joint_td3_sac/checkpoints/best.pt`, identified
  by the frozen feasible-first, minimum-delay rule as the episode-50
  checkpoint.
- The matched replay reproduces the original checkpoint evaluation:
  `965.711 ms` weighted task delay, `833.873 W` mean power, and `835.259 W`
  one-sided power UCB95.
- TD3, P0, and P1-prime use evaluation seeds `1000..1019`, the same random
  8--12-period load-switch schedule for each seed, the checkpoint's own micro
  SAC, and the checkpoint multiplier. All external-trace hashes match within
  every CRN triplet.

## 2. Selected-checkpoint result

| Policy | Weighted delay (ms) | Mean power (W) | Power UCB95 (W) | Feasible |
|---|---:|---:|---:|:---:|
| TD3-SAC | 965.711 | 833.873 | 835.259 | yes |
| P0 centroid tracking | 960.037 | 833.877 | 835.258 | yes |
| P1-prime causal support | 951.385 | 841.239 | 842.488 | no |

Paired TD3 minus P0 delay is `+5.674 ms`, with 95% CI
`[-3.873, +15.220] ms` and paired standardized effect `d_z=+0.278`.
TD3 is faster on `12/20` evaluation seeds, but its mean direction is harmful.
Training seed 1 therefore fails the frozen seed-level beneficial-direction
condition. Its power condition passes.

P1-prime minus P0 delay is `-8.652 ms`, with 95% CI
`[-21.665, +4.361] ms`; it is faster on `15/20` seeds but neither passes the
5-ms paired significance gate nor the 840-W power constraint. Its historical
verdict is not reused: this is a new matched-tuple verdict.

The descriptive TD3 capture fraction is `-0.656`. It is not a success metric
because TD3 does not improve on P0 and the matched P1-prime reference is both
uncertain and power-infeasible.

## 3. Behavior and transition windows

- Mean total three-UAV path length is `603.419 m` for TD3 and `602.930 m` for
  P0. Their representative trajectories are nearly identical; the selected
  TD3 checkpoint has not learned the non-trivial support displacement.
- In the first five macro periods after each realized switch, TD3 minus P0
  delay is `+6.792 ms` on average and is beneficial on `11/20` seeds.
  P1-prime minus P0 is `-7.358 ms` and beneficial on `13/20` seeds.
- Therefore the aggregate result is not hiding a reliable transition-window
  benefit from TD3.

## 4. Late-training mechanism

The final episode-1000 checkpoint is feasible (`831.554 W` power UCB95) but
has `10900.693 ms` weighted task delay and is correctly rejected by the frozen
checkpoint rule. The fixed drift-probe audit separates two sequential effects:

1. During approximately episodes 100--450, the delay Critic itself usually
   ranks the drift/hover-directed action above returning toward the centroid.
2. From approximately episode 500 onward, the delay component increasingly
   favors return, but the multiplier-weighted energy component reverses the
   total Actor objective on many states. At episode 900, delay favors return
   on `98.2%` of fixed drift probes while the energy term overturns that
   preference on `74.1%`.

This supports a path-dependent two-stage failure: Critic misranking first
creates the displacement; the energy term later helps lock in the displaced
state. It is not literal hovering: late Actor actions are closer in action
space to the hover residual, but still imply about `17 m` first-period motion
on the fixed probes.

## 5. Frozen interpretation

- Engineering gates pass.
- Training seed 1 is power-feasible but does not beat the matched P0 baseline.
- Because the v32b primary claim requires a beneficial TD3-minus-P0 direction
  for every training seed, that strong claim cannot pass unchanged after this
  seed.
- No budget, reward, target, seed, checkpoint, P1-prime parameter, or training
  hyperparameter is changed in response.

Primary artifacts:

- `result_archive/v32b_matched_reference/seed1/summary.json`
- `result_archive/v32b_matched_reference/seed1/paired_deltas.csv`
- `result_archive/v32b_matched_reference/seed1/matched_reference_summary.pdf`
- `result_archive/v32b_seed1_actor_energy_lock/checkpoint_summary.csv`
- `result_archive/v32b_seed1_actor_energy_lock/actor_energy_lock_audit.pdf`
