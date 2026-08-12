# v30 Gate B micro-representation protocol

> Status: FROZEN
>
> Date: 2026-08-07
>
> No joint TD3 training is permitted under this protocol.

## 1. Question and attribution boundary

Gate B asks whether the complete proposed U2U observation package is safe for
the micro SAC layer under deterministic centroid tracking.

The old nominal-rate feature and normalized spectral efficiency are related by
a constant one-to-one scaling. Consequently, the treatment changes both the
numerical conditioning and the causal contention information. Any Gate B
difference is attributable only to the complete package, not uniquely to
spectral efficiency or to the previous active-link count.

The old observation is not described as leaking the current action. It embeds
a fixed six-link nominal-bandwidth assumption and therefore is not the
executed dynamic rate. The treatment exposes action-independent channel
quality and separately supplies a causal summary of previous contention.

## 2. Arms

| Item | Control | Treatment |
|---|---|---|
| U2U matrix | nominal six-link rate / 60 Mbps | eta / fixed anchor eta |
| Previous contention scalar | absent | previous frame mean active-link count / 6 |
| Empty/reset fallback | historical behavior | 6 active directed links |
| Macro policy | deterministic zero residual | same |
| Physical interpretation | dynamic business-centroid tracking | same |
| SAC initialization | fresh | fresh |
| Replay reuse | prohibited | prohibited |

The previous-frame count is the arithmetic mean of the five physical
integration-step active-link counts within the completed 0.5 s resource frame.
It is updated only after that resource frame and uses no EMA.

## 3. Frozen training budget

- training seeds: 1, 42, 123;
- micro pretraining: 300 episodes per arm and seed;
- macro action: deterministic zero residual in `centroid_waypoint` mode;
- old and treatment arms use the same current code, replay RNG isolation,
  optimizer settings, workload, mobility, power budget, and dual protocol;
- checkpoint selection: existing common UCB-feasible-first rule, unchanged;
- treatment and control are trained sequentially on the single GPU; no
  concurrent GPU jobs are allowed.

Historical v24 is descriptive only. Its difference from the fresh control
contains both training randomness and replay-RNG plumbing changes and must not
be labeled pure run-to-run variance.

## 4. Frozen evaluation

- common evaluation seeds: 71000 through 71019;
- each of the six selected checkpoints is evaluated deterministically;
- primary metric: completed-task priority-weighted delay, using the existing
  evaluation definition;
- safety metrics: system average power and UCB95, queue/holding objective;
- behavioral metric: per-episode mean routing total-variation distance,
  averaged over priorities and source UAVs;
- raw per-training-seed and per-evaluation-seed values must be archived.

Let `delta = delay_treatment - delay_control`. The primary estimator is the
equal-weight mean over the crossed 3 training seeds by 20 evaluation seeds.
Its uncertainty is computed by independently resampling training seeds and
evaluation seeds (two-way hierarchical bootstrap, fixed analysis seed 30030,
20,000 replicates).

## 5. Gate B verdict

The non-inferiority margin is fixed at +3 ms, approximately 0.4 percent of the
current centroid-reference delay and consistent with the existing checkpoint
reproducibility tolerance.

Gate B passes when all conditions hold:

1. the one-sided 95 percent bootstrap upper bound for mean `delta` is at most
   +3 ms;
2. no training seed has a paired mean degradation greater than +10 ms;
3. all treatment checkpoints satisfy power UCB95 <= 840 W;
4. task conservation and external-trace pairing checks pass.

Strict improvement is not required. A tie passes because Gate B is a safety
qualification for the micro layer, while the proposed representation's main
test remains macro-Critic ranking in Gate C.

Routing total variation is diagnostic only:

- delay tie and near-zero routing difference: micro SAC did not materially use
  the new information; Gate C carries the remaining burden of proof;
- delay tie with nonzero routing difference: behavior changed without harming
  performance;
- delay improvement: positive secondary evidence, not a prerequisite.

## 6. Stop conditions

- failure of Gate B blocks joint training;
- no hyperparameter or checkpoint rescue is allowed after observing results;
- no BC, Advisor, distance penalty, coverage injection, n-step target, or
  Actor-objective change is permitted;
- a decomposition arm may be proposed later, but it cannot retroactively alter
  this two-arm verdict.

## 7. Archive

```
result_archive/v30_gate_b_micro_representation/
|-- protocol_snapshot.json
|-- per_episode_metrics.csv
|-- per_training_seed_summary.csv
|-- paired_statistics.json
|-- routing_distance.csv
|-- gate_b_summary.png
`-- verdict.md
```
