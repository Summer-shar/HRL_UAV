# v32c Stage-1 Continuation Audit Protocol

> Status: FROZEN before execution. This is a read-only diagnostic. It does not
> modify v32b training, checkpoint selection, rewards, or the remaining formal
> seed queue.

## 1. Question

The existing Actor-loss audit compares `Q^pi(s, a_actor)` with
`Q^pi(s, a_zero)`. The zero-residual action is applied for one macro period and
the learned Actor is the continuation policy. It therefore does not estimate
the value of a policy that keeps returning to the dynamic centroid.

This audit distinguishes:

1. a one-step action-ranking error in the delay Critic; and
2. an on-policy self-consistency trap in which one corrective step is genuinely
   ineffective under the drifting Actor continuation, while sustained
   correction would improve the realized outcome.

## 2. Frozen Objects

- Training seed: `1`.
- Checkpoints: evaluation candidates at episodes `100, 150, 300, 450`.
- Micro policy, multiplier, environment, and load-switch configuration: loaded
  from each checkpoint without alteration.
- Evaluation episode seeds: `41000..41019`.
- Episode length: 30 macro periods.
- Drift threshold: mean UAV-to-dynamic-centroid distance at least `50 m`.
- Eligible fork indices: macro periods `1..20`, leaving at least nine periods
  after the fork.

For each checkpoint and seed, an Actor rollout selects the first eligible fork
state. Seeds without an eligible state are reported and excluded; no threshold
or fork range is changed after observing the result.

## 3. Three Continuation Arms

All arms execute the frozen Actor before the selected fork and start from the
same state under common random numbers.

- `ACTOR`: frozen Actor at the fork and thereafter.
- `ZERO_ONCE`: zero residual at the fork, then frozen Actor thereafter.
- `RETURN`: zero residual at the fork and every remaining macro period. This
  continually targets the current dynamic centroid and remains subject to the
  physical 20 m movement limit.

The final exogenous-trace hash must match across all three arms for every seed.

## 4. Critic and Realized-Return Semantics

At the fork state, record the delay Critic values for the Actor and zero
actions using the checkpoint's frozen Actor aggregation rule.

The realized comparison for `ACTOR` versus `ZERO_ONCE` is the discounted sum,
from the fork to episode truncation, of the exact macro replay reward used to
train the delay Critic. This is the only comparison with matching continuation
policy and therefore the only valid direct audit of the one-step Q ranking.

`RETURN` has a different continuation policy. Its discounted replay return and
task-level weighted delay measure whether sustained recovery is valuable; they
must not be treated as a target that `Q^pi(s, a_zero)` was supposed to predict.

Report task delay, power, trajectory distance, and recovery time as supporting
metrics. Statistical units are episode seeds, not macro periods.

## 5. Frozen Interpretation

- Critic error: Q favors `ACTOR`, but the paired realized replay return favors
  `ZERO_ONCE` with a 95% CI excluding zero.
- Self-consistency trap: Q and realized return both favor `ACTOR` over
  `ZERO_ONCE`, while `RETURN` improves realized replay return and weighted task
  delay over `ACTOR` with paired 95% CIs excluding zero.
- Recovery not valuable in sampled states: `RETURN` does not improve either
  realized metric.
- Mixed or intervals crossing zero: inconclusive. No extra seeds or changed
  thresholds are added to rescue a label.

The audit establishes action-value semantics and temporal mechanism evidence;
it is not by itself a causal intervention on online training.

## 6. Outputs

```
result_archive/v32c_stage1_continuation_audit/
|-- per_seed_branches.csv
|-- per_fork_q_and_returns.csv
|-- checkpoint_summary.json
|-- stage1_continuation_audit.png
|-- stage1_continuation_audit.pdf
`-- verdict.md
```

