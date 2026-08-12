# v32c Stage-1 Continuation Audit Result

> Status: COMPLETE. This report applies the frozen rules in
> `docs/v32c_stage1_continuation_audit_protocol.md`. No threshold, checkpoint,
> seed, or interpretation branch was changed after observing the results.

## 1. Validity

- Checkpoints: episodes 100, 150, 300, and 450 from v32b seed 1.
- Evaluation seeds: 41000--41019; all 20 seeds were eligible at every checkpoint.
- ACTOR, ZERO_ONCE, and RETURN branches passed the common-exogenous-trace hash
  check for every fork.
- The one-step Critic audit uses the exact discounted macro replay reward used
  to train the delay Critic. RETURN is reported separately because it changes
  the continuation policy.

## 2. Frozen Verdicts

| Checkpoint | Q(actor)-Q(zero) | G(zero once)-G(actor) | G(return)-G(actor) | Delay(return)-Delay(actor) | Frozen verdict |
|---|---:|---:|---:|---:|---|
| ep100 | +0.0128 [0.0093, 0.0163] | -0.0087 [-0.0203, 0.0029] | +0.1827 [0.0393, 0.3262] | -105.8 ms [-176.2, -35.4] | INCONCLUSIVE_OR_MIXED |
| ep150 | +0.0155 [0.0071, 0.0240] | -0.0227 [-0.0311, -0.0142] | -0.1115 [-0.1485, -0.0745] | +1278.9 ms [916.0, 1641.8] | SUSTAINED_RECOVERY_NOT_VALUABLE |
| ep300 | +0.0898 [0.0838, 0.0958] | +0.0159 [0.0029, 0.0289] | -0.0055 [-0.0239, 0.0129] | +205.9 ms [5.5, 406.2] | ONE_STEP_CRITIC_RANKING_ERROR |
| ep450 | +0.0886 [0.0676, 0.1096] | +0.2000 [0.1798, 0.2201] | +0.9904 [0.9390, 1.0417] | +3104.4 ms [2698.4, 3510.4] | ONE_STEP_CRITIC_RANKING_ERROR |

Values in brackets are paired 95% confidence intervals across episode seeds.
Larger discounted replay return is better; lower completed-task delay is
better.

## 3. Interpretation

The experiment rejects a single-mechanism explanation of stage-1 degradation.

At ep100, sustained zero-residual control returns all UAVs close to their
dynamic centroids and improves both the training objective and completed-task
delay. A single zero action is not significantly better than the Actor action,
however. This is compatible with a one-step correction being too weak under
the original Actor continuation, but the preregistered self-consistency label
is not awarded because its one-step confidence interval crosses zero.

At ep150, the Critic and realized one-step return agree that the Actor action is
better. Forcing sustained return physically restores the positions but makes
delay substantially worse. At the sampled states, recovery cost and the
existing queue state dominate any geometric benefit.

At ep300 and ep450, the delay Critic ranks the Actor action above the zero
action while the exact realized replay return ranks the zero action above it.
This is direct evidence of a one-step action-ranking error relative to the
Critic's own training target. The sign-agreement rate falls from 70% at
ep100/150 to 25% at ep300 and 5% at ep450.

At ep450, sustained return exposes a second issue: it improves discounted
training replay return by 0.9904 but worsens completed-task delay by 3.10 s.
Under severe backlog, completed-task delay is affected by which tasks finish
inside the finite window, whereas the replay objective also accounts for queue
and constraint costs. The two quantities must therefore not be treated as
interchangeable at this checkpoint. This result is evidence of an
objective/evaluation divergence under heavy congestion, not evidence that
sustained return improves user-visible delay.

## 4. Mechanism Boundary

The supported conclusion is:

1. an early sustained-recovery opportunity exists at ep100 but is not exposed
   as a decisive one-step advantage;
2. sustained recovery becomes operationally costly after the queue has
   degraded, despite reducing power, because repeated deployment interrupts
   service while backlog is already large;
3. explicit one-step Critic ranking errors emerge by ep300 and persist at
   ep450; and
4. late-stage replay return and completed-task delay diverge under severe
   backlog/censoring.

The data do not support either blanket claim that all drift is caused by a
Critic error or that the Critic is always correct because one-step correction
is futile.

## 5. Artifacts

Raw paired branches, fork-level Q values and returns, the machine-readable
summary, and the figure are archived under
`result_archive/v32c_stage1_continuation_audit/`.
