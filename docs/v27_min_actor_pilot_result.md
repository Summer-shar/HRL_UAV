# v27 min-Actor single-factor pilot result

> Status: FINAL PRE-REGISTERED VERDICT
>
> Date: 2026-08-05
>
> Verdict: **FAIL**. Do not extend training, select an intermediate checkpoint,
> or start multi-seed formal runs from this variant.

## 1. Question and isolation

The v27 pilot tested one scientific change relative to the v24 dual-on control:

| Item | v24 control | v27 experiment |
|---|---:|---:|
| TD3 Actor delay-value aggregation | `Q1` | `min(Q1,Q2)` |

The environment, seed, 100-episode budget, TD3 hyperparameters, system dual,
840 W budget, micro SAC checkpoint, reward definition, and U2U representation
were unchanged. Joint neutral coverage, BC, Advisor, distance penalties, and
action regularization remained disabled.

This change is a diagnostic variant, not standard TD3. In standard TD3, the
clipped minimum is used in the Critic target while the deterministic Actor is
updated through Q1. Practical twin-Q SAC provides precedent for using a
conservative twin-Q estimate in a policy objective, but its stochastic policy
derivation is not identical to deterministic TD3.

## 2. Fixed endpoint behavior audit

The fixed endpoint was episode 100 (`final.pt`). Twenty common evaluation
seeds (1000--1019) were replayed for v27, v24, and the same-v27-checkpoint
centroid reference. External user and centroid traces matched by hash for all
pairs.

### 2.1 v27 versus v24 Q1-Actor control

| Metric | v27 | v24 | Paired difference (v27-v24), 95% CI | Criterion |
|---|---:|---:|---:|---:|
| Weighted end-to-end delay | 780.159 ms | 773.440 ms | +6.720 [4.352, 9.087] ms | FAIL |
| Mean centroid distance | 67.024 m | 55.413 m | +11.611 [11.478, 11.743] m | FAIL |
| Mean power | 839.884 W | 837.948 W | +1.935 [1.489, 2.381] W | descriptive |
| Total path length | 723.316 m | 696.482 m | +26.833 [22.766, 30.901] m | descriptive |

The one-sided 95% power upper confidence bound was **841.583 W**, above the
pre-registered 840 W budget. Therefore all three behavior gates failed:

1. delay did not improve;
2. centroid distance did not improve;
3. the power UCB95 was infeasible.

### 2.2 Same-checkpoint centroid reference

With the v27 micro policy and state architecture held fixed, replacing the
learned macro action by deterministic centroid tracking reduced weighted delay
from 780.159 ms to 761.406 ms. The learned-minus-centroid paired difference was
**+18.753 ms**, 95% CI **[15.522, 21.984]**. Thus the v27 failure cannot be
explained by centroid tracking becoming valueless under the new architecture.

## 3. Same-state mechanism audit

Twenty independent episodes (seeds 62000--62019) were evaluated at macro
indices 0/5/10/15/20/25. Branch points were averaged within each episode;
episodes, not the 120 correlated branch points, were the statistical units.
Both branches used the same Actor after their different first action, and all
exogenous traces matched.

### 3.1 Actor-trajectory states

| Quantity (Actor action - zero action) | Mean, 95% CI |
|---|---:|
| Q1 | +0.002982 [0.002815, 0.003150] |
| Q2 | -0.000576 [-0.000943, -0.000209] |
| min(Q1,Q2) | -0.000336 [-0.000687, 0.000015] |
| True continuation return | -0.001444 [-0.001981, -0.000908] |

At states already visited by the learned Actor, Q2 and the clipped minimum had
the correct mean direction, but conservative-Q ranking agreement was 65.0%,
exactly the same mean as v26. This local property did not produce better
end-to-end behavior.

### 3.2 Centroid-reference states

| Quantity (Actor action - zero action) | Mean, 95% CI |
|---|---:|
| Q1 | +0.005371 [0.005169, 0.005572] |
| Q2 | +0.003267 [0.002872, 0.003663] |
| min(Q1,Q2) | +0.003279 [0.002886, 0.003673] |
| True continuation return | -0.001744 [-0.002938, -0.000551] |

Both Critic heads were optimistic in the wrong direction at centroid states.
The minimum selected the less optimistic head but could not reverse its sign.
Conservative-Q ranking agreement was 33.33% in v27 versus 35.83% in v26. The
paired difference was **-2.50 percentage points**, 95% CI
**[-16.88, 11.88]**, so the pre-registered mechanism gate also failed.

## 4. Scientific conclusion

The audit supports four bounded conclusions:

1. The standard TD3 convention of updating the Actor through Q1 is **not an
   implementation bug**.
2. In v26, Q1 had a larger optimistic error than Q2 at some drift states, so
   Q1-only optimization was a plausible amplifier.
3. The v27 intervention shows that Q1-only optimization is **not a sufficient
   cause** of the observed harmful drift. When the rollout distribution
   changed under min-Actor training, both heads became optimistic at centroid
   states.
4. The unresolved failure lies upstream in Critic learning/support or state
   representation. Changing only the Actor aggregation does not repair it.

This result therefore rejects min-Actor as the next formal algorithm. It does
not establish that conservative twin-Q policy objectives are generally
ineffective; it establishes only that this single-factor variant failed under
the frozen project protocol.

## 5. Next boundary

No additional training is authorized by this result. The next experiment, if
approved, must be pre-registered as a separate single-factor test targeting
the state-action joint-support gap (for example, same-state step-level probe
actions). The U2U state representation correction remains a separate arm and
must not be combined with that test.

## 6. Artifacts

- `result_archive/v27_min_actor_pilot_audit/summary.json`
- `result_archive/v27_min_actor_pilot_audit/per_seed_raw.csv`
- `result_archive/v27_min_actor_pilot_audit/paired_vs_v24.csv`
- `result_archive/v27_min_actor_pilot_audit/paired_vs_centroid.csv`
- `result_archive/v27_min_actor_pilot_audit/mechanism_summary.json`
- `result_archive/v27_min_actor_pilot_audit/min_actor_pilot_audit.pdf`
- `result_archive/v27_min_actor_pilot_audit/min_actor_fixed_probe.pdf`
