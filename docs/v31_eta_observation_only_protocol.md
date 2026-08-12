# v31 eta-observation-only micro protocol

> Status: FROZEN
>
> Date: 2026-08-10
>
> Joint TD3 remains prohibited during this experiment.

## 1. Question

Does replacing only the direct U2U matrix observed by micro SAC with normalized,
action-independent spectral efficiency preserve micro-layer performance?

This experiment follows the v30 Gate B failure. It removes the three treatment
couplings found by the post-result implementation audit: derived-goal rescaling,
contention-aware pressure normalization, and the previous-contention scalar.

## 2. Arms

The control is the already trained v30 fresh-control arm. The new arm changes
exactly one micro-state block:

| Path | v30 control | v31 eta-only |
|---|---|---|
| Direct U2U matrix | nominal fixed-six-link rate / 60 Mbps | eta / fixed eta reference |
| Derived routing goal | historical nominal-rate input | historical nominal-rate input |
| Queue-pressure capacity | historical fixed-six-link nominal rate | historical fixed-six-link nominal rate |
| Previous active-link scalar | absent | absent |
| Macro policy | deterministic centroid tracking | same |

The macro U2U summary follows the configured representation, but the macro
policy is deterministic and does not learn or use that difference in this gate.

## 3. Frozen budget

- training seeds: 1, 42, 123;
- fresh micro SAC training: 300 episodes per seed;
- configuration: `configs/tcom_dynamic_u2u_sparse_v31_eta_only.json`;
- the v30 control checkpoints are reused as the frozen comparator;
- no treatment checkpoint or replay is reused;
- no hyperparameter, reward, physical-model, dual, or checkpoint-selection
  change is permitted.

## 4. Evaluation and verdict

Evaluation reuses v30 Gate B seeds 71000-71019, deterministic centroid
tracking, CRN trace hashing, exact task conservation, the two-way hierarchical
bootstrap (seed 30030, 20,000 replicates), and the same UCB-feasible-first
checkpoint rule.

Let `delta = delay_eta_only - delay_control`. The arm passes only if:

1. the one-sided 95% bootstrap upper bound is at most +3 ms;
2. no training seed has paired mean degradation above +10 ms;
3. every eta-only checkpoint has power UCB95 <= 840 W;
4. CRN trace identity and exact task conservation both pass.

Strict improvement is not required. Failure is archived without rescue,
additional training, or checkpoint selection after inspecting results.

## 5. Interpretation boundary

- PASS means the action-independent eta matrix is safe as a direct SAC feature;
  it does not validate changing the derived goal or adding contention state.
- FAIL means this direct rescaling is not robustly non-inferior under fresh SAC
  training; it does not imply that eta is physically incorrect.
- The observed v30 treatment-seed spread is not called a direct estimate of SAC
  variance because it also contains treatment-by-seed interaction.
