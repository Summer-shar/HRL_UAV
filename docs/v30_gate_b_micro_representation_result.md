# v30 Gate B micro-representation result

> Status: FINAL
>
> Date: 2026-08-10
>
> Frozen protocol: `docs/v30_gate_b_micro_representation_protocol.md`

## 1. Verdict

Gate B **failed**. The complete treatment package is not non-inferior to the
historical U2U representation under deterministic business-centroid tracking.
Joint TD3 training remains blocked by the frozen stop rule.

The primary treatment-minus-control delay estimate was **+8.302 ms**. The
two-way hierarchical bootstrap 95% interval was **[-35.337, +38.959] ms**, and
the one-sided 95% upper bound was **+35.624 ms**, above the preregistered
+3 ms non-inferiority margin.

## 2. Frozen-condition results

| Condition | Result |
|---|---:|
| One-sided 95% upper bound <= +3 ms | FAIL |
| No training-seed mean degradation > +10 ms | FAIL |
| Every treatment power UCB95 <= 840 W | PASS |
| External user/arrival traces identical within each pair | PASS |
| Integer task conservation exact in every episode | PASS |

The failed verdict is therefore not caused by a power violation, task-accounting
error, or broken common-random-number pairing.

## 3. Training-seed heterogeneity

| Training seed | Control delay (ms) | Treatment delay (ms) | Delta (ms) | Paired 95% CI (ms) | Treatment power UCB95 (W) | Coop-ratio delta | Routing TV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 765.934 | 729.471 | -36.463 | [-41.594, -31.333] | 835.419 | +0.0035 | 0.2384 |
| 42 | 718.030 | 758.002 | +39.973 | [+34.570, +45.376] | 835.613 | +0.0615 | 0.2369 |
| 123 | 725.239 | 746.636 | +21.397 | [+15.858, +26.936] | 834.083 | +0.0681 | 0.1174 |

The treatment materially changed routing: mean routing total variation was
0.1976. Seed 1 improved, while seeds 42 and 123 degraded substantially. In the
two degraded seeds, the treatment also increased the cooperation ratio by
about 0.06-0.07. With only three training seeds, this association is diagnostic
and must not be presented as proof that excess cooperation caused the delay
increase.

A post-result cluster mapping further limits that interpretation. Seeds 1 and
42 both occupied the high-routing-TV cluster (mean TV about 0.238) but had
opposite delay outcomes; seed 123 occupied the low-TV cluster (mean TV about
0.117) and still degraded. Routing-change magnitude therefore does not explain
the verdict. Cooperation-ratio change tracked delay direction across the three
training seeds, but three clustered training seeds are insufficient for a
causal or inferential correlation claim.

## 4. Post-result implementation audit

The frozen treatment was a package, not a one-variable intervention. It changed
four micro-state pathways:

1. the direct U2U matrix changed from the fixed-six-link nominal-rate feature
   (`eta / 120`) to normalized spectral efficiency (`eta / eta_ref`);
2. the same matrix enters the derived routing-goal matrix. Off-diagonal
   attractiveness was rescaled by about 18.49 times, while the diagonal is
   overwritten by local CPU availability. Row normalization therefore does not
   cancel the scale change, so the treatment also changed the hand-crafted
   local-versus-offload prior;
3. transmitted-queue pressure normalization changed from fixed-six-link
   nominal capacity to a capacity estimate based on the previous resource
   frame's active-link count;
4. the previous active-link fraction was appended as an additional SAC state
   scalar.

The macro U2U summary also changes representation, but Gate B used a
deterministic zero-residual macro policy. It therefore did not create a learned
macro-action difference in this gate; it remains relevant to any future joint
training experiment.

This audit narrows the valid conclusion: Gate B rejects the **complete package**.
It does not prove that spectral efficiency is harmful, that previous contention
is harmful, or that either component alone caused the seed instability. Gate A's
formula/environment equivalence result remains valid.

## 5. Decision and next admissible experiment

- Do not rescue Gate B by increasing training time, changing checkpoints, or
  tuning a seed-specific hyperparameter.
- Do not start Gate C or the formal joint TD3 matrix with this treatment.
- Before another training matrix, separate the direct channel feature from the
  derived routing prior and from contention-aware pressure normalization.
- A follow-up must be preregistered as a decomposition experiment. Its first
  requirement is that changing the channel feature cannot silently rescale the
  local-versus-offload goal prior.

The raw paired data, bootstrap result, routing diagnostic, and publication-scale
figure are archived in `result_archive/v30_gate_b_micro_representation/`.
