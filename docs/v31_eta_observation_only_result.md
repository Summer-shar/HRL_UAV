# v31 eta-observation-only micro result

> Status: FINAL
>
> Date: 2026-08-10
>
> Frozen protocol: `docs/v31_eta_observation_only_protocol.md`

## 1. Verdict

The eta-observation-only arm **failed** its preregistered non-inferiority gate.
Joint TD3 remains blocked.

The equal-weight treatment-minus-control estimate was **-5.650 ms**, but its
two-way hierarchical bootstrap 95% interval was **[-50.785, +35.173] ms** and
the one-sided 95% upper bound was **+24.909 ms**, above the +3 ms margin. One
training seed also exceeded the +10 ms degradation limit.

This is a robustness failure, not evidence that eta is physically wrong or
that its average delay is worse.

## 2. Frozen-condition results

| Condition | Result |
|---|---:|
| One-sided 95% upper bound <= +3 ms | FAIL |
| No training-seed mean degradation > +10 ms | FAIL |
| Every eta-only power UCB95 <= 840 W | PASS |
| External user/arrival traces identical within each pair | PASS |
| Integer task conservation exact in every episode | PASS |

## 3. Training-seed results

| Training seed | Control delay (ms) | Eta-only delay (ms) | Delta (ms) | Paired 95% CI (ms) | Eta-only power UCB95 (W) | Coop-ratio delta | Routing TV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 765.934 | 714.046 | -51.888 | [-56.710, -47.066] | 834.016 | -0.0124 | 0.2742 |
| 42 | 718.030 | 753.865 | +35.836 | [+33.127, +38.544] | 834.186 | +0.0134 | 0.2425 |
| 123 | 725.239 | 724.341 | -0.898 | [-5.060, +3.265] | 833.503 | -0.0071 | 0.2934 |

Across the three training seeds, the control arm's seed-level delay standard
deviation was 25.83 ms, the eta-only arm's was 20.67 ms, and the paired-delta
standard deviation was 44.05 ms. The latter is a treatment-by-training-seed
interaction and must not be labeled a direct estimate of generic SAC
run-to-run variance.

## 4. Interpretation

The direct historical U2U feature and the eta-only feature are related by a
constant one-to-one scale transformation:

```
historical feature = eta / 120
eta-only feature   = eta / eta_ref
```

Consequently, v31 added no channel information to micro SAC. It tested whether
the alternative scale was numerically safe after removing the v30 goal,
pressure, and previous-contention couplings. The answer is that the scale can
train successfully and can improve an individual seed, but it is not robustly
non-inferior under the frozen three-seed protocol.

The mean cooperation-ratio change was only -0.0020, so v31 did not reproduce
v30's systematic cooperation increase in the two degraded seeds. The v30
derived-goal rescaling remains an important confound, but v31 proves it was not
the sole source of training-seed sensitivity.

## 5. Decision

1. Do not rescue eta-only with more episodes, a different checkpoint, or
   seed-specific tuning.
2. Keep the historical numerical U2U matrix for the micro SAC mainline.
3. Correct its terminology in code and paper: it is an action-independent,
   fixed-scale channel-quality proxy proportional to spectral efficiency, not
   an executed dynamic-link rate.
4. Keep the derived routing goal on its historical calibration unless a
   separately preregistered goal ablation is scientifically required.
5. Do not add `K_prev` or contention-aware pressure to the mainline on the
   basis of v30/v31.

The experiment answers the direct-feature question. Further micro retraining
arms are not justified unless they target a new scientific question rather
than another normalization choice.
