# v31 Eta-Only Gate verdict

Status: **FAIL**

## Frozen conditions

- [ ] `noninferiority_upper_le_3ms`
- [ ] `no_training_seed_degrades_over_10ms`
- [x] `all_treatment_power_ucb95_le_840w`
- [x] `external_traces_match`
- [x] `task_conservation_exact`

## Primary estimate

Treatment - control delay: **-5.650 ms**; two-way bootstrap 95% CI [-50.785, +35.173] ms; one-sided 95% upper bound +24.909 ms.

## Training-seed results

| Seed | Control delay (ms) | Treatment delay (ms) | Delta (ms) | Paired 95% CI (ms) | Treatment power UCB95 (W) | Coop-ratio delta | Routing TV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 765.934 | 714.046 | -51.888 | [-56.710, -47.066] | 834.016 | -0.0124 | 0.2742 |
| 42 | 718.030 | 753.865 | +35.836 | [+33.127, +38.544] | 834.186 | +0.0134 | 0.2425 |
| 123 | 725.239 | 724.341 | -0.898 | [-5.060, +3.265] | 833.503 | -0.0071 | 0.2934 |

## Attribution boundary

The verdict applies only to the complete treatment arm defined by the frozen protocol. It does not validate or reject components that the protocol held fixed or excluded.

Power feasibility, common-random-number trace identity, and exact task conservation all passed. The failure is therefore a performance and cross-training-seed robustness failure, not a power-accounting or pairing failure.