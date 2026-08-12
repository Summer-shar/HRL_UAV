# v30 Gate B verdict

Status: **FAIL**

## Frozen conditions

- [ ] `noninferiority_upper_le_3ms`
- [ ] `no_training_seed_degrades_over_10ms`
- [x] `all_treatment_power_ucb95_le_840w`
- [x] `external_traces_match`
- [x] `task_conservation_exact`

## Primary estimate

Treatment - control delay: **+8.302 ms**; two-way bootstrap 95% CI
[-35.337, +38.959] ms; one-sided 95% upper bound +35.624 ms.

## Training-seed results

| Seed | Control delay (ms) | Treatment delay (ms) | Delta (ms) | Paired 95% CI (ms) | Treatment power UCB95 (W) | Coop-ratio delta | Routing TV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 765.934 | 729.471 | -36.463 | [-41.594, -31.333] | 835.419 | +0.0035 | 0.2384 |
| 42 | 718.030 | 758.002 | +39.973 | [+34.570, +45.376] | 835.613 | +0.0615 | 0.2369 |
| 123 | 725.239 | 746.636 | +21.397 | [+15.858, +26.936] | 834.083 | +0.0681 | 0.1174 |

## Attribution boundary

The verdict rejects the complete treatment package under the frozen Gate B
protocol. It does not isolate spectral-efficiency scaling, the
previous-contention scalar, contention-aware pressure normalization, or the
derived-goal rescaling as an individual cause.

Power feasibility, common-random-number trace identity, and exact task
conservation all passed. The failure is therefore a performance and
cross-training-seed robustness failure, not a power-accounting or pairing
failure.
