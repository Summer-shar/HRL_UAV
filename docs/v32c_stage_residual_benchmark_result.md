# v32c Noncausal Stage-Residual Benchmark Result

> Status: COMPLETE. This report applies the frozen protocol in
> `docs/v32c_stage_residual_benchmark_protocol.md`. Search seeds, validation
> seeds, CEM budget, power gates, and the policy family were not changed after
> observing any candidate.

## 1. Validity and Scope

- Frozen checkpoint: v32b seed-1 selected `best.pt`, including its matched
  micro SAC and multiplier.
- Search seeds: 2000--2004. Held-out validation seeds: 3000--3019.
- Two independent CEM searches each ran 24 candidates for 6 iterations.
- All candidate/P0 comparisons passed the common-exogenous-trace hash check.
- The result is a noncausal best-found benchmark in a restricted 18-D
  load-stage residual family. It is not a theoretical position-value bound.

## 2. Search Readout

| Search result | Delay (ms) | Power mean (W) | Power UCB95 (W) | 838-W search gate |
|---|---:|---:|---:|:---:|
| P0 / zero residual | 993.192 | 835.853 | 838.715 | no |
| Unconstrained best | 920.779 | 852.538 | 856.138 | no |
| Feasibility-first best | 993.192 | 835.853 | 838.715 | no |

The unconstrained arm found a 72.414-ms search-seed improvement, but it did so
with a large power violation. Across all 288 evaluated candidates, none passed
the conservative five-seed 838-W search gate. The minimum search UCB was the
zero-residual P0 candidate itself at 838.715 W, so the feasibility-first arm
retained P0.

This is a limitation of the frozen prescreen as well as a result: the 2-W
reserve combined with only five search seeds was conservative enough to reject
P0, even though P0 later passed the 20-seed 840-W validation gate. The gate was
not relaxed after this became visible.

## 3. Held-Out Validation

| Policy | Delay mean (ms) | Paired delay change vs P0 (ms) | Wins / 20 | Power mean (W) | Power UCB95 (W) | Feasible |
|---|---:|---:|---:|---:|---:|:---:|
| P0 | 938.872 | 0 | - | 833.584 | 835.397 | yes |
| Unconstrained candidate | 936.415 | -2.457 [-17.398, 12.483] | 11 | 849.958 | 852.049 | no |
| Feasibility-first candidate | 938.872 | 0 | 0 (20 ties) | 833.584 | 835.397 | yes |

The unconstrained candidate's standardized paired effect is `d_z=-0.077`.
Its large search-set improvement collapses to 2.457 ms on held-out seeds, the
confidence interval crosses zero, and power is infeasible. The
feasibility-first candidate is exactly P0 and therefore has zero gain.

Frozen verdict:

`NO_MATERIAL_FEASIBLE_GAIN_FOUND_IN_FROZEN_FAMILY`

## 4. Scientific Interpretation

The current v32b seed-1 architecture does not provide evidence for a robust
greater-than-5-ms trajectory increment in this structured policy family. The
result is stronger than a power-only failure: even after ignoring power, the
best search candidate does not retain a statistically supported delay gain on
held-out seeds. Search-set overfitting is the dominant observed failure.

This result does not prove that trajectory has no physical value. It is
limited to one training-seed tuple, a frozen micro policy and multiplier, the
specified stage-residual family, CEM budget, and conservative prescreen. A
different policy family or jointly adapted micro policy may perform better.

Together with the matched v32b seed-1 results (TD3 is 5.674 ms slower than P0;
P1-prime is 8.652 ms faster but its CI crosses zero and it is power-infeasible),
the evidence currently does not support a strong claim that the learned macro
trajectory captures a material load-switching benefit. Remaining formal seeds
must still be reported, but they cannot retroactively change this seed-1
benchmark verdict.

If trajectory optimization must remain a primary paper contribution, the next
scenario should first pass a new preregistered physical-value gate on held-out
seeds before any RL training is launched. A negative result here is not a
reason to tune TD3 again inside the same weakly identified scene.

## 5. Artifacts

Candidate-level search metrics, full CEM histories, held-out per-seed values,
paired deltas, machine-readable summary, and the figure are archived under
`result_archive/v32c_stage_residual_benchmark/`.
