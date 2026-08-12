# v32c Noncausal Stage-Residual Benchmark Protocol

> Status: FROZEN before execution. This is a pure replay/search experiment and
> does not alter the running v32b formal seeds.

## 1. Scope and Naming

The experiment asks how much delay improvement can be found by a structured,
load-stage-aware trajectory controller when the v32b seed-1 micro policy,
multiplier, communication model, and power ledger are frozen.

The result is called the **noncausal best-found stage-residual benchmark**. It
is not a theoretical position-value upper bound:

- the policy family is restricted;
- CEM is a heuristic and need not find the family optimum; and
- the micro policy and multiplier are frozen rather than jointly optimized.

Future load-switch timing makes the controller optimistic relative to causal
policies, while incomplete search makes the best-found gain conservative
relative to the family optimum. These effects prevent a formal bound claim.

## 2. Frozen Architecture and Seeds

- Checkpoint: v32b seed-1 selected `best.pt`.
- Reference: matched zero-residual dynamic-centroid policy P0.
- Search episode seeds: `2000..2004`.
- Held-out validation seeds: `3000..3019`.
- Load switching: the checkpoint's cyclic-random 8--12-period schedule.
- Power budget: `840 W` system-average power.
- Search-stage conservative feasibility threshold: one-sided five-seed
  UCB95 at most `838 W`. This fixed 2 W reserve addresses search selection and
  small-sample uncertainty; it is an engineering guard, not a coverage
  guarantee.
- Final feasibility: held-out 20-seed one-sided UCB95 at most `840 W`.

## 3. Policy Family

The parameter vector contains one 2-D centroid residual for every load phase
and UAV: `3 phases x 3 UAVs x 2 coordinates = 18` parameters.

- Each UAV residual vector is projected to a disk of radius `80 m`.
- The waypoint is the current dynamic business centroid plus the active phase
  residual.
- Physical movement remains limited by the environment to `20 m` per macro
  period.
- The controller is allowed to know the next load-switch time. During the five
  macro periods before a switch, it linearly blends from the current phase
  residual to the next phase residual. This is the sole noncausal information.
- It receives no future user motion, arrivals, or channel realization.

This is a P1-shaped family with freely searched directions and all-UAV phase
roles. A negative result applies only to this family and search budget.

## 4. Frozen CEM Configuration

Run two independent CEM searches, one unconstrained and one feasibility-first.

- Population: `24`.
- Iterations: `6`.
- Elite fraction: `0.25`.
- Initial coordinate standard deviation: `40 m`.
- Minimum coordinate standard deviation: `5 m`.
- CEM random seeds: `3201` unconstrained and `3202` feasibility-first.
- Candidate zero is the current distribution mean in every iteration.

Unconstrained ranking uses mean weighted task delay. Feasibility-first ranking
first separates candidates by the frozen `838 W` search UCB gate, then ranks
feasible candidates by delay; if none is feasible, it ranks by UCB violation
and then delay. No weighted delay-power penalty is used.

The final candidate from each search is evaluated once on the held-out 20
seeds. Among these two already-frozen candidates, the best candidate whose
validation UCB95 is at most `840 W` is the reported feasible best found. This
is not a fallback search or retuning step. If neither candidate is feasible,
the result reports that fact without selecting an earlier CEM candidate.

## 5. Frozen Interpretation

All gains are paired against P0 on held-out seeds. "Feasible best-found" means
the better held-out delay among the two final candidates that pass `840 W`.

- Feasible best-found gain greater than `15 ms`, with the delay 95% CI
  excluding zero: strong evidence that the current macro learning problem has
  meaningful attainable position value.
- Feasible gain from `5` to `15 ms`: position is a secondary increment under
  this architecture.
- No feasible gain greater than `5 ms`: no material gain was found in this
  policy family and search budget; this does not prove that all trajectories
  lack value.
- Unconstrained gain greater than `15 ms` but feasible gain below `5 ms`:
  power feasibility, rather than geometry alone, is the dominant restriction.

This result may motivate a scene or algorithm change, but cannot alone prove
that the physical environment has no position value.

## 6. Outputs

```
result_archive/v32c_stage_residual_benchmark/
|-- protocol_snapshot.json
|-- search_baselines.csv
|-- unconstrained/history.json
|-- feasibility_first/history.json
|-- candidate_search_metrics.csv
|-- validation_raw.csv
|-- validation_paired.csv
|-- benchmark_summary.json
|-- stage_residual_benchmark.png
|-- stage_residual_benchmark.pdf
`-- verdict.md
```
