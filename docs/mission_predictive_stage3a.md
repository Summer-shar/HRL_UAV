# Mission V1 Stage 3A

No learning or model modification. Read-only deterministic SAC101 window2000;
mission physics19976306281cf213f1e7f6973b999f2dcd39d593. Preserve existing
uncommitted Stage2 sources and all prior results.

## Frozen design

- Scenes33600,33601; branch times100,200,...,800s. One future realization each.
- Branch after `advance_macro_clock`, before next action. Full physical MDP,
  parent/chunk queues, mission/local mobility and global/environment RNG snapshots.
- Unchanged Predictive controller: current centroid plus10s causal velocity lead,
  original20m cap. First-macro residual is zero or +/-0.05 along one of6 direct
  action dimensions. Remaining5 macros use Predictive on each branch's own state.
- Native clipping/unit-disk/map/safety/continuous-motion execution. No coordinate
  writes, no baseline route playback. Each branch recomputes120 SAC decisions.
- 16x13x6=1248 macro intervals;208 branches. No other scenes/scales/horizons.
- Two0..860s prefix replays with the original90-macro load schedule construct
  actual states and uninterrupted zero references. They are not new900s evaluations.

## Evidence and tolerances

`C60` sums600 original physical costs; holding and signed objective energy are
recorded separately from full energy. `J60=C60/600` is a local horizon average,
not the earlier900s policy score. Queue means use120 frame ends.

PPO return uses native frame rewards and the exact `experiment.rollout`
20-frame discounted mean; `learning.py` uses `gamma_frame**20` between macros.
No tail value or entropy is added. Unit tests compare the diagnostic sequence
with a real no-learning rollout, plus an independent600-cost scalar calculation.

Numerical tolerances: physical cost1e-8, return1e-10, trajectory1e-8m.
An ineffective nonzero perturbation has no trajectory change over600 samples.
The statewise oracle excludes these; fixed-direction means retain all16 cases.
No significance threshold or effect-size success threshold is introduced.

Sources and configuration are hashed before execution. Snapshots preserve object
aliases via one pickle graph; audit callbacks are installed on new environments,
never copied from the original. Audit hash prefixes are retained in memory;
portable snapshots store physics/RNG and can be replayed, but do not serialize
the hashlib internal states. Those prefixes can be recreated by the saved replay.

All original parent/queue conservation assertions, actual rate/work/time checks
and energy closure remain enabled. Completion within a branch is divided by
initially unfinished parents plus arrivals; completed-only delay is auxiliary.

## Outputs and stop

`results/mission_predictive_stage3a/` is exclusive: refuse an existing directory.
Manifest/protocol/source_evidence,208 branch records,16 full snapshots, frame
queues/routes/links/uplink/commands/trajectories, CRN checks, physical costs,
macro rewards, per-state table, all12 fixed directions and descriptive associations.
Do not overwrite history, commit, push, fine-tune SAC or start any follow-up.

Retrospective selection under one future cannot establish stable state dependence,
learnability, final-test generalization, statistical significance or RL necessity.

## Executed preflight

`D:\Anaconda3\envs\pytorch-gpu\python.exe -m unittest tests.test_mission_opportunity -v`

Five tests passed (5.157s): bounded matrix, native20s rollout equivalence,
real frozen-SAC full-state restore/zero continuation/perturbed CRN,60s service and
objective conservation, and oracle/ranking arithmetic. The initial harness
attempt referenced a nonexistent PPO class and failed before simulation;
the guard was corrected to the actual PPO ActorNet/CriticNet constructors.
No production fix or learning operation was made.
Functional tests and bounded run results are documented in the new result folder
only after they actually complete.
