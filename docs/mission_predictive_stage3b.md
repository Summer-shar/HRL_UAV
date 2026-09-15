# Mission Stage 3B: frozen directions, fresh conditional futures

Stage3A code is archived at47115d7c2f5cda972e52a514b70c2c7ca784a5ad.
The separate prerequisite archive is81b2778 (Stage2 provenance dependency).
Neither commit includes results. Mission physics and SAC101 source are unchanged.

## Design fixed before running

Reuse the16 full Stage3A snapshots: scenes33600/33601, t100,...,800s.
No new baseline replay,900s evaluation, other development or reserved scene.
For each state use exactly four labels: zero, original stage3a_best, e5_plus
(fixed_best), and the opposite of stage3a_best. No action reselection.
Even duplicate action vectors are run under both labels and checked for identical
physical terminal states and metrics. All labels remain in paired summaries.

Each label runs6 macros/60s. Only the first receives the frozen +/-0.05 normalized
direct-action residual. Subsequent5 macros restore the unchanged causal Predictive
rule on that branch's state. SAC101 deterministically reroutes on its own state.
Existing action projection, physical services, objective and statistics are reused.

Five future indices1..5 use env RandomState seeds:

`610000000+(scene-33600)*10000+branch_time_s*10+future_index`.

All80 seeds are fixed before evaluation, distinct and not evaluation scene IDs.
Restore the entire original snapshot and global RNG first, then reseed the existing
environment RandomState only. This discards its cached future Gaussian draw, not
the physical user velocities. Assert every other physical field/history unchanged.

**Conditional-future scope:** task Poisson arrivals and future group-local mobility
innovations are resampled. Mission reference routes and the pre-generated
load-switch schedule remain fixed because they belong to the full saved state.
This is not a robustness check over new load-switch dwell realizations. Regenerating
that schedule would require a separately specified conditional protocol; it is not
silently done here. Current phase, age, queues, unfinished tasks and velocity
memory are not reset. Retained irrelevant legacy group-noise draws follow the
original code and also consume the new environment RNG without physical edits.

Audit digests begin empty at the snapshot, so comparisons concern future suffixes.
Within each future, four candidates share identical births, mobility innovations,
user/group trajectories, load schedule and environment RNG progression. Across
futures require distinct arrival/user/innovation digests. Original Stage3A future
is selection data only and is excluded from all fresh-validation means.

## Scores and interpretation

Reuse C60, holding/energy, full energy/power, auxiliary completed delay, available-
parent completion rate, queues, actual radio work and time, and native truncated
PPO return. Numerical tolerances remain1e-8 cost,1e-10 return and1e-8m trajectory.
No new effect-size or significance threshold.

Report paired differences for each80 state/future pair. First average five futures
within each state, then average16 states. Report both sign frequency and magnitude,
state-specific minus fixed/opposite, original-future versus fresh-future shrinkage,
holding/energy and physical/return ranking alignment.80 pairs are not80 training
initializations; these are two reused development scenes, not unseen tests.
Rank comparisons among four labels are diagnostic only and never select actions.
Five draws cannot prove stable state predictability or RL learnability.

## Protection and stop

New independent results/mission_predictive_stage3b; refuse overwriting it.
Save manifest/protocol/frozen_candidates/source evidence, all320 branch files,
paired differences,16-state summaries, aggregate report and checks.
No PPO/TD3 instance, training, backward, replay, config/physics changes, new scale,
extra future, SAC fine-tuning, follow-up training or automatic post-run commit/push.
Stop after320 branches /1920 macros. Source and original selection evidence hashed
before/after. Snapshots are local trusted experiment artifacts, never external
untrusted pickle files.

## Executed preflight (2026-09-13)

- Stage3A regression: `python -m unittest tests.test_mission_opportunity -v`,5/5 passed,5.170s.
- New tests: `python -m unittest tests.test_mission_future_check -v`,4/4 passed,3.019s.
- Interpreter: `D:\Anaconda3\envs\pytorch-gpu\python.exe`.
- New checks cover all80 unique future seeds and frozen candidates, restoration of
  the actual saved full state except future RNG, a four-way10s frozen-SAC smoke
  including duplicate directions, another10s future with changed arrivals/mobility,
  and synthetic analysis verifying that poor selections are not replaced by winners.
- First new-test run had3 passes/1 failure: the test passed a120-window config to
  strict2000-window provenance checking. The test now reads Stage3A's effective
  configuration. No physics, source evidence or production check was relaxed.
- Real bounded320-branch completion is recorded separately in the result manifest;
  these tests alone are not a performance result.
