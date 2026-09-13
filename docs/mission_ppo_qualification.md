# Stage4A Qualification Runner

This is the runner extension to the audited Stage4A commit
`cfffe7dfca8cfe81e527dfd1636fd95bfcafc8f2`. Implementing this entry point
does not authorize or constitute a formal training run.

## Frozen Protocol

- Fresh PPO initialization seeds201/202/203 share deterministic frozen SAC101
  window2000, SHA256 `be946d3a576918460690a16f1a57f2c6dc33dde13db86d360db5690858a76932`.
- Unchanged mission physics: `19976306281cf213f1e7f6973b999f2dcd39d593`.
  Environment, MDP, wireless, CPU, movement, reward and original PPO math stay untouched.
- Each initialization visits72000 through72039 in the same order. Each stream
  is one900s/90-macro call.300s and600s boundaries only produce statistics.
-40 streams =120 windows =3600 macro transitions =12 complete300-sample updates.
  No forced or final partial update. SAC inference remains deterministic in training;
  only PPO explores. No SAC replay or SAC learning is constructed/restored/executed.
- Monitors at0/30/60/90/120 use33600 and33601, each900s with deterministic PPO/SAC.
  Only window120 is the endpoint. No33610-33612 or33700-33709 evaluation here.
- Window0 must match archived Hover on both complete scenes before any training.
  Trajectories, micro observations and zero macro actions are checked bitwise;
  route/queue/accounting records and reported metrics use rtol1e-12/atol1e-8.
  Current/Progressive/Predictive are locked historical comparators, not teachers.

## Invocation After Separate Authorization

`python scripts/run_mission_ppo_qualification.py --output results/NEW_DIRECTORY --run-authorized-qualification`

Resume the SAME directory with the same command plus `--resume`. There are no
CLI overrides for seeds, hyperparameters, horizons, scenes or baseline selection.
Imports, missing opt-in and `--help` never start evaluation/training.

## Recovery and Integrity

Formal checkpoints have their own stage schema, not a falsified SAC pretrain
continuation. Smoke checkpoints are explicitly rejected. At each complete stream
the checkpoint includes both PPO networks/optimizers, fixed exploration log_std,
the PPO RNG, unconsumed macro rollout, counters, gradients, module modes, next
scene index, current window, effective config and source/physics/evidence locks.
Pending90/180/270 (and other incomplete batch sizes) are preserved without updates.

Each boundary or monitor is written into a new sibling `.attempt-*` directory.
Only after all artifacts and their SHA256 manifest exist is it renamed to its
canonical directory. Interrupted attempts remain on disk and are never treated as
completed data. Resume validates the contiguous boundary chain and latest full
checkpoint; a mid-stream interruption repeats that whole stream. It does not
recover a partial environment or claim to preserve work since the last boundary.
A completed monitor is reused only when its checkpoint matches the resumed policy.

There is no concurrent-run support. One process must exclusively own one output
directory. Hard power-loss/filesystem durability is not guaranteed by directory
rename. Hash/corruption failures stop rather than silently falling back or changing
configuration. Recovery is limited to identical source/config/runtime/device context.

Untracked Stage3B diagnostics are neither required nor included in the formal
dependency lock. They are not committed by this task. Original result directories
are only read. Source SAC hashes, frozen SAC state and exogenous CRN checks remain
active. Across PPO initializations the training streams' exogenous records are
compared; UAV/queue/routing trajectories are not forced equal.

## Artifacts and Statistical Unit

Each stream/monitor reuses the existing mission meter and `save_case`: full trajectory,
business/group tracking error, routes by priority/source/role, queues, links, energy,
macro actions, per-window metrics, summaries and CRN records. The runner additionally
saves actual PPO batches (original GAE outputs), update statistics and checkpoints.
These files can be large; sufficient disk space is a prerequisite for formal runs.

Final aggregation first averages the two scenes within each PPO initialization.
It then computes mean and sample std across the three initialization means. Fixed
controllers get no training std. Per-scene completion counts/rates, unfinished tasks,
stage queues, power and full energy components remain available alongside delay/J.
Monitor comparisons and final summaries include all four fixed comparators, not only
the weaker Hover. No significance, convergence, generalization or guaranteed advantage
is inferred from development data.

## Acceptance Status

Actual acceptance on2026-09-13, Python3.9.25 in `pytorch-gpu`:

```powershell
& 'D:\Anaconda3\envs\pytorch-gpu\python.exe' -m unittest tests.test_mission_ppo_qualification.RunnerTests tests.test_mission_ppo_qualification.SyntheticTests tests.test_mission_ppo_stage4a.Stage4ATests.test_selective_restore_fresh_seeds_zero_means_and_empty_optimizers tests.test_mission_ppo_stage4a.Stage4ATests.test_protocol_separates_scenes_and_preserves_all_other_configuration tests.test_mission_ppo_stage4a.Stage4ATests.test_original_full_batch_update_and_logging_equivalence tests.test_mission_ppo_stage4a.Stage4ATests.test_sac_guard_rejects_learning_replay_and_detects_mutation -v
```

13 passed,0 failed,15.471s. Includes the real scheduler with a nonphysical clock
double (all three120-window schedules,12 batches each, monitor ordering, gate-failure
stop, interrupted-stream retry and completed-run no-op recovery). Synthetic samples
exercise original PPO updates and checkpoint round trips at90/180/270/360/630 samples.
Continued policy/optimizer/gradient/RNG/pending-rollout states match bitwise. Evaluation
does not mutate pending PPO state or global RNG. Log-on/off equivalence and frozen SAC
assertions pass. Baseline/config/source preflight was read-only; no environment calls.

```powershell
& 'D:\Anaconda3\envs\pytorch-gpu\python.exe' -m unittest tests.test_mission_ppo_qualification.RealPrefix -v
```

1 passed,0 failed,3.798s. This command was executed only once: scene72000,10s,
one stochastic PPO macro action,20 deterministic SAC decisions, zero PPO/SAC updates,
no replay writes/sampling, one pending macro sample. Its trajectory/queue/route/energy
output path was exercised in a temporary test directory, then removed by the test
fixture. SAC and locked sources were unchanged. The partial prefix was correctly
rejected as a formal stream-boundary checkpoint. Later metadata/CSV-preflight checks
were tested statically; the physical prefix was not rerun.

`python scripts/run_mission_ppo_qualification.py --help` and `git diff --check`
also completed successfully. No old multi-prefix smoke suite was rerun.

These14 distinct tests do not include a real900s monitor, a real full training stream
or formal120-window qualification. The window0 gate is implemented and tested on
synthetic records, NOT already passed on real900s PPO trajectories. No reserved
scene was run. Next: exact-commit audit, separate training authorization, then the
full900s window0 gate before learning. Disk space and exclusive output ownership
must be checked before that run. No performance improvement is claimed here.
