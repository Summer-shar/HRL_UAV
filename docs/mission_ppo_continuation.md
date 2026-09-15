# PPO-only continuation: cumulative window120 to600

## Scope

User authorized this independent continuation after the completed Stage4A120
qualification at9f98e5142b563e134a2f90bd2b6f555fd7ce4bf1. Original source files,
configuration, learning math, physics and all historical results are unchanged.
Only four independent continuation files are added. They are not committed;
their raw SHA256 values are locked in the new run manifest.

System model, optimization target, MDP observations/actions/reward, SAC and PPO
networks and update mathematics are unchanged. This changes the total training
budget and extends the existing scene sequence, not the physical task. No paper
or historical figure is edited; future learning curves must label cumulative
windows and treat this as a separate continuation experiment.

## Frozen protocol

- Restore each of PPO201/202/203 from its own boundary040/window120, matching
  its evaluated monitor120 state bitwise. Retain networks, optimizers, gradients,
  module modes, private RNG and cumulative counters. No checkpoint selection.
- Same source SAC101 window2000 and verified SHA256, deterministic and frozen
  in training and evaluation; no SAC replay construction/writes/sampling.
- Each seed adds160 continuous900s streams, scenes72040 through72199 in the
  same order.300/600s within a stream are statistics boundaries, not resets.
- Each adds480 windows /14400 macros /48 full300-sample updates. Final counts
  are600 windows /18000 macros /60 PPO updates; no partial force update.
- Existing config remains identical, including joint_windows=120 as the
  original progress normalization reference. The original collector passes
  cumulative_macros/3600, which is intentionally allowed above1. The unchanged
  learning code multiplies by120/200 and clamps at1. Std therefore reaches1/30
  at cumulative window200 and remains there. It is not restarted or stretched.
- New monitors:240/360/480/600, deterministic PPO and SAC, only33600/33601,
  900s each. Window120 results and full window0 gates are reused with hashes.
  Final endpoint is600. No33610-33612 or33700-33709 execution.
- Original Hover/Current/Progressive/Predictive comparators remain included.
  Statistics average the two development scenes per PPO seed first, then take
  mean/sample std over three initialization means; these are not fresh seeds.

## Implementation and storage

The new collector retains the frozen Stage4A body verbatim from meter creation
through reward, sampling, progress, accounting and return. Only scene/budget
guards are extended. An AST regression test verifies the shared body; the
original collector and120 runner are not modified or monkeypatched in production.

The original persist_case/statistics are reused. Each new training case is
written to a temporary directory inside its transaction and archived as
stream.zip. Every archived byte is checked against its original SHA256 before
the generated temporary directory is removed. No old files are removed.
The archive contains original0.5s queue records, full trajectories, actions,
CRN, energy/link/UAV records and summaries. Monitors retain the original layout.

Checkpoints and update batches remain uncompressed standalone files. Complete
transaction directories, source/runtime/evidence locks and an exclusive OS file
lock protect restart integrity. Low disk space (<512MiB) stops before the next
stream. Interruptions retain failed attempts; --resume restores only the latest
complete900s boundary. A monitor never changes PPO/RNG or SAC states.

## Commands

```powershell
python -m unittest tests.test_mission_ppo_continuation
python scripts/run_mission_ppo_continuation.py --output results/mission_ppo_continue120to600 --run-authorized-continuation
# Only for interruption, with the same output and unchanged sources/runtime:
python scripts/run_mission_ppo_continuation.py --output results/mission_ppo_continue120to600 --run-authorized-continuation --resume
```

This document describes the implemented protocol, not an assertion of completed
training or improved performance. Actual commands/results are reported separately.

## Acceptance executed before launch (2026-09-13)

- `python -m unittest tests.test_mission_ppo_continuation -v`:8 passed,
  25.546s. Includes all three schedules using clock doubles, interruption/recovery,
  real source restore, deterministic action equality on all540 archived states,
  original PPO synthetic updates across pending boundaries and the window200
  std floor, and two identical10s physical prefixes at scene72040. Those prefixes
  perform20 SAC inferences each and no PPO update; synthetic tests use temporary
  PPO networks and do not alter any source checkpoint.
- `python -m unittest tests.test_mission_ppo_qualification.RunnerTests
  tests.test_mission_ppo_qualification.SyntheticTests -v`:9 passed,17.161s.
- Read-only preflight passed: exact original HEAD, physics/source/evidence locks,
  all eight fixed comparator rows and source/runtime/config identity. Original
  tracked-file diff remains empty. No formal continuation was included in these
  test counts.
