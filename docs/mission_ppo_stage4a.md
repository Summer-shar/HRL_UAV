# Stage 4A: fresh direct macro PPO + frozen deterministic SAC101

## Scope and authorization

This revision implements an independent entry and functional tests/smoke only.
It does not authorize or expose the formal three-seed120-window training loop.
No physics/config/learning implementation, historical results, paper, or Git
baseline is changed. No commit/push is part of this revision.

Implementation base: `47115d7c2f5cda972e52a514b70c2c7ca784a5ad`, branch
`codex/mission_rpgm_v1`. Stage3B uncommitted diagnostic files remain untouched.
Frozen mission physics: `19976306281cf213f1e7f6973b999f2dcd39d593`.

SAC source: `results/sac_only_wireless6_long_seed101/checkpoints/window2000.pt`.
SHA256: `be946d3a576918460690a16f1a57f2c6dc33dde13db86d360db5690858a76932`.
The existing source loader verifies checkpoint/manifest hashes, configuration,
SAC-only stage, window2000 endpoint and unchanged learning/source dependencies.
The mission provenance checker verifies the frozen physical files separately.

## Frozen prospective protocol

- Fresh PPO initializations:201,202,203; all paired with the same SAC101.
- Environment scenes:72000,72001,...,72039, in this order for every PPO seed.
- Each scene is a continuous900s stream:90 macros,20 service frames per macro.
- A300s window is statistics only. No reset at300/600s; reset at900s.
-120windows per seed:3600 macro transitions,12 full300-transition PPO updates.
- Monitor windows:0,30,60,90,120. Only final window120 may be used as endpoint.
- Validation:33600/33601,900s deterministic PPO and deterministic SAC.
- Comparison includes Hover/Current/Progressive/Predictive, not just Hover.
- No33700-33709, additional scenarios, best-checkpoint selection or joint tuning.
- All PPO hyperparameters remain those in the source configuration:
  initial std1/3, final std1/30, decay200windows, batch300macros,10epochs,
  actor LR3e-4, critic LR1e-3, clip0.2, target KL0.03.
- Source `training.seed` is replaced by the PPO initialization seed only in the
  new effective config; `joint_windows=120` supplies the unchanged PPO progress
  formula. The source config/checkpoint is never edited. Other fields match the
  existing mission effective config.

## Initialization and action semantics

`ReadOnlySAC` selectively restores the SAC actor, twin critic, target critic,
entropy and its optimizer states/observation metadata. It does not instantiate
or restore a replay. The old replay counter is recorded only as source evidence.

`MissionMacroPPO` constructs only fresh original PPO networks and optimizers.
It inherits the original macro sampler and calls the original PPO update/GAE;
it does not call `DirectHRL.__init__` or `restore_agent`. No source PPO network,
optimizer, rollout or RNG is restored. PPO initialization/exploration RNG is
separate from the explicitly enumerated environment scenes.

The unchanged zero output head gives a zero deterministic mean initially.
In mission mode this is Hover-equivalent. The initial stochastic training
trajectory is NOT Hover: its Gaussian std is1/3.

Raw Gaussian actions/logprobs remain the training data. The unchanged execution
chain is component clip[-1,1], unit-disk projection,20m scaling, map/safety
projection. Logs distinguish raw, clipped, disk-projected, requested and actual
displacement, saturation, speed, role and safety/map modifications.

Neither Predictive nor progressive/B helpers generate the PPO training action.
Predictive is read only as a historical comparator and a short fixed smoke
baseline; it is not a teacher, initialization or residual center.

## Macro-only collection

`collect_stream` reuses the original MDP, physical step and MissionPilotMeter.
It collects only `(state, raw_action, old_logprob, macro_reward, next_state,
trace_continue)` for PPO. SAC inference is always inside no_grad with
`deterministic=True`; no micro transition or pending replay object is created.
There is no extra tail macro action/decision after the stream ends.

The macro reward remains the original discounted20-frame mean; frame rewards
retain the original within-frame physical-step discounts. Macro discount is
gamma_frame**20. The last transition cuts the GAE trace across the reset but
retains the original successor-value bootstrap, exactly as the old update.
PPO batches persist across900s stream resets until300 samples are available.
No partial batch is force-updated during the prospective formal protocol.

SAC learn/soft-update/optimizer-step and replay construction/store/sample/load
raise if called inside the guard. SAC parameters, optimizer states, alpha,
gradients, module modes, action bounds and update counters are checked unchanged.
PPO learning is allowed only in designated short tests/smoke.

The optional journal observes the actual original GAE call, logging states,
raw actions, rewards, values, advantages, returns, logprobs, std, KL, epochs,
losses and gradient counts without introducing reference-controller targets.

## Functional verification entry

```powershell
& 'D:\Anaconda3\envs\pytorch-gpu\python.exe' -m unittest tests.test_mission_ppo_stage4a -v
& 'D:\Anaconda3\envs\pytorch-gpu\python.exe' scripts/run_mission_ppo_stage4a.py
```

The script rejects command-line overrides. It creates a new
`results/mission_ppo_stage4a_smoke` directory and refuses overwrites.

Real smoke: all three fresh initial policies are compared with Hover for20s;
only PPO201 collects60s/six macro samples on72000. One explicitly marked
`force=True` smoke update uses these six samples, without changing the300-sample
formal threshold or any PPO hyperparameter. A20s post-update paired check uses
33600 and the four fixed controls. No900s physical evaluation is run.

Separately, a nonphysical clock test double covers40x900s stream orchestration,
300/600s state carry-over,3600 collected/consumed macros, and12 complete batches.
These are control-flow assertions, NOT real900s training or performance data.
Another unit test uses a synthetic300-transition batch to compare the unchanged
PPO update with logging enabled/disabled. It is a numerical update test, not
environment training evidence.

## Interpretation and remaining work

Passing smoke means the isolated execution path functions; it does not show
PPO learns useful tracking, converges or beats a baseline. J and holding/service
must be reported together in any future qualification. Three PPO seeds do not
constitute three SAC retrainings or unseen-scene evaluation.

Formal training remains gated pending authorization. A complete120-window
launcher, monitor orchestration, final multi-initialization aggregation and
interruption recovery are not exposed here. The smoke checkpoint is labeled
`mission_direct_ppo_smoke_only` and must not be used to start qualification.
Recorded tests/results are in the independent smoke output; until commands
actually finish, this document is an implementation protocol, not a pass claim.
