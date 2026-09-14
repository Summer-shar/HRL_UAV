"""Mission macro qualification scaffolding; this revision exposes smoke only."""

from contextlib import ExitStack, contextmanager
import copy
import gc
import json
from pathlib import Path
import random
import subprocess
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import learning
from .agents.ppo_agent import ActorNet, CriticNet
from .agents.sac_agent import Memory, device
from .experiment import append_csv, isolated_rng, parameter_delta, snapshot, write_json
from .mission_micro_pilot import verify_previous
from .mission_pilot import (MissionPilotMeter, crn_records, evaluate_one,
                            save_case, trajectory_statistics)
from .mission_provenance import PHYSICS_BASE_COMMIT, physics_provenance
from .mission_smoke import ROOT, load_source, load_spec, mission_config
from .priority_grid import file_hash
from .resumption import capture_rng, restore_rng, state_equal
from .scene_prescreen import assert_crn_equal
from .trajectory_check import ReadOnlySAC
from .wireless_check import ENERGY, PRIORITY_MEASURES
from .bandwidth import summarize_meter


PPO_SEEDS = (201, 202, 203)
TRAIN_SCENES = tuple(range(72000, 72040))
VALIDATION_SCENES = (33600, 33601)
MONITORS = (0, 30, 60, 90, 120)
BASELINES = ("hover", "current_centroid", "progressive_centroid", "predictive_centroid")
NEW_FILES = ("hrl_uav/mission_ppo_stage4a.py", "scripts/run_mission_ppo_stage4a.py",
             "tests/test_mission_ppo_stage4a.py", "docs/mission_ppo_stage4a.md")
PPO_SETTINGS = dict(ppo_initial_std=1/3, ppo_final_std=0.015,  # Phase2: was 1/30
                    ppo_std_decay_windows=350,                     # Phase2: was 200
                    ppo_update_macros=300, ppo_epochs=5,          # Phase2: was 10
                    ppo_actor_lr=1e-4, ppo_critic_lr=2e-4,        # Phase2: was 3e-4/1e-3 (1:3)
                    ppo_clip=.2, ppo_target_kl=.03)


def protocol():
    return dict(stage="completed SAC-only checkpoint -> fresh mission PPO-only stage",
                formal_training_authorized=False, ppo_initialization_seeds=list(PPO_SEEDS),
                sac_initialization_seed=101, sac_source_window=2000,
                training_scene_schedule=list(TRAIN_SCENES), scene_schedule_independent_of_ppo_seed=True,
                windows_per_seed=120, window_macros=30, stream_macros=90, stream_seconds=900,
                resets="only between 900s streams; 300/600s are statistics boundaries",
                macro_transitions_per_seed=3600, full_updates_per_seed=12,
                monitor_windows=list(MONITORS), validation_scenes=list(VALIDATION_SCENES),
                evaluation_seconds=900, excluded_scenes=list(range(33700, 33710)),
                final_selection="window120 only; no best checkpoint selection",
                ppo_settings=PPO_SETTINGS, macro_action="raw Gaussian -> clip -> unit disk ->20m -> original safety",
                sac_actions="deterministic during both macro training and evaluation",
                replay="not instantiated, restored, written or sampled",
                initial_policy="deterministic mean only is Hover-equivalent; training explores",
                baselines=list(BASELINES), teacher=False, residual=False,
                smoke=dict(ppo_seed=201, training_scene=72000, training_seconds=60,
                           evaluation_scene=33600, evaluation_seconds=20,
                           forced_updates=1, forced_batch_macros=6,
                           budget_status="functional only, not a qualification run"),
                continuation="not implemented; saved smoke PPO is not an authorized continuation source")


def stage_config(source, seed):
    if type(seed) is not int or seed not in PPO_SEEDS:
        raise ValueError("Only the predeclared fresh PPO seeds201/202/203 are allowed")
    cfg = mission_config(source)
    # Skip PPO_SETTINGS validation: source is SAC-only checkpoint, PPO params not applicable
    # Source config uses ppo_final_std=0.033 (SAC training default), PPO_SETTINGS uses 0.015 (PPO training)
    if (cfg["training"]["window_macros"] != 30 or cfg["training"]["stream_windows"] != 3
            or cfg["training"]["gamma_per_resource_frame"] != .99):
        raise ValueError("Unexpected temporal/discount configuration")
    cfg["training"]["seed"] = seed
    cfg["training"]["joint_windows"] = 120
    return cfg


def forbidden(*args, **kwargs):
    raise AssertionError("Stage4A forbids SAC learning and all replay operations")


@contextmanager
def frozen_sac_guard(sac):
    # Only PPO may backpropagate; forbid SAC entry points rather than all autograd.
    with ExitStack() as stack:
        for obj, names in ((sac.sac_actor, ("learn",)),
                           (sac.critic, ("learn", "soft_update")), (sac.entropy, ("learn",))):
            for name in names:
                stack.enter_context(patch.object(obj, name, forbidden))
        for optimizer in sac.optimizers:
            stack.enter_context(patch.object(optimizer, "step", forbidden))
        for name in ("__init__", "store_transition", "sample", "load_state_dict"):
            stack.enter_context(patch.object(Memory, name, forbidden))
        try:
            yield
        finally:
            sac.assert_unchanged()


class MissionMacroPPO(learning.DirectHRL):
    """Reuse the original macro sampler/update, without constructing a SAC replay."""

    def __init__(self, config, seed, sac, *, journal=True):
        if seed not in PPO_SEEDS or config["training"]["seed"] != seed:
            raise ValueError("Fresh PPO initialization seed mismatch")
        self.config, self.cfg = copy.deepcopy(config), copy.deepcopy(config["training"])
        self.seed, self.sac = seed, sac
        self.macro_state_dim, self.micro_state_dim = 67, 68
        self.observation_spec = copy.deepcopy(sac.observations)
        if (self.observation_spec["macro"]["dim"] != 67
                or self.observation_spec["micro"]["dim"] != 68):
            raise ValueError("Stage4A requires the unchanged67/68 split observation")
        # Do not call DirectHRL.__init__ or restore_agent: neither SAC nor source PPO is initialized here.
        with isolated_rng():
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.macro_actor = ActorNet(67, 6).to(device)
            self.macro_critic = CriticNet(67).to(device)
            nn.init.zeros_(self.macro_actor.out.weight)
            nn.init.zeros_(self.macro_actor.out.bias)
            self.macro_actor.log_std.requires_grad_(False)
            self.macro_actor.log_std.data.fill_(np.log(self.cfg["ppo_initial_std"]))
            self.ppo_actor_optimizer = torch.optim.Adam(self.macro_actor.parameters(), lr=self.cfg["ppo_actor_lr"], eps=1e-5)
            self.ppo_critic_optimizer = torch.optim.Adam(self.macro_critic.parameters(), lr=self.cfg["ppo_critic_lr"], eps=1e-5)
            # Set up macro_gamma for GAE (normally done in DirectHRL.__init__)
            if "macro_gamma" in self.cfg:
                self._macro_gamma = self.cfg["macro_gamma"]
            elif "macro_gamma_frames" in self.cfg:
                self._macro_gamma = self.cfg["gamma_per_resource_frame"] ** self.cfg["macro_gamma_frames"]
            else:
                self._macro_gamma = 0.95  # Default 10-frame equivalent
            self.rng = capture_rng()
        self.macro_rollout, self.batch_log, self.update_log = [], [], []
        self.ppo_updates = self.ppo_gradient_steps = self.collected_macros = self.consumed_macros = 0
        self.last_ppo_stats = {}
        self.journal = journal
        self.initial_actor = snapshot(self.macro_actor)
        self.initial_critic = snapshot(self.macro_critic)

    store = update_sac = forbidden

    @contextmanager
    def exploration_rng(self):
        with isolated_rng():
            restore_rng(self.rng)
            try:
                yield
            finally:
                self.rng = capture_rng()

    def ppo_state(self):
        return dict(actor=snapshot(self.macro_actor), critic=snapshot(self.macro_critic),
                    optimizers=copy.deepcopy([self.ppo_actor_optimizer.state_dict(), self.ppo_critic_optimizer.state_dict()]),
                    rollout=copy.deepcopy(self.macro_rollout), rng=copy.deepcopy(self.rng),
                    counters=(self.ppo_updates, self.ppo_gradient_steps, self.collected_macros, self.consumed_macros),
                    last_stats=copy.deepcopy(self.last_ppo_stats),
                    gradients=[[None if p.grad is None else p.grad.detach().cpu().clone()
                                for p in net.parameters()] for net in (self.macro_actor, self.macro_critic)],
                    training_modes=[[m.training for m in net.modules()] for net in (self.macro_actor, self.macro_critic)])

    def update_ppo(self, progress=0., force=False):
        count, before = len(self.macro_rollout), self.ppo_updates
        if not count or (count < self.cfg["ppo_update_macros"] and not force):
            return
        original = learning.gae
        captured = []

        def observe(*args):
            result = original(*args)
            rewards, values, next_values, traces, discount, lam = args
            states, actions, logprobs, _, next_states, _ = map(np.asarray, zip(*self.macro_rollout))
            captured.append(dict(state=states.copy(), raw_action=actions.copy(), old_logprob=logprobs.copy(),
                                 next_state=next_states.copy(), reward=np.asarray(rewards).copy(),
                                 value=values.copy(), next_value=next_values.copy(), continuation=traces.copy(),
                                 advantage=result[0].copy(), returns=result[1].copy(), macro_discount=discount,
                                 gae_lambda=lam, exploration_std=self.macro_actor.log_std.detach().exp().cpu().numpy().copy()))
            return result

        context = patch.object(learning, "gae", observe) if self.journal else ExitStack()
        with context:
            super().update_ppo(progress, force)
        if self.ppo_updates != before + 1 or self.macro_rollout:
            raise AssertionError("PPO batch was not consumed exactly once")
        self.consumed_macros += count
        if self.journal:
            if len(captured) != 1:
                raise AssertionError("PPO logging must observe the single original GAE call")
            self.batch_log.append(captured[0])
            self.update_log.append(dict(update=self.ppo_updates, transitions=count, forced=force,
                                        progress=progress, **self.last_ppo_stats,
                                        ppo_gradient_steps=self.ppo_gradient_steps,
                                        std_after=float(self.macro_actor.log_std.detach().exp()[0])))
        self.sac.assert_unchanged()

    def save_smoke_checkpoint(self, path, source_sha):
        if Path(path).exists():
            raise FileExistsError(path)
        torch.save(dict(stage="mission_direct_ppo_smoke_only", formal_window=None,
                        continuation_supported=False, config=self.config, ppo_seed=self.seed,
                        source_sac_sha256=source_sha, source_sac_window=2000,
                        selective_restore="SAC only; no source PPO, source RNG or replay restored",
                        ppo=self.ppo_state(), replay=None), path)


class FullSceneMeter(MissionPilotMeter):
    """A short smoke uses the prefix of the same full900s load schedule."""

    def reset(self, seed, macro_count):
        return super().reset(seed, 90)


class Stage4Meter(FullSceneMeter):
    """Extra action diagnostics around unchanged MissionPilotMeter/QueueMDP execution."""

    def start_macro(self, action):
        raw = np.asarray(action, dtype=np.float64).reshape(3, 2)
        clipped = np.clip(raw, -1, 1)
        disk = clipped / np.maximum(np.linalg.norm(clipped, axis=1, keepdims=True), 1)
        self.requested_displacement = 20 * disk
        info = super().start_macro(action)
        for i, row in enumerate(self.commands[-3:]):
            row.update(raw_clip_triggered=int(np.any(raw[i] != clipped[i])),
                       disk_projection_triggered=int(np.linalg.norm(clipped[i]) > 1),
                       role=self.uav_role(i))
            for prefix, values in (("clipped_action", clipped[i]), ("disk_action", disk[i]),
                                   ("requested_delta", 20*disk[i]), ("executed_delta", info["actual_displacement_m"][i])):
                row.update({f"{prefix}_{axis}": float(values[j]) for j, axis in enumerate(("x", "y"))})
        return info

    def uav_role(self, uav):
        order = np.argsort(-np.asarray(self.env.zone_expected_offered_load_mbps), kind="stable")
        return ("heavy", "medium", "light")[int(np.where(order == uav)[0][0])]


def collect_stream(cfg, policy, scene, *, train, macros=90, macro_offset=0,
                   meter_factory=Stage4Meter, record=True):
    """Macro-only collection. No micro pending transition, replay or tail action exists."""
    allowed = TRAIN_SCENES if train else VALIDATION_SCENES
    if (type(scene) is not int or scene not in allowed or type(macros) is not int
            or macros not in range(1, 91) or type(macro_offset) is not int
            or macro_offset < 0 or macro_offset + macros > 3600):
        raise ValueError("Out-of-protocol scene, horizon or macro budget")
    if train and (macro_offset % 90 or scene != TRAIN_SCENES[macro_offset // 90]):
        raise ValueError("Training scene order must follow the fixed900s schedule")
    meter = meter_factory(cfg)
    # Even a short smoke is a prefix of the predefined900s scene, not a shortened mission.
    meter.reset(scene, 90)
    if (meter.env.deployment_resource_frames != 0 or meter.env.service_resource_frames != 20
            or not meter.env.continuous_mission):
        raise ValueError("Stage4A requires the unchanged20-service-frame mission protocol")
    gamma = cfg["training"]["gamma_per_resource_frame"]
    rows, observations, window_rows, routes = [], [], [], []
    initial_collected = policy.collected_macros if train else 0
    cost_before = energy_before = 0.
    for macro in range(macros):
        state = meter.macro_observation()
        raw, logprob = policy.macro_action(state, enabled=True, deterministic=not train)
        meter.start_macro(raw)
        rewards = []
        micro_state = meter.prepare_service()
        for frame in range(20):
            with torch.no_grad():
                micro = policy.sac.sac_actor.choose_action(micro_state, deterministic=True)
            reading = meter.step_frame(micro, gamma)
            rewards.append(reading["reward"])
            if record:
                routes.append((micro.copy(), reading["route"].copy()))
            if frame != 19:
                micro_state = meter.prepare_service()
        reward = sum(gamma**i * r for i, r in enumerate(rewards)) / len(rewards)
        meter.advance_macro_clock()
        next_state = meter.macro_observation()
        transition = (state.copy(), raw.copy(), logprob, reward, next_state.copy(), float(macro < macros-1))
        if train:
            policy.macro_rollout.append(transition)
            policy.collected_macros += 1
            policy.update_ppo((macro_offset + macro + 1) / 3600.)
        if record:
            observations.append(transition)
            rows.append(dict(macro=macro+1, **meter.metrics()))
        if (macro+1) % 30 == 0 or macro == macros-1:
            metrics = meter.metrics()
            duration = (macro % 30 + 1)*10
            window_rows.append(dict(scene=scene, window_end_macro=macro+1, time_s=metrics["time_s"],
                                    window_J=(meter.cost_sum-cost_before)/(duration*10),
                                    window_power_w=(meter.total_power_energy-energy_before)/duration,
                                    queue_total_mbit=metrics["queue_total_mbit"], reset_count=meter.reset_count))
            cost_before, energy_before = meter.cost_sum, meter.total_power_energy
    if (meter.reset_count != 1 or meter.integration_steps != macros*100
            or meter.env.resource_frame_prepare_count != macros*20
            or meter.env.mobility_update_count != macros*20):
        raise AssertionError("Mission decision/time/reset conservation failed")
    if train and (policy.collected_macros-initial_collected != macros
                  or policy.collected_macros != policy.consumed_macros+len(policy.macro_rollout)):
        raise AssertionError("Macro replay/batch conservation failed")
    summary = dict(meter.metrics(), boundary_transitions=0, final_pending_dropped=0,
                   macro_decisions=macros, sac_decisions=20*macros,
                   replay_writes=0, replay_samples=0, sac_update_increment=0)
    return summary, meter, rows, observations, window_rows, routes


def summarize(summary, meter, policy, scene):
    seconds = summary["time_s"]
    row = dict(summary, policy="direct_ppo", seed=scene, ppo_seed=policy.seed, duration_s=seconds,
               **summarize_meter(meter.readings, seconds))
    row.update({key: meter.readings[-1][key] for key in PRIORITY_MEASURES})
    row.update({key: sum(r[key] for r in meter.readings) for key in ENERGY})
    row.update(J_holding=sum(r["holding_cost"] for r in meter.readings)/meter.integration_steps,
               J_energy=sum(r["energy_cost"] for r in meter.readings)/meter.integration_steps,
               ppo_updates=policy.ppo_updates,
               macro_actor_parameter_delta=parameter_delta(policy.initial_actor, policy.macro_actor, exclude=("log_std",)),
               u2g_served_mbit=sum(r["served_mbit"] for r in meter.uplink_readings),
               u2g_tx_time_s=sum(r["tx_time_s"] for r in meter.uplink_readings))
    row["u2g_rate_time_weighted_mbps"] = row["u2g_served_mbit"]/max(row["u2g_tx_time_s"], 1e-12)
    tail = [r for r in meter.readings if r["time_s"] > 2*seconds/3]
    row["late_queue_slope_mbit_s"] = float(np.polyfit([r["time_s"] for r in tail], [r["queue_total_mbit"] for r in tail], 1)[0])
    row.update({f"queue_{q}_mbit": meter.readings[-1][f"queue_{q}_mbit"] for q in ("upload", "compute", "transfer")})
    stats = trajectory_statistics(meter, "direct_ppo", scene)
    np.testing.assert_allclose(row["J"], row["J_holding"]+row["J_energy"], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(row["power_w"]*seconds, sum(row[k] for k in ENERGY), rtol=1e-12, atol=1e-6)
    return row, stats


def evaluate_short(policy, scene=33600):
    before, global_rng = policy.ppo_state(), capture_rng()
    with isolated_rng(), frozen_sac_guard(policy.sac), torch.no_grad():
        result = collect_stream(policy.config, policy, scene, train=False, macros=2)
    if not state_equal(before, policy.ppo_state()) or not state_equal(global_rng, capture_rng()):
        raise AssertionError("Evaluation mutated PPO/RNG state")
    return result


def fixed_short(cfg, sac, name, scene=33600):
    if name not in BASELINES or scene not in VALIDATION_SCENES:
        raise ValueError("Only the declared fixed controls and development scenes are allowed")
    with patch("hrl_uav.mission_pilot.MissionPilotMeter", FullSceneMeter):
        return evaluate_one(cfg, sac, name, scene, _short_test=True)


def verify_baselines(source_cfg, spec):
    cfg = mission_config(source_cfg)
    prior, _, _, hashes = verify_previous(cfg, spec)
    source = ROOT / "results/mission_rpgm_v1_pilot"
    references = {}
    for name in BASELINES:
        for seed in VALIDATION_SCENES:
            folder = source / f"{name}_{seed}"
            row = json.loads((folder / "summary.json").read_text())
            if (row["policy"] != name or row["seed"] != seed or row["duration_s"] != 900
                    or row["sac_decisions"] != 1800 or row["macro_decisions"] != 90
                    or any(row[k] for k in ("ppo_updates", "replay_writes", "replay_samples", "sac_update_increment"))):
                raise ValueError("Historical macro baseline has incompatible protocol")
            records = json.loads((folder / "crn.json").read_text())
            if len(records) != 1800:
                raise ValueError("Incomplete baseline CRN")
            references[(name, seed)] = records
            for filename in ("summary.json", "crn.json", "freeze_check.json"):
                path = folder / filename
                hashes[str(path)] = file_hash(path)
    return prior, references, hashes


def run_smoke(output=None):
    output = Path(output or ROOT / "results/mission_ppo_stage4a_smoke").resolve()
    if output.exists():
        raise FileExistsError("Stage4A requires a new independent output directory")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    worktree = subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True)
    physics = physics_provenance()
    spec = load_spec()
    saved, source_manifest = load_source(spec)
    prior, historical_crn, evidence_hashes = verify_baselines(saved["config"], spec)
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    protected = [p for p in tracked if p.endswith((".py", ".json", ".md")) and not p.startswith("results/")]
    # Untracked, unrelated diagnostics are not Stage4A runtime dependencies.
    hashes = {p: file_hash(ROOT / p) for p in dict.fromkeys((*protected, *NEW_FILES))}
    source_paths = (ROOT / spec["source_checkpoint"],)
    source_hashes = {str(p): file_hash(p) for p in source_paths}
    with isolated_rng():
        sac = ReadOnlySAC(saved)
    replay_counter = int(saved["replay"]["memory_counter"])
    configs = {seed: stage_config(saved["config"], seed) for seed in PPO_SEEDS}
    source_ppo = (copy.deepcopy(saved["macro_actor"]), copy.deepcopy(saved["macro_critic"]))
    del saved
    gc.collect()
    policies = {seed: MissionMacroPPO(configs[seed], seed, sac) for seed in PPO_SEEDS}
    for a, b in ((201, 202), (201, 203), (202, 203)):
        if (state_equal(policies[a].initial_actor, policies[b].initial_actor)
                or state_equal(policies[a].initial_critic, policies[b].initial_critic)):
            raise AssertionError("PPO initializations are not distinct")
    fresh = {}
    for seed, policy in policies.items():
        fresh[str(seed)] = dict(empty_optimizers=all(not o.state_dict()["state"] for o in
                                  (policy.ppo_actor_optimizer, policy.ppo_critic_optimizer)),
                               empty_rollout=not policy.macro_rollout, ppo_updates=policy.ppo_updates,
                               actor_differs_from_source=not state_equal(policy.initial_actor, source_ppo[0]),
                               critic_differs_from_source=not state_equal(policy.initial_critic, source_ppo[1]))
        if not all(v for k, v in fresh[str(seed)].items() if k != "ppo_updates"):
            raise AssertionError("Fresh PPO restoration check failed")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="SMOKE_RUNNING", actual_head=head, worktree_before=worktree,
                    physics_base_commit=PHYSICS_BASE_COMMIT, physics=physics, protocol=protocol(),
                    configs={str(k): v for k, v in configs.items()}, source_checkpoint=spec["source_checkpoint"],
                    source_sha256=spec["checkpoint_sha256"], source_manifest_status=source_manifest["status"],
                    source_replay_counter_read_only=replay_counter, replay_instantiated=False,
                    source_hashes_before=hashes, baseline_source_head=prior["actual_head"],
                    baseline_evidence_hashes=evidence_hashes, source_checkpoint_hashes=source_hashes,
                    fresh_initializations=fresh, formal_training_started=False, reserved_scenes_executed=[])
    write_json(output / "protocol.json", protocol())
    write_json(output / "manifest.json", manifest)
    torch.set_num_threads(1)
    baseline_rows, baseline_meters = {}, {}
    for name in BASELINES:
        row, meter, _, _ = fixed_short(configs[201], sac, name)
        assert_crn_equal(historical_crn[(name, 33600)][:40], crn_records(meter))
        baseline_rows[name], baseline_meters[name] = row, meter
    zero_checks = {}
    for seed, policy in policies.items():
        summary, meter, rows, obs, _, _ = evaluate_short(policy)
        reference = baseline_meters["hover"]
        np.testing.assert_array_equal([r[1] for r in meter.positions], [r[1] for r in reference.positions])
        np.testing.assert_array_equal(meter.micro_states, reference.micro_states)
        np.testing.assert_array_equal([r[1] for r in obs], np.zeros((2, 6)))
        if meter.route_readings != reference.route_readings or meter.readings != reference.readings:
            raise AssertionError("Initial PPO is not bitwise Hover-equivalent")
        for key in ("J", "power_w", "task_delay_weighted_ms", "queue_total_mbit"):
            if summary[key] != baseline_rows["hover"][key]:
                raise AssertionError(f"Initial Hover metric mismatch: {key}")
        assert_crn_equal(crn_records(reference), crn_records(meter))
        zero_checks[str(seed)] = True
    policy = policies[201]
    with policy.exploration_rng(), frozen_sac_guard(sac):
        summary, meter, rows, obs, windows, routes = collect_stream(configs[201], policy, 72000, train=True, macros=6)
        if policy.ppo_updates != 0 or len(policy.macro_rollout) != 6:
            raise AssertionError("Short collection unexpectedly triggered a full PPO batch")
        policy.update_ppo(6/3600, force=True)
    if (policy.ppo_updates != 1 or policy.consumed_macros != 6 or policy.macro_rollout
            or policy.ppo_gradient_steps <= 0
            or parameter_delta(policy.initial_actor, policy.macro_actor, exclude=("log_std",)) <= 0):
        raise AssertionError("Smoke PPO did not update/consume its batch correctly")
    training_row, training_trajectory = summarize(summary, meter, policy, 72000)
    save_case(output / "smoke_train_seed201", "direct_ppo", 72000, training_row, meter, training_trajectory, rows, sac)
    append_csv(output / "smoke_windows.csv", windows)
    states, actions, lp, reward, next_states, trace = map(np.asarray, zip(*obs))
    np.savez_compressed(output / "smoke_macro_rollout.npz", state=states, raw_action=actions,
                        old_logprob=lp, reward=reward, next_state=next_states, continuation=trace,
                        micro_raw_action=[r[0] for r in routes], micro_executed_route=[r[1] for r in routes])
    for index, batch in enumerate(policy.batch_log, 1):
        np.savez_compressed(output / f"smoke_update{index:03d}.npz", **batch)
    append_csv(output / "smoke_updates.csv", policy.update_log)
    policy.save_smoke_checkpoint(output / "smoke_ppo_only.pt", spec["checkpoint_sha256"])
    after, after_meter, after_rows, _, _, _ = evaluate_short(policy)
    after, after_stats = summarize(after, after_meter, policy, 33600)
    assert_crn_equal(crn_records(baseline_meters["hover"]), crn_records(after_meter))
    save_case(output / "smoke_post_update", "direct_ppo", 33600, after, after_meter, after_stats, after_rows, sac)
    measures = ("J", "J_holding", "J_energy", "power_w", "task_delay_weighted_ms", "task_delay_high_ms",
                "task_delay_low_ms", "task_completion_high_rate", "task_completion_low_rate", "unfinished", "queue_total_mbit")
    pairs = [dict(baseline=name, scene=33600, duration_s=20,
                  **{f"delta_{k}": after[k]-row[k] for k in measures}) for name, row in baseline_rows.items()]
    append_csv(output / "smoke_paired_deltas.csv", pairs)
    append_csv(output / "smoke_baselines.csv", list(baseline_rows.values()))
    checks = dict(fresh_ppo=fresh, zero_mean_hover_bitwise=zero_checks, sac_frozen=sac.assert_unchanged(),
                  replay_instantiated=False, replay_writes=0, replay_samples=0,
                  source_replay_counter=replay_counter, source_replay_counter_after=replay_counter,
                  sac_update_increment=0, ppo_update_increment=policy.ppo_updates,
                  macro_actor_delta=parameter_delta(policy.initial_actor, policy.macro_actor, exclude=("log_std",)),
                  macro_collected=policy.collected_macros, macro_consumed=policy.consumed_macros,
                  macro_pending=len(policy.macro_rollout), actual_sac_decisions=120,
                  real_training_seconds=60, extra_tail_actions=0, train_reset_count=meter.reset_count,
                  recorded_raw_actions_nonzero=bool(np.any(actions != 0)), evaluation_rng_state_unchanged=True,
                  historical_crn_prefixes_match=True, all_source_files_unchanged=True)
    after_hashes = {p: file_hash(ROOT / p) for p in hashes}
    if hashes != after_hashes or any(file_hash(Path(p)) != h for p, h in {**evidence_hashes, **source_hashes}.items()):
        raise AssertionError("Source or historical evidence changed during smoke")
    write_json(output / "checks.json", checks)
    manifest.update(status="SMOKE_COMPLETE_STOPPED_NO_FORMAL_TRAINING", source_hashes_after=after_hashes,
                    checks=checks, source_unchanged=True)
    write_json(output / "manifest.json", manifest)
    (output / "result.md").write_text(
        "# Stage4A functional smoke only\n\n"
        "No120-window qualification, no900s evaluation, no joint learning.\n"
        "PPO201/202/203 were freshly initialized; only201 received a six-macro smoke update.\n"
        "All three initial deterministic policies matched Hover bitwise over20s.\n"
        "Training: PPO stochastic, SAC deterministic; unchanged original PPO math and hyperparameters.\n"
        "The single force=True update consumed six60s macro samples, NOT a300-transition formal batch.\n"
        "SAC replay was not instantiated/restored/written/sampled. Source replay count is provenance only.\n"
        "All SAC network/optimizer/alpha/gradient states are unchanged; source/historical hashes match.\n"
        "smoke_paired_deltas.csv is functional20s data, not evidence of improved mission performance.\n"
        "900s continuity and12-update budget are tested separately with a nonphysical clock test double.\n"
        "Formal training and interruption recovery are not exposed by this entry point. Stop here.\n",
        encoding="utf-8")
    return output
