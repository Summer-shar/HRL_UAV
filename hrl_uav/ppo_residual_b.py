"""Opt-in B-centered residual PPO. Only a bounded real-checkpoint smoke is exposed.

The native Gaussian policy and rollout store delta, NOT the translated/projected
action. This adapter changes only the deterministic macro execution map. The
original MDP still owns motion, safety, service, energy and reward calculations.
"""

from contextlib import ExitStack, contextmanager
import copy
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import torch

from .bandwidth import summarize_meter
from .experiment import append_csv, isolated_rng, parameter_delta, rollout, snapshot, write_json
from .learning import tensor
from .mdp import QueueMDP
from .ppo_diagnostics import LoggedDirectHRL, distribution
from .ppo_frozen import (
    DEFAULT_SOURCE, SOURCE_DIGESTS, STAGE_TRANSITION, VALIDATION_SEEDS, DEVELOPMENT_SEEDS,
    FrozenSACUpdates, assert_sac_frozen, load_json, prepare_source, sac_learning_state,
)
from .priority_grid import file_hash
from .resumption import capture_rng, restore_agent, save_boundary_checkpoint, state_equal
from .scene_prescreen import ROOT, assert_crn_equal, source_hashes
from .target_tracking_check import TrackingPolicy, tracking_action
from .trajectory_check import OpportunityMeter, crn_records
from .wireless_check import ENERGY, PRIORITY_MEASURES


BASE_HEAD = "1e6b02c92a71dc2a1f0971a30ce707cd8f5a89ba"
ORIGINAL_PPO_HEAD = "b4908d3e78d4c807013c8ff6991637dbae546129"
SEMANTICS = "delta_a Gaussian; execute native_projection(a_B(current_geometry) + delta_a)"
HYPERPARAMETERS = dict(ppo_initial_std=1/3, ppo_final_std=1/30,
                      ppo_std_decay_windows=200, ppo_update_macros=300, ppo_epochs=10,
                      ppo_actor_lr=3e-4, ppo_critic_lr=1e-3, ppo_clip=.2, ppo_target_kl=.03)


def project_action(action):
    """Match QueueMDP.start_macro: component clipping, then per-UAV unit disk."""
    raw = np.asarray(action, dtype=np.float64)
    if raw.shape not in ((6,), (3, 2)) or not np.isfinite(raw).all():
        raise ValueError("Expected six finite macro action components")
    clipped = np.clip(raw.reshape(3, 2), -1, 1)
    feasible = clipped / np.maximum(np.linalg.norm(clipped, axis=1, keepdims=True), 1)
    return feasible.ravel(), np.any(np.abs(feasible-raw.reshape(3, 2)) > 1e-12, axis=1)


def baseline_b(current, centroids, loads, radius):
    # Pure existing helper: no results files, schedule lookahead or learned teacher.
    return tracking_action(current, centroids, loads, [[0, 0]] * 3, radius, .5)


class ResidualMeter(OpportunityMeter):
    def __init__(self, config, *, agent=None, residual=True, journal=None, global_offset=0):
        super().__init__(config)
        self.agent, self.residual, self.journal = agent, residual, journal
        self.global_offset = global_offset
        self.controls = []

    def reset(self, seed, macro_count):
        self.scene_seed, self.requested_macros = seed, macro_count
        return super().reset(seed, macro_count)

    def start_macro(self, action):
        current, centroid = self.env.uav_coords.copy(), self.env.weighted_centroids.copy()
        loads = self.env.zone_expected_offered_load_mbps.copy()
        a_b, detail = baseline_b(current, centroid, loads, self.config["waypoint_radius_m"])
        policy_action = np.asarray(action).copy()
        pre = a_b + policy_action if self.residual else policy_action.copy()
        executed, projected = project_action(pre)
        state = self.macro_observation().copy()
        row = dict(scene_seed=self.scene_seed, macro_index=self.macro_index,
                   global_macro_index=self.global_offset+self.macro_index,
                   state=state, current_uav=current, current_centroid=centroid, current_load=loads,
                   B_target=detail["interpolated_target"], B_feasible_target=detail["feasible_target"],
                   B_projection=detail["projected"], a_B=a_b.copy(),
                   sampled_delta=policy_action if self.residual else np.zeros(6),
                   pre_projection=pre.copy(), a_exec=executed.copy(), projected=projected,
                   actual_displacement=None, execution_complete=False,
                   tail_only=self.macro_index >= self.requested_macros)
        if self.journal is not None and not row["tail_only"]:
            with torch.no_grad():
                mean, std = self.agent.macro_actor(tensor(state).unsqueeze(0))
            row.update(residual_mean=mean.cpu().numpy().ravel().copy(),
                       residual_std=std.cpu().numpy().ravel().copy())
            self.journal.records.append(row)
        self.controls.append(row)
        # Do NOT replace the action retained in the native PPO rollout with a_exec.
        info = super().start_macro(executed)
        row["propulsion_per_uav_j"] = np.asarray(info["E_macro_per_uav"]).copy()
        return info

    def step_frame(self, action, gamma_frame=.99):
        frame = super().step_frame(action, gamma_frame)
        if frame["phase"] == "deployment" and self.env.macro_service_pause_steps_remaining == 0:
            row = self.controls[-1]
            row["actual_displacement"] = self.env.uav_coords.copy()-row["current_uav"]
            row["execution_complete"] = True
        return frame


class ResidualJournal:
    """Observe actual native GAE, with exact causal geometry recorded at execution."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.records, self.cursor, self.pending = [], 0, None
        write_json(self.directory / "schema.json", {
            "probability_semantics": SEMANTICS,
            "geometry": "exact online current environment geometry, not decoded float32 observations",
            "B": "existing coefficient=0.5 pure tracking helper; original feasible B projection",
            "projection": "clip each component to [-1,1], then normalize rows exceeding unit norm",
            "value_advantage_return": "actual pre-update values and native GAE outputs; raw and normalized advantage",
            "KL": "native last checked epoch KL, not an extra post-update estimate",
            "critic_epochs": "counted from actual optimizer calls",
            "tail": "extra deployment completing SAC boundary experience is NOT a macro training sample",
            "indices": "zero-based scene macro index and global macro transition index",
        })

    def before(self, agent, gae_arguments, gae_result):
        if self.pending is not None:
            raise AssertionError("Residual journal has an unconsumed update")
        states, actions, logprobs, rewards, next_states, continuations = map(np.asarray, zip(*agent.macro_rollout))
        actual_rewards, values, next_values, traces, discount, trace_lambda = gae_arguments
        records = self.records[self.cursor:self.cursor+len(states)]
        if len(records) != len(states) or not all(r["execution_complete"] and not r["tail_only"] for r in records):
            raise AssertionError("Missing completed macro execution metadata")
        np.testing.assert_array_equal(states, [r["state"] for r in records])
        np.testing.assert_array_equal(actions, [r["sampled_delta"] for r in records])
        np.testing.assert_array_equal(actual_rewards, rewards)
        np.testing.assert_array_equal(traces, continuations)
        advantages, returns = gae_result
        normalized = tensor((advantages-advantages.mean())/(advantages.std()+1e-8)).cpu().numpy()
        fields = ("scene_seed", "macro_index", "global_macro_index", "current_uav", "current_centroid",
                  "current_load", "B_target", "B_feasible_target", "B_projection", "a_B", "sampled_delta",
                  "residual_mean", "pre_projection", "a_exec", "projected", "actual_displacement", "residual_std")
        data = {k: np.asarray([r[k] for r in records]) for k in fields}
        data.update(state=states, next_state=next_states, old_logprob=logprobs, reward=rewards,
                    value=values, next_value=next_values, advantage=advantages, returns=returns,
                    advantage_normalized=normalized, continuation=continuations)
        update = agent.ppo_updates+1
        path = self.directory / f"update{update:03d}.npz"
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, **data)
        self.pending = dict(update=update, transitions=len(states), std_before=float(data["residual_std"][0, 0]),
                            gradients_before=agent.ppo_gradient_steps,
                            macro_discount=float(discount), gae_lambda=float(trace_lambda),
                            projection_uav_count=int(data["projected"].sum()),
                            projection_macro_count=int(data["projected"].any(axis=1).sum()),
                            projection_fraction=float(data["projected"].mean()),
                            **distribution(np.linalg.norm(actions, axis=1), "residual_norm"))

    def after(self, agent):
        row, stats = self.pending, agent.last_ppo_stats
        if row is None or row["update"] != agent.ppo_updates:
            raise AssertionError("Residual journal/update mismatch")
        row.update(actual_actor_epochs=stats["epochs"], critic_epochs=agent.observed_critic_steps,
                   kl=stats["kl"], actor_loss=stats["actor_loss"], critic_loss=stats["critic_loss"],
                   pre_logprob_error=stats["pre_logprob_error"], ppo_gradient_steps=agent.ppo_gradient_steps,
                   actor_steps_added=agent.ppo_gradient_steps-row["gradients_before"],
                   std_after=float(agent.macro_actor.log_std.detach().exp()[0]))
        if row["critic_epochs"] != agent.cfg["ppo_epochs"]:
            raise AssertionError("Native critic step count changed")
        append_csv(self.directory / "updates.csv", [row])
        self.cursor += row["transitions"]
        self.pending = None


class ResidualAgent(LoggedDirectHRL):
    """Same nets, distributions and native updates; optional optimizer-call observation."""
    def update_ppo(self, progress=0.0, force=False):
        if getattr(self, "ppo_journal", None) is None:
            return super().update_ppo(progress, force)
        self.observed_critic_steps = 0
        original = self.ppo_critic_optimizer.step

        def step(*args, **kwargs):
            result = original(*args, **kwargs)
            self.observed_critic_steps += 1
            return result

        with patch.object(self.ppo_critic_optimizer, "step", step):
            return super().update_ppo(progress, force)


class ResidualFrozen(FrozenSACUpdates):
    def checkpoint(self, config, include_replay=True):
        saved = super().checkpoint(config, include_replay)
        saved["experiment"].update(macro_control="B_centered_residual", probability_semantics=SEMANTICS,
                                   B_coefficient=.5, smoke_only=True,
                                   resume_boundary="Audit snapshot only; do not load via absolute-action PPO runner")
        return saved


def forbidden(*args, **kwargs):
    raise AssertionError("SAC learning/replay sampling or evaluation mutation is forbidden")


@contextmanager
def frozen_guards(agent, *, evaluation=False):
    with ExitStack() as stack:
        for optimizer in (agent.sac_actor.optimizer, agent.sac_critic.optimizer, agent.entropy.optimizer):
            stack.enter_context(patch.object(optimizer, "step", forbidden))
        stack.enter_context(patch.object(agent.replay, "sample", forbidden))
        if evaluation:
            for method in ("store", "update_sac", "update_ppo"):
                stack.enter_context(patch.object(agent, method, forbidden))
        yield


def evaluate_pair(cfg, agent, output, label, *, seed=33600, macros=6, require_zero=False):
    if type(seed) is not int or seed not in VALIDATION_SEEDS+DEVELOPMENT_SEEDS:
        raise ValueError("Only declared development scenes are accessible")
    if type(macros) is not int or macros != 6:
        raise ValueError("This entry permits only six-macro functional evaluations")
    before, rng = copy.deepcopy(agent.checkpoint(cfg)), capture_rng()
    results, meters, arrays = {}, {}, {}
    for name, residual in (("residual", True), ("B", False)):
        directory = Path(output) / label / name
        directory.mkdir(parents=True, exist_ok=False)
        policy = agent if residual else TrackingPolicy(agent, dict(
            name="B", offsets_m=[[0, 0]]*3, tracking_coefficient=.5))

        def factory(config):
            meter = ResidualMeter(config, residual=residual)
            if not residual:
                policy.meter = meter
            return meter

        with isolated_rng(), frozen_guards(agent, evaluation=True), patch("hrl_uav.experiment.QueueMDP", factory):
            summary, rows, trajectory, routes, meter = rollout(
                cfg, policy, seed, macros, macro_enabled=True, train=False, complete_tail=False, record=True)
        if meter.reset_calls != 1 or summary["time_s"] != macros*10 or summary["boundary_transitions"] != 0:
            raise AssertionError("Evaluation boundary protocol changed")
        summary.update(summarize_meter(meter.readings, summary["time_s"]))
        summary.update({k: meter.readings[-1][k] for k in PRIORITY_MEASURES})
        summary.update({k: sum(r[k] for r in meter.readings) for k in ENERGY})
        summary.update(J_holding=meter.objective_holding/meter.integration_steps,
                       J_energy=meter.objective_energy/meter.integration_steps)
        np.testing.assert_allclose(summary["J"], summary["J_holding"]+summary["J_energy"], atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(summary["power_w"]*summary["time_s"], sum(summary[k] for k in ENERGY), atol=1e-6, rtol=1e-12)
        uav, users, centroids, variable = map(np.asarray, zip(*trajectory))
        raw, executed_route = map(np.asarray, zip(*routes))
        if len(routes) != macros*meter.env.service_resource_frames:
            raise AssertionError("Changed number of SAC routing decisions")
        data = dict(uav=uav, users=users, centroid=centroids, policy_variable=variable,
                    sac_raw=raw, executed_route=executed_route)
        for k in ("B_target", "B_feasible_target", "a_B", "sampled_delta", "a_exec", "actual_displacement"):
            data[k] = np.asarray([r[k] for r in meter.controls])
        np.savez_compressed(directory / "trajectory.npz", **data)
        append_csv(directory / "frames.csv", meter.readings)
        append_csv(directory / "deployments.csv", meter.deployments)
        append_csv(directory / "routes.csv", meter.route_readings)
        append_csv(directory / "crn.csv", crn_records(meter))
        write_json(directory / "metrics.json", summary)
        results[name], meters[name], arrays[name] = summary, meter, data
    assert_crn_equal(crn_records(meters["residual"]), crn_records(meters["B"]))
    for key in ("users", "centroid"):
        np.testing.assert_array_equal(arrays["residual"][key], arrays["B"][key])
    errors = {k: float(results["residual"][k]-results["B"][k]) for k in results["residual"]
              if isinstance(results["residual"][k], (int, float, np.number))}
    if require_zero:
        np.testing.assert_array_equal(arrays["residual"]["sampled_delta"], 0)
        for key in ("B_target", "B_feasible_target", "a_B", "a_exec", "actual_displacement", "uav", "sac_raw", "executed_route"):
            np.testing.assert_allclose(arrays["residual"][key], arrays["B"][key], atol=1e-9, rtol=0, err_msg=key)
        for key, error in errors.items():
            np.testing.assert_allclose(error, 0, atol=1e-9, rtol=0, err_msg=key)
    if not state_equal(before, agent.checkpoint(cfg)) or not state_equal(rng, capture_rng()):
        raise AssertionError("Evaluation mutated learning/replay state or global RNG")
    result = dict(seed=seed, macros=macros, duration_s=macros*10, window0_B_equal=require_zero,
                  CRN_equal=True, learning_replay_rng_unchanged=True, metrics=results,
                  residual_minus_B=errors)
    write_json(Path(output) / label / "pair.json", result)
    return result


def prepare_residual_source(source=DEFAULT_SOURCE):
    cfg, saved, manifest, report, scenes = prepare_source(source, 120, 72000)
    prior_path = ROOT / "results/ppo_frozen_sac_pilot120_seed101/manifest.json"
    prior = load_json(prior_path)
    if (prior["actual_head"] != ORIGINAL_PPO_HEAD or prior["config"] != cfg
            or prior["source_checkpoint_sha256"] != SOURCE_DIGESTS["checkpoint"]
            or prior["training_scene_seeds"] != scenes):
        raise ValueError("Residual experiment does not match original PPO120 provenance/config")
    if any(cfg["training"].get(k) != v for k, v in HYPERPARAMETERS.items()):
        raise ValueError("Original PPO120 hyperparameters required, not low std")
    for name, digest in prior["source_hashes"].items():
        if name != "hrl_uav/ppo_frozen.py" and file_hash(ROOT/name) != digest:
            raise ValueError(f"Original PPO120 source changed: {name}")
    # The sole previously reviewed runner extension is opt-in logging/low-std support.
    subprocess.run(["git", "diff", "--exit-code", BASE_HEAD, "--", "hrl_uav/ppo_frozen.py"],
                   cwd=ROOT, capture_output=True, check=True)
    b_path = ROOT / "results/progressive_centroid_dev_recheck/manifest.json"
    b_manifest = load_json(b_path)
    if (b_manifest["status"] != "COMPLETE" or b_manifest["source_digests"] != SOURCE_DIGESTS
            or b_manifest["B_controller_sha256"] != file_hash(ROOT/"hrl_uav/target_tracking_check.py")
            or any(b_manifest["config"][k] != cfg[k] for k in cfg if k != "training")):
        raise ValueError("B controller/model does not match its completed development recheck")
    return cfg, saved, dict(sac_manifest_sha256=file_hash(Path(source).parent.parent/"manifest.json"),
                            sac_report_sha256=file_hash(Path(source).parent.parent/"report.json"),
                            original_ppo_manifest_sha256=file_hash(prior_path),
                            B_manifest_sha256=file_hash(b_path), B_controller_sha256=b_manifest["B_controller_sha256"],
                            source_status=report["status"], original_source_hashes=manifest["source_hashes"])


def train_smoke_window(cfg, agent, output, *, journal=None):
    """One native window, then native force flush; the formal 300 threshold is unchanged."""
    agent.ppo_journal = journal
    proxy = ResidualFrozen(agent, SOURCE_DIGESTS["checkpoint"])
    frozen = sac_learning_state(agent, cfg)
    actor, critic = snapshot(agent.macro_actor), snapshot(agent.macro_critic)
    replay = agent.replay.memory_counter
    instances = []

    def factory(config):
        meter = ResidualMeter(config, agent=agent, journal=journal)
        instances.append(meter)
        return meter

    macros = cfg["training"]["window_macros"]
    with frozen_guards(agent), patch("hrl_uav.experiment.QueueMDP", factory):
        summary, rows, trajectory, routes, meter = rollout(cfg, proxy, 72000, macros,
            macro_enabled=True, train=True, complete_tail=True, record=True)
        proxy.update_ppo(1/cfg["training"]["joint_windows"], force=True)
    if (len(instances) != 1 or agent.macro_rollout or proxy.consumed_macro_transitions != macros
            or summary["boundary_transitions"] != macros or summary["final_pending_dropped"] != 0
            or agent.replay.memory_counter-replay != macros*meter.env.service_resource_frames
            or proxy.blocked_update_calls != macros*meter.env.service_resource_frames//cfg["training"]["sac_update_frames"]
            or not np.isclose(summary["boundary_discount_min"], cfg["training"]["gamma_per_resource_frame"]**3)
            or len(meter.controls) != macros+1 or not meter.controls[-1]["tail_only"]):
        raise AssertionError("Native replay/boundary/macro conservation failed")
    if journal is not None and (journal.cursor != macros or len(journal.records) != macros or journal.pending is not None):
        raise AssertionError("Journal did not consume each macro transition exactly once")
    checks = assert_sac_frozen(agent, cfg, frozen)
    actor_delta = parameter_delta(actor, agent.macro_actor, ("log_std",))
    critic_delta = parameter_delta(critic, agent.macro_critic)
    if agent.ppo_updates != 1 or actor_delta <= 0 or critic_delta <= 0:
        raise AssertionError("Smoke must update both PPO networks exactly one batch")
    controls = meter.controls[:macros]
    report = dict(ppo_updates=agent.ppo_updates, actor_gradient_steps=agent.ppo_gradient_steps,
                  critic_optimizer_steps=agent.observed_critic_steps if journal else agent.cfg["ppo_epochs"],
                  critic_step_count_source="observed optimizer calls" if journal else "native fixed epoch count",
                  actor_max_delta=actor_delta,
                  critic_max_delta=critic_delta, sac_frozen_checks=checks,
                  sac_update_increment=0, sac_updates_at_entry=frozen["sac_updates"],
                  sac_updates_at_exit=sac_learning_state(agent, cfg)["sac_updates"],
                  update_events=proxy.ppo_update_events, last_ppo_stats=agent.last_ppo_stats,
                  replay_added=agent.replay.memory_counter-replay, boundary_transitions=summary["boundary_transitions"],
                  boundary_discount=summary["boundary_discount_min"], pending_dropped=summary["final_pending_dropped"],
                  macro_transitions_consumed=proxy.consumed_macro_transitions, macro_transitions_pending=0,
                  projection_uav_count=int(sum(r["projected"].sum() for r in controls)),
                  projection_macro_count=int(sum(r["projected"].any() for r in controls)),
                  extra_tail_deployments=1, blocked_sac_update_calls=proxy.blocked_update_calls,
                  training_scene=72000, logged_macro_count=journal.cursor if journal else 0,
                  summary_time_s=summary["time_s"], environment_tail_time_s=meter.env._current_time())
    append_csv(Path(output)/"training.csv", [r for r in rows if "window" in r])
    append_csv(Path(output)/"training_macros.csv", [r for r in rows if "window" not in r])
    append_csv(Path(output)/"training_deployments.csv", meter.deployments)
    write_json(Path(output)/"training_controls.json", [
        {k: v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else v
         for k, v in row.items()} for row in meter.controls])
    write_json(Path(output)/"training_audit.json", report)
    save_boundary_checkpoint(proxy, cfg, Path(output)/"smoke_window1.pt", 1, 72001)
    return report, trajectory, routes


def run_smoke(output):
    """No formal-training switch is exposed in this round."""
    cfg, saved, provenance = prepare_residual_source()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    model = QueueMDP(cfg)
    model.reset(72000, cfg["training"]["window_macros"])
    agent = ResidualAgent.from_mdp(model, cfg)
    audit = restore_agent(agent, saved, dict(checkpoint=str(DEFAULT_SOURCE),
        checkpoint_sha256=SOURCE_DIGESTS["checkpoint"], transition=STAGE_TRANSITION,
        source_stage="pretrain", source_sac_window=2000, target_stage="B_residual_ppo_only",
        completed_ppo_windows=0, replay_counter=int(saved["replay"]["memory_counter"])))
    del saved
    frozen = sac_learning_state(agent, cfg)
    agent.set_stage("joint")
    assert_sac_frozen(agent, cfg, frozen)
    if (agent.ppo_updates or agent.ppo_gradient_steps or agent.macro_rollout
            or agent.ppo_actor_optimizer.state or agent.ppo_critic_optimizer.state
            or torch.count_nonzero(agent.macro_actor.out.weight).item()
            or torch.count_nonzero(agent.macro_actor.out.bias).item()):
        raise AssertionError("PPO must restore the untouched zero-mean source initialization")
    np.testing.assert_allclose(agent.macro_actor.log_std.detach().exp().cpu().numpy(), 1/3, atol=1e-7, rtol=0)
    hashes = source_hashes()
    # Explicitly include the pure B helper and new path even if the inventory evolves.
    for name in ("hrl_uav/ppo_residual_b.py", "hrl_uav/target_tracking_check.py",
                 "scripts/run_ppo_residual_b.py", "tests/test_ppo_residual_b.py"):
        hashes[name] = file_hash(ROOT/name)
    manifest = dict(status="RESTORED_NOT_TRAINED", method="B-centered residual PPO + frozen SAC",
        probability_semantics=SEMANTICS, config=cfg, hyperparameter_overrides={},
        actual_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        worktree=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        source_hashes=hashes, source_checkpoint=str(DEFAULT_SOURCE),
        source_checkpoint_sha256=SOURCE_DIGESTS["checkpoint"], provenance=provenance, restore_audit=audit,
        stage_transition=STAGE_TRANSITION, target_stage="B_residual_ppo_only", smoke_only=True,
        formal_budget_config_windows=120, executed_training_windows=1, formal_training_authorized=False,
        smoke_batch="30 native macro transitions, native force flush at 1/120 progress; threshold remains 300",
        evaluation_scene=33600, evaluation_duration_s=60, training_scene=72000,
        sac_training_actions="stochastic, unchanged frozen-PPO protocol", sac_evaluation_actions="deterministic",
        replay_collection=True, replay_sampling=False, resume_supported=False,
        reserved_scenes_executed=[], scope="Functional check only, no performance inference")
    write_json(output/"manifest.json", manifest)
    initial = evaluate_pair(cfg, agent, output, "window0", require_zero=True)
    journal = ResidualJournal(output/"ppo_samples")
    manifest["status"] = "WINDOW0_B_EQUAL_SMOKE_UPDATING"
    write_json(output/"manifest.json", manifest)
    training, _, _ = train_smoke_window(cfg, agent, output, journal=journal)
    final = evaluate_pair(cfg, agent, output, "after_smoke")
    assert_sac_frozen(agent, cfg, frozen)
    if hashes != {p: file_hash(ROOT/p) for p in hashes}:
        raise AssertionError("Source changed during smoke")
    for name, path in (("checkpoint", DEFAULT_SOURCE), ("manifest.json", DEFAULT_SOURCE.parent.parent/"manifest.json"),
                       ("report.json", DEFAULT_SOURCE.parent.parent/"report.json")):
        if file_hash(path) != SOURCE_DIGESTS[name]:
            raise AssertionError("SAC source changed during smoke")
    report = dict(status="RESIDUAL_B_SMOKE_ONLY_COMPLETE", initial=initial, training=training,
                  final=final, reserved_scenes_executed=[], source_unchanged=True, resume_supported=False,
                  limitation="One forced partial batch, not a 300-macro natural update or performance evidence")
    write_json(output/"report.json", report)
    manifest.update(status=report["status"], source_unchanged=True,
                    smoke_checkpoint_sha256=file_hash(output/"smoke_window1.pt"))
    write_json(output/"manifest.json", manifest)
    return report
