"""Train the macro PPO policy on top of a completed SAC-only checkpoint.

This is a component-isolation runner for the candidate 6 MHz model.  The
restored SAC continues to act at every service frame, but all SAC optimizer
updates are blocked during the PPO stage.  Replay collection is retained so a
later, separately authorized joint fine-tuning experiment can start from the
actual PPO-induced state distribution without changing the frozen-stage policy.
"""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import torch

from .experiment import append_csv, isolated_rng, parameter_delta, rollout, snapshot, write_json
from .learning import DirectHRL
from .mdp import QueueMDP
from .priority_grid import file_hash
from .qualification import means, metered_evaluate, verify_crn
from .resumption import capture_rng, restore_agent, save_boundary_checkpoint, state_equal
from .scene_prescreen import ROOT, PairedMeter, assert_crn_equal, digest_update, source_hashes


DEFAULT_SOURCE = ROOT / "results/sac_only_wireless6_long_seed101/checkpoints/window2000.pt"
DEFAULT_TRAINING_SEED_BASE = 72000
VALIDATION_SEEDS = (33600, 33601)
DEVELOPMENT_SEEDS = (33610, 33611, 33612)
RESERVED_SEEDS = tuple(range(33700, 33710))
# Seed101 provenance frozen by ab5287d; new branch files are not old source inputs.
SOURCE_DIGESTS = {
    "checkpoint": "be946d3a576918460690a16f1a57f2c6dc33dde13db86d360db5690858a76932",
    "manifest.json": "74d6ee2568a96332ebb1f8e5e635a6685cd9cfc4dce2437ca1206dc248a2220e",
    "report.json": "3f8738199d1b00f7f2d725d1709efa8e94cd8758cd3bf90cd322d3f64675761c",
}
STAGE_TRANSITION = "completed_sac_only_checkpoint_to_ppo_only_macro_stage"


class FrozenSACUpdates:
    """Transparent agent proxy that blocks SAC learning but keeps SAC acting.

    ``store`` is intentionally delegated to the underlying agent.  Therefore
    replay can collect the PPO-induced experience distribution while the SAC
    actor, Critic/target Critic, entropy temperature and their optimizers remain
    bitwise frozen.
    """

    def __init__(self, agent, source_checkpoint_sha256=None):
        self._agent = agent
        self.source_checkpoint_sha256 = source_checkpoint_sha256
        self.blocked_update_calls = 0
        self.ppo_update_events = []
        self.consumed_macro_transitions = 0

    def __getattr__(self, name):
        return getattr(self._agent, name)

    def update_sac(self):
        self.blocked_update_calls += 1
        return None

    def checkpoint(self, config, include_replay=True):
        state = self._agent.checkpoint(config, include_replay=include_replay)
        state["experiment"] = {
            "stage_transition": STAGE_TRANSITION, "learning_scope": "ppo_only_sac_frozen",
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "resume_supported": False,
        }
        return state

    def update_ppo(self, progress, force=False):
        count, before = len(self._agent.macro_rollout), self._agent.ppo_updates
        result = self._agent.update_ppo(progress, force=force)
        if self._agent.ppo_updates != before:
            if self._agent.ppo_updates != before + 1 or self._agent.macro_rollout:
                raise AssertionError("PPO update did not consume exactly one rollout batch")
            self.consumed_macro_transitions += count
            self.ppo_update_events.append({
                "update": self._agent.ppo_updates, "transitions": count,
                "progress": float(progress), "forced": bool(force),
                "std_after": self._agent.macro_actor.log_std.detach().exp().cpu().tolist(),
            })
        return result


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sac_learning_state(agent, config):
    """State that must not change while SAC is frozen; replay data is excluded."""
    state = agent.checkpoint(config, include_replay=False)
    update_keys = ("sac_critic", "sac_critic_gradients", "sac_target", "sac_actor")
    networks = (agent.sac_actor.action_net, agent.sac_critic.critic_v, agent.sac_critic.target_critic_v)
    return {
        "sac_actor": snapshot(networks[0]),
        "sac_critic": snapshot(networks[1]),
        "sac_target": snapshot(networks[2]),
        "log_alpha": state["log_alpha"].clone(),
        "cached_alpha": agent.entropy.alpha.detach().clone(),
        "behavior": {
            "action_bounds": (agent.sac_actor.min_action, agent.sac_actor.max_action),
            "target_entropy": agent.entropy.target_entropy,
            "training_modes": [[module.training for module in net.modules()] for net in networks],
            "requires_grad": [[p.requires_grad for p in net.parameters()] for net in networks],
            "route_width": [net.route_width for net in networks[1:]],
            "config": copy.deepcopy(agent.cfg),
            "observations": copy.deepcopy(agent.observation_spec),
        },
        "gradients": [[None if p.grad is None else p.grad.detach().clone()
                       for p in net.parameters()] for net in networks],
        "alpha_gradient": None if agent.entropy.log_alpha.grad is None else agent.entropy.log_alpha.grad.detach().clone(),
        "sac_optimizers": copy.deepcopy(state["optimizers"][:3]),
        "sac_updates": {key: int(state["updates"][key]) for key in update_keys},
        "stage_initial_updates": int(state["training_state"]["stage_initial_updates"]),
        "last_sac_stats": copy.deepcopy(state["training_state"]["last_sac_stats"]),
        "replay_rng_state": copy.deepcopy(agent.replay.state_dict()["rng_state"]),
    }


def assert_sac_frozen(agent, config, reference):
    current = sac_learning_state(agent, config)
    checks = {key: state_equal(reference[key], current[key]) for key in reference}
    if not all(checks.values()):
        raise AssertionError(f"Frozen SAC state changed during PPO training: {checks}")
    return checks


def _source_run(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("Source must be a checkpoint under a completed SAC-only run/checkpoints directory")
    return checkpoint.parent.parent


def prepare_source(checkpoint, ppo_windows, training_seed_base):
    checkpoint = Path(checkpoint).resolve()
    run = _source_run(checkpoint)
    for name, path in (("checkpoint", checkpoint), ("manifest.json", run / "manifest.json"),
                       ("report.json", run / "report.json")):
        if not path.is_file() or file_hash(path) != SOURCE_DIGESTS[name]:
            raise ValueError(f"Seed101 frozen-baseline provenance mismatch: {name}")
    manifest = load_json(run / "manifest.json")
    report = load_json(run / "report.json")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if manifest.get("status") != "LONG_COMPLETE" or report.get("status") != "LONG_COMPLETE":
        raise ValueError("PPO bootstrap requires a completed SAC-only long run")
    if report.get("final_window") != 2000 or manifest.get("training_initialization_seed") != 101:
        raise ValueError("This first PPO isolation protocol is anchored to the predeclared seed101 window2000 reference")
    if manifest.get("macro_enabled") is not False or saved.get("stage") != "pretrain":
        raise ValueError("Source is not a pure SAC-only pretraining checkpoint")
    if (saved["updates"].get("ppo") != 0 or saved["updates"].get("ppo_gradients") != 0
            or saved["training_state"].get("macro_rollout")):
        raise ValueError("Source already contains PPO learning state")
    if saved["optimizers"][3]["state"] or saved["optimizers"][4]["state"]:
        raise ValueError("Source PPO optimizer is not fresh")
    run_state = saved.get("run_state", {})
    if (run_state.get("completed_windows") != 2000 or not run_state.get("at_rollout_boundary")
            or saved["replay"]["memory_counter"] <= 0 or not run_state.get("rng")):
        raise ValueError("Source is not the complete window2000 rollout boundary")
    if saved["config"] != manifest["config"]:
        raise ValueError("Checkpoint and source manifest configuration differ")

    recorded = [item for item in manifest.get("checkpoints", []) if int(item.get("window", -1)) == 2000]
    digest = file_hash(checkpoint)
    if not recorded or all(item.get("sha256") != digest for item in recorded):
        raise ValueError("Source window2000 checkpoint hash is not registered in the completed run")

    source_mismatches = []
    for relative, expected in manifest["source_hashes"].items():
        path = ROOT / relative
        if not path.is_file() or file_hash(path) != expected:
            source_mismatches.append(relative)
    if source_mismatches:
        raise ValueError(f"Existing model/training source changed since SAC101: {source_mismatches}")

    cfg = copy.deepcopy(saved["config"])
    if (cfg["environment"].get("wireless_model") != "psd_per_uav_budget"
            or cfg["environment"].get("u2u_bandwidth_mhz") != 6
            or not cfg["environment"].get("compute_during_deployment")
            or not cfg["environment"].get("pause_service_during_flight")
            or cfg["training"]["seed"] != 101
            or cfg["training"]["pretrain_windows"] != 2000
            or cfg["training"]["joint_windows"] != 0
            or cfg["training"].get("sac_stage_policy") != "preserve"):
        raise ValueError("Source is not the frozen candidate-6MHz SAC2000 configuration")
    if tuple(cfg["validation_seeds"]) != VALIDATION_SEEDS or tuple(cfg["test_seeds"]) != DEVELOPMENT_SEEDS:
        raise ValueError("Development scene contract differs from the SAC-only reference")
    if tuple(cfg["pilot"]["reserved_final_seeds"]) != RESERVED_SEEDS:
        raise ValueError("Reserved unseen-scene contract differs")

    if type(ppo_windows) is not int or ppo_windows <= 0:
        raise ValueError("ppo_windows must be a positive integer")
    if type(training_seed_base) is not int or training_seed_base < 0:
        raise ValueError("training_seed_base must be a nonnegative integer")
    cfg["training"]["joint_windows"] = ppo_windows
    stream = int(cfg["training"]["stream_windows"])
    scene_count = (ppo_windows + stream - 1) // stream
    training_scenes = list(range(int(training_seed_base), int(training_seed_base) + scene_count))
    forbidden = set(VALIDATION_SEEDS) | set(DEVELOPMENT_SEEDS) | set(RESERVED_SEEDS)
    if set(training_scenes) & forbidden:
        raise ValueError("PPO training scenes overlap development or reserved scenes")

    return cfg, saved, manifest, report, training_scenes


class MacroPairMeter(PairedMeter):
    """Reuse task/schedule hashes, but exclude controlled UAV motion from CRN."""

    def __init__(self, config):
        super().__init__(config)
        self.users_digest = hashlib.sha256()
        integrate = self.env._integrate_user_positions

        def record_users():
            result = integrate()
            digest_update(self.users_digest, self.env.global_micro_step, self.env.user_coords)
            return result

        self.env._integrate_user_positions = record_users

    def reset(self, seed, macro_count):
        result = super().reset(seed, macro_count)
        digest_update(self.users_digest, self.env.global_micro_step, self.env.user_coords)
        return result

    def step_frame(self, action, gamma_frame=0.99):
        frame = super().step_frame(action, gamma_frame)
        self.crn_frames[-1]["users_microsteps_sha256"] = self.users_digest.hexdigest()
        return frame


def exogenous_records(meter):
    keys = ("time_s", "task_arrived_count", "arrivals_sha256", "load_sha256",
            "schedule_sha256", "users_microsteps_sha256")
    return [{key: row[key] for key in keys} for row in meter.crn_frames]


def paired_evaluation(config, agent, seeds, output, label, window, *, macros=90):
    """Compare learned trajectory against dynamic-centroid motion with the same SAC."""
    seeds = list(seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed not in VALIDATION_SEEDS + DEVELOPMENT_SEEDS for seed in seeds)):
        raise ValueError("Only the declared validation/development scenes may be evaluated")
    if type(macros) is not int or macros < 6:
        raise ValueError("At least six evaluation macros are needed for the existing queue-slope meter")
    before_rng = capture_rng()
    # state_dict tensors alias live parameters; an independent copy is required.
    before = copy.deepcopy(agent.checkpoint(config))
    before_sac = sac_learning_state(agent, config)
    ppo_label = f"{label}_ppo_w{window}"
    centroid_label = f"{label}_centroid_w{window}"
    meters = []

    def factory(cfg):
        meter = MacroPairMeter(cfg)
        meters.append(meter)
        return meter

    with patch("hrl_uav.qualification.MeteredMDP", factory):
        ppo = metered_evaluate(config, agent, seeds, macros, output, ppo_label, macro_enabled=True)
        centroid = metered_evaluate(config, agent, seeds, macros, output, centroid_label, macro_enabled=False)
    if not state_equal(before, agent.checkpoint(config)) or not state_equal(before_rng, capture_rng()):
        raise AssertionError("Read-only paired evaluation changed learning state or RNG")
    assert_sac_frozen(agent, config, before_sac)
    if [row["seed"] for row in ppo] != seeds or [row["seed"] for row in centroid] != seeds:
        raise AssertionError("Paired evaluation scene identities differ")
    crn = verify_crn(output, list(seeds), ppo_label, centroid_label)
    if not all(item["users"] and item["arrivals"] for item in crn.values()):
        raise AssertionError("PPO and centroid paired evaluations do not share exogenous streams")
    if len(meters) != 2 * len(seeds):
        raise AssertionError("Missing paired physical meters")
    for index, seed in enumerate(seeds):
        left, right = exogenous_records(meters[index]), exogenous_records(meters[index + len(seeds)])
        assert_crn_equal(left, right)
        for pair_label, records in ((ppo_label, left), (centroid_label, right)):
            append_csv(output / f"{pair_label}_{seed}_crn.csv", records)
        crn[str(seed)].update(task_births=True, user_microsteps=True, load_schedule=True, frames=len(left))
    deltas = []
    for a, b in zip(ppo, centroid):
        deltas.append({
            "window": int(window), "seed": int(a["seed"]),
            "ppo_minus_centroid_J": a["J"] - b["J"],
            "ppo_minus_centroid_delay_ms": a["task_delay_weighted_ms"] - b["task_delay_weighted_ms"],
            "ppo_minus_centroid_high_ms": a["task_delay_high_ms"] - b["task_delay_high_ms"],
            "ppo_minus_centroid_low_ms": a["task_delay_low_ms"] - b["task_delay_low_ms"],
            "ppo_minus_centroid_queue_mbit": a["queue_total_mbit"] - b["queue_total_mbit"],
            "ppo_minus_centroid_completion": a["task_completion_rate"] - b["task_completion_rate"],
            "ppo_minus_centroid_power_w": a["power_w"] - b["power_w"],
        })
    append_csv(output / "paired_macro_deltas.csv", deltas)
    return {"window": int(window), "evaluation_macros": macros, "learning_state_unchanged": True,
            "rng_unchanged": True, "ppo": ppo, "centroid": centroid,
            "ppo_mean": means(ppo), "centroid_mean": means(centroid), "deltas": deltas, "crn": crn}


def run_frozen_ppo(output, *, source=DEFAULT_SOURCE, ppo_windows, training_seed_base=DEFAULT_TRAINING_SEED_BASE,
                   validation_every=30, smoke=False, initial_std=None, sample_diagnostics=False):
    """Run PPO-only macro training on one frozen SAC101 checkpoint.

    This routine never evaluates the reserved 33700--33709 scenes.  It is a
    development experiment until a separate protocol explicitly authorizes an
    untouched test.
    """
    output = Path(output).resolve()
    source = Path(source).resolve()
    cfg, saved, source_manifest, source_report, training_scenes = prepare_source(
        source, ppo_windows, training_seed_base)
    if type(validation_every) is not int or validation_every <= 0:
        raise ValueError("validation_every must be a positive integer")
    if smoke and ppo_windows != 1:
        raise ValueError("Smoke is exactly one native window; no algorithm hyperparameters are changed")
    if initial_std is not None and initial_std != 1 / 6:
        raise ValueError("Only the predeclared low-exploration initial std 1/6 is authorized")
    reference_pilot = None
    if initial_std is not None:
        prior_dir = ROOT / "results/ppo_frozen_sac_pilot120_seed101"
        prior = load_json(prior_dir / "manifest.json")
        expected_config = copy.deepcopy(prior["config"])
        if smoke:
            expected_config["training"]["joint_windows"] = 1
        if (cfg != expected_config or training_scenes != prior["training_scene_seeds"][:len(training_scenes)]
                or prior["actual_head"] != "b4908d3e78d4c807013c8ff6991637dbae546129"
                or prior["source_checkpoint_sha256"] != SOURCE_DIGESTS["checkpoint"]
                or validation_every != 30):
            raise ValueError("Low-std pilot must otherwise match the original PPO120 protocol")
        for name, digest in prior["source_hashes"].items():
            if name != "hrl_uav/ppo_frozen.py" and file_hash(ROOT / name) != digest:
                raise ValueError(f"Original PPO120 model/training implementation changed: {name}")
        reference_pilot = {
            "manifest": str(prior_dir / "manifest.json"),
            "manifest_sha256": file_hash(prior_dir / "manifest.json"),
            "report_sha256": file_hash(prior_dir / "report.json"),
            "source_comparison": "All original inputs unchanged except opt-in ppo_frozen runner extension",
        }
    if sample_diagnostics:
        from .ppo_diagnostics import LoggedDirectHRL, PPOJournal
        agent_type = LoggedDirectHRL
    else:
        agent_type = DirectHRL
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    validation_seeds = VALIDATION_SEEDS[:1] if smoke else VALIDATION_SEEDS
    development_seeds = DEVELOPMENT_SEEDS[:1] if smoke else DEVELOPMENT_SEEDS
    evaluation_macros = 6 if smoke else 90

    torch.set_num_threads(1)
    model = QueueMDP(cfg)
    model.reset(training_scenes[0], cfg["training"]["window_macros"])
    agent = agent_type.from_mdp(model, cfg)
    restore_meta = {
        "checkpoint": str(source), "checkpoint_sha256": file_hash(source),
        "transition": STAGE_TRANSITION, "source_stage": "pretrain", "source_sac_window": 2000,
        "target_stage": "ppo_only", "completed_ppo_windows": 0,
        "target_first_training_scene_seed": training_scenes[0],
        "replay_counter": int(saved["replay"]["memory_counter"]),
        "rng_mode": "exact SAC2000 bootstrap before PPO-only macro training",
    }
    restore_audit = restore_agent(agent, saved, restore_meta)
    del saved
    if agent.stage != "pretrain" or agent.ppo_updates != 0:
        raise AssertionError("Restored source is not a PPO-fresh SAC-only agent")
    # Apply only AFTER the complete, unmodified SAC2000 bootstrap is verified.
    overrides = {}
    if initial_std is not None:
        overrides["ppo_initial_std"] = {"source": cfg["training"]["ppo_initial_std"], "effective": initial_std}
        cfg["training"]["ppo_initial_std"] = initial_std
        with torch.no_grad():
            agent.macro_actor.log_std.fill_(np.log(initial_std))
    if sample_diagnostics:
        agent.ppo_journal = PPOJournal(output / "ppo_samples", agent.observation_spec, cfg)
    frozen_reference = sac_learning_state(agent, cfg)
    # The shared scheduler's runtime stage is joint, but SAC updates stay blocked.
    # This is not continuation validation, and saved['config'] is never rewritten.
    agent.set_stage("joint")
    assert_sac_frozen(agent, cfg, frozen_reference)
    initial_macro = snapshot(agent.macro_actor)
    initial_replay = agent.replay.memory_counter
    proxy = FrozenSACUpdates(agent, restore_meta["checkpoint_sha256"])

    with isolated_rng():
        probe = QueueMDP(cfg)
        probe.reset(validation_seeds[0], evaluation_macros)
        initial_macro_action, _ = agent.macro_action(probe.macro_observation(), enabled=True, deterministic=True)
    if float(np.max(np.abs(initial_macro_action))) > 1e-7:
        raise AssertionError("Source PPO mean is not the expected zero-head centroid-equivalent initialization")

    current_sources = source_hashes()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    manifest = {
        "status": "RESTORED_NOT_TRAINED", "method": "PPO + frozen pretrained SAC",
        "stage_transition": STAGE_TRANSITION, "runtime_stage": "joint", "smoke": bool(smoke),
        "resume_supported": False,
        "resume_boundary": "Boundary snapshots are audit artifacts; this entry has no interruption-resume protocol",
        "sac_action_mode": "stochastic during macro training, deterministic during paired evaluation",
        "evaluation_macros": evaluation_macros,
        "actual_head": head, "branch": branch, "config": cfg,
        "authorized_post_restore_overrides": overrides,
        "ppo_sample_diagnostics": bool(sample_diagnostics),
        "reference_pilot": reference_pilot,
        "source_checkpoint": str(source), "source_checkpoint_sha256": file_hash(source),
        "source_run": str(_source_run(source)), "source_manifest_sha256": file_hash(_source_run(source) / "manifest.json"),
        "source_report_sha256": file_hash(_source_run(source) / "report.json"),
        "source_status": source_manifest["status"], "source_final_window": source_report["final_window"],
        "training_initialization_seed": 101, "ppo_training_from_frozen_sac": True,
        "sac_updates_enabled": False, "sac_replay_collection_enabled": True,
        "training_scene_seeds": training_scenes, "validation_seeds": list(validation_seeds),
        "development_seeds": list(development_seeds), "reserved_scenes_executed": [],
        "source_hashes": current_sources, "restore_audit": restore_audit,
        "selection": "Fixed PPO budget, last weights only; development checkpoints are monitoring only",
        "scope": "Component-isolation development experiment; not unseen-scene generalization or formal significance",
    }
    write_json(output / "manifest.json", manifest)

    history = []
    initial = paired_evaluation(cfg, agent, validation_seeds, output, "validation", 0, macros=evaluation_macros)
    history.append(initial)
    if not all(item["uav"] for item in initial["crn"].values()):
        raise AssertionError("Initial PPO and centroid deployment trajectories differ")
    for seed in validation_seeds:
        with np.load(output / f"validation_ppo_w0_{seed}_trajectory.npz") as a, np.load(
                output / f"validation_centroid_w0_{seed}_trajectory.npz") as b:
            if not all(np.array_equal(a[key], b[key]) for key in ("raw_action", "executed_route")):
                raise AssertionError("Initial PPO and centroid SAC routing differs")
    for a, b in zip(initial["ppo"], initial["centroid"]):
        for key in ("J", "task_delay_weighted_ms", "queue_total_mbit", "power_w"):
            if not np.isclose(a[key], b[key], atol=1e-9, rtol=0):
                raise AssertionError("Untrained PPO must reproduce dynamic-centroid execution before learning")

    stream = int(cfg["training"]["stream_windows"])
    windows = int(cfg["training"]["joint_windows"])
    next_validation = validation_every
    started = time.perf_counter()
    manifest["status"] = "TRAINING"
    write_json(output / "manifest.json", manifest)

    for stream_index, offset in enumerate(range(0, windows, stream)):
        count = min(stream, windows - offset)
        scene = training_scenes[stream_index]
        replay_before = agent.replay.memory_counter
        blocked_before = proxy.blocked_update_calls
        summary, rows, _, _, mdp = rollout(
            cfg, proxy, scene, count * cfg["training"]["window_macros"],
            macro_enabled=True, train=True, window_offset=offset,
            progress_offset=offset / max(windows, 1), complete_tail=True,
        )
        end = offset + count
        macros = count * cfg["training"]["window_macros"]
        replay_added = agent.replay.memory_counter - replay_before
        if (replay_added != macros * mdp.env.service_resource_frames
                or summary["boundary_transitions"] != macros or summary["final_pending_dropped"] != 0
                or not np.isclose(summary["boundary_discount_min"],
                                  cfg["training"]["gamma_per_resource_frame"] ** (1 + mdp.env.deployment_resource_frames))
                or proxy.blocked_update_calls - blocked_before != macros * mdp.env.service_resource_frames // cfg["training"]["sac_update_frames"]
                or proxy.consumed_macro_transitions + len(agent.macro_rollout) != end * cfg["training"]["window_macros"]):
            raise AssertionError("Replay/boundary/PPO transition conservation failed")
        # Final residual batch must be consumed BEFORE final validation/snapshots.
        if end == windows:
            proxy.update_ppo(1.0, force=True)
            if agent.macro_rollout or proxy.consumed_macro_transitions != windows * cfg["training"]["window_macros"]:
                raise AssertionError("Final PPO rollout was not consumed exactly once")
        append_csv(output / "training.csv", rows)
        append_csv(output / "boundary_audit.csv", [{
            "start_window": offset, "end_window": end, "stream_seed": scene,
            "summary_time_s": summary["time_s"],
            "boundary_transitions": summary["boundary_transitions"],
            "pending_dropped": summary["final_pending_dropped"],
            "boundary_discount": summary["boundary_discount_min"],
            "replay_added": replay_added,
            "blocked_sac_update_calls": proxy.blocked_update_calls - blocked_before,
            "ppo_updates_total": agent.ppo_updates,
            "ppo_transitions_consumed": proxy.consumed_macro_transitions,
            "ppo_transitions_pending": len(agent.macro_rollout),
        }])
        assert_sac_frozen(agent, cfg, frozen_reference)
        save_boundary_checkpoint(proxy, cfg, output / "boundary_latest.pt", end, scene + 1)

        if end >= next_validation or end == windows:
            history.append(paired_evaluation(cfg, agent, validation_seeds, output, "validation", end, macros=evaluation_macros))
            save_boundary_checkpoint(proxy, cfg, output / "checkpoints" / f"window{end}.pt", end, scene + 1)
            next_validation = (end // validation_every + 1) * validation_every
        write_json(output / "progress.json", {
            "status": "TRAINING", "completed_windows": end, "target_windows": windows,
            "ppo_updates": agent.ppo_updates, "blocked_sac_update_calls": proxy.blocked_update_calls,
            "replay_count": agent.replay.memory_counter,
        })

    frozen_checks = assert_sac_frozen(agent, cfg, frozen_reference)
    final_checkpoint = output / "final.pt"
    save_boundary_checkpoint(proxy, cfg, final_checkpoint, windows, training_scenes[-1] + 1)
    final = paired_evaluation(cfg, agent, development_seeds, output, "development_final", windows, macros=evaluation_macros)

    for name, path in (("checkpoint", source), ("manifest.json", _source_run(source) / "manifest.json"),
                       ("report.json", _source_run(source) / "report.json")):
        if not path.is_file() or file_hash(path) != SOURCE_DIGESTS[name]:
            raise AssertionError(f"Frozen SAC source changed during PPO experiment: {name}")
    if any(not (ROOT / relative).is_file() or file_hash(ROOT / relative) != digest
           for relative, digest in current_sources.items()):
        raise AssertionError("Source/config/test file changed during PPO experiment")

    report = {
        "status": "SMOKE_ONLY" if smoke else "PPO_FROZEN_SAC_DEVELOPMENT_COMPLETE",
        "stage_transition": STAGE_TRANSITION, "resume_supported": False,
        "source_seed": 101, "source_sac_window": 2000,
        "ppo_windows": windows, "ppo_updates": agent.ppo_updates,
        "blocked_sac_update_calls": proxy.blocked_update_calls,
        "sac_updates_at_entry": frozen_reference["sac_updates"],
        "sac_updates_at_exit": sac_learning_state(agent, cfg)["sac_updates"],
        "ppo_update_events": proxy.ppo_update_events,
        "ppo_transitions_consumed": proxy.consumed_macro_transitions,
        "ppo_transitions_pending": len(agent.macro_rollout),
        "sac_frozen_checks": frozen_checks,
        "sac_actor_max_delta": parameter_delta(frozen_reference["sac_actor"], agent.sac_actor.action_net),
        "macro_mean_max_delta": parameter_delta(initial_macro, agent.macro_actor, ("log_std",)),
        "ppo_gradients": agent.ppo_gradient_steps,
        "replay_at_entry": initial_replay, "replay_at_exit": agent.replay.memory_counter,
        "validation_history": history, "development_final": final,
        "reserved_scenes_executed": [],
        "interpretation_boundary": (
            "This run tests whether learned macro trajectory control adds value above the same frozen SAC. "
            "It does not test joint fine-tuning, unseen scenes, global optimality or infinite-horizon stability."
        ),
    }
    write_json(output / "report.json", report)
    manifest.update(status=report["status"], elapsed_seconds=time.perf_counter() - started,
                    final_checkpoint=str(final_checkpoint), final_checkpoint_sha256=file_hash(final_checkpoint))
    write_json(output / "manifest.json", manifest)
    write_json(output / "progress.json", {"status": report["status"], "completed_windows": windows,
                                           "target_windows": windows, "ppo_updates": agent.ppo_updates})
    return report
