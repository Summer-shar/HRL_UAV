"""Audited stage-preserving continuation at completed rollout boundaries."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch


def validate_resume_config(saved, requested, completed, stage="pretrain", *, allowed_training_overrides=()):
    if stage not in ("pretrain", "joint"):
        raise ValueError(f"Unsupported continuation stage: {stage}")
    restored = copy.deepcopy(requested)
    allowed = set(allowed_training_overrides)
    critic_key = "sac_critic_steps_per_update"
    if allowed - {critic_key}:
        raise ValueError("Unsupported experimental continuation override")
    changes = {}
    old_fits = saved["training"].get(critic_key, 1)
    new_fits = requested["training"].get(critic_key, 1)
    if type(new_fits) is not int or new_fits < 1:
        raise ValueError("sac_critic_steps_per_update must be a positive integer")
    if new_fits != old_fits and critic_key in allowed:
        changes[critic_key] = {"from": old_fits, "to": new_fits}
    if new_fits == old_fits or critic_key in allowed:
        if critic_key in saved["training"]:
            restored["training"][critic_key] = old_fits
        else:
            restored["training"].pop(critic_key, None)
    budget = f"{stage}_windows"
    restored["training"][budget] = saved["training"][budget]
    restored["qualification"]["max_stage_seconds"] = saved["qualification"]["max_stage_seconds"]
    if stage == "joint":
        old_decay = saved["training"].get("ppo_std_decay_windows", saved["training"]["joint_windows"])
        new_decay = requested["training"].get("ppo_std_decay_windows", requested["training"]["joint_windows"])
        if old_decay != new_decay:
            raise ValueError("Joint continuation must preserve the original PPO exploration decay horizon")
        if "ppo_std_decay_windows" not in saved["training"]:
            restored["training"].pop("ppo_std_decay_windows", None)
        for key in ("joint_evaluation_mode", "checkpoint_every_windows"):
            if key in saved["qualification"]:
                restored["qualification"][key] = saved["qualification"][key]
            else:
                restored["qualification"].pop(key, None)
    if restored != saved:
        raise ValueError("Continuation may change only the window budget, wall-clock cap, joint monitoring cadence and explicitly authorized Critic fit count")
    if not 0 < completed < requested["training"][budget]:
        raise ValueError("Continuation needs a larger total window budget")
    return changes


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("Saved and current CUDA RNG device counts differ")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def load_source(path, config, *, allowed_training_overrides=()):
    path = Path(path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stage = checkpoint["stage"]
    if stage not in ("pretrain", "joint"):
        raise ValueError(f"Unsupported continuation stage: {stage}")
    if stage == "pretrain" and checkpoint["updates"]["ppo"] != 0:
        raise ValueError("Micro-pretraining checkpoint already contains PPO updates")
    run_state = checkpoint.get("run_state")
    joint_history = {}
    source_kind = "completed_run"
    if stage == "joint":
        if run_state is None or "macro_rollout" not in checkpoint.get("training_state", {}):
            raise ValueError("Joint continuation requires full RNG and pending PPO rollout state")
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        if manifest["teacher"] or manifest["config"] != checkpoint["config"] or manifest["pretrain_only"]:
            raise ValueError("Joint checkpoint provenance does not match its manifest")
        if path.name == "joint_latest.pt":
            source_kind = "rollout_boundary"
            admission = json.loads((path.parent / "pretrain_admission.json").read_text(encoding="utf-8"))
            decision = json.loads((path.parent / "joint_start_decision.json").read_text(encoding="utf-8"))
            if (not decision["allowed"] or not run_state["at_rollout_boundary"]
                    or not manifest["start_window"] < run_state["completed_windows"] <= checkpoint["config"]["training"]["joint_windows"]):
                raise ValueError("Joint checkpoint is not an admitted saved rollout boundary")
            interruption = path.parent / "interruption.json"
            if interruption.exists():
                recorded = json.loads(interruption.read_text(encoding="utf-8"))
                if recorded["checkpoint_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError("Interrupted checkpoint hash changed")
        else:
            report = json.loads((path.parent / "report.json").read_text(encoding="utf-8"))
            audit = report["stage_audits"][-1]
            if (path.name != "final.pt" or report["status"] != "JOINT_PILOT_COMPLETE_NOT_FORMAL"
                    or not report["joint_training_started"] or report["pretrain_only"]
                    or audit["stage"] != "joint" or audit["completed_windows"] != run_state["completed_windows"]
                    or audit["replay_at_stage_exit"] != checkpoint["replay"]["memory_counter"]
                    or any(report["updates_total"][key] != checkpoint["updates"][key]
                           for key in ("sac_actor", "sac_critic", "ppo"))):
                raise ValueError("Joint checkpoint provenance does not match the completed run")
            admission, decision = report["pretrain_admission"], report["joint_start_decision"]
        joint_history = {"original_pretrain_admission": admission,
                         "original_joint_start_decision": decision,
                         "ppo_std_decay_windows": checkpoint["config"]["training"].get(
                             "ppo_std_decay_windows", checkpoint["config"]["training"]["joint_windows"])}
    if run_state is None:
        report = json.loads((path.parent / "report.json").read_text(encoding="utf-8"))
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        if (path.name != "micro_pretrained.pt" or not report["pretrain_only"]
                or report["status"] != "MICRO_ONLY_DIAGNOSTIC_COMPLETE"
                or len(report["stage_audits"]) != 1 or manifest["teacher"]
                or manifest["config"] != checkpoint["config"]):
            raise ValueError("Legacy checkpoint provenance is not a completed pure micro run")
        audit = report["stage_audits"][0]
        if (audit["replay_at_stage_entry"] != 0
                or audit["critic_updates"] != checkpoint["updates"]["sac_critic"]
                or audit["actor_updates"] != checkpoint["updates"]["sac_actor"]):
            raise ValueError("Legacy update counters do not match the archived run")
        completed = int(audit["completed_windows"])
        stream = config["training"]["stream_windows"]
        if completed % stream:
            raise ValueError("Legacy checkpoint must end at a full rollout boundary")
        counter = int(audit["replay_at_stage_exit"])
        next_seed = config["qualification"]["training_scene_seed_base"] + completed // stream
    else:
        if not run_state["at_rollout_boundary"]:
            raise ValueError("Mid-rollout continuation requires an environment snapshot")
        completed = int(run_state["completed_windows"])
        counter = int(checkpoint["replay"]["memory_counter"])
        next_seed = int(run_state["next_training_scene_seed"])
    overrides = validate_resume_config(checkpoint["config"], config, completed, stage,
                                       allowed_training_overrides=allowed_training_overrides)
    if not checkpoint.get("replay") or counter <= 0:
        raise ValueError("Continuation requires the nonempty original replay")
    mode = "restored_global_rng" if run_state else "legacy_global_rng_reseeded"
    metadata = {"checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_kind": source_kind,
                "stage": stage, "start_window": completed, "next_training_scene_seed": next_seed,
                "replay_counter": counter, "rng_mode": mode,
                "legacy_rng_seed": None if run_state else config["training"]["seed"] + 1000000 + completed,
                "scope": "Continuation of learned state at saved rollout boundaries; not a claim of equivalence to a preplanned longer run",
                "authorized_training_overrides": overrides,
                **joint_history}
    return checkpoint, metadata


def state_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(state_equal(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)):
        return (isinstance(right, type(left)) and len(left) == len(right)
                and all(state_equal(a, b) for a, b in zip(left, right)))
    return left == right


def restore_agent(agent, checkpoint, metadata):
    if agent.micro_algorithm != checkpoint["config"]["training"].get("micro_algorithm", "sac"):
        raise ValueError("SAC and TQC checkpoints are not interchangeable")
    if (agent.micro_state_dim != checkpoint["micro_state_dim"]
            or agent.macro_state_dim != checkpoint["macro_state_dim"]
            or agent.observation_spec != checkpoint["observations"]):
        raise ValueError("Checkpoint observation schema does not match this environment")
    networks = {"sac_actor": agent.sac_actor.action_net, "sac_critic": agent.sac_critic.critic_v,
                "sac_target": agent.sac_critic.target_critic_v,
                "macro_actor": agent.macro_actor, "macro_critic": agent.macro_critic}
    for name, network in networks.items():
        network.load_state_dict(checkpoint[name], strict=True)
    with torch.no_grad():
        agent.entropy.log_alpha.copy_(checkpoint["log_alpha"].to(agent.entropy.log_alpha.device))
    agent.entropy.alpha = agent.entropy.log_alpha.exp()
    optimizers = [agent.sac_actor.optimizer, agent.sac_critic.optimizer, agent.entropy.optimizer,
                  agent.ppo_actor_optimizer, agent.ppo_critic_optimizer]
    if len(checkpoint["optimizers"]) != len(optimizers):
        raise ValueError("Checkpoint is missing optimizer states")
    for optimizer, state in zip(optimizers, checkpoint["optimizers"]):
        optimizer.load_state_dict(state)
    agent.sac_updates = int(checkpoint["updates"]["sac_critic"])
    agent.sac_critic_gradient_steps = int(checkpoint["updates"].get("sac_critic_gradients", agent.sac_updates))
    agent.sac_target_updates = int(checkpoint["updates"].get("sac_target", agent.sac_updates))
    agent.sac_actor_updates = int(checkpoint["updates"]["sac_actor"])
    agent.ppo_updates = int(checkpoint["updates"]["ppo"])
    agent.ppo_gradient_steps = int(checkpoint["updates"]["ppo_gradients"])
    agent.stage = checkpoint["stage"]
    training = checkpoint.get("training_state", {})
    agent.stage_initial_updates = int(training.get("stage_initial_updates", 0))
    agent.last_sac_stats = copy.deepcopy(training.get("last_sac_stats", {}))
    agent.last_ppo_stats = copy.deepcopy(training.get("last_ppo_stats", {}))
    agent.macro_rollout = copy.deepcopy(training.get("macro_rollout", []))
    agent.replay.load_state_dict(checkpoint["replay"], memory_counter=metadata["replay_counter"])
    checks = {name: state_equal(network.state_dict(), checkpoint[name]) for name, network in networks.items()}
    checks["log_alpha"] = state_equal(agent.entropy.log_alpha.detach(), checkpoint["log_alpha"])
    checks["optimizers"] = all(state_equal(o.state_dict(), s) for o, s in zip(optimizers, checkpoint["optimizers"]))
    replay = agent.replay.state_dict()
    checks["replay_transitions"] = state_equal(replay["transitions"], checkpoint["replay"]["transitions"])
    checks["replay_rng"] = state_equal(replay["rng_state"], checkpoint["replay"]["rng_state"])
    checks["pending_ppo_rollout"] = state_equal(agent.macro_rollout, training.get("macro_rollout", []))
    checks["critic_gradient_counter"] = agent.sac_critic_gradient_steps == checkpoint["updates"].get("sac_critic_gradients", checkpoint["updates"]["sac_critic"])
    checks["target_update_counter"] = agent.sac_target_updates == checkpoint["updates"].get("sac_target", checkpoint["updates"]["sac_critic"])
    if not all(checks.values()):
        raise AssertionError(f"Restoration differs from saved state: {checks}")
    if checkpoint.get("run_state"):
        restore_rng(checkpoint["run_state"]["rng"])
        checks["global_rng"] = state_equal(capture_rng(), checkpoint["run_state"]["rng"])
        if not checks["global_rng"]:
            raise AssertionError("Global RNG restoration differs from the saved state")
    else:
        seed = metadata["legacy_rng_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    rates = [[group["lr"] for group in optimizer.param_groups] for optimizer in optimizers]
    if any(any(rate != expected for rate in groups)
           for groups, expected in zip(rates[:3], agent.configured_sac_rates())):
        raise AssertionError("Restored optimizer learning rates differ from frozen config")
    actor_warmup = agent.sac_actor_warmup()
    return {**metadata, "exact_restore_checks": checks, "optimizer_learning_rates": rates,
            "updates_at_entry": dict(checkpoint["updates"]),
            "retained_transitions": len(replay["transitions"]),
            "stage_initial_updates": agent.stage_initial_updates,
            "ppo_exploration_std": agent.macro_actor.log_std.detach().exp().cpu().tolist(),
            "next_update_actor_due": bool(
                agent.sac_updates + 1 - agent.stage_initial_updates > actor_warmup
                and (agent.sac_updates + 1) % agent.cfg["sac_policy_delay"] == 0)}


def verify_initial_evaluation(rows, metadata):
    path = Path(metadata["checkpoint"]).parent / "validation.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        archived = {int(row["seed"]): row for row in csv.DictReader(handle)
                    if row["stage"] == metadata.get("stage", "pretrain") and int(row["window"]) == metadata["start_window"]}
    if set(archived) != {int(row["seed"]) for row in rows}:
        raise ValueError("Source checkpoint has no matching validation rows")
    keys = ("task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms",
            "task_completion_rate", "queue_total_mbit", "J", "power_w", "late_queue_slope_mbit_s")
    deltas = {str(row["seed"]): {key: abs(row[key] - float(archived[int(row["seed"])][key]))
                               for key in keys} for row in rows}
    if any(not np.isfinite(value) or value > 1e-9 for values in deltas.values() for value in values.values()):
        raise AssertionError(f"Restored policy evaluation did not reproduce source: {deltas}")
    return deltas


def save_boundary_checkpoint(agent, config, path, completed, next_seed):
    state = agent.checkpoint(config)
    state["run_state"] = {"completed_windows": int(completed), "at_rollout_boundary": True,
                          "next_training_scene_seed": int(next_seed), "rng": capture_rng()}
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
