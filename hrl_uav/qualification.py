"""Explicit joint-stage scheduling and read-only quality monitoring."""

import copy
import csv
import hashlib
import random
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .bandwidth import MeteredMDP, UniformRouting, summarize_meter
from .experiment import append_csv, isolated_rng, parameter_delta, rollout, snapshot, write_json
from .learning import DirectHRL
from .mdp import QueueMDP
from .resumption import capture_rng, load_source, restore_agent, save_boundary_checkpoint, state_equal, verify_initial_evaluation


def metered_evaluate(config, agent, seeds, macros, output, label, macro_enabled=False):
    readings = []
    with isolated_rng(), patch("hrl_uav.experiment.QueueMDP", MeteredMDP):
        for seed in seeds:
            summary, rows, trajectory, routes, mdp = rollout(
                config, agent, seed, macros, macro_enabled=macro_enabled, record=True,
            )
            tail = rows[len(rows)*2//3:]
            slope = float(np.polyfit([r["time_s"] for r in tail],
                                    [r["queue_total_mbit"] for r in tail], 1)[0])
            summary.update(summarize_meter(mdp.readings, summary["time_s"]))
            summary["late_queue_slope_mbit_s"] = slope
            readings.append({"policy": label, "seed": seed, **summary})
            append_csv(output/f"{label}_{seed}_queues.csv", rows)
            append_csv(output/f"{label}_{seed}_uavs.csv", mdp.uav_readings)
            uav, users, centroid, macro_action = map(np.asarray, zip(*trajectory))
            raw_action, executed_route = map(np.asarray, zip(*routes))
            np.savez_compressed(output/f"{label}_{seed}_trajectory.npz", uav=uav,
                                users=users, centroid=centroid, macro_action=macro_action,
                                raw_action=raw_action, executed_route=executed_route)
    return readings


def means(rows):
    keys = ("task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms",
            "task_completion_rate", "queue_total_mbit", "mean_queue_compute_mbit",
            "mean_queue_transfer_mbit", "J", "power_w", "late_queue_slope_mbit_s")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def pretraining_verdict(history, baseline, rules):
    """Engineering tolerances, not a hypothesis test or infinite-horizon proof."""
    baseline_by_seed = {int(row["seed"]): row for row in baseline}
    recent = history[-rules["tail_checkpoints"]:]
    checks = {"enough_checkpoints": len(recent) == rules["tail_checkpoints"]}
    expected_seeds = set(baseline_by_seed)
    checks["paired_seeds"] = all({int(r["seed"]) for r in item["rows"]} == expected_seeds
                                and len(item["rows"]) == len(expected_seeds) for item in recent)
    if not recent or not baseline or not checks["paired_seeds"]:
        return {"passed": False, "checks": checks, "reason": "Incomplete validation pairs"}
    readings = [row for item in recent for row in item["rows"]]
    keys = ("task_delay_weighted_ms", "J", "task_completion_rate", "queue_total_mbit",
            "mean_queue_compute_mbit", "mean_queue_transfer_mbit", "mean_queue_upload_mbit",
            "late_queue_slope_mbit_s")
    checks["finite_readings"] = bool(all(np.isfinite(row[k]) for row in readings+baseline for k in keys))
    for key, tolerance, name in (
        ("task_delay_weighted_ms", "max_relative_delay_regression", "delay_quality"),
        ("J", "max_relative_cost_regression", "objective_quality"),
    ):
        checks[name] = all(row[key] <= baseline_by_seed[int(row["seed"])][key]*(1+rules[tolerance])
                           for row in readings)
    checks["completion"] = all(row["task_completion_rate"] >= rules["min_completion_rate"] for row in readings)
    checks["terminal_queue"] = all(row["queue_total_mbit"] <= baseline_by_seed[int(row["seed"])]["queue_total_mbit"]
                                    + rules["max_queue_excess_mbit"] for row in readings)
    queue_keys = ("mean_queue_compute_mbit", "mean_queue_transfer_mbit", "mean_queue_upload_mbit")
    checks["mean_queue"] = all(sum(row[k] for k in queue_keys) <=
                               sum(baseline_by_seed[int(row["seed"])][k] for k in queue_keys)
                               + rules["max_queue_excess_mbit"] for row in readings)
    checks["late_queue_growth"] = all(row["late_queue_slope_mbit_s"] <= rules["max_late_queue_slope_mbit_s"]
                                      for row in readings)
    ranges = {}
    for seed in sorted(expected_seeds):
        for key in ("task_delay_weighted_ms", "J"):
            values = [next(r[key] for r in item["rows"] if int(r["seed"]) == seed) for item in recent]
            ranges[f"{seed}_{key}"] = float(np.ptp(values)/max(abs(np.mean(values)), 1e-12))
    checks["validation_stability"] = all(value <= rules["max_relative_validation_range"] for value in ranges.values())
    return {"passed": bool(all(checks.values())), "checks": checks, "relative_ranges": ranges,
            "windows": [item["window"] for item in recent],
            "scope": "Validation-only engineering admission; not formal significance or stability proof"}


def joint_start_decision(admission, policy="quality_gate", smoke=False):
    if policy not in ("quality_gate", "fixed_budget"):
        raise ValueError(f"Unknown joint start policy: {policy}")
    checks = admission.get("checks", {})
    valid = all(checks.get(key, False) for key in
                ("enough_checkpoints", "paired_seeds", "finite_readings"))
    allowed = smoke or (valid if policy == "fixed_budget" else admission["passed"])
    return {"policy": policy, "allowed": bool(allowed),
            "quality_verdict_passed": admission["passed"], "validation_integrity": bool(valid),
            "smoke_override": bool(smoke),
            "scope": ("Fixed pretraining budget; quality verdict remains descriptive and is not overwritten"
                      if policy == "fixed_budget" else "Joint training requires the original quality gate")}


def verify_crn(output, seeds, left, right):
    results = {}
    for seed in seeds:
        with np.load(output/f"{left}_{seed}_trajectory.npz") as a, np.load(output/f"{right}_{seed}_trajectory.npz") as b:
            equal = {key: bool(np.array_equal(a[key], b[key])) for key in ("users", "uav")}
        arrivals = []
        for label in (left, right):
            with (output/f"{label}_{seed}_queues.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
                arrivals.append([float(row["task_arrived_count"]) for row in rows])
        equal["arrivals"] = arrivals[0] == arrivals[1]
        results[str(seed)] = equal
    return results


def training_stage_seed(config, stage):
    rules, cfg = config["qualification"], config["training"]
    schedule = rules.get("training_scene_schedule", "legacy_stage_offset")
    if schedule not in ("continuous", "legacy_stage_offset"):
        raise ValueError(f"Unknown training scene schedule: {schedule}")
    if stage not in ("pretrain", "joint"):
        raise ValueError(f"Unknown training stage: {stage}")
    offset = 0
    if stage == "joint":
        offset = ((cfg["pretrain_windows"] + cfg["stream_windows"] - 1) // cfg["stream_windows"]
                  if schedule == "continuous" else 10000)
    return rules["training_scene_seed_base"] + offset


def run_mainline(config, output, smoke=False, pretrain_only=False, resume_from=None,
                 resume_training_overrides=()):
    config = copy.deepcopy(config)
    if resume_from is not None and smoke:
        raise ValueError("Continuation is unavailable in smoke mode")
    output = Path(output)
    rules, cfg = config["qualification"], config["training"]
    joint_evaluation_mode = rules.get("joint_evaluation_mode", "periodic")
    checkpoint_every = rules.get("checkpoint_every_windows", rules["validation_every_windows"])
    if joint_evaluation_mode not in ("periodic", "final_only"):
        raise ValueError("joint_evaluation_mode must be periodic or final_only")
    if not isinstance(checkpoint_every, int) or isinstance(checkpoint_every, bool) or checkpoint_every <= 0:
        raise ValueError("checkpoint_every_windows must be a positive integer")
    joint_policy = cfg.get("joint_start_policy", "quality_gate")
    if joint_policy not in ("quality_gate", "fixed_budget"):
        raise ValueError(f"Unknown joint start policy: {joint_policy}")
    config["validation_seeds"] = rules["validation_seeds"]
    config["test_seeds"] = rules["test_seeds"]
    if smoke:
        from .experiment import smoke_config
        config = smoke_config(config)
        rules, cfg = config["qualification"], config["training"]
        rules["validation_macros"] = 6
        checkpoint_every = rules.get("checkpoint_every_windows", rules["validation_every_windows"])
    if set(config["validation_seeds"]) & set(config["test_seeds"]):
        raise ValueError("Validation and test seeds must be disjoint")
    stages = ("pretrain",) if pretrain_only else ("pretrain", "joint")
    if resume_training_overrides and resume_from is None:
        raise ValueError("Experimental continuation overrides require a source checkpoint")
    saved_state, resume = (load_source(resume_from, config, allowed_training_overrides=resume_training_overrides)
                           if resume_from is not None else (None, None))
    resume_stage = resume["stage"] if resume else None
    if resume:
        if pretrain_only != (resume_stage == "pretrain"):
            raise ValueError("Use --pretrain-only for micro continuation, and omit it for joint continuation")
        stages = (resume_stage,)
    start_window = resume["start_window"] if resume else 0
    monitoring = {"joint_evaluation_mode": joint_evaluation_mode,
                  "checkpoint_every_windows": checkpoint_every,
                  "initial_resume_evaluation": not (resume_stage == "joint" and joint_evaluation_mode == "final_only"),
                  "final_evaluation_unchanged": True}
    training_seeds = {
        training_stage_seed(config, stage) + i
        for stage in stages
        for i in range((cfg[f"{stage}_windows"]+cfg["stream_windows"]-1)//cfg["stream_windows"])
    }
    if resume:
        training_seeds = {resume["next_training_scene_seed"] + i for i in range(
            (cfg[f"{resume_stage}_windows"] - start_window + cfg["stream_windows"] - 1) // cfg["stream_windows"])}
    if training_seeds & (set(config["validation_seeds"]) | set(config["test_seeds"])):
        raise ValueError("Training scenes overlap validation/test scenes")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    env = QueueMDP(config)
    env.reset(cfg["seed"], cfg["window_macros"])
    agent = DirectHRL.from_mdp(env, config)
    resume_audit = restore_agent(agent, saved_state, resume) if resume else None
    del saved_state
    root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ("hrl_uav", "scripts", "configs")
               for p in (root/folder).rglob("*") if p.is_file() and p.suffix in (".py", ".json")}
    write_json(output/"manifest.json", {
        "config": config, "source_hashes": sources, "teacher": False,
        "micro_algorithm": agent.micro_algorithm,
        "old_checkpoint": resume["checkpoint"] if resume else None,
        "method": (f"continued native direct HRL {resume_stage} stage" if resume else
                   f"fresh direct {agent.micro_algorithm.upper()}, then {joint_policy} PPO+{agent.micro_algorithm.upper()} synchronous updates"),
        "joint_start_policy": joint_policy,
        "sac_stage_policy": agent.sac_stage_policy,
        "training_scene_schedule": rules.get("training_scene_schedule", "legacy_stage_offset"),
        "resume": resume, "start_window": start_window, "resume_stage": resume_stage,
        "state_dim": agent.micro_state_dim, "observation_slices": env.micro_feature_slices,
        "micro_state_dim": agent.micro_state_dim, "macro_state_dim": agent.macro_state_dim,
        "observations": agent.observation_spec,
        "training_seed_count": 1, "training_scene_seeds": sorted(training_seeds),
        "scope": "Engineering pilot, not a formal multi-training-seed result",
        "uniform_scope": "Evaluation-only diagnostic; no replay insertion, actor cloning or action fallback",
        "selection": "Fixed budget, last weights only; joint start follows the recorded policy, tests never select weights",
        "evaluation_scope": "Repeated development scenes, not an untouched formal test set",
        "smoke_bypasses_quality_gate": smoke,
        "pretrain_only": pretrain_only,
        "monitoring": monitoring,
    })
    started = time.perf_counter()
    initial_actor = snapshot(agent.sac_actor.action_net)
    initial_macro = snapshot(agent.macro_actor)
    replay_at_initial_evaluation = agent.replay.memory_counter
    initial_stage = resume_stage or "pretrain"
    baseline, initial = [], []
    if not monitoring["initial_resume_evaluation"]:
        resume_audit["initial_validation"] = {
            "status": "NOT_RUN_FINAL_ONLY_MONITORING",
            "restoration_basis": "Exact network, optimizer, replay, pending PPO rollout and RNG restoration",
        }
        write_json(output/"resume_audit.json", resume_audit)
        print(f"[restored {initial_stage}] exact state restored; evaluation deferred to final weights", flush=True)
    else:
        baseline = metered_evaluate(config, UniformRouting(), config["validation_seeds"],
                                    rules["validation_macros"], output, "uniform_validation")
        append_csv(output/"baseline_validation.csv", baseline)
        initial = metered_evaluate(config, agent, config["validation_seeds"],
                                   rules["validation_macros"], output, f"validation_{initial_stage}_{start_window}",
                                   macro_enabled=initial_stage == "joint")
        append_csv(output/"validation.csv", [{"stage": initial_stage, "window": start_window, **row} for row in initial])
        if resume:
            resume_audit["initial_validation_absolute_differences"] = verify_initial_evaluation(initial, resume)
            write_json(output/"resume_audit.json", resume_audit)
        else:
            for a, b in zip(initial, baseline):
                for key in ("task_delay_weighted_ms", "J", "queue_total_mbit"):
                    if not np.isclose(a[key], b[key], atol=1e-9, rtol=0):
                        raise AssertionError("Fresh deterministic actor must match uniform routing")
        print(f"[{'restored' if resume else 'fresh'} baseline] {means(initial)}", flush=True)
    if agent.replay.memory_counter != replay_at_initial_evaluation:
        raise AssertionError("Read-only baseline evaluation populated replay")
    histories, stage_audits = {"pretrain": [], "joint": []}, []
    admission, joint_started, budget_stop, joint_decision = None, False, False, None
    if resume_stage == "joint":
        admission = resume["original_pretrain_admission"]
        write_json(output/"pretrain_admission.json", admission)
    for stage in stages:
        continuing = bool(resume and stage == resume_stage)
        if stage == "joint":
            joint_decision = ({**resume["original_joint_start_decision"],
                               "continued_existing_joint_stage": True, "resume_start_window": start_window}
                              if continuing else joint_start_decision(admission, joint_policy, smoke))
            write_json(output/"joint_start_decision.json", joint_decision)
            if not joint_decision["allowed"]:
                break
            for network in (agent.sac_actor.action_net, agent.sac_critic.critic_v,
                            agent.sac_critic.target_critic_v, agent.macro_actor, agent.macro_critic):
                if not all(torch.isfinite(p).all().item() for p in network.parameters()):
                    raise FloatingPointError("Non-finite network at the joint-stage boundary")
            if not torch.isfinite(agent.entropy.log_alpha).all().item():
                raise FloatingPointError("Non-finite entropy temperature at the joint-stage boundary")
            joint_started = True
        actor_before, macro_before = snapshot(agent.sac_actor.action_net), snapshot(agent.macro_actor)
        before_updates = (agent.sac_updates, agent.sac_actor_updates, agent.ppo_updates)
        before_critic_gradients, before_target_updates = agent.sac_critic_gradient_steps, agent.sac_target_updates
        replay_before, replay_id = agent.replay.memory_counter, id(agent.replay)
        stage_state = (copy.deepcopy(agent.checkpoint(config, include_replay=False))
                       if stage == "joint" and not continuing else None)
        stage_rng = capture_rng() if stage_state is not None else None
        replay_rng = copy.deepcopy(agent.replay.rng.bit_generator.state) if stage_state is not None else None
        warmup_before = agent.sac_actor_warmup()
        if not continuing:
            agent.set_stage(stage)
        if agent.replay.memory_counter != replay_before or id(agent.replay) != replay_id:
            raise AssertionError("Stage transition reset the current run's replay")
        if continuing:
            print(f"[continue {stage}] window={start_window}; replay={replay_before}; "
                  f"SAC={agent.sac_actor_updates}/{agent.sac_updates}; PPO={agent.ppo_updates}", flush=True)
        if stage_state is not None:
            after = agent.checkpoint(config, include_replay=False)
            preserved = {key: state_equal(stage_state[key], after[key]) for key in
                         ("sac_actor", "sac_critic", "sac_target", "log_alpha", "macro_actor", "macro_critic")}
            preserved["optimizer_moments"] = all(state_equal(a["state"], b["state"])
                                                 for a, b in zip(stage_state["optimizers"], after["optimizers"]))
            sac_preserved = {
                "full_optimizers_including_learning_rates": state_equal(stage_state["optimizers"][:3], after["optimizers"][:3]),
                "lifetime_actor_warmup_clock": state_equal(stage_state["training_state"]["stage_initial_updates"],
                                                           after["training_state"]["stage_initial_updates"]),
                "update_counters": state_equal(stage_state["updates"], after["updates"]),
                "actor_warmup_budget": warmup_before == agent.sac_actor_warmup(),
                "global_rng_at_boundary": state_equal(stage_rng, capture_rng()),
                "replay_sampling_rng": state_equal(replay_rng, agent.replay.rng.bit_generator.state),
            }
            if agent.sac_stage_policy == "preserve":
                preserved.update(sac_preserved)
            if not all(preserved.values()):
                raise AssertionError(f"Stage transition changed learned state: {preserved}")
            write_json(output/"joint_transition_audit.json", {
                "after_pretrain_windows": cfg["pretrain_windows"], "preserved": preserved,
                "replay_counter": replay_before, "updates_at_entry": after["updates"],
                "sac_stage_policy": agent.sac_stage_policy,
                "sac_preservation_checks": sac_preserved,
                "sac_learning_rates_before": [[group["lr"] for group in opt["param_groups"]]
                                              for opt in stage_state["optimizers"][:3]],
                "sac_learning_rates": [opt["param_groups"][0]["lr"] for opt in after["optimizers"][:3]],
                "joint_actor_warmup_updates": agent.sac_actor_warmup(),
                "actor_warmup_clock_origin": agent.stage_initial_updates,
                "actor_warmup_remaining_critic_updates": max(0, agent.sac_actor_warmup() -
                                                             (agent.sac_updates - agent.stage_initial_updates)),
                "training_scene_schedule": rules.get("training_scene_schedule", "legacy_stage_offset"),
                "joint_first_scene_seed": training_stage_seed(config, "joint"),
                "teacher": False, "old_checkpoint": None,
            })
            print(f"[joint start] after {cfg['pretrain_windows']} pretrain windows; "
                  f"policy={joint_policy}; quality_passed={admission['passed']}; replay={replay_before}", flush=True)
            del stage_state
        stage_start = start_window if continuing else 0
        stage_started, completed = time.perf_counter(), stage_start
        windows = cfg[f"{stage}_windows"]
        next_seed = (resume["next_training_scene_seed"] if continuing else
                     training_stage_seed(config, stage))
        next_validation = (stage_start // rules["validation_every_windows"] + 1) * rules["validation_every_windows"]
        next_checkpoint = (stage_start // checkpoint_every + 1) * checkpoint_every
        periodic_validation = stage != "joint" or joint_evaluation_mode == "periodic"
        write_json(output/"progress.json", {"stage": stage, "completed_windows": completed,
                   "planned_windows": windows, "start_window": stage_start,
                   "joint_training_started": joint_started, "minutes": (time.perf_counter()-started)/60})
        for offset in range(stage_start, windows, cfg["stream_windows"]):
            if time.perf_counter()-stage_started > rules["max_stage_seconds"]:
                budget_stop = True
                break
            count = min(cfg["stream_windows"], windows-offset)
            seed = next_seed
            summary, rows, _, _, _ = rollout(
                config, agent, seed, count*cfg["window_macros"], macro_enabled=stage == "joint",
                train=True, window_offset=offset, progress_offset=offset/max(windows, 1), complete_tail=True,
            )
            append_csv(output/"training.csv", [{**row, "global_window": row["window"] +
                       (cfg["pretrain_windows"] if stage == "joint" else 0)} for row in rows])
            append_csv(output/"boundary_audit.csv", [{"stage": stage, "seed": seed,
                        "pending_dropped": summary["final_pending_dropped"],
                        "boundary_discount": summary["boundary_discount_min"]}])
            completed = offset+count
            next_seed += 1
            if completed == windows and stage == "joint":
                agent.update_ppo(1.0, force=True)
            if periodic_validation and (completed == windows or completed >= next_validation):
                readings = metered_evaluate(config, agent, config["validation_seeds"],
                                            rules["validation_macros"], output,
                                            f"validation_{stage}_{completed}", macro_enabled=stage == "joint")
                histories[stage].append({"window": completed, "rows": readings})
                append_csv(output/"validation.csv", [{"stage": stage, "window": completed, **row} for row in readings])
                print(f"[validation {stage} {completed}] {means(readings)}", flush=True)
                if stage == "joint" and rules.get("paired_joint_validation", False):
                    centroid_label = f"validation_joint_centroid_{completed}"
                    control = metered_evaluate(config, agent, config["validation_seeds"],
                                               rules["validation_macros"], output, centroid_label)
                    append_csv(output/"validation_joint_centroid.csv", [{"window": completed, **row} for row in control])
                    pairs = verify_crn(output, config["validation_seeds"], f"validation_joint_{completed}", centroid_label)
                    if not all(row["users"] and row["arrivals"] for row in pairs.values()):
                        raise AssertionError("Joint/centroid validation exogenous streams diverged")
                    deltas = [{"window": completed, "seed": a["seed"],
                               **{f"ppo_minus_centroid_{key}": a[key]-b[key] for key in
                                  ("task_delay_weighted_ms", "J", "queue_total_mbit", "task_completion_rate")}}
                              for a, b in zip(readings, control)]
                    append_csv(output/"validation_macro_deltas.csv", deltas)
                next_validation = (completed // rules["validation_every_windows"] + 1) * rules["validation_every_windows"]
            if completed == windows or completed >= next_checkpoint:
                latest = "resume_latest.pt" if stage == "pretrain" else "joint_latest.pt"
                save_boundary_checkpoint(agent, config, output/latest, completed, next_seed)
                next_checkpoint = (completed // checkpoint_every + 1) * checkpoint_every
            write_json(output/"progress.json", {"stage": stage, "completed_windows": completed,
                       "planned_windows": windows, "start_window": stage_start,
                       "joint_training_started": joint_started,
                       "minutes": (time.perf_counter()-started)/60})
        stage_audits.append({"stage": stage, "completed_windows": completed,
                             "start_window": stage_start, "trained_windows_this_run": completed-stage_start,
                             "sac_actor_max_delta": parameter_delta(actor_before, agent.sac_actor.action_net),
                             "macro_mean_max_delta": parameter_delta(macro_before, agent.macro_actor, ("log_std",)),
                             "critic_updates": agent.sac_updates-before_updates[0],
                             "critic_gradient_steps": agent.sac_critic_gradient_steps-before_critic_gradients,
                             "target_updates": agent.sac_target_updates-before_target_updates,
                             "actor_updates": agent.sac_actor_updates-before_updates[1],
                             "ppo_updates": agent.ppo_updates-before_updates[2],
                             "replay_at_stage_entry": replay_before, "replay_at_stage_exit": agent.replay.memory_counter})
        if stage == "pretrain" and stage_audits[-1]["macro_mean_max_delta"] != 0:
            raise AssertionError("Macro actor changed during micro-only pretraining")
        checkpoint = "partial.pt" if budget_stop else ("micro_pretrained.pt" if stage == "pretrain" else "final.pt")
        save_boundary_checkpoint(agent, config, output/checkpoint, completed, next_seed)
        if budget_stop:
            break
        if stage == "pretrain":
            admission = pretraining_verdict(histories[stage], baseline, rules)
            write_json(output/"pretrain_admission.json", admission)
            print(f"[pretrain admission] {admission}", flush=True)
    label = "partial" if budget_stop else ("final" if joint_started else "pretrain_final")
    final = metered_evaluate(config, agent, config["test_seeds"], config["test_macros"],
                             output, label, macro_enabled=joint_started)
    uniform = metered_evaluate(config, UniformRouting(), config["test_seeds"], config["test_macros"], output, "uniform_test")
    matched = metered_evaluate(config, agent, config["test_seeds"], config["test_macros"],
                               output, "centroid_same_sac") if joint_started else []
    append_csv(output/"final_evaluation.csv", final+uniform+matched)
    crn = verify_crn(output, config["test_seeds"], label, "uniform_test")
    if not all(v["users"] and v["arrivals"] and (joint_started or v["uav"]) for v in crn.values()):
        raise AssertionError("Paired evaluation exogenous streams diverged")
    matched_crn = verify_crn(output, config["test_seeds"], label, "centroid_same_sac") if matched else {}
    if not all(row["users"] and row["arrivals"] for row in matched_crn.values()):
        raise AssertionError("Matched centroid evaluation exogenous streams diverged")
    unchanged = all(hashlib.sha256((root/p).read_bytes()).hexdigest() == digest for p, digest in sources.items())
    report = {
        "micro_algorithm": agent.micro_algorithm,
        "status": "SMOKE_ONLY" if smoke else ("BUDGET_STOP_INCOMPLETE" if budget_stop else
                   ("MICRO_ONLY_DIAGNOSTIC_COMPLETE" if pretrain_only else
                    ("JOINT_PILOT_COMPLETE_NOT_FORMAL" if joint_started else "MICRO_NOT_QUALIFIED_JOINT_NOT_STARTED"))),
        "minutes": (time.perf_counter()-started)/60, "stage_audits": stage_audits,
        "resume": resume_audit, "start_window": start_window, "resume_stage": resume_stage,
        "updates_total": {"sac_critic": agent.sac_updates,
                          "sac_critic_gradients": agent.sac_critic_gradient_steps,
                          "sac_target": agent.sac_target_updates,
                          "sac_actor": agent.sac_actor_updates, "ppo": agent.ppo_updates},
        "pretrain_admission": admission, "joint_training_started": joint_started,
        "joint_start_decision": joint_decision,
        "pretrain_only": pretrain_only,
        "monitoring": monitoring,
        "final_label": label, "final": final, "uniform": uniform, "matched_centroid": matched,
        "mean": means(final), "uniform_mean": means(uniform),
        "paired_delay_ms": [a["task_delay_weighted_ms"]-b["task_delay_weighted_ms"] for a, b in zip(final, uniform)],
        "paired_macro_delay_ms": [a["task_delay_weighted_ms"]-b["task_delay_weighted_ms"] for a, b in zip(final, matched)],
        "sac_actor_max_delta": parameter_delta(initial_actor, agent.sac_actor.action_net),
        "macro_mean_max_delta": parameter_delta(initial_macro, agent.macro_actor, ("log_std",)),
        "training_seed_count": 1, "evaluation_seed_count": len(config["test_seeds"]),
        "crn": crn, "matched_centroid_crn": matched_crn, "source_hashes_unchanged": unchanged,
        "baseline_scope": "Uniform is untrained diagnostic, matched_centroid is component replay, neither is an independently trained competitor",
        "convergence": "Finite fixed-seed validation and 900s queues; no global optimality or infinite-horizon claim",
    }
    write_json(output/"report.json", report)
    write_json(output/"progress.json", {"stage": "complete", "status": report["status"], "minutes": report["minutes"]})
    return report
