"""Training/evaluation orchestration, with explicit continuation boundaries."""

from contextlib import contextmanager
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .learning import DirectHRL
from .mdp import QueueMDP


@contextmanager
def isolated_rng():
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)


def append_csv(path, rows):
    if not rows:
        return
    exists = Path(path).exists()
    with Path(path).open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def snapshot(network):
    return {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}


def parameter_delta(old, network, exclude=()):
    values = [(v.detach().cpu()-old[k]).abs().max().item() for k, v in network.state_dict().items() if k not in exclude]
    return max(values, default=0.0)


def boundary_reward(last_reward, deployment_rewards, gamma):
    result = float(last_reward)
    for index, reward in enumerate(deployment_rewards, 1):
        result += gamma ** index * reward
    return result, gamma ** (1 + len(deployment_rewards))


def rollout(config, agent, seed, macro_count, macro_enabled=True, train=False,
            window_offset=0, progress_offset=0.0, record=False, complete_tail=False,
            on_window=None):
    mdp = QueueMDP(config)
    mdp.reset(seed, macro_count)
    cfg = config["training"]
    gamma = cfg["gamma_per_resource_frame"]
    pending = None
    boundary_count = 0
    boundary_discounts = []
    rows, trajectory, routes = [], [], []
    frames_seen = 0
    window_cost, window_energy, window_reward = 0.0, 0.0, 0.0
    previous_completed = dict(mdp.env.completed_priority_count)
    previous_delays = dict(mdp.env.completed_priority_delay_sum)
    last_summary = None

    def complete_pending(deployment, next_state):
        nonlocal pending, boundary_count
        if pending is not None:
            state, action, route, reward = pending
            r, discount = boundary_reward(reward, [f["reward"] for f in deployment], gamma)
            agent.store(state, action, route, r, next_state, discount)
            boundary_count += 1
            boundary_discounts.append(discount)
            pending = None

    for macro in range(macro_count):
        state = mdp.macro_observation()
        action, logprob = agent.macro_action(state, macro_enabled, deterministic=not train)
        mdp.start_macro(action)
        cost_before, energy_before = mdp.cost_sum, mdp.total_power_energy
        deployment = [mdp.step_frame(np.zeros(mdp.micro_action_dim), gamma) for _ in range(mdp.env.deployment_resource_frames)]
        rewards = [frame["reward"] for frame in deployment]
        micro_state = mdp.prepare_service()
        if train:
            complete_pending(deployment, micro_state)
        for frame_index in range(mdp.env.service_resource_frames):
            micro_action = agent.sac_actor.choose_action(micro_state, deterministic=not train)
            frame = mdp.step_frame(micro_action, gamma)
            rewards.append(frame["reward"])
            last = frame_index == mdp.env.service_resource_frames - 1
            if record:
                routes.append((micro_action.copy(), frame["route"].copy()))
            if train:
                if last:
                    pending = (micro_state.copy(), micro_action.copy(), frame["route"].copy(), frame["reward"])
                else:
                    next_micro_state = mdp.prepare_service()
                    agent.store(micro_state, micro_action, frame["route"], frame["reward"], next_micro_state, gamma)
                frames_seen += 1
                if frames_seen % cfg["sac_update_frames"] == 0:
                    agent.update_sac()
            if not last:
                micro_state = next_micro_state if train else mdp.prepare_service()
        macro_reward = sum(gamma ** i * r for i, r in enumerate(rewards)) / len(rewards)
        window_reward += macro_reward
        window_cost += mdp.cost_sum - cost_before
        window_energy += mdp.total_power_energy - energy_before
        mdp.advance_macro_clock()
        if train and macro_enabled:
            next_state = mdp.macro_observation()
            agent.macro_rollout.append((state, action, logprob, macro_reward, next_state, float(macro < macro_count - 1)))
            progress = min(1.0, progress_offset + (macro+1) / (cfg["joint_windows"] * cfg["window_macros"]))
            agent.update_ppo(progress)
        if record:
            metrics = mdp.metrics()
            trajectory.append((mdp.env.uav_coords.copy(), mdp.env.user_coords.copy(), mdp.env.weighted_centroids.copy(), action.copy()))
            rows.append({"macro": macro+1, "phase_age": mdp.phase_age, **metrics})
        if (macro + 1) % cfg["window_macros"] == 0 or macro == macro_count-1:
            metrics = mdp.metrics()
            completed = {p: mdp.env.completed_priority_count[p] - previous_completed[p] for p in ("high","low")}
            delays = {p: mdp.env.completed_priority_delay_sum[p] - previous_delays[p] for p in ("high","low")}
            duration_macros = (macro % cfg["window_macros"]) + 1
            metrics["window_delay_ms"] = 1000 * (10*delays["high"]+delays["low"]) / max(10*completed["high"]+completed["low"], 1)
            metrics["window_high_ms"] = 1000*delays["high"]/max(completed["high"],1)
            metrics["window_low_ms"] = 1000*delays["low"]/max(completed["low"],1)
            metrics["window_J"] = window_cost / (duration_macros * mdp.env.micro_steps_per_macro)
            metrics["window_reward"] = window_reward / duration_macros
            metrics["window_power_w"] = window_energy / (duration_macros * mdp.env.macro_dt)
            last_summary = metrics
            if train:
                stage_window = window_offset + (macro+1+cfg["window_macros"]-1)//cfg["window_macros"]
                rows.append({
                    "stage": agent.stage, "window": stage_window, "seed": seed, **metrics,
                    "sac_critic_updates": agent.sac_updates, "sac_actor_updates": agent.sac_actor_updates,
                    "sac_critic_gradient_steps": agent.sac_critic_gradient_steps,
                    "sac_target_updates": agent.sac_target_updates,
                    "ppo_updates": agent.ppo_updates,
                    "critic_loss": agent.last_sac_stats.get("critic_loss",0),
                    "critic_loss_last_fit": agent.last_sac_stats.get("critic_loss_last_fit",0),
                    "alpha": agent.entropy.log_alpha.exp().item(),
                    "target_std": agent.last_sac_stats.get("target_std",0),
                })
                print(f"[{agent.stage} {stage_window}] delay={metrics['window_delay_ms']:.1f}ms low={metrics['window_low_ms']:.1f}ms Q={metrics['queue_total_mbit']:.1f}Mbit SAC={agent.sac_actor_updates}/{agent.sac_updates} PPO={agent.ppo_updates}", flush=True)
                # Observational checkpoint hook; keep the live stream and pending transition.
                if on_window is not None:
                    on_window(dict(rows[-1]), mdp, pending, macro == macro_count - 1)
            previous_completed = dict(mdp.env.completed_priority_count)
            previous_delays = dict(mdp.env.completed_priority_delay_sum)
            window_cost = window_energy = window_reward = 0.0
    summary = dict(last_summary)
    # At an outer sampling truncation, obtain the real next service decision
    # state before resetting. Do not bootstrap into an empty reset state.
    if train and complete_tail:
        action, _ = agent.macro_action(mdp.macro_observation(), macro_enabled)
        mdp.start_macro(action)
        deployment = [mdp.step_frame(np.zeros(mdp.micro_action_dim), gamma) for _ in range(mdp.env.deployment_resource_frames)]
        complete_pending(deployment, mdp.prepare_service())
    summary.update(boundary_transitions=boundary_count, boundary_discount_min=min(boundary_discounts, default=0), final_pending_dropped=int(pending is not None))
    return summary, rows, trajectory, routes, mdp


def evaluate(config, agent, seeds, macros, output, label, macro_enabled=True):
    summaries = []
    with isolated_rng():
        for seed in seeds:
            summary, rows, trajectory, routes, _ = rollout(config, agent, seed, macros, macro_enabled=macro_enabled, record=True)
            summaries.append({"policy": label, "seed": seed, **summary})
            append_csv(output / f"{label}_{seed}_queues.csv", rows)
            uav, users, centroid, macro_action = map(np.asarray, zip(*trajectory))
            raw_action, executed_route = map(np.asarray, zip(*routes))
            np.savez_compressed(output / f"{label}_{seed}_trajectory.npz", uav=uav, users=users, centroid=centroid, macro_action=macro_action, raw_action=raw_action, executed_route=executed_route)
    return summaries


def run(config, output, method="ppo", smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = config["training"]
    torch.set_num_threads(1)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    model_env = QueueMDP(config)
    model_env.reset(cfg["seed"], cfg["window_macros"])
    agent = DirectHRL.from_mdp(model_env, config)
    initial_sac, initial_macro = snapshot(agent.sac_actor.action_net), snapshot(agent.macro_actor)
    root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for folder in ("hrl_uav","scripts","configs") for p in (root/folder).rglob("*") if p.is_file() and p.suffix in (".py",".json")}
    write_json(output / "manifest.json", {"config":config,"method":method,"smoke":smoke,"source_hashes":sources,
        "observations": agent.observation_spec, "observation_slices": model_env.micro_feature_slices,
        "state_dim": agent.micro_state_dim, "micro_state_dim": agent.micro_state_dim,
        "macro_state_dim": agent.macro_state_dim, "micro_action_dim": model_env.micro_action_dim,
        "device":str(next(agent.macro_actor.parameters()).device),"teacher":False,"old_checkpoint":None})
    started = time.perf_counter()
    all_validation = []
    stage_audits = []
    for stage in ("pretrain", "joint"):
        agent.set_stage(stage)
        stage_sac = snapshot(agent.sac_actor.action_net)
        stage_macro = snapshot(agent.macro_actor)
        stage_updates = (agent.sac_updates, agent.sac_actor_updates, agent.ppo_updates)
        windows = cfg[f"{stage}_windows"]
        for offset in range(0, windows, cfg["stream_windows"]):
            chunk_windows = min(cfg["stream_windows"], windows-offset)
            seed = cfg["seed"] + offset//cfg["stream_windows"] + (10000 if stage == "joint" else 0)
            summary, rows, _, _, _ = rollout(
                config, agent, seed, chunk_windows*cfg["window_macros"],
                macro_enabled=stage == "joint" and method == "ppo", train=True,
                window_offset=offset, progress_offset=offset/max(windows,1), complete_tail=True,
            )
            append_csv(output / "training.csv", rows)
            append_csv(output / "boundary_audit.csv", [{"stage":stage,"stream_seed":seed,"simulated_s":summary["time_s"],"boundary_transitions":summary["boundary_transitions"],"bootstrap_discount":summary["boundary_discount_min"],"pending_dropped":summary["final_pending_dropped"]}])
            completed = offset + chunk_windows
            if completed == windows or (stage == "joint" and completed % 15 == 0):
                label = f"validation_{stage}_{completed}"
                readings = evaluate(config, agent, config["validation_seeds"], cfg["window_macros"], output, label, macro_enabled=stage=="joint" and method=="ppo")
                all_validation.extend(readings)
                append_csv(output / "validation.csv", readings)
                print(f"[validation {stage} {completed}] {np.mean([r['task_delay_weighted_ms'] for r in readings]):.2f} ms",flush=True)
        if stage == "joint" and method == "ppo":
            agent.update_ppo(1.0, force=True)
        stage_audits.append({"stage":stage,"sac_actor_max_delta":parameter_delta(stage_sac,agent.sac_actor.action_net),"macro_mean_max_delta":parameter_delta(stage_macro,agent.macro_actor,exclude=("log_std",)),"sac_critic_updates":agent.sac_updates-stage_updates[0],"sac_actor_updates":agent.sac_actor_updates-stage_updates[1],"ppo_updates":agent.ppo_updates-stage_updates[2]})
        torch.save(agent.checkpoint(config), output / ("micro_pretrained.pt" if stage=="pretrain" else "final.pt"))
    full = evaluate(config, agent, config["test_seeds"], config["test_macros"], output, "final", macro_enabled=method=="ppo")
    matched = evaluate(config, agent, config["test_seeds"], config["test_macros"], output, "centroid_same_sac", macro_enabled=False)
    append_csv(output / "final_evaluation.csv", full+matched)
    differences = [a["task_delay_weighted_ms"]-b["task_delay_weighted_ms"] for a,b in zip(full,matched)]
    report = {
        "status":"SMOKE_ONLY" if smoke else "SHORT_PILOT_NOT_FORMAL",
        "minutes":(time.perf_counter()-started)/60,"stage_audits":stage_audits,
        "sac_max_delta":parameter_delta(initial_sac,agent.sac_actor.action_net),
        "macro_mean_max_delta":parameter_delta(initial_macro,agent.macro_actor,exclude=("log_std",)),
        "final":full,"matched_centroid":matched,"paired_macro_delay_ms":differences,
        "training_seed_count":1,"evaluation_seed_count":len(config["test_seeds"]),
        "convergence":"Not established by completion of training; inspect fixed-seed validation and continuous queues.",
        "baseline_scope":"centroid_same_sac is a frozen component ablation, not an independently trained competitor.",
    }
    write_json(output / "report.json",report)
    return report


def smoke_config(config):
    config = copy.deepcopy(config)
    config["training"].update(pretrain_windows=1,joint_windows=1,stream_windows=1,window_macros=2,
                              batch_size=16,replay_warmup=16,sac_update_frames=1,sac_actor_warmup_updates=0,
                              ppo_update_macros=2,ppo_epochs=2,replay_capacity=1000)
    config["validation_seeds"] = [31000]
    config["test_seeds"] = [32000]
    config["test_macros"] = 6
    return config
