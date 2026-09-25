"""Explicit P7 jobs. Importing/dry-running this module does not import torch."""
import copy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

from .industrial_baselines import CVXRouting, RandomRouting, hover_scores, local_scores
from .industrial_mdp import IndustrialMDP
from .industrial_rollout import IndustrialRollout, LearningControllers, build_agent, store_transitions
from .industrial_scene import load_config

METHODS = ("ppo_sac", "ppo_cvx", "ppo_random", "ppo_local")


def json_write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_protocol(path, root):
    protocol = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    scene = load_config(Path(root) / protocol["scene_config"])
    budget_summary(protocol, scene)
    return protocol, scene


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def whole_count(seconds, dt):
    if not np.isfinite(seconds) or seconds <= 0 or not np.isclose(seconds / dt, round(seconds / dt)):
        raise ValueError("Duration must be a positive whole number of decision intervals")
    return round(seconds / dt)


def budget_summary(protocol, scene):
    if protocol["schema"] != "industrial_p7_v1" or tuple(protocol["methods"]) != METHODS:
        raise ValueError("Unsupported protocol or method list")
    seeds = protocol["training_seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Training seeds must be distinct nonnegative integers")
    b, l, e, r, t = (protocol[k] for k in ("budget", "learning", "evaluation", "randomness", "runtime"))
    streams = positive_integer(b["streams"], "streams")
    positive_integer(b["checkpoint_every_streams"], "checkpoint interval")
    for k in ("batch_size", "replay_capacity", "replay_warmup", "sac_update_frames", "sac_policy_delay", "ppo_update_macros", "ppo_epochs", "sac_critic_steps_per_update"):
        positive_integer(l[k], k)
    if l["replay_capacity"] < max(l["replay_warmup"], l["batch_size"]):
        raise ValueError("Replay cannot reach its learning threshold")
    if any(k in l for k in ("macro_gamma", "macro_gamma_frames", "ppo_std_decay_windows", "joint_windows", "joint_lr")):
        raise ValueError("Legacy time/stage overrides are not allowed")
    if l["micro_algorithm"] != "sac" or l["sac_stage_policy"] != "preserve":
        raise ValueError("This protocol uses SAC with one continuous learning schedule")
    if l["gamma_per_resource_frame"] != scene["objective"]["gamma_per_routing_frame"]:
        raise ValueError("Scene and learning discounts disagree")
    macros = whole_count(b["seconds_per_stream"], scene["time"]["motion_s"])
    frames = whole_count(b["seconds_per_stream"], scene["time"]["routing_s"])
    per_macro = whole_count(scene["time"]["motion_s"], scene["time"]["routing_s"])
    if l["sac_update_frames"] % per_macro:
        raise ValueError("SAC updates must fall on completed macro blocks for this runner")
    whole_count(e["seconds"], scene["time"]["motion_s"])
    if e["warmup_seconds"] != 0 or e["selection"] != "fixed_budget_final_only" or e["hover_pair"] != "same_frozen_sac":
        raise ValueError("Unexpected evaluation/selection policy")
    lengths = [len(e[k]) for k in ("arrival_seeds", "layout_seeds", "routing_seeds")]
    if lengths[0] == 0 or len(set(lengths)) != 1:
        raise ValueError("Evaluation seed tuples must have equal nonzero lengths")
    if r["seed_stride"] < streams:
        raise ValueError("Training seed ranges overlap")
    all_ranges = []
    for field in ("training_arrival_base", "training_layout_base", "routing_base"):
        values = [r[field] + seed*r["seed_stride"] + i for seed in seeds for i in range(streams)]
        if len(values) != len(set(values)):
            raise ValueError("Duplicate training seed")
        all_ranges.append(set(values))
    all_ranges.extend(set(e[k]) for k in ("arrival_seeds", "layout_seeds", "routing_seeds"))
    if any(len(set(e[k])) != lengths[0] for k in ("arrival_seeds", "layout_seeds", "routing_seeds")):
        raise ValueError("Duplicate evaluation seed")
    for i, values in enumerate(all_ranges):
        if any(values & other for other in all_ranges[i+1:]):
            raise ValueError("Seed namespaces overlap")
    if t["device"] not in ("cpu", "cuda"):
        raise ValueError("Explicit cpu or cuda device required")
    positive_integer(t["torch_threads"], "torch_threads")
    positive_integer(t["max_wall_seconds_per_job"], "wall limit")
    total_frames, total_macros = frames*streams, macros*streams
    warmup = max(l["batch_size"], l["replay_warmup"])
    first_update = ((warmup+l["sac_update_frames"]-1)//l["sac_update_frames"])*l["sac_update_frames"]
    sac_updates = max(0, (total_frames-first_update)//l["sac_update_frames"]+1)
    actor_updates = sum(i > l["sac_actor_warmup_updates"] and i % l["sac_policy_delay"] == 0 for i in range(1, sac_updates+1))
    return dict(streams_per_job=streams, seconds_per_job=b["seconds_per_stream"]*streams,
                physical_steps_per_job=whole_count(b["seconds_per_stream"], scene["time"]["physical_s"])*streams,
                frames_per_stream=frames, macros_per_stream=macros, frames_per_job=total_frames,
                macros_per_job=total_macros, ppo_updates_per_job=(total_macros+l["ppo_update_macros"]-1)//l["ppo_update_macros"],
                sac_updates_in_sac_job=sac_updates, sac_critic_fits=sac_updates*l["sac_critic_steps_per_update"],
                sac_actor_updates=actor_updates, independent_training_jobs=len(seeds)*len(METHODS),
                evaluation_trajectories=len(seeds)*(len(METHODS)+1)*lengths[0],
                evaluation_seconds=len(seeds)*(len(METHODS)+1)*lengths[0]*e["seconds"],
                macro_gamma=scene["objective"]["gamma_per_routing_frame"]**per_macro,
                endpoint="fixed budget final checkpoint; no best-model selection")


def job_seeds(protocol, seed, index):
    r = protocol["randomness"]
    return tuple(r[k]+seed*r["seed_stride"]+index for k in ("training_arrival_base", "training_layout_base", "routing_base"))


def runtime_setup(protocol, seed):
    # No environment variables or existing source files are changed.
    import torch
    from . import learning
    from .agents import ppo_agent, sac_agent
    requested = protocol["runtime"]["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; no silent device substitution")
    device = torch.device(requested)
    learning.device = ppo_agent.device = sac_agent.device = device
    torch.set_num_threads(protocol["runtime"]["torch_threads"])
    torch.manual_seed(seed)
    if requested == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return torch


def runtime_versions():
    versions = {"python": sys.version}
    for name in ("numpy", "torch", "cvxpy", "scipy", "clarabel"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def make_agent(protocol, mdp, seed, method):
    settings = copy.deepcopy(protocol["learning"])
    settings["seed"] = seed
    if method != "ppo_sac":
        # Construct the same initial PPO RNG sequence without an unused large replay.
        settings["replay_capacity"] = 2
    agent = build_agent(mdp, settings)
    agent.set_stage("joint")
    return agent


def lower_controller(method, controllers, scene, seed):
    if method in ("ppo_sac", "hover_sac"):
        return controllers.micro
    if method == "ppo_cvx":
        return CVXRouting(scene)
    if method == "ppo_random":
        return RandomRouting(seed)
    if method == "ppo_local":
        return local_scores
    raise ValueError(method)


class TrainingSchedule:
    """PPO updates before the next committed macro action; SAC after true successor."""
    def __init__(self, agent, method, protocol, total_macros):
        self.agent, self.method, self.protocol = agent, method, protocol
        self.total_macros = total_macros
        self.macros = self.frames = 0
        self.ppo_seconds = self.sac_seconds = 0.

    def macro_boundary(self, transition):
        self.macros += 1
        self.agent.macro_rollout.append((transition.state, transition.action, transition.logprob,
            transition.reward, transition.next_state, transition.trace_continue))
        started = time.perf_counter()
        self.agent.update_ppo(progress=self.macros/self.total_macros, force=self.macros == self.total_macros)
        self.ppo_seconds += time.perf_counter()-started

    def completed_micro(self, collected):
        for transition in collected["micro"]:
            self.frames += 1
            if self.method != "ppo_sac":
                continue
            store_transitions(self.agent, {"micro": [transition]}, macro=False)
            if self.frames % self.protocol["learning"]["sac_update_frames"] == 0:
                started = time.perf_counter()
                self.agent.update_sac()
                self.sac_seconds += time.perf_counter()-started


def network_fingerprint(agent):
    h = hashlib.sha256()
    for module in (agent.macro_actor, agent.macro_critic, agent.sac_actor.action_net,
                   agent.sac_critic.critic_v, agent.sac_critic.target_critic_v):
        for key, value in module.state_dict().items():
            h.update(key.encode())
            h.update(value.detach().cpu().numpy().tobytes())
    h.update(agent.entropy.log_alpha.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def counters(agent):
    return dict(ppo=agent.ppo_updates, ppo_actor_gradient_steps=agent.ppo_gradient_steps,
                sac=agent.sac_updates, sac_actor=agent.sac_actor_updates,
                sac_critic_fits=agent.sac_critic_gradient_steps, replay_writes=agent.replay.memory_counter)


def source_hashes(root):
    names = ["industrial_scene.py", "industrial_env.py", "industrial_mdp.py", "industrial_observation.py",
             "industrial_rollout.py", "industrial_baselines.py", "industrial_experiment.py",
             "env.py", "mdp.py", "geometry.py", "load_switching.py", "routing.py", "learning.py", "xu_cvx_r.py", "agents/ppo_agent.py", "agents/sac_agent.py", "agents/tqc.py"]
    paths = {"hrl_uav/"+name: Path(root)/"hrl_uav"/name for name in names}
    paths["hrl_uav/industrial_experiment.py"] = Path(__file__)
    paths["scripts/run_industrial_p7.py"] = Path(__file__).resolve().parents[1]/"scripts/run_industrial_p7.py"
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def timed(controller, accumulator):
    def call(view):
        start = time.perf_counter()
        result = controller(view)
        accumulator[0] += time.perf_counter()-start
        accumulator[1] += 1
        return result
    return call


def check_wall(start, protocol):
    if time.perf_counter()-start > protocol["runtime"]["max_wall_seconds_per_job"]:
        raise TimeoutError("Job wall limit reached; partial run is not a fixed-budget endpoint")


def collect_evaluation(mdp, upper, lower, seconds, wall_check, *, trajectory_enabled):
    upper_time, lower_time = [0., 0], [0., 0]
    rollout = IndustrialRollout(mdp, timed(upper, upper_time), timed(lower, lower_time))
    trajectory = [[0., *np.asarray(mdp.controller_information().positions_m).ravel().tolist()]] if trajectory_enabled else []
    for _ in range(whole_count(seconds, mdp.config["time"]["motion_s"])):
        wall_check()
        collected = rollout.run(1, truncate=False)
        if trajectory_enabled:
            trajectory.append([mdp.env.time_s, *np.asarray(mdp.controller_information().positions_m).ravel().tolist()])
    return collected["metrics"], trajectory, upper_time, lower_time


def train_job(protocol, scene, method, seed, output, root):
    summary = budget_summary(protocol, scene)
    if method not in METHODS or seed not in protocol["training_seeds"]:
        raise ValueError("Job is not registered in this protocol")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(phase="train", status="running", method=method, training_seed=seed,
                    protocol=protocol, scene=scene, protocol_sha256=digest(protocol), scene_sha256=digest(scene),
                    source_sha256=source_hashes(root), runtime_versions=runtime_versions(), budget=summary, selection="final_only")
    json_write(output/"manifest.json", manifest)
    start = time.perf_counter()
    try:
        torch = runtime_setup(protocol, seed)
        mdp = IndustrialMDP(scene)
        agent = make_agent(protocol, mdp, seed, method)
        scheduler = TrainingSchedule(agent, method, protocol, summary["macros_per_job"])
        upper_time, lower_time = [0., 0], [0., 0]
        cvx_fallbacks = 0
        for stream in range(protocol["budget"]["streams"]):
            arrival, layout, routing = job_seeds(protocol, seed, stream)
            mdp.reset(arrival, layout)
            controllers = LearningControllers(agent, mdp.encoder, deterministic=False)
            lower = lower_controller(method, controllers, scene, routing)
            rollout = IndustrialRollout(mdp, timed(controllers.macro, upper_time), timed(lower, lower_time),
                                        on_macro_boundary=scheduler.macro_boundary)
            for macro in range(summary["macros_per_stream"]):
                check_wall(start, protocol)
                collected = rollout.run(1, truncate=macro == summary["macros_per_stream"]-1)
                scheduler.completed_micro(collected)
            fallback = sum(row["fallback"] for row in getattr(lower, "records", []))
            cvx_fallbacks += fallback
            row = dict(stream=stream, arrival_seed=arrival, layout_seed=layout, routing_seed=routing,
                       metrics=collected["metrics"], updates=counters(agent), cvx_fallbacks=fallback)
            with (output/"training.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False)+"\n")
            if hasattr(lower, "records"):
                json_write(output/f"cvx_stream_{stream:04d}.json", lower.records)
            final = stream+1 == protocol["budget"]["streams"]
            if final or (stream+1) % protocol["budget"]["checkpoint_every_streams"] == 0:
                config = dict(schema="industrial_static_v1", training=agent.cfg)
                checkpoint = agent.checkpoint(config, include_replay=False)
                checkpoint["industrial_run"] = dict(protocol_sha256=digest(protocol), scene_sha256=digest(scene),
                    source_sha256=manifest["source_sha256"], method=method, training_seed=seed,
                    completed_streams=stream+1, final=final, resumable=False)
                torch.save(checkpoint, output/("final.pt" if final else f"stream_{stream+1:04d}.pt"))
            json_write(output/"progress.json", dict(completed_streams=stream+1,
                completed_macros=scheduler.macros, updates=counters(agent), elapsed_seconds=time.perf_counter()-start))
        actual = counters(agent)
        expected_sac = summary["sac_updates_in_sac_job"] if method == "ppo_sac" else 0
        assert actual["ppo"] == summary["ppo_updates_per_job"] and actual["sac"] == expected_sac
        assert scheduler.frames == summary["frames_per_job"] and scheduler.macros == summary["macros_per_job"]
        if source_hashes(root) != manifest["source_sha256"]:
            raise RuntimeError("Source files changed during training; endpoint cannot be marked complete")
        manifest.update(status="complete", updates=actual, network_sha256=network_fingerprint(agent),
            elapsed_seconds=time.perf_counter()-start, upper_inference=upper_time, lower_inference=lower_time,
            ppo_update_seconds=scheduler.ppo_seconds, sac_update_seconds=scheduler.sac_seconds,
            cvx_fallbacks=cvx_fallbacks, checkpoint_sha256=hashlib.sha256((output/"final.pt").read_bytes()).hexdigest())
        json_write(output/"manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter()-start)
        json_write(output/"manifest.json", manifest)
        raise


def evaluate_job(protocol, scene, run_directory, output, root):
    budget_summary(protocol, scene)
    run_directory, output = Path(run_directory), Path(output)
    trained = json.loads((run_directory/"manifest.json").read_text(encoding="utf-8"))
    expected_hashes = source_hashes(root)
    if (trained["status"] != "complete" or trained["protocol_sha256"] != digest(protocol)
            or trained["scene_sha256"] != digest(scene) or trained["source_sha256"] != expected_hashes):
        raise ValueError("Training must be complete with the same protocol, scene and source files")
    checkpoint_path = run_directory/"final.pt"
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != trained["checkpoint_sha256"]:
        raise ValueError("Final checkpoint fingerprint mismatch")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(phase="evaluate", status="running", method=trained["method"], training_seed=trained["training_seed"],
                  protocol_sha256=digest(protocol), scene_sha256=digest(scene), source_sha256=expected_hashes,
                  training_directory=str(run_directory.resolve()), runtime_versions=runtime_versions(), rows=[])
    json_write(output/"evaluation.json", report)
    started = time.perf_counter()
    try:
        torch = runtime_setup(protocol, trained["training_seed"])
        mdp = IndustrialMDP(scene)
        agent = make_agent(protocol, mdp, trained["training_seed"], trained["method"])
        # Only this locally produced, hash-checked checkpoint is accepted.
        saved = torch.load(checkpoint_path, map_location=protocol["runtime"]["device"], weights_only=False)
        metadata = saved["industrial_run"]
        if not metadata["final"] or metadata["completed_streams"] != protocol["budget"]["streams"]:
            raise ValueError("Intermediate checkpoint cannot be used as the selected endpoint")
        if any(metadata[key] != trained[key] for key in ("protocol_sha256", "scene_sha256", "source_sha256", "method", "training_seed")):
            raise ValueError("Checkpoint identity does not match training manifest")
        for key, module in (("macro_actor", agent.macro_actor), ("macro_critic", agent.macro_critic),
                            ("sac_actor", agent.sac_actor.action_net), ("sac_critic", agent.sac_critic.critic_v),
                            ("sac_target", agent.sac_critic.target_critic_v)):
            module.load_state_dict(saved[key], strict=True)
        with torch.no_grad():
            agent.entropy.log_alpha.copy_(saved["log_alpha"])
        initial, counts = network_fingerprint(agent), counters(agent)
        assert initial == trained["network_sha256"]
        eval_cfg = protocol["evaluation"]
        variants = [trained["method"]] + (["hover_sac"] if trained["method"] == "ppo_sac" else [])
        for variant in variants:
            for index, (arrival, layout, routing) in enumerate(zip(eval_cfg["arrival_seeds"], eval_cfg["layout_seeds"], eval_cfg["routing_seeds"])):
                mdp.reset(arrival, layout)
                controllers = LearningControllers(agent, mdp.encoder, deterministic=True)
                lower = lower_controller(variant, controllers, scene, routing)
                upper = hover_scores if variant == "hover_sac" else controllers.macro
                metrics, trajectory, upper_time, lower_time = collect_evaluation(mdp, upper, lower,
                    eval_cfg["seconds"], lambda: check_wall(started, protocol),
                    trajectory_enabled=variant in ("ppo_sac", "hover_sac"))
                row = dict(variant=variant, training_seed=trained["training_seed"], scene_index=index,
                    arrival_seed=arrival, layout_seed=layout, routing_seed=routing, metrics=metrics,
                    upper_inference=upper_time, lower_inference=lower_time,
                    cvx_fallbacks=sum(r["fallback"] for r in getattr(lower, "records", [])))
                if hasattr(lower, "records"):
                    json_write(output/f"cvx_scene_{index:02d}.json", lower.records)
                if variant in ("ppo_sac", "hover_sac"):
                    json_write(output/f"trajectory_{variant}_{index:02d}.json", trajectory)
                report["rows"].append(row)
                json_write(output/"evaluation.json", report)
        if network_fingerprint(agent) != initial or counters(agent) != counts:
            raise AssertionError("Evaluation changed network parameters or learning counters")
        if source_hashes(root) != expected_hashes:
            raise RuntimeError("Source files changed during evaluation")
        report.update(status="complete", learning_updates=0, replay_operations=0,
                      parameters_unchanged=True, elapsed_seconds=time.perf_counter()-started)
        json_write(output/"evaluation.json", report)
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        json_write(output/"evaluation.json", report)
        raise


def aggregate_evaluations(protocol, scene, reports):
    """Equal scene weighting within seed, then independent training seed weighting."""
    import math
    grouped, pair_hashes, seen = {}, {}, set()
    expected_sources = None
    for report in reports:
        if report["status"] != "complete" or report["protocol_sha256"] != digest(protocol) or report["scene_sha256"] != digest(scene):
            raise ValueError("Incomplete or incompatible evaluation")
        if expected_sources is None:
            expected_sources = report["source_sha256"]
        if report["source_sha256"] != expected_sources:
            raise ValueError("Source revisions differ across methods")
        job = (report["method"], report["training_seed"])
        if job[0] not in METHODS or job[1] not in protocol["training_seeds"]:
            raise ValueError("Unregistered training job")
        if job in seen:
            raise ValueError("Duplicate evaluation job")
        seen.add(job)
        variants = {job[0]} | ({"hover_sac"} if job[0] == "ppo_sac" else set())
        expected_rows = {(variant, index) for variant in variants for index in range(len(protocol["evaluation"]["arrival_seeds"]))}
        if len(report["rows"]) != len(expected_rows) or {(r["variant"], r["scene_index"]) for r in report["rows"]} != expected_rows:
            raise ValueError("Incomplete or duplicate variant/scenario rows")
        for row in report["rows"]:
            if row["training_seed"] != job[1]:
                raise ValueError("Row training seed disagrees with its job")
            for key, field in (("arrival_seed", "arrival_seeds"), ("layout_seed", "layout_seeds"), ("routing_seed", "routing_seeds")):
                if row[key] != protocol["evaluation"][field][row["scene_index"]]:
                    raise ValueError("Evaluation seed differs from registered tuple")
            if row["metrics"]["seconds"] != protocol["evaluation"]["seconds"]:
                raise ValueError("Evaluation duration differs from registered duration")
            key = row["scene_index"]
            event = row["metrics"]["event_sha256"]
            if key in pair_hashes and pair_hashes[key] != event:
                raise ValueError("Exogenous event hashes differ within a paired comparison")
            pair_hashes[key] = event
            grouped.setdefault((row["variant"], row["training_seed"]), []).append(row)
    per_seed = []
    scalar_metrics = ("mean_cost_per_s", "weighted_delay_ms", "average_power_w", "holding_task_s",
        "unfinished_parents", "oldest_unfinished_age_s", "upload_pending_bits", "ground_pending_cycles",
        "ready_cycles", "transfer_pending_bits", "transfer_future_cycles", "cpu_pending_cycles")
    for (variant, seed), rows in sorted(grouped.items()):
        if sorted(r["scene_index"] for r in rows) != list(range(len(protocol["evaluation"]["arrival_seeds"]))):
            raise ValueError("Missing or duplicate evaluation scenes")
        values = {}
        for key in scalar_metrics:
            observations = [r["metrics"][key] for r in rows]
            values[key] = None if any(v is None for v in observations) else float(np.mean(observations))
        for priority in range(2):
            for key in ("mean_delay_ms", "completion_rate"):
                observations = [r["metrics"]["priority"][priority][key] for r in rows]
                values[f"priority_{priority}_{key}"] = None if any(v is None for v in observations) else float(np.mean(observations))
        rates = [sum(p["completed"] for p in r["metrics"]["priority"])/sum(p["arrived"] for p in r["metrics"]["priority"]) if sum(p["arrived"] for p in r["metrics"]["priority"]) else None for r in rows]
        values["completion_rate"] = None if any(v is None for v in rates) else float(np.mean(rates))
        energy_keys = sorted(rows[0]["metrics"]["energy_j"])
        for key in energy_keys:
            values["energy_j_"+key] = float(np.mean([r["metrics"]["energy_j"][key] for r in rows]))
        values["energy_j_total"] = float(np.mean([sum(r["metrics"]["energy_j"].values()) for r in rows]))
        objective = scene["objective"]
        seconds = protocol["evaluation"]["seconds"]
        values["holding_cost_per_s"] = objective["holding_weight"]*values["holding_task_s"]/objective["holding_reference_task_s"]/seconds
        values["energy_cost_per_s"] = (1-objective["holding_weight"])*values["energy_j_total"]/objective["energy_reference_j"]/seconds
        per_seed.append(dict(variant=variant, training_seed=seed, means=values,
                             cvx_fallbacks=sum(r["cvx_fallbacks"] for r in rows)))
    summary = {}
    for variant in sorted({row["variant"] for row in per_seed}):
        rows = [r for r in per_seed if r["variant"] == variant]
        statistics = {}
        for key in rows[0]["means"]:
            values = [r["means"][key] for r in rows]
            valid = all(v is not None and math.isfinite(v) for v in values)
            statistics[key] = dict(mean=float(np.mean(values)) if valid else None,
                training_seed_std=float(np.std(values, ddof=1)) if valid and len(values) > 1 else None)
        summary[variant] = dict(training_seeds=[r["training_seed"] for r in rows], statistics=statistics)
    expected = {(method, seed) for method in METHODS for seed in protocol["training_seeds"]}
    matched_seeds = sorted(seed for seed in protocol["training_seeds"] if all((method, seed) in seen for method in METHODS))
    pairs = []
    seed_lookup = {(r["variant"], r["training_seed"]): r for r in per_seed}
    for seed in protocol["training_seeds"]:
        if ("ppo_sac", seed) in seed_lookup and ("hover_sac", seed) in seed_lookup:
            moved, hovered = seed_lookup[("ppo_sac", seed)]["means"], seed_lookup[("hover_sac", seed)]["means"]
            pairs.append(dict(training_seed=seed, difference_moving_minus_hover={
                key: moved[key]-hovered[key] if moved[key] is not None and hovered[key] is not None else None for key in moved}))
    return dict(complete_matrix=seen == expected, matched_training_seeds=matched_seeds,
                missing_jobs=sorted(expected-seen), paired_events_verified=True,
                aggregation="equal scenario means, then equal independent training seed means; no significance claims",
                per_seed=per_seed, methods=summary, motion_pairs=pairs)
