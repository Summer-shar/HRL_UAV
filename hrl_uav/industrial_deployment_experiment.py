"""Explicit deployment V3 jobs. Dry-run does not import a neural runtime."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from .industrial_baselines import CVXRouting, RandomRouting, local_scores
from .industrial_experiment import (METHODS, aggregate_evaluations as _aggregate,
    counters, digest, network_fingerprint, positive_integer, runtime_setup,
    runtime_versions, timed, whole_count)
from .industrial_rollout import IndustrialRollout
from .industrial_deployment_training import (DeploymentLearningControllers,
    DeploymentTrainingSchedule)


def json_write(path, data):
    """Atomically publish JSON so progress readers never consume half a record."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
        allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_json(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_protocol(path, root):
    from .industrial_deployment_scene import validate_deployment_config
    path = Path(path)
    if not path.is_absolute():
        path = Path(root) / path
    protocol = json.loads(path.read_text(encoding="utf-8-sig"))
    relative = Path(protocol["scene_config"])
    candidates = [relative] if relative.is_absolute() else [path.parent.parent / relative, Path(root) / relative]
    scene_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if scene_path is None:
        raise FileNotFoundError("The deployment scene referenced by the protocol is missing")
    scene = json.loads(scene_path.read_text(encoding="utf-8-sig"))
    validate_deployment_config(scene)
    budget_summary(protocol, scene)
    return protocol, scene


def stream_durations(seconds, stream_seconds):
    if seconds == 0:
        return []
    count = int(math.ceil(seconds / stream_seconds))
    return [min(stream_seconds, seconds - i * stream_seconds) for i in range(count)]


def job_seeds(protocol, seed, index, stage):
    settings = protocol["randomness"]
    if stage not in ("pretrain", "joint") or not 0 <= index < settings["seed_stride"]:
        raise ValueError("Unregistered stage or exhausted seed namespace")
    return tuple(settings[stage][name] + seed * settings["seed_stride"] + index
        for name in ("arrival_base", "layout_base", "routing_base"))


def budget_summary(protocol, scene):
    from .industrial_deployment_scene import validate_deployment_config
    validate_deployment_config(scene)
    if protocol.get("schema") != "industrial_deployment_v3" or protocol.get("observation_version") != "industrial_deployment_v3":
        raise ValueError("Unsupported deployment protocol or observation contract")
    if protocol.get("purpose") not in ("research", "gate") or tuple(protocol["methods"]) != METHODS:
        raise ValueError("Explicit research/gate purpose and complete registered methods required")
    seeds = protocol["training_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Training seeds must be distinct nonnegative integers")
    base, budget, learning = scene["base_scene"], protocol["budget"], protocol["learning"]
    macro, frame, physical = (base["time"][key] for key in ("motion_s", "routing_s", "physical_s"))
    for key in ("pretrain_seconds", "joint_seconds", "stream_seconds", "log_window_seconds"):
        whole_count(budget[key], macro)
    whole_count(budget["stream_seconds"], budget["log_window_seconds"])
    for stage in ("pretrain", "joint"):
        whole_count(budget[stage + "_seconds"], budget["log_window_seconds"])
    for key in ("batch_size", "replay_capacity", "replay_warmup", "sac_update_frames",
            "sac_policy_delay", "ppo_update_macros", "ppo_epochs", "sac_critic_steps_per_update",
            "ppo_std_decay_updates"):
        positive_integer(learning[key], key)
    if learning["replay_capacity"] < max(learning["batch_size"], learning["replay_warmup"]):
        raise ValueError("Replay capacity cannot reach its learning threshold")
    if learning["micro_algorithm"] != "sac" or learning["sac_stage_policy"] != "preserve":
        raise ValueError("Deployment requires continuous SAC learning across stages")
    if learning["ppo_update_guard"] != "post_step_batch_kl_v1":
        raise ValueError("Explicit post-step PPO guard required")
    if learning["gamma_per_resource_frame"] != base["objective"]["gamma_per_routing_frame"]:
        raise ValueError("Learning and physical discounts disagree")
    if learning["sac_update_frames"] % whole_count(macro, frame):
        raise ValueError("SAC update frequency must align with completed macro blocks")
    forbidden = {"macro_gamma", "macro_gamma_frames", "ppo_std_decay_windows", "joint_windows", "joint_lr"}
    if forbidden & learning.keys():
        raise ValueError("Legacy clock/stage overrides are not supported")
    runtime, evaluation = protocol["runtime"], protocol["evaluation"]
    if runtime["device"] not in ("cpu", "cuda") or runtime["resume"] != "stream_boundary":
        raise ValueError("Explicit device and stream-boundary resume contract required")
    positive_integer(runtime["torch_threads"], "torch_threads")
    positive_integer(runtime["max_wall_seconds_per_job"], "wall limit")
    whole_count(evaluation["seconds"], macro)
    if evaluation["warmup_seconds"] != 0 or evaluation["selection"] != "fixed_budget_final_only" or evaluation["hover_pair"] != "same_frozen_sac":
        raise ValueError("Unexpected final evaluation policy")
    lengths = [len(evaluation[k]) for k in ("arrival_seeds", "layout_seeds", "routing_seeds")]
    if not lengths[0] or len(set(lengths)) != 1:
        raise ValueError("Evaluation seed tuples must have equal nonzero lengths")
    stride = positive_integer(protocol["randomness"]["seed_stride"], "seed_stride")
    namespaces = []
    stages = {}
    cumulative_frames = previous_updates = previous_actors = 0
    threshold = max(learning["batch_size"], learning["replay_warmup"])
    first = math.ceil(threshold / learning["sac_update_frames"]) * learning["sac_update_frames"]
    for stage in ("pretrain", "joint"):
        seconds = budget[stage + "_seconds"]
        streams = stream_durations(seconds, budget["stream_seconds"])
        if len(streams) > stride:
            raise ValueError("Stage exceeds its stream seed namespace")
        for column in range(3):
            values = [job_seeds(protocol, seed, i, stage)[column] for seed in seeds for i in range(len(streams))]
            if len(values) != len(set(values)):
                raise ValueError("Duplicate training seeds")
            namespaces.append(set(values))
        frames, macros = whole_count(seconds, frame), whole_count(seconds, macro)
        cumulative_frames += frames
        updates = max(0, (cumulative_frames - first) // learning["sac_update_frames"] + 1)
        actors = max(0, updates // learning["sac_policy_delay"] - learning["sac_actor_warmup_updates"] // learning["sac_policy_delay"])
        stages[stage] = dict(seconds=seconds, stream_count=len(streams), stream_durations=streams,
            log_windows=whole_count(seconds, budget["log_window_seconds"]),
            physical_steps=whole_count(seconds, physical), frames=frames, macros=macros,
            ppo_samples=macros if stage == "joint" else 0,
            ppo_update_events=math.ceil(macros / learning["ppo_update_macros"]) if stage == "joint" else 0,
            sac_update_events=updates - previous_updates,
            sac_critic_fits=(updates - previous_updates) * learning["sac_critic_steps_per_update"],
            sac_actor_updates=actors - previous_actors)
        previous_updates, previous_actors = updates, actors
    for key in ("arrival_seeds", "layout_seeds", "routing_seeds"):
        if len(set(evaluation[key])) != len(evaluation[key]):
            raise ValueError("Duplicate evaluation seeds")
        namespaces.append(set(evaluation[key]))
    if any(a & b for i, a in enumerate(namespaces) for b in namespaces[i + 1:]):
        raise ValueError("Registered random seed namespaces overlap")
    return dict(stages=stages, total_seconds=sum(s["seconds"] for s in stages.values()),
        total_physical_steps=sum(s["physical_steps"] for s in stages.values()),
        macro_gamma=base["objective"]["gamma_per_routing_frame"] ** whole_count(macro, frame),
        endpoint="fixed-budget final only", resume_granularity="complete independent stream",
        baseline_extra_pretrain_seconds=0, sac_extra_pretrain_seconds=budget["pretrain_seconds"])


def source_hashes(root):
    """Hash resolved imported sources; keys survive staged installation unchanged."""
    names = ("industrial_scene", "industrial_env", "industrial_mdp", "industrial_observation",
        "industrial_rollout", "industrial_baselines", "industrial_experiment", "env", "mdp",
        "geometry", "load_switching", "routing", "learning", "xu_cvx_r", "agents.ppo_agent",
        "agents.sac_agent", "agents.tqc", "industrial_deployment_scene", "industrial_deployment_env",
        "industrial_deployment_observation", "industrial_deployment_mdp", "industrial_deployment_learning",
        "industrial_deployment_training", "industrial_deployment_checkpoint", "industrial_deployment_experiment")
    result = {}
    for name in names:
        spec = importlib.util.find_spec("hrl_uav." + name)
        if spec is None or spec.origin is None:
            raise FileNotFoundError("Missing runtime dependency: " + name)
        result["hrl_uav/" + name.replace(".", "/") + ".py"] = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    script = Path(__file__).resolve().parents[1] / "scripts/run_industrial_deployment.py"
    result["scripts/run_industrial_deployment.py"] = hashlib.sha256(script.read_bytes()).hexdigest()
    return result


def make_agent(protocol, mdp, seed, method):
    from .industrial_deployment_learning import build_deployment_agent
    settings = copy.deepcopy(protocol["learning"])
    settings["seed"] = seed
    if method != "ppo_sac":
        settings["replay_capacity"] = 2
    return build_deployment_agent(mdp, settings)


class BaseReportController:
    def __init__(self, controller):
        self.controller = controller

    def __call__(self, view):
        return self.controller(view.base)

    @property
    def records(self):
        return getattr(self.controller, "records", [])


def lower_controller(method, controllers, scene, seed):
    if method in ("ppo_sac", "hover_sac"):
        return controllers.micro
    if method == "ppo_cvx":
        return BaseReportController(CVXRouting(scene["base_scene"]))
    if method == "ppo_random":
        return BaseReportController(RandomRouting(seed))
    if method == "ppo_local":
        return BaseReportController(local_scores)
    raise ValueError("Unknown method: " + method)


def ppo_fingerprint(agent):
    value = hashlib.sha256()
    for module in (agent.macro_actor, agent.macro_critic):
        for key, tensor in module.state_dict().items():
            value.update(key.encode())
            value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _seed_table(protocol, stream_plan=None):
    return {stage: {str(seed): [dict(stream=i, arrival_seed=values[0], layout_seed=values[1], routing_seed=values[2])
        for i in range(len(stream_plan[stage] if stream_plan is not None else stream_durations(protocol["budget"][stage + "_seconds"], protocol["budget"]["stream_seconds"])))
        for values in [job_seeds(protocol, seed, i, stage)]] for seed in protocol["training_seeds"]}
        for stage in ("pretrain", "joint")}


def dry_run(protocol, scene, output, root, *, method=None, seed=None, stage="both"):
    from .industrial_deployment_mdp import DeploymentIndustrialMDP
    summary = budget_summary(protocol, scene)
    # Construction publishes the initial report; no service/arrival preparation.
    mdp = DeploymentIndustrialMDP(scene)
    result = dict(phase="dry-run", training_started=False, environment_service_steps=0,
        torch_imported="torch" in sys.modules, protocol_sha256=digest(protocol), scene_sha256=digest(scene),
        source_sha256=source_hashes(root), observations=mdp.observation_spec(),
        macro_dim=85, micro_dim=86, critic_input_dim=104, replay_row_width=210,
        macro_action_dim=6, micro_action_dim=18, deployment=scene["deployment"],
        max_inter_macro_migration_m_per_s=scene["deployment"]["max_target_step_m"] / scene["deployment"]["macro_period_s"],
        clocks=scene["base_scene"]["time"], budget=summary,
        selected_job=dict(method=method, training_seed=seed, stage=stage),
        all_registered_jobs_are_not_started=True)
    if result["torch_imported"]:
        raise RuntimeError("Dry-run must not import torch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("protocol.json", protocol), ("scene.json", scene),
            ("observation_schema.json", result["observations"]), ("seed_table.json", _seed_table(protocol)),
            ("dry_run.json", result)):
        json_write(output / name, value)
    return result


def _window_row(previous, current, *, stage, stream, global_seconds, routes, objective):
    seconds = current["seconds"] - previous.get("seconds", 0.)
    holding = current["holding_task_s"] - previous.get("holding_task_s", 0.)
    energy = {key: value - previous.get("energy_j", {}).get(key, 0.) for key, value in current["energy_j"].items()}
    cost = current["cost_sum"] - previous.get("cost_sum", 0.)
    row = dict(stage=stage, stream=stream, end_stream_seconds=current["seconds"],
        end_global_seconds=global_seconds, seconds=seconds, cost_sum=cost,
        mean_cost_per_s=cost / seconds, holding_task_s=holding, energy_j=energy,
        holding_cost_per_s=objective["holding_weight"] * holding / objective["holding_reference_task_s"] / seconds,
        energy_cost_per_s=(1 - objective["holding_weight"]) * sum(energy.values()) / objective["energy_reference_j"] / seconds,
        metrics_at_end=current)
    for name in ("raw", "executed"):
        values = np.asarray(routes[name])
        row["routing_" + name] = dict(mean=values.mean(axis=0).tolist(), std=values.std(axis=0).tolist(),
            minimum=values.min(axis=0).tolist(), maximum=values.max(axis=0).tolist())
    return row


def _assert_counters(agent, scheduler, protocol, scene, method):
    base = scene["base_scene"]
    if scheduler.frames != scheduler.macros * whole_count(base["time"]["motion_s"], base["time"]["routing_s"]):
        raise AssertionError("Physical and policy clocks disagree")
    if scheduler.physical_steps != scheduler.frames * whole_count(base["time"]["routing_s"], base["time"]["physical_s"]):
        raise AssertionError("Physical service step count mismatch")
    if scheduler.ppo_samples != scheduler.stage_macros["joint"] or scheduler.ppo_samples != scheduler.consumed_ppo_samples + len(agent.macro_rollout):
        raise AssertionError("Lost or duplicated PPO samples")
    settings = protocol["learning"]
    first = math.ceil(max(settings["batch_size"], settings["replay_warmup"]) / settings["sac_update_frames"]) * settings["sac_update_frames"]
    expected = max(0, (scheduler.frames - first) // settings["sac_update_frames"] + 1) if method == "ppo_sac" else 0
    actor = max(0, expected // settings["sac_policy_delay"] - settings["sac_actor_warmup_updates"] // settings["sac_policy_delay"])
    if (agent.sac_updates != expected or agent.sac_critic_gradient_steps != expected * settings["sac_critic_steps_per_update"]
            or agent.sac_actor_updates != actor or agent.replay.memory_counter != (scheduler.frames if method == "ppo_sac" else 0)):
        raise AssertionError("SAC lifecycle update or replay counter mismatch")


def train_job(protocol, scene, method, seed, output, root, *, stage="both", checkpoint=None,
              extend_stage_seconds=0, phase="train"):
    """Run one explicit job; resume/handoff never edits the parent run."""
    from .industrial_deployment_mdp import DeploymentIndustrialMDP
    from .industrial_deployment_checkpoint import load_checkpoint, make_checkpoint, restore_checkpoint, save_checkpoint
    if phase not in ("train", "resume") or stage not in ("pretrain", "joint", "both"):
        raise ValueError("Invalid phase or stage")
    protocol = copy.deepcopy(protocol)
    budget_summary(protocol, scene)
    if method not in METHODS or seed not in protocol["training_seeds"]:
        raise ValueError("This method/seed is not registered")
    if method != "ppo_sac" and stage != "joint" and phase != "resume":
        raise ValueError("Non-SAC baselines run only the joint stage")
    if phase == "resume" and checkpoint is None:
        raise ValueError("Resume requires a complete stream checkpoint")
    if extend_stage_seconds and phase != "resume":
        raise ValueError("Budget extension is explicit resume only")
    if phase == "train" and method == "ppo_sac" and stage == "joint" and checkpoint is None:
        raise ValueError("SAC joint training requires its complete pretraining endpoint")
    hashes = source_hashes(root)
    identity = dict(method=method, training_seed=seed, source_sha256=hashes)
    saved = load_checkpoint(checkpoint) if checkpoint is not None else None
    runtime_setup(protocol, seed)
    mdp = DeploymentIndustrialMDP(scene)
    agent = make_agent(protocol, mdp, seed, method)
    state = dict(stage="pretrain" if stage in ("pretrain", "both") else "joint",
        stage_plan=["pretrain", "joint"] if stage == "both" else [stage],
        stream_plan={part: stream_durations(protocol["budget"][part + "_seconds"], protocol["budget"]["stream_seconds"])
                     for part in ("pretrain", "joint")},
        stage_completed_seconds={"pretrain": 0., "joint": 0.}, next_stream={"pretrain": 0, "joint": 0},
        committed_streams=0, committed_windows=0, completed_global_seconds=0.,
        ppo_pretrain_fingerprint=ppo_fingerprint(agent), parent_checkpoint=None, at_stream_boundary=True,
        upper_inference=[0., 0], lower_inference=[0., 0], cvx_fallbacks=0)
    if saved is not None:
        state = restore_checkpoint(agent, saved, protocol=protocol, scene=scene, identity=identity)
        state["parent_checkpoint"] = saved.get("_load_metadata")
        if phase == "train":
            if stage != "joint" or method != "ppo_sac" or state["stage"] != "pretrain" or state["stage_completed_seconds"]["pretrain"] != protocol["budget"]["pretrain_seconds"]:
                raise ValueError("Only a completed SAC pretraining endpoint can unlock joint training")
            state["stage_plan"] = ["joint"]
            state["stage"] = "joint"
        if extend_stage_seconds:
            whole_count(extend_stage_seconds, protocol["budget"]["log_window_seconds"])
            active = state["stage"]
            if state["stage_completed_seconds"][active] != protocol["budget"][active + "_seconds"]:
                raise ValueError("Extend only a completed stage; incomplete jobs use ordinary resume")
            protocol["budget"][active + "_seconds"] += extend_stage_seconds
            state["stream_plan"][active].extend(stream_durations(extend_stage_seconds, protocol["budget"]["stream_seconds"]))
            if len(state["stream_plan"][active]) > protocol["randomness"]["seed_stride"]:
                raise ValueError("Extension exhausted the registered stage seed namespace")
            state["stage_plan"] = [active]
            state["extension_seconds"] = extend_stage_seconds
            budget_summary(protocol, scene)
    if all(state["stage_completed_seconds"][part] >= protocol["budget"][part + "_seconds"] for part in state["stage_plan"]):
        raise ValueError("This run is already complete; use explicit budget extension")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(phase=phase, status="running", method=method, training_seed=seed,
        requested_stage=stage, protocol=protocol, scene=scene, protocol_sha256=digest(protocol),
        scene_sha256=digest(scene), source_sha256=hashes, observations=mdp.observation_spec(),
        runtime_versions=runtime_versions(), budget=budget_summary(protocol, scene),
        parent_checkpoint=state.get("parent_checkpoint"), selection="fixed_budget_final_only",
        committed_streams=state["committed_streams"], resumed=phase == "resume",
        execution_stream_plan=state["stream_plan"], stage_plan=state["stage_plan"])
    for part in ("pretrain", "joint"):
        manifest["budget"]["stages"][part]["stream_count"] = len(state["stream_plan"][part])
        manifest["budget"]["stages"][part]["stream_durations"] = state["stream_plan"][part]
    for name, value in (("manifest.json", manifest), ("protocol.json", protocol), ("scene.json", scene),
            ("observation_schema.json", mdp.observation_spec()), ("seed_table.json", _seed_table(protocol, state["stream_plan"]))):
        json_write(output / name, value)
    scheduler = DeploymentTrainingSchedule(agent, method, protocol,
        on_update=lambda row: append_json(output / "learning.jsonl", row))
    if saved is not None:
        scheduler.load_state_dict(state["schedule"])
    started = time.perf_counter()
    base = scene["base_scene"]
    macros_per_window = whole_count(protocol["budget"]["log_window_seconds"], base["time"]["motion_s"])
    try:
        for active in state["stage_plan"]:
            target = protocol["budget"][active + "_seconds"]
            if state["stage_completed_seconds"][active] >= target:
                continue
            state["stage"] = active
            scheduler.set_stage(active, whole_count(target, base["time"]["motion_s"]))
            while state["stage_completed_seconds"][active] < target:
                if time.perf_counter() - started > protocol["runtime"]["max_wall_seconds_per_job"]:
                    raise TimeoutError("Wall limit reached; resume from the last committed stream")
                stream = state["next_stream"][active]
                duration = state["stream_plan"][active][stream]
                if duration > target - state["stage_completed_seconds"][active]:
                    raise ValueError("Saved stream plan exceeds the explicit stage budget")
                arrival, layout, routing = job_seeds(protocol, seed, stream, active)
                state["at_stream_boundary"] = False
                mdp.reset(arrival, layout)
                controllers = DeploymentLearningControllers(agent, mdp.encoder,
                    deterministic=False, macro_enabled=active == "joint")
                lower = lower_controller(method, controllers, scene, routing)
                pending_commands = []
                original_start = mdp.start_macro

                def start_command(action):
                    before = np.asarray(mdp.controller_information().base.positions_m).copy()
                    command = original_start(action)
                    pending_commands.append(dict(time_s=mdp.env.time_s, raw_action=np.asarray(action).tolist(),
                        start_positions_m=before.tolist(), command=_jsonable(command)))
                    return command

                def complete_macro(transition):
                    command = pending_commands.pop(0)
                    if active == "pretrain" and not np.array_equal(np.asarray(command["start_positions_m"]), mdp.env.physics.uav_coords):
                        raise AssertionError("Pretraining UAV moved despite a zero upper action")
                    append_json(output / "motions.jsonl", dict(stage=active, stream=stream,
                        macro=scheduler.macros, reward=transition.reward,
                        end_positions_m=mdp.env.physics.uav_coords.tolist(), **command))
                    scheduler.macro_boundary(transition)

                mdp.start_macro = start_command
                rollout = IndustrialRollout(mdp, timed(controllers.macro, state["upper_inference"]),
                    timed(lower, state["lower_inference"]), on_macro_boundary=complete_macro)
                previous, routes = {}, {"raw": [], "executed": []}
                macro_count = whole_count(duration, base["time"]["motion_s"])
                try:
                    for index in range(macro_count):
                        if time.perf_counter() - started > protocol["runtime"]["max_wall_seconds_per_job"]:
                            raise TimeoutError("Wall limit reached; partial stream is uncommitted")
                        collected = rollout.run(1, truncate=index + 1 == macro_count)
                        scheduler.completed_micro(collected)
                        for transition in collected["micro"]:
                            routes["raw"].append(transition.action.ravel())
                            routes["executed"].append(transition.route.ravel())
                        if (index + 1) % macros_per_window == 0:
                            current = collected["metrics"]
                            elapsed_global = state["completed_global_seconds"] + current["seconds"]
                            row = _window_row(previous, current, stage=active, stream=stream,
                                global_seconds=elapsed_global, routes=routes, objective=base["objective"])
                            row["updates"] = counters(agent)
                            row["macro_observation_at_end_before_priming"] = collected["macro"][-1].next_state.tolist()
                            row["uncommitted_stream"] = True
                            append_json(output / "windows.jsonl", row)
                            previous, routes = current, {"raw": [], "executed": []}
                            json_write(output / "progress.json", dict(status="running", stage=active,
                                committed_streams=state["committed_streams"], stage_stream=stream,
                                simulated_seconds=elapsed_global, committed_seconds=state["completed_global_seconds"],
                                stage_seconds=state["stage_completed_seconds"][active] + current["seconds"],
                                stage_target_seconds=target, updates=counters(agent),
                                elapsed_seconds=time.perf_counter() - started, latest_window=row))
                finally:
                    mdp.start_macro = original_start
                fallback = sum(record["fallback"] for record in getattr(lower, "records", []))
                state["cvx_fallbacks"] += fallback
                state["stage_completed_seconds"][active] += duration
                state["completed_global_seconds"] += duration
                state["next_stream"][active] += 1
                state["committed_streams"] += 1
                state["committed_windows"] += whole_count(duration, protocol["budget"]["log_window_seconds"])
                state["schedule"] = scheduler.state_dict()
                stage_final = state["stage_completed_seconds"][active] == target
                state["stage_complete"] = stage_final
                state["at_stream_boundary"] = True
                state["run_complete"] = all(state["stage_completed_seconds"][part] >= protocol["budget"][part + "_seconds"] for part in state["stage_plan"])
                _assert_counters(agent, scheduler, protocol, scene, method)
                if active == "pretrain" and ppo_fingerprint(agent) != state["ppo_pretrain_fingerprint"]:
                    raise AssertionError("Pretraining changed PPO parameters or standard deviation")
                if source_hashes(root) != hashes:
                    raise RuntimeError("Runtime sources changed; checkpoint cannot be committed")
                row = dict(stage=active, stream=stream, arrival_seed=arrival, layout_seed=layout,
                    routing_seed=routing, metrics=collected["metrics"], updates=counters(agent),
                    cvx_fallbacks=fallback, committed_stream=state["committed_streams"],
                    completed_global_seconds=state["completed_global_seconds"])
                # Log is committed only when the immutable checkpoint/index publishes.
                append_json(output / "training.jsonl", row)
                payload = make_checkpoint(agent, protocol, scene, identity, state)
                persistent = ("pretrain_final.pt" if active == "pretrain" else "final.pt") if stage_final else None
                receipt = save_checkpoint(output / "checkpoints", payload, persistent_name=persistent)
                manifest.update(committed_streams=state["committed_streams"], committed_seconds=state["completed_global_seconds"],
                    latest_checkpoint=receipt, stage=active, updates=counters(agent))
                json_write(output / "manifest.json", manifest)
                json_write(output / "progress.json", dict(status="running", stage=active,
                    stage_stream=stream, committed_streams=state["committed_streams"],
                    simulated_seconds=state["completed_global_seconds"], committed_seconds=state["completed_global_seconds"],
                    stage_seconds=state["stage_completed_seconds"][active], stage_target_seconds=target,
                    updates=counters(agent), elapsed_seconds=time.perf_counter() - started,
                    latest_window=row, checkpoint=receipt))
        final_name = "final.pt" if state["stage"] == "joint" else "pretrain_final.pt"
        endpoint = output / "checkpoints" / final_name
        manifest.update(status="complete", completed_stage=state["stage"], run_state=state,
            updates=counters(agent), network_sha256=network_fingerprint(agent),
            elapsed_seconds=time.perf_counter() - started, checkpoint_path=str(endpoint.resolve()),
            checkpoint_sha256=hashlib.sha256(endpoint.read_bytes()).hexdigest(),
            upper_inference=state["upper_inference"], lower_inference=state["lower_inference"],
            ppo_update_seconds=scheduler.ppo_seconds, sac_update_seconds=scheduler.sac_seconds)
        json_write(output / "manifest.json", manifest)
        json_write(output / "progress.json", dict(status="complete", stage=state["stage"],
            committed_seconds=state["completed_global_seconds"], committed_streams=state["committed_streams"],
            updates=counters(agent), checkpoint_path=str(endpoint.resolve())))
        return manifest
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed",
            error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter() - started)
        json_write(output / "manifest.json", manifest)
        json_write(output / "failure.json", dict(error=manifest["error"],
            committed_streams=manifest["committed_streams"], resume="last published checkpoints/latest.json only"))
        raise


def resume_job(checkpoint, output, root, *, extend_stage_seconds=0):
    from .industrial_deployment_checkpoint import load_checkpoint
    saved = load_checkpoint(checkpoint)
    identity = saved["identity"]
    return train_job(saved["protocol"], saved["scene"], identity["method"], identity["training_seed"],
        output, root, phase="resume", checkpoint=checkpoint, extend_stage_seconds=extend_stage_seconds)


def evaluate_job(protocol, scene, run_directory, output, root):
    from .industrial_deployment_mdp import DeploymentIndustrialMDP
    from .industrial_deployment_checkpoint import load_checkpoint, restore_checkpoint
    budget_summary(protocol, scene)
    run_directory, output = Path(run_directory), Path(output)
    trained = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    expected = source_hashes(root)
    if (trained["status"] != "complete" or trained["completed_stage"] != "joint"
            or trained["protocol_sha256"] != digest(protocol) or trained["scene_sha256"] != digest(scene)
            or trained["source_sha256"] != expected):
        raise ValueError("Evaluation requires a complete joint endpoint with matching contracts")
    checkpoint = run_directory / "checkpoints/final.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != trained["checkpoint_sha256"]:
        raise ValueError("Final checkpoint digest mismatch")
    runtime_setup(protocol, trained["training_seed"])
    mdp = DeploymentIndustrialMDP(scene)
    agent = make_agent(protocol, mdp, trained["training_seed"], trained["method"])
    identity = dict(method=trained["method"], training_seed=trained["training_seed"], source_sha256=expected)
    saved = load_checkpoint(checkpoint)
    restored = restore_checkpoint(agent, saved, protocol=protocol, scene=scene, identity=identity, restore_random=False)
    if not restored["run_complete"] or restored["stage"] != "joint":
        raise ValueError("Intermediate checkpoints are not final evaluation endpoints")
    initial, count = network_fingerprint(agent), counters(agent)
    if initial != trained["network_sha256"]:
        raise ValueError("Loaded network differs from sealed final endpoint")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(phase="evaluate", status="running", method=trained["method"],
        training_seed=trained["training_seed"], protocol_sha256=digest(protocol), scene_sha256=digest(scene),
        source_sha256=expected, training_directory=str(run_directory.resolve()), runtime_versions=runtime_versions(), rows=[])
    json_write(output / "evaluation.json", report)
    started, event_pairs = time.perf_counter(), {}
    try:
        cfg = protocol["evaluation"]
        variants = [trained["method"]] + (["hover_sac"] if trained["method"] == "ppo_sac" else [])
        for variant in variants:
            for index, (arrival, layout, routing) in enumerate(zip(cfg["arrival_seeds"], cfg["layout_seeds"], cfg["routing_seeds"])):
                mdp.reset(arrival, layout)
                controllers = DeploymentLearningControllers(agent, mdp.encoder,
                    deterministic=True, macro_enabled=variant != "hover_sac")
                lower = lower_controller(variant, controllers, scene, routing)
                upper_time, lower_time = [0., 0], [0., 0]
                rollout = IndustrialRollout(mdp, timed(controllers.macro, upper_time), timed(lower, lower_time))
                trajectory = [[0., *np.asarray(mdp.controller_information().base.positions_m).ravel().tolist()]]
                for _ in range(whole_count(cfg["seconds"], mdp.config["time"]["motion_s"])):
                    if time.perf_counter() - started > protocol["runtime"]["max_wall_seconds_per_job"]:
                        raise TimeoutError("Evaluation wall limit exceeded")
                    collected = rollout.run(1, truncate=False)
                    if variant in ("ppo_sac", "hover_sac"):
                        trajectory.append([mdp.env.time_s, *np.asarray(mdp.controller_information().base.positions_m).ravel().tolist()])
                metrics = collected["metrics"]
                if index in event_pairs and event_pairs[index] != metrics["event_sha256"]:
                    raise AssertionError("Motion/hover exogenous events are not paired")
                event_pairs[index] = metrics["event_sha256"]
                report["rows"].append(dict(variant=variant, training_seed=trained["training_seed"],
                    scene_index=index, arrival_seed=arrival, layout_seed=layout, routing_seed=routing,
                    metrics=metrics, upper_inference=upper_time, lower_inference=lower_time,
                    cvx_fallbacks=sum(row["fallback"] for row in getattr(lower, "records", []))))
                if variant in ("ppo_sac", "hover_sac"):
                    json_write(output / f"trajectory_{variant}_{index:02d}.json", trajectory)
                json_write(output / "evaluation.json", report)
        if network_fingerprint(agent) != initial or counters(agent) != count:
            raise AssertionError("Frozen evaluation altered parameters or learning counters")
        if source_hashes(root) != expected:
            raise RuntimeError("Runtime sources changed during evaluation")
        report.update(status="complete", learning_updates=0, replay_operations=0,
            parameters_unchanged=True, elapsed_seconds=time.perf_counter() - started)
        json_write(output / "evaluation.json", report)
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        json_write(output / "evaluation.json", report)
        raise


def aggregate_evaluations(protocol, scene, reports):
    # The legacy aggregation has no schema dependency; retain its paired-seed checks.
    return _aggregate(protocol, scene["base_scene"], reports)
