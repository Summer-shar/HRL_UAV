"""Random-layout/role episodes with checkpointed admission and direct joint learning."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .industrial_experiment import (METHODS, counters, digest, network_fingerprint,
    positive_integer, runtime_setup, runtime_versions, timed, whole_count)
from .industrial_deployment_experiment import (append_json, json_write, make_agent,
    ppo_fingerprint, _jsonable, _window_row, _assert_counters,
    source_hashes as _deployment_sources)
from .industrial_deployment_training import DeploymentLearningControllers, DeploymentTrainingSchedule
from .industrial_rollout import IndustrialRollout


def load_protocol(path, root):
    from .industrial_hotspot_scene import validate_hotspot_config
    path = Path(path)
    if not path.is_absolute():
        path = Path(root) / path
    protocol = json.loads(path.read_text(encoding="utf-8-sig"))
    relative = Path(protocol["scene_config"])
    candidates = [relative] if relative.is_absolute() else [path.parent.parent / relative, Path(root) / relative]
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None:
        raise FileNotFoundError("Referenced hotspot scene is missing")
    scene = json.loads(source.read_text(encoding="utf-8-sig"))
    validate_hotspot_config(scene)
    budget_summary(protocol, scene)
    return protocol, scene


def source_hashes(root):
    result = _deployment_sources(root)
    names = ("industrial_hotspot_scene", "industrial_hotspot_env", "industrial_hotspot_mdp",
        "industrial_hotspot_gate", "industrial_hotspot_evaluation", "industrial_hotspot_checkpoint",
        "industrial_hotspot_experiment")
    for name in names:
        spec = importlib.util.find_spec("hrl_uav." + name)
        if spec is None or spec.origin is None:
            raise FileNotFoundError("Missing hotspot source: " + name)
        result["hrl_uav/" + name + ".py"] = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    root_source = Path(__file__).resolve().parents[1]
    for name in ("run_industrial_hotspot.py", "validate_industrial_hotspot.py"):
        result["scripts/" + name] = hashlib.sha256((root_source / "scripts" / name).read_bytes()).hexdigest()
    return result


def episode_seeds(protocol, seed, index, stage):
    random = protocol["randomness"]
    if stage not in ("pretrain", "joint") or type(index) is not int or not 0 <= index < random["training_stride"]:
        raise ValueError("Invalid episode cursor or exhausted seed namespace")
    return {name + "_seed": random[stage][name + "_base"] + seed * random["training_stride"] + index
        for name in ("arrival", "layout", "load", "routing")}


def budget_summary(protocol, scene):
    from .industrial_hotspot_scene import validate_hotspot_config
    validate_hotspot_config(scene)
    if protocol.get("schema") != "industrial_hotspot_v4" or protocol.get("purpose") not in ("research", "gate"):
        raise ValueError("Explicit hotspot V4 research/gate protocol required")
    if protocol.get("observation_version") != "industrial_deployment_v3" or tuple(protocol["methods"]) != METHODS:
        raise ValueError("The V4 protocol preserves the deployment observations and registered methods")
    budget, learning, base = protocol["budget"], protocol["learning"], scene["base_scene"]
    episode = budget["episode_seconds"]
    macros = whole_count(episode, base["time"]["motion_s"])
    frames = whole_count(episode, base["time"]["routing_s"])
    steps = whole_count(episode, base["time"]["physical_s"])
    whole_count(episode, budget["log_window_seconds"])
    whole_count(budget["log_window_seconds"], base["time"]["motion_s"])
    pretrain = positive_integer(budget["pretrain_max_episodes"], "pretrain_max_episodes")
    joint = positive_integer(budget["joint_episodes"], "joint_episodes")
    for name, maximum in (("pretrain_check_episodes", pretrain), ("joint_development_check_episodes", joint)):
        nodes = budget[name]
        if protocol["purpose"] == "gate" and name == "joint_development_check_episodes" and not nodes:
            continue
        if not nodes or nodes != sorted(set(nodes)) or any(type(x) is not int or x <= 0 or x > maximum for x in nodes) or nodes[-1] != maximum:
            raise ValueError("Check nodes must be distinct, increasing and include the stage endpoint")
    if len(budget["pretrain_check_episodes"]) < protocol["admission"]["consecutive_checkpoints_required"]:
        raise ValueError("Not enough distinct admission checkpoints")
    if protocol["purpose"] == "research":
        if episode != 900 or budget["log_window_seconds"] != 300 or pretrain != 667 or joint != 600:
            raise ValueError("Research V4 uses complete 900s episodes and the registered 667/600 limits")
        if budget["pretrain_check_episodes"] != [200, 300, 400, 500, 600, 667] or budget["joint_development_check_episodes"] != [200, 400, 600]:
            raise ValueError("Research checkpoint schedule changed; register a new protocol")
        if protocol["training_seeds"] != [1]:
            raise ValueError("The first V4 batch registers only training seed 1")
    if budget.get("stream_seconds_alias", episode) != episode or budget.get("no_short_tail_episode") is not True:
        raise ValueError("Short tail episodes are not supported")
    if learning["micro_algorithm"] != "sac" or learning["sac_stage_policy"] != "preserve":
        raise ValueError("SAC must preserve its learning lifecycle across the two stages")
    for name in ("batch_size", "replay_capacity", "replay_warmup", "sac_update_frames", "sac_policy_delay",
            "sac_critic_steps_per_update", "ppo_update_macros", "ppo_epochs", "ppo_std_decay_updates"):
        positive_integer(learning[name], name)
    if learning["replay_capacity"] < max(learning["batch_size"], learning["replay_warmup"]):
        raise ValueError("Replay cannot reach its learning threshold")
    if learning["sac_update_frames"] % whole_count(base["time"]["motion_s"], base["time"]["routing_s"]):
        raise ValueError("SAC update boundaries must coincide with complete macro blocks")
    if learning["gamma_per_resource_frame"] != base["objective"]["gamma_per_routing_frame"]:
        raise ValueError("Physical and learning discounts disagree")
    if protocol["runtime"]["device"] not in ("cpu", "cuda") or protocol["runtime"]["resume"] != "stream_boundary":
        raise ValueError("Explicit device and full-episode continuation required")
    positive_integer(protocol["runtime"]["torch_threads"], "torch_threads")
    positive_integer(protocol["runtime"]["max_wall_seconds_per_job"], "wall limit")
    if type(protocol.get("opportunity_diagnostic", {}).get("failure_blocks_automatic_training", True)) is not bool:
        raise ValueError("Spatial diagnostic blocking policy must be an explicit boolean")
    seeds = protocol["training_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Distinct nonnegative training seeds required")
    namespaces = []
    for stage, count in (("pretrain", pretrain), ("joint", joint)):
        for field in ("arrival_seed", "layout_seed", "load_seed", "routing_seed"):
            values = [episode_seeds(protocol, seed, i, stage)[field] for seed in seeds for i in range(count)]
            if len(values) != len(set(values)):
                raise ValueError("Training seed collision")
            namespaces.append(set(values))
    for name in ("development_manifest", "final_test_manifest"):
        manifest = protocol[name]
        whole_count(manifest["seconds"], base["time"]["motion_s"])
        scenarios = manifest["scenarios"]
        if not scenarios or len({row["scenario_id"] for row in scenarios}) != len(scenarios):
            raise ValueError("Evaluation scenes need distinct IDs")
        for row in scenarios:
            if sorted(row["profile_permutation"]) != [0, 1, 2]:
                raise ValueError("Invalid registered business profile permutation")
        if protocol["purpose"] == "research" and manifest["seconds"] != 900:
            raise ValueError("Research evaluation uses whole 900s scenes")
        for field in ("arrival_seed", "layout_seed", "load_seed", "routing_seed"):
            namespaces.append({row[field] for row in scenarios})
    if any(a & b for i, a in enumerate(namespaces) for b in namespaces[i + 1:]):
        raise ValueError("Training/development/final seed namespaces overlap")
    if protocol["purpose"] == "research" and (len(protocol["development_manifest"]["scenarios"]) != 6
            or len({tuple(s["profile_permutation"]) for s in protocol["development_manifest"]["scenarios"]}) != 6
            or len(protocol["final_test_manifest"]["scenarios"]) != 20):
        raise ValueError("Research development/final scenario registration differs")
    first = math.ceil(max(learning["batch_size"], learning["replay_warmup"])/learning["sac_update_frames"]) * learning["sac_update_frames"]

    def endpoint(pre_episodes):
        total_frames = (pre_episodes + joint) * frames
        updates = max(0, (total_frames - first)//learning["sac_update_frames"] + 1)
        actor = max(0, updates//learning["sac_policy_delay"] - learning["sac_actor_warmup_updates"]//learning["sac_policy_delay"])
        return dict(pretrain_episodes=pre_episodes, joint_episodes=joint,
            seconds=(pre_episodes + joint)*episode, physical_steps=(pre_episodes + joint)*steps,
            replay_writes=total_frames, sac_updates=updates,
            sac_critic_fits=updates*learning["sac_critic_steps_per_update"], sac_actor_updates=actor,
            ppo_samples=joint*macros, ppo_events=math.ceil(joint*macros/learning["ppo_update_macros"]))
    earliest = budget["pretrain_check_episodes"][protocol["admission"]["consecutive_checkpoints_required"] - 1]
    return dict(episode_seconds=episode, physical_steps_per_episode=steps, frames_per_episode=frames,
        macros_per_episode=macros, earliest_admitted_endpoint=endpoint(earliest),
        maximum_admitted_endpoint=endpoint(pretrain), pretrain_cap_requires_admission=True,
        development_physical_step_cap=protocol["evaluation_budget"]["first_batch_total_physical_steps_max"],
        automatically_run_final_tests=False)


def dry_run(protocol, scene, output, root, *, method=None, seed=None):
    from .industrial_hotspot_mdp import HotspotIndustrialMDP
    summary = budget_summary(protocol, scene)
    mdp = HotspotIndustrialMDP(scene)
    if "torch" in sys.modules:
        raise RuntimeError("Dry-run cannot import torch")
    result = dict(phase="dry-run", training_started=False, environment_service_steps=0,
        torch_imported=False, source_sha256=source_hashes(root), protocol_sha256=digest(protocol),
        scene_sha256=digest(scene), observations=mdp.observation_spec(), budget=summary,
        selected_job=dict(method=method, training_seed=seed), macro_dim=85, micro_dim=86,
        critic_input_dim=104, replay_row_width=210, explicit_training_prerequisites=["functional acceptance", "P3 spatial benefit passed"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("protocol.json", protocol), ("scene.json", scene), ("dry_run.json", result)):
        json_write(output / name, value)
    return result


def _read_evidence(path, protocol, scene, sources, label):
    if path is None:
        raise ValueError(label + " evidence is required before training")
    path = Path(path).absolute()
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    recorded_spatial_failure = (
        label in ("spatial diagnostic", "diagnostic")
        and protocol.get("opportunity_diagnostic", {}).get("failure_blocks_automatic_training", True) is False
        and value.get("schema") == "industrial_hotspot_spatial_diagnostic_v1"
        and value.get("passed") is False
        and value.get("status") == "spatial_benefit_unconfirmed")
    if not recorded_spatial_failure and (value.get("passed") is not True or value.get("status", "complete") not in ("complete", "passed")):
        raise ValueError(label + " must explicitly pass; completion alone is insufficient")
    for key, expected in (("protocol_sha256", digest(protocol)), ("scene_sha256", digest(scene)), ("source_sha256", sources)):
        if value.get(key) != expected:
            raise ValueError(label + " evidence identity mismatch: " + key)
    if recorded_spatial_failure:
        # Only a completed, technically valid quality comparison is informational.
        # Incomplete runs, source mismatches and technical failures still abort.
        from .industrial_hotspot_gate import paired_rows
        panel = protocol["development_manifest"]
        if value.get("dev_manifest_sha") != digest(panel):
            raise ValueError("Spatial diagnostic development panel mismatch")
        if any(type(value.get(key)) is not int or value[key] != 0
               for key in ("learning_updates", "replay_writes", "replay_samples")):
            raise ValueError("Spatial diagnostic must contain no learning or replay operations")
        pairs = paired_rows(value["reference_rows"], value["hover_rows"])
        expected = {str(row["scenario_id"]): row for row in panel["scenarios"]}
        if {str(a["scenario_id"]) for a, _ in pairs} != set(expected):
            raise ValueError("Spatial diagnostic is missing registered scenarios")
        for moving, hover in pairs:
            scenario = expected[str(moving["scenario_id"])]
            for row, variant in ((moving, "reference_cvx"), (hover, "hover_cvx")):
                if any(type(row.get(key)) is not int or row[key] != 0
                       for key in ("learning_updates", "replay_writes", "replay_samples")):
                    raise ValueError("Spatial diagnostic trajectory contains learning or replay operations")
                if (row.get("variant") != variant or row.get("scenario") != scenario
                        or row["scenario_sha"] != digest(scenario)
                        or row.get("profile_permutation") != scenario["profile_permutation"]
                        or row["metrics"]["seconds"] != panel["seconds"]):
                    raise ValueError("Spatial diagnostic trajectory contract mismatch")
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), record=value)


def _verify_saved_evidence(state, protocol, scene, sources):
    if state["startup_evidence"].get("fixture_only") is True:
        if protocol["purpose"] != "gate":
            raise ValueError("Synthetic startup evidence cannot authorize research training")
        for name in ("validation", "diagnostic"):
            value = state["startup_evidence"][name]["record"]
            if value.get("protocol_sha256") != digest(protocol) or value.get("scene_sha256") != digest(scene) or value.get("source_sha256") != sources:
                raise ValueError("Functional fixture identity changed")
        return
    for name in ("validation", "diagnostic"):
        sealed = state["startup_evidence"][name]
        current = _read_evidence(sealed["path"], protocol, scene, sources, name)
        if current["sha256"] != sealed["sha256"]:
            raise ValueError("Startup evidence changed after the checkpoint")


def _evaluate_subprocess(kind, checkpoint, output, root, cache_root, budget_path, *, timeout):
    script = Path(__file__).resolve().parents[1] / "scripts/run_industrial_hotspot.py"
    command = [sys.executable, "-B", str(script), "--phase", "worker-evaluate", "--evaluation-kind", kind,
        "--project-root", str(root), "--checkpoint", str(checkpoint), "--output", str(output),
        "--cache-root", str(cache_root), "--budget-path", str(budget_path)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    with (output / "worker.stdout.txt").open("a", encoding="utf-8") as stdout, (output / "worker.stderr.txt").open("a", encoding="utf-8") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, check=False, **kwargs)
    report_path = output / "report.json"
    if result.returncode != 0 or not report_path.is_file():
        raise RuntimeError("Frozen evaluation failed; the candidate remains pending: " + str(output))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return dict(report, report_path=str(report_path), evaluation_file_sha=hashlib.sha256(report_path.read_bytes()).hexdigest())


def _route_controller(method, controllers, mdp, seed):
    from .industrial_baselines import CVXRouting, RandomRouting, local_scores
    from .industrial_deployment_experiment import BaseReportController
    if method == "ppo_sac":
        return controllers.micro
    if method == "ppo_cvx":
        return BaseReportController(CVXRouting(mdp.config))
    if method == "ppo_random":
        return BaseReportController(RandomRouting(seed))
    return BaseReportController(local_scores)


def train_job(protocol, scene, method, seed, output, root, *, stage="both", checkpoint=None,
              phase="train", validation_record=None, diagnostic_record=None,
              evaluation_dispatch=None, stop_after_episodes=None):
    """One job. Candidate decisions are committed before any following episode."""
    from .industrial_hotspot_checkpoint import load_checkpoint, make_checkpoint, restore_checkpoint, save_checkpoint, capture_rng
    from .industrial_deployment_checkpoint import _same
    from .industrial_hotspot_mdp import HotspotIndustrialMDP
    from .industrial_hotspot_scene import generate_realization
    from .industrial_hotspot_gate import initial_gate_state, advance_gate, evaluate_admission
    budget_summary(protocol, scene)
    if stage not in ("pretrain", "joint", "both") or phase not in ("train", "resume"):
        raise ValueError("Unknown training stage or phase")
    if method not in METHODS or seed not in protocol["training_seeds"]:
        raise ValueError("Unregistered method or training seed")
    if protocol["purpose"] != "gate" and (evaluation_dispatch is not None or stop_after_episodes is not None):
        raise ValueError("Functional-test dispatch/interruption hooks cannot be used by research training")
    if stop_after_episodes is not None:
        positive_integer(stop_after_episodes, "stop_after_episodes")
    if phase == "resume" and checkpoint is None:
        raise ValueError("Resume requires a complete V4 checkpoint")
    if method != "ppo_sac" and stage != "joint" and phase != "resume":
        raise ValueError("Rule baselines train their PPO only in the joint stage")
    hashes = source_hashes(root)
    identity = dict(method=method, training_seed=seed, source_sha256=hashes)
    saved = load_checkpoint(checkpoint) if checkpoint is not None else None
    if saved is None:
        if protocol["purpose"] == "gate" and evaluation_dispatch is not None and validation_record is None and diagnostic_record is None:
            record = dict(fixture_only=True, protocol_sha256=digest(protocol), scene_sha256=digest(scene), source_sha256=hashes)
            evidence = dict(fixture_only=True, validation=dict(path=None, sha256=digest(record), record=record),
                diagnostic=dict(path=None, sha256=digest(record), record=record))
        else:
            evidence = dict(validation=_read_evidence(validation_record, protocol, scene, hashes, "functional acceptance"),
                diagnostic=_read_evidence(diagnostic_record, protocol, scene, hashes, "spatial diagnostic"))
    runtime_setup(protocol, seed)
    mdp = HotspotIndustrialMDP(scene)
    agent = make_agent(protocol, mdp, seed, method)
    budget = protocol["budget"]
    episode_seconds = budget["episode_seconds"]
    if saved is not None:
        state = restore_checkpoint(agent, saved, protocol=protocol, scene=scene, identity=identity)
        _verify_saved_evidence(state, protocol, scene, hashes)
        state["parent_checkpoint"] = saved["_load_metadata"]
        if phase == "train":
            if stage != "joint" or method != "ppo_sac" or state["stage"] != "pretrain" or state["gate"].get("status") != "admitted":
                raise ValueError("Only an admitted hotspot SAC endpoint can start joint training")
            state["requested_stage"] = "both"
    else:
        if method == "ppo_sac" and stage == "joint":
            raise ValueError("Joint SAC training requires an admitted V4 pretraining checkpoint")
        state = dict(stage="pretrain" if method == "ppo_sac" else "joint", requested_stage=stage,
            episodes={"pretrain": 0, "joint": 0}, completed_episodes=0, completed_seconds=0.,
            at_stream_boundary=True, gate=initial_gate_state(budget["pretrain_check_episodes"], budget["pretrain_max_episodes"]),
            pending_evaluation=None, joint_evaluated_nodes=[], startup_evidence=evidence,
            pretrain_ppo_sha=ppo_fingerprint(agent), parent_checkpoint=None, upper_inference=[0., 0], lower_inference=[0., 0],
            runtime_versions=runtime_versions(), role_counts={"pretrain": np.zeros((3, 3), int).tolist(), "joint": np.zeros((3, 3), int).tolist()})
        if method != "ppo_sac":
            state["gate"]["status"] = "not_applicable"
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    resumed_count = state["completed_episodes"]
    diagnostic = state["startup_evidence"]["diagnostic"]["record"]
    state.setdefault("evaluation_cache_root", diagnostic.get("cache_root", str(Path(root) / "industrial_hotspot_records" / ("cache-" + digest(protocol)[:16]))))
    state.setdefault("evaluation_budget_path", diagnostic.get("budget_path", str(Path(state["evaluation_cache_root"]) / "budget.json")))
    manifest = dict(schema="industrial_hotspot_run_v4", phase=phase, status="running", method=method,
        training_seed=seed, protocol=protocol, scene=scene, protocol_sha256=digest(protocol), scene_sha256=digest(scene),
        source_sha256=hashes, observations=mdp.observation_spec(), runtime_versions=runtime_versions(),
        budget=budget_summary(protocol, scene), parent_checkpoint=state["parent_checkpoint"],
        startup_evidence=state["startup_evidence"], committed_episodes=state["completed_episodes"])
    for name, value in (("manifest.json", manifest), ("protocol.json", protocol), ("scene.json", scene),
            ("development_manifest.json", protocol["development_manifest"]), ("observation_schema.json", mdp.observation_spec())):
        json_write(output / name, value)
    if saved is not None and state.get("last_episode_record") is not None:
        json_write(output / "restored_episode.json", dict(record=state["last_episode_record"],
            parent_checkpoint=saved["_load_metadata"], note="Frozen episode record recovered from the committed checkpoint; not a new episode"))
    scheduler = DeploymentTrainingSchedule(agent, method, protocol,
        on_update=lambda row: append_json(output / "learning.jsonl", row))
    if saved is not None:
        scheduler.load_state_dict(state["schedule"])
    started = time.perf_counter()
    base = scene["base_scene"]
    episode_macros = whole_count(episode_seconds, base["time"]["motion_s"])
    window_macros = whole_count(budget["log_window_seconds"], base["time"]["motion_s"])

    def wall_remaining():
        remaining = protocol["runtime"]["max_wall_seconds_per_job"] - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError("Wall limit reached; latest complete episode remains resumable")
        return remaining

    def publish(endpoint=None):
        state["schedule"] = scheduler.state_dict()
        state["at_stream_boundary"] = True
        if source_hashes(root) != hashes:
            raise RuntimeError("Source files changed; refusing to publish a mixed-version checkpoint")
        receipt = save_checkpoint(output / "checkpoints", make_checkpoint(agent, protocol, scene, identity, state), persistent_name=endpoint)
        manifest.update(committed_episodes=state["completed_episodes"], completed_seconds=state["completed_seconds"],
            stage=state["stage"], episodes=state["episodes"].copy(), updates=counters(agent), latest_checkpoint=receipt,
            gate=copy.deepcopy(state["gate"]), pending_evaluation=copy.deepcopy(state["pending_evaluation"]))
        json_write(output / "manifest.json", manifest)
        return receipt

    def evaluate_pending():
        pending = state["pending_evaluation"]
        if pending is None:
            return
        candidate = load_checkpoint(pending["candidate_path"])
        candidate_sha = candidate["_load_metadata"]["sha256"]
        if pending.get("candidate_sha") not in (None, candidate_sha):
            raise ValueError("Pending candidate fingerprint changed")
        pending["candidate_sha"] = candidate_sha
        # Keep paths stable across resumes so matching completed scenario results
        # are reused and a failed candidate is never sampled again for quality.
        destination = Path(pending["evaluation_output"])
        json_write(output / "candidate_pending.json", pending)
        before_rng = capture_rng()
        if evaluation_dispatch is None:
            report = _evaluate_subprocess(pending["kind"], pending["candidate_path"], destination, root,
                state["evaluation_cache_root"], state["evaluation_budget_path"], timeout=wall_remaining())
        else:
            report = evaluation_dispatch(pending["kind"], Path(pending["candidate_path"]), destination, root,
                Path(state["evaluation_cache_root"]), Path(state["evaluation_budget_path"]))
        if not _same(before_rng, capture_rng()):
            raise AssertionError("Frozen evaluation consumed the parent training RNG")
        if report.get("status") == "needs_review":
            manifest["status"] = "needs_review"
            publish("needs_review.pt")
            return
        if report.get("status") != "complete" or report.get("candidate_sha") != candidate_sha:
            raise ValueError("Candidate evaluation is incomplete or belongs to another checkpoint")
        for key, expected in (("protocol_sha256", digest(protocol)), ("scene_sha256", digest(scene)),
                ("source_sha256", hashes), ("dev_manifest_sha", digest(protocol["development_manifest"]))):
            if report.get(key) != expected:
                raise ValueError("Frozen evaluation identity mismatch: " + key)
        report_file = Path(report["report_path"])
        if not report_file.is_file() or hashlib.sha256(report_file.read_bytes()).hexdigest() != report["evaluation_file_sha"]:
            raise ValueError("Frozen evaluation receipt is not backed by its exact result file")
        disk_report = json.loads(report_file.read_text(encoding="utf-8"))
        if disk_report != {key: value for key, value in report.items() if key not in ("report_path", "evaluation_file_sha")}:
            raise ValueError("Frozen evaluation receipt contents differ from the committed file")
        if pending["kind"] == "admission":
            expected_identity = dict(candidate_sha=candidate_sha, scene_contract_sha=digest(scene),
                dev_manifest_sha=digest(protocol["development_manifest"]),
                admission_rule_version=protocol["admission"]["version"], evaluator_source_sha=digest(hashes))
            if report.get("identity") != expected_identity:
                raise ValueError("Admission cache identity differs from the sealed candidate contract")
            recomputed = evaluate_admission(report["sac_rows"], report["local_rows"], report["cvx_rows"], protocol["admission"])
            if recomputed != report["gate_result"] or recomputed["quality_passed"] != report["quality_passed"]:
                raise ValueError("Reported admission result disagrees with its frozen metrics")
        if pending["kind"] == "admission":
            state["gate"] = advance_gate(state["gate"], pending["episode"], candidate_sha, report)
        else:
            if pending["episode"] not in state["joint_evaluated_nodes"]:
                state["joint_evaluated_nodes"].append(pending["episode"])
        state["last_evaluation"] = dict(episode=pending["episode"], kind=pending["kind"],
            candidate_sha=candidate_sha, report_path=report["report_path"], evaluation_file_sha=report["evaluation_file_sha"])
        append_json(output / "decisions.jsonl", dict(**state["last_evaluation"], gate=state["gate"]))
        state["pending_evaluation"] = None
        endpoint = "pretrain_final.pt" if state["stage"] == "pretrain" and state["gate"].get("status") == "admitted" else None
        if state["gate"].get("status") == "needs_review":
            endpoint = "needs_review.pt"
        publish(endpoint)

    try:
        while True:
            wall_remaining()
            if state["pending_evaluation"] is not None:
                evaluate_pending()
            if state["gate"].get("status") == "needs_review" or manifest["status"] == "needs_review":
                manifest["status"] = "needs_review"
                break
            if state["stage"] == "pretrain" and state["gate"].get("status") == "admitted":
                if state["requested_stage"] == "pretrain":
                    manifest["status"] = "complete"
                    break
                if agent.ppo_updates or agent.macro_rollout or getattr(agent, "ppo_accepted_update_events", 0):
                    raise AssertionError("PPO did not remain untouched before joint admission")
                state["stage"] = "joint"
            if state["stage"] == "joint" and state["episodes"]["joint"] == budget["joint_episodes"]:
                manifest["status"] = "complete"
                publish("final.pt")
                break
            if stop_after_episodes is not None and state["completed_episodes"] - resumed_count >= stop_after_episodes:
                manifest["status"] = "interrupted"
                break
            active = state["stage"]
            maximum = budget["pretrain_max_episodes"] if active == "pretrain" else budget["joint_episodes"]
            if state["episodes"][active] >= maximum:
                state["gate"]["status"] = "needs_review"
                manifest["status"] = "needs_review"
                publish("needs_review.pt")
                break
            scheduler.set_stage(active, maximum * episode_macros)
            index = state["episodes"][active]
            seeds = episode_seeds(protocol, seed, index, active)
            realization = generate_realization(scene, seeds["layout_seed"], seeds["load_seed"])
            mdp.reset(seeds["arrival_seed"], layout_seed=seeds["layout_seed"], load_seed=seeds["load_seed"], realization=realization)
            audit_dir = output / "realizations"
            audit_dir.mkdir(exist_ok=True)
            realization_path = audit_dir / f"{active}-{index + 1:04d}.json"
            json_write(realization_path, realization)
            state["at_stream_boundary"] = False
            controllers = DeploymentLearningControllers(agent, mdp.encoder, deterministic=False, macro_enabled=active == "joint")
            lower = _route_controller(method, controllers, mdp, seeds["routing_seed"])
            pending_commands = []
            original_start, original_frame = mdp.start_macro, mdp.step_frame
            phase_totals = {name: dict(frames=0, cost_sum=0., holding_task_s=0., energy_j=0., routes=np.zeros((2, 3, 3)), cpu_cycles_sum=np.zeros((2, 3))) for name in ("adjusting", "holding")}

            def start_command(action):
                before = np.asarray(mdp.controller_information().base.positions_m).copy()
                command = original_start(action)
                pending_commands.append(dict(time_s=mdp.env.time_s, raw_action=np.asarray(action).tolist(),
                    start_positions_m=before.tolist(), command=_jsonable(command)))
                return command

            def step_frame(action):
                view = mdp.controller_information()
                phase_name = view.deployment_phase
                before_cost, before_holding, before_energy = mdp.env.cost_sum, mdp.env.holding, sum(mdp.env.energy.values())
                result = original_frame(action)
                target = phase_totals[phase_name]
                target["frames"] += 1
                target["cost_sum"] += mdp.env.cost_sum - before_cost
                target["holding_task_s"] += mdp.env.holding - before_holding
                target["energy_j"] += sum(mdp.env.energy.values()) - before_energy
                target["routes"] += result["route"]
                target["cpu_cycles_sum"] += np.asarray(view.base.compute_cycles)
                return result

            def complete_macro(transition):
                command = pending_commands.pop(0)
                if active == "pretrain" and not np.array_equal(np.asarray(command["start_positions_m"]), mdp.env.physics.uav_coords):
                    raise AssertionError("SAC pretraining physically moved a UAV")
                append_json(output / "motions.jsonl", dict(stage=active, episode=index + 1,
                    macro=scheduler.macros, reward=transition.reward, end_positions_m=mdp.env.physics.uav_coords.tolist(), **command))
                scheduler.macro_boundary(transition)

            mdp.start_macro, mdp.step_frame = start_command, step_frame
            rollout = IndustrialRollout(mdp, timed(controllers.macro, state["upper_inference"]),
                timed(lower, state["lower_inference"]), on_macro_boundary=complete_macro)
            previous, routes = {}, {"raw": [], "executed": []}
            discounted_macro_return = 0.
            try:
                for macro in range(episode_macros):
                    wall_remaining()
                    collected = rollout.run(1, truncate=macro + 1 == episode_macros,
                        collect_metrics=(macro + 1) % window_macros == 0 or macro + 1 == episode_macros)
                    scheduler.completed_micro(collected)
                    discounted_macro_return += mdp.gamma_macro ** macro * collected["macro"][0].reward
                    for transition in collected["micro"]:
                        routes["raw"].append(transition.action.ravel())
                        routes["executed"].append(transition.route.ravel())
                    if (macro + 1) % window_macros == 0:
                        current = collected["metrics"]
                        window = _window_row(previous, current, stage=active, stream=index,
                            global_seconds=state["completed_seconds"] + current["seconds"], routes=routes, objective=base["objective"])
                        window.update(episode=index + 1, committed=False, updates=counters(agent))
                        append_json(output / "windows.jsonl", window)
                        previous, routes = current, {"raw": [], "executed": []}
                        json_write(output / "progress.json", dict(status="running", stage=active, episodes=state["episodes"],
                            committed_episodes=state["completed_episodes"], committed_seconds=state["completed_seconds"],
                            simulated_seconds=state["completed_seconds"] + current["seconds"], latest_window=window,
                            updates=counters(agent), gate=state["gate"], elapsed_seconds=time.perf_counter() - started))
            finally:
                mdp.start_macro, mdp.step_frame = original_start, original_frame
            # collected metrics were frozen before tail priming; never re-read
            # environment.metrics() here for episode arrivals or unfinished work.
            metrics = collected["metrics"]
            state["episodes"][active] += 1
            state["completed_episodes"] += 1
            state["completed_seconds"] += episode_seconds
            state["at_stream_boundary"] = True
            _assert_counters(agent, scheduler, protocol, scene, method)
            if active == "pretrain" and ppo_fingerprint(agent) != state["pretrain_ppo_sha"]:
                raise AssertionError("Pretraining changed PPO parameters or exploration state")
            for physical, profile in enumerate(realization["profile_by_region"]):
                state["role_counts"][active][physical][profile] += 1
            priority = [{**row, "unfinished": row["arrived"] - row["completed"]} for row in metrics["priority"]]
            row = dict(stage=active, episode=index + 1, global_episode=state["completed_episodes"],
                seconds=episode_seconds, total_training_seconds=state["completed_seconds"], **seeds,
                realization_path=str(realization_path), realization_sha=digest(realization),
                profile_by_region=realization["profile_by_region"], metrics=metrics, priority=priority,
                physical_cost_sum=metrics["cost_sum"], undiscounted_physical_return=-metrics["cost_sum"],
                discounted_physical_return=discounted_macro_return,
                mean_weighted_unfinished_parents=metrics["holding_task_s"] / episode_seconds,
                mean_unfinished_parents=metrics["unfinished_parent_time_s"] / episode_seconds if "unfinished_parent_time_s" in metrics else None,
                phase_statistics={name: {key: _jsonable(value) for key, value in data.items()} for name, data in phase_totals.items()},
                updates=counters(agent), truncated=True, terminated=False)
            state["last_episode_record"] = copy.deepcopy(row)
            node = state["episodes"][active]
            candidate_node = (active == "pretrain" and node in budget["pretrain_check_episodes"]) or (active == "joint" and method == "ppo_sac" and node in budget["joint_development_check_episodes"])
            endpoint = None
            if candidate_node:
                endpoint = f"candidate-{active}-{node:04d}.pt"
                state["pending_evaluation"] = dict(kind="admission" if active == "pretrain" else "joint", episode=node,
                    candidate_path=str(output / "checkpoints" / endpoint), candidate_sha=None,
                    evaluation_output=str(output / "evaluations" / f"{active}-{node:04d}"))
            receipt = publish(endpoint)
            row["checkpoint_sha256"] = receipt["sha256"]
            row["checkpoint_path"] = receipt["persistent_path"] or receipt["path"]
            # A row is appended only after its full checkpoint is durable.
            append_json(output / "episodes.jsonl", row)
            json_write(output / "progress.json", dict(status="running", stage=active, episodes=state["episodes"],
                committed_episodes=state["completed_episodes"], committed_seconds=state["completed_seconds"],
                updates=counters(agent), latest_episode=row, gate=state["gate"], pending_evaluation=state["pending_evaluation"],
                checkpoint=receipt, elapsed_seconds=time.perf_counter() - started))
        manifest.update(run_state=state, updates=counters(agent), network_sha256=network_fingerprint(agent),
            elapsed_seconds=time.perf_counter() - started, completed_stage=state["stage"])
        json_write(output / "manifest.json", manifest)
        json_write(output / "progress.json", dict(status=manifest["status"], stage=state["stage"], episodes=state["episodes"],
            committed_episodes=state["completed_episodes"], committed_seconds=state["completed_seconds"],
            gate=state["gate"], updates=counters(agent), checkpoint=manifest.get("latest_checkpoint")))
        return manifest
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed",
            error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter() - started)
        json_write(output / "manifest.json", manifest)
        json_write(output / "failure.json", dict(error=manifest["error"],
            committed_episodes=manifest["committed_episodes"], resume="latest complete V4 checkpoint; pending evaluation precedes training"))
        raise


def resume_job(checkpoint, output, root, **kwargs):
    from .industrial_hotspot_checkpoint import load_checkpoint
    saved = load_checkpoint(checkpoint)
    identity = saved["identity"]
    return train_job(saved["protocol"], saved["scene"], identity["method"], identity["training_seed"],
        output, root, phase="resume", checkpoint=checkpoint, **kwargs)
