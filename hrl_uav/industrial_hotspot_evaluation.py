"""Frozen hotspot development runs, strict cache identities and durable budgets.

This module imports no torch until a checkpoint evaluation is explicitly called.
The training runner dispatches these functions in a separate process.
"""
import copy
from contextlib import contextmanager
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np

from .industrial_hotspot_gate import (TechnicalEvaluationError, digest,
    evaluate_admission, paired_rows, pooled_metrics, validate_metric_row)


class EvaluationBudgetExceeded(RuntimeError):
    pass


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class EvaluationBudget:
    """A killed attempt retains its whole reservation: undercounting is impossible.

    A normal/handled-failure completion releases only definitely unused steps.
    Persisted progress is informative; reservations remain charged until finish.
    A stale process lock is an explicit error, never silently stolen.
    """
    def __init__(self, path, cap, identity):
        self.path = Path(path).absolute()
        self.lock = self.path.with_name(self.path.name+".lock")
        if type(cap) is not int or cap <= 0:
            raise ValueError("Evaluation budget must be a positive physical-step count")
        self.cap, self.identity = cap, copy.deepcopy(identity)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if not self.path.exists():
                atomic_json(self.path, dict(schema="industrial_hotspot_evaluation_budget_v1",
                    cap=cap, identity=self.identity, charged_steps=0, attempts=[]))
            self._read()

    @contextmanager
    def _locked(self):
        try:
            descriptor = os.open(str(self.lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError("Evaluation budget is locked; do not concurrently run or steal a stale lock") from exc
        try:
            os.write(descriptor, str(os.getpid()).encode())
            os.close(descriptor)
            yield
        finally:
            if self.lock.exists():
                self.lock.unlink()

    def _read(self):
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if (value.get("schema") != "industrial_hotspot_evaluation_budget_v1" or
                value.get("cap") != self.cap or value.get("identity") != self.identity):
            raise ValueError("Evaluation budget contract changed")
        if value["charged_steps"] != sum(a["charged_steps"] for a in value["attempts"]):
            raise ValueError("Corrupt evaluation budget accounting")
        if not 0 <= value["charged_steps"] <= self.cap:
            raise ValueError("Invalid evaluation budget total")
        return value

    def begin(self, key, steps):
        if type(steps) is not int or steps <= 0:
            raise ValueError("Invalid trajectory reservation")
        with self._locked():
            value = self._read()
            if value["charged_steps"]+steps > self.cap:
                raise EvaluationBudgetExceeded("Insufficient development physical-step budget; needs_review")
            token = uuid.uuid4().hex
            value["attempts"].append(dict(attempt_id=token, cache_key=key,
                planned_steps=steps, charged_steps=steps, attempted_steps_recorded=0,
                status="reserved", pid=os.getpid()))
            value["charged_steps"] += steps
            atomic_json(self.path, value)
        return token

    def progress(self, token, attempted_steps):
        with self._locked():
            value = self._read()
            row = next(a for a in value["attempts"] if a["attempt_id"] == token)
            if row["status"] != "reserved" or not row["attempted_steps_recorded"] <= attempted_steps <= row["planned_steps"]:
                raise ValueError("Invalid attempt progress")
            row["attempted_steps_recorded"] = int(attempted_steps)
            atomic_json(self.path, value)

    def finish(self, token, attempted_steps, status, error=None):
        if status not in ("complete", "failed"):
            raise ValueError("Invalid attempt outcome")
        with self._locked():
            value = self._read()
            row = next(a for a in value["attempts"] if a["attempt_id"] == token)
            if row["status"] != "reserved" or not row["attempted_steps_recorded"] <= attempted_steps <= row["planned_steps"]:
                raise ValueError("Invalid attempt completion")
            if status == "complete" and attempted_steps != row["planned_steps"]:
                raise ValueError("A complete trajectory must consume its registered physical steps")
            value["charged_steps"] -= row["charged_steps"]-attempted_steps
            row.update(charged_steps=int(attempted_steps), attempted_steps_recorded=int(attempted_steps),
                status=status, error=error)
            atomic_json(self.path, value)
            return copy.deepcopy(row)

    def snapshot(self):
        with self._locked():
            return copy.deepcopy(self._read())


def _hash_value(h, value):
    """Deterministic typed hashing, including optimizer/replay tensors, without pickle."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        array = value.detach().cpu().contiguous().numpy()
        h.update(b"tensor")
        _hash_value(h, array)
    elif isinstance(value, np.ndarray):
        h.update(b"ndarray"+str(value.dtype).encode()+str(value.shape).encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        h.update(b"dict")
        for key in sorted(value, key=lambda k: (type(k).__name__, repr(k))):
            _hash_value(h, key)
            _hash_value(h, value[key])
    elif isinstance(value, (list, tuple)):
        h.update(type(value).__name__.encode()+str(len(value)).encode())
        for item in value:
            _hash_value(h, item)
    elif isinstance(value, np.generic):
        _hash_value(h, value.item())
    else:
        h.update((type(value).__name__+":"+repr(value)).encode("utf-8"))


def frozen_fingerprint(agent):
    from .industrial_deployment_checkpoint import capture_rng
    networks = (agent.macro_actor, agent.macro_critic, agent.sac_actor.action_net,
        agent.sac_critic.critic_v, agent.sac_critic.target_critic_v)
    optimizers = (agent.sac_actor.optimizer, agent.sac_critic.optimizer, agent.entropy.optimizer,
        agent.ppo_actor_optimizer, agent.ppo_critic_optimizer)
    h = hashlib.sha256()
    for network in networks:
        _hash_value(h, network.state_dict())
        _hash_value(h, network.training)
        _hash_value(h, [p.grad for p in network.parameters()])
    _hash_value(h, [o.state_dict() for o in optimizers])
    _hash_value(h, dict(log_alpha=agent.entropy.log_alpha, alpha=agent.entropy.alpha,
        stage=agent.stage, stage_initial_updates=agent.stage_initial_updates,
        sac_updates=agent.sac_updates, sac_actor_updates=agent.sac_actor_updates,
        sac_critic_gradient_steps=agent.sac_critic_gradient_steps, sac_target_updates=agent.sac_target_updates,
        ppo_updates=agent.ppo_updates, ppo_gradient_steps=agent.ppo_gradient_steps,
        macro_rollout=agent.macro_rollout, last_sac_stats=agent.last_sac_stats,
        last_ppo_stats=agent.last_ppo_stats, guard=agent.guard_state(), replay=agent.replay.state_dict(),
        global_rng=capture_rng()))
    return h.hexdigest()


def validate_panel(panel, *, development=True, research=True):
    scenarios = panel.get("scenarios", [])
    seconds = panel.get("seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid evaluation duration")
    if research and seconds != 900:
        raise ValueError("Research evaluation uses sealed 900-second episodes")
    if development and len(scenarios) != 6:
        raise ValueError("Development panel must contain exactly six scenarios")
    if len({s["scenario_id"] for s in scenarios}) != len(scenarios) or not scenarios:
        raise ValueError("Duplicate or absent evaluation scenarios")
    for scenario in scenarios:
        for key in ("layout_seed", "load_seed", "arrival_seed", "routing_seed"):
            if type(scenario.get(key)) is not int or scenario[key] < 0:
                raise ValueError("Invalid registered evaluation seed: "+key)
        if sorted(scenario.get("profile_permutation", [])) != [0, 1, 2]:
            raise ValueError("Evaluation business roles must be explicitly sealed")
    if development and {tuple(s["profile_permutation"]) for s in scenarios} != set(itertools.permutations(range(3))):
        raise ValueError("Development panel must cover each business-role assignment once")
    return copy.deepcopy(panel)


class CausalReferenceDeployment:
    def __init__(self, *, report_seconds=10., step_m=10., tolerance_m=.001, weights=(10., 1.)):
        self.report_seconds, self.step_m, self.tolerance_m = report_seconds, step_m, tolerance_m
        self.weights = np.asarray(weights)
        self.target = None
        self.target_time_s = None

    def __call__(self, wrapper):
        view = wrapper.base
        if view.time_s < self.report_seconds-1e-9:
            return np.zeros(6)
        if self.target is None:
            targets = []
            for source in range(3):
                terminals = [t for t in view.terminals if t.source == source]
                points = np.asarray([t.position_m for t in terminals])
                mass = np.asarray([np.asarray(t.recent_bits) @ self.weights for t in terminals])
                targets.append((points*mass[:, None]).sum(axis=0)/mass.sum() if mass.sum() > 0 else points.mean(axis=0))
            self.target = np.asarray(targets)
            self.target_time_s = float(view.time_s)
        delta = self.target-np.asarray(view.positions_m)
        distance = np.linalg.norm(delta, axis=1)
        delta[distance <= self.tolerance_m] = 0.
        return (delta/np.maximum(distance[:, None], self.step_m)).ravel()


def _cache_identity(context, scenario, variant, candidate_sha):
    return dict(context, scenario_sha=digest(scenario), variant=variant,
        candidate_sha=candidate_sha if variant in ("ppo_sac", "hover_sac", "ppo_cvx", "ppo_random", "ppo_local") else None)


def _load_cache(path, identity):
    if not path.exists():
        return None
    cached = json.loads(path.read_text(encoding="utf-8"))
    if cached.get("identity") != identity or cached.get("row_sha256") != digest(cached.get("row")):
        raise TechnicalEvaluationError("Cached trajectory identity or digest mismatch")
    validate_metric_row(cached["row"])
    return cached["row"]


def _source_hashes(root):
    from .industrial_hotspot_experiment import source_hashes
    return source_hashes(root)


def _context(protocol, scene, panel, sources):
    return dict(protocol_sha256=digest(protocol), scene_contract_sha=digest(scene),
        dev_manifest_sha=digest(panel), admission_rule_version=protocol["admission"]["version"],
        evaluator_source_sha=digest(sources), seconds=panel["seconds"])


def _ledger(protocol, scene, root, panel, budget_path, *, final=False):
    cap_key = "reserved_final_test_physical_steps" if final else "first_batch_total_physical_steps_max"
    identity = dict(protocol_sha256=digest(protocol), scene_sha256=digest(scene),
        panel_sha256=digest(panel), evaluator_source_sha=digest(_source_hashes(root)),
        scope="final" if final else "development")
    return EvaluationBudget(budget_path, int(protocol["evaluation_budget"][cap_key]), identity)


def _run_trajectory(protocol, scene, scenario, variant, seconds, agent, budget, key, root, sources):
    from .industrial_hotspot_mdp import HotspotIndustrialMDP
    from .industrial_hotspot_scene import generate_realization
    from .industrial_baselines import CVXRouting, RandomRouting, local_scores
    from .industrial_deployment_training import DeploymentLearningControllers
    mdp = HotspotIndustrialMDP(scene)
    realization = generate_realization(scene, scenario["layout_seed"], scenario["load_seed"],
        profile_permutation=scenario["profile_permutation"])
    mdp.reset(scenario["arrival_seed"], scenario["layout_seed"], scenario["load_seed"], realization)
    dt, macro = mdp.env.dt, scene["deployment"]["macro_period_s"]
    if not math.isclose(seconds/dt, round(seconds/dt)) or not math.isclose(seconds/macro, round(seconds/macro)):
        raise ValueError("Evaluation duration must contain complete macros")
    planned = round(seconds/dt)
    if variant in ("ppo_sac", "hover_sac", "ppo_cvx", "ppo_random", "ppo_local") and agent is None:
        raise ValueError("A learned upper or lower controller requires its checkpoint")
    controllers = DeploymentLearningControllers(agent, mdp.encoder, deterministic=True,
        macro_enabled=variant != "hover_sac") if agent is not None else None
    reference = None
    upper = (lambda view: np.zeros(6))
    if variant.startswith("ppo_"):
        upper = controllers.macro
    elif variant == "reference_cvx":
        settings = protocol["opportunity_diagnostic"]
        reference = CausalReferenceDeployment(report_seconds=settings["report_collection_seconds"],
            step_m=scene["deployment"]["max_target_step_m"],
            tolerance_m=settings["reference_target_tolerance_m"],
            weights=scene["base_scene"]["objective"]["priority_weights"])
        upper = reference
    cvx = None
    if variant in ("ppo_sac", "hover_sac"):
        lower = controllers.micro
    elif variant in ("hover_cvx", "reference_cvx", "ppo_cvx"):
        cvx = CVXRouting(mdp.config)
        lower = lambda view: cvx(view.base)
    elif variant in ("hover_local", "ppo_local"):
        lower = lambda view: local_scores(view.base)
    elif variant == "ppo_random":
        random_rule = RandomRouting(scenario["routing_seed"])
        lower = lambda view: random_rule(view.base)
    else:
        raise ValueError("Unknown frozen evaluation variant: "+variant)
    before = frozen_fingerprint(agent) if agent is not None else None
    attempt = budget.begin(key, planned)
    attempted = 0
    original_step = mdp.env.micro_step
    def counted_step(route):
        nonlocal attempted
        if attempted >= planned:
            raise AssertionError("Evaluation exceeded its physical-step reservation")
        attempted += 1
        return original_step(route)
    mdp.env.micro_step = counted_step
    started = time.perf_counter()
    curve, commands, raw_routes, routes = [], [], [], []
    discounted_cost = 0.
    try:
        for _ in range(round(seconds/macro)):
            if time.perf_counter()-started > protocol["runtime"]["max_wall_seconds_per_job"]:
                raise TimeoutError("Frozen trajectory wall limit reached")
            result = upper(mdp.controller_information())
            action = result[0] if isinstance(result, tuple) else result
            action = np.asarray(action, dtype=float)
            if action.size != 6 or not np.isfinite(action).all():
                raise TechnicalEvaluationError("Invalid upper action")
            movement = mdp.start_macro(action)
            if movement["minimum_separation_m"] < mdp.config["motion"]["min_separation_m"]-1e-7:
                raise TechnicalEvaluationError("Unsafe executed deployment")
            commands.append(dict(time_s=mdp.env.time_s, raw_action=action.tolist(),
                target_m=np.asarray(movement["target_m"]).tolist(),
                displacement_m=np.asarray(movement["displacement_m"]).tolist(),
                safety_scale=float(movement["safety_scale"])))
            for _ in range(mdp.frames_per_macro):
                mdp.prepare_service()
                action = np.asarray(lower(mdp.controller_information()), dtype=float)
                if action.size != 18 or not np.isfinite(action).all():
                    raise TechnicalEvaluationError("Invalid lower action")
                start_step = mdp.env.step_index
                reading = mdp.step_frame(action)
                discounted_cost += sum(mdp.gamma_step**(start_step+j)*row["cost"]
                    for j, row in enumerate(reading["readings"]))
                raw_routes.append(action.ravel())
                routes.append(reading["route"].ravel())
            # No next-boundary priming: the t=900 snapshot excludes t=900 arrivals.
            metrics = copy.deepcopy(mdp.metrics())
            curve.append(dict(time_s=mdp.env.time_s, cost_sum=metrics["cost_sum"],
                discounted_cost=discounted_cost, positions_m=mdp.env.physics.uav_coords.tolist(),
                unfinished=[p["arrived"]-p["completed"] for p in metrics["priority"]]))
            budget.progress(attempt, attempted)
        if attempted != planned or mdp.env.step_index != planned:
            raise AssertionError("Evaluation physical-step count mismatch")
        scale_bits, scale_cycles = max(1., mdp.env.generated_bits), max(1., mdp.env.generated_cycles)
        tolerances = {name: 1e-5+1e-9*(scale_cycles if name.endswith("cycles") else scale_bits)
            for name in metrics["conservation"]}
        if any(not math.isfinite(value) or abs(value) > tolerances[name]
                for name, value in metrics["conservation"].items()):
            raise TechnicalEvaluationError("Task conservation failed")
        if agent is not None and frozen_fingerprint(agent) != before:
            raise TechnicalEvaluationError("Frozen evaluation changed learning state or RNG")
        if _source_hashes(root) != sources:
            raise TechnicalEvaluationError("Evaluation sources changed during the trajectory")
        objective = mdp.config["objective"]
        row = dict(status="complete", technical_valid=True, frozen_verified=True,
            metrics_frozen_before_priming=True, scenario_id=scenario["scenario_id"], scenario=copy.deepcopy(scenario),
            scenario_sha=digest(scenario), realization_sha=realization["realization_sha256"],
            profile_permutation=realization["profile_by_region"], business_roles_by_physical_uav=realization["profile_by_region"],
            event_sha256=metrics["event_sha256"], canonical_event_sha256=metrics["canonical_event_sha256"],
            variant=variant, metrics=metrics, low_unfinished=metrics["priority"][1]["arrived"]-metrics["priority"][1]["completed"],
            discounted_cost=discounted_cost, physical_return=-metrics["cost_sum"],
            holding_cost=objective["holding_weight"]*metrics["holding_task_s"]/objective["holding_reference_task_s"],
            energy_cost=(1-objective["holding_weight"])*sum(metrics["energy_j"].values())/objective["energy_reference_j"],
            conservation_tolerances=tolerances, frozen_fingerprint=before,
            learning_updates=0, replay_writes=0, replay_samples=0,
            curve=curve, commands=commands, realization=realization,
            raw_route_mean=np.asarray(raw_routes).mean(axis=0).reshape(2,3,3).tolist(),
            executed_route_mean=np.asarray(routes).mean(axis=0).reshape(2,3,3).tolist(),
            cvx_fallbacks=sum(r["fallback"] for r in cvx.records) if cvx is not None else 0)
        if reference is not None:
            row["reference"] = dict(target_m=reference.target.tolist() if reference.target is not None else None,
                target_time_s=reference.target_time_s, reached_tolerance_m=reference.tolerance_m,
                residual_m=(np.linalg.norm(reference.target-mdp.env.physics.uav_coords, axis=1).tolist()
                    if reference.target is not None else None), target_uses_legal_reports_only=True)
        validate_metric_row(row)
        row["budget_attempt"] = budget.finish(attempt, attempted, "complete")
        return row
    except BaseException as error:
        budget.finish(attempt, attempted, "failed", f"{type(error).__name__}: {error}")
        raise
    finally:
        mdp.env.micro_step = original_step


def _rows(protocol, scene, panel, variant, agent, candidate_sha, context, cache_root, budget, root, sources):
    rows = []
    directory = Path(cache_root)/"trajectories"
    directory.mkdir(parents=True, exist_ok=True)
    for scenario in panel["scenarios"]:
        identity = _cache_identity(context, scenario, variant, candidate_sha)
        key = digest(identity)
        path = directory/(key+".json")
        row = _load_cache(path, identity)
        if row is None:
            row = _run_trajectory(protocol, scene, scenario, variant, panel["seconds"], agent,
                budget, key, root, sources)
            atomic_json(path, dict(schema="industrial_hotspot_frozen_cache_v1", identity=identity,
                row_sha256=digest(row), row=row))
        rows.append(row)
    return rows


def _receipt(report_path):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return dict(report, report_path=str(Path(report_path).resolve()), evaluation_file_sha=file_sha(report_path))


def _prepare_report(output, identity):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/"report.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        for key, value in identity.items():
            if old.get(key) != value:
                raise TechnicalEvaluationError("Output directory belongs to another evaluation: "+key)
        if old.get("status") in ("complete", "spatial_benefit_unconfirmed"):
            return path, _receipt(path)
    return path, None


def _load_agent(checkpoint, root):
    from .industrial_hotspot_checkpoint import load_checkpoint, restore_checkpoint
    from .industrial_hotspot_experiment import make_agent
    from .industrial_hotspot_mdp import HotspotIndustrialMDP
    from .industrial_experiment import runtime_setup
    saved = load_checkpoint(checkpoint)
    protocol, scene, identity = saved["protocol"], saved["scene"], saved["identity"]
    sources = _source_hashes(root)
    if identity["source_sha256"] != sources:
        raise TechnicalEvaluationError("Checkpoint sources differ from the evaluation sources")
    runtime_setup(protocol, identity["training_seed"])
    agent = make_agent(protocol, HotspotIndustrialMDP(scene), identity["training_seed"], identity["method"])
    restore_checkpoint(agent, saved, protocol=protocol, scene=scene, identity=identity, restore_random=True)
    metadata = saved.get("_load_metadata", {})
    checkpoint_sha = metadata.get("sha256", metadata.get("checkpoint_sha256"))
    if checkpoint_sha is None:
        source_path = metadata.get("path", metadata.get("checkpoint_path", checkpoint))
        if str(source_path).endswith(".json"):
            raise ValueError("Checkpoint loader must return the resolved payload digest for an index")
        checkpoint_sha = file_sha(source_path)
    return saved, agent, checkpoint_sha, sources


def candidate_evaluate(checkpoint, output, root, cache_root, budget_path):
    saved, agent, candidate_sha, sources = _load_agent(checkpoint, root)
    protocol, scene = saved["protocol"], saved["scene"]
    if saved["identity"]["method"] != "ppo_sac" or saved["stage"] != "pretrain":
        raise ValueError("Admission evaluates a SAC pretraining candidate only")
    panel = validate_panel(protocol["development_manifest"], research=protocol["purpose"] == "research")
    context = _context(protocol, scene, panel, sources)
    identity = {key: context[key] for key in ("scene_contract_sha", "dev_manifest_sha", "admission_rule_version", "evaluator_source_sha")}
    identity["candidate_sha"] = candidate_sha
    report = dict(schema="industrial_hotspot_candidate_evaluation_v1", status="running",
        candidate_sha=candidate_sha, identity=identity, source_sha256=sources,
        protocol_sha256=digest(protocol), scene_sha256=digest(scene), dev_manifest_sha=digest(panel))
    path, existing = _prepare_report(output, {key: report[key] for key in report if key != "status"})
    if existing is not None:
        return existing
    atomic_json(path, report)
    budget = _ledger(protocol, scene, root, panel, budget_path)
    try:
        report["local_rows"] = _rows(protocol, scene, panel, "hover_local", None, None, context, cache_root, budget, root, sources)
        report["cvx_rows"] = _rows(protocol, scene, panel, "hover_cvx", None, None, context, cache_root, budget, root, sources)
        report["sac_rows"] = _rows(protocol, scene, panel, "hover_sac", agent, candidate_sha, context, cache_root, budget, root, sources)
        measured = evaluate_admission(report["sac_rows"], report["local_rows"], report["cvx_rows"], protocol["admission"])
        report.update(status="complete", quality_passed=measured["quality_passed"], gate_result=measured,
            budget=budget.snapshot(), learning_updates=0, replay_writes=0, replay_samples=0)
        atomic_json(path, report)
        return _receipt(path)
    except EvaluationBudgetExceeded as error:
        report.update(status="needs_review", quality_passed=False, error=str(error), budget=budget.snapshot())
        atomic_json(path, report)
        return _receipt(path)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", budget=budget.snapshot())
        atomic_json(path, report)
        raise


def paired_evaluate(checkpoint, output, root, cache_root, budget_path, panel="development"):
    saved, agent, candidate_sha, sources = _load_agent(checkpoint, root)
    protocol, scene = saved["protocol"], saved["scene"]
    if saved["stage"] != "joint" or panel not in ("development", "final"):
        raise ValueError("Paired evaluations require a joint-stage endpoint and explicit panel")
    final = panel == "final"
    manifest = validate_panel(protocol["final_test_manifest" if final else "development_manifest"],
        development=not final, research=protocol["purpose"] == "research")
    context = _context(protocol, scene, manifest, sources)
    report = dict(schema="industrial_hotspot_paired_evaluation_v1", status="running",
        candidate_sha=candidate_sha, source_sha256=sources, protocol_sha256=digest(protocol),
        scene_sha256=digest(scene), dev_manifest_sha=digest(manifest), panel=panel)
    path, existing = _prepare_report(output, {key: report[key] for key in report if key != "status"})
    if existing is not None:
        return existing
    atomic_json(path, report)
    budget = _ledger(protocol, scene, root, manifest, budget_path, final=final)
    try:
        method = saved["identity"]["method"]
        moving = _rows(protocol, scene, manifest, method, agent, candidate_sha, context, cache_root, budget, root, sources)
        report.update(moving_rows=moving, moving_pooled=pooled_metrics(moving))
        if method == "ppo_sac":
            hover = _rows(protocol, scene, manifest, "hover_sac", agent, candidate_sha, context, cache_root, budget, root, sources)
            pairs = paired_rows(moving, hover)
            report.update(hover_rows=hover, hover_pooled=pooled_metrics(hover), same_frozen_sac=True,
                paired_cost_differences=[dict(scenario_id=a["scenario_id"], moving_minus_hover=
                    a["metrics"]["cost_sum"]-b["metrics"]["cost_sum"]) for a,b in pairs])
        report.update(status="complete", budget=budget.snapshot(), learning_updates=0, replay_writes=0, replay_samples=0)
        atomic_json(path, report)
        return _receipt(path)
    except EvaluationBudgetExceeded as error:
        report.update(status="needs_review", error=str(error), budget=budget.snapshot())
        atomic_json(path, report)
        return _receipt(path)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", budget=budget.snapshot())
        atomic_json(path, report)
        raise


def diagnose(protocol, scene, output, root, cache_root, budget_path):
    panel = validate_panel(protocol["development_manifest"], research=protocol["purpose"] == "research")
    sources = _source_hashes(root)
    context = _context(protocol, scene, panel, sources)
    report = dict(schema="industrial_hotspot_spatial_diagnostic_v1", status="running", passed=False,
        source_sha256=sources, protocol_sha256=digest(protocol), scene_sha256=digest(scene),
        dev_manifest_sha=digest(panel))
    path, existing = _prepare_report(output, {key: report[key] for key in report if key not in ("status", "passed")})
    if existing is not None:
        return existing
    atomic_json(path, report)
    budget = _ledger(protocol, scene, root, panel, budget_path)
    try:
        hover = _rows(protocol, scene, panel, "hover_cvx", None, None, context, cache_root, budget, root, sources)
        moving = _rows(protocol, scene, panel, "reference_cvx", None, None, context, cache_root, budget, root, sources)
        pairs = paired_rows(moving, hover)
        cfg = protocol["opportunity_diagnostic"]
        qualified = int(sum(bool(a["metrics"]["cost_sum"] <= cfg["per_scene_cost_ratio_max"]*b["metrics"]["cost_sum"]) for a,b in pairs))
        cost_a = float(sum(float(a["metrics"]["cost_sum"]) for a,b in pairs))
        cost_b = float(sum(float(b["metrics"]["cost_sum"]) for a,b in pairs))
        passed = bool(cost_a < cost_b and qualified >= int(cfg["per_scene_required_count"]))
        comparisons = []
        for a,b in pairs:
            if [x["time_s"] for x in a["curve"]] != [x["time_s"] for x in b["curve"]]:
                raise TechnicalEvaluationError("Paired diagnostic sample times differ")
            first = next((x["time_s"] for x,y in zip(a["curve"],b["curve"]) if x["cost_sum"] < y["cost_sum"]-1e-9), None)
            comparisons.append(dict(scenario_id=a["scenario_id"], reference_minus_hover=a["metrics"]["cost_sum"]-b["metrics"]["cost_sum"],
                discounted_reference_minus_hover=a["discounted_cost"]-b["discounted_cost"],
                first_positive_cumulative_benefit_sample_s=first, sampling_resolution_s=scene["deployment"]["macro_period_s"]))
        report.update(status="complete" if passed else "spatial_benefit_unconfirmed", passed=passed,
            hover_rows=hover, reference_rows=moving, hover_pooled=pooled_metrics(hover),
            reference_pooled=pooled_metrics(moving), qualifying_scenes=qualified,
            total_cost_benefit=cost_b-cost_a, comparisons=comparisons, budget=budget.snapshot(),
            learning_updates=0, replay_writes=0, replay_samples=0)
        atomic_json(path, report)
        return _receipt(path)
    except EvaluationBudgetExceeded as error:
        report.update(status="needs_review", error=str(error), budget=budget.snapshot())
        atomic_json(path, report)
        return _receipt(path)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", budget=budget.snapshot())
        atomic_json(path, report)
        raise


def final_evaluate(checkpoint, output, root, cache_root, budget_path):
    """Explicit final-test entry; never dispatched automatically by admission."""
    from .industrial_hotspot_checkpoint import load_checkpoint
    saved = load_checkpoint(checkpoint)
    state = saved["run_state"]
    # The public final-test entry accepts only the registered joint endpoint.
    completed = state.get("episodes", state.get("stage_completed_episodes", {}))
    if isinstance(completed, dict):
        completed = completed.get("joint")
    if completed != saved["protocol"]["budget"]["joint_episodes"]:
        raise ValueError("Final testing requires the complete registered joint budget")
    return paired_evaluate(checkpoint, output, root, cache_root, budget_path, panel="final")
