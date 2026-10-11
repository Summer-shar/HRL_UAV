"""Bounded causal diagnostics for motion and splitting; no learning or tuning.

python -m hrl_v2.control_diagnostics --output-dir outputs/control_diagnostics
All workload tapes and controls are fixed before the paired runs. Production
resource parameters, physical transition code and reward remain unchanged.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np

from hrl_uav.industrial_env import IndustrialEnv
from hrl_uav.industrial_scene import Arrival
from hrl_v2.collector import collect_metrics
from hrl_v2.hybrid_simplex import SUPPORTS
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.timing import ClockConfig


SCENE_PATH = Path(__file__).parent / "configs/physical_scene.json"
SCENE = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
SECONDS = 60
LIMIT = 18000  # Together with the previous 1,451 steps, still below 20,000.
LAYOUTS = ((92, 93), (102, 103))
PATTERNS = ("staggered_busy", "synchronous_busy")
MOTIONS = ("hover", "toward", "away")
ROUTES = ("local", "equal")


def write_json(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Budget:
    """Reserve a whole case before running; a crash cannot refund its budget."""
    def __init__(self, directory, *, resume=False):
        self.path = directory / "budget.json"
        if self.path.exists() and not resume:
            raise RuntimeError("Existing diagnostic evidence is immutable; inspect it before rerunning")
        self.records = json.loads(self.path.read_text())["cases"] if resume else []
        if any(row["status"] != "passed" for row in self.records):
            raise RuntimeError("Incomplete previous case requires budget review before continuation")
        self.current = None

    @contextmanager
    def case(self, label, steps):
        if any(row["label"] == label for row in self.records):
            raise RuntimeError("A diagnostic case must not be silently overwritten or repeated")
        if self.current is not None or sum(r["reserved_steps"] for r in self.records) + steps > LIMIT:
            raise RuntimeError("Physical step reservation exceeds diagnostic limit")
        row = dict(label=label, reserved_steps=steps, physical_steps=0, status="running")
        self.records.append(row)
        self.current = row
        write_json(self.path, dict(limit=LIMIT, cases=self.records))
        try:
            yield
            if row["physical_steps"] != steps:
                raise AssertionError("Unexpected physical step count")
            row["status"] = "passed"
        except BaseException:
            row["status"] = "failed"
            raise
        finally:
            self.current = None
            write_json(self.path, dict(limit=LIMIT, cases=self.records,
                physical_steps=sum(r["physical_steps"] for r in self.records)))

    def wrap(self, original):
        def counted(env, route):
            if self.current is None or self.current["physical_steps"] >= self.current["reserved_steps"]:
                raise RuntimeError("Physical service outside its reserved diagnostic case")
            before = env.step_index
            try:
                return original(env, route)
            finally:
                self.current["physical_steps"] += env.step_index - before
        return counted


class Tape:
    def __init__(self, scene, pattern):
        self.events = []
        self.by_frame = {}
        self.calls = []
        # Deterministic fractional counters reproduce configured mean rates
        # and high-priority fractions without any stochastic arrival sampling.
        counts = np.asarray([(user * 17 % 39) / 39 for user in range(39)], float)
        high = np.asarray([(user * 11 % 39) / 39 for user in range(39)], float)
        dt = scene.config["time"]["routing_s"]
        for frame in range(round(SECONDS / dt)):
            now = frame * dt
            for user, terminal in enumerate(scene.terminals):
                profile = scene._profiles[user]
                _, slot = scene._logical[user]
                parameters = profile["slots"][slot]["parameters"]
                busy = (20 <= now < 40) if pattern == "synchronous_busy" else (
                    20 * terminal.source <= now < 20 * (terminal.source + 1))
                activity = profile["activity_parameters"]["busy_multiplier" if busy else "normal_multiplier"]
                counts[user] += parameters["rate_per_s"] * profile["business_multiplier"] * activity * dt
                while counts[user] >= 1.:
                    counts[user] -= 1.
                    high[user] += parameters["high_probability"]
                    priority = 0 if high[user] >= 1. else 1
                    if priority == 0:
                        high[user] -= 1.
                    bits = float(parameters["input_bits"][priority])
                    arrival = Arrival(len(self.events), user, terminal.source, terminal.task_type,
                        priority, now, bits, bits * parameters["cycles_per_bit"][priority])
                    self.events.append(arrival)
                    self.by_frame.setdefault(frame, []).append(arrival)
        self.sha256 = digest([asdict(a) for a in self.events])

    def __call__(self, frame):
        if frame != len(self.calls):
            raise AssertionError("Arrival preparation duplicated or skipped")
        self.calls.append(frame)
        return tuple(self.by_frame.get(frame, ()))


class Script:
    def __init__(self, report, motion, route, *, deployment):
        self.motion, self.route = motion, route
        self.max_target_step_m = deployment["max_target_step_m"]
        self.starts = np.asarray([u.position for u in report.uavs])
        self.centroids = np.asarray([np.mean([t.position for t in report.terminals if t.source == s], axis=0)
                                     for s in range(3)])
        self.delta = self.centroids - self.starts
        self.distance = np.linalg.norm(self.delta, axis=1)
        self.heading = np.divide(self.delta, self.distance[:, None], out=np.zeros((3, 2)),
                                 where=self.distance[:, None] > 0)
        self.upper_requests = []

    def upper(self, report, history):
        macro = report.tick // 50
        length = np.clip(self.distance - self.max_target_step_m * macro, 0., self.max_target_step_m)
        sign = {"hover": 0., "toward": 1., "away": -1.}[self.motion]
        request = sign * self.heading * length[:, None] / self.max_target_step_m
        self.upper_requests.append([report.tick, request.tolist()])
        return request.ravel(), {}

    def lower(self, report, history):
        if self.route == "local":
            route = np.tile(np.eye(3), (2, 1, 1))
        elif self.route == "equal":
            route = np.full((2, 3, 3), 1/3)
        elif self.route == "high_local_low_equal":
            route = np.full((2, 3, 3), 1/3)
            route[0] = np.eye(3)
        else:
            raise ValueError("Unknown fixed routing script")
        support = [SUPPORTS.index(tuple(np.flatnonzero(row))) for row in route.reshape(6, 3)]
        return route.astype(np.float64), np.asarray(support)


class Probe(PhysicsAdapter):
    def __init__(self, layout, load, pattern):
        super().__init__(SCENE, ledger=True, timing=ClockConfig.from_scene(SCENE, SECONDS))
        self.reset(arrival_seed=91, layout_seed=layout, load_seed=load)
        self.tape = Tape(self.env._scene, pattern)
        self.env._scene.frame = self.tape
        self.stages = dict(upload=0., ready=0., has_transfer=0., cpu_only=0.)
        self.tail_holding = 0.
        self.series = []
        self.speed_norms = []
        self.motion_records = []
        self.active_upload_rate_sum = self.active_upload_samples = 0.
        self.initial_rates = self.env.physics.compute_u2g_rate_bps().tolist()

    def start_macro(self, request):
        result = super().start_macro(request)
        self.motion_records.append(dict(tick=self.env.step_index,
            displacement=np.asarray(result["displacement_m"]).tolist(),
            safety_scale=result["safety_scale"]))
        return result

    def step(self, route):
        env = self.env
        for parent in env._active_parents.values():
            weight = (10 if parent.arrival.priority == 0 else 1) * env.dt
            children = [env.fragments[fid] for fid in parent.fragment_ids]
            stage = "upload" if parent.upload_remaining > 0 else "ready" if parent.route_time_s is None else (
                "has_transfer" if any(f.stage == "u2u" for f in children) else "cpu_only")
            self.stages[stage] += weight
            if any(f.finish_s is not None for f in children):
                self.tail_holding += weight
        rates = env.physics.compute_u2g_rate_bps()
        active = np.asarray([bool(queue) for queue in env.upload])
        self.active_upload_rate_sum += float(rates[active].sum())
        self.active_upload_samples += int(active.sum())
        self.speed_norms.append(np.linalg.norm(env.velocity, axis=1).tolist())
        result = super().step(route)
        if env.step_index % 50 == 0:
            self.series.append(dict(time_s=env.time_s, cost=env.cost_sum, holding=env.holding,
                energy=dict(env.energy), unfinished=len(env._active_parents),
                upload_bits=sum(p.upload_remaining for p in env._active_parents.values()),
                cpu_cycles=sum(f.compute_remaining for f in env._active_fragments.values() if f.stage == "cpu"),
                transfer_bits=sum(f.transfer_remaining for f in env._active_fragments.values()),
                positions=env.physics.uav_coords.tolist()))
        return result


def run_case(layout, load, pattern, motion, route):
    adapter = Probe(layout, load, pattern)
    policy = Script(adapter.report(), motion, route, deployment=SCENE["deployment"])
    metrics = collect_metrics(adapter, policy, identity={})
    env = adapter.env
    # Independently integrate parent residence; no call to a cost function.
    independent_holding = sum((10 if a.priority == 0 else 1) *
        ((env.parents[a.task_id].finish_s or SECONDS) - a.birth_s) for a in adapter.tape.events)
    np.testing.assert_allclose(env.holding, independent_holding, rtol=1e-11, atol=1e-8)
    np.testing.assert_allclose(sum(adapter.stages.values()), env.holding, rtol=1e-11, atol=1e-8)
    assert adapter.tape.calls == list(range(121))
    assert len(env.parents) == len(adapter.tape.events)
    assert [p.arrival for p in env.parents.values()] == adapter.tape.events
    conservation = env.conservation()
    for name, error in conservation.items():
        scale = env.generated_cycles if "cycles" in name else env.generated_bits
        assert abs(error) <= max(1e-8, scale * 1e-11), (name, error)
    completed = [p for p in env.parents.values() if p.finish_s is not None]
    uploaded = [p for p in env.parents.values() if p.upload_finish_s is not None]
    split_completed = [p for p in completed if len(p.fragment_ids) > 1]
    tail_delays = [max(env.fragments[f].finish_s for f in p.fragment_ids) -
                   min(env.fragments[f].finish_s for f in p.fragment_ids) for p in split_completed]
    cpu_service = [sum(row["cycles"] for row in env.ledger if row["kind"] == "cpu_service"
                       and env.fragments[row["fragment_id"]].destination == node) for node in range(3)]
    components = dict(holding=.09 * env.holding / SECONDS,
        **{name: value / 140 / SECONDS for name, value in env.energy.items()})
    np.testing.assert_allclose(sum(components.values()), metrics["cost_per_s"], rtol=1e-12)
    result = dict(layout_seed=layout, load_seed=load, pattern=pattern, motion=motion, route=route,
        seconds=SECONDS, physical_steps=env.step_index, arrivals=len(env.parents),
        tape_sha256=adapter.tape.sha256, generated_bits=env.generated_bits, generated_cycles=env.generated_cycles,
        cost_per_s=metrics["cost_per_s"], cost_components_per_s=components,
        stage_holding_task_s=adapter.stages, tail_holding_task_s=adapter.tail_holding,
        holding_task_s=env.holding, holding_oracle_error=env.holding-independent_holding,
        energy_j=env.energy, completed=len(completed), unfinished=len(env._active_parents),
        priority=[{k: row[k] for k in ("arrived", "completed", "unfinished", "mean_delay_ms", "oldest_unfinished_age_s")}
                  for row in metrics["priority"]],
        cutoff_backlog=metrics["cutoff_backlog"],
        cpu_utilization=[cycles / (SECONDS * hz) for cycles, hz in zip(cpu_service, env.config["resources"]["cpu_hz"])],
        uploaded=len(uploaded), mean_completed_upload_delay_s=float(np.mean([p.upload_finish_s-p.arrival.birth_s for p in uploaded])),
        mean_split_tail_gap_s=float(np.mean(tail_delays)) if tail_delays else None,
        final_upload_rates_bps=env.physics.compute_u2g_rate_bps().tolist(), initial_upload_rates_bps=adapter.initial_rates,
        active_upload_mean_rate_bps=adapter.active_upload_rate_sum/adapter.active_upload_samples,
        upload_served_bits=env.upload_served_bits, cpu_served_cycles=env.cpu_served_cycles, tx_served_bits=env.tx_served_bits,
        initial_positions=policy.starts.tolist(), centroids=policy.centroids.tolist(),
        final_positions=env.physics.uav_coords.tolist(),
        motion_records=adapter.motion_records, time_series=adapter.series,
        conservation=conservation, max_speed_mps=float(np.max(adapter.speed_norms)))
    return result, adapter.speed_norms


def verify_pairs(results, speeds):
    checks = []
    for layout, load in LAYOUTS:
        for pattern in PATTERNS:
            group = [r for r in results if r["layout_seed"] == layout and r["pattern"] == pattern]
            assert len({r["tape_sha256"] for r in group}) == 1
            for route in ROUTES:
                toward = next(r for r in group if r["motion"] == "toward" and r["route"] == route)
                away = next(r for r in group if r["motion"] == "away" and r["route"] == route)
                a = speeds[(layout, pattern, "toward", route)]
                b = speeds[(layout, pattern, "away", route)]
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)
                np.testing.assert_allclose(toward["energy_j"]["propulsion"], away["energy_j"]["propulsion"], rtol=1e-12)
                assert all(r["safety_scale"] == 1 for r in toward["motion_records"] + away["motion_records"])
                checks.append(dict(layout_seed=layout, pattern=pattern, route=route,
                    same_arrivals=True, matched_speed=True, same_propulsion_energy=True, safety_projection_inactive=True))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--priority-followup", action="store_true",
        help="Append the four fixed high-local/low-equal hypothesis checks to a completed panel")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.priority_followup:
        priority_followup(args.output_dir)
        return
    budget = Budget(args.output_dir)
    plan = dict(layouts=LAYOUTS, patterns=PATTERNS, motions=MOTIONS, routes=ROUTES,
        seconds=SECONDS, planned_panel_steps=14400, planned_mechanism_steps=450,
        limit=LIMIT, scene_sha256=hashlib.sha256(SCENE_PATH.read_bytes()).hexdigest(),
        no_training=True, unchanged_physical_parameters=True,
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note="Fixed scripted mechanism comparisons, not a parameter search or stochastic performance evaluation")
    write_json(args.output_dir / "plan.json", plan)
    results, speeds = [], {}
    begin = time.perf_counter()
    with patch.object(IndustrialEnv, "micro_step", budget.wrap(IndustrialEnv.micro_step)):
        for layout, load in LAYOUTS:
            for pattern in PATTERNS:
                for motion in MOTIONS:
                    for route in ROUTES:
                        label = f"{layout}/{pattern}/{motion}/{route}"
                        with budget.case(label, 600):
                            result, speed = run_case(layout, load, pattern, motion, route)
                        results.append(result)
                        speeds[(layout, pattern, motion, route)] = speed
                        write_json(args.output_dir / "panel.json", results)
                        print(label, "cost/s", round(result["cost_per_s"], 6), "unfinished", result["unfinished"], flush=True)
        checks = verify_pairs(results, speeds)
        write_json(args.output_dir / "checks.json", checks)
        from hrl_v2.control_mechanisms import run_mechanisms
        with budget.case("split_mechanisms", 450):
            mechanisms = run_mechanisms()
        write_json(args.output_dir / "mechanisms.json", mechanisms)
    write_json(args.output_dir / "summary.json", dict(passed=True, panel_runs=len(results),
        mechanism_runs=len(mechanisms), physical_steps=sum(r["physical_steps"] for r in budget.records),
        elapsed_s=time.perf_counter()-begin, training_started=False))


def priority_followup(directory):
    plan = json.loads((directory / "plan.json").read_text())
    if plan["scene_sha256"] != hashlib.sha256(SCENE_PATH.read_bytes()).hexdigest():
        raise RuntimeError("Physical configuration changed since paired controls")
    summary = json.loads((directory / "summary.json").read_text())
    if not summary["passed"] or summary["panel_runs"] != 24:
        raise RuntimeError("A completed original panel is required")
    budget = Budget(directory, resume=True)
    controls = json.loads((directory / "panel.json").read_text())
    write_json(directory / "priority_followup_plan.json", dict(planned_steps=2400,
        reason="Test whether local high priority avoids the observed extra forwarding delay while low priority retains parallel CPU capacity",
        layouts=LAYOUTS, patterns=PATTERNS, motion="hover", route="high_local_low_equal",
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tuning=False, changed_physical_parameters=False))
    results = []
    begin = time.perf_counter()
    with patch.object(IndustrialEnv, "micro_step", budget.wrap(IndustrialEnv.micro_step)):
        for layout, load in LAYOUTS:
            for pattern in PATTERNS:
                label = f"{layout}/{pattern}/hover/high_local_low_equal"
                with budget.case(label, 600):
                    result, _ = run_case(layout, load, pattern, "hover", "high_local_low_equal")
                    control = next(r for r in controls if r["layout_seed"] == layout
                                   and r["pattern"] == pattern and r["motion"] == "hover" and r["route"] == "equal")
                    assert result["tape_sha256"] == control["tape_sha256"]
                    assert result["upload_served_bits"] == control["upload_served_bits"]
                    assert result["energy_j"]["propulsion"] == control["energy_j"]["propulsion"]
                    assert result["final_positions"] == control["final_positions"]
                results.append(result)
                write_json(directory / "priority_followup.json", results)
                print(label, "cost/s", round(result["cost_per_s"], 6), "unfinished", result["unfinished"], flush=True)
    summary.update(priority_followup_runs=len(results), priority_followup_elapsed_s=time.perf_counter()-begin,
        physical_steps=sum(r["physical_steps"] for r in budget.records))
    write_json(directory / "summary.json", summary)


if __name__ == "__main__":
    main()
