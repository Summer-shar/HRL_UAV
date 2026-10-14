"""Closed-loop evaluation: dev selection, final test, parameter sweeps.

Every learned method is executed deterministically (PPO tanh(mean), SAC
argmax support + Dirichlet mean).  Each run writes one small JSON record and,
for test runs, the UAV / terminal trajectories at every 5 s decision.
"""
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import perf

SUPPORT_EQUAL = np.full(6, 6)
SUPPORT_LOCAL = np.tile(np.arange(3), 2)


class ScriptedPolicy:
    """Torch-free hover/equal/local controller (used for the optional CA reference and tests)."""
    observer = None

    def __init__(self, upper, lower):
        if upper != 'hover' or lower not in ('equal', 'local'):
            raise ValueError('Scripted policies are hover + equal/local only')
        self.lower_rule = lower

    def begin_episode(self, identity):
        pass

    def upper(self, report, history):
        return np.zeros(6), {}

    def lower(self, report, history):
        if self.lower_rule == 'equal':
            return np.full((2, 3, 3), 1/3), SUPPORT_EQUAL
        return np.tile(np.eye(3), (2, 1, 1)), SUPPORT_LOCAL


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------
def checkpoint(root, seed, stage, episode):
    rd = C.run_dir(root, seed, stage)
    return rd/'final.pt' if episode == 'final' else rd/'ckpt'/f'ep{int(episode):04d}.pt'


def candidate_paths(root, seed, label, candidate, selected=None):
    """(method, learner_state_path, upper_state_path or None) of one dev candidate."""
    if label == 'ProposedUpper':
        return 'Proposed', checkpoint(root, seed, 'M2', candidate), None
    if label == 'ProposedLower':
        if int(candidate) == 0:
            upper = checkpoint(root, seed, 'M2', selected['ProposedUpper'])
            return 'Proposed', checkpoint(root, seed, 'M1', 'final'), upper
        return 'Proposed', checkpoint(root, seed, 'M3', candidate), None
    if label == 'HoverSAC':
        if int(candidate) == 0:
            return 'HoverSAC', checkpoint(root, seed, 'M1', 'final'), None
        return 'HoverSAC', checkpoint(root, seed, 'HoverSAC', candidate), None
    if label in C.PPO_BASELINES:
        return label, checkpoint(root, seed, label, candidate), None
    raise ValueError(label)


def method_paths(root, seed, method):
    """Checkpoint paths of a method after dev selection."""
    if method == 'CA':
        return 'CA', None, None
    selected = C.read_json(C.seed_dir(root, seed)/'selected.json')
    label = dict(Proposed='ProposedLower').get(method, method)
    return candidate_paths(root, seed, label, selected[label], selected)


def factory(method, seed, learner_path, upper_path):
    if C.METHOD_SPEC[method]['lower'] in ('sac',) or C.METHOD_SPEC[method]['upper'] == 'ppo':
        from .learn import LoadedMethod, load_state
        learner_state = load_state(learner_path)['learner']
        upper_state = load_state(upper_path)['learner'] if upper_path is not None else None
        loaded = LoadedMethod(method, seed, learner_state=learner_state, upper_state=upper_state)
        return loaded.policy
    spec = C.METHOD_SPEC[method]
    return lambda scene, split, scenario: ScriptedPolicy(spec['upper'], spec['lower'])


# ---------------------------------------------------------------------------
# One closed-loop run
# ---------------------------------------------------------------------------
def evaluate_one(make_policy, scene, split, scenario, out_json, *, smoke=False, traj_npz=None, identity=None,
                 burst=None):
    """One deterministic 900 s run.  burst: window dict of p2.burst (controlled
    overload of the heaviest region); the run then also records the delay by
    arrival time."""
    out_json = Path(out_json)
    if out_json.exists():
        return C.read_json(out_json)
    perf.apply()
    from hrl_v2.collector import collect_metrics
    from hrl_v2.physics_adapter import PhysicsAdapter
    config = C.load_config()
    adapter = PhysicsAdapter(scene, timing=C.clocks(scene, smoke))
    adapter.reset(**C.scenario_seeds(split, scenario))
    burst_info = None
    if burst is not None:
        from . import burst as B
        burst_info = B.install(adapter.env, **burst)
    policy = make_policy(scene, split, scenario)
    per_upper = adapter.timing.physical_per_upper
    env = adapter.env
    times, uavs, users = [0.], [env.physics.uav_coords.copy()], [env.physics.user_coords.copy()]

    def tick(ad, reading):
        if ad.env.step_index % per_upper == 0:
            times.append(ad.env.time_s)
            uavs.append(ad.env.physics.uav_coords.copy())
            users.append(ad.env.physics.user_coords.copy())

    started = time.perf_counter()
    metrics = collect_metrics(adapter, policy, identity=dict(identity or {}, split=split, scenario=scenario),
                              on_tick=tick, reward_scale=config['objective']['reward_scale'])
    row = C.summarize(metrics)
    assigned = np.asarray(metrics['assigned_cycles'], dtype=float)
    total = assigned.sum()
    row.update(identity or {}, split=split, scenario=scenario, wall_seconds=time.perf_counter()-started,
               cycles_by_destination=(assigned.sum(axis=(0, 1))/total).tolist() if total else None,
               cycles_by_source=(assigned.sum(axis=(0, 2))/total).tolist() if total else None,
               offloaded_fraction=float(1-np.trace(assigned.sum(axis=0))/total) if total else None,
               uav_path_m=float(np.linalg.norm(np.diff(np.asarray(uavs), axis=0), axis=-1).sum(axis=0).mean()),
               cpu_ghz=scene['base_scene']['resources']['cpu_hz'][0]/1e9,
               u2u_mhz=scene['base_scene']['resources']['u2u_bandwidth_hz']/1e6,
               load=scene['base_scene']['region_multipliers'][0]/C.load_base_scene()['base_scene']['region_multipliers'][0])
    if burst is not None:
        from . import burst as B
        row.update(burst=burst_info, burst_bins=B.record(env, burst['bin_s'], env.time_s,
                                                          tuple(config['objective']['priority_weights'])))
    override = getattr(policy, 'lower_override', None)
    records = getattr(getattr(override, 'rule', None), 'records', None)
    if records is not None:
        row.update(solver_calls=len(records), solver_fallbacks=sum(bool(r.get('fallback')) for r in records))
    if traj_npz is not None:
        Path(traj_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(traj_npz, time_s=np.asarray(times), uav_xy=np.asarray(uavs), user_xy=np.asarray(users),
                            user_source=np.asarray([t.source for t in env._scene.terminals]))
    C.write_json(out_json, C.clean(row))
    return row


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
def dev_job(root, seed, label, candidate, smoke=False):
    d = C.design(smoke)
    sd = C.seed_dir(root, seed)
    selected = C.read_json(sd/'selected.json') if (sd/'selected.json').exists() else {}
    method, learner_path, upper_path = candidate_paths(root, seed, label, candidate, selected)
    make = factory(method, seed, learner_path, upper_path)
    scene = C.load_scene(root)
    folder = sd/'dev'/label/str(candidate)
    rows = []
    for scenario in d['dev']:
        rows.append(evaluate_one(make, scene, 'dev', scenario, folder/f'{scenario:02d}.json', smoke=smoke,
                                 identity=dict(seed=seed, label=label, candidate=candidate, method=method,
                                               learner=str(learner_path), upper=str(upper_path) if upper_path else None)))
    summary = dict(label=label, candidate=candidate, seed=seed, scenarios=list(d['dev']),
                   **{k: float(np.mean([r[k] for r in rows])) for k in (C.SELECTION_METRIC, 'truncated_delay_s', 'energy_w_total')})
    C.write_json(folder/'summary.json', summary)
    return summary


def select(root, seed, labels, smoke=False):
    """Pick, per label, the candidate with the lowest mean dev cost per second (ties: later)."""
    d = C.design(smoke)
    sd = C.seed_dir(root, seed)
    path = sd/'selected.json'
    selected = C.read_json(path) if path.exists() else {}
    report = C.read_json(sd/'selection_report.json') if (sd/'selection_report.json').exists() else {}
    for label in labels:
        candidates = d['upper_candidates'] if label in ('ProposedUpper',)+tuple(C.PPO_BASELINES) else d['lower_candidates']
        scores = {}
        for c in candidates:
            s = C.read_json(sd/'dev'/label/str(c)/'summary.json')
            scores[c] = s[C.SELECTION_METRIC]
        best = min(candidates, key=lambda c: (scores[c], -c))
        selected[label] = best
        report[label] = dict(metric=C.SELECTION_METRIC, scores={str(k): v for k, v in scores.items()}, selected=best)
    C.write_json(path, selected)
    C.write_json(sd/'selection_report.json', report)
    return selected


def test_job(root, seed, method, smoke=False):
    d = C.design(smoke)
    method_, learner_path, upper_path = method_paths(root, seed, method)
    make = factory(method_, seed, learner_path, upper_path)
    scene = C.load_scene(root)
    folder = C.seed_dir(root, seed)/'test'/method
    for scenario in C.test_scenarios(root, d):
        evaluate_one(make, scene, 'test', scenario, folder/f'{scenario:02d}.json', smoke=smoke,
                     traj_npz=folder/'traj'/f'{scenario:02d}.npz',
                     identity=dict(seed=seed, method=method, learner=str(learner_path) if learner_path else None,
                                   upper=str(upper_path) if upper_path else None))


def burst_job(root, seed, method, smoke=False):
    """Controlled burst (p2.burst) on the sweep scenarios, nominal parameters."""
    from . import burst as B
    if C.scene_override(root):
        raise ValueError('burst runs belong to the nominal root, not to a retrained sweep point')
    d = C.design(smoke)
    method_, learner_path, upper_path = method_paths(root, seed, method)
    make = factory(method_, seed, learner_path, upper_path)
    scene = C.load_scene(root)
    folder = C.seed_dir(root, seed)/'burst'/method
    for scenario in d['sweep_scenarios']:
        evaluate_one(make, scene, 'test', scenario, folder/f'{scenario:02d}.json', smoke=smoke,
                     burst=B.window(smoke), identity=dict(seed=seed, method=method, experiment='burst'))


def sweep_job(root, seed, method, param, smoke=False, saturated=False):
    """Stable sweep points, or with saturated=True the points with rho >= 0.95."""
    if C.scene_override(root):
        raise ValueError('a retrained sweep point root has no nested sweeps')
    d = C.design(smoke)
    values = d['saturated' if saturated else 'sweeps'][param]
    if not values:
        return
    method_, learner_path, upper_path = method_paths(root, seed, method)
    make = factory(method_, seed, learner_path, upper_path)
    base = C.load_scene(root)
    for value in values:
        if float(value) == C.NOMINAL[param]:
            continue    # nominal point = the test runs on the same scenarios
        scene = C.scaled_scene(base, **{param: value})
        folder = C.seed_dir(root, seed)/'sweep'/f'{param}={value:g}'/method
        for scenario in d['sweep_scenarios']:
            evaluate_one(make, scene, 'test', scenario, folder/f'{scenario:02d}.json', smoke=smoke,
                         identity=dict(seed=seed, method=method, param=param, value=value))
