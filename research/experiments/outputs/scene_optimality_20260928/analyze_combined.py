"""Read-only combined probe: U2U power split x GT mobility (low tier), upper x lower heuristics.

No source/config change, no learner, no training. Reuses unchanged pieces of earlier probes:
  Mobility (analyze_gt_mobility, tier 'low', same RNG), paper_split / wft (analyze_lower_headroom).
U2U: current (v2.3.0, full 1 W per link) or paper_split (1 W per UAV split over active outgoing links).

groups  CA       hover   + equal
        TOO_h    tracker + equal     (tracker: oracle true-position demand centroid, <= 20 m per 5 s)
        TFOO_h   tracker + cvx       (frozen CVX baseline)
        P_h      tracker + wft       (heuristic stand-in for Proposed; a lower bound on reachable, not SAC)
        hover_wft hover  + wft       (lower-only decomposition)
Checks: current CA / TOO_h scenario 1 must reproduce gt_mobility/low a_01 / b_01 bit-for-bit.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(HERE))
from analyze_gt_mobility import Mobility
from analyze_lower_headroom import paper_split, wft

OUT = HERE/'combined'
CONFIG = ROOT/'src/hrl_v2/configs/v2_3_0_lite.json'
MASTER, STREAM, TIER = 4101, 1, 'low'
GROUPS = dict(CA=('hover', 'equal'), TOO_h=('tracker', 'equal'), TFOO_h=('tracker', 'cvx'),
              P_h=('tracker', 'wft'), hover_wft=('hover', 'wft'))
PLAN = dict(current=('CA', 'TOO_h', 'TFOO_h', 'P_h'), paper_split=tuple(GROUPS))


def run(job):
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    u2u, group, scenario = job
    sid = scenario['scenario']
    path = OUT/u2u/f'{group}_{sid:02d}.json'
    if path.exists(): return json.loads(path.read_text())['line']
    from hrl_v2.run import read_config
    from hrl_v2.runtime import configure
    from hrl_v2.policy import Policy
    from hrl_v2.randomness import Streams
    from hrl_v2.physics_adapter import PhysicsAdapter
    from hrl_v2.collector import collect_metrics
    from hrl_v2.timing import ClockConfig
    from hrl_v2.paths import source_root
    from hrl_v2.baselines import Routing
    from hrl_v2.hybrid_simplex import SUPPORTS
    from hrl_v2.checkpoint import atomic_json
    configure(test_cpu=True)
    config = read_config(CONFIG, ROOT/'outputs')
    scene = json.loads((source_root()/'hrl_v2/configs/physical_scene.json').read_text(encoding='utf-8'))
    adapter = PhysicsAdapter(scene, timing=ClockConfig.from_config(config))
    adapter.reset(**{k: v for k, v in scenario.items() if k.endswith('_seed')})
    if u2u == 'paper_split':
        adapter.env.physics.allocate_u2u_resources = paper_split(adapter.env.physics)
        adapter.env._invalidate_view()
    mobility = Mobility(adapter.env, sid, TIER)
    step = scene['deployment']['max_target_step_m']
    upper, route = GROUPS[group]

    class Tracker(Policy):
        # identical to analyze_gt_mobility's tracker
        def upper(self, report, history):
            observed = np.asarray(history[max(0, report.history_cursor-20):report.history_cursor], dtype=float).reshape(-1, 39, 6)
            elapsed = min(10., report.history_cursor*.5)
            rates = observed.sum(0)/elapsed if elapsed else np.zeros((39, 6))
            xy = np.asarray(adapter.env.physics.user_coords)
            result = []
            for u in report.uavs:
                terminals = [t for t in report.terminals if t.source == u.key]
                w = np.asarray([10*(t.upload[0][1]+5*rates[t.key, 1])+t.upload[1][1]+5*rates[t.key, 4] for t in terminals])
                target = np.average([xy[t.key] for t in terminals], axis=0, weights=w) if w.sum() else np.asarray(u.position)
                delta = target-np.asarray(u.position)
                result.extend(delta/max(step, np.linalg.norm(delta)))
            return np.asarray(result), {}

    cls = Tracker if upper == 'tracker' else Policy
    policy = cls(None, None, None, Streams(MASTER, episode=sid, split='dev', policy_stream=STREAM), mode='hover',
                 deployment=scene['deployment'])
    if route == 'equal':
        policy.lower_override = lambda report, history: (np.full((2, 3, 3), 1/3, dtype=np.float64), np.full(6, 6))
    elif route == 'cvx':
        policy.lower_override = Routing('cvx', scene, MASTER)
    else:
        def lower(report, history):
            r = np.asarray(wft(report, history), dtype=np.float64)
            return r, np.asarray([SUPPORTS.index(tuple(np.flatnonzero(row > 0))) for row in r.reshape(6, 3)])
        policy.lower_override = lower
    tick = time.perf_counter()
    item = dict(probe='combined', u2u=u2u, tier=TIER, group=group, scenario=sid, stream=STREAM, master=MASTER)
    m = collect_metrics(adapter, policy, identity=item, on_tick=mobility, reward_scale=config['objective']['reward_scale'])
    line = dict(u2u=u2u, group=group, scenario=sid, truncated_delay_s=m['truncated_delay_s'], cost_per_s=m['cost_per_s'],
                unfinished_parents=m['unfinished_parents'], stage_delay_s=m['stage_delay_s'], energy_j=m['energy_j'],
                priority_truncated_delay_s=[p['truncated_delay_s'] for p in m['priority']],
                priority_unfinished=[p['unfinished'] for p in m['priority']],
                priority_unfinished_rate=[p['unfinished_rate'] for p in m['priority']],
                event_sha256=m['event_sha256'], wall_seconds=time.perf_counter()-tick, **mobility.summary())
    atomic_json(path, dict(identity=item, line=line))
    print(f'{u2u} {group} scenario={sid} delay={line["truncated_delay_s"]:.4f} cost/s={line["cost_per_s"]:.4f} '
          f'unfinished={line["unfinished_parents"]} disp={line["mean_displacement_m"]:.0f}m wall={line["wall_seconds"]:.0f}s', flush=True)
    return line


def main():
    from hrl_v2.evaluate import gate_manifest
    scenarios = gate_manifest(20)['scenarios']
    for group, old in (('CA', 'a'), ('TOO_h', 'b')):
        check = run(('current', group, scenarios[0]))
        ref = json.loads((HERE/f'gt_mobility/low/{old}_01.json').read_text())['line']
        ok = check['event_sha256'] == ref['event_sha256'] and check['truncated_delay_s'] == ref['truncated_delay_s']
        print(f'current {group} reproduces gt_mobility/low {old}_01:', ok, flush=True)
        if not ok: raise RuntimeError('Combined probe does not reproduce the mobility probe')
    jobs = [(u, g, s) for u, groups in PLAN.items() for g in groups for s in scenarios]
    with ProcessPoolExecutor(max_workers=8) as pool:
        lines = list(pool.map(run, jobs))
    (OUT/'lines.json').write_text(json.dumps(lines, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
