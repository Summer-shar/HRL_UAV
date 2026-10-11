"""Check that v2.4.0 env physics reproduces probe combined/paper_split scenario 1 bit-for-bit."""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
SPEED = Path('D:/HRL_UAV_V2_speedup')
PROBE = Path('D:/HRL_UAV_V2/outputs/scene_optimality_20260928')
OUT = PROBE/'verify_v2_4_0'
sys.path.insert(0, str(SPEED/'src'))
MASTER, STREAM = 4101, 1
GROUPS = dict(CA=('hover', 'equal'), TOO_h=('tracker', 'equal'), TFOO_h=('tracker', 'cvx'))


def run(group):
    from hrl_v2.run import read_config
    from hrl_v2.runtime import configure
    from hrl_v2.policy import Policy
    from hrl_v2.randomness import Streams
    from hrl_v2.physics_adapter import PhysicsAdapter
    from hrl_v2.collector import collect_metrics
    from hrl_v2.timing import ClockConfig
    from hrl_v2.paths import source_root
    from hrl_v2.baselines import Routing
    from hrl_v2.evaluate import gate_manifest
    assert 'hrl_uav_v2_speedup' in str(source_root()).lower(), source_root()
    configure(test_cpu=True)
    config = read_config(SPEED/'src/hrl_v2/configs/v2_4_0_lite.json', OUT)
    scene = json.loads((source_root()/'hrl_v2/configs/physical_scene.json').read_text(encoding='utf-8'))
    scenario = next(s for s in gate_manifest(20)['scenarios'] if s['scenario'] == 1)
    sid = scenario['scenario']
    adapter = PhysicsAdapter(scene, timing=ClockConfig.from_config(config))
    adapter.reset(**{k: v for k, v in scenario.items() if k.endswith('_seed')})
    env = adapter.env
    assert env._mobility is not None and env._mobility.steps == 10
    # Same RNG as the probe's Mobility(env, sid, 'low').
    env._mobility.rng = np.random.default_rng(np.random.SeedSequence([sid, 7001]))
    step = scene['deployment']['max_target_step_m']
    upper, route = GROUPS[group]

    class Tracker(Policy):
        # identical to analyze_combined's tracker
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
    else:
        policy.lower_override = Routing('cvx', scene, MASTER)
    item = dict(probe='combined', u2u='paper_split', tier='low', group=group, scenario=sid, stream=STREAM, master=MASTER)
    tick = time.perf_counter()
    m = collect_metrics(adapter, policy, identity=item, on_tick=None, reward_scale=config['objective']['reward_scale'])
    ref = json.loads((PROBE/f'combined/paper_split/{group}_01.json').read_text())['line']
    new = dict(event_sha256=m['event_sha256'], truncated_delay_s=m['truncated_delay_s'],
               unfinished_parents=m['unfinished_parents'], cost_per_s=m['cost_per_s'],
               mean_displacement_m=float(np.linalg.norm(env.physics.user_coords-env._mobility_start, axis=1).mean()),
               terminals_reflected=int((env._mobility.reflections > 0).sum()))
    same = {k: bool(new[k] == ref[k]) for k in new}
    return dict(group=group, bitwise_match=all(same.values()), fields_equal=same, new=new,
                probe={k: ref[k] for k in new}, wall_seconds=time.perf_counter()-tick)


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(run, GROUPS))
    (OUT/'verify_probe.json').write_text(json.dumps(rows, indent=1, default=lambda o: o.item()), encoding='utf-8')
    print(json.dumps(rows, indent=1, default=lambda o: o.item()), flush=True)
