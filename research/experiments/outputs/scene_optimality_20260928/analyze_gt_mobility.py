"""Read-only GT mobility probe: how much could the upper layer gain if terminals drift (paper [11] GM model)?

No source change, no learner, no training. Terminal coordinates are moved inside this script only,
through the collector's on_tick hook, every 1 s (paper tau = 1 s):
    v <- c1 v + (1-c1) v_mean + sqrt(1-c1^2) Phi,   Phi ~ N(0, sigma_v^2)
    a <- c2 a + (1-c2) a_mean + sqrt(1-c2^2) Psi,   Psi ~ N(0, sigma_a^2)
    x <- x + v (cos a, sin a) tau, reflected at [0, map_size]
Paper: v_mean 0.5 m/s, a_mean pi/4, c1 = c2 = 0.5.  Ours (paper does not give): sigma and boundary.
Terminal-to-UAV association stays the fixed region mapping, as in the current environment.

groups   a hover+equal (CA)   b tracker+equal   c hover+local   d hover+cvx
tracker  each UAV steps (<= 20 m per 5 s macro) toward the demand-weighted centroid of its own region's
         terminals at their TRUE current positions (weights as baselines.centroid); an oracle-position
         heuristic, i.e. an estimate of what an upper policy could reach, not an upper bound proof.
"""
import json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
OUT = Path(__file__).resolve().parent/'gt_mobility'
CONFIG = ROOT/'src/hrl_v2/configs/v2_3_0_lite.json'
REF = ROOT/'outputs/lite_v2_3_0/training_4101_20260929_002123/baseline_reference'
MASTER, STREAM = 4101, 1
TIERS = dict(none=dict(v_mean=0., a_mean=math.pi/4, sigma_v=0., sigma_a=0.),
             low=dict(v_mean=.5, a_mean=math.pi/4, sigma_v=.1, sigma_a=math.pi/8),
             high=dict(v_mean=.5, a_mean=math.pi/4, sigma_v=.5, sigma_a=math.pi/4))
C1 = C2 = .5
GROUPS = dict(a=('hover', 'equal'), b=('tracker', 'equal'), c=('hover', 'local'), d=('hover', 'cvx'))


class Mobility:
    def __init__(self, env, scenario, tier):
        self.env, self.p = env, TIERS[tier]
        self.rng = np.random.default_rng(np.random.SeedSequence([scenario, 7001]))
        self.start = np.array(env.physics.user_coords, dtype=float)
        self.speed = np.full(39, self.p['v_mean']); self.angle = np.full(39, self.p['a_mean'])
        self.size = env.config['map_size_m']
        self.reflections = np.zeros(39, dtype=int)
        self.distance_sum, self.samples = 0., 0

    def __call__(self, adapter, reading):
        e = adapter.env
        if e.step_index % 10: return
        p = self.p
        self.speed = C1*self.speed+(1-C1)*p['v_mean']+math.sqrt(1-C1**2)*self.rng.normal(0, 1, 39)*p['sigma_v']
        self.angle = C2*self.angle+(1-C2)*p['a_mean']+math.sqrt(1-C2**2)*self.rng.normal(0, 1, 39)*p['sigma_a']
        xy = np.array(e.physics.user_coords, dtype=float)
        xy += self.speed[:, None]*np.stack([np.cos(self.angle), np.sin(self.angle)], 1)
        for axis in range(2):
            low, high = xy[:, axis] < 0, xy[:, axis] > self.size
            xy[low, axis] = -xy[low, axis]; xy[high, axis] = 2*self.size-xy[high, axis]
            flip = low | high
            self.reflections += flip
            # reflect the heading so the terminal keeps moving away from the wall
            if axis == 0: self.angle[flip] = math.pi-self.angle[flip]
            else: self.angle[flip] = -self.angle[flip]
        xy.setflags(write=False)
        e.physics.user_coords = xy
        e._invalidate_view()
        uav = np.asarray(e.physics.uav_coords)[e.physics.user_allocations]
        self.distance_sum += float(np.linalg.norm(xy-uav, axis=1).mean()); self.samples += 1

    def summary(self):
        xy = np.asarray(self.env.physics.user_coords)
        return dict(mean_displacement_m=float(np.linalg.norm(xy-self.start, axis=1).mean()),
                    terminals_reflected=int((self.reflections > 0).sum()),
                    mean_gt_to_serving_uav_m=self.distance_sum/max(self.samples, 1))


def run(job):
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    tier, group, scenario = job
    path = OUT/tier/f'{group}_{scenario["scenario"]:02d}.json'
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
    from hrl_v2.checkpoint import atomic_json
    configure(test_cpu=True)
    config = read_config(CONFIG, ROOT/'outputs')
    scene = json.loads((source_root()/'hrl_v2/configs/physical_scene.json').read_text(encoding='utf-8'))
    upper, route = GROUPS[group]
    sid = scenario['scenario']
    adapter = PhysicsAdapter(scene, timing=ClockConfig.from_config(config))
    adapter.reset(**{k: v for k, v in scenario.items() if k.endswith('_seed')})
    mobility = Mobility(adapter.env, sid, tier)
    step = scene['deployment']['max_target_step_m']

    class Tracker(Policy):
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
    else: policy.lower_override = Routing(route, scene, MASTER)
    tick = time.perf_counter()
    item = dict(probe='gt_mobility', tier=tier, group=group, scenario=sid, stream=STREAM, master=MASTER)
    m = collect_metrics(adapter, policy, identity=item, on_tick=mobility, reward_scale=config['objective']['reward_scale'])
    line = dict(tier=tier, group=group, scenario=sid, truncated_delay_s=m['truncated_delay_s'], cost_per_s=m['cost_per_s'],
                unfinished_parents=m['unfinished_parents'], stage_delay_s=m['stage_delay_s'], energy_j=m['energy_j'],
                priority_truncated_delay_s=[p['truncated_delay_s'] for p in m['priority']],
                event_sha256=m['event_sha256'], wall_seconds=time.perf_counter()-tick, **mobility.summary())
    atomic_json(path, dict(identity=item, tier_parameters=TIERS[tier], c1=C1, c2=C2, line=line))
    print(f'{tier} {group} scenario={sid} delay={line["truncated_delay_s"]:.4f} cost/s={line["cost_per_s"]:.4f} '
          f'disp={line["mean_displacement_m"]:.0f}m wall={line["wall_seconds"]:.0f}s', flush=True)
    return line


def main():
    from hrl_v2.evaluate import gate_manifest
    scenarios = gate_manifest(20)['scenarios']
    # Zero-motion hover+equal must reproduce the frozen reference exactly.
    check = run(('none', 'a', scenarios[0]))
    ref = json.loads((REF/'hover_equal_01_1.json').read_text())['metrics']
    ok = check['event_sha256'] == ref['event_sha256'] and check['truncated_delay_s'] == ref['truncated_delay_s']
    print('zero-motion reproduces hover_equal scenario 1:', ok, flush=True)
    if not ok: raise RuntimeError('Mobility hook changes the static environment')
    jobs = [('none', 'b', s) for s in scenarios]
    jobs += [(t, g, s) for t in ('low', 'high') for g in ('c', 'd', 'a', 'b') for s in scenarios]
    with ProcessPoolExecutor(max_workers=8) as pool:
        lines = list(pool.map(run, jobs))
    (OUT/'lines.json').write_text(json.dumps([check]+lines, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
