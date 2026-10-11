"""Read-only lower-layer headroom probe: can any causal state-aware routing beat equal, and by how much?

No source/config change, no learner, no training. Hover at the initial positions, 20 gate scenarios,
stream 1 (every rule here is deterministic; hover_equal's two streams are bit-identical).

U2U models (override only inside this process):
  current      v2.3.0 scene = [11] eq. 13 literal: 3 MHz shared by active links, full 1 W per link, -100 dBm
  paper_split  same, but each UAV's 1 W is split over its active outgoing links

Rules (lower_override, executed exactly like baselines):
  equal   all rows 1/3
  a050    fixed local share 1/2, rest split evenly
  greedy  frozen GreedyRouting baseline
  cvx     frozen CVXRouting baseline (TFOO lower)
  wf      water-filling on CPU backlog only: one row per priority, shared by all sources;
          equalizes projected finish time (strict priority: low sees high+low backlog)
  wft     sequential per-row water-filling with transfer: time_d = cpu_backlog_d/f + [d!=s](link_queue_sd/rate
          + x_d*bits/rate) + x_d*cycles/f; rows processed high first, each row's assignment added to the
          backlog of the later rows. Rates are the all-active (conservative) link rates in the report.
Work per row = ready work + ground work that finishes upload within the 0.5 s routing frame (causal).
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
OUT = Path(__file__).resolve().parent/'lower_headroom'
CONFIG = ROOT/'src/hrl_v2/configs/v2_3_0_lite.json'
REF = ROOT/'outputs/lite_v2_3_0/training_4101_20260929_002123/baseline_reference'
MASTER, STREAM, FRAME = 4101, 1, .5
U2U = ('current', 'paper_split')
RULES = ('equal', 'a050', 'greedy', 'cvx', 'wf', 'wft')


def paper_split(phys, bandwidth=3e6, noise=1e-13):
    def allocate(active_link_mask, uav_coords=None):
        active = np.asarray(active_link_mask, bool).copy()
        np.fill_diagonal(active, False)
        bw = np.where(active, bandwidth/max(int(active.sum()), 1), 0.)
        out = active.sum(1)
        power = np.divide(phys.u2u_tx_budget_w*active, out[:, None], out=np.zeros((3, 3)), where=out[:, None] > 0)
        rate = np.where(active, bw*np.log2(1+power*phys._u2u_channel_gain(uav_coords)/noise), 0.)
        return dict(u2u_link_bandwidth_hz=bw, u2u_link_power_w=power,
                    u2u_link_noise_w=np.where(active, noise, 0.), u2u_rate_bps=rate)
    return allocate


def fill(offset, slope):
    """x >= 0, sum x = 1, minimizing max_d offset_d + slope_d x_d over used d (water level by bisection)."""
    lo, hi = offset.min(), offset.max()+slope.max()
    for _ in range(100):
        level = (lo+hi)/2
        if np.maximum(0., (level-offset)/slope).sum() > 1: hi = level
        else: lo = level
    x = np.maximum(0., (hi-offset)/slope)
    return x/x.sum()


def work(report):
    """(2, 3) bits and cycles per (priority, source) expected to be routed during this frame."""
    rate = {t.key: t.rate for t in report.terminals}
    bits, cycles = np.zeros((2, 3)), np.zeros((2, 3))
    for p in report.parents:
        soon = p.stage == 'ground' and rate[p.terminal] > 0 and p.upload_remaining/rate[p.terminal] <= FRAME
        if p.stage == 'ready' or soon:
            bits[p.priority, p.source] += p.bits; cycles[p.priority, p.source] += p.cycles
    return bits, cycles


def backlog(report):
    """(2, 3) cycles already committed to each CPU (queued + in transit), per priority."""
    return np.array([[u.compute[p]+u.in_transit[p] for u in report.uavs] for p in range(2)])


def wf(report, history):
    f = np.array([u.cpu_hz for u in report.uavs])
    bits, cycles = work(report)
    load = backlog(report)
    route = np.full((2, 3, 3), 1/3)
    for p in range(2):
        total = cycles[p].sum()
        if total > 0:
            route[p] = fill(load[:p+1].sum(0)/f, total/f)
    return route


def wft(report, history):
    f = np.array([u.cpu_hz for u in report.uavs])
    bits, cycles = work(report)
    load = backlog(report)
    rate, queued = np.full((3, 3), np.inf), np.zeros((2, 3, 3))
    for link in report.links:
        rate[link.source, link.destination] = link.all_active_rate
        queued[:, link.source, link.destination] = link.bits
    route = np.full((2, 3, 3), 1/3)
    for p in range(2):
        for s in range(3):
            if cycles[p, s] <= 0: continue
            remote = np.arange(3) != s
            offset = load[:p+1].sum(0)/f + np.where(remote, queued[:p+1, s].sum(0)/rate[s], 0.)
            slope = cycles[p, s]/f + np.where(remote, bits[p, s]/rate[s], 0.)
            x = fill(offset, slope)
            route[p, s] = x
            load[p] += x*cycles[p, s]; queued[p, s] += np.where(remote, x*bits[p, s], 0.)
    return route


def run(job):
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    u2u, rule, scenario = job
    sid = scenario['scenario']
    path = OUT/u2u/f'{rule}_{sid:02d}.json'
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
    policy = Policy(None, None, None, Streams(MASTER, episode=sid, split='dev', policy_stream=STREAM), mode='hover',
                    deployment=scene['deployment'])

    def wrap(rule_fn):
        def lower(report, history):
            route = np.asarray(rule_fn(report, history), dtype=np.float64)
            ids = np.asarray([SUPPORTS.index(tuple(np.flatnonzero(row > 0))) for row in route.reshape(6, 3)])
            return route, ids
        return lower

    if rule in ('greedy', 'cvx'):
        routing = Routing(rule, scene, MASTER)
        if rule == 'greedy' and u2u == 'paper_split':
            routing.rule.physics.allocate_u2u_resources = paper_split(routing.rule.physics)
        policy.lower_override = routing
    elif rule == 'equal':
        policy.lower_override = wrap(lambda r, h: np.full((2, 3, 3), 1/3))
    elif rule == 'a050':
        a = np.full((3, 3), .25); np.fill_diagonal(a, .5)
        policy.lower_override = wrap(lambda r, h: np.stack([a, a]))
    else:
        policy.lower_override = wrap(dict(wf=wf, wft=wft)[rule])
    tick = time.perf_counter()
    item = dict(probe='lower_headroom', u2u=u2u, rule=rule, scenario=sid, stream=STREAM, master=MASTER)
    m = collect_metrics(adapter, policy, identity=item, reward_scale=config['objective']['reward_scale'])
    line = dict(u2u=u2u, rule=rule, scenario=sid, truncated_delay_s=m['truncated_delay_s'], cost_per_s=m['cost_per_s'],
                unfinished_parents=m['unfinished_parents'], stage_delay_s=m['stage_delay_s'], energy_j=m['energy_j'],
                priority_truncated_delay_s=[p['truncated_delay_s'] for p in m['priority']],
                priority_unfinished=[p['unfinished'] for p in m['priority']],
                event_sha256=m['event_sha256'], wall_seconds=time.perf_counter()-tick)
    atomic_json(path, dict(identity=item, line=line))
    print(f'{u2u} {rule} scenario={sid} delay={line["truncated_delay_s"]:.4f} cost/s={line["cost_per_s"]:.4f} '
          f'unfinished={line["unfinished_parents"]} wall={line["wall_seconds"]:.0f}s', flush=True)
    return line


def main():
    from hrl_v2.evaluate import gate_manifest
    scenarios = gate_manifest(20)['scenarios']
    # Unchanged physics + frozen rules must reproduce the frozen reference exactly.
    for rule, name in (('equal', 'hover_equal'), ('cvx', 'hover_cvx')):
        check = run(('current', rule, scenarios[0]))
        ref = json.loads((REF/f'{name}_01_1.json').read_text())['metrics']
        ok = check['event_sha256'] == ref['event_sha256'] and check['truncated_delay_s'] == ref['truncated_delay_s']
        print(f'current {rule} reproduces {name} scenario 1:', ok, flush=True)
        if not ok: raise RuntimeError(f'Probe setup changes {name}')
    jobs = [(u, r, s) for u in U2U for r in RULES for s in scenarios]
    with ProcessPoolExecutor(max_workers=8) as pool:
        lines = list(pool.map(run, jobs))
    (OUT/'lines.json').write_text(json.dumps(lines, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
