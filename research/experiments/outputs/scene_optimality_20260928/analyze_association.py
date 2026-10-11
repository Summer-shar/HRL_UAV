"""Read-only: what static UAV placement can achieve under nearest-UAV association ([11] eq. 30).

No training and no source/config change. Association and paper-style U2U are applied only to
in-memory objects of this analysis process.

A  exact replay of upload FIFO + tick routing for any static positions/association, validated
   against the earlier fixed-association analysis (fixed part, pooled SRPT bound).
B  position search with nearest association: (i) upload-optimal, (ii) local-only-bound-optimal.
C  exact env runs (hover, local or equal routing) at those positions, with the current U2U and
   with paper-style U2U (3 MHz equally shared by active links, fixed -100 dBm noise).
"""
import dataclasses, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
from hrl_v2.evaluate import gate_manifest
from analyze import srpt_area, BASE, T, DT, W, scene_cfg

OUT = Path(__file__).resolve().parent
PRIOR = {r['scenario']: r for r in json.loads((OUT/'results.json').read_text())['rows']}
FIXED_BEST = {r['scenario']: np.array(r['uav_best_static']) for r in json.loads((OUT/'upper_upload.json').read_text())}
INIT = np.array(scene_cfg['base_scene']['uav_initial_positions_m'], float)
MAP = scene_cfg['base_scene']['map_size_m']
SEP = scene_cfg['base_scene']['motion']['min_separation_m']
CPU = 1e10


def nearest(users, pos):
    return np.argmin(((users[:, None, :]-pos[None])**2).sum(-1), axis=1)


def feasible(pos):
    if (pos < 0).any() or (pos > MAP).any():
        return False
    d = np.linalg.norm(pos[:, None]-pos[None], axis=-1)
    return d[np.triu_indices(3, 1)].min() >= SEP


class Replay:
    def __init__(self, sc):
        env = HotspotIndustrialEnv(scene_cfg)
        env.reset(int(sc['arrival_seed']), int(sc['layout_seed']), int(sc['load_seed']))
        arr = [a for k in range(1800) for a in env._scene.frame(k)]
        self.physics = env.physics
        self.users = env.physics.user_coords.copy()
        self.region = env.physics.user_allocations.copy()
        self.u = np.array([a.user_id for a in arr])
        self.b = np.array([a.birth_s for a in arr])
        self.bits = np.array([a.input_bits for a in arr])
        self.cyc = np.array([a.compute_cycles for a in arr])
        self.prio = np.array([a.priority for a in arr])
        self.w = np.where(self.prio == 0, W[0], W[1])
        self.wpop = float(self.w.sum())
        self.groups = [np.flatnonzero(self.u == k) for k in range(39)]
        self.cache = {}

    def finish(self, rates):
        """Continuous FIFO per terminal: F_k = S_k + max_{j<=k}(b_j - S_{j-1})."""
        F = np.empty(len(self.u))
        for k, idx in enumerate(self.groups):
            if idx.size:
                s = self.bits[idx]/rates[k]
                S = np.cumsum(s)
                F[idx] = S+np.maximum.accumulate(self.b[idx]-(S-s))
        return F

    def release(self, pos, assoc, rates=None):
        if rates is None:
            rates = self.physics.compute_u2g_rate_bps(uav_coords=pos, associations=assoc)
        F = self.finish(rates)
        # Upload finishing in tick k is routed at the start of tick k+1.
        return F, np.ceil(F/DT-1e-9)*DT

    def compute_areas(self, rt, assoc):
        routed = rt < T
        hi = routed & (self.prio == 0)
        pooled = 9*srpt_area(rt[hi], self.cyc[hi], 3*CPU, T)+srpt_area(rt[routed], self.cyc[routed], 3*CPU, T)
        a = assoc[self.u]
        local = sum(9*srpt_area(rt[hi & (a == j)], self.cyc[hi & (a == j)], CPU, T)
                    + srpt_area(rt[routed & (a == j)], self.cyc[routed & (a == j)], CPU, T) for j in range(3))
        return pooled, local

    def bounds(self, pos, assoc, rates=None):
        F, rt = self.release(pos, assoc, rates)
        pooled, local = self.compute_areas(rt, assoc)
        fixed = float((self.w*(np.minimum(rt, T)-self.b)).sum())
        a = assoc[self.u]
        return dict(upload=float((self.w*(np.minimum(F, T)-self.b)).sum()/self.wpop), fixed=fixed/self.wpop,
                    lb_pooled=(fixed+pooled)/self.wpop, lb_local=(fixed+local)/self.wpop,
                    util=[float(self.cyc[a == j].sum()/T/CPU) for j in range(3)])

    def upload_obj(self, pos):
        F, _ = self.release(pos, nearest(self.users, pos))
        return float((self.w*(np.minimum(F, T)-self.b)).sum()/self.wpop)

    def local_obj(self, pos):
        """Search surrogate: exact fixed part; per-UAV SRPT area cached per association."""
        assoc = nearest(self.users, pos)
        _, rt = self.release(pos, assoc)
        key = assoc.tobytes()
        if key not in self.cache:
            self.cache[key] = self.compute_areas(rt, assoc)[1]
        return (float((self.w*(np.minimum(rt, T)-self.b)).sum())+self.cache[key])/self.wpop


def search(obj, starts, spans):
    best_val, best = np.inf, None
    for start in starts:
        pos = np.array(start, float)
        val = obj(pos)
        for half, step, passes in spans:
            offs = np.arange(-half, half+1e-9, step)
            for _ in range(passes):
                improved = False
                for i in range(3):
                    base = pos[i].copy()
                    for dx in offs:
                        for dy in offs:
                            cand = pos.copy()
                            cand[i] = base+(dx, dy)
                            if not feasible(cand):
                                continue
                            v = obj(cand)
                            if v < val-1e-12:
                                val, pos, improved = v, cand, True
                if not improved:
                    break
        if val < best_val:
            best_val, best = val, pos
    return best


def paper_u2u(phys, bandwidth=3e6, noise=1e-13):
    def allocate(active_link_mask, uav_coords=None):
        active = np.asarray(active_link_mask, bool).copy()
        np.fill_diagonal(active, False)
        n = int(active.sum())
        bw = np.where(active, bandwidth/max(n, 1), 0.)
        out = active.sum(1)
        power = np.divide(phys.u2u_tx_budget_w*active, out[:, None], out=np.zeros((3, 3)), where=out[:, None] > 0)
        gain = phys._u2u_channel_gain(uav_coords)
        rate = np.where(active, bw*np.log2(1+power*gain/noise), 0.)
        return dict(u2u_link_bandwidth_hz=bw, u2u_link_power_w=power,
                    u2u_link_noise_w=np.where(active, noise, 0.), u2u_rate_bps=rate)
    return allocate


def run_env(sc, pos, assoc, routing, u2u):
    env = HotspotIndustrialEnv(scene_cfg)
    env.reset(int(sc['arrival_seed']), int(sc['layout_seed']), int(sc['load_seed']))
    ph = env.physics
    assert all(np.allclose(ph.user_coords[t.user_id], t.position_m) for t in env._scene.terminals)
    if assoc is not None:
        env._scene.terminals = tuple(dataclasses.replace(t, source=int(assoc[t.user_id])) for t in env._scene.terminals)
        ph.user_allocations = np.asarray(assoc, np.int64).copy()
    ph.uav_coords[:] = pos
    env._target = ph.uav_coords.copy()
    if u2u == 'paper':
        ph.allocate_u2u_resources = paper_u2u(ph)
    env._invalidate_view()
    route = np.stack([np.eye(3)]*2) if routing == 'local' else np.full((2, 3, 3), 1/3)
    for step in range(int(round(T/DT))):
        if step % env.frame_steps == 0:
            env.prepare_frame()
        env.micro_step(route)
    par = list(env.parents.values())
    b = np.array([p.arrival.birth_s for p in par])
    w = np.where(np.array([p.arrival.priority for p in par]) == 0, W[0], W[1])
    up = np.array([np.inf if p.upload_finish_s is None else p.upload_finish_s for p in par])
    rt = np.array([np.inf if p.route_time_s is None else p.route_time_s for p in par])
    fin = np.array([np.inf if p.finish_s is None else p.finish_s for p in par])
    wp = w.sum()
    return dict(holding=float((w*(np.minimum(fin, T)-b)).sum()), delay=float((w*(np.minimum(fin, T)-b)).sum()/wp),
                upload=float((w*(np.minimum(up, T)-b)).sum()/wp),
                after_route=float((w*(np.minimum(fin, T)-np.minimum(rt, T))).sum()/wp),
                unfinished=int(np.isinf(fin).sum()))


def steps_to_reach(pos, per_macro):
    d = np.linalg.norm(pos-INIT, axis=1).max()
    return float(np.ceil(d/per_macro-1e-9)*5.)


def per_scene(sc):
    sid = sc['scenario']
    t0 = time.time()
    r = Replay(sc)
    region = r.region
    # A: validation against the earlier fixed-association analysis
    fixed_init = r.bounds(INIT, region)
    prior = PRIOR[sid]
    assert abs(fixed_init['fixed']-prior['delay_fixed_upload_ready_s']) < 1e-9, (sid, fixed_init, prior['delay_fixed_upload_ready_s'])
    assert abs(fixed_init['lb_pooled']-prior['delay_lb_pooled_srpt_s']) < 1e-9, (sid, fixed_init['lb_pooled'], prior['delay_lb_pooled_srpt_s'])
    fixed_best = r.bounds(FIXED_BEST[sid], region)
    above = r.physics.compute_u2g_rate_bps(user_coords=r.users[:1], uav_coords=np.repeat(r.users[:1], 3, 0),
                                           associations=np.zeros(1, np.int64))
    overhead = np.full(39, float(above[0]))
    floor = r.bounds(INIT, region, rates=overhead)
    floor = dict(upload=floor['upload'], fixed=floor['fixed'], lb_pooled=floor['lb_pooled'])
    # B: nearest-association searches
    p1 = search(r.upload_obj, [INIT, FIXED_BEST[sid]], [(150, 10, 3), (8, 2, 2)])
    a1 = nearest(r.users, p1)
    p2 = search(r.local_obj, [p1, INIT], [(200, 20, 3), (20, 5, 2)])
    a2 = nearest(r.users, p2)
    near1, near2 = r.bounds(p1, a1), r.bounds(p2, a2)
    near_init = r.bounds(INIT, nearest(r.users, INIT))
    t1 = time.time()
    # C: exact env runs
    runs = dict(
        fixed_init_equal=run_env(sc, INIT, None, 'equal', 'current'),
        near_upload_local=run_env(sc, p1, a1, 'local', 'current'),
        near_upload_equal=run_env(sc, p1, a1, 'equal', 'current'),
        near_local_local=run_env(sc, p2, a2, 'local', 'current'),
        near_local_equal=run_env(sc, p2, a2, 'equal', 'current'),
        fixed_init_equal_paperu2u=run_env(sc, INIT, None, 'equal', 'paper'),
        near_local_equal_paperu2u=run_env(sc, p2, a2, 'equal', 'paper'))
    ref = json.loads((BASE/f'baselines/hover_equal_{sid:02d}_1.json').read_text())['metrics']
    assert abs(runs['fixed_init_equal']['holding']-ref['holding_task_s']) < 1e-6*ref['holding_task_s'], (sid, runs['fixed_init_equal'])
    row = dict(scenario=sid, profile_by_region=prior['profile_by_region'],
               fixed_init=fixed_init, fixed_best_static=fixed_best, floor_overhead_every_terminal=floor,
               nearest_at_init=near_init,
               nearest_upload_opt=dict(pos=p1.tolist(), reassigned=int((a1 != region).sum()),
                                       assoc=a1.tolist(), move_m=np.linalg.norm(p1-INIT, axis=1).tolist(), **near1),
               nearest_local_opt=dict(pos=p2.tolist(), reassigned=int((a2 != region).sum()),
                                      assoc=a2.tolist(), move_m=np.linalg.norm(p2-INIT, axis=1).tolist(),
                                      reach_s_10m_per_macro=steps_to_reach(p2, 10.), reach_s_20m_per_macro=steps_to_reach(p2, 20.),
                                      **near2),
               env=runs,
               baseline=dict(hover_equal=ref['truncated_delay_s'],
                             hover_local=json.loads((BASE/f'baselines/hover_local_{sid:02d}_1.json').read_text())['metrics']['truncated_delay_s'],
                             centroid_equal=json.loads((BASE/f'baselines/centroid_equal_{sid:02d}_1.json').read_text())['metrics']['truncated_delay_s']),
               seconds=dict(search=t1-t0, env=time.time()-t1))
    print(f"scene {sid:2d} fixedLB {fixed_init['lb_pooled']:.3f} floorLB {floor['lb_pooled']:.3f} | "
          f"up {fixed_init['upload']:.4f}/{fixed_best['upload']:.4f}/{near1['upload']:.4f}/{floor['upload']:.4f} | "
          f"localLB fixed {fixed_init['lb_local']:.3f} near {near2['lb_local']:.3f} util {np.round(near2['util'], 3)} re {row['nearest_local_opt']['reassigned']} | "
          f"env eq {runs['fixed_init_equal']['delay']:.3f} nearL {runs['near_local_local']['delay']:.3f} "
          f"nearE {runs['near_local_equal']['delay']:.3f} paperE {runs['fixed_init_equal_paperu2u']['delay']:.3f} "
          f"({t1-t0:.0f}+{time.time()-t1:.0f}s)", flush=True)
    return row


def u2u_table():
    env = HotspotIndustrialEnv(scene_cfg)
    ph = env.physics
    out = {}
    for name, alloc in (('current', ph.allocate_u2u_resources), ('paper', paper_u2u(ph))):
        one = np.zeros((3, 3), bool); one[0, 1] = True
        full = ~np.eye(3, dtype=bool)
        out[name] = dict(single_link_0_1_mbps=float(alloc(one)['u2u_rate_bps'][0, 1]/1e6),
                         all_six_mbps=(alloc(full)['u2u_rate_bps']/1e6).round(3).tolist(),
                         distance_m=float(np.linalg.norm(INIT[0]-INIT[1])))
    return out


def main():
    scenes = gate_manifest(20)['scenarios']
    only = [int(x) for x in sys.argv[1:]]
    if only:
        scenes = [s for s in scenes if s['scenario'] in only]
    print('u2u', json.dumps(u2u_table()), flush=True)
    with ProcessPoolExecutor(max_workers=min(10, len(scenes))) as pool:
        rows = list(pool.map(per_scene, scenes))
    name = 'association.json' if not only else f'association_test_{"_".join(map(str, only))}.json'
    (OUT/name).write_text(json.dumps(dict(u2u=u2u_table(), rows=rows), indent=1))


if __name__ == '__main__':
    main()
