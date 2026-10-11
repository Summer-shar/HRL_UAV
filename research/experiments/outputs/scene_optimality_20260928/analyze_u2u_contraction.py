"""Read-only: paper-literal U2U ([11] eq. 13) and the "UAVs move closer" lever.

(i)  hover at INIT, region association, equal routing, with three U2U models:
       current        6 MHz shared by active links, 1 W per UAV split over its outgoing links,
                      noise = PSD x link bandwidth (audited env, checked against hover_equal)
       paper_split    3 MHz shared by active links, 1 W split, fixed -100 dBm noise
                      (checked against association.json)
       paper_literal  3 MHz shared by active links (assumption: [11] does not state the split),
                      full P_M = 1 W on every link (no power split), fixed sigma^2 = -100 dBm
(ii) under paper_literal: static contraction toward the common centroid,
     pos(s) = c + (1-s)(P0-c), from INIT and from the fixed-association upload-optimal positions,
     with fixed routing local share alpha in {1/3, 1/2, 2/3}; region association unchanged.
     Current U2U is swept at INIT / equal routing for comparison.

Propulsion under the proposed caliber (deployment segment only, p(v)*tau, tau = 1 s,
20 m per 5 s period) is formula-based; the env runs keep the audited per-tick propulsion.
No source/config change; all overrides live only in this process.
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_association import (HotspotIndustrialEnv, gate_manifest, paper_u2u, BASE, T, DT, W, scene_cfg,
                                 INIT, FIXED_BEST)

OUT = Path(__file__).resolve().parent
PRIOR = {r['scenario']: r for r in json.loads((OUT/'association.json').read_text())['rows']}
SHRINK = (0., .1, .2, .3, .4, .6, .8)
ALPHA = (1/3, 1/2, 2/3)
B_M, NOISE_W = 3e6, 1e-13        # [11]: B_M = 3 MHz, sigma^2 = -100 dBm
STEP_M, TAU_S, MACRO_S = 20., 1., 5.
PERIODS = int(round(T/MACRO_S))


def paper_u2u_literal(phys, bandwidth=B_M, noise=NOISE_W):
    def allocate(active_link_mask, uav_coords=None):
        active = np.asarray(active_link_mask, bool).copy()
        np.fill_diagonal(active, False)
        bw = np.where(active, bandwidth/max(int(active.sum()), 1), 0.)
        power = np.where(active, phys.u2u_tx_budget_w, 0.)
        gain = phys._u2u_channel_gain(uav_coords)
        rate = np.where(active, bw*np.log2(1+power*gain/noise), 0.)
        return dict(u2u_link_bandwidth_hz=bw, u2u_link_power_w=power,
                    u2u_link_noise_w=np.where(active, noise, 0.), u2u_rate_bps=rate)
    return allocate


ALLOC = dict(current=None, paper_split=paper_u2u, paper_literal=paper_u2u_literal)


def routing(alpha):
    if alpha == 1/3:
        return np.full((2, 3, 3), 1/3)
    r = np.full((3, 3), (1-alpha)/2)
    np.fill_diagonal(r, alpha)
    return np.stack([r]*2)


def shrink(pos, s):
    c = pos.mean(0)
    return c+(1-s)*(pos-c)


def flying_power(ph, v):
    return float(ph._compute_tcom_flying_power(np.array([float(v)]))[0])


def deploy_propulsion_j(ph, start, end):
    """Proposed caliber: every UAV pays p(v)*tau once per 5 s period, straight-line move at
    <= 20 m per period, p(0)*tau once it holds."""
    total = 0.
    for d in np.linalg.norm(end-start, axis=1):
        n = int(np.floor(d/STEP_M+1e-9))
        rem = d-n*STEP_M
        moving = n+int(rem > 1e-9)
        total += TAU_S*(n*flying_power(ph, STEP_M/TAU_S)+(flying_power(ph, rem/TAU_S) if rem > 1e-9 else 0.)
                        + (PERIODS-moving)*flying_power(ph, 0.))
    return total


def run(sc, pos, alpha, u2u):
    env = HotspotIndustrialEnv(scene_cfg)
    env.reset(int(sc['arrival_seed']), int(sc['layout_seed']), int(sc['load_seed']))
    ph = env.physics
    ph.uav_coords[:] = pos
    env._target = ph.uav_coords.copy()
    if ALLOC[u2u] is not None:
        ph.allocate_u2u_resources = ALLOC[u2u](ph)
    env._invalidate_view()
    route = routing(alpha)
    for step in range(int(round(T/DT))):
        if step % env.frame_steps == 0:
            env.prepare_frame()
        env.micro_step(route)
    assert not env.motion_ticks and np.allclose(ph.uav_coords, pos)
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
                unfinished=int(np.isinf(fin).sum()), energy_j={k: float(v) for k, v in env.energy.items()},
                env_holding_task_s=float(env.holding), env_cost_sum=float(env.cost_sum))


def per_scene(sc):
    sid = sc['scenario']
    t0 = time.time()
    ph = HotspotIndustrialEnv(scene_cfg).physics
    hover = {k: run(sc, INIT, 1/3, k) for k in ALLOC}
    ref = json.loads((BASE/f'baselines/hover_equal_{sid:02d}_1.json').read_text())['metrics']
    assert abs(hover['current']['holding']-ref['holding_task_s']) < 1e-6*ref['holding_task_s'], (sid, hover['current'])
    prior = PRIOR[sid]['env']['fixed_init_equal_paperu2u']
    assert abs(hover['paper_split']['delay']-prior['delay']) < 1e-9, (sid, hover['paper_split'], prior)
    starts = dict(init=INIT, upload_best=FIXED_BEST[sid])
    sweep = []
    for u2u, names, alphas in (('paper_literal', starts, ALPHA), ('current', ('init',), (1/3,))):
        for name in names:
            for s in SHRINK:
                pos = shrink(starts[name], s)
                for alpha in alphas:
                    if name == 'init' and s == 0 and alpha == 1/3:
                        res = hover[u2u]
                    else:
                        res = run(sc, pos, alpha, u2u)
                    move = np.linalg.norm(pos-INIT, axis=1)
                    sweep.append(dict(u2u=u2u, start=name, s=s, alpha=alpha, pos=pos.tolist(),
                                      min_uav_distance_m=float(min(np.linalg.norm(pos[i]-pos[j]) for i, j in ((0, 1), (0, 2), (1, 2)))),
                                      move_max_m=float(move.max()),
                                      reach_s_20m=float(np.ceil(move.max()/STEP_M-1e-9)*MACRO_S),
                                      reach_s_10m=float(np.ceil(move.max()/10.-1e-9)*MACRO_S),
                                      deploy_propulsion_j=deploy_propulsion_j(ph, INIT, pos), **res))
    he = hover['paper_literal']['delay']
    best = min((r for r in sweep if r['u2u'] == 'paper_literal'), key=lambda r: r['delay'])
    print(f"scene {sid:2d} hover_equal cur {hover['current']['delay']:.3f} split {hover['paper_split']['delay']:.3f} "
          f"literal {he:.3f} | best literal {best['delay']:.3f} ({best['start']} s={best['s']} a={best['alpha']:.2f}, "
          f"{(best['delay']/he-1)*100:+.1f}%) ({time.time()-t0:.0f}s)", flush=True)
    return dict(scenario=sid, hover=hover, baseline_hover_equal=ref['truncated_delay_s'], sweep=sweep, seconds=time.time()-t0)


def tables():
    env = HotspotIndustrialEnv(scene_cfg)
    ph = env.physics
    full = ~np.eye(3, dtype=bool)
    one = np.zeros((3, 3), bool)
    one[0, 1] = True
    u2u = {}
    for s in SHRINK:
        pos = shrink(INIT, s)
        u2u[f'{s}'] = dict(distance_01_m=float(np.linalg.norm(pos[0]-pos[1])))
        for name, make in (('current', lambda p: p.allocate_u2u_resources), ('paper_split', paper_u2u),
                           ('paper_literal', paper_u2u_literal)):
            alloc = make(ph)
            u2u[f'{s}'][name] = dict(single_link_0_1_mbps=float(alloc(one, pos)['u2u_rate_bps'][0, 1]/1e6),
                                     all_six_mbps=(alloc(full, pos)['u2u_rate_bps']/1e6).round(3).tolist(),
                                     all_six_power_w=alloc(full, pos)['u2u_link_power_w'].round(3).tolist())
    v = np.linspace(0, 20, 201)
    p = ph._compute_tcom_flying_power(v)
    power = dict(p0_w=flying_power(ph, 0), p10_w=flying_power(ph, 10), p20_w=flying_power(ph, 20),
                 v_min_mps=float(v[np.argmin(p)]), p_min_w=float(p.min()),
                 hover_all_periods_j=3*PERIODS*TAU_S*flying_power(ph, 0),
                 current_caliber_hover_j=3*T*flying_power(ph, 0))
    return dict(u2u=u2u, propulsion=power)


def main():
    scenes = gate_manifest(20)['scenarios']
    only = [int(x) for x in sys.argv[1:]]
    if only:
        scenes = [s for s in scenes if s['scenario'] in only]
    tab = tables()
    print('tables', json.dumps(tab), flush=True)
    with ProcessPoolExecutor(max_workers=min(10, len(scenes))) as pool:
        rows = list(pool.map(per_scene, scenes))
    name = 'u2u_contraction.json' if not only else f'u2u_contraction_test_{"_".join(map(str, only))}.json'
    (OUT/name).write_text(json.dumps(dict(tables=tab, rows=rows), indent=1))


if __name__ == '__main__':
    main()
