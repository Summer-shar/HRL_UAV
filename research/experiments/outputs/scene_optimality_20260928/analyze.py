"""Read-only scene optimality analysis (no learner, no training, no source change).

Part A  exogenous replay of the 20 dev gate scenes (busy states, realized load)
Part B  hover + equal routing driven directly on the audited env, only to obtain
        each parent's exact routing (ready) time; checked against frozen baseline
Part C  pooled 30 GHz SRPT lower bound on holding (valid for any hover policy)
Part D  fluid LP: minimum achievable max-CPU utilization under the U2U
        time-sharing capacity region at hover positions
"""
import heapq, itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
from hrl_v2.evaluate import gate_manifest

OUT = Path(__file__).resolve().parent
T, DT, W = 900.0, 0.1, (10.0, 1.0)
scene_cfg = json.loads((ROOT/'src/hrl_v2/configs/physical_scene.json').read_text(encoding='utf-8'))
BASE = ROOT/'outputs/lite_v2_2_1'
EVAL = BASE/'seed_4101_20260928_003614_657543/evaluations/episode-0300'
DIAG = json.loads((ROOT/'outputs/diagnosis_scene9_20260928/results.json').read_text())


def srpt_area(release, size, rate, horizon):
    """Integral over [0,horizon] of jobs-in-service under preemptive SRPT (pointwise optimal N(t))."""
    order = np.argsort(release, kind='stable')
    release, size = release[order], size[order]
    heap, t, k, area, n = [], 0.0, 0, 0.0, len(release)
    while t < horizon and (k < n or heap):
        if not heap:
            t = max(t, release[k])
        while k < n and release[k] <= t:
            heapq.heappush(heap, size[k]/rate); k += 1
        nxt = release[k] if k < n else np.inf
        if not heap:
            continue
        rem = heapq.heappop(heap)
        end = min(t+rem, nxt, horizon)
        area += (len(heap)+1)*(end-t)
        rem -= end-t
        t = end
        if rem > 1e-12:
            heapq.heappush(heap, rem)
    return area


def u2u_region(physics):
    links = [(i, j) for i in range(3) for j in range(3) if i != j]
    subsets, rates = [], []
    for r in range(1, 7):
        for combo in itertools.combinations(range(6), r):
            mask = np.zeros((3, 3), bool)
            for c in combo:
                mask[links[c]] = True
            rate = physics.allocate_u2u_resources(mask)['u2u_rate_bps']
            subsets.append(combo); rates.append([rate[l] for l in links])
    return links, np.array(rates)


def min_max_util(cyc, bits, links, rates, cpu=1e10):
    """cyc,bits: (2,3) per-second loads by (priority, source). Returns u*, routing x, link usage."""
    nx, nt = 18, len(rates)
    idx = lambda p, i, j: 1+9*p+3*i+j
    nv = 1+nx+nt
    c = np.zeros(nv); c[0] = 1
    A_eq, b_eq = [], []
    for p in range(2):
        for i in range(3):
            row = np.zeros(nv); row[[idx(p, i, j) for j in range(3)]] = 1
            A_eq.append(row); b_eq.append(1)
    A_ub, b_ub = [], []
    for j in range(3):
        row = np.zeros(nv); row[0] = -cpu
        for p in range(2):
            for i in range(3):
                row[idx(p, i, j)] = cyc[p, i]
        A_ub.append(row); b_ub.append(0)
    for l, (i, j) in enumerate(links):
        row = np.zeros(nv)
        for p in range(2):
            row[idx(p, i, j)] = bits[p, i]
        row[1+nx:] = -rates[:, l]
        A_ub.append(row); b_ub.append(0)
    row = np.zeros(nv); row[1+nx:] = 1; A_ub.append(row); b_ub.append(1)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq,
                  bounds=[(0, None)]*nv, method='highs')
    assert res.status == 0, res.message
    x = res.x[1:1+nx].reshape(2, 3, 3)
    fwd_bits = sum(x[p, i, j]*bits[p, i] for p in range(2) for i in range(3) for j in range(3) if i != j)
    return float(res.x[0]), x, float(fwd_bits), float(res.x[1+nx:].sum())


def expected_loads(env):
    """Analytic (2,3) cycle and bit rates from the scene's actual profile assignment."""
    sc = env._scene
    a = sc.scene_wrapper['base_scene']['activity']
    pn, pb = (-np.expm1(-0.5/a[k]) for k in ('normal_mean_s', 'busy_mean_s'))
    busy = pn/(pn+pb)
    mult = (1-busy)*a['normal_multiplier']+busy*a['busy_multiplier']
    cyc, bits = np.zeros((2, 3)), np.zeros((2, 3))
    for k, term in enumerate(sc.terminals):
        prof = sc._profiles[k]; par = prof['slots'][sc._logical[k][1]]['parameters']
        rate = par['rate_per_s']*prof['business_multiplier']*mult
        for p, q in enumerate((par['high_probability'], 1-par['high_probability'])):
            bits[p, term.source] += rate*q*par['input_bits'][p]
            cyc[p, term.source] += rate*q*par['input_bits'][p]*par['cycles_per_bit'][p]
    return cyc, bits, busy


def pooled(files):
    h = pop = 0.
    for f in files:
        m = json.loads(Path(f).read_text())['metrics']
        h += m['holding_task_s']; pop += sum(w*p['served_population'] for w, p in zip(W, m['priority']))
    return h/pop


def main():
    manifest = gate_manifest(20)
    env = HotspotIndustrialEnv(scene_cfg)
    route = np.full((2, 3, 3), 1/3)
    rows, scene9 = [], None
    for sc in manifest['scenarios']:
        sid = sc['scenario']
        env.reset(int(sc['arrival_seed']), int(sc['layout_seed']), int(sc['load_seed']))
        perm = env._scene.realization['profile_by_region']
        links, rates = u2u_region(env.physics)
        busy_trace = np.zeros((1800, 39), bool)
        for step in range(int(round(T/DT))):
            if step % env.frame_steps == 0:
                env.prepare_frame()
                busy_trace[step//env.frame_steps] = env._scene._busy
            env.micro_step(route)
        par = list(env.parents.values())
        birth = np.array([p.arrival.birth_s for p in par]); prio = np.array([p.arrival.priority for p in par])
        cyc = np.array([p.arrival.compute_cycles for p in par]); bits = np.array([p.arrival.input_bits for p in par])
        src = np.array([p.arrival.source for p in par])
        rt = np.array([np.inf if p.route_time_s is None else p.route_time_s for p in par])
        fin = np.array([np.inf if p.finish_s is None else p.finish_s for p in par])
        w = np.where(prio == 0, W[0], W[1])
        holding_env = float((w*(np.minimum(fin, T)-birth)).sum())
        ref = json.loads((BASE/f'baselines/hover_equal_{sid:02d}_1.json').read_text())['metrics']
        assert abs(holding_env-ref['holding_task_s']) < 1e-6*max(1, ref['holding_task_s']), (sid, holding_env)
        wpop = float(w.sum())
        # fixed part: upload + ready (identical for all hover policies)
        fixed = float((w*(np.minimum(rt, T)-birth)).sum())
        routed = rt < T
        hi = routed & (prio == 0)
        area_h = srpt_area(rt[hi], cyc[hi], 3e10, T)
        area_all = srpt_area(rt[routed], cyc[routed], 3e10, T)
        lb = fixed + 9*area_h + area_all
        noq = fixed + float((w[routed]*np.minimum(cyc[routed]/3e10, T-rt[routed])).sum())
        # realized and expected loads
        e_cyc, e_bits, busy_p = expected_loads(env)
        r_cyc = np.array([[cyc[(prio == p) & (src == i)].sum()/T for i in range(3)] for p in range(2)])
        r_bits = np.array([[bits[(prio == p) & (src == i)].sum()/T for i in range(3)] for p in range(2)])
        u_exp, x_exp, fwd_exp, tshare_exp = min_max_util(e_cyc, e_bits, links, rates)
        u_real, _, fwd_real, _ = min_max_util(r_cyc, r_bits, links, rates)
        # 60 s sliding windows (10 s step) on realized arrivals
        wins = []
        for start in np.arange(0, T-60+1e-9, 10.):
            m = (birth >= start) & (birth < start+60)
            wc = np.array([[cyc[m & (prio == p) & (src == i)].sum()/60 for i in range(3)] for p in range(2)])
            wb = np.array([[bits[m & (prio == p) & (src == i)].sum()/60 for i in range(3)] for p in range(2)])
            u, _, _, _ = min_max_util(wc, wb, links, rates)
            wins.append(dict(start_s=float(start), total_gcps=float(wc.sum()/1e9),
                             region_gcps=(wc.sum(0)/1e9).tolist(), lp_util=u))
        # policies
        sac_det = next(t for t in DIAG['traces'] if t['scenario'] == sid)['metrics']['truncated_delay_s']
        sac_sto = pooled([EVAL/f'stochastic_hover_{sid:02d}_{s}.json' for s in (1, 2)])
        base = {k: pooled([BASE/f'baselines/{k}_{sid:02d}_{s}.json' for s in (1, 2)])
                for k in ('hover_local', 'hover_equal', 'hover_cvx')}
        row = dict(scenario=sid, profile_by_region=perm,
                   expected_region_gcps=(e_cyc.sum(0)/1e9).tolist(),
                   realized_region_gcps=(r_cyc.sum(0)/1e9).tolist(),
                   expected_total_gcps=float(e_cyc.sum()/1e9), realized_total_gcps=float(r_cyc.sum()/1e9),
                   local_only_util_expected=(e_cyc.sum(0)/1e10).tolist(),
                   lp_util_expected=u_exp, lp_util_realized=u_real,
                   lp_forward_mbps_expected=fwd_exp/1e6, lp_u2u_time_share_expected=tshare_exp,
                   lp_route_expected=x_exp.round(4).tolist(),
                   input_mbps_expected=float(e_bits.sum()/1e6),
                   max_window=max(wins, key=lambda r: r['lp_util']),
                   windows_lp_over_1=int(sum(r['lp_util'] > 1 for r in wins)),
                   windows_total_over_30=int(sum(r['total_gcps'] > 30 for r in wins)),
                   n_windows=len(wins),
                   delay_fixed_upload_ready_s=fixed/wpop, delay_lb_pooled_srpt_s=lb/wpop,
                   delay_lb_noqueue_s=noq/wpop,
                   delay=dict(sac_deterministic=sac_det, sac_stochastic=sac_sto, **base),
                   equal_holding_check=holding_env)
        rows.append(row)
        print(f"scene {sid:2d} perm={perm} exp={row['expected_total_gcps']:.2f} real={row['realized_total_gcps']:.2f} "
              f"LPu={u_exp:.3f}/{u_real:.3f} maxwin={row['max_window']['lp_util']:.3f} "
              f"fixed={row['delay_fixed_upload_ready_s']:.3f} LB={row['delay_lb_pooled_srpt_s']:.3f} "
              f"eq={base['hover_equal']:.3f} sacD={sac_det:.3f}", flush=True)
        if sid == 9:
            frames = np.arange(1800)*0.5
            by_region_busy = np.stack([busy_trace[:, r*13:(r+1)*13].mean(1) for r in range(3)], 1)
            bins = np.arange(0, T+1e-9, 10.)
            region_bins = [[float(cyc[(src == r) & (birth >= a) & (birth < a+10)].sum()/10/1e9)
                            for a in bins[:-1]] for r in range(3)]
            scene9 = dict(windows=wins, busy_fraction_10s=[by_region_busy[int(a/0.5):int(a/0.5)+20].mean(0).tolist()
                                                          for a in bins[:-1]],
                          realized_gcps_10s=region_bins, bin_start_s=bins[:-1].tolist(),
                          u2u_links=links, u2u_rates_single_link_mbps=[float(r) for r in rates[[k for k in range(len(rates)) if k < 6]].max(1)/1e6],
                          u2u_rates_all6_mbps=(rates[-1]/1e6).tolist(),
                          uplink_mbps=(env.physics.compute_u2g_rate_bps()/1e6).tolist())
            np.savez_compressed(OUT/'scene9_busy.npz', busy=busy_trace, t=frames)
    (OUT/'results.json').write_text(json.dumps(dict(rows=rows, scene9=scene9), indent=1))


if __name__ == '__main__':
    main()
