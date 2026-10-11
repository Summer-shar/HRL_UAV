"""Read-only follow-up: U2U requirement per permutation, energy controllability, cost lower bound."""
import itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
from hrl_v2.evaluate import gate_manifest
from analyze import u2u_region, expected_loads, BASE

OUT = Path(__file__).resolve().parent
scene_cfg = json.loads((ROOT/'src/hrl_v2/configs/physical_scene.json').read_text(encoding='utf-8'))
KAPPA, T = 1e-28, 900.


def min_share(cyc, bits, links, rates, u_cap, cpu=1e10):
    """Minimum U2U time share (and forwarded Mbps) so every CPU utilization <= u_cap."""
    nx, nt = 18, len(rates); nv = nx+nt
    idx = lambda p, i, j: 9*p+3*i+j
    c = np.zeros(nv); c[nx:] = 1
    A_eq = []; b_eq = []
    for p in range(2):
        for i in range(3):
            row = np.zeros(nv); row[[idx(p, i, j) for j in range(3)]] = 1; A_eq.append(row); b_eq.append(1)
    A_ub = []; b_ub = []
    for j in range(3):
        row = np.zeros(nv)
        for p in range(2):
            for i in range(3): row[idx(p, i, j)] = cyc[p, i]
        A_ub.append(row); b_ub.append(u_cap*cpu)
    for l, (i, j) in enumerate(links):
        row = np.zeros(nv)
        for p in range(2): row[idx(p, i, j)] = bits[p, i]
        row[nx:] = -rates[:, l]; A_ub.append(row); b_ub.append(0)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq,
                  bounds=[(0, None)]*nv, method='highs')
    if res.status != 0: return None
    x = res.x[:nx].reshape(2, 3, 3)
    fwd = sum(x[p, i, j]*bits[p, i] for p in range(2) for i in range(3) for j in range(3) if i != j)
    return dict(time_share=float(res.x[nx:].sum()), forward_mbps=float(fwd/1e6),
                forward_gcps=float(sum(x[p, i, j]*cyc[p, i] for p in range(2) for i in range(3) for j in range(3) if i != j)/1e9))


def main():
    env = HotspotIndustrialEnv(scene_cfg)
    manifest = gate_manifest(20)
    out = dict(permutations={}, scenes=[])
    seen = set()
    for sc in manifest['scenarios']:
        env.reset(int(sc['arrival_seed']), int(sc['layout_seed']), int(sc['load_seed']))
        perm = tuple(env._scene.realization['profile_by_region'])
        if perm in seen: continue
        seen.add(perm)
        links, rates = u2u_region(env.physics)
        cyc, bits, _ = expected_loads(env)
        region = cyc.sum(0)
        need = {str(u): min_share(cyc, bits, links, rates, u) for u in (0.95, 0.9, 0.85, float(region.sum()/3e10)+1e-6)}
        out['permutations'][str(list(perm))] = dict(region_gcps=(region/1e9).tolist(),
            bits_per_cycle_by_region=(bits.sum(0)/cyc.sum(0)).tolist(), requirement=need)
    # U2U single-link rate against spacing (same formula, hypothetical positions)
    spacing = {}
    for d in (50, 100, 150, 200, 300, 400):
        coords = np.array([[0., 0.], [d, 0.], [d/2, d*np.sqrt(3)/2]])
        m = np.zeros((3, 3), bool); m[0, 1] = True
        one = env.physics.allocate_u2u_resources(m, coords)['u2u_rate_bps'][0, 1]
        all6 = env.physics.allocate_u2u_resources(~np.eye(3, dtype=bool), coords)['u2u_rate_bps'][0, 1]
        spacing[d] = dict(single_mbps=float(one/1e6), all_six_each_mbps=float(all6/1e6))
    out['u2u_rate_vs_equilateral_spacing_m'] = spacing
    # energy / cost bounds per scene (equal-route arrivals are policy-invariant)
    for sc in manifest['scenarios']:
        sid = sc['scenario']
        m = json.loads((BASE/f'baselines/hover_equal_{sid:02d}_1.json').read_text())['metrics']
        cycles = sum(r['compute_cycles'] for r in m['arrivals_by_region'])
        f = cycles/T/3
        e_cpu_min = 3*KAPPA*f**3*T
        out['scenes'].append(dict(scenario=sid, realized_gcps=cycles/T/1e9,
            cpu_energy_equal_j=m['energy_j']['cpu'], cpu_energy_jensen_lb_j=e_cpu_min,
            cpu_energy_at_10ghz_j=KAPPA*1e20*cycles,
            propulsion_j=m['energy_j']['propulsion'], u2u_j=m['energy_j']['u2u'], upload_j=m['energy_j']['upload'],
            cost_equal_per_s=m['mean_cost_per_s'],
            holding_part_equal_per_s=0.09*m['holding_task_s']/T,
            energy_part_equal_per_s=sum(m['energy_j'].values())/T/140))
    (OUT/'capacity.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out['permutations'], indent=1))
    print(json.dumps(spacing, indent=1))
    for s in out['scenes'][:3]+[out['scenes'][8]]: print(s)


if __name__ == '__main__':
    main()
