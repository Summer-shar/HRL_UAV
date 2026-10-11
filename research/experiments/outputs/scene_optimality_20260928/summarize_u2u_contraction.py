"""Read-only summary of u2u_contraction.json (paper-literal U2U, contraction lever)."""
import json, sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
d = json.loads((OUT/(sys.argv[1] if sys.argv[1:] else 'u2u_contraction.json')).read_text())
rows, tab = d['rows'], d['tables']
n = len(rows)


def rng(v):
    v = np.asarray(v, float)
    return f'{v.mean():.4f} [{v.min():.4f}, {v.max():.4f}]'


print('scenes', n)
print('U2U rates at INIT-contraction (Mbps, all six links active; single link 0->1)')
for s, t in tab['u2u'].items():
    print(f"  s={s:4s} d01={t['distance_01_m']:6.1f} m | current {t['current']['all_six_mbps'][0][1:]} single {t['current']['single_link_0_1_mbps']:.2f}"
          f" | split {t['paper_split']['all_six_mbps'][0][1:]} | literal {t['paper_literal']['all_six_mbps'][0][1:]} single {t['paper_literal']['single_link_0_1_mbps']:.2f}")
print('propulsion (formula)', json.dumps({k: round(v, 1) for k, v in tab['propulsion'].items()}))

print('\n(i) hover at INIT, region association, equal routing')
cur = np.array([r['hover']['current']['delay'] for r in rows])
for k in ('current', 'paper_split', 'paper_literal'):
    h = [r['hover'][k] for r in rows]
    v = np.array([x['delay'] for x in h])
    print(f"  {k:13s} delay {rng(v)} rel {np.mean(v/cur-1)*100:+.2f}% worse in {(v > cur).sum()}/{n} | upload {np.mean([x['upload'] for x in h]):.4f}"
          f" after_route {np.mean([x['after_route'] for x in h]):.4f} unfinished {np.mean([x['unfinished'] for x in h]):.1f}"
          f" | u2u energy {np.mean([x['energy_j']['u2u'] for x in h]):.1f} J")

lit = np.array([r['hover']['paper_literal']['delay'] for r in rows])
sw = lambda r, **k: [x for x in r['sweep'] if all(x[a] == b for a, b in k.items())]
print('\n(ii) paper_literal: mean delay (rel vs hover literal) by start / alpha / s')
alphas = sorted({x['alpha'] for x in rows[0]['sweep'] if x['u2u'] == 'paper_literal'})
shr = sorted({x['s'] for x in rows[0]['sweep']})
for start in ('init', 'upload_best'):
    for a in alphas:
        cells = []
        for s in shr:
            v = np.array([sw(r, u2u='paper_literal', start=start, alpha=a, s=s)[0]['delay'] for r in rows])
            cells.append(f's{s}:{v.mean():.3f}({np.mean(v/lit-1)*100:+.1f}%)')
        print(f'  {start:11s} a={a:.2f} ' + ' '.join(cells))
print('  current U2U, init, equal: ' + ' '.join(
    f"s{s}:{np.mean([sw(r, u2u='current', start='init', s=s)[0]['delay'] for r in rows]):.3f}" for s in shr))

best_route = np.array([min(x['delay'] for x in sw(r, u2u='paper_literal', start='init', s=0.)) for r in rows])
best_all = [min(sw(r, u2u='paper_literal'), key=lambda x: x['delay']) for r in rows]
best_eq = [min(sw(r, u2u='paper_literal', alpha=1/3), key=lambda x: x['delay']) for r in rows]
bv = np.array([x['delay'] for x in best_all])
be = np.array([x['delay'] for x in best_eq])
print('\nper-scene best (paper_literal)')
print(f'  routing lever only (INIT, best alpha)       {rng(best_route)} rel {np.mean(best_route/lit-1)*100:+.2f}%')
print(f'  position lever only (equal, best start/s)   {rng(be)} rel {np.mean(be/lit-1)*100:+.2f}% better in {(be < lit-1e-12).sum()}/{n}')
print(f'  both (best start/s/alpha)                   {rng(bv)} rel {np.mean(bv/lit-1)*100:+.2f}%; vs routing-only {np.mean(bv/best_route-1)*100:+.2f}%')
print(f'  vs current-U2U hover_equal                  rel {np.mean(bv/cur-1)*100:+.2f}%')
for name, b in (('equal', best_eq), ('both', best_all)):
    print(f'  {name}: best s {[x["s"] for x in b]}')
    print(f'  {name}: best start {[x["start"][0] for x in b]} alpha {[round(x["alpha"], 2) for x in b]}')
    print(f'  {name}: move_max {rng([x["move_max_m"] for x in b])} m, reach 20 m/period {rng([x["reach_s_20m"] for x in b])} s,'
          f' 10 m/period {rng([x["reach_s_10m"] for x in b])} s, min UAV distance {rng([x["min_uav_distance_m"] for x in b])} m')
    print(f'  {name}: upload {np.mean([x["upload"] for x in b]):.4f} after_route {np.mean([x["after_route"] for x in b]):.4f}'
          f' unfinished {np.mean([x["unfinished"] for x in b]):.1f}')

print('\nenergy per 900 s (J), hover INIT equal, paper_literal')
h = [r['hover']['paper_literal'] for r in rows]
for k in ('upload', 'cpu', 'u2u', 'propulsion'):
    print(f"  env {k:10s} {np.mean([x['energy_j'][k] for x in h]):10.1f}")
print(f"  proposed propulsion (deployment only, formula, hold at INIT) {np.mean([sw(r, u2u='paper_literal', start='init', s=0., alpha=1/3)[0]['deploy_propulsion_j'] for r in rows]):.1f}")
print(f"  proposed propulsion at per-scene best (both)                 {rng([x['deploy_propulsion_j'] for x in best_all])}")
hold = np.mean([x['env_holding_task_s'] for x in h])
print(f'  env holding_task_s {hold:.1f}')
