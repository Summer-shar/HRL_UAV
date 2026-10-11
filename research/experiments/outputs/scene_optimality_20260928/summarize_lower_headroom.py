"""Tabulate lower_headroom/: per U2U model, each rule vs equal (paired over 20 scenarios), plus SAC ep300 (D1, current U2U)."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
LH = OUT/'lower_headroom'
RULES = ('equal', 'a050', 'greedy', 'cvx', 'wf', 'wft')
KEYS = ('truncated_delay_s', 'cost_per_s')


def load(u2u, rule):
    return {int(p.stem.split('_')[1]): json.loads(p.read_text())['line'] for p in sorted((LH/u2u).glob(f'{rule}_*.json'))}


def row(lines, base, label):
    ids = sorted(base)
    out = dict(rule=label, n=len(ids))
    for k in KEYS:
        a, b = np.array([lines[i][k] for i in ids]), np.array([base[i][k] for i in ids])
        out[k] = float(a.mean()); out[k+'_vs_equal'] = float(a.mean()/b.mean()-1); out[k+'_wins'] = int((a < b).sum())
    out['unfinished_per_ep'] = float(np.mean([lines[i]['unfinished_parents'] for i in ids]))
    out['priority_delay'] = np.mean([lines[i]['priority_truncated_delay_s'] for i in ids], 0).round(4).tolist()
    out['stage_u2u'] = float(np.mean([lines[i]['stage_delay_s']['u2u'] for i in ids])) if isinstance(base[ids[0]]['stage_delay_s'], dict) and 'u2u' in base[ids[0]]['stage_delay_s'] else None
    out['energy_cpu_kj'] = float(np.mean([lines[i]['energy_j']['cpu'] for i in ids])/1e3)
    out['energy_u2u_kj'] = float(np.mean([lines[i]['energy_j']['u2u'] for i in ids])/1e3)
    return out


def main():
    result = {}
    for u2u in ('current', 'paper_split'):
        base = load(u2u, 'equal')
        rows = [row(load(u2u, r), base, r) for r in RULES if len(load(u2u, r)) == len(base)]
        best = {i: min((load(u2u, r)[i] for r in RULES), key=lambda l: l['cost_per_s']) for i in base}
        rows.append(row(best, base, 'per_scenario_best_cost'))
        if u2u == 'current':
            d1 = {int(p.stem.split('_')[0]): json.loads(p.read_text())['metrics'] for p in sorted((OUT/'g1_diag/D1').glob('*_0.json'))}
            d1 = {i: dict(truncated_delay_s=m['truncated_delay_s'], cost_per_s=m['cost_per_s'], unfinished_parents=m['unfinished_parents'],
                          priority_truncated_delay_s=[p['truncated_delay_s'] for p in m['priority']], stage_delay_s=m['stage_delay_s'],
                          energy_j=m['energy_j']) for i, m in d1.items()}
            rows.append(row(d1, base, 'SAC_ep300_deterministic'))
        result[u2u] = rows
        print(f'== {u2u}')
        print(f'{"rule":26s} delay    vs eq   wins  cost/s  vs eq   wins  unfin/ep  hi/lo delay     cpu kJ  u2u kJ')
        for r in rows:
            print(f'{r["rule"]:26s} {r["truncated_delay_s"]:.4f} {r["truncated_delay_s_vs_equal"]*100:+6.1f}% {r["truncated_delay_s_wins"]:3d}  '
                  f'{r["cost_per_s"]:.3f} {r["cost_per_s_vs_equal"]*100:+6.1f}% {r["cost_per_s_wins"]:3d}  {r["unfinished_per_ep"]:7.1f}  '
                  f'{r["priority_delay"]}  {r["energy_cpu_kj"]:6.1f}  {r["energy_u2u_kj"]:6.2f}')
    (LH/'comparison.json').write_text(json.dumps(result, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
