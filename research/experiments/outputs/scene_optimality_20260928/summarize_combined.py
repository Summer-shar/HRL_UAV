"""Tabulate combined/: per U2U model, each group vs CA (paired over 20 scenarios) and the ordering P_h < TFOO_h < TOO_h < CA."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
C = HERE/'combined'
KEYS = ('truncated_delay_s', 'cost_per_s')
ORDER = ('CA', 'TOO_h', 'TFOO_h', 'P_h', 'hover_wft')


def load(u2u, group):
    return {int(p.stem.rsplit('_', 1)[1]): json.loads(p.read_text())['line'] for p in sorted((C/u2u).glob(f'{group}_*.json'))}


def main():
    result = {}
    for u2u in ('current', 'paper_split'):
        groups = {g: load(u2u, g) for g in ORDER if load(u2u, g)}
        ca = groups['CA']; ids = sorted(ca)
        rows = []
        for g, lines in groups.items():
            r = dict(group=g, n=len(lines))
            for k in KEYS:
                a, b = np.array([lines[i][k] for i in ids]), np.array([ca[i][k] for i in ids])
                r[k] = float(a.mean()); r[k+'_vs_CA'] = float(a.mean()/b.mean()-1); r[k+'_wins_vs_CA'] = int((a < b).sum())
            r['unfinished_per_ep'] = float(np.mean([lines[i]['unfinished_parents'] for i in ids]))
            r['priority_delay'] = np.mean([lines[i]['priority_truncated_delay_s'] for i in ids], 0).round(4).tolist()
            r['priority_unfinished_rate'] = np.mean([lines[i]['priority_unfinished_rate'] for i in ids], 0).tolist()
            r['upload_s'] = float(np.mean([lines[i]['stage_delay_s']['upload'] for i in ids]))
            r['cpu_kj'] = float(np.mean([lines[i]['energy_j']['cpu'] for i in ids])/1e3)
            r['u2u_kj'] = float(np.mean([lines[i]['energy_j']['u2u'] for i in ids])/1e3)
            rows.append(r)
        pairs = {}
        for a, b in (('P_h', 'TFOO_h'), ('TFOO_h', 'TOO_h'), ('P_h', 'TOO_h'), ('TOO_h', 'CA')):
            if a in groups and b in groups:
                for k in KEYS:
                    x, y = np.array([groups[a][i][k] for i in ids]), np.array([groups[b][i][k] for i in ids])
                    pairs[f'{a}<{b}:{k}'] = dict(rel=float(x.mean()/y.mean()-1), wins=int((x < y).sum()))
        result[u2u] = dict(rows=rows, pairs=pairs)
        print(f'== {u2u} (GT mobility low)')
        print(f'{"group":10s} delay   vs CA   wins  cost/s  vs CA   wins  unfin/ep  hi/lo delay        upload  cpu kJ  u2u kJ')
        for r in rows:
            print(f'{r["group"]:10s} {r["truncated_delay_s"]:.4f} {r["truncated_delay_s_vs_CA"]*100:+6.1f}% {r["truncated_delay_s_wins_vs_CA"]:3d}  '
                  f'{r["cost_per_s"]:.3f} {r["cost_per_s_vs_CA"]*100:+6.1f}% {r["cost_per_s_wins_vs_CA"]:3d}  {r["unfinished_per_ep"]:7.1f}  '
                  f'{r["priority_delay"]}  {r["upload_s"]:.3f}  {r["cpu_kj"]:6.1f}  {r["u2u_kj"]:5.2f}')
        for k, v in pairs.items():
            print(f'  {k:32s} {v["rel"]*100:+6.1f}%  wins {v["wins"]}/20')
    (C/'comparison.json').write_text(json.dumps(result, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
