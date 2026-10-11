"""Tabulate the GT mobility probe (analyze_gt_mobility.py); static rows come from the frozen reference."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE/'gt_mobility'
REF = HERE.parents[1]/'outputs/lite_v2_3_0/training_4101_20260929_002123/baseline_reference'
names = dict(a='hover+equal (CA)', b='tracker+equal', c='hover+local', d='hover+cvx')
rows = {}
for f in OUT.glob('*/*.json'):
    line = json.loads(f.read_text())['line']
    rows[(line['tier'], line['group'], line['scenario'])] = line
for group, label in (('a', 'equal'), ('c', 'local'), ('d', 'cvx')):
    for s in range(1, 21):
        m = json.loads((REF/f'hover_{label}_{s:02d}_1.json').read_text())['metrics']
        rows.setdefault(('none', group, s), dict(truncated_delay_s=m['truncated_delay_s'], cost_per_s=m['cost_per_s'],
            unfinished_parents=m['unfinished_parents'], stage_delay_s=m['stage_delay_s'],
            priority_truncated_delay_s=[p['truncated_delay_s'] for p in m['priority']],
            mean_displacement_m=0., terminals_reflected=0, mean_gt_to_serving_uav_m=None))
report = {}
for tier in ('none', 'low', 'high'):
    report[tier] = {}
    base = np.array([rows[(tier, 'a', s)]['truncated_delay_s'] for s in range(1, 21)])
    base_cost = np.array([rows[(tier, 'a', s)]['cost_per_s'] for s in range(1, 21)])
    print(f'\n[{tier}]  {"group":18s} {"delay":>7s} {"vs_a":>7s} {"wins":>4s} {"cost/s":>7s} {"vs_a":>7s} {"unf":>5s} '
          f'{"upload":>7s} {"u2u":>6s} {"cpu":>6s} {"H":>6s} {"L":>6s} {"disp":>5s} {"refl":>4s} {"gt-uav":>6s}')
    for g in 'abcd':
        r = [rows[(tier, g, s)] for s in range(1, 21)]
        d = np.array([x['truncated_delay_s'] for x in r]); c = np.array([x['cost_per_s'] for x in r])
        st = {k: np.mean([x['stage_delay_s'][k] for x in r]) for k in ('upload', 'ready', 'u2u', 'cpu')}
        pr = np.mean([x['priority_truncated_delay_s'] for x in r], axis=0)
        gap = [x['mean_gt_to_serving_uav_m'] for x in r if x['mean_gt_to_serving_uav_m'] is not None]
        report[tier][g] = dict(label=names[g], mean_delay_s=float(d.mean()), delay_vs_a=float(d.mean()/base.mean()-1),
            wins_vs_a=int((d < base).sum()), cost_per_s=float(c.mean()), cost_vs_a=float(c.mean()/base_cost.mean()-1),
            cost_wins_vs_a=int((c < base_cost).sum()), unfinished=int(sum(x['unfinished_parents'] for x in r)),
            stage_delay_s=st, priority_delay_s=pr.tolist(),
            mean_displacement_m=float(np.mean([x['mean_displacement_m'] for x in r])),
            terminals_reflected_mean=float(np.mean([x['terminals_reflected'] for x in r])),
            mean_gt_to_serving_uav_m=float(np.mean(gap)) if gap else None,
            per_scenario_delay=d.tolist(), per_scenario_cost=c.tolist())
        q = report[tier][g]
        print(f'        {names[g]:18s} {q["mean_delay_s"]:7.4f} {q["delay_vs_a"]:+7.1%} {q["wins_vs_a"]:4d} {q["cost_per_s"]:7.3f} '
              f'{q["cost_vs_a"]:+7.1%} {q["unfinished"]:5d} {st["upload"]:7.4f} {st["u2u"]:6.3f} {st["cpu"]:6.3f} '
              f'{pr[0]:6.3f} {pr[1]:6.3f} {q["mean_displacement_m"]:5.0f} {q["terminals_reflected_mean"]:4.1f} '
              f'{q["mean_gt_to_serving_uav_m"] or 0:6.1f}')
(OUT/'comparison.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
