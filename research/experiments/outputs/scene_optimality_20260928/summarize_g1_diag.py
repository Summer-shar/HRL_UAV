"""Tabulate G1 diagnostics against the frozen G1 stochastic panel and the hover_equal/local references."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from hrl_v2.metrics import PooledMetrics
from hrl_v2.gates import g1

OUT = Path(__file__).resolve().parent/'g1_diag'
RUN = ROOT/'outputs/lite_v2_3_0/training_4101_20260929_002123/runs/Proposed/seed_4101'
reference = json.loads((RUN/'baseline_reference.json').read_text())
per_ref = reference['per_scenario']


def g1_panel():
    per = {}
    for f in sorted((RUN/'evaluations/episode-0300').glob('stochastic_hover_*.json')):
        m = json.loads(f.read_text())['metrics']
        per.setdefault(str(int(f.stem.split('_')[2])), PooledMetrics()).add(m)
    return {k: v.result() for k, v in per.items()}


rows = dict(G1_stochastic=g1_panel(), hover_equal=per_ref['hover_equal'], hover_local=per_ref['hover_local'])
for part in ('D1', 'D2a', 'D2b'):
    f = OUT/f'summary_{part}.json'
    if f.exists(): rows[part] = json.loads(f.read_text())['per_scenario']

keys = sorted(rows['hover_equal'], key=int)
eq = np.array([rows['hover_equal'][k]['truncated_delay_s'] for k in keys])
eqc = np.array([rows['hover_equal'][k]['cost_per_s'] for k in keys])
report = {}
print('unfinished and energy columns are normalised to 40 episodes (20 scenarios x 2 streams)')
print(f'{"panel":14s} {"mean_s":>8s} {"wins":>5s} {"unf":>5s} {"cost/s":>7s} {"cwin":>5s} {"H_tr":>7s} {"L_tr":>7s} '
      f'{"H_unf":>5s} {"L_unf":>5s} {"u2u":>7s} {"cpu":>7s} {"E_cpu":>9s} {"E_u2u":>9s} pass')
for name, per in rows.items():
    a = np.array([per[k]['truncated_delay_s'] for k in keys])
    c = np.array([per[k]['cost_per_s'] for k in keys])
    pr = [dict(tr=sum(per[k]['priority'][i]['residence_sum_s'] for k in keys)/sum(per[k]['priority'][i]['served_population'] for k in keys),
               unf=sum(per[k]['priority'][i]['unfinished'] for k in keys)) for i in range(2)]
    wpop = sum(w*sum(per[k]['priority'][i]['served_population'] for k in keys) for i, w in enumerate((10, 1)))
    stage = {s: sum(w*sum(per[k]['priority'][i]['stage_residence_sum_s'][s] for k in keys) for i, w in enumerate((10, 1)))/wpop
             for s in ('upload', 'ready', 'u2u', 'cpu')}
    energy = {e: sum(per[k]['energy_j'][e] for k in keys) for e in ('upload', 'u2u', 'cpu', 'propulsion')}
    episodes = sum(per[k]['episodes'] for k in keys)
    gate = g1(per, per_ref, reference['required_wins']) if name not in ('hover_equal', 'hover_local') else None
    report[name] = dict(mean_delay_s=float(a.mean()), wins_vs_equal=int((a < eq).sum()),
                        unfinished=int(sum(per[k]['unfinished_parents'] for k in keys)), episodes=episodes,
                        unfinished_per_40_episodes=40*sum(per[k]['unfinished_parents'] for k in keys)/episodes,
                        cost_per_s_mean=float(c.mean()), cost_wins_vs_equal=int((c < eqc).sum()),
                        priority=pr, stage_delay_s=stage, energy_j=energy,
                        g1_passed=None if gate is None else gate['passed'],
                        per_scenario_delay={k: float(x) for k, x in zip(keys, a)},
                        per_scenario_cost={k: float(x) for k, x in zip(keys, c)})
    r = report[name]
    print(f'{name:14s} {r["mean_delay_s"]:8.4f} {r["wins_vs_equal"]:5d} {r["unfinished_per_40_episodes"]:5.0f} {r["cost_per_s_mean"]:7.4f} '
          f'{r["cost_wins_vs_equal"]:5d} {pr[0]["tr"]:7.4f} {pr[1]["tr"]:7.4f} {pr[0]["unf"]*40/episodes:5.0f} {pr[1]["unf"]*40/episodes:5.0f} '
          f'{stage["u2u"]:7.4f} {stage["cpu"]:7.4f} {energy["cpu"]*40/episodes:9.0f} {energy["u2u"]*40/episodes:9.0f} {r["g1_passed"]}')
print('\nper-scenario truncated delay (s)')
print('sc  ' + ' '.join(f'{n[:9]:>9s}' for n in rows))
for k in keys:
    print(f'{k:>3s} ' + ' '.join(f'{report[n]["per_scenario_delay"][k]:9.4f}' for n in rows))
(OUT/'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
