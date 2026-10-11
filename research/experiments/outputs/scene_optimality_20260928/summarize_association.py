"""Read-only summary of association.json (20 dev scenes)."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
d = json.loads((OUT/'association.json').read_text())
rows = d['rows']


def m(f):
    return float(np.mean([f(r) for r in rows]))


def rng(f):
    v = [f(r) for r in rows]
    return f'{np.mean(v):.4f} [{np.min(v):.4f}, {np.max(v):.4f}]'


print('scenes', len(rows))
print('upload  fixed_init', rng(lambda r: r['fixed_init']['upload']),
      '| fixed_best', rng(lambda r: r['fixed_best_static']['upload']),
      '| nearest_opt', rng(lambda r: r['nearest_upload_opt']['upload']),
      '| floor', rng(lambda r: r['floor_overhead_every_terminal']['upload']))
print('pooledLB fixed_init', rng(lambda r: r['fixed_init']['lb_pooled']),
      '| nearest_upload_opt', rng(lambda r: r['nearest_upload_opt']['lb_pooled']),
      '| floor (any policy)', rng(lambda r: r['floor_overhead_every_terminal']['lb_pooled']))
print('nearest_upload_opt reassigned', rng(lambda r: r['nearest_upload_opt']['reassigned']),
      'max move', rng(lambda r: max(r['nearest_upload_opt']['move_m'])))
print('localLB fixed_init', rng(lambda r: r['fixed_init']['lb_local']),
      '| nearest_local_opt', rng(lambda r: r['nearest_local_opt']['lb_local']))
print('nearest_local_opt util max', rng(lambda r: max(r['nearest_local_opt']['util'])),
      'min', rng(lambda r: min(r['nearest_local_opt']['util'])),
      '| fixed util max', rng(lambda r: max(r['fixed_init']['util'])))
print('nearest_local_opt reassigned', rng(lambda r: r['nearest_local_opt']['reassigned']),
      'max move', rng(lambda r: max(r['nearest_local_opt']['move_m'])),
      'upload', rng(lambda r: r['nearest_local_opt']['upload']))
print('reach s at 10 m/macro', rng(lambda r: r['nearest_local_opt']['reach_s_10m_per_macro']),
      '| 20 m/macro', rng(lambda r: r['nearest_local_opt']['reach_s_20m_per_macro']))
for k in rows[0]['env']:
    print(f'env {k:28s} delay', rng(lambda r: r['env'][k]['delay']),
          'upload', f"{m(lambda r: r['env'][k]['upload']):.4f}",
          'after_route', f"{m(lambda r: r['env'][k]['after_route']):.4f}",
          'unfinished', f"{m(lambda r: r['env'][k]['unfinished']):.1f}")
for k in rows[0]['baseline']:
    print(f'baseline {k:14s}', rng(lambda r: r['baseline'][k]))
he = np.array([r['env']['fixed_init_equal']['delay'] for r in rows])
for k in ('near_upload_equal', 'near_local_local', 'near_local_equal'):
    v = np.array([r['env'][k]['delay'] for r in rows])
    print(f'{k} vs hover_equal: better in {(v < he).sum()}/20, mean rel change {np.mean(v/he-1)*100:+.2f}%')
best = np.array([min(r['env'][k]['delay'] for k in ('fixed_init_equal', 'near_upload_equal', 'near_local_local', 'near_local_equal'))
                 for r in rows])
print(f'best static option per scene vs hover_equal: mean rel change {np.mean(best/he-1)*100:+.2f}%')
pu = np.array([r['env']['fixed_init_equal_paperu2u']['delay'] for r in rows])
print(f'paper-style U2U (shared 3 MHz) at init, equal routing: mean rel change {np.mean(pu/he-1)*100:+.2f}%')
