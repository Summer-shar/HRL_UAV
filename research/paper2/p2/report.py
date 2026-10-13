"""Tables and paired statistics from the test and sweep records (torch-free)."""
import csv
from pathlib import Path

import numpy as np

from . import common as C

LABELS = {'Proposed': 'Proposed (HRL)', 'TFOO': 'TFOO (PPO+CVX)', 'TOO': 'TOO (PPO+Equal)',
          'NC': 'NC (no cooperation)', 'HoverSAC': 'Hover-SAC', 'CA': 'CA (hover+equal)'}
METRICS = [('truncated_delay_s', 'Avg. delay (s)'), ('high_truncated_delay_s', 'High-priority delay (s)'),
           ('low_truncated_delay_s', 'Low-priority delay (s)'), ('high_completed_p95_s', 'High p95 (s)'),
           ('low_completed_p95_s', 'Low p95 (s)'),
           ('weighted_completed_delay_s', 'Completed-task delay (s)'),
           ('energy_w_compute_comm', 'Compute+comm. energy (W)'), ('energy_w_total', 'Total energy (W)'),
           ('cost_per_s', 'Weighted cost (/s)'), ('unfinished_rate', 'Unfinished rate'),
           ('offloaded_fraction', 'Offloaded fraction'), ('uav_path_m', 'UAV path (m)'),
           ('upload_stage_s', 'Upload stage (s)'), ('u2u_stage_s', 'U2U stage (s)'), ('cpu_stage_s', 'CPU stage (s)')]


def load_rows(root, seeds, methods, kind='test', param=None, value=None, scenarios=None):
    rows = []
    for seed in seeds:
        for method in methods:
            if kind == 'test':
                folder = C.seed_dir(root, seed)/'test'/method
            else:
                folder = C.seed_dir(root, seed)/'sweep'/f'{param}={value:g}'/method
            for path in sorted(folder.glob('[0-9][0-9].json')):
                r = C.read_json(path)
                for stage in ('upload', 'u2u', 'cpu', 'ready'):
                    r[f'{stage}_stage_s'] = (r.get('stage_delay_s') or {}).get(stage)
                if scenarios is None or r['scenario'] in scenarios:
                    rows.append(dict(r, seed=seed, method=method))
    return rows


def per_seed(rows, method, metric):
    """Scenario-mean per seed."""
    out = {}
    for seed in sorted({r['seed'] for r in rows}):
        vals = [r[metric] for r in rows if r['method'] == method and r['seed'] == seed and r.get(metric) is not None]
        if vals:
            out[seed] = float(np.mean(vals))
    return out


def per_scenario(rows, method, metric):
    """Seed-mean per scenario (paired unit for the Wilcoxon test)."""
    out = {}
    for sc in sorted({r['scenario'] for r in rows}):
        vals = [r[metric] for r in rows if r['method'] == method and r['scenario'] == sc and r.get(metric) is not None]
        if vals:
            out[sc] = float(np.mean(vals))
    return out


def main_table(rows, methods):
    table = []
    for m in methods:
        line = dict(method=m, label=LABELS.get(m, m))
        for key, _ in METRICS:
            seeds = per_seed(rows, m, key)
            if seeds:
                v = np.asarray(list(seeds.values()))
                line[key] = float(v.mean())
                line[key+'_std_over_seeds'] = float(v.std(ddof=1)) if len(v) > 1 else 0.
        table.append(line)
    return table


def paired(rows, methods, metric='truncated_delay_s', reference='Proposed'):
    from scipy.stats import wilcoxon
    ref = per_scenario(rows, reference, metric)
    ref_seed = per_seed(rows, reference, metric)
    result = {}
    for m in methods:
        if m == reference:
            continue
        other = per_scenario(rows, m, metric)
        keys = sorted(set(ref) & set(other))
        if not keys:
            continue
        a, b = np.asarray([ref[k] for k in keys]), np.asarray([other[k] for k in keys])
        other_seed = per_seed(rows, m, metric)
        try:
            p = float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.
        except ValueError:
            p = None
        result[m] = dict(metric=metric, scenarios=len(keys), proposed_mean=float(a.mean()), other_mean=float(b.mean()),
                         relative_change=float(a.mean()/b.mean()-1), proposed_better_scenarios=int((a < b).sum()),
                         wilcoxon_p=p, seeds_proposed_better=int(sum(ref_seed[s] < other_seed[s] for s in ref_seed if s in other_seed)),
                         seeds=len(set(ref_seed) & set(other_seed)))
    return result


def sweep_table(root, seeds, methods, param, d, values=None):
    nominal = C.NOMINAL[param]
    test_rows = load_rows(root, seeds, methods, 'test', scenarios=set(d['sweep_scenarios']))
    table = []
    for value in (d['sweeps'][param] if values is None else values):
        rows = test_rows if float(value) == nominal else load_rows(root, seeds, methods, 'sweep', param, value,
                                                                   scenarios=set(d['sweep_scenarios']))
        for m in methods:
            line = dict(param=param, value=value, utilization=C.utilization(param, value), method=m)
            for key, _ in METRICS:
                seeds_ = per_seed(rows, m, key)
                if seeds_:
                    v = np.asarray(list(seeds_.values()))
                    line[key] = float(v.mean())
                    line[key+'_std_over_seeds'] = float(v.std(ddof=1)) if len(v) > 1 else 0.
            table.append(line)
    return table


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(x, digits=4):
    return '—' if x is None else f'{x:.{digits}g}'


def build(root, seeds, smoke=False, methods=C.METHODS):
    d = C.design(smoke)
    res = Path(root)/'results'
    rows = load_rows(root, seeds, methods, 'test')
    present = [m for m in methods if any(r['method'] == m for r in rows)]
    table = main_table(rows, present)
    stats = {metric: paired(rows, present, metric) for metric in
             ('truncated_delay_s', 'energy_w_compute_comm', 'cost_per_s', 'high_truncated_delay_s')}
    sweeps = {p: sweep_table(root, seeds, present, p, d) for p in d['sweeps']}
    saturated = {p: [r for r in sweep_table(root, seeds, present, p, d, values=v) if r.get('truncated_delay_s') is not None]
                 for p, v in d.get('saturated', {}).items() if v}
    write_csv(res/'main_table.csv', table)
    for p, t in sweeps.items():
        write_csv(res/f'sweep_{p}.csv', t)
    for p, t in saturated.items():
        if t:
            write_csv(res/f'saturated_{p}.csv', t)
    C.write_json(res/'stats.json', C.clean(dict(seeds=list(seeds), test_scenarios=list(d['test']),
                                                 paired=stats, main_table=table, sweeps=sweeps,
                                                 saturated=saturated)))
    lines = ['# Paper-2 results', '',
             f'Seeds: {", ".join(map(str, seeds))}; test scenarios {d["test"][0]}–{d["test"][-1]} (split "test"); '
             'deterministic execution; values = mean over seeds of scenario means (± std over seeds).', '',
             '| Method | '+' | '.join(label for _, label in METRICS)+' |',
             '|---|'+'---:|'*len(METRICS)]
    for line in table:
        cells = [f'{fmt(line.get(k))} ± {fmt(line.get(k+"_std_over_seeds"), 2)}' if line.get(k) is not None else '—'
                 for k, _ in METRICS]
        lines.append(f'| {line["label"]} | '+' | '.join(cells)+' |')
    for metric, result in stats.items():
        lines += ['', f'## Proposed vs others: {metric}', '',
                  '| Other | Proposed | Other | Change | Better scenarios | Wilcoxon p | Better seeds |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for m, r in result.items():
            lines.append(f'| {LABELS.get(m, m)} | {fmt(r["proposed_mean"])} | {fmt(r["other_mean"])} | '
                         f'{100*r["relative_change"]:+.1f}% | {r["proposed_better_scenarios"]}/{r["scenarios"]} | '
                         f'{fmt(r["wilcoxon_p"], 3)} | {r["seeds_proposed_better"]}/{r["seeds"]} |')
    def sweep_lines(title, p, t, values):
        out = ['', title, '', '| '+p+' | demand/capacity | '+' | '.join(LABELS.get(m, m) for m in present)+' |',
               '|---:|---:|'+'---:|'*len(present)]
        for value in values:
            cells = []
            for m in present:
                r = next((x for x in t if x['value'] == value and x['method'] == m), {})
                cells.append(f'{fmt(r.get("truncated_delay_s"))} s / {fmt(r.get("energy_w_compute_comm"))} W / {fmt(r.get("cost_per_s"))}')
            out.append(f'| {value:g} | {C.utilization(p, value):.2f} | '+' | '.join(cells)+' |')
        return out
    for p, t in sweeps.items():
        lines += sweep_lines(f'## Sweep: {p} (demand/capacity < 0.95; cells: delay / compute+comm power / joint cost)', p, t, d['sweeps'][p])
    for p, t in saturated.items():
        if t:
            lines += sweep_lines(f'## Near or above total capacity: {p} (demand/capacity >= 0.95; reported separately)',
                                 p, t, sorted({r['value'] for r in t}))
    (res/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return res
