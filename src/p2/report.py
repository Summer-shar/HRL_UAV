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
                         seeds=len(set(ref_seed) & set(other_seed)),
                         per_seed={str(s): dict(proposed=float(ref_seed[s]), other=float(other_seed[s]))
                                   for s in sorted(set(ref_seed) & set(other_seed))})
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


def burst_table(root, seeds, methods, smoke=False):
    """Delay by task arrival time under the controlled burst (pooled over seeds and scenarios)."""
    from . import burst as B
    win = B.window(smoke)
    bins, windows = [], []
    for m in methods:
        runs = [C.read_json(p) for seed in seeds
                for p in sorted((C.seed_dir(root, seed)/'burst'/m).glob('[0-9][0-9].json'))]
        runs = [r for r in runs if r.get('burst_bins')]
        if not runs:
            continue
        edges = runs[0]['burst_bins']['edges']
        tot = {k: np.sum([r['burst_bins'][k] for r in runs], axis=0)
               for k in ('w_sojourn', 'w', 'high_sojourn', 'high_n', 'low_sojourn', 'low_n', 'unfinished')}
        ratio = lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
        wd, hd, ld = ratio(tot['w_sojourn'], tot['w']), ratio(tot['high_sojourn'], tot['high_n']), ratio(tot['low_sojourn'], tot['low_n'])
        for i in range(len(edges)-1):
            bins.append(dict(method=m, t_start_s=edges[i], t_end_s=edges[i+1], weighted_delay_s=wd[i],
                             high_delay_s=hd[i], low_delay_s=ld[i], tasks=tot['high_n'][i]+tot['low_n'][i],
                             unfinished=tot['unfinished'][i], runs=len(runs)))
        span = win['end_s'] - win['start_s']
        for name, (a, b) in dict(before=(win['start_s']-span, win['start_s']), burst=(win['start_s'], win['end_s']),
                                 after=(win['end_s'], win['end_s']+span)).items():
            sel = [i for i in range(len(edges)-1) if a <= edges[i] < b]
            if sel:
                windows.append(dict(method=m, window=name, t_start_s=a, t_end_s=b,
                                    weighted_delay_s=float(tot['w_sojourn'][sel].sum()/max(tot['w'][sel].sum(), 1e-12)),
                                    high_delay_s=float(tot['high_sojourn'][sel].sum()/max(tot['high_n'][sel].sum(), 1e-12)),
                                    low_delay_s=float(tot['low_sojourn'][sel].sum()/max(tot['low_n'][sel].sum(), 1e-12)),
                                    runs=len(runs)))
    return dict(window=win, bins=bins, windows=windows)


def _point_root(main_root, base, param, value):
    from .pipeline import point_name
    if C.point_value(param, value) == C.point_value(param, C.NOMINAL[param]):
        return Path(main_root)
    return Path(base)/point_name(param, value)


def retrain_table(main_root, base, seeds, param, d, methods=C.METHODS, values=None):
    """Design-v3 sweep table: every non-nominal value from its own retrained root,
    the nominal value from the main run; all on the test split (scenarios 21-40)."""
    scen = set(d['test'])
    table = []
    for value in (d['retrain_sweeps'][param] if values is None else values):
        root = _point_root(main_root, base, param, value)
        rows = load_rows(root, seeds, methods, 'test', scenarios=scen)
        nominal = root == Path(main_root)
        for m in methods:
            line = dict(param=param, value=C.point_value(param, value), utilization=C.utilization(param, value),
                        terminals=3*(C.point_value(param, value) if param == 'users' else C.NOMINAL['users']),
                        method=m, retrained=not nominal, source=str(root))
            for key, _ in METRICS:
                seeds_ = per_seed(rows, m, key)
                if seeds_:
                    v = np.asarray(list(seeds_.values()))
                    line[key] = float(v.mean())
                    line[key+'_std_over_seeds'] = float(v.std(ddof=1)) if len(v) > 1 else 0.
                    line['seeds'] = len(v)
            line['scenarios'] = len({r['scenario'] for r in rows if r['method'] == m})
            table.append(line)
    return table


def retrain_paired(main_root, base, seeds, param, d, methods=C.METHODS, values=None,
                   metrics=('truncated_delay_s', 'cost_per_s', 'high_truncated_delay_s', 'energy_w_compute_comm')):
    """Proposed vs every other method at every value of a parameter: the paired
    Wilcoxon test of the main table (seed-mean per test scenario), per metric."""
    out = []
    for value in (d['retrain_sweeps'][param] if values is None else values):
        root = _point_root(main_root, base, param, value)
        rows = load_rows(root, seeds, methods, 'test', scenarios=set(d['test']))
        present = [m for m in methods if any(r['method'] == m for r in rows)]
        if 'Proposed' not in present:
            continue
        for metric in metrics:
            for other, r in paired(rows, present, metric).items():
                out.append(dict(param=param, value=C.point_value(param, value), metric=metric, other=other,
                                **{k: v for k, v in r.items() if k not in ('metric', 'per_seed')}))
    return out


def convergence_rows(root, seeds, window=20):
    """Training-curve plateau check per stage and seed (reported, never used to drop
    or extend anything): mean truncated delay and cost of the last `window`
    episodes vs the `window` before them."""
    out = []
    for seed in seeds:
        for stage in ('M1', 'M2', 'M3', 'HoverSAC', 'TFOO', 'TOO', 'NC'):
            rows = C.read_jsonl(C.run_dir(root, seed, stage)/'train.jsonl')
            if len(rows) < 2*window:
                continue
            line = dict(root=str(root), seed=seed, stage=stage, episodes=len(rows))
            for key in ('truncated_delay_s', 'cost_per_s'):
                last = float(np.mean([r[key] for r in rows[-window:]]))
                prev = float(np.mean([r[key] for r in rows[-2*window:-window]]))
                line[f'{key}_last{window}'] = last
                line[f'{key}_prev{window}'] = prev
                line[f'{key}_relative_change'] = last/prev-1 if prev else None
            line['plateau'] = abs(line['cost_per_s_relative_change'] or 0.) <= 0.10
            out.append(line)
    return out


def retrain_build(main_root, base, seeds, points, smoke=False, methods=C.METHODS):
    """Tables, paired tests and training checks of the design-v3 retrained sweep from
    the points finished so far (nominal = main run)."""
    from .pipeline import point_name
    d = C.design(smoke)
    res = Path(base)/'results'
    res.mkdir(parents=True, exist_ok=True)
    lines = ['# Retrained sweep (design v3, docs/PROTOCOL.md)', '',
             'Every learned method trained from scratch at each non-nominal value with the main pipeline '
             '(same stages, budgets, hyper-parameters, seeds, training scenarios, initial weights, fine-tuning '
             f'recipe and dev selection); evaluated on the test scenarios {", ".join(map(str, d["test"]))}. '
             f'Nominal value of each parameter: the main run {main_root}. Values: mean over seeds of scenario '
             'means (std over seeds in the CSV files). users = terminals per region, nominal computation load.', '']
    status = {}
    for param, value in points:
        root = Path(base)/point_name(param, value)
        status[point_name(param, value)] = all(
            (C.seed_dir(root, s)/'test'/m/f'{sc:02d}.json').exists()
            for s in seeds for m in methods for sc in d['test'])
    lines += ['| point | finished |', '|---|---|'] + [f'| {k} | {"yes" if v else "no"} |' for k, v in status.items()]
    tables, pairs = {}, []
    for param in C.RETRAIN_ORDER:
        values = [v for v in d['retrain_sweeps'][param]
                  if C.point_value(param, v) == C.point_value(param, C.NOMINAL[param]) or status.get(point_name(param, v))]
        if len(values) < 2:
            continue
        complete = len(values) == len(d['retrain_sweeps'][param])
        t = retrain_table(main_root, base, seeds, param, d, methods, values)
        tables[param] = t
        write_csv(res/f'retrain_sweep_{param}.csv', t)
        pr = retrain_paired(main_root, base, seeds, param, d, methods, values)
        pairs += pr
        lines += ['', f'## {param} (retrained{"" if complete else ", finished points only"}; '
                  'cells: delay / compute+comm power / joint cost)', '',
                  '| '+param+' | demand/capacity | '+' | '.join(LABELS.get(m, m) for m in methods)+' |',
                  '|---:|---:|'+'---:|'*len(methods)]
        for value in values:
            cells = []
            for m in methods:
                r = next((x for x in t if x['value'] == C.point_value(param, value) and x['method'] == m), {})
                cells.append(f'{fmt(r.get("truncated_delay_s"))} s / {fmt(r.get("energy_w_compute_comm"))} W / '
                             f'{fmt(r.get("cost_per_s"))}')
            lines.append(f'| {C.point_value(param, value):g} | {C.utilization(param, value):.2f} | '+' | '.join(cells)+' |')
        lines += ['', f'Proposed vs others ({param}): relative change of the delay / cost, Wilcoxon p (20 scenarios), '
                  'better seeds', '', '| '+param+' | '+' | '.join(LABELS.get(m, m) for m in methods if m != 'Proposed')+' |',
                  '|---:|'+'---:|'*(len(methods)-1)]
        for value in values:
            cells = []
            for m in methods:
                if m == 'Proposed':
                    continue
                dl = next((x for x in pr if x['value'] == C.point_value(param, value) and x['other'] == m
                           and x['metric'] == 'truncated_delay_s'), None)
                co = next((x for x in pr if x['value'] == C.point_value(param, value) and x['other'] == m
                           and x['metric'] == 'cost_per_s'), None)
                cells.append('—' if not dl or not co else
                             f'{100*dl["relative_change"]:+.1f}% / {100*co["relative_change"]:+.1f}%, '
                             f'p={fmt(dl["wilcoxon_p"], 2)}/{fmt(co["wilcoxon_p"], 2)}, '
                             f'{dl["seeds_proposed_better"]}/{dl["seeds"]}')
            lines.append(f'| {C.point_value(param, value):g} | '+' | '.join(cells)+' |')
    if pairs:
        write_csv(res/'retrain_paired.csv', pairs)
    conv = []
    for name, ok in status.items():
        if ok:
            conv += convergence_rows(Path(base)/name, seeds)
            root = Path(base)/name
            rows = load_rows(root, seeds, methods, 'test')
            tbl = main_table(rows, [m for m in methods if any(r['method'] == m for r in rows)])
            write_csv(res/'points'/f'{name}.csv', tbl)
    if conv:
        write_csv(res/'retrain_convergence.csv', conv)
        flagged = [c for c in conv if not c['plateau']]
        lines += ['', f'## Training curves: {len(conv)} stage runs checked, {len(flagged)} with a cost change above 10 % '
                  'between the last two 20-episode windows (reported, nothing dropped or extended)', '']
        lines += [f'- {Path(c["root"]).name} seed {c["seed"]} {c["stage"]}: cost {fmt(c["cost_per_s_prev20"])} -> '
                  f'{fmt(c["cost_per_s_last20"])} ({100*c["cost_per_s_relative_change"]:+.1f}%)' for c in flagged]
    C.write_json(res/'retrain_stats.json', C.clean(dict(design='v3', seeds=list(seeds), test_scenarios=list(d['test']),
                                                        points=status, tables=tables, paired=pairs, convergence=conv)))
    (res/'retrain_summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return res


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
    point = C.scene_override(root)
    sweeps = {} if point else {p: sweep_table(root, seeds, present, p, d) for p in d['sweeps']}
    saturated = {} if point else {
        p: [r for r in sweep_table(root, seeds, present, p, d, values=v) if r.get('truncated_delay_s') is not None]
        for p, v in d.get('saturated', {}).items() if v}
    write_csv(res/'main_table.csv', table)
    for p, t in sweeps.items():
        write_csv(res/f'sweep_{p}.csv', t)
    for p, t in saturated.items():
        if t:
            write_csv(res/f'saturated_{p}.csv', t)
    recipe = C.finetune(root)
    burst = dict(window=None, bins=[], windows=[]) if point else burst_table(root, seeds, present, smoke)
    if burst['bins']:
        write_csv(res/'burst_bins.csv', burst['bins'])
        write_csv(res/'burst_windows.csv', burst['windows'])
    for seed in seeds:     # which dev candidate each label used, next to the tables
        src = C.seed_dir(root, seed)/'selection_report.json'
        if src.exists():
            C.write_json(res/'selection'/f'seed_{seed}.json', C.read_json(src))
    C.write_json(res/'stats.json', C.clean(dict(seeds=list(seeds), test_scenarios=list(d['test']), finetune=recipe,
                                                 paired=stats, main_table=table, sweeps=sweeps,
                                                 saturated=saturated, burst_windows=burst['windows'])))
    lines = ['# Paper-2 results', '',
             (f'Retrained sweep point: {point["param"]} = {point["value"]:g} (every learned method trained at this value). '
              if point else '') +
             f'Seeds: {", ".join(map(str, seeds))}; test scenarios {", ".join(map(str, C.test_scenarios(root, d)))} (split "test"); '
             'deterministic execution; values = mean over seeds of scenario means (± std over seeds).',
             f'Fine-tuning recipe (M3, HoverSAC): {recipe["recipe"]} (alpha {recipe["alpha"]:g}, actor lr '
             f'{recipe["actor_lr"]:g}, {recipe["actor_frozen_episodes"]} critic-only episodes).', '',
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
    if burst['windows']:
        w = burst['window']
        lines += ['', f'## Controlled burst: heaviest region fully busy in [{w["start_s"]:g}, {w["end_s"]:g}) s '
                  '(delay by task arrival time; pooled over seeds and sweep scenarios)', '',
                  '| Method | before: weighted / high (s) | burst: weighted / high (s) | after: weighted / high (s) |',
                  '|---|---:|---:|---:|']
        for m in present:
            cells = []
            for name in ('before', 'burst', 'after'):
                r = next((x for x in burst['windows'] if x['method'] == m and x['window'] == name), None)
                cells.append(f'{fmt(r["weighted_delay_s"])} / {fmt(r["high_delay_s"])}' if r else '—')
            lines.append(f'| {LABELS.get(m, m)} | '+' | '.join(cells)+' |')
    (res/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return res
