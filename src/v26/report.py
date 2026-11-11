"""Tables and consolidated records of a v2.6 run (torch-free).

Written to <out-base>/results/ (pushed to GitHub by the status publisher):

  v26_selection.csv            dev selection of both variants (all candidate scores)
  v26_decision.json            decision table (written by core.decide)
  v26_summary.md               main table, ablation, burst and zero-shot comparison
  v26_records_test.json        new test records (changed seeds / methods of the adopted variant)
  v26_records_burst.json       new burst records
  v26_records_zeroshot.json    new zero-shot records
  v26_records_ablation.json    MH-TPCO without stage 3, test scenarios
  v26_records_burstdev.json    burst experiment on the dev scenarios (decision data)
  v26_train_logs.json          training log rows of the continued windows, both variants
"""
import csv
import statistics as st
from pathlib import Path

import numpy as np

from p2 import common as C
from . import core as V
from . import nosupport as NS

ORDER = ('NC', 'CA', 'HoverSAC', 'TOO', 'TFOO', 'Proposed')
KEYS = (('D', lambda r: V._mean_delay(r)), ('DH', lambda r: r['high_truncated_delay_s']),
        ('DL', lambda r: r['low_truncated_delay_s']), ('P', lambda r: r['energy_w_compute_comm']),
        ('C', lambda r: r['cost_per_s']), ('off', lambda r: r['offloaded_fraction']))


def _records(folder):
    out = {}
    for p in sorted(Path(folder).glob('[0-9][0-9].json')):
        r = C.read_json(p)
        out[int(r['scenario'])] = r
    return out


def _wilcoxon(a, b):
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(a, b).pvalue)
    except Exception:                                           # noqa: BLE001 (scipy missing, all-zero differences)
        return None


def _stats(records):
    """records: {(seed, scenario): row} -> means over all records and per-scenario seed means."""
    rows = list(records.values())
    mean = {k: st.mean(f(r) for r in rows) for k, f in KEYS}
    scen = sorted({c for _, c in records})
    seeds = sorted({s for s, _ in records})
    per_scen = {k: [st.mean(f(records[(s, c)]) for s in seeds if (s, c) in records) for c in scen] for k, f in KEYS}
    per_seed = {s: {k: st.mean(f(r) for (s2, _), r in records.items() if s2 == s) for k, f in KEYS} for s in seeds}
    return dict(mean=mean, per_scen=per_scen, per_seed=per_seed, n=len(rows))


def _compare(a, b):
    """a against b on the per-scenario seed means: relative change, scenarios where a is lower, Wilcoxon p."""
    out = {}
    for k in ('D', 'DH', 'C'):
        x, y = np.asarray(a['per_scen'][k]), np.asarray(b['per_scen'][k])
        if len(x) != len(y) or not len(x):
            out[k] = None
            continue
        out[k] = dict(change=float(a['mean'][k]/b['mean'][k]-1), better=int((x < y).sum()), n=len(x),
                      p=_wilcoxon(x, y))
    return out


def _fmt_compare(c):
    if c is None:
        return '-'
    p = '' if c['p'] is None else f', p={c["p"]:.2g}'
    return f'{100*c["change"]:+.2f}% ({c["better"]}/{c["n"]}{p})'


def _table(lines, title, stats, reference='v2.6'):
    lines += ['', f'## {title}', '', '| 方法 | 记录数 | D/s | DH/s | DL/s | P/W | C | 转发比例 |', '|---|---|---|---|---|---|---|---|']
    for name, s in stats.items():
        m = s['mean']
        lines.append(f'| {name} | {s["n"]} | {m["D"]:.4f} | {m["DH"]:.4f} | {m["DL"]:.4f} | {m["P"]:.1f} | '
                     f'{m["C"]:.3f} | {100*m["off"]:.1f}% |')
    if reference in stats:
        lines += ['', f'{reference} 相对各方法（按场景的种子均值配对；括号内为更优场景数与 Wilcoxon p）：', '',
                  '| 对比 | D | DH | C |', '|---|---|---|---|']
        for name, s in stats.items():
            if name == reference:
                continue
            c = _compare(stats[reference], s)
            lines.append(f'| {name} | {_fmt_compare(c["D"])} | {_fmt_compare(c["DH"])} | {_fmt_compare(c["C"])} |')


def merged(base, main, adopted, seeds, method, kind, point=None, zeroshot_main=None):
    """Records of a method under v2.6: new records for the changed seeds of the adopted
    variant, the main run's records otherwise.  Returns ({(seed, scenario): row}, new rows)."""
    out, new = {}, []
    for seed in seeds:
        is_new = adopted != 'main' and method in V.METHOD_LABEL and V.changed(base, adopted, seed, method)
        if is_new:
            folder = dict(test=V.test_dir, burst=V.burst_dir)[kind](base, adopted, seed, method) if kind != 'zeroshot' \
                else V.zeroshot_dir(base, adopted, point, seed, method)
        elif kind == 'zeroshot':
            folder = None if zeroshot_main is None else Path(zeroshot_main)/point/f'seed_{seed}'/'test'/method
        else:
            folder = C.seed_dir(main, seed)/kind/method
        rows = _records(folder) if folder is not None and Path(folder).exists() else {}
        for scenario, r in rows.items():
            out[(seed, scenario)] = r
            if is_new:
                new.append(r)
    return out, new


def _main(main, seeds, method, kind, point=None, zeroshot_main=None):
    return merged(None, main, 'main', seeds, method, kind, point, zeroshot_main)[0]


def build(base, main, seeds, smoke=False, zeroshot_main=None):
    base, main = Path(base), Path(main)
    out = base/'results'
    out.mkdir(parents=True, exist_ok=True)
    s = V.sizes(smoke)
    lines = ['# v2.6 结果汇总', '', f'生成时间：{V.now()}；主实验目录 `{main}`（只读）；输出目录 `{base}`。',
             '', 'D 为全部任务的平均处理时延（不按优先级加权），DH、DL 为高、低优先级任务时延，P 为计算与通信功率，'
             'C 为单位时间系统代价。']

    # ---- dev selection -------------------------------------------------------------------
    rows = []
    for variant in V.VARIANTS:
        for seed in seeds:
            path = C.seed_dir(V.variant_root(base, variant), seed)/'v26_selection.json'
            if not path.exists():
                continue
            for label, r in C.read_json(path).items():
                rows.append(dict(variant=variant, seed=seed, label=label, main_selected=r['main_selected'],
                                 main_dev_cost=r['main_dev_cost'], selected=r['selected'], dev_cost=r['dev_cost'],
                                 changed=r['changed'], **{f'score_{c}': r['scores'].get(str(c)) for c in s['all']}))
    if rows:
        with open(out/'v26_selection.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        lines += ['', '## 验证集选择', '', '| 版本 | 种子 | 标签 | 原选中（代价） | 现选中（代价） | 各候选代价 |', '|---|---|---|---|---|---|']
        for r in rows:
            scores = ' '.join(f'{c}:{r[f"score_{c}"]:.3f}' for c in s['all'] if r[f'score_{c}'] is not None)
            lines.append(f'| {r["variant"]} | {r["seed"]} | {r["label"]} | {r["main_selected"]} ({r["main_dev_cost"]:.4f}) | '
                         f'{r["selected"]} ({r["dev_cost"]:.4f}) | {scores} |')

    decision_path = out/'v26_decision.json'
    decision = C.read_json(decision_path) if decision_path.exists() else None
    adopted = decision['adopted'] if decision else 'main'
    if decision:
        lines += ['', '## 选择', '', f'采用 **{adopted}**：{decision["reason"]}。详见 `DECISION.md`。']

    # ---- ablation: no stage 3 ------------------------------------------------------------
    ablation = {}
    for seed in seeds:
        for scenario, r in _records(base/'ablation'/'NoStage3'/f'seed_{seed}').items():
            ablation[(seed, scenario)] = r
    C.write_json(out/'v26_records_ablation.json', list(ablation.values()))
    nosup = {}
    for seed in seeds:
        for scenario, r in _records(NS.test_dir(base, seed)).items():
            nosup[(seed, scenario)] = r
    C.write_json(out/'v26_records_nosupport.json', list(nosup.values()))
    ns_rows = []
    for seed in seeds:
        path = C.seed_dir(NS.root(base), seed)/'selection.json'
        if path.exists():
            r = C.read_json(path)
            ns_rows.append(f'| {seed} | {r["selected"]} | ' + ' '.join(f'{c}:{v:.3f}' for c, v in r['scores'].items()) + ' |')
    if ns_rows:
        lines += ['', '## 无支撑集选择消融（NoSupport）的验证集选择', '', '| 种子 | 选中的窗口 | 各候选验证代价 |',
                  '|---|---|---|'] + ns_rows

    # ---- main table ----------------------------------------------------------------------
    stats, new_test = {}, []
    for method in ORDER:
        rec = _main(main, seeds, method, 'test')
        if rec:
            stats[method if method != 'Proposed' else 'Proposed（现有）'] = _stats(rec)
    if ablation:
        stats['NoStage3（无第三阶段）'] = _stats(ablation)
    if nosup:
        stats['NoSupport（无支撑集选择）'] = _stats(nosup)
    if adopted != 'main':
        for method, name in (('HoverSAC', 'HoverSAC（v2.6）'), ('Proposed', 'v2.6')):
            rec, new = merged(base, main, adopted, seeds, method, 'test')
            new_test += new
            if rec:
                stats[name] = _stats(rec)
    reference = 'v2.6' if 'v2.6' in stats else 'Proposed（现有）'
    if stats:
        _table(lines, f'标称参数，测试场景（{reference} 为比较基准）', stats, reference)
        lines += ['', '各种子的 MH-TPCO（D / DH / C）：', '']
        for name in ('Proposed（现有）', 'NoStage3（无第三阶段）', 'NoSupport（无支撑集选择）', 'v2.6'):
            if name in stats:
                lines.append(f'- {name}：' + '；'.join(f'{seed}: {v["D"]:.4f} / {v["DH"]:.4f} / {v["C"]:.3f}'
                                                    for seed, v in stats[name]['per_seed'].items()))
    C.write_json(out/'v26_records_test.json', new_test)

    # ---- burst experiment (test scenarios) -----------------------------------------------
    bstats, new_burst = {}, []
    for method in ORDER:
        rec = _main(main, seeds, method, 'burst')
        if rec:
            bstats[method if method != 'Proposed' else 'Proposed（现有）'] = _stats(rec)
    if adopted != 'main':
        for method, name in (('HoverSAC', 'HoverSAC（v2.6）'), ('Proposed', 'v2.6')):
            rec, new = merged(base, main, adopted, seeds, method, 'burst')
            new_burst += new
            if rec:
                bstats[name] = _stats(rec)
    if bstats:
        _table(lines, '突发实验（测试场景，整个窗口）', bstats, 'v2.6' if 'v2.6' in bstats else 'Proposed（现有）')
    C.write_json(out/'v26_records_burst.json', new_burst)

    # ---- zero-shot sweep -----------------------------------------------------------------
    new_zero = []
    if adopted != 'main':
        zl = []
        for name, param, value in V.zeroshot_points(smoke):
            rec, new = merged(base, main, adopted, seeds, 'Proposed', 'zeroshot', name, zeroshot_main)
            new_zero += new
            new_zero += merged(base, main, adopted, seeds, 'HoverSAC', 'zeroshot', name, zeroshot_main)[1]
            if not rec:
                continue
            a = _stats(rec)
            cells = [name, f'{a["mean"]["D"]:.4f}', f'{a["mean"]["C"]:.3f}']
            for other in ('Proposed', 'TFOO', 'TOO'):
                rec_o = _main(main, seeds, other, 'zeroshot', name, zeroshot_main) if zeroshot_main else {}
                if rec_o and set(rec_o) == set(rec):
                    c = _compare(a, _stats(rec_o))
                    cells.append(f'{_fmt_compare(c["D"])} / {_fmt_compare(c["C"])}')
                else:
                    cells.append('-')
            zl.append('| ' + ' | '.join(cells) + ' |')
        if zl:
            lines += ['', '## 零样本扫参（v2.6 的 MH-TPCO；对比列为 D / C 的相对变化）', '',
                      '| 参数点 | D/s | C | 对现有 MH-TPCO | 对 TFOO | 对 TOO |', '|---|---|---|---|---|---|'] + zl
            if not zeroshot_main:
                lines += ['', '（未给出 --zeroshot-main，没有与原零样本结果比较。）']
    C.write_json(out/'v26_records_zeroshot.json', new_zero)

    # ---- decision data and training logs ---------------------------------------------------
    burstdev = []
    for variant in ('main',)+V.VARIANTS:
        for seed in seeds:
            burstdev += [dict(r, variant=variant) for r in _records(V.burstdev_dir(base, variant, seed)).values()]
    C.write_json(out/'v26_records_burstdev.json', burstdev)
    logs = []
    for variant in V.VARIANTS:
        for seed in seeds:
            for stage in V.LABELS.values():
                rd = C.run_dir(V.variant_root(base, variant), seed, stage)
                logs += [dict(r, variant=variant) for r in C.read_jsonl(rd/'train.jsonl') if r['episode'] > s['base']]
    C.write_json(out/'v26_train_logs.json', logs)
    if logs:
        lines += ['', '## 继续微调的训练过程（每 50 个窗口的平均代价 / 策略相对熵）', '',
                  '| 版本 | 阶段 | 种子 | 窗口数 | 分段均值 |', '|---|---|---|---|---|']
        for variant in V.VARIANTS:
            for stage in V.LABELS.values():
                for seed in seeds:
                    rows = [r for r in logs if (r['variant'], r['stage'], r['seed']) == (variant, stage, seed)]
                    if not rows:
                        continue
                    step = max(1, (s['total']-s['base'])//4)
                    parts = []
                    for i in range(0, len(rows), step):
                        chunk = rows[i:i+step]
                        ent = [r['actor_relative_entropy'] for r in chunk if r.get('actor_relative_entropy') is not None]
                        parts.append(f'{st.mean(r["cost_per_s"] for r in chunk):.3f} / '
                                     + (f'{st.mean(ent):.2f}' if ent else '-'))
                    lines.append(f'| {variant} | {stage} | {seed} | {len(rows)} | ' + '；'.join(parts) + ' |')
    (out/'v26_summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return out/'v26_summary.md'
