"""Torch-free test of the design-v4 zero-shot sweep (nominal policies evaluated unchanged at every point).

Checks: point roots (scene.json design v4, the nominal point refused, a v3 root
refused), the job plan (one job per point, seed and method; slow jobs first;
terminals instead of the superseded users design), zero-shot evaluation of the
scripted CA method in each point's own scene on the full test split, the report
(zeroshot_* files, paired tests, per-seed values, retrained = False, no training
checks, a stale users table renamed superseded_*), the consistency checks
(p2/consistency.py: identical records pass, a changed main-run record fails), and
the ZeroShot stage with a stand-in run.py (stops while points are missing, passes
once every record exists).
Usage:  python tests/test_zeroshot.py
"""
import importlib
import json
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from p2 import common as C  # noqa: E402

try:
    importlib.import_module('torch')
except ImportError:
    stub = types.ModuleType('hrl_v2.hybrid_simplex')
    stub.SUPPORTS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
    sys.modules['hrl_v2.hybrid_simplex'] = stub

from p2 import consistency, evaluate, feasibility as F, pipeline, report, retrain  # noqa: E402

FAKE = 'import sys\nprint("fake", " ".join(sys.argv[1:]))\n'


def main():
    tmp = Path(tempfile.mkdtemp(prefix='p2zs_'))
    main_root, base = tmp/'main', tmp/'zs'
    d = C.design(smoke=True)
    seed = d['seeds'][0]
    points = retrain.zeroshot_points(smoke=True)
    assert [p for p, _ in points] == list(C.ZEROSHOT_ORDER) and ('terminals', 7) in points, points
    roots = [retrain.prepare_zeroshot_point(base, p, v, main_root, smoke=True) for p, v in points]
    assert retrain.prepare_zeroshot_point(base, *points[0], main_root, smoke=True) == roots[0]
    assert C.read_json(roots[0]/'scene.json')['design'] == 'v4'
    for root in roots:
        assert C.test_scenarios(root, d) == tuple(d['test'])
    try:
        retrain.prepare_zeroshot_point(base, 'cpu_ghz', 10.0, main_root)
    except ValueError as e:
        print('refused as expected:', e)
    else:
        raise AssertionError('the nominal point must be refused')
    v3 = retrain.prepare_point(tmp/'v3', 'cpu_ghz', 9.0, smoke=True)
    try:
        evaluate.zeroshot_test_job(v3, main_root, seed, 'CA', smoke=True)
    except ValueError as e:
        print('refused as expected:', e)
    else:
        raise AssertionError('a design-v3 root must be refused')

    full = retrain.zeroshot_points()
    assert len(full) == 16 and not any(p == 'users' for p, _ in full)
    assert [v for p, v in full if p == 'terminals'] == [7, 9, 11, 15]
    jobs = pipeline.zeroshot_plan(base, full, C.SEEDS, main_root)
    assert len(jobs) == len(full)*len(C.SEEDS)*len(C.METHODS) and len({j.id for j in jobs}) == len(jobs)
    assert all(j.args[0] == 'zeroshot-test' and f'--main={main_root}' in j.args and not j.deps for j in jobs)
    top = max(jobs, key=lambda j: j.priority)
    assert top.id.startswith(('u2u_mhz=1.5/', 'u2u_mhz=2/')) and top.id.endswith(('_TOO', '_CA')), top.id

    # CA evaluated zero-shot in every point's scene (scripted, no checkpoints needed)
    for root in roots:
        C.set_active_root(root)
        evaluate.zeroshot_test_job(root, main_root, seed, 'CA', smoke=True)
        C.set_active_root(None)
    rec = sorted((C.seed_dir(roots[2], seed)/'test'/'CA').glob('*.json'))
    assert [int(r.stem) for r in rec] == list(d['test'])
    row = C.read_json(rec[0])
    assert row['users'] == 7 and row['trained_at'].startswith('nominal'), row
    assert abs(row['load'] - C.terminals_demand(7)/C.MEAN_DEMAND_GCYCLES) < 1e-9, row['load']   # total load follows K
    assert C.read_json(C.seed_dir(roots[3], seed)/'test'/'CA'/f'{d["test"][0]:02d}.json')['u2u_mhz'] == 1.5

    # report with fabricated learned methods
    evaluate.test_job(main_root, seed, 'CA', smoke=True)
    for root in [main_root] + roots:
        for i, m in enumerate(['Proposed', 'TFOO', 'TOO', 'NC', 'HoverSAC']):
            for src in (C.seed_dir(root, seed)/'test'/'CA').glob('*.json'):
                r = json.loads(src.read_text())
                r['truncated_delay_s'] *= .8 + .1*i
                r['cost_per_s'] *= .8 + .1*i
                r['method'] = m
                C.write_json(C.seed_dir(root, seed)/'test'/m/src.name, r)
    (base/'results').mkdir(parents=True, exist_ok=True)
    (base/'results'/'zeroshot_sweep_users.csv').write_text('stale\n')
    res = report.retrain_build(main_root, base, [seed], points, smoke=True, kind='zeroshot')
    for p in C.ZEROSHOT_ORDER:
        assert (res/f'zeroshot_sweep_{p}.csv').exists(), p
    assert (res/'zeroshot_paired.csv').exists() and not (res/'retrain_convergence.csv').exists()
    assert not (res/'zeroshot_sweep_users.csv').exists() and (res/'superseded_zeroshot_sweep_users.csv').exists()
    import csv
    rows = list(csv.DictReader(open(res/'zeroshot_sweep_terminals.csv', encoding='utf-8')))
    assert {r['retrained'] for r in rows} == {'False'} and {r['terminals'] for r in rows} == {'21', '39'}, rows[0]
    assert rows[0]['truncated_delay_s_min_over_seeds'] and rows[0]['truncated_delay_s_max_over_seeds']
    seed_rows = list(csv.DictReader(open(res/'zeroshot_per_seed.csv', encoding='utf-8')))
    assert {r['param'] for r in seed_rows} == set(C.ZEROSHOT_ORDER) and {r['seed'] for r in seed_rows} == {str(seed)}
    summary = (res/'zeroshot_summary.md').read_text()
    assert summary.startswith('# Zero-shot sweep (design v4') and '(zero-shot;' in summary, summary[:300]
    assert 'Weighted delay per training seed (terminals' in summary and '| 7 | 0.46 |' in summary, summary
    print(summary[:700])

    # consistency checks (CA only: scripted, no checkpoints): the zero-shot path at the nominal scene vs
    # the main run's records, and the terminals=7 point recomputed vs its records
    kw = dict(methods=('CA',), regression_methods=('CA',), regression_points=(('terminals', 7), ('users', 10)))
    assert consistency.run(base, main_root, [seed], 2, smoke=True, **kw) == 0
    out = C.read_json(base/'_zeroshot'/'consistency.json')
    assert out['all_identical'] and [c['kind'] for c in out['checks']] == ['nominal', 'regression'], out
    assert out['compared'] == 2*len(d['test']) and out['skipped_regression_points'] == ['users=10'], out
    main_rec = C.seed_dir(main_root, seed)/'test'/'CA'/f'{d["test"][0]:02d}.json'
    saved = main_rec.read_text()
    r = json.loads(saved)
    r['truncated_delay_s'] += 1e-12
    main_rec.write_text(json.dumps(r))
    assert consistency.run(base, main_root, [seed], 2, smoke=True, **kw) == 1
    out = C.read_json(base/'_zeroshot'/'consistency.json')
    assert not out['all_identical'] and out['value_differences'] == 1, out
    assert 'truncated_delay_s' in out['checks'][0]['differences'][0], out['checks'][0]
    main_rec.write_text(saved)
    print('consistency checks OK')

    # the stage with a stand-in run.py
    fake = tmp/'fake_run.py'
    fake.write_text(FAKE)
    stage = F.ZeroShot(main_root, tmp/'stage', 2, tuple(C.SEEDS), publish=False)
    stage.run_py = str(fake)
    assert stage.run() == 1
    md = (tmp/'stage'/'ZEROSHOT.md').read_text(encoding='utf-8')
    assert '16 个点未完成' in md and 'records: 0/5760' in md and 'terminals=7' in md, md
    for p, v in full:                                     # every record present -> passes
        root = tmp/'stage'/pipeline.point_name(p, v)
        for s in C.SEEDS:
            for m in C.METHODS:
                for sc in C.design()['test']:
                    path = C.seed_dir(root, s)/'test'/m/f'{sc:02d}.json'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('{}')
    stage = F.ZeroShot(main_root, tmp/'stage', 2, tuple(C.SEEDS), publish=False)
    stage.run_py = str(fake)
    assert stage.run() == 0
    md = (tmp/'stage'/'ZEROSHOT.md').read_text(encoding='utf-8')
    assert '零样本扫参完成' in md and 'records: 5760/5760' in md, md[:600]
    st = json.loads((tmp/'stage'/'_zeroshot'/'state.json').read_text())
    assert st['steps']['smoke']['status'] == 'skipped'      # passed in the first run, same code
    assert st['steps']['consistency']['status'] == 'skipped' and '逐位一致性检查' in md
    print('OK; outputs in', tmp)


if __name__ == '__main__':
    main()
