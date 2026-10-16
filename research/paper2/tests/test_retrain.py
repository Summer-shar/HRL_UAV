"""Torch-free test of design-v3 retrained sweep points (p2/retrain.py, pipeline.retrain_plan, report.retrain_build).

Checks the per-root scene override (cpu, load and users points), the shared
multi-root job graph and its priorities (pilot, then cpu, load, users), that a
point root evaluates its own scene on the full test split, the cross-root
report with paired tests, and the records-only export/merge.  Learned results
are fabricated from scripted CA runs.
Usage:  python tests/test_retrain.py
"""
import importlib
import json
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2 import common as C  # noqa: E402

try:
    importlib.import_module('torch')
except ImportError:
    stub = types.ModuleType('hrl_v2.hybrid_simplex')
    stub.SUPPORTS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
    sys.modules['hrl_v2.hybrid_simplex'] = stub

from p2 import evaluate, pipeline, report, retrain  # noqa: E402


def main():
    tmp = Path(tempfile.mkdtemp(prefix='p2retrain_'))
    base, main_root = tmp/'retrain', tmp/'main'
    d = C.design(smoke=True)
    seed = d['seeds'][0]
    points = retrain.all_points(smoke=True)
    assert points == [('cpu_ghz', 9.0), ('load', 0.8), ('users', 7)], points
    full = retrain.all_points()
    assert full[0] == C.PILOT_POINT and len(full) == 12 and ('users', 19) in full, full
    roots = [retrain.prepare_point(base, p, v, smoke=True) for p, v in points]
    assert retrain.prepare_point(base, *points[0], smoke=True) == roots[0]          # idempotent
    for bad in (lambda: retrain.prepare_point(base, 'users', 13), lambda: retrain.parse_points('users=7.5')):
        try:
            bad()
        except ValueError as e:
            print('refused as expected:', e)
        else:
            raise AssertionError('nominal point / fractional users must be refused')
    assert retrain.parse_points('cpu_ghz=9,users=19') == [('cpu_ghz', 9.0), ('users', 19)]

    # scene override is per root, nominal scene untouched
    cpu_root, load_root, users_root = roots
    assert C.load_scene(cpu_root)['base_scene']['resources']['cpu_hz'] == [9e9]*3
    assert C.load_scene(users_root)['base_scene']['num_users'] == 21
    assert C.load_base_scene()['base_scene']['num_users'] == 39
    assert C.load_scene(main_root) == C.load_base_scene()
    for root in roots + [main_root]:
        assert C.test_scenarios(root, d) == tuple(d['test'])
    assert C.finetune(users_root)['recipe'] == 'v2' and C.prune_snapshots(users_root)
    assert not C.prune_snapshots(main_root)
    design = C.read_json(users_root/'design.json')
    assert design['point'] == dict(param='users', value=7, design='v3', prune_snapshots=True)
    assert abs(design['demand_over_capacity'] - C.utilization('cpu_ghz', 10.)) < 1e-12 and design['scene_sha256']

    # one job graph for all points; pilot first, then cpu > load > users; markers root-local
    jobs = pipeline.retrain_plan(base, points, d['seeds'], smoke=True)
    ids = [j.id for j in jobs]
    assert len(ids) == len(set(ids)) and all(dep in ids for j in jobs for dep in j.deps)
    assert not any('_sweep_' in i or '_burst_' in i or '_sat_' in i for i in ids)
    for j in jobs:
        assert j.id == f'{Path(j.out).name}/{j.marker}', (j.id, j.out, j.marker)
    prio = {name: {j.priority for j in jobs if j.id.startswith(name+'/')} for name in
            (pipeline.point_name(*p) for p in points)}
    assert min(prio['cpu_ghz=9']) > max(prio['load=0.8']) and min(prio['load=0.8']) > max(prio['users=7']), prio
    print(f'retrain plan: {len(jobs)} jobs for {len(points)} points')

    # a point root evaluates its own scene on the test split
    for root in roots:
        C.set_active_root(root)
        evaluate.test_job(root, seed, 'CA', smoke=True)
        C.set_active_root(None)
    rec = sorted((C.seed_dir(users_root, seed)/'test'/'CA').glob('*.json'))
    assert [int(r.stem) for r in rec] == list(d['test'])
    row = C.read_json(rec[0])
    assert row['users'] == 7 and abs(row['load']-1.) < 1e-9 and row['cpu_ghz'] == 10.0, row
    row = C.read_json(C.seed_dir(cpu_root, seed)/'test'/'CA'/f'{d["test"][0]:02d}.json')
    assert row['cpu_ghz'] == 9.0 and 'users' not in row
    for bad in (lambda: evaluate.sweep_job(users_root, seed, 'CA', 'cpu_ghz', smoke=True),
                lambda: evaluate.burst_job(users_root, seed, 'CA', smoke=True)):
        try:
            bad()
        except ValueError as e:
            print('refused as expected:', e)
        else:
            raise AssertionError('nested sweep / burst in a point root must be refused')

    # cross-root report with fabricated learned methods (Proposed best, as a plumbing check)
    evaluate.test_job(main_root, seed, 'CA', smoke=True)
    for root in [main_root] + roots:
        for i, m in enumerate(['Proposed', 'TFOO', 'TOO', 'NC', 'HoverSAC']):
            for src in (C.seed_dir(root, seed)/'test'/'CA').glob('*.json'):
                r = json.loads(src.read_text())
                r['truncated_delay_s'] *= .8 + .1*i
                r['cost_per_s'] *= .8 + .1*i
                r['method'] = m
                C.write_json(C.seed_dir(root, seed)/'test'/m/src.name, r)
    res = report.retrain_build(main_root, base, [seed], points, smoke=True)
    for p in C.RETRAIN_ORDER:
        t = [json.loads(json.dumps(x)) for x in report.retrain_table(main_root, base, [seed], p, d)]
        assert {x['retrained'] for x in t} == {True, False}, p
        assert {x['scenarios'] for x in t} == {len(d['test'])}
        assert (res/f'retrain_sweep_{p}.csv').exists(), p
    users_rows = report.retrain_table(main_root, base, [seed], 'users', d)
    assert {x['terminals'] for x in users_rows} == {21, 39}
    pairs = C.read_json(res/'retrain_stats.json')['paired']
    assert all(x['relative_change'] < 0 for x in pairs if x['metric'] == 'truncated_delay_s')
    print((res/'retrain_summary.md').read_text()[:900])

    # export (records only) from a split run, merged elsewhere, gives the same tables
    import zipfile
    for junk in ('seed_%d/train/M1/final.pt' % seed, 'seed_%d/test/CA/traj/21.npz' % seed, '_jobs/x.log'):
        (users_root/junk).parent.mkdir(parents=True, exist_ok=True)
        (users_root/junk).write_text('junk')
    try:
        retrain.export(base, tmp/'none.zip', points, [seed], 'A', smoke=True)     # no report.done yet
    except ValueError as e:
        print('refused as expected:', e)
    else:
        raise AssertionError('points without report.done must not be exported')
    merged = tmp/'merged'
    for machine, part in (('A', points[:1]), ('B', points[1:])):
        info = retrain.export(base, tmp/f'export_{machine}.zip', part, [seed], machine, smoke=True,
                              require_report=False)
        names = zipfile.ZipFile(info['dest']).namelist()
        assert f'manifest_{machine}.json' in names and not any(n.endswith(('.pt', '.npz', '.log')) for n in names)
        man = json.loads(zipfile.ZipFile(info['dest']).read(f'manifest_{machine}.json'))
        assert man['finished'] == [pipeline.point_name(*p) for p in part] and man['environment']['code_sha256']
        zipfile.ZipFile(info['dest']).extractall(merged)
    res2 = report.retrain_build(main_root, merged, [seed], points, smoke=True)
    for p in C.RETRAIN_ORDER:
        a = (res/f'retrain_sweep_{p}.csv').read_text().replace(str(base), '')
        b = (res2/f'retrain_sweep_{p}.csv').read_text().replace(str(merged), '')
        assert a == b, p
    print('OK; outputs in', tmp)


if __name__ == '__main__':
    main()
