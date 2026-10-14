"""Torch-free test of retrained sweep points (p2/retrain.py, pipeline.retrain_plan, report.retrain_build).

Checks the per-root scene override, the shared multi-root job graph, that a
point root evaluates on the scaled scene and the sweep scenarios, and the
cross-root report.  Learned results are fabricated from scripted CA runs.
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
    assert all(v != C.NOMINAL[p] for p, v in points) and len(points) == 3, points
    roots = [retrain.prepare_point(base, p, v, smoke=True) for p, v in points]
    assert retrain.prepare_point(base, *points[0], smoke=True) == roots[0]          # idempotent

    # scene override is per root, nominal scene untouched
    u2u_root = base/pipeline.point_name('u2u_mhz', 2.0)
    assert C.load_scene(u2u_root)['base_scene']['resources']['u2u_bandwidth_hz'] == 2e6
    assert C.load_base_scene()['base_scene']['resources']['u2u_bandwidth_hz'] == 3e6
    assert C.load_scene(main_root) == C.load_base_scene()
    assert C.test_scenarios(u2u_root, d) == tuple(d['sweep_scenarios'])
    assert C.test_scenarios(main_root, d) == tuple(d['test'])
    assert C.finetune(u2u_root)['recipe'] == 'v2' and C.prune_snapshots(u2u_root)
    assert not C.prune_snapshots(main_root)

    # one job graph for all points; first points boosted; markers root-local
    first = [('u2u_mhz', 2.0)]
    jobs = pipeline.retrain_plan(base, points, d['seeds'], smoke=True, first=first)
    ids = [j.id for j in jobs]
    assert len(ids) == len(set(ids)) and all(dep in ids for j in jobs for dep in j.deps)
    assert not any('_sweep_' in i or '_burst_' in i or '_sat_' in i for i in ids)
    for j in jobs:
        assert j.id == f'{Path(j.out).name}/{j.marker}', (j.id, j.out, j.marker)
    boosted = {j.priority for j in jobs if j.id.startswith('u2u_mhz=2/')}
    others = {j.priority for j in jobs if not j.id.startswith('u2u_mhz=2/')}
    assert min(boosted) > max(others), (boosted, others)
    print(f'retrain plan: {len(jobs)} jobs for {len(points)} points')

    # a point root evaluates the scaled scene on the sweep scenarios
    C.set_active_root(u2u_root)
    evaluate.test_job(u2u_root, seed, 'CA', smoke=True)
    C.set_active_root(None)
    rec = sorted((C.seed_dir(u2u_root, seed)/'test'/'CA').glob('*.json'))
    assert [int(r.stem) for r in rec] == list(d['sweep_scenarios'])
    row = C.read_json(rec[0])
    assert row['u2u_mhz'] == 2.0 and row['load'] == 1.0 and row['cpu_ghz'] == 10.0, row
    for bad in (lambda: evaluate.sweep_job(u2u_root, seed, 'CA', 'cpu_ghz', smoke=True),
                lambda: evaluate.burst_job(u2u_root, seed, 'CA', smoke=True)):
        try:
            bad()
        except ValueError as e:
            print('refused as expected:', e)
        else:
            raise AssertionError('nested sweep / burst in a point root must be refused')

    # cross-root report with fabricated learned methods
    evaluate.test_job(main_root, seed, 'CA', smoke=True)
    for root in roots:
        if root != u2u_root:
            C.set_active_root(root)
            evaluate.test_job(root, seed, 'CA', smoke=True)
            C.set_active_root(None)
    for root in [main_root] + roots:
        for i, m in enumerate(x for x in C.METHODS if x != 'CA'):
            for src in (C.seed_dir(root, seed)/'test'/'CA').glob('*.json'):
                r = json.loads(src.read_text())
                r['truncated_delay_s'] *= 1 + .1*i
                r['method'] = m
                C.write_json(C.seed_dir(root, seed)/'test'/m/src.name, r)
    res = report.retrain_build(main_root, base, [seed], points, smoke=True)
    for p in d['sweeps']:
        t = [json.loads(json.dumps(x)) for x in report.retrain_table(main_root, base, [seed], p, d)]
        assert {x['retrained'] for x in t} == {True, False}
        assert (res/f'retrain_sweep_{p}.csv').exists(), p
    print((res/'retrain_summary.md').read_text()[:800])

    # export (records only) from a split run, merged elsewhere, gives the same tables
    import zipfile
    for junk in ('seed_%d/train/M1/final.pt' % seed, 'seed_%d/test/CA/traj/21.npz' % seed, '_jobs/x.log'):
        (u2u_root/junk).parent.mkdir(parents=True, exist_ok=True)
        (u2u_root/junk).write_text('junk')
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
    for p in d['sweeps']:
        a = (res/f'retrain_sweep_{p}.csv').read_text().replace(str(base), '')
        b = (res2/f'retrain_sweep_{p}.csv').read_text().replace(str(merged), '')
        assert a == b, p
    print('OK; outputs in', tmp)


if __name__ == '__main__':
    main()
