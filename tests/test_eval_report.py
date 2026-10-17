"""Torch-free plumbing test: scripted evaluation, sweeps, tables, figures, job plan.

Learned methods need torch; here their result files are fabricated from the
scripted CA runs purely to exercise the report and plotting code.
Usage:  python tests/test_eval_report.py
"""
import importlib
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np

from p2 import common as C

try:
    importlib.import_module('torch')
except ImportError:   # collector imports SUPPORTS from a torch module; stub it
    stub = types.ModuleType('hrl_v2.hybrid_simplex')
    stub.SUPPORTS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
    sys.modules['hrl_v2.hybrid_simplex'] = stub

from p2 import evaluate, pipeline, plots, report  # noqa: E402


def fabricate(root, seed, method, source='CA', scale=1.):
    for src in C.seed_dir(root, seed).glob(f'**/{source}/*.json'):
        dst = src.parent.with_name(method) / src.name
        row = json.loads(src.read_text())
        for k in ('truncated_delay_s', 'high_truncated_delay_s', 'energy_w_compute_comm', 'cost_per_s'):
            if row.get(k) is not None:
                row[k] *= scale
        row['method'] = method
        C.write_json(dst, row)
    tsrc = C.seed_dir(root, seed)/'test'/source/'traj'
    if tsrc.exists():
        shutil.copytree(tsrc, C.seed_dir(root, seed)/'test'/method/'traj', dirs_exist_ok=True)


def main():
    root = Path(tempfile.mkdtemp(prefix='p2test_'))
    d = C.design(smoke=True)
    seed = d['seeds'][0]
    evaluate.test_job(root, seed, 'CA', smoke=True)
    for p in d['sweeps']:
        evaluate.sweep_job(root, seed, 'CA', p, smoke=True)
    rows = sorted((C.seed_dir(root, seed)/'test'/'CA').glob('*.json'))
    assert len(rows) == len(d['test']), rows
    r = C.read_json(rows[0])
    print('CA smoke test row:', {k: r[k] for k in ('scenario', 'truncated_delay_s', 'energy_w_compute_comm',
                                                    'offloaded_fraction', 'uav_path_m', 'wall_seconds')})
    assert r['uav_path_m'] == 0. and abs(r['offloaded_fraction']-2/3) < 1e-9
    # second call must reuse the record, not recompute
    again = evaluate.evaluate_one(None, None, 'test', d['test'][0], rows[0])
    assert again == r
    # sweep files
    for p, values in d['sweeps'].items():
        for v in values:
            if v != C.NOMINAL[p]:
                assert list((C.seed_dir(root, seed)/'sweep'/f'{p}={v:g}'/'CA').glob('*.json')), (p, v)
    for i, m in enumerate(m for m in C.METHODS if m != 'CA'):
        fabricate(root, seed, m, scale=1.+.1*i)
        rd = C.run_dir(root, seed, {'Proposed': 'M2'}.get(m, m))
        for ep in range(1, 21):
            C.append_jsonl(rd/'train.jsonl', dict(episode=ep, truncated_delay_s=1./ep+.1*i))
    res = report.build(root, [seed], smoke=True)
    figs = plots.build(root, [seed], smoke=True)
    for name in ('main_table.csv', 'stats.json', 'summary.md', 'sweep_cpu_ghz.csv', 'sweep_load.csv'):
        assert (res/name).exists(), name
    for name in ('fig_main_bars', 'fig_trajectory', 'fig_sweep_cpu_ghz', 'fig_sweep_load', 'fig_convergence',
                 'fig_cooperation'):
        assert (figs/f'{name}.png').exists(), name
    print((res/'summary.md').read_text()[:1500])
    # job plan: every dependency exists, ids unique
    for smoke in (True, False):
        jobs = pipeline.plan(C.design(smoke)['seeds'], smoke=smoke)
        ids = [j.id for j in jobs]
        assert len(ids) == len(set(ids))
        assert all(dep in ids for j in jobs for dep in j.deps)
        print(f'plan smoke={smoke}: {len(jobs)} jobs')
    print('OK; outputs in', root)


if __name__ == '__main__':
    main()
