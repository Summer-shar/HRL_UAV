"""Torch-free test of the controlled burst (p2/burst.py) with the scripted CA policy.

Same scenario with and without the burst: arrivals before the window must be
identical, arrivals inside the window must increase, the burst must hit the
region with the largest expected demand, and the scene must be left unchanged.
Usage:  python tests/test_burst.py
"""
import importlib
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

from p2 import burst as B, evaluate, pipeline, report  # noqa: E402


def main():
    root = Path(tempfile.mkdtemp(prefix='p2burst_'))
    d = C.design(smoke=True)
    win = B.window(smoke=True)
    make = evaluate.factory('CA', d['seeds'][0], None, None)
    scenario = d['sweep_scenarios'][0]
    base = evaluate.evaluate_one(make, C.load_scene(), 'test', scenario, root/'plain.json', smoke=True)
    scene = C.load_scene()
    before = [dict(p['activity_parameters']) for p in scene['business_profiles']['catalog']]
    hot = evaluate.evaluate_one(make, scene, 'test', scenario, root/'burst.json', smoke=True, burst=win)
    assert [p['activity_parameters'] for p in scene['business_profiles']['catalog']] == before, 'scene modified'
    info, bins = hot['burst'], hot['burst_bins']
    assert info['region'] == int(np.argmax(info['region_demand_gcycles'])), info
    assert info['busy_multiplier'] == 3.0

    # reference bins of the run without burst, recomputed on the same grid
    from hrl_v2.physics_adapter import PhysicsAdapter
    from hrl_v2.collector import collect_metrics
    adapter = PhysicsAdapter(C.load_scene(), timing=C.clocks(C.load_scene(), True))
    adapter.reset(**C.scenario_seeds('test', scenario))
    collect_metrics(adapter, make(None, 'test', scenario), identity={}, reward_scale=100.)
    ref = B.record(adapter.env, win['bin_s'], adapter.env.time_s)
    edges = np.asarray(bins['edges'][:-1])
    n_hot = np.asarray(bins['high_n']) + np.asarray(bins['low_n'])
    n_ref = np.asarray(ref['high_n']) + np.asarray(ref['low_n'])
    pre = edges < win['start_s']
    inside = (edges >= win['start_s']) & (edges < win['end_s'])
    assert np.array_equal(n_hot[pre], n_ref[pre]), (n_hot, n_ref)
    assert n_hot[inside].sum() > 1.3*n_ref[inside].sum(), (n_hot[inside], n_ref[inside])
    print('arrivals per bin, plain:', n_ref.astype(int).tolist())
    print('arrivals per bin, burst:', n_hot.astype(int).tolist())
    print('burst region', info['region'], 'demand', np.round(info['region_demand_gcycles'], 2))
    print('plain delay %.3f s, burst delay %.3f s' % (base['truncated_delay_s'], hot['truncated_delay_s']))

    # job + report plumbing
    evaluate.burst_job(root, d['seeds'][0], 'CA', smoke=True)
    t = report.burst_table(root, d['seeds'][:1], ['CA'], smoke=True)
    assert t['bins'] and {w['window'] for w in t['windows']} == {'before', 'burst', 'after'}, t['windows']
    ids = [j.id for j in pipeline.plan(d['seeds'], smoke=True)]
    assert any('_burst_' in i for i in ids)
    print('OK; outputs in', root)


if __name__ == '__main__':
    main()
