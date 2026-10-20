"""Terminals per region as a scene parameter (v2.5.0 simulator): the users design
(nominal total load; design v3 and the first v4 run) and the terminals design (nominal
load per terminal; design v4 since 2026-10-20).

For every users value of the design: the generated scene passes the simulator's
own validation, every region has K terminals numbered region by region, the
cluster / background split and the type counts follow the nominal template,
the expected computation load of every profile permutation equals the nominal
24.725 Gcycles/s, every terminal lies in its region, the uplink bandwidth per
terminal is B_G / (3K), and a short closed-loop run produces Lite observations
of width 3K x 17 + 66.  K = 13 must return the nominal scene unchanged.
The users scenes (and the cpu/load/u2u scenes) must be byte-identical to those of
commit 87e9a27, which produced the first zero-shot records (SHA-256 below).
terminals: the same terminals, clusters and type counts as users, but every
business multiplier unchanged, so the expected load is C.terminals_demand(K)
(from the simulator's workload_summary for every profile permutation) and
C.utilization equals it over the 30 Gcycles/s of the three UAVs.
Usage:  python tests/test_users.py
"""
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np

from p2 import common as C


SCENE_SHA256_87E9A27 = {
    ('users', 7): '5c18c5abdc84404086c5ccf222b8c2a041b2a45ee753b52ca474ffc8b8aa358b',
    ('users', 10): '43b62884cec3c3b60d8902b21c4f620340170ac268252d4ec14c8fd93c05533b',
    ('users', 16): '92efe9e593b6ccb6537158b827a45a17a7c7ea8135a95c94bda4d2f39754494f',
    ('users', 19): '5971c00a4cf2b5db8e10cdc6df2bb7624dabb99bdd6b2df1d7849391df2159e1',
    ('cpu_ghz', 11.0): 'd21560b1c3f453245ce021bb1c2cb2973d556d511a923fb57bde86b24fa8668b',
    ('load', 0.8): '44adaa4a70caaf42948ad03c252c6e7f7afb05ca3fab63c1d6a1fd1cd3a5fa77',
    ('u2u_mhz', 2.5): 'bd2aa8a9f3bad01ac20eb5e54b6b00c70a06d76357d91d011643f3134e21dc3c'}


def sha(scene):
    return hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def terminals(base):
    from hrl_uav.industrial_hotspot_scene import validate_hotspot_config, generate_realization, effective_config
    from hrl_uav.industrial_scene import workload_summary
    from hrl_v2.physics_adapter import PhysicsAdapter
    from hrl_v2.timing import ClockConfig
    assert C.scaled_scene(base, terminals=13) == base and C.utilization('terminals', 13) == C.utilization('load', 1.0)
    assert C.load_scene.__doc__ and C.point_value('terminals', '15') == 15
    try:
        C.scaled_scene(base, users=7, terminals=7)
    except ValueError:
        pass
    else:
        raise AssertionError('users and terminals together must be refused')
    previous = 0.
    for k in C.ZEROSHOT_SWEEPS['terminals']:
        s, u = C.scaled_scene(base, terminals=k), C.scaled_scene(base, users=k)
        validate_hotspot_config(s)
        for a, b, p0 in zip(s['business_profiles']['catalog'], u['business_profiles']['catalog'],
                            base['business_profiles']['catalog']):
            assert a['device_counts_by_type'] == b['device_counts_by_type']
            assert a['cluster_counts_by_type'] == b['cluster_counts_by_type'] and a['slots'] == b['slots']
            assert a['business_multiplier'] == p0['business_multiplier']
        assert s['base_scene']['region_multipliers'] == base['base_scene']['region_multipliers']
        assert s['geometry'] == u['geometry'] and s['base_scene']['num_users'] == 3*k
        loads = [workload_summary(effective_config(s, generate_realization(s, 7, 8, list(perm))))['total_compute_gcycles_s']
                 for perm in itertools.permutations(range(3))]
        assert np.allclose(loads, C.terminals_demand(k), rtol=1e-12, atol=0), (k, loads, C.terminals_demand(k))
        cap = 3*C.NOMINAL['cpu_ghz']
        assert abs(C.utilization('terminals', k) - loads[0]/cap) < 1e-12
        assert loads[0] > previous and loads[0]/cap < 0.95, (k, loads[0])
        previous = loads[0]
        a = PhysicsAdapter(s, timing=ClockConfig.from_scene(s, rollout_seconds=10.))
        a.reset(**C.scenario_seeds('test', 21))
        assert a.env.physics.M == 3*k
        assert abs(a.env._expected_workload['total_compute_gcycles_s'] - C.terminals_demand(k)) < 1e-9
        print(f'OK terminals={k:2d}: {3*k} terminals, load {loads[0]:.4f} Gcycles/s, demand/capacity '
              f'{C.utilization("terminals", k):.3f}, uplink {a.env.physics.B_gt_total/a.env.physics.M/1e3:.1f} kHz '
              'per terminal')
    assert 30.0 < C.terminals_demand(16)            # 16 per region would exceed the total UAV capacity


def main():
    from hrl_uav.industrial_hotspot_scene import validate_hotspot_config, generate_realization, effective_config
    from hrl_uav.industrial_scene import workload_summary
    from hrl_v2.physics_adapter import PhysicsAdapter
    from hrl_v2.timing import ClockConfig
    from hrl_v2.lite_observation import LiteObserver, upper_dim

    base = C.load_scene()
    for (param, value), digest in SCENE_SHA256_87E9A27.items():
        assert sha(C.scaled_scene(base, **{param: value})) == digest, (param, value)
    terminals(base)
    assert C.scaled_scene(base, users=13) == base and C.users_scene(base, 13) == base
    assert C.largest_remainder([5, 4, 4], 7) == [3, 2, 2] and sum(C.largest_remainder([8, 4, 1], 19)) == 19
    for k in C.RETRAIN_SWEEPS['users']:
        s = C.scaled_scene(base, users=k)
        validate_hotspot_config(s)
        g = s['geometry']
        assert g['cluster_count_per_region'] + g['background_count_per_region'] == k
        assert g['cluster_count_per_region'] == int(np.floor(9*k/13+.5))
        for profile in s['business_profiles']['catalog']:
            assert sum(profile['device_counts_by_type']) == k == len(profile['slots'])
            assert sum(profile['cluster_counts_by_type']) == g['cluster_count_per_region']
            assert all(q <= c for q, c in zip(profile['cluster_counts_by_type'], profile['device_counts_by_type']))
        loads = []
        for perm in itertools.permutations(range(3)):
            r = generate_realization(s, 7, 8, list(perm))
            assert [t['user_id'] for t in r['terminals']] == list(range(3*k))
            assert all(t['source'] == t['user_id']//k for t in r['terminals'])
            for t in r['terminals']:
                lo, hi = np.asarray(s['base_scene']['regions_m'][t['source']], float)
                assert np.all(np.asarray(t['position_m']) >= lo) and np.all(np.asarray(t['position_m']) <= hi)
            loads.append(workload_summary(effective_config(s, r))['total_compute_gcycles_s'])
        assert np.allclose(loads, C.MEAN_DEMAND_GCYCLES, rtol=1e-12, atol=0), (k, loads)
        a = PhysicsAdapter(s, timing=ClockConfig.from_scene(s, rollout_seconds=10.))
        a.reset(**C.scenario_seeds('test', 21))
        assert a.env.physics.M == 3*k and len(a.env.upload) == 3*k
        rates = a.env.physics.compute_u2g_rate_bps()
        assert rates.shape == (3*k,)
        obs = LiteObserver()
        for macro in range(2):
            a.start_macro(np.zeros(6))
            for frame in range(10):
                rep = a.prepare()
                o = obs.update(rep, a.arrivals)
                assert o.upper.shape == (upper_dim(3*k),) and o.lower.shape == (110,)
                for _ in range(5):
                    a.step(np.full((2, 3, 3), 1/3))
        print(f'OK users={k:2d}: {3*k} terminals, cluster/background {g["cluster_count_per_region"]}/'
              f'{g["background_count_per_region"]}, load {loads[0]:.6f} Gcycles/s, '
              f'uplink {a.env.physics.B_gt_total/a.env.physics.M/1e3:.1f} kHz per terminal, upper obs {o.upper.shape[0]}')
    print('OK')


if __name__ == '__main__':
    main()
