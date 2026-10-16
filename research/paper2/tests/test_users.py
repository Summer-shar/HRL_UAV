"""Terminals per region as a scene parameter (design-v3 users sweep, v2.5.0 simulator).

For every users value of the design: the generated scene passes the simulator's
own validation, every region has K terminals numbered region by region, the
cluster / background split and the type counts follow the nominal template,
the expected computation load of every profile permutation equals the nominal
24.725 Gcycles/s, every terminal lies in its region, the uplink bandwidth per
terminal is B_G / (3K), and a short closed-loop run produces Lite observations
of width 3K x 17 + 66.  K = 13 must return the nominal scene unchanged.
Usage:  python tests/test_users.py
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from p2 import common as C


def main():
    from hrl_uav.industrial_hotspot_scene import validate_hotspot_config, generate_realization, effective_config
    from hrl_uav.industrial_scene import workload_summary
    from hrl_v2.physics_adapter import PhysicsAdapter
    from hrl_v2.timing import ClockConfig
    from hrl_v2.lite_observation import LiteObserver, upper_dim

    base = C.load_scene()
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
