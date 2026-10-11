"""v2.4.0 physics and gate contracts: terminal mobility, U2U budget, report-only unfinished."""
import copy
import json
import unittest

import numpy as np

from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
from hrl_v2.gates import g1
from hrl_v2.paths import source_root
from hrl_v2.run import read_config
from hrl_v2.tests.test_physical_contract import quiet_scene, step

ROOT = source_root().parent
CONFIGS = source_root()/'hrl_v2/configs'


def static_scene():
    scene = quiet_scene()
    scene['base_scene'].pop('terminal_mobility')
    return scene


def run(env, steps):
    for _ in range(steps):
        step(env)
    return env.physics.user_coords.copy()


class MobilityTests(unittest.TestCase):
    def test_moves_only_on_one_second_ticks_and_is_seeded(self):
        env = HotspotIndustrialEnv(quiet_scene())
        env.reset(4101, 3)
        start = env.physics.user_coords.copy()
        self.assertEqual(env._mobility.steps, 10)
        np.testing.assert_array_equal(run(env, 9), start)
        moved = run(env, 1)
        self.assertTrue(np.all(np.linalg.norm(moved-start, axis=1) > 0))
        np.testing.assert_array_equal(run(env, 9), moved)
        again = HotspotIndustrialEnv(quiet_scene())
        again.reset(4101, 3)
        np.testing.assert_array_equal(run(again, 200), run(env, 181))
        other = HotspotIndustrialEnv(quiet_scene())
        other.reset(4102, 3)
        self.assertFalse(np.array_equal(run(other, 200), again.physics.user_coords))

    def test_stays_inside_map_and_keeps_association(self):
        env = HotspotIndustrialEnv(quiet_scene())
        env.reset(7, 7)
        allocation = env.physics.user_allocations.copy()
        size = env.config['map_size_m']
        for _ in range(60):
            xy = run(env, 100)
            self.assertTrue(np.all((xy >= 0) & (xy <= size)))
        np.testing.assert_array_equal(env.physics.user_allocations, allocation)
        report = env.metrics()['terminal_mobility']
        self.assertGreater(report['mean_displacement_m'], 0)

    def test_reports_publish_current_positions(self):
        env = HotspotIndustrialEnv(quiet_scene())
        env.reset(11, 11)
        run(env, 2*env.frame_steps)
        self.assertEqual([r.position_m for r in env._terminal_reports],
                         [tuple(xy) for xy in env.physics.user_coords.tolist()])

    def test_static_scene_has_no_mobility_and_legacy_positions(self):
        env = HotspotIndustrialEnv(static_scene())
        env.reset(11, 11)
        start = env.physics.user_coords.copy()
        self.assertIsNone(env._mobility)
        np.testing.assert_array_equal(run(env, 2*env.frame_steps), start)
        self.assertEqual([r.position_m for r in env._terminal_reports],
                         [t.position_m for t in env._scene.terminals])
        self.assertNotIn('terminal_mobility', env.metrics())

    def test_invalid_mobility_is_rejected(self):
        for key, value in (('update_s', .15), ('update_s', 0.), ('speed_memory', 1.5),
                           ('sigma_v_mps', -1.), ('boundary', 'wrap'), ('extra', 1.), ('v_mean_mps', float('nan'))):
            scene = quiet_scene()
            scene['base_scene']['terminal_mobility'][key] = value
            with self.assertRaises(ValueError, msg=(key, value)):
                HotspotIndustrialEnv(scene)


class U2UBudgetTests(unittest.TestCase):
    def test_single_active_link_gets_full_budget(self):
        p = HotspotIndustrialEnv(quiet_scene()).physics
        active = np.zeros((3, 3), dtype=bool); active[0, 1] = active[0, 2] = active[1, 0] = True
        power = p.allocate_u2u_resources(active)['u2u_link_power_w']
        np.testing.assert_array_equal(power, [[0, .5, .5], [1., 0, 0], [0, 0, 0]])


class ConfigAndGateTests(unittest.TestCase):
    def test_physics_binding(self):
        for name in ('v2_4_0.json', 'v2_4_0_lite.json'):
            read_config(CONFIGS/name, ROOT)
        with self.assertRaisesRegex(ValueError, 'physics version'):
            read_config(CONFIGS/'v2_3_0_lite.json', ROOT)

    def test_g1_report_only_ignores_unfinished(self):
        def rows(delay, unfinished):
            return {str(k): dict(truncated_delay_s=delay, unfinished_parents=unfinished, episodes=2,
                priority=[dict(unfinished=unfinished), dict(unfinished=0)]) for k in range(12)}
        hover, baselines = rows(1., 10), dict(hover_equal=rows(2., 4), hover_local=rows(3., 0))
        self.assertFalse(g1(hover, baselines, 9)['passed'])
        result = g1(hover, baselines, 9, unfinished='report_only')
        self.assertTrue(result['passed'])
        self.assertEqual(result['unfinished']['per_episode']['candidate']['total'], 60.)
        self.assertEqual(result['unfinished']['per_episode']['hover_equal']['by_priority'], [24., 0.])
        self.assertFalse(result['unfinished']['candidate_within_equal'])
        with self.assertRaises(ValueError):
            g1(hover, baselines, 9, unfinished='drop')


if __name__ == '__main__':
    unittest.main()
