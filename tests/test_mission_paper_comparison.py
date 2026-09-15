import copy
import unittest

import numpy as np

from hrl_uav.bandwidth import UniformRouting
from hrl_uav.mission_paper_comparison import CONDITIONS, RandomRouting, condition_config, protocol
from hrl_uav.routing import project_rows_to_simplex
from hrl_uav.scene_prescreen import LocalRouting


class PaperComparisonTests(unittest.TestCase):
    def test_random_uses_private_repeatable_rng_and_feasible_full_simplex(self):
        before = np.random.get_state()
        left, right = RandomRouting(33600), RandomRouting(33600)
        actions = []
        for _ in range(100):
            a = left.choose_action(None, deterministic=True)
            b = right.choose_action(np.ones(68), deterministic=True)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(project_rows_to_simplex(a.reshape(2, 3, 3)), a.reshape(2, 3, 3), atol=1e-14)
            actions.append(a)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertGreater(np.std(actions), .1)
        self.assertEqual(left.calls, 100)

    def test_fixed_routes_match_definitions(self):
        local = project_rows_to_simplex(LocalRouting().choose_action(None).reshape(2, 3, 3))
        uniform = project_rows_to_simplex(UniformRouting().choose_action(None).reshape(2, 3, 3))
        np.testing.assert_array_equal(local, np.tile(np.eye(3), (2, 1, 1)))
        np.testing.assert_allclose(uniform, 1/3)

    def test_only_declared_parameter_fields_change(self):
        cfg = dict(phase_loads=[14.375, 6.325, 4.025], environment=dict(u2u_bandwidth_mhz=6, untouched=1),
                   reward=dict(delay_weight=.9), cpu_ghz=10)
        original = copy.deepcopy(cfg)
        for name, scale, bandwidth in CONDITIONS:
            result = condition_config(cfg, name)
            np.testing.assert_allclose(result["phase_loads"], np.array(cfg["phase_loads"])*scale)
            self.assertEqual(result["environment"]["u2u_bandwidth_mhz"], bandwidth)
            result["phase_loads"] = cfg["phase_loads"]
            result["environment"]["u2u_bandwidth_mhz"] = 6
            self.assertEqual(result, cfg)
        self.assertEqual(cfg, original)

    def test_budget_and_protected_scenes(self):
        p = protocol()
        self.assertEqual(p["cases"], 5*(3*4+1)*2)
        self.assertEqual(p["cases"]-p["reused_cases"], p["new_evaluations"])
        self.assertFalse(p["training"])
        for scene in range(33700, 33710):
            with self.assertRaises(ValueError):
                RandomRouting(scene)
        with self.assertRaises(ValueError):
            condition_config({}, "unregistered")


if __name__ == "__main__":
    unittest.main()
