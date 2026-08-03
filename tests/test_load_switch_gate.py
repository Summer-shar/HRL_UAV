import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_load_switch_gate import GatePolicyController
from run_load_switch_gate import LOAD_PHASES_MBPS
from run_load_switch_gate import paired_summary
from run_load_switch_gate import phase_index_for_step
from run_load_switch_gate import switch_happens_at_step


class FakeBaseEnv:
    def __init__(self):
        self.N = 3
        self.zone_expected_offered_load_mbps = LOAD_PHASES_MBPS[0].copy()
        self.Q_loc_high = np.zeros(3)
        self.Q_loc_low = np.zeros(3)
        self.Q_tx_high = np.zeros((3, 3))
        self.Q_tx_low = np.zeros((3, 3))
        self.uav_coords = np.asarray(
            [[350.0, 350.0], [350.0, 650.0], [650.0, 500.0]],
            dtype=np.float64,
        )
        self.weighted_centroids = self.uav_coords.copy()

    def get_raw_state(self):
        return {
            "uav_coords": self.uav_coords.copy(),
            "weighted_centroids": self.weighted_centroids.copy(),
        }


class FakeWrapper:
    def __init__(self):
        self.env = FakeBaseEnv()
        self.macro_queue_norm = 10.0
        self.macro_action_mode = "centroid_waypoint"
        self.macro_waypoint_radius_m = 200.0


class LoadSwitchGateTests(unittest.TestCase):
    def test_phase_schedule_switches_before_steps_10_and_20(self):
        self.assertEqual(phase_index_for_step(0), 0)
        self.assertEqual(phase_index_for_step(9), 0)
        self.assertEqual(phase_index_for_step(10), 1)
        self.assertEqual(phase_index_for_step(19), 1)
        self.assertEqual(phase_index_for_step(20), 2)
        self.assertFalse(switch_happens_at_step(0))
        self.assertTrue(switch_happens_at_step(10))
        self.assertTrue(switch_happens_at_step(20))

    def test_p0_is_exact_zero_centroid_residual(self):
        env = FakeWrapper()
        controller = GatePolicyController("P0")
        controller.reset(env)
        controller.before_macro(env, 0, switched=False)
        np.testing.assert_allclose(controller.action(env), 0.0, atol=1e-9)

    def test_p1_waits_one_period_after_observed_switch(self):
        env = FakeWrapper()
        controller = GatePolicyController("P1")
        controller.reset(env)

        initial_action = controller.action(env)
        self.assertGreater(np.linalg.norm(initial_action[1]), 0.0)
        self.assertEqual(controller.active_heavy, 0)
        self.assertEqual(controller.active_support, 1)

        env.env.zone_expected_offered_load_mbps = LOAD_PHASES_MBPS[1].copy()
        controller.before_macro(env, 10, switched=True)
        np.testing.assert_allclose(controller.action(env), 0.0, atol=1e-9)
        self.assertIsNone(controller.active_support)

        controller.before_macro(env, 11, switched=False)
        self.assertEqual(controller.active_heavy, 1)
        self.assertIsNotNone(controller.active_support)
        self.assertGreater(np.linalg.norm(controller.action(env)), 0.0)

    def test_p2_starts_next_phase_support_five_periods_early(self):
        env = FakeWrapper()
        controller = GatePolicyController("P2")
        controller.reset(env)
        controller.before_macro(env, 5, switched=False)
        self.assertEqual(controller.active_heavy, 1)
        self.assertEqual(controller.last_event, "noncausal_anticipation_started")

        env.env.zone_expected_offered_load_mbps = LOAD_PHASES_MBPS[1].copy()
        controller.before_macro(env, 10, switched=True)
        self.assertEqual(controller.active_heavy, 1)
        self.assertEqual(controller.last_event, "anticipated_role_continues")

    def test_paired_summary_uses_within_seed_differences(self):
        reference = [
            {"seed": 1, "delay": 100.0},
            {"seed": 2, "delay": 200.0},
            {"seed": 3, "delay": 300.0},
        ]
        candidate = [
            {"seed": 1, "delay": 94.0},
            {"seed": 2, "delay": 195.0},
            {"seed": 3, "delay": 296.0},
        ]
        summary = paired_summary(candidate, reference, "delay")
        self.assertAlmostEqual(summary["mean"], -5.0)
        np.testing.assert_allclose(summary["values"], [-6.0, -5.0, -4.0])


if __name__ == "__main__":
    unittest.main()
