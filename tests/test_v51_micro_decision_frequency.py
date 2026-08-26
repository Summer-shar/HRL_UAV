import unittest

import numpy as np

from scripts.audit_v51_micro_decision_frequency import (
    ActionHoldPolicy,
    paired_summary,
)


class CountingPolicy:
    def __init__(self):
        self.calls = 0

    def __call__(self, _env, _eval_mode):
        self.calls += 1
        return np.asarray([self.calls, -self.calls], dtype=np.float32)


class MicroDecisionFrequencyAuditTest(unittest.TestCase):
    def test_hold_two_frames_reuses_executed_action(self):
        base = CountingPolicy()
        policy = ActionHoldPolicy(base, hold_frames=2)
        actions = [policy(None, True) for _ in range(5)]
        self.assertEqual(base.calls, 3)
        self.assertEqual(policy.base_action_updates, 3)
        np.testing.assert_array_equal(actions[0], actions[1])
        np.testing.assert_array_equal(actions[2], actions[3])
        self.assertFalse(np.array_equal(actions[1], actions[2]))
        actions[0][0] = 999.0
        self.assertNotEqual(actions[1][0], 999.0)

    def test_invalid_hold_length_is_rejected(self):
        with self.assertRaises(ValueError):
            ActionHoldPolicy(CountingPolicy(), hold_frames=0)

    def test_paired_summary_uses_hold_minus_every_frame(self):
        rows = [
            {"seed": 1, "policy": "every_frame", "delay": 10.0},
            {"seed": 1, "policy": "hold_two_frames", "delay": 12.0},
            {"seed": 2, "policy": "every_frame", "delay": 20.0},
            {"seed": 2, "policy": "hold_two_frames", "delay": 19.0},
        ]
        report = paired_summary(rows, "delay")
        self.assertEqual(report["per_seed"], {"1": 2.0, "2": -1.0})
        self.assertAlmostEqual(report["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
