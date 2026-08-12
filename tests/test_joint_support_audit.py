import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_v26_joint_support import action_bins
from audit_v26_joint_support import action_norm
from audit_v26_joint_support import state_bins
from audit_v26_joint_support import state_distance_m
from diagnose_td3_counterfactual import load_support_replay
from diagnose_td3_counterfactual import state_action_support_metrics


class JointSupportAuditTests(unittest.TestCase):
    def test_state_distance_uses_first_two_features_per_uav(self):
        states = np.zeros((2, 12), dtype=np.float32)
        states[0, 0:2] = [0.003, 0.004]
        states[0, 4:6] = [0.0, 0.01]
        states[0, 8:10] = [0.0, 0.0]
        states[1, 0:2] = [0.02, 0.0]
        states[1, 4:6] = [0.0, 0.02]
        states[1, 8:10] = [0.012, 0.016]

        observed = state_distance_m(states, num_agents=3)

        self.assertTrue(np.allclose(observed, [5.0, 20.0]))

    def test_action_norm_and_frozen_bin_boundaries(self):
        actions = np.asarray(
            [
                [0.10, 0.0, 0.10, 0.0, 0.10, 0.0],
                [0.40, 0.0, 0.40, 0.0, 0.40, 0.0],
                [0.60, 0.0, 0.60, 0.0, 0.60, 0.0],
            ],
            dtype=np.float32,
        )
        norms = action_norm(actions, num_agents=3)

        self.assertTrue(np.allclose(norms, [0.10, 0.40, 0.60]))
        self.assertTrue(np.array_equal(action_bins(norms), [0, 1, 2]))
        self.assertTrue(
            np.array_equal(
                state_bins(np.asarray([20.0, 50.0, 50.1])),
                [0, 1, 2],
            )
        )

    def test_conditional_support_distinguishes_seen_action_at_same_state(self):
        states = np.zeros((4, 12), dtype=np.float32)
        states[2:, 0] = 0.08
        actions = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.1, 0.0, 0.1, 0.0],
                [0.8, 0.0, 0.8, 0.0, 0.8, 0.0],
                [0.9, 0.0, 0.9, 0.0, 0.9, 0.0],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "support.npz"
            np.savez(path, state=states, action=actions)
            support = load_support_replay(path, num_agents=3, neighbors=2)
            metrics = state_action_support_metrics(
                support,
                states[0],
                actions[2],
            )

        self.assertEqual(metrics["zero_support_coarse_cell_count"], 2)
        self.assertGreater(
            metrics["actor_support_conditional_action_nn_distance"],
            metrics["zero_support_conditional_action_nn_distance"],
        )


if __name__ == "__main__":
    unittest.main()
