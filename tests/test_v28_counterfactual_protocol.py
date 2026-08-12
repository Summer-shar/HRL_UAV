import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_v28a_target_horizon import aggregate_episode
from audit_v28a_target_horizon import summarize
from generate_v28_counterfactual_dataset import branch_to_arrays
from generate_v28_counterfactual_dataset import discounted_return


class V28CounterfactualProtocolTests(unittest.TestCase):
    def test_discounted_return_obeys_requested_fixed_horizon(self):
        rewards = [1.0, 2.0, 100.0]

        observed = discounted_return(rewards, horizon=2, gamma=0.5)

        self.assertAlmostEqual(observed, 2.0)

    def test_transition_schema_keeps_episode_and_fork_identity(self):
        records = [
            {
                "state": np.zeros(4, dtype=np.float32),
                "action": np.ones(2, dtype=np.float32),
                "reward": -0.1,
                "next_state": np.ones(4, dtype=np.float32),
                "external_trace_hash": "abc",
            },
            {
                "state": np.ones(4, dtype=np.float32),
                "action": np.zeros(2, dtype=np.float32),
                "reward": -0.2,
                "next_state": np.full(4, 2.0, dtype=np.float32),
                "external_trace_hash": "def",
            },
        ]

        arrays = branch_to_arrays(
            records,
            split="test",
            episode_seed=64000,
            snapshot_policy="centroid",
            fork_macro_index=5,
            branch_action="zero",
        )

        self.assertTrue(np.array_equal(arrays["branch_step"], [0, 1]))
        self.assertTrue(np.all(arrays["episode_seed"] == 64000))
        self.assertEqual(
            set(arrays["fork_state_id"]), {"test:64000:centroid:5"}
        )
        self.assertEqual(
            set(arrays["branch_id"]), {"test:64000:centroid:5:zero"}
        )

    def test_horizon_gate_uses_episode_level_sign_agreement_without_filtering(self):
        rows = []
        for seed in (64000, 64001):
            for policy in ("actor", "centroid"):
                for fork in range(6):
                    rows.append(
                        {
                            "episode_seed": seed,
                            "snapshot_policy": policy,
                            "fork_macro_index": fork,
                            "g30_g60_sign_match": 1.0,
                            "y1_ranking_agreement_g30": 0.0,
                            "y5_ranking_agreement_g30": 0.5,
                            "y15_ranking_agreement_g30": 1.0,
                        }
                    )

        episode_rows = aggregate_episode(rows)
        summary = summarize(episode_rows)

        self.assertEqual(len(episode_rows), 4)
        self.assertTrue(summary["horizon_stability_gate_pass"])
        self.assertEqual(summary["verdict"], "VALID_TARGET_AUDIT")


if __name__ == "__main__":
    unittest.main()
