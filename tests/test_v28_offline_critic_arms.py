import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_v28_offline_critic_arms import apply_gates_and_verdict
from run_v28_offline_critic_arms import initial_fork_queries
from run_v28_offline_critic_arms import prepare_finite_horizon_dataset
from run_v28_offline_critic_arms import validate_crn_pairs


def toy_payload(horizon=3):
    rows = []
    for branch, rewards in (("actor", [1.0, 2.0, 3.0]), ("zero", [0.0, 1.0, 1.0])):
        for step in range(horizon):
            rows.append(
                {
                    "branch_id": f"train:1:centroid:0:{branch}",
                    "branch_action": branch,
                    "branch_step": step,
                    "fork_state_id": "train:1:centroid:0",
                    "episode_seed": 1,
                    "snapshot_policy": "centroid",
                    "fork_macro_index": 0,
                    "state": np.asarray([0.0, float(step)], dtype=np.float32),
                    "action": np.asarray([1.0 if branch == "actor" else 0.0], dtype=np.float32),
                    "reward": rewards[step],
                    "next_state": np.asarray([0.0, float(step + 1)], dtype=np.float32),
                    "external_trace_hash": f"trace-{step}",
                }
            )
    return {
        key: np.asarray([row[key] for row in rows])
        for key in rows[0]
    }


class V28OfflineCriticArmTests(unittest.TestCase):
    def test_remaining_horizon_is_part_of_state_and_return_label(self):
        payload = toy_payload()
        prepared = prepare_finite_horizon_dataset(payload, horizon=3, gamma=0.5)

        self.assertEqual(prepared["state"].shape, (6, 3))
        self.assertTrue(
            np.allclose(prepared["state"][:3, -1], [1.0, 2.0 / 3.0, 1.0 / 3.0])
        )
        self.assertTrue(np.allclose(prepared["next_state"][:3, -1], [2.0 / 3.0, 1.0 / 3.0, 0.0]))
        self.assertTrue(np.allclose(prepared["truncated_return"][:3, 0], [2.75, 3.5, 3.0]))

    def test_initial_queries_use_same_state_and_fixed_horizon_return(self):
        queries = initial_fork_queries(toy_payload(), horizon=3, gamma=0.5)

        self.assertEqual(len(queries), 1)
        self.assertAlmostEqual(queries[0]["return_actor"], 2.75)
        self.assertAlmostEqual(queries[0]["return_zero"], 0.75)
        self.assertAlmostEqual(queries[0]["true_delta"], 2.0)

    def test_verdict_requires_all_pipeline_gates(self):
        arm = {
            "agreement_ci95": [0.95, 0.90, 1.0],
            "predicted_delta_mean": -1.0,
            "classification": "CORRECT",
        }
        summary = {
            split: {
                policy: {
                    "true_delta_mean": -1.0,
                    "td": {**arm, "classification": "WRONG" if policy == "centroid" else "CORRECT"},
                    "truncated_return": dict(arm),
                }
                for policy in ("actor", "centroid")
            }
            for split in ("train", "test")
        }

        gates, verdict = apply_gates_and_verdict(summary)
        self.assertTrue(all(gates.values()))
        self.assertEqual(verdict, "SUPPORTS_BOOTSTRAP_BIAS_PROPAGATION")

        gates, verdict = apply_gates_and_verdict(summary, crn_gate=False)
        self.assertFalse(gates["crn_and_counts"])
        self.assertEqual(verdict, "INVALID_OR_INCONCLUSIVE")

    def test_crn_gate_checks_every_paired_step(self):
        payload = toy_payload()
        self.assertTrue(validate_crn_pairs(payload, horizon=3))

        payload["external_trace_hash"][-1] = "mismatch"
        self.assertFalse(validate_crn_pairs(payload, horizon=3))


if __name__ == "__main__":
    unittest.main()
