import unittest

import numpy as np

from scripts.run_v32c_stage_residual_benchmark import StageResidualPolicy
from scripts.run_v32c_stage_residual_benchmark import candidate_rank
from scripts.run_v32c_stage_residual_benchmark import select_best_feasible
from scripts.run_v32c_stage_residual_benchmark import project_residuals


class _Schedule:
    switch_steps = (10, 20)

    @staticmethod
    def phase_index(index):
        return min(int(index) // 10, 2)


class StageResidualBenchmarkTests(unittest.TestCase):
    def test_projection_limits_each_uav_phase_vector(self):
        offsets = np.full((3, 3, 2), 100.0)
        projected = project_residuals(offsets, 80.0)
        np.testing.assert_allclose(
            np.linalg.norm(projected, axis=2),
            80.0,
        )

    def test_noncausal_blend_reaches_next_phase_before_switch(self):
        offsets = np.zeros((3, 3, 2))
        offsets[1, :, 0] = 50.0
        policy = StageResidualPolicy(offsets, _Schedule(), anticipation_steps=5)
        np.testing.assert_allclose(policy.desired_residual(4), 0.0)
        np.testing.assert_allclose(policy.desired_residual(5)[:, 0], 10.0)
        np.testing.assert_allclose(policy.desired_residual(9)[:, 0], 50.0)
        np.testing.assert_allclose(policy.desired_residual(10)[:, 0], 50.0)

    def test_feasibility_first_rank_precedes_lower_delay_infeasible_candidate(self):
        feasible = {"delay_mean_ms": 100.0, "power_ucb95_w": 837.0}
        infeasible = {"delay_mean_ms": 50.0, "power_ucb95_w": 839.0}
        self.assertLess(
            candidate_rank(feasible, "feasibility_first", 838.0),
            candidate_rank(infeasible, "feasibility_first", 838.0),
        )

    def test_validation_can_select_feasible_unconstrained_final_candidate(self):
        summaries = {
            "unconstrained": {"gain_ms": 12.0, "power_ucb95_w": 839.0},
            "feasibility_first": {"gain_ms": 5.0, "power_ucb95_w": 838.0},
        }
        arm, summary = select_best_feasible(summaries, 840.0)
        self.assertEqual(arm, "unconstrained")
        self.assertEqual(summary["gain_ms"], 12.0)


if __name__ == "__main__":
    unittest.main()
