import unittest

import numpy as np

from scripts.run_v34_cooperative_mobility_gate import (
    cooperative_motion_metrics,
    soft_objective_metrics,
)


class V36OffsetAuditMetricTests(unittest.TestCase):
    def test_motion_metrics_measure_arrival_and_occupancy(self):
        records = []
        for x_coord in (20.0, 40.0, 60.0):
            records.append(
                {
                    "phase_index": 0,
                    "heavy_uav": 1,
                    "support_uav": 0,
                    "residual_command_m": np.asarray(
                        [[60.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
                    ),
                    "uav_after_m": np.asarray(
                        [[x_coord, 0.0], [0.0, 0.0], [0.0, 0.0]]
                    ),
                    "centroids_after_m": np.zeros((3, 2)),
                }
            )
        metrics = cooperative_motion_metrics(
            {"macro_records": records},
            macro_period_s=10.0,
            target_tolerance_m=10.0,
        )
        self.assertAlmostEqual(metrics["target_occupancy_fraction"], 1.0 / 3.0)
        self.assertEqual(metrics["target_reach_segment_fraction"], 1.0)
        self.assertEqual(metrics["target_arrival_time_reached_s"], 30.0)
        self.assertEqual(metrics["target_arrival_time_capped_s"], 30.0)

    def test_soft_objective_is_negative_mean_reward(self):
        trace = {
            "macro_records": [
                {"average_micro_reward": -0.2},
                {"average_micro_reward": -0.4},
            ]
        }
        metrics = soft_objective_metrics(trace, micro_reward_scale=1.0)
        self.assertAlmostEqual(metrics["soft_objective_j"], 0.3)


if __name__ == "__main__":
    unittest.main()
