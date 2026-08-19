import unittest

import numpy as np

from scripts.run_v37_load_imbalance_gate import mean_ci, paired_delta


class V37LoadImbalanceGateTests(unittest.TestCase):
    def test_paired_delta_uses_cooperative_minus_centroid(self):
        rows = [
            {"seed": 1, "policy": "centroid", "metric": 10.0},
            {"seed": 1, "policy": "cooperative", "metric": 8.0},
            {"seed": 2, "policy": "centroid", "metric": 12.0},
            {"seed": 2, "policy": "cooperative", "metric": 9.0},
        ]
        result = paired_delta(rows, "metric")
        self.assertAlmostEqual(result["mean"], -2.5)
        self.assertEqual(result["cooperative_better_count"], 2)

    def test_mean_ci_keeps_values_for_matched_difference(self):
        result = mean_ci(np.asarray([1.0, 2.0, 3.0]))
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["values"], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
