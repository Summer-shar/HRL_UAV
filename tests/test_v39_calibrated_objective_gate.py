import unittest

from scripts.run_v39_calibrated_objective_gate import (
    DELAY_WEIGHT,
    ENERGY_WEIGHT,
    exchange_rate_summary,
    raw_soft_objective_value,
)


class V39CalibratedObjectiveGateTests(unittest.TestCase):
    def test_finite_exchange_rate(self):
        result = exchange_rate_summary(10.0, 20.0)
        self.assertEqual(result["status"], "FINITE_EXCHANGE_RATE")
        self.assertAlmostEqual(result["ms_per_w"], 0.5)
        self.assertTrue(result["meets_declared_threshold"])

    def test_pareto_dominance(self):
        result = exchange_rate_summary(4.0, -2.0)
        self.assertEqual(result["status"], "PARETO_DOMINANT")
        self.assertTrue(result["meets_declared_threshold"])

    def test_no_delay_improvement_is_invalid_purchase(self):
        result = exchange_rate_summary(-1.0, 10.0)
        self.assertEqual(result["status"], "NO_DELAY_IMPROVEMENT")
        self.assertFalse(result["meets_declared_threshold"])

    def test_raw_soft_objective_uses_frozen_exact_weights(self):
        row = {"cost_delay_raw_mean": 2.0, "cost_energy_raw_mean": 3.0}
        expected = DELAY_WEIGHT * 2.0 + ENERGY_WEIGHT * 3.0
        self.assertAlmostEqual(raw_soft_objective_value(row), expected)


if __name__ == "__main__":
    unittest.main()
