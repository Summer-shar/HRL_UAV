import unittest

from scripts.audit_v50_conservative_micro_gate import conservative_twin_advantage


class ConservativeTwinAdvantageTests(unittest.TestCase):
    def test_requires_candidate_lower_bound_to_clear_baseline_upper_bound(self):
        self.assertAlmostEqual(
            conservative_twin_advantage(1.4, 1.2, 1.0, 1.1),
            0.1,
        )

    def test_rejects_candidate_when_twin_intervals_overlap(self):
        self.assertLess(
            conservative_twin_advantage(1.4, 1.0, 1.1, 1.2),
            0.0,
        )

    def test_rejects_candidate_below_baseline(self):
        self.assertLess(
            conservative_twin_advantage(0.8, 0.9, 1.0, 1.1),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
