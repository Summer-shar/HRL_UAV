import unittest

from scripts.prescreen_p1_prime import integrated_offset_exposure
from scripts.prescreen_p1_prime import select_candidate
from scripts.prescreen_p1_prime import split_traversal


class P1PrimePrescreenTests(unittest.TestCase):
    def test_split_traversal_preserves_offset(self):
        self.assertEqual(split_traversal(40.0, 10.0), [10.0] * 4)
        self.assertEqual(split_traversal(20.0, 15.0), [15.0, 5.0])
        self.assertAlmostEqual(sum(split_traversal(30.0, 20.0)), 30.0)

    def test_exposure_rewards_earlier_geometric_response(self):
        self.assertGreater(
            integrated_offset_exposure(40.0, 10.0),
            integrated_offset_exposure(40.0, 5.0),
        )

    def test_selection_is_lexicographic_and_ignores_unsafe_points(self):
        rows = [
            {
                "offset_m": 40.0,
                "residual_cap_per_macro_m": 15.0,
                "analytically_safe": False,
            },
            {
                "offset_m": 30.0,
                "residual_cap_per_macro_m": 20.0,
                "analytically_safe": True,
            },
            {
                "offset_m": 40.0,
                "residual_cap_per_macro_m": 5.0,
                "analytically_safe": True,
            },
            {
                "offset_m": 40.0,
                "residual_cap_per_macro_m": 10.0,
                "analytically_safe": True,
            },
        ]
        selected = select_candidate(rows)
        self.assertEqual(selected["offset_m"], 40.0)
        self.assertEqual(selected["residual_cap_per_macro_m"], 10.0)


if __name__ == "__main__":
    unittest.main()

