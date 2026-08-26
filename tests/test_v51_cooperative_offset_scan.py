import unittest

from scripts.audit_v51_cooperative_offset_scan import (
    offset_name,
    paired_vs_zero,
    parse_offsets,
)


class CooperativeOffsetScanTest(unittest.TestCase):
    def test_offsets_require_unique_zero_baseline(self):
        self.assertEqual(parse_offsets("-10,0,10"), (-10.0, 0.0, 10.0))
        with self.assertRaises(ValueError):
            parse_offsets("-10,10")
        with self.assertRaises(ValueError):
            parse_offsets("0,10,10")

    def test_offset_name_preserves_direction(self):
        self.assertEqual(offset_name(-40), "offset_-040m")
        self.assertEqual(offset_name(0), "offset_+000m")
        self.assertEqual(offset_name(60), "offset_+060m")

    def test_paired_difference_is_offset_minus_centroid(self):
        rows = [
            {"seed": 1, "offset_m": 0, "delay": 10.0},
            {"seed": 1, "offset_m": 20, "delay": 8.0},
            {"seed": 2, "offset_m": 0, "delay": 20.0},
            {"seed": 2, "offset_m": 20, "delay": 21.0},
        ]
        report = paired_vs_zero(rows, (0.0, 20.0), "delay")["20"]
        self.assertEqual(report["per_seed"], {"1": -2.0, "2": 1.0})
        self.assertAlmostEqual(report["mean"], -0.5)


if __name__ == "__main__":
    unittest.main()
