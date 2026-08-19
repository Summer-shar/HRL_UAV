import unittest

from scripts.audit_v42_macro_only_pilot import mean_ci95, parse_seed_spec


class V42MacroOnlyAuditTests(unittest.TestCase):
    def test_parse_seed_spec(self):
        self.assertEqual(parse_seed_spec("7000-7002,7010"), [7000, 7001, 7002, 7010])

    def test_mean_ci95_preserves_signed_deltas(self):
        summary = mean_ci95([-3.0, -2.0, -1.0])
        self.assertEqual(summary["mean"], -2.0)
        self.assertLess(summary["ci_low"], summary["mean"])
        self.assertGreater(summary["ci_high"], summary["mean"])


if __name__ == "__main__":
    unittest.main()
