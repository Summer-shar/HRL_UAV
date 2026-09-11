import unittest
from unittest.mock import patch

from scripts import run_progressive_centroid_recheck as check


class ProgressiveRecheckTests(unittest.TestCase):
    def test_only_three_authorized_scenes_admitted(self):
        for seed in (33600, 33601, 33700, 33709, 33610.0, True):
            with self.assertRaises(ValueError):
                check.evaluate_b({}, None, seed, None)

    def test_exact_existing_B_controller_invocation(self):
        original = check.tracking.SEEDS
        calls = []

        def recorder(*args):
            calls.append((args, check.tracking.SEEDS))
            return "row", "meter"

        with patch.object(check.tracking, "evaluate", recorder):
            for seed in check.RECHECK:
                self.assertEqual(check.evaluate_b({"sentinel": 1}, None, seed, "out"), ("row", "meter"))
        self.assertEqual(check.tracking.SEEDS, original)
        self.assertEqual(len(calls), 3)
        for (args, allowed), seed in zip(calls, check.RECHECK):
            self.assertEqual(args[2:], ("zero", .5, seed, 90, "out", "B"))
            self.assertEqual(allowed, check.RECHECK)

    def test_aggregate_requires_complete_unique_scenes(self):
        rows = [dict(policy=p, seed=s, **{k: float(s) for k in check.FIELDS})
                for p in ("A", "B", "PPO120") for s in check.RECHECK]
        result = check.aggregate(rows, check.RECHECK, "development_recheck")
        self.assertEqual(result[0]["J"], 33611)
        with self.assertRaises(ValueError):
            check.aggregate(rows[:-1], check.RECHECK, "development_recheck")
        with self.assertRaises(ValueError):
            check.aggregate(rows + [rows[0]], check.RECHECK, "development_recheck")


if __name__ == "__main__":
    unittest.main()
