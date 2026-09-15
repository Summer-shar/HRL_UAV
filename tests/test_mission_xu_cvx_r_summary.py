import csv
from pathlib import Path
import tempfile
import unittest

from hrl_uav.mission_xu_cvx_r import METRICS, summarize_results


class XuSummaryTests(unittest.TestCase):
    def test_solver_only_columns_do_not_break_reused_sac_csv(self):
        rows = []
        for seed in (201, 202, 203):
            for scene in (33600, 33601):
                for method in ("SAC", "Xu-CVX-R"):
                    row = dict(micro=method, ppo_seed=seed, seed=scene, **{k: 1. for k in METRICS})
                    if method == "Xu-CVX-R":
                        row["solver_calls"] = 1800
                        row["solver_mean_ms"] = 2.
                    rows.append(row)
        with tempfile.TemporaryDirectory() as name:
            summarize_results(Path(name), rows)
            with open(Path(name)/"per_seed.csv", encoding="utf-8-sig", newline="") as stream:
                recorded = list(csv.DictReader(stream))
            self.assertEqual(len(recorded), 12)
            self.assertTrue(all(r["solver_calls"] == "" for r in recorded if r["micro"] == "SAC"))
            self.assertTrue(all(r["solver_calls"] == "1800" for r in recorded if r["micro"] == "Xu-CVX-R"))


if __name__ == "__main__":
    unittest.main()
