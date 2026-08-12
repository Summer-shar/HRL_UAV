import unittest
from unittest import mock

import numpy as np

from scripts.evaluate_v30_gate_b import mean_ci95
from scripts.evaluate_v30_gate_b import pair_rows
from scripts.evaluate_v30_gate_b import summarize
from scripts.evaluate_v30_gate_b import two_way_bootstrap


def _arm_row(
    arm,
    training_seed,
    eval_seed,
    delay,
    power,
    route_shift=0.0,
    coop_ratio=0.5,
):
    row = {
        "arm": arm,
        "training_seed": training_seed,
        "eval_seed": eval_seed,
        "task_delay_weighted_ms": delay,
        "average_power_w": power,
        "users_trace_hash": f"users-{eval_seed}",
        "arrival_high_trace_hash": f"high-{eval_seed}",
        "arrival_low_trace_hash": f"low-{eval_seed}",
        "task_conservation_residual": 0,
        "coop_ratio": coop_ratio,
    }
    for prefix in ("route_high", "route_low"):
        for i in range(3):
            for j in range(3):
                value = 1.0 if i == j else 0.0
                if i == 0 and j == 0:
                    value -= route_shift
                if i == 0 and j == 1:
                    value += route_shift
                row[f"{prefix}_{i}_{j}"] = value
    return row


class GateBEvaluationTest(unittest.TestCase):
    def test_pairing_preserves_crn_and_computes_route_tv(self):
        rows = []
        for training_seed in (1, 42, 123):
            rows.append(_arm_row("control", training_seed, 71000, 100.0, 820.0))
            rows.append(
                _arm_row(
                    "treatment",
                    training_seed,
                    71000,
                    98.0,
                    821.0,
                    route_shift=0.3,
                )
            )
        paired = pair_rows(rows)
        self.assertEqual(len(paired), 3)
        self.assertTrue(all(row["external_trace_match"] for row in paired))
        self.assertTrue(all(row["delay_delta_ms"] == -2.0 for row in paired))
        self.assertTrue(all(row["coop_delta"] == 0.0 for row in paired))
        self.assertTrue(all(row["routing_total_variation"] > 0.0 for row in paired))

    def test_two_way_bootstrap_is_deterministic_and_centered(self):
        matrix = np.asarray([[-2.0, -1.0], [-3.0, -2.0], [-1.0, -3.0]])
        with mock.patch(
            "scripts.evaluate_v30_gate_b.DEFAULT_BOOTSTRAP_REPLICATES",
            1000,
        ):
            first = two_way_bootstrap(matrix)
            second = two_way_bootstrap(matrix)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], -2.0)
        self.assertLess(first["one_sided_95_upper"], 0.0)

    def test_constant_mean_interval_is_finite(self):
        self.assertEqual(mean_ci95([2.0, 2.0, 2.0]), (2.0, 2.0))

    def test_summary_applies_all_frozen_gate_conditions(self):
        rows = []
        for training_seed in (1, 42, 123):
            for eval_seed in (71000, 71001, 71002):
                rows.append(
                    _arm_row("control", training_seed, eval_seed, 100.0, 825.0)
                )
                rows.append(
                    _arm_row("treatment", training_seed, eval_seed, 98.0, 826.0)
                )
        paired = pair_rows(rows)
        with mock.patch(
            "scripts.evaluate_v30_gate_b.DEFAULT_BOOTSTRAP_REPLICATES",
            1000,
        ):
            per_seed, summary = summarize(rows, paired)
        self.assertEqual(len(per_seed), 3)
        self.assertEqual(summary["status"], "PASS")
        self.assertTrue(all(summary["conditions"].values()))


if __name__ == "__main__":
    unittest.main()
