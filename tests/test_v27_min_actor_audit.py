import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_v27_min_actor import METRICS
from audit_v27_min_actor import paired_statistics
from summarize_v27_min_actor_probe import aggregate_episode
from summarize_v27_min_actor_probe import paired_comparison


def metric_row(seed, base):
    row = {
        "seed": seed,
        "centroid_trace_hash": f"centroid-{seed}",
        "users_trace_hash": f"users-{seed}",
    }
    for index, metric in enumerate(METRICS):
        row[metric] = float(base + index)
    return row


def probe_row(policy, episode_index, seed, q1_diff, q2_diff, return_diff):
    return {
        "snapshot_policy": policy,
        "episode_index": str(episode_index),
        "seed": str(seed),
        "q1_actor": str(q1_diff),
        "q1_zero": "0.0",
        "q2_actor": str(q2_diff),
        "q2_zero": "0.0",
        "q_actor_minus_zero": str(min(q1_diff, q2_diff)),
        "return_actor_minus_zero": str(return_diff),
        "trace_match": "True",
    }


class V27MinActorAuditTests(unittest.TestCase):
    def test_paired_statistics_use_common_seed_differences(self):
        left = [metric_row(1, 3.0), metric_row(2, 5.0)]
        right = [metric_row(1, 2.0), metric_row(2, 2.0)]

        rows, summary = paired_statistics(left, right, "left", "right")

        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(summary["J"]["left_minus_right_mean"], 2.0)
        self.assertTrue(all(row["external_trajectory_match"] for row in rows))

    def test_episode_probe_aggregation_keeps_heads_separate(self):
        rows = [
            probe_row("centroid", 0, 62000, 0.4, -0.2, -0.3),
            probe_row("centroid", 0, 62000, 0.2, -0.4, -0.1),
        ]

        aggregate = aggregate_episode(rows)[0]

        self.assertAlmostEqual(aggregate["q1_diff_mean"], 0.3)
        self.assertAlmostEqual(aggregate["q2_diff_mean"], -0.3)
        self.assertAlmostEqual(aggregate["qmin_diff_mean"], -0.3)
        self.assertAlmostEqual(aggregate["q1_agreement_mean"], 0.0)
        self.assertAlmostEqual(aggregate["q2_agreement_mean"], 1.0)
        self.assertAlmostEqual(aggregate["qmin_agreement_mean"], 1.0)

    def test_probe_comparison_is_paired_by_policy_episode_and_seed(self):
        v27 = aggregate_episode(
            [probe_row("centroid", 0, 62000, -0.2, -0.3, -0.1)]
        )
        v26 = aggregate_episode(
            [probe_row("centroid", 0, 62000, 0.2, 0.3, -0.1)]
        )
        for policy in ("actor",):
            v27.extend(
                aggregate_episode(
                    [probe_row(policy, 0, 62000, -0.1, -0.2, -0.1)]
                )
            )
            v26.extend(
                aggregate_episode(
                    [probe_row(policy, 0, 62000, -0.1, -0.2, -0.1)]
                )
            )

        rows, summary = paired_comparison(v27, v26)

        centroid = [row for row in rows if row["snapshot_policy"] == "centroid"]
        self.assertEqual(len(centroid), 1)
        self.assertAlmostEqual(
            centroid[0]["qmin_agreement_mean_v27_minus_v26"], 1.0
        )
        self.assertAlmostEqual(
            summary["centroid"][
                "qmin_agreement_mean_v27_minus_v26_ci95"
            ][0],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
