import unittest

import numpy as np

from scripts.audit_v41_micro_extension import paired_summary


class V41MicroExtensionAuditTests(unittest.TestCase):
    def test_paired_summary_uses_within_seed_differences(self):
        rows = []
        for seed, baseline, treatment in ((1, 10.0, 8.0), (2, 100.0, 99.0)):
            for name, delay in (("v33_final", baseline), ("v41_final", treatment)):
                rows.append(
                    {
                        "checkpoint": name,
                        "seed": seed,
                        "task_delay_weighted_ms": delay,
                        "mean_queue_mbits": delay,
                        "terminal_queue_mbits": delay,
                        "average_power_w": delay,
                    }
                )
        rows.extend(
            {
                "checkpoint": "v40_final",
                "seed": seed,
                "task_delay_weighted_ms": value,
                "mean_queue_mbits": value,
                "terminal_queue_mbits": value,
                "average_power_w": value,
            }
            for seed, value in ((1, 9.0), (2, 99.5))
        )
        result = paired_summary(rows, "v41_final", "v33_final")
        self.assertAlmostEqual(result["delay_ms"]["mean"], np.mean([-2.0, -1.0]))
        self.assertEqual(result["delay_ms"]["negative_count"], 2)


if __name__ == "__main__":
    unittest.main()
