"""Matched continuation budgets, update clocks and history provenance."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.compare_sac_tqc import configurations
from scripts.continue_sac_tqc import extended_config, scheduled_updates
from scripts.plot_sac_tqc import history_rows
from scripts.run_convergence_watch import snapshot_windows
from hrl_uav.resumption import validate_resume_config


class SACTQCContinuationTests(unittest.TestCase):
    def test_only_budget_changes_for_both_algorithms(self):
        for name, source in configurations().items():
            with self.subTest(algorithm=name):
                original = copy.deepcopy(source)
                target = extended_config(source, 200)
                self.assertEqual(target["training"]["joint_windows"], 1000)
                self.assertEqual(validate_resume_config(source, target, 800, "joint"), {})
                target["training"]["joint_windows"] = 800
                self.assertEqual(target, original)
                self.assertEqual(source, original)

    def test_invalid_budget_rejected(self):
        source = configurations()["sac"]
        for additional in (0, -1, True, 200.5, "200"):
            with self.subTest(additional=additional), self.assertRaises(ValueError):
                extended_config(source, additional)

    def test_second_continuation_keeps_all_learning_settings(self):
        for name, fresh in configurations().items():
            with self.subTest(algorithm=name):
                source = extended_config(fresh, 200)
                target = extended_config(source, 400)
                self.assertEqual(target["training"]["joint_windows"], 1400)
                self.assertEqual(validate_resume_config(source, target, 1000, "joint"), {})
                target["training"]["joint_windows"] = source["training"]["joint_windows"]
                self.assertEqual(target, source)

    def test_kl_early_stop_is_not_a_scheduled_clock_mismatch(self):
        a = dict(sac_critic=26951, sac_critic_gradients=53902, sac_target=26951,
                 sac_actor=13425, ppo=80, ppo_gradients=154)
        b = {**a, "ppo_gradients": 175}
        self.assertEqual(scheduled_updates(a), scheduled_updates(b))
        b["ppo"] += 1
        self.assertNotEqual(scheduled_updates(a), scheduled_updates(b))

    def test_snapshots_use_actual_rollout_boundaries(self):
        self.assertEqual(snapshot_windows(200, 800, 1000, 3, 100), [1102, 1200])
        self.assertEqual(snapshot_windows(200, 1000, 1400, 3, 100), [1302, 1401, 1500, 1600])

    def test_history_is_concatenated_not_overwritten(self):
        result = {"protocol": {"history_runs": ["old"]}}
        records = {Path("old/sac/training.csv"): [{"global_window": "999"}, {"global_window": "1000"}],
                   Path("new/sac/training.csv"): [{"global_window": "1001"}, {"global_window": "1002"}]}
        with patch("pathlib.Path.exists", return_value=True), patch(
                "scripts.plot_sac_tqc.read_rows", side_effect=lambda p: records[p]):
            rows = history_rows("new", result, "sac", "training.csv")
            self.assertEqual([int(r["global_window"]) for r in rows], [999, 1000, 1001, 1002])
            records[Path("new/sac/training.csv")][0]["global_window"] = "1000"
            with self.assertRaisesRegex(ValueError, "unique increasing"):
                history_rows("new", result, "sac", "training.csv")

    def test_final_only_resume_does_not_require_validation_csv(self):
        with patch("pathlib.Path.exists", return_value=False):
            self.assertEqual(history_rows("new", {"protocol": {}}, "tqc", "validation.csv"), [])


if __name__ == "__main__":
    unittest.main()
