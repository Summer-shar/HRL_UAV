import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs" / "tcom_v39_calibrated_soft_objective.json"
BASE = ROOT / "configs" / "tcom_v39_calibrated_soft_objective_base.json"
FROZEN = (
    ROOT
    / "result_archive"
    / "v38_objective_calibration"
    / "frozen_weights.json"
)


class V39CalibratedObjectiveProtocolTests(unittest.TestCase):
    def setUp(self):
        self.main = json.loads(MAIN.read_text(encoding="utf-8"))
        self.base = json.loads(BASE.read_text(encoding="utf-8"))
        self.frozen = json.loads(FROZEN.read_text(encoding="utf-8"))

    def test_weights_match_frozen_calibration(self):
        for config in (self.main, self.base):
            self.assertAlmostEqual(
                config["delay_objective_weight"],
                self.frozen["delay_weight"],
            )
            self.assertAlmostEqual(
                config["energy_objective_weight"],
                self.frozen["energy_weight"],
            )
            self.assertAlmostEqual(
                config["delay_objective_weight"]
                + config["energy_objective_weight"],
                1.0,
            )

    def test_configs_differ_only_in_load_and_result_identity(self):
        allowed = {
            "results_version",
            "zone_loads_mbps",
            "load_switch_phases_mbps",
        }
        differing = {
            key
            for key in self.main
            if self.main[key] != self.base[key]
        }
        self.assertEqual(differing, allowed)

    def test_training_uses_enhanced_load_and_final_checkpoint(self):
        self.assertEqual(self.main["zone_loads_mbps"], [14.5, 4.5, 2.5])
        self.assertEqual(self.main["micro_pretrain_episodes"], 300)
        self.assertEqual(self.main["micro_pretrain_macro_policy"], "zero")
        self.assertEqual(self.main["micro_pretrain_checkpoint"], "final")
        self.assertEqual(self.main["objective_mode"], "weighted")
        self.assertEqual(self.main["power_budget_w"], 0.0)


if __name__ == "__main__":
    unittest.main()
