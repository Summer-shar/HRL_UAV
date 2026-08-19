import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class V42MacroOnlyPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v40 = json.loads(
            (ROOT / "configs" / "tcom_v40_micro_warmstart.json").read_text(
                encoding="utf-8"
            )
        )
        cls.v42 = json.loads(
            (ROOT / "configs" / "tcom_v42_macro_only_pilot.json").read_text(
                encoding="utf-8"
            )
        )

    def test_v42_uses_v40_final_with_frozen_micro(self):
        self.assertEqual(
            self.v42["micro_pretrain_source_results_version"],
            "tcom_v40_micro_warmstart",
        )
        self.assertEqual(self.v42["micro_pretrain_source_stage"], "formal")
        self.assertEqual(self.v42["micro_pretrain_source_seed"], 1)
        self.assertEqual(self.v42["micro_pretrain_checkpoint"], "final")
        self.assertTrue(self.v42["freeze_micro_in_joint"])
        self.assertFalse(self.v42["resume_micro_replay"])
        self.assertFalse(self.v42["resume_micro_optimizer_in_joint"])

    def test_v42_changes_only_experiment_identity_and_micro_source(self):
        ignored = {
            "results_version",
            "micro_pretrain_source_results_version",
            "micro_pretrain_source_stage",
            "micro_pretrain_source_seed",
            "micro_pretrain_actor_init_checkpoint",
            "micro_pretrain_prefill_episodes",
            "micro_pretrain_actor_freeze_episodes",
            "training_episodes",
            "freeze_micro_in_joint",
            "resume_micro_replay",
            "resume_micro_optimizer_in_joint",
        }
        common_keys = (set(self.v40) & set(self.v42)) - ignored
        differences = {
            key: (self.v40[key], self.v42[key])
            for key in common_keys
            if self.v40[key] != self.v42[key]
        }
        self.assertEqual(differences, {})

    def test_v42_keeps_final_stack_and_single_seed_budget(self):
        self.assertEqual(self.v42["formal_seeds"], [1])
        self.assertEqual(self.v42["training_episodes"], 300)
        self.assertEqual(self.v42["objective_mode"], "weighted")
        self.assertAlmostEqual(
            self.v42["delay_objective_weight"], 0.960408028711155
        )
        self.assertAlmostEqual(
            self.v42["energy_objective_weight"], 0.03959197128884489
        )
        self.assertEqual(self.v42["zone_loads_mbps"], [14.5, 4.5, 2.5])
        self.assertEqual(self.v42["gm_heading_mode"], "reference_point_group")


if __name__ == "__main__":
    unittest.main()
