import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_experiment import build_experiments
from run_experiment import build_micro_pretrain
from run_experiment import stage_config


class JointTrainingProtocolTest(unittest.TestCase):
    def test_main_stages_use_twenty_macro_steps_without_zone_confinement(self):
        config = stage_config("pilot")
        self.assertEqual(config["macro_steps"], 20)
        experiment = build_micro_pretrain(
            "pilot",
            seed=1,
            config=config,
            device="cpu",
        )
        radius_index = experiment["args"].index("--user_zone_radius")
        self.assertEqual(float(experiment["args"][radius_index + 1]), 0.0)

    def test_micro_pretraining_freezes_only_the_macro_layer(self):
        config = stage_config("smoke")
        experiment = build_micro_pretrain(
            "smoke",
            seed=1,
            config=config,
            device="cpu",
        )

        self.assertIn("--freeze_macro", experiment["args"])
        self.assertNotIn("--freeze_micro", experiment["args"])

    def test_main_method_continues_training_the_micro_layer(self):
        config = stage_config("smoke")
        experiments = build_experiments(
            "smoke",
            seeds=[1],
            config=config,
            device="cpu",
        )
        experiment = next(
            item
            for item in experiments
            if item["method"] == "joint_td3_sac"
        )

        self.assertIn("--micro_ckpt", experiment["args"])
        self.assertNotIn("--freeze_micro", experiment["args"])
        capacity_index = experiment["args"].index("--td3_replay_capacity")
        self.assertEqual(experiment["args"][capacity_index + 1], "3000")


if __name__ == "__main__":
    unittest.main()
