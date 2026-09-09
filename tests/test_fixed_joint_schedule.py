import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from hrl_uav.experiment import smoke_config
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.qualification import joint_start_decision, run_mainline


ROOT = Path(__file__).resolve().parents[1]


class FixedJointScheduleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = json.loads((ROOT / "configs/hrl_pretrain200_joint200.json").read_text())
        self.failed = {"passed": False, "checks": {
            "enough_checkpoints": True, "paired_seeds": True,
            "finite_readings": True, "delay_quality": False}}

    def test_fixed_budget_does_not_relabel_failed_quality(self):
        original = copy.deepcopy(self.failed)
        result = joint_start_decision(self.failed, "fixed_budget")
        self.assertTrue(result["allowed"])
        self.assertFalse(result["quality_verdict_passed"])
        self.assertFalse(result["smoke_override"])
        self.assertEqual(self.failed, original)
        self.assertFalse(joint_start_decision(self.failed)["allowed"])

    def test_fixed_budget_still_requires_valid_measurements(self):
        for key in ("enough_checkpoints", "paired_seeds", "finite_readings"):
            bad = copy.deepcopy(self.failed)
            bad["checks"][key] = False
            self.assertFalse(joint_start_decision(bad, "fixed_budget")["allowed"])
        with self.assertRaises(ValueError):
            joint_start_decision(self.failed, "typo")

    def test_only_schedule_and_joint_monitoring_differ_from_micro_run(self):
        previous = json.loads((ROOT / "configs/micro_resume800.json").read_text())
        current = copy.deepcopy(self.config)
        self.assertEqual(current["training"]["pretrain_windows"], 200)
        self.assertEqual(current["training"]["joint_windows"], 200)
        self.assertEqual(current["training"].pop("joint_start_policy"), "fixed_budget")
        self.assertEqual(current["training"].pop("joint_sac_actor_warmup_updates"), 0)
        self.assertTrue(current["qualification"].pop("paired_joint_validation"))
        for key in ("pretrain_windows", "joint_windows"):
            current["training"][key] = previous["training"][key]
        current["qualification"]["max_stage_seconds"] = previous["qualification"]["max_stage_seconds"]
        self.assertEqual(current, previous)

    def test_joint_actor_does_not_repeat_pretrain_warmup(self):
        config = smoke_config(self.config)
        config["training"]["sac_actor_warmup_updates"] = 100
        env = QueueMDP(config)
        env.reset(1, 2)
        agent = DirectHRL.from_mdp(env, config)
        state = env.micro_observation()
        action = np.zeros(env.micro_action_dim, dtype=np.float32)
        route = np.full((2, 3, 3), 1 / 3)
        for _ in range(config["training"]["replay_warmup"]):
            agent.store(state, action, route, -0.1, state, 0.99)
        agent.update_sac()
        self.assertEqual(agent.sac_actor_updates, 0)
        agent.set_stage("joint")
        agent.update_sac()
        self.assertEqual(agent.sac_actor_updates, 1)
        self.assertEqual(agent.sac_updates, 2)

    def test_real_two_stage_run_preserves_sac_and_trains_both_layers(self):
        config = smoke_config(self.config)
        config["training"].update(pretrain_windows=2, joint_windows=2)
        config["qualification"].update(
            validation_every_windows=1, tail_checkpoints=1, validation_macros=6,
            validation_seeds=[33000], test_seeds=[33500],
            max_relative_delay_regression=-0.99,
        )
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            report = run_mainline(config, output)
            self.assertFalse(report["pretrain_admission"]["passed"])
            self.assertTrue(report["joint_training_started"])
            self.assertTrue(report["joint_start_decision"]["allowed"])
            self.assertFalse(report["joint_start_decision"]["smoke_override"])
            pretrain, joint = report["stage_audits"]
            self.assertEqual(pretrain["macro_mean_max_delta"], 0)
            self.assertEqual(pretrain["ppo_updates"], 0)
            self.assertGreater(joint["ppo_updates"], 0)
            self.assertGreater(joint["actor_updates"], 0)
            self.assertGreater(joint["macro_mean_max_delta"], 0)
            self.assertEqual(joint["replay_at_stage_entry"], pretrain["replay_at_stage_exit"])
            audit = json.loads((output / "joint_transition_audit.json").read_text())
            self.assertTrue(all(audit["preserved"].values()))
            self.assertEqual(audit["after_pretrain_windows"], 2)
            self.assertEqual(audit["joint_actor_warmup_updates"], 0)
            self.assertEqual(audit["updates_at_entry"]["ppo"], 0)
            self.assertEqual(len(report["paired_macro_delay_ms"]), 1)
            self.assertTrue(all(row["users"] and row["arrivals"]
                                for row in report["matched_centroid_crn"].values()))
            self.assertTrue((output / "validation_macro_deltas.csv").is_file())
            checkpoint = torch.load(output / "final.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["stage"], "joint")
            self.assertEqual(checkpoint["run_state"]["completed_windows"], 2)
            self.assertIn("rng", checkpoint["run_state"])
            self.assertIn("macro_rollout", checkpoint["training_state"])
            self.assertEqual(checkpoint["training_state"]["macro_rollout"], [])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertIsNone(manifest["old_checkpoint"])
            self.assertFalse(manifest["teacher"])
            self.assertEqual(manifest["joint_start_policy"], "fixed_budget")


if __name__ == "__main__":
    unittest.main()
