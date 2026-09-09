import contextlib
import copy
import csv
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
from hrl_uav.qualification import run_mainline
from hrl_uav.resumption import load_source, restore_agent, state_equal, validate_resume_config


ROOT = Path(__file__).resolve().parents[1]


class JointContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = smoke_config(json.loads((ROOT / "configs/hrl_pretrain200_joint200.json").read_text()))
        self.config["training"].update(pretrain_windows=2, joint_windows=2, replay_capacity=64)
        self.config["qualification"].update(validation_every_windows=1, tail_checkpoints=1,
                                            validation_macros=6, validation_seeds=[33000], test_seeds=[33500])
        self.config["validation_seeds"], self.config["test_seeds"] = [33000], [33500]

    def target(self):
        target = copy.deepcopy(self.config)
        target["training"].update(joint_windows=4, ppo_std_decay_windows=2)
        return target

    def test_extension_cannot_reset_exploration_or_change_learning(self):
        target = self.target()
        validate_resume_config(self.config, target, 2, "joint")
        bad = copy.deepcopy(target)
        bad["training"].pop("ppo_std_decay_windows")
        with self.assertRaisesRegex(ValueError, "exploration"):
            validate_resume_config(self.config, bad, 2, "joint")
        for key in ("pretrain_windows", "sac_policy_delay"):
            bad = copy.deepcopy(target)
            bad["training"][key] += 1
            with self.assertRaisesRegex(ValueError, "only"):
                validate_resume_config(self.config, bad, 2, "joint")

    def test_production_config_only_extends_joint_budget(self):
        old = json.loads((ROOT / "configs/hrl_pretrain200_joint200.json").read_text())
        new = json.loads((ROOT / "configs/hrl_resume_joint600.json").read_text())
        self.assertEqual(new["training"]["pretrain_windows"], 200)
        self.assertEqual(new["training"]["joint_windows"], 600)
        self.assertEqual(new["training"]["ppo_std_decay_windows"], 200)
        validate_resume_config(old, new, 200, "joint")

    def test_joint_without_complete_rng_and_rollout_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            torch.save({"stage": "joint", "updates": {"ppo": 2}}, path)
            with self.assertRaisesRegex(ValueError, "full RNG"):
                load_source(path, self.target())

    def test_joint_resume_restores_both_layers_and_matches_control(self):
        target = self.target()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            run_mainline(target, root / "full")
            part_report = run_mainline(self.config, root / "part")
            saved, metadata = load_source(root / "part/final.pt", target)
            self.assertEqual(metadata["stage"], "joint")
            with self.assertRaisesRegex(ValueError, "pretrain-only"):
                run_mainline(target, root / "invalid", pretrain_only=True, resume_from=root / "part/final.pt")
            self.assertFalse((root / "invalid").exists())
            env = QueueMDP(target)
            env.reset(1, 2)
            restored = DirectHRL.from_mdp(env, target)
            audit = restore_agent(restored, saved, metadata)
            self.assertEqual(restored.stage, "joint")
            self.assertTrue(all(audit["exact_restore_checks"].values()))
            self.assertEqual(restored.stage_initial_updates, saved["training_state"]["stage_initial_updates"])
            self.assertEqual(restored.sac_actor.optimizer.param_groups[0]["lr"], target["training"]["joint_lr"][0])
            self.assertTrue(np.allclose(audit["ppo_exploration_std"], target["training"]["ppo_final_std"]))

            result = run_mainline(target, root / "resumed", resume_from=root / "part/final.pt")
            self.assertEqual(len(result["stage_audits"]), 1)
            stage = result["stage_audits"][0]
            self.assertEqual(stage["stage"], "joint")
            self.assertEqual(stage["start_window"], 2)
            self.assertEqual(stage["completed_windows"], 4)
            self.assertEqual(stage["trained_windows_this_run"], 2)
            self.assertGreater(stage["actor_updates"], 0)
            self.assertGreater(stage["ppo_updates"], 0)
            self.assertEqual(stage["replay_at_stage_entry"], part_report["stage_audits"][-1]["replay_at_stage_exit"])
            self.assertEqual(result["pretrain_admission"], part_report["pretrain_admission"])
            self.assertTrue(result["joint_start_decision"]["continued_existing_joint_stage"])
            self.assertTrue(all(value == 0 for row in result["resume"]["initial_validation_absolute_differences"].values()
                                for value in row.values()))
            self.assertFalse((root / "resumed/micro_pretrained.pt").exists())
            self.assertFalse((root / "resumed/joint_transition_audit.json").exists())
            def rows(path):
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    return list(csv.DictReader(handle))
            full_rows = [r for r in rows(root / "full/training.csv")
                         if r["stage"] == "joint" and int(r["window"]) > 2]
            self.assertEqual(full_rows, rows(root / "resumed/training.csv"))
            full = torch.load(root / "full/final.pt", map_location="cpu", weights_only=False)
            resumed = torch.load(root / "resumed/final.pt", map_location="cpu", weights_only=False)
            self.assertTrue(state_equal(full, resumed))
            self.assertEqual(resumed["run_state"]["next_training_scene_seed"], metadata["next_training_scene_seed"] + 2)


if __name__ == "__main__":
    unittest.main()
