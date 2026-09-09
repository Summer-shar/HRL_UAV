"""Monitoring is observational: no new update events, random draws or transitions."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from hrl_uav.experiment import smoke_config
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.qualification import run_mainline
from hrl_uav.resumption import state_equal, validate_resume_config
from scripts.run_convergence_watch import RetainBoundaries, continuation_config, paired_readings, snapshot_windows, trend_summary


ROOT = Path(__file__).resolve().parents[1]


class ConvergenceWatchTests(unittest.TestCase):
    def test_snapshot_windows_match_sampling_boundaries_for_both_continuations(self):
        self.assertEqual(snapshot_windows(200, 1400, 1600, 3, 50), [1651, 1702, 1750, 1800])
        self.assertEqual(snapshot_windows(200, 1600, 1800, 3, 50), [1851, 1902, 1950, 2000])
        self.assertEqual(snapshot_windows(200, 1600, 1800, 3, 100), [1902, 2000])

    def test_default_continuation_preserves_all_settings_except_joint_budget(self):
        original = copy.deepcopy(self.config)
        target = continuation_config(self.config)
        self.assertEqual(validate_resume_config(self.config, target, 4, "joint"), {})
        target["training"]["joint_windows"] -= 200
        self.assertEqual(target, original)
        self.assertEqual(self.config, original)

    def test_explicit_checkpoint_override_is_monitoring_only(self):
        target = continuation_config(self.config, 100)
        self.assertEqual(target["qualification"]["checkpoint_every_windows"], 100)
        self.assertEqual(validate_resume_config(self.config, target, 4, "joint"), {})
        for invalid in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                continuation_config(self.config, invalid)

    def setUp(self):
        torch.set_num_threads(1)
        self.config = smoke_config(json.loads((ROOT / "configs/hrl_ppo_only_unlock.json").read_text()))
        self.config["training"].update(pretrain_windows=2, joint_windows=4, replay_capacity=64,
                                       ppo_update_macros=7, ppo_std_decay_windows=4,
                                       sac_critic_steps_per_update=2)
        self.config["qualification"].update(validation_every_windows=1, tail_checkpoints=1,
                                            validation_macros=6, validation_seeds=[33000], test_seeds=[33500],
                                            checkpoint_every_windows=1, joint_evaluation_mode="final_only")
        self.config["validation_seeds"], self.config["test_seeds"] = [33000], [33500]

    def test_snapshot_copy_leaves_full_training_bitwise_unchanged(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            control = run_mainline(self.config, root / "control")
            recorder = RetainBoundaries(root / "snapshots")
            with patch("hrl_uav.qualification.save_boundary_checkpoint", side_effect=recorder):
                treatment = run_mainline(self.config, root / "watched")
            a = torch.load(root / "control/final.pt", map_location="cpu", weights_only=False)
            b = torch.load(root / "watched/final.pt", map_location="cpu", weights_only=False)
            self.assertTrue(state_equal(a, b))
            self.assertEqual(control["mean"], treatment["mean"])
            self.assertEqual(control["updates_total"], treatment["updates_total"])
            self.assertEqual([r["global_window"] for r in recorder.records], [3, 4, 5, 6])
            last = torch.load(recorder.records[-1]["checkpoint"], map_location="cpu", weights_only=False)
            self.assertTrue(state_equal(b, last))

    def test_paired_replay_preserves_networks_optimizers_replay_and_rng(self):
        mdp = QueueMDP(self.config)
        mdp.reset(1, 6)
        agent = DirectHRL.from_mdp(mdp, self.config)
        with tempfile.TemporaryDirectory() as directory:
            readings = paired_readings(self.config, agent, [33000], 6, Path(directory), "check")
        self.assertTrue(all(readings["checks"].values()))
        self.assertEqual(agent.sac_updates, 0)
        self.assertEqual(agent.ppo_updates, 0)

    def test_only_monitoring_and_budget_change(self):
        target = copy.deepcopy(self.config)
        target["training"]["joint_windows"] += 200
        target["qualification"]["checkpoint_every_windows"] = 50
        self.assertEqual(validate_resume_config(self.config, target, 4, "joint"), {})
        target["training"]["pretrain_lr"][0] /= 2
        with self.assertRaises(ValueError):
            validate_resume_config(self.config, target, 4, "joint")

    def test_trends_use_actual_indices_and_do_not_create_a_pass_verdict(self):
        points = []
        for x in (1600, 1651, 1702, 1750, 1800):
            values = {k: .1*x for k in ("task_delay_weighted_ms", "task_delay_low_ms", "J")}
            points.append({"global_window": x, "mean": values, "centroid_mean": values})
        result = trend_summary(points)
        self.assertAlmostEqual(result["last_three"]["mean"]["J"]["slope_per_100_windows"], 10)
        self.assertNotIn("passed", result)


if __name__ == "__main__":
    unittest.main()
