"""Fresh initialization and statistical-level checks; no performance evaluation."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hrl_uav.bandwidth import UniformRouting
from hrl_uav.experiment import isolated_rng, rollout
from hrl_uav.scene_prescreen import assert_crn_equal
from hrl_uav.wireless_long import rows, stream_plan
from hrl_uav.wireless_pilot import MEASURES, PilotMeter
from hrl_uav.wireless_replicates import (REFERENCE, fresh_agent, fresh_config,
    load_json, nested_stats, record_window_optimization)


class ReplicateTest(unittest.TestCase):
    def test_only_initialization_changes_and_no_weights_are_loaded(self):
        baseline = load_json(REFERENCE / "manifest.json")["config"]
        actors = []
        with isolated_rng(), tempfile.TemporaryDirectory() as tmp, patch("torch.load", side_effect=AssertionError("load forbidden")):
            for seed in (102, 103):
                cfg = fresh_config(baseline, seed)
                expected = copy.deepcopy(baseline)
                expected["training"]["seed"] = seed
                self.assertEqual(cfg, expected)
                agent, _, _, _, audit = fresh_agent(cfg, Path(tmp))
                self.assertTrue(all(audit["checks"].values()))
                self.assertEqual(agent.sac_actor_updates, 0)
                actors.append(audit["actor_sha256"])
            self.assertNotEqual(*actors)
        plan = list(stream_plan(0, 71000))
        self.assertEqual(plan[0], (0, 3, 71000))
        self.assertEqual(plan[-1], (1998, 2, 71666))
        self.assertEqual(sum(n for _, n, _ in plan), 2000)

    def test_scene_crn_independent_of_initialization_setting(self):
        baseline = load_json(REFERENCE / "manifest.json")["config"]
        frames = []
        with isolated_rng():
            for seed in (102, 103):
                cfg = fresh_config(baseline, seed)
                with patch("hrl_uav.experiment.QueueMDP", PilotMeter):
                    _, _, _, _, meter = rollout(cfg, UniformRouting(), 71000, 2,
                                                macro_enabled=False, train=False, complete_tail=False)
                frames.append(meter.crn_frames)
        assert_crn_equal(*frames)

    def test_warmup_without_updates_has_an_empty_window_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            record_window_optimization(output, 1, {})
            row = rows(output / "optimization_windows.csv")[0]
            self.assertEqual(row["events"], 0)
            self.assertEqual(row["critic_loss_mean"], "")

    def test_summary_unit_is_training_seed_not_scene(self):
        per_seed = [dict(training_seed=s, **dict.fromkeys(MEASURES, value))
                    for s, value in zip((101, 102, 103), (1., 2., 6.))]
        stats = nested_stats(per_seed)
        self.assertEqual(stats[0]["training_seed_count"], 3)
        self.assertAlmostEqual(stats[0]["mean"], 3.)
        self.assertAlmostEqual(stats[0]["std"], np.std([1., 2., 6.], ddof=1))
        with self.assertRaises(ValueError):
            nested_stats(per_seed + [per_seed[0]])


if __name__ == "__main__":
    unittest.main()
