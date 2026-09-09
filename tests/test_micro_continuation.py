import copy
import csv
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from hrl_uav.experiment import smoke_config
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.qualification import run_mainline
from hrl_uav.resumption import (capture_rng, load_source, restore_agent, restore_rng,
                                state_equal, validate_resume_config)


ROOT = Path(__file__).resolve().parents[1]


class MicroContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = smoke_config(json.loads((ROOT / "configs/micro_budget600.json").read_text()))
        self.config["training"].update(pretrain_windows=2, replay_capacity=64,
                                       sac_actor_warmup_updates=3)
        self.config["qualification"].update(validation_every_windows=1, validation_macros=6,
                                            validation_seeds=[33000], test_seeds=[33500])
        self.config["validation_seeds"] = [33000]
        self.config["test_seeds"] = [33500]

    def test_only_budget_changes_allowed(self):
        target = copy.deepcopy(self.config)
        target["training"]["pretrain_windows"] = 4
        validate_resume_config(self.config, target, 2)
        target["training"]["pretrain_lr"][0] /= 2
        with self.assertRaisesRegex(ValueError, "only"):
            validate_resume_config(self.config, target, 2)
        with self.assertRaisesRegex(ValueError, "larger"):
            validate_resume_config(self.config, self.config, 2)

    def test_rng_roundtrip(self):
        state = capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(4))
        restore_rng(state)
        actual = (random.random(), np.random.rand(), torch.rand(4))
        self.assertTrue(state_equal(expected, actual))

    def test_smoke_cannot_resume(self):
        with self.assertRaisesRegex(ValueError, "smoke"):
            run_mainline(self.config, "unused", smoke=True, resume_from="missing.pt")

    def test_full_state_split_run_matches_uninterrupted_training(self):
        target = copy.deepcopy(self.config)
        target["training"]["pretrain_windows"] = 4
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_mainline(target, root / "full", pretrain_only=True)
            run_mainline(self.config, root / "part", pretrain_only=True)
            result = run_mainline(target, root / "resumed", pretrain_only=True,
                                  resume_from=root / "part/micro_pretrained.pt")
            def read(path):
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    return list(csv.DictReader(handle))
            full_rows = read(root / "full/training.csv")
            self.assertEqual(full_rows[2:], read(root / "resumed/training.csv"))
            full = torch.load(root / "full/micro_pretrained.pt", map_location="cpu", weights_only=False)
            resumed = torch.load(root / "resumed/micro_pretrained.pt", map_location="cpu", weights_only=False)
            self.assertTrue(state_equal(full, resumed))
            self.assertEqual(result["stage_audits"][0]["trained_windows_this_run"], 2)
            self.assertEqual(result["resume"]["stage_initial_updates"], 0)
            self.assertEqual(result["resume"]["rng_mode"], "restored_global_rng")
            self.assertTrue(all(result["resume"]["exact_restore_checks"].values()))

    def test_legacy_restore_recovers_counter_without_rewarming(self):
        target = copy.deepcopy(self.config)
        target["training"]["pretrain_windows"] = 4
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "legacy"
            run_mainline(self.config, root, pretrain_only=True)
            path = root / "micro_pretrained.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint.pop("run_state")
            checkpoint.pop("training_state")
            checkpoint["replay"].pop("memory_counter")
            checkpoint["replay"].pop("capacity")
            torch.save(checkpoint, path)
            saved, metadata = load_source(path, target)
            env = QueueMDP(target)
            env.reset(1, 2)
            agent = DirectHRL.from_mdp(env, target)
            audit = restore_agent(agent, saved, metadata)
            self.assertEqual(agent.replay.memory_counter, 72)
            self.assertEqual(agent.stage_initial_updates, 0)
            self.assertEqual(audit["rng_mode"], "legacy_global_rng_reseeded")
            before = agent.sac_actor_updates
            agent.update_sac()
            self.assertGreater(agent.sac_actor_updates, before)


if __name__ == "__main__":
    unittest.main()
