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
from hrl_uav.qualification import run_mainline, training_stage_seed
from hrl_uav.resumption import (
    capture_rng, load_source, restore_agent, restore_rng, state_equal, validate_resume_config,
)


ROOT = Path(__file__).resolve().parents[1]


class PPOOnlyUnlockTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.production = json.loads((ROOT / "configs/hrl_ppo_only_unlock.json").read_text())
        self.config = smoke_config(self.production)
        self.config["training"].update(pretrain_windows=2, joint_windows=2, replay_capacity=64,
                                       ppo_std_decay_windows=2)
        self.config["qualification"].update(
            validation_every_windows=1, tail_checkpoints=1, validation_macros=6,
            validation_seeds=[33000], test_seeds=[33500], checkpoint_every_windows=1,
        )
        self.config["validation_seeds"], self.config["test_seeds"] = [33000], [33500]

    def filled_agent(self, warmup=0):
        config = copy.deepcopy(self.config)
        config["training"]["sac_actor_warmup_updates"] = warmup
        env = QueueMDP(config)
        env.reset(1, 2)
        agent = DirectHRL.from_mdp(env, config)
        state = env.micro_observation()
        action = np.zeros(env.micro_action_dim, dtype=np.float32)
        route = np.full((2, 3, 3), 1 / 3)
        for _ in range(config["training"]["replay_warmup"]):
            agent.store(state, action, route, -0.1, state, 0.99)
        return agent, config

    def assert_micro_equal(self, left, right, config):
        a, b = left.checkpoint(config), right.checkpoint(config)
        for key in ("sac_actor", "sac_critic", "sac_target", "log_alpha", "replay", "updates"):
            self.assertTrue(state_equal(a[key], b[key]), key)
        self.assertTrue(state_equal(a["optimizers"][:3], b["optimizers"][:3]))
        self.assertEqual(left.stage_initial_updates, right.stage_initial_updates)
        self.assertEqual(left.last_sac_stats, right.last_sac_stats)

    def clone_agent(self, agent, config):
        saved = copy.deepcopy(agent.checkpoint(config))
        saved["run_state"] = {"rng": capture_rng()}
        env = QueueMDP(config)
        env.reset(1, 2)
        cloned = DirectHRL.from_mdp(env, config)
        restore_agent(cloned, saved, {"replay_counter": agent.replay.memory_counter})
        return cloned

    def test_only_unlock_preserves_entire_sac_and_next_updates(self):
        agent, config = self.filled_agent()
        for _ in range(4):
            agent.update_sac()
        self.assertGreater(agent.sac_actor_updates, 0)
        control = self.clone_agent(agent, config)
        rng = capture_rng()
        agent.set_stage("joint")
        self.assertTrue(state_equal(rng, capture_rng()))
        self.assert_micro_equal(agent, control, config)
        for _ in range(4):
            rng = capture_rng()
            control.update_sac()
            control_rng_after = capture_rng()
            restore_rng(rng)
            agent.update_sac()
            self.assertTrue(state_equal(control_rng_after, capture_rng()))
            self.assert_micro_equal(agent, control, config)
        self.assertGreater(agent.sac_actor_updates, 2)

    def test_unlock_neither_restarts_nor_skips_unfinished_actor_warmup(self):
        agent, config = self.filled_agent(warmup=3)
        agent.update_sac()
        agent.update_sac()
        control = self.clone_agent(agent, config)
        agent.set_stage("joint")
        self.assertEqual(agent.stage_initial_updates, 0)
        for expected_actor_updates in (0, 1):
            rng = capture_rng()
            control.update_sac()
            restore_rng(rng)
            agent.update_sac()
            self.assert_micro_equal(agent, control, config)
            self.assertEqual(agent.sac_actor_updates, expected_actor_updates)

    def test_preserve_rejects_ambiguous_legacy_sac_overrides(self):
        env = QueueMDP(self.config)
        env.reset(1, 2)
        for key, value in (("joint_lr", [1e-5] * 3), ("joint_sac_actor_warmup_updates", 0)):
            bad = copy.deepcopy(self.config)
            bad["training"][key] = value
            with self.assertRaisesRegex(ValueError, "remove joint SAC overrides"):
                DirectHRL.from_mdp(env, bad)
        bad = copy.deepcopy(self.config)
        bad["training"]["sac_stage_policy"] = "typo"
        with self.assertRaisesRegex(ValueError, "Unknown SAC stage policy"):
            DirectHRL.from_mdp(env, bad)

    def test_invalid_stage_does_not_mutate_agent(self):
        agent, _ = self.filled_agent()
        with self.assertRaisesRegex(ValueError, "Unknown training stage"):
            agent.set_stage("typo")
        self.assertEqual(agent.stage, "pretrain")

    def test_new_recipe_does_not_change_mdp_sac_algorithm_or_ppo_hyperparameters(self):
        old = json.loads((ROOT / "configs/hrl_resume_joint600.json").read_text())
        new = copy.deepcopy(self.production)
        self.assertEqual(new["training"].pop("sac_stage_policy"), "preserve")
        old["training"].pop("joint_lr")
        old["training"].pop("joint_sac_actor_warmup_updates")
        self.assertEqual(new["qualification"].pop("training_scene_schedule"), "continuous")
        self.assertEqual(new["qualification"].pop("joint_evaluation_mode"), "final_only")
        self.assertEqual(new["qualification"].pop("checkpoint_every_windows"), 6)
        self.assertEqual(old, new)

    def test_continuous_seeds_and_legacy_compatibility(self):
        self.assertEqual(training_stage_seed(self.production, "pretrain"), 61000)
        self.assertEqual(training_stage_seed(self.production, "joint"), 61067)
        legacy = json.loads((ROOT / "configs/hrl_resume_joint600.json").read_text())
        self.assertEqual(training_stage_seed(legacy, "joint"), 71000)
        bad = copy.deepcopy(self.config)
        bad["qualification"]["training_scene_schedule"] = "typo"
        with self.assertRaisesRegex(ValueError, "Unknown training scene schedule"):
            training_stage_seed(bad, "pretrain")

    def test_legacy_joint_checkpoint_cannot_silently_become_corrected_training(self):
        old = json.loads((ROOT / "configs/hrl_resume_joint600.json").read_text())
        with self.assertRaisesRegex(ValueError, "only"):
            validate_resume_config(old, self.production, 320, "joint")

    def test_real_run_both_layers_update_preserved_sac_restores_exactly(self):
        target = copy.deepcopy(self.config)
        target["training"]["joint_windows"] = 4
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            run_mainline(target, root / "full")
            report = run_mainline(self.config, root / "part")
            pretrain, joint = report["stage_audits"]
            self.assertEqual(pretrain["ppo_updates"], 0)
            self.assertGreater(joint["ppo_updates"], 0)
            self.assertGreater(joint["actor_updates"], 0)
            self.assertGreater(joint["critic_updates"], 0)
            self.assertEqual(joint["replay_at_stage_entry"], pretrain["replay_at_stage_exit"])
            audit = json.loads((root / "part/joint_transition_audit.json").read_text())
            self.assertEqual(audit["sac_stage_policy"], "preserve")
            self.assertTrue(all(audit["preserved"].values()))
            self.assertTrue(all(audit["sac_preservation_checks"].values()))
            self.assertEqual(audit["sac_learning_rates"], self.config["training"]["pretrain_lr"])
            self.assertEqual(audit["actor_warmup_clock_origin"], 0)
            self.assertEqual(audit["joint_first_scene_seed"], 61002)
            manifest = json.loads((root / "part/manifest.json").read_text())
            self.assertEqual(manifest["training_scene_seeds"], [61000, 61001, 61002, 61003])
            saved, metadata = load_source(root / "part/final.pt", target)
            env = QueueMDP(target)
            env.reset(1, 2)
            restored = DirectHRL.from_mdp(env, target)
            restored_audit = restore_agent(restored, saved, metadata)
            self.assertTrue(all(restored_audit["exact_restore_checks"].values()))
            self.assertEqual([rates[0] for rates in restored_audit["optimizer_learning_rates"][:3]],
                             self.config["training"]["pretrain_lr"])
            run_mainline(target, root / "resumed", resume_from=root / "part/final.pt")
            full = torch.load(root / "full/final.pt", map_location="cpu", weights_only=False)
            resumed = torch.load(root / "resumed/final.pt", map_location="cpu", weights_only=False)
            self.assertTrue(state_equal(full, resumed))


if __name__ == "__main__":
    unittest.main()
