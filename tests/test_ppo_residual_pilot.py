"""Formal-run orchestration checks; synthetic batches do not run the pilot."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_uav.mdp import QueueMDP
from hrl_uav.learning import tensor
from hrl_uav.ppo_residual_b import ResidualAgent, ResidualJournal, baseline_b, project_action
from hrl_uav.ppo_residual_pilot import (
    ROOT, PilotJournal, FullBatchFrozen, accepted_source, assert_budget, evaluate_full,
)
from hrl_uav.resumption import capture_rng, restore_rng, state_equal


class PilotProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = json.loads((ROOT/"configs/sac_only_wireless6_pilot.json").read_text())
        self.cfg["training"].update(joint_windows=120, replay_capacity=64)
        self.mdp = QueueMDP(self.cfg)
        self.mdp.reset(72000, 2)

    def test_accepted_implementation_is_unchanged(self):
        self.assertEqual(len(accepted_source()), 3)

    def test_forced_and_oversized_updates_fail_before_learning(self):
        agent = ResidualAgent.from_mdp(self.mdp, self.cfg)
        proxy = FullBatchFrozen(agent)
        with self.assertRaisesRegex(AssertionError, "forbids forced"):
            proxy.update_ppo(1, force=True)
        agent.macro_rollout = [None]*301
        with self.assertRaisesRegex(AssertionError, "exceeded"):
            proxy.update_ppo(.1)
        self.assertEqual(agent.ppo_updates, 0)

    def test_wrong_scenes_and_observation_points_never_enter_environment(self):
        with patch("hrl_uav.ppo_residual_pilot.rollout") as run:
            for seed, window in [(s, 120) for s in range(33700, 33710)]+[(72000, 120), (33610, 30), (33600, 60.1)]:
                with self.subTest(seed=seed, window=window), self.assertRaises(ValueError):
                    evaluate_full(self.cfg, None, ROOT, seed, window)
            run.assert_not_called()

    def test_pilot_log_extension_matches_logging_off_full_300_batch(self):
        rng = capture_rng()
        plain = ResidualAgent.from_mdp(self.mdp, self.cfg)
        restore_rng(rng)
        logged = ResidualAgent.from_mdp(self.mdp, self.cfg)
        state = self.mdp.macro_observation()
        e = self.mdp.env
        b, detail = baseline_b(e.uav_coords, e.weighted_centroids, e.zone_expected_offered_load_mbps, 60)
        records = []
        for index in range(300):
            delta, lp = plain.macro_action(state)
            mean, std = plain.macro_actor(tensor(state).unsqueeze(0))
            executed, projected = project_action(b+delta)
            plain.macro_rollout.append((state.copy(), delta.copy(), lp,
                float(-1+.02*np.sin(index)), state.copy(), float(index % 90 != 89)))
            # Explicit synthetic execution metadata, not measured physical displacement.
            records.append(dict(scene_seed=72000+index//90, macro_index=index % 90,
                global_macro_index=index, state=state.copy(), current_uav=e.uav_coords.copy(),
                current_centroid=e.weighted_centroids.copy(), current_load=e.zone_expected_offered_load_mbps.copy(),
                B_target=detail["interpolated_target"], B_feasible_target=detail["feasible_target"],
                B_projection=detail["projected"], a_B=b.copy(), sampled_delta=delta.copy(),
                residual_mean=mean.detach().cpu().numpy().ravel(), residual_std=std.detach().cpu().numpy().ravel(),
                pre_projection=b+delta, a_exec=executed, projected=projected,
                actual_displacement=np.zeros((3, 2)), execution_complete=True, tail_only=False))
        logged.macro_rollout = copy.deepcopy(plain.macro_rollout)
        with tempfile.TemporaryDirectory() as tmp:
            journal = PilotJournal(Path(tmp)/"samples")
            journal.records = records
            logged.ppo_journal = journal
            before = capture_rng()
            plain_proxy, logged_proxy = FullBatchFrozen(plain), FullBatchFrozen(logged)
            plain_proxy.update_ppo(10/120)
            rng_after = capture_rng()
            restore_rng(before)
            logged_proxy.update_ppo(10/120)
            self.assertTrue(state_equal(plain.checkpoint(self.cfg), logged.checkpoint(self.cfg)))
            self.assertTrue(state_equal(rng_after, capture_rng()))
            assert_budget(logged_proxy, journal, 10)
            with np.load(journal.directory/"update001.npz") as data:
                self.assertEqual(data["state"].shape, (300, 67))
                np.testing.assert_array_equal(data["update_id"], 1)
                np.testing.assert_array_equal(data["training_initialization_seed"], 101)
                np.testing.assert_array_equal(data["global_macro_index"], np.arange(300))
                self.assertTrue(np.isfinite(data["advantage"]).all())
            self.assertEqual(logged_proxy.ppo_update_events[0]["transitions"], 300)
            self.assertFalse(logged_proxy.ppo_update_events[0]["forced"])
            journal.cursor -= 1
            with self.assertRaisesRegex(AssertionError, "conservation"):
                assert_budget(logged_proxy, journal, 10)
            saved = logged_proxy.checkpoint(self.cfg)
            self.assertFalse(saved["experiment"]["smoke_only"])
            self.assertFalse(saved["experiment"]["resume_supported"])


if __name__ == "__main__":
    unittest.main()
