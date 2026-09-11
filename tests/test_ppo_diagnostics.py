import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from hrl_uav.experiment import rollout, smoke_config
from hrl_uav.learning import DirectHRL, gae, tensor
from hrl_uav.mdp import QueueMDP
from hrl_uav.ppo_diagnostics import LoggedDirectHRL, PPOJournal, action_geometry, b_actions
from hrl_uav.ppo_frozen import FrozenSACUpdates, assert_sac_frozen, sac_learning_state
from hrl_uav.resumption import capture_rng, restore_rng, state_equal
from hrl_uav.target_tracking_check import tracking_action


ROOT = Path(__file__).resolve().parents[1]


class PPOJournalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        base = json.loads((ROOT / "configs/sac_only_wireless6_pilot.json").read_text())
        self.cfg = smoke_config(base)
        self.cfg["training"].update(joint_windows=1, ppo_initial_std=1/6, replay_capacity=64,
                                    ppo_update_macros=4)
        self.mdp = QueueMDP(self.cfg)
        self.mdp.reset(72000, 2)

    def test_state_derived_b_matches_real_controller_with_float32_tolerance(self):
        for displacement in (np.zeros((3, 2)), np.array([[5, -4], [200, 0], [0, -180]])):
            self.mdp.env.uav_coords += displacement
            state = self.mdp.macro_observation()
            actual, projected = b_actions([state], self.mdp.observation_spec())
            expected, detail = tracking_action(self.mdp.env.uav_coords, self.mdp.env.weighted_centroids,
                                               self.mdp.env.zone_expected_offered_load_mbps,
                                               [[0, 0]]*3, 60, .5)
            np.testing.assert_allclose(actual[0], expected, atol=2e-6, rtol=0)
            self.assertEqual(projected[0], detail["projected"].sum())

    def test_geometry_handles_zero_and_distinguishes_alignment_from_closeness(self):
        b = np.array([[0., 0.], [1., 0.], [1., 0.]])
        d = action_geometry([[0, 0], [0, 1], [3, 0]], b)
        self.assertTrue(np.isnan(d["angle_deg"][0]))
        np.testing.assert_allclose(d["angle_deg"][1:], [90, 0])
        np.testing.assert_allclose(d["projection_gain"][1:], [0, 3])
        self.assertEqual(d["distance_to_B"][2], 2)

    def test_actual_300_sample_update_bitwise_equal_with_logging(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["training"]["ppo_update_macros"] = 300
        initialization_rng = capture_rng()
        plain = DirectHRL.from_mdp(self.mdp, cfg)
        restore_rng(initialization_rng)
        logged = LoggedDirectHRL.from_mdp(self.mdp, cfg)
        state = self.mdp.macro_observation()
        rng = np.random.default_rng(1234)
        for index in range(300):
            action, lp = plain.macro_action(state)
            plain.macro_rollout.append((state.copy(), action.copy(), lp,
                                       float(-.1 + rng.normal(0, .01)), state.copy(), float(index % 30 != 29)))
        logged.macro_rollout = copy.deepcopy(plain.macro_rollout)
        frozen = sac_learning_state(logged, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            logged.ppo_journal = PPOJournal(Path(tmp) / "journal", logged.observation_spec, cfg)
            before = capture_rng()
            plain.update_ppo(.25)
            after = capture_rng()
            restore_rng(before)
            logged.update_ppo(.25)
            self.assertTrue(state_equal(after, capture_rng()))
            self.assertTrue(state_equal(plain.checkpoint(cfg), logged.checkpoint(cfg)))
            assert_sac_frozen(logged, cfg, frozen)
            self.assertEqual(logged.ppo_updates, 1)
            with np.load(Path(tmp) / "journal/update001.npz") as saved:
                self.assertEqual(saved["state"].shape, (300, 67))
                adv, returns = gae(saved["reward"], saved["value"], saved["next_value"],
                                   saved["continuation"], cfg["training"]["gamma_per_resource_frame"]**20,
                                   cfg["training"]["gae_lambda"])
                np.testing.assert_array_equal(saved["advantage"], adv)
                np.testing.assert_array_equal(saved["returns"], returns)
                np.testing.assert_array_equal(saved["advantage_normalized"], tensor((adv-adv.mean())/(adv.std()+1e-8)).cpu().numpy())
                np.testing.assert_allclose(saved["exploration_std"], 1/6, atol=1e-8)
            logged.update_ppo(1, force=True)
            self.assertEqual(len(list((Path(tmp)/"journal").glob("update*.npz"))), 1)
            with (Path(tmp)/"journal/updates.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["ppo_gradient_steps"]), logged.ppo_gradient_steps)

    def test_real_rollout_logging_preserves_rng_replay_boundaries_and_all_learning_state(self):
        initialization_rng = capture_rng()
        plain = DirectHRL.from_mdp(self.mdp, self.cfg)
        plain.set_stage("joint")
        restore_rng(initialization_rng)
        logged = LoggedDirectHRL.from_mdp(self.mdp, self.cfg)
        logged.set_stage("joint")
        frozen = sac_learning_state(logged, self.cfg)
        with tempfile.TemporaryDirectory() as tmp:
            logged.ppo_journal = PPOJournal(Path(tmp)/"journal", logged.observation_spec, self.cfg)
            results = []
            start_rng = capture_rng()
            for agent in (plain, logged):
                restore_rng(start_rng)
                proxy = FrozenSACUpdates(agent)
                summary, rows, _, _, _ = rollout(self.cfg, proxy, 72000, 2, macro_enabled=True,
                                                 train=True, complete_tail=True)
                self.assertEqual(agent.ppo_updates, 0)
                proxy.update_ppo(1, force=True)
                self.assertEqual(summary["boundary_transitions"], 2)
                self.assertEqual(summary["final_pending_dropped"], 0)
                self.assertEqual(agent.replay.memory_counter, 36)
                self.assertEqual(proxy.consumed_macro_transitions, 2)
                self.assertGreater(agent.ppo_updates, 0)
                self.assertEqual(agent.sac_updates, 0)
                assert_sac_frozen(agent, self.cfg, frozen)
                results.append((summary, rows, capture_rng(), copy.deepcopy(agent.checkpoint(self.cfg))))
            self.assertTrue(state_equal(results[0], results[1]))


if __name__ == "__main__":
    unittest.main()
