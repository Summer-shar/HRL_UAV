import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_uav.experiment import rollout, smoke_config, snapshot, parameter_delta
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP, compress


ROOT = Path(__file__).resolve().parents[1]


class SplitObservationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = json.loads((ROOT / "configs/main.json").read_text())
        self.mdp = QueueMDP(self.config)
        self.mdp.reset(123, 12)

    def feature(self, level, key):
        state = getattr(self.mdp, f"{level}_observation")()
        start, end = getattr(self.mdp, f"{level}_feature_slices")[key]
        return state[start:end]

    def test_distinct_semantic_layouts_and_dimensions(self):
        spec = self.mdp.observation_spec()
        self.assertEqual(spec["layout"], "split_v1")
        self.assertEqual(spec["micro"]["dim"], 68)
        self.assertEqual(spec["macro"]["dim"], 67)
        for item in ("uav_coordinates", "centroid_offset", "centroid_velocity"):
            self.assertNotIn(item, spec["micro"]["slices"])
            self.assertIn(item, spec["macro"]["slices"])
        for item in ("current_arrivals", "macro_clock", "transfer_queue"):
            self.assertIn(item, spec["micro"]["slices"])
            self.assertNotIn(item, spec["macro"]["slices"])
        for level in ("micro", "macro"):
            for item in ("centroids", "unfinished_chunks", "assigned_cpu_seconds"):
                self.assertNotIn(item, spec[level]["slices"])
        np.testing.assert_array_equal(self.mdp.observation(), self.mdp.micro_observation())

    def test_shared_quantities_have_identical_values_not_separate_physics(self):
        self.mdp.env.Q_loc_low[1] = 20
        self.mdp.phase_age = 3
        spec = self.mdp.observation_spec()
        shared = spec["micro"]["slices"].keys() & spec["macro"]["slices"].keys()
        for key in shared:
            np.testing.assert_array_equal(self.feature("micro", key), self.feature("macro", key))
        self.assertGreater(self.feature("macro", "compute_queue")[4], 0)
        self.assertAlmostEqual(float(self.feature("micro", "phase_age")[0]), 0.3)

    def test_macro_transfer_is_destination_aggregate_before_compression(self):
        self.mdp.env.Q_tx_low.fill(0)
        self.mdp.env.Q_tx_low[0, 2] = 10
        self.mdp.env.Q_tx_low[1, 2] = 30
        expected = np.zeros((2, 3))
        expected[1, 2] = 40
        np.testing.assert_allclose(self.feature("macro", "inbound_transfer_queue"),
                                   compress(expected, 10).ravel(), rtol=1e-6)
        micro = self.feature("micro", "transfer_queue").reshape(2, 3, 3)
        self.assertGreater(micro[1, 0, 2], 0)
        self.assertGreater(micro[1, 1, 2], 0)

    def test_observation_reads_are_causal_and_do_not_advance_environment(self):
        before = self.mdp.observation_spec()
        states = [self.mdp.micro_observation(), self.mdp.macro_observation()]
        queues = self.mdp.env.Q_loc_low.copy()
        rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()
        now = self.mdp.env._current_time()
        self.mdp.schedule.phase_index_by_macro[5:] = 2
        self.assertEqual(before, self.mdp.observation_spec())
        for old, new in zip(states, (self.mdp.micro_observation(), self.mdp.macro_observation())):
            np.testing.assert_array_equal(old, new)
        np.testing.assert_array_equal(queues, self.mdp.env.Q_loc_low)
        np.testing.assert_array_equal(rng[1], np.random.get_state()[1])
        self.assertEqual(rng[2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(now, self.mdp.env._current_time())

    def test_macro_motion_changes_micro_link_conditions(self):
        before = self.feature("micro", "u2u_spectral_efficiency").copy()
        action = np.array([0.5, 0, -0.5, 0, 0, 0], dtype=np.float32)
        self.mdp.start_macro(action)
        for _ in range(self.mdp.env.deployment_resource_frames):
            self.mdp.step_frame(np.zeros(18))
        self.mdp.prepare_service()
        after = self.feature("micro", "u2u_spectral_efficiency")
        self.assertGreater(float(np.max(np.abs(before - after))), 1e-5)

    def test_split_and_legacy_observations_preserve_fixed_action_transitions(self):
        legacy = copy.deepcopy(self.mdp)
        legacy.observation_layout = "shared_v1"
        rng = np.random.default_rng(98)
        for _ in range(2):
            macro = rng.uniform(-0.4, 0.4, 6)
            for mdp in (self.mdp, legacy):
                mdp.macro_observation()
                mdp.start_macro(macro)
            for frame in range(self.mdp.env.resource_frames_per_macro):
                if frame >= self.mdp.env.deployment_resource_frames:
                    self.mdp.prepare_service()
                    legacy.prepare_service()
                action = rng.uniform(-1, 1, 18)
                a, b = self.mdp.step_frame(action), legacy.step_frame(action)
                self.assertEqual(a["reward"], b["reward"])
                np.testing.assert_array_equal(a["route"], b["route"])
                np.testing.assert_array_equal(self.mdp.env.Q_loc_low, legacy.env.Q_loc_low)
                np.testing.assert_array_equal(self.mdp.env.Q_tx_low, legacy.env.Q_tx_low)
            for mdp in (self.mdp, legacy):
                mdp.advance_macro_clock()
        self.assertEqual(self.mdp.metrics(), legacy.metrics())

    def test_pretrain_then_joint_preserves_micro_and_uses_separate_buffers(self):
        config = smoke_config(self.config)
        agent = DirectHRL.from_mdp(self.mdp, config)
        before_macro = snapshot(agent.macro_actor)
        rollout(config, agent, 1, 2, macro_enabled=False, train=True, complete_tail=True)
        self.assertGreater(agent.sac_actor_updates, 0)
        self.assertEqual(agent.ppo_updates, 0)
        self.assertEqual(parameter_delta(before_macro, agent.macro_actor), 0)
        before_micro = snapshot(agent.sac_actor.action_net)
        before_critic = snapshot(agent.sac_critic.critic_v)
        replay = agent.replay
        count = replay.memory_counter
        actor_updates = agent.sac_actor_updates
        optimizers = (agent.sac_actor.optimizer, agent.sac_critic.optimizer, agent.entropy.optimizer)
        agent.set_stage("joint")
        self.assertIs(agent.replay, replay)
        self.assertEqual(count, agent.replay.memory_counter)
        self.assertEqual(parameter_delta(before_micro, agent.sac_actor.action_net), 0)
        self.assertEqual(parameter_delta(before_critic, agent.sac_critic.critic_v), 0)
        self.assertEqual(optimizers, (agent.sac_actor.optimizer, agent.sac_critic.optimizer, agent.entropy.optimizer))
        with patch.object(agent, "macro_action", wraps=agent.macro_action) as macro_call, \
                patch.object(agent, "store", wraps=agent.store) as store_call:
            report, _, _, _, _ = rollout(config, agent, 2, 2, train=True, complete_tail=True)
        self.assertEqual(report["final_pending_dropped"], 0)
        self.assertEqual(report["boundary_transitions"], 2)
        for call in macro_call.call_args_list:
            self.assertEqual(call.args[0].shape, (67,))
        for call in store_call.call_args_list:
            self.assertEqual(call.args[0].shape, (68,))
            self.assertEqual(call.args[4].shape, (68,))
        self.assertGreater(agent.sac_actor_updates, actor_updates)
        self.assertGreater(agent.ppo_updates, 0)
        self.assertGreater(parameter_delta(before_macro, agent.macro_actor, ("log_std",)), 0)
        self.assertEqual(replay.state_dict()["transitions"].shape[1], 2*68+2*18+2)
        checkpoint = agent.checkpoint(config)
        self.assertEqual(checkpoint["micro_state_dim"], 68)
        self.assertEqual(checkpoint["macro_state_dim"], 67)
        self.assertEqual(checkpoint["observations"]["layout"], "split_v1")

    def test_wrong_level_or_implicit_macro_dimension_fails_early(self):
        with self.assertRaisesRegex(ValueError, "macro_state_dim"):
            DirectHRL(68, 18, 6, self.config)
        agent = DirectHRL.from_mdp(self.mdp, smoke_config(self.config))
        with self.assertRaisesRegex(ValueError, "PPO requires"):
            agent.macro_action(self.mdp.micro_observation(), enabled=False)
        with self.assertRaisesRegex(ValueError, "SAC replay requires"):
            agent.store(self.mdp.macro_observation(), np.zeros(18), np.zeros(18),
                        -1, self.mdp.micro_observation(), 0.99)

    def test_old_config_keeps_104_or_116_and_cannot_load_into_split_actor(self):
        config = copy.deepcopy(self.config)
        config.pop("observation")
        legacy = QueueMDP(config)
        legacy.reset(1, 6)
        spec = legacy.observation_spec()
        self.assertEqual(spec["micro"]["dim"], 104)
        self.assertEqual(spec["macro"]["dim"], 104)
        old = DirectHRL.from_mdp(legacy, config)
        new = DirectHRL.from_mdp(self.mdp, self.config)
        with self.assertRaises(RuntimeError):
            new.sac_actor.action_net.load_state_dict(old.sac_actor.action_net.state_dict())
        legacy.observe_routing_ready = True
        spec = legacy.observation_spec()
        self.assertEqual(spec["micro"]["dim"], 116)
        self.assertEqual(spec["macro"]["dim"], 116)


if __name__ == "__main__":
    unittest.main()
