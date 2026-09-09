import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from hrl_uav.experiment import rollout
from hrl_uav.reference import (
    SB3ReplayView, comparison_config, fixed_batch_audit, make_agent,
)


ROOT = Path(__file__).resolve().parents[1]


class ReferenceSACTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.base = json.loads((ROOT/"configs/main.json").read_text())
        self.cfg = comparison_config(self.base, smoke=True)

    def test_comparison_does_not_change_mdp(self):
        before = copy.deepcopy(self.base)
        actual = comparison_config(self.base)
        for key in ("environment", "reward", "phase_loads", "phase_dwell", "cpu_ghz", "waypoint_radius_m"):
            self.assertEqual(actual[key], before[key])
        self.assertEqual(self.base, before)
        self.assertEqual(actual["training"]["sac_update_frames"], 1)
        self.assertEqual(actual["training"]["joint_windows"], 0)
        self.assertEqual(actual["training"]["critic_action_representation"], "raw_score")

    def test_upstream_actual_target_and_gradients(self):
        result = fixed_batch_audit(self.cfg)
        self.assertTrue(result["passed"], result)
        self.assertAlmostEqual(result["native_critic_loss"], 2*result["sb3_critic_loss"], places=3)

    def test_replay_discount_and_executed_route_not_confused(self):
        agent, state = make_agent(self.cfg, "sb3")
        n, a = agent.state_dim, agent.action_dim
        for discount in (0.99, 0.99**3, 0.0):
            agent.store(state, np.full(a, 0.1), np.full(a, 1/3), -7, state+0.01, discount)
        data = agent.model.replay_buffer.convert(agent.replay.state_dict()["transitions"])
        np.testing.assert_allclose(data.discounts.cpu().numpy().ravel(), [0.99,0.99**3,0.0])
        np.testing.assert_allclose(data.actions.cpu().numpy(), 0.1)
        np.testing.assert_allclose(data.rewards.cpu().numpy(), -7)
        np.testing.assert_allclose(data.next_observations.cpu().numpy(), np.tile(state+0.01,(3,1)))
        self.assertEqual(data.observations.shape, (3,n))
        self.assertEqual(data.actions.shape, (3,a))

    def test_reference_sampler_does_not_advance_environment_rng(self):
        agent, state = make_agent(self.cfg, "sb3")
        agent.store(state, np.zeros(18), np.full(18,1/3), -1, state, 0.99)
        np.random.seed(729)
        before = np.random.get_state()
        agent.model.replay_buffer.sample(8)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2], after[2])

    def test_fresh_common_initialization_and_uniform_warmup(self):
        native, state = make_agent(self.cfg, "native")
        sb3, _ = make_agent(self.cfg, "sb3")
        for _ in range(8):
            np.testing.assert_array_equal(native.sac_actor.choose_action(state), sb3.sac_actor.choose_action(state))
        np.testing.assert_array_equal(native.sac_actor.choose_action(state, True), sb3.sac_actor.choose_action(state, True))

    def test_real_updates_equal_and_macro_frozen(self):
        counts = []
        for implementation in ("native", "sb3"):
            agent, _ = make_agent(self.cfg, implementation)
            macro = {k:v.clone() for k,v in agent.macro_actor.state_dict().items()}
            summary, _, _, _, _ = rollout(self.cfg, agent, 51000, 2, macro_enabled=False,
                                          train=True, complete_tail=True)
            counts.append((agent.sac_updates, agent.sac_actor_updates, agent.replay.memory_counter))
            self.assertGreater(agent.sac_actor_updates, 0)
            self.assertEqual(summary["final_pending_dropped"], 0)
            for key, value in macro.items():
                self.assertTrue(torch.equal(value, agent.macro_actor.state_dict()[key]))
            rows = agent.replay.state_dict()["transitions"]
            from hrl_uav.routing import project_rows_to_simplex
            for row in rows:
                n, a = agent.micro_state_dim, agent.action_dim
                route = project_rows_to_simplex(row[n:n+a].reshape(2,3,3))
                np.testing.assert_allclose(route.ravel(), row[n+a+2:n+2*a+2], atol=1e-7)
        self.assertEqual(counts[0], counts[1])


if __name__ == "__main__":
    unittest.main()
