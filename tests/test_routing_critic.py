import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from hrl_uav.experiment import smoke_config, rollout
from hrl_uav.learning import DirectHRL, tensor
from hrl_uav.mdp import QueueMDP
from hrl_uav.routing import project_rows_to_simplex, project_rows_to_simplex_tensor


ROOT = Path(__file__).resolve().parents[1]


class RoutingCriticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = smoke_config(json.loads((ROOT/"configs/main.json").read_text()))
        self.config["training"]["critic_action_representation"] = "executed_route"

    def test_projection_matches_execution_including_sparse_boundaries(self):
        raw = np.random.default_rng(881).uniform(-1, 1, (128, 2, 3, 3))
        raw[0] = 0
        raw[1] = np.tile(2*np.eye(3)-1, (2,1,1))
        actual = project_rows_to_simplex_tensor(torch.tensor(raw)).numpy()
        np.testing.assert_allclose(actual, project_rows_to_simplex(raw), atol=1e-12)

    def test_projection_derivative_is_correct_away_from_kinks(self):
        x = torch.tensor([[0.11,0.22,-0.13], [0.8,0.2,-0.7]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(project_rows_to_simplex_tensor, (x,)))

    def test_equivalent_routes_have_equal_q_and_zero_null_gradient(self):
        env = QueueMDP(self.config)
        state = tensor(env.reset(1,6)).unsqueeze(0)
        agent = DirectHRL.from_mdp(env, self.config)
        raw = torch.zeros((1,18), device=state.device, requires_grad=True)
        for network in (agent.sac_critic.critic_v, agent.sac_critic.target_critic_v):
            q1, q2 = network(state, raw)
            for shift in (-0.25, 0.25):
                p1, p2 = network(state, raw+shift)
                torch.testing.assert_close(q1,p1,atol=1e-5,rtol=1e-5)
                torch.testing.assert_close(q2,p2,atol=1e-5,rtol=1e-5)
            grad, = torch.autograd.grad(q1.sum(), raw)
            torch.testing.assert_close(grad.reshape(1,6,3).sum(-1), torch.zeros((1,6),device=grad.device),atol=1e-6,rtol=0)
            self.assertGreater(float(grad.abs().sum()), 0)

    def test_initial_actor_and_network_parameters_remain_matched(self):
        env = QueueMDP(self.config)
        state = env.reset(1,6)
        raw_config = copy.deepcopy(self.config)
        raw_config["training"]["critic_action_representation"] = "raw_score"
        torch.manual_seed(77)
        raw_agent = DirectHRL.from_mdp(env, raw_config)
        torch.manual_seed(77)
        route_agent = DirectHRL.from_mdp(env, self.config)
        for left, right in ((raw_agent.sac_actor.action_net, route_agent.sac_actor.action_net),
                            (raw_agent.sac_critic.critic_v, route_agent.sac_critic.critic_v)):
            for key,value in left.state_dict().items():
                self.assertTrue(torch.equal(value, right.state_dict()[key]))
        np.testing.assert_array_equal(raw_agent.sac_actor.choose_action(state, True),
                                      route_agent.sac_actor.choose_action(state, True))

    def test_real_updates_and_checkpoint_config_preserve_representation(self):
        env = QueueMDP(self.config)
        state = env.reset(1,6)
        agent = DirectHRL.from_mdp(env, self.config)
        rollout(self.config,agent,1,2,macro_enabled=False,train=True,complete_tail=True)
        self.assertGreater(agent.sac_actor_updates,0)
        checkpoint = agent.checkpoint(self.config)
        restored = DirectHRL.from_mdp(env, checkpoint["config"])
        restored.sac_critic.critic_v.load_state_dict(checkpoint["sac_critic"])
        self.assertEqual(restored.sac_critic.critic_v.route_width,3)
        for a,b in zip(agent.sac_critic.get_v(tensor(state)[None],tensor(np.zeros(18))[None]),
                       restored.sac_critic.get_v(tensor(state)[None],tensor(np.zeros(18))[None])):
            torch.testing.assert_close(a,b)


if __name__ == "__main__":
    unittest.main()
