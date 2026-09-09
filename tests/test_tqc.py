"""TQC math, legacy SAC equivalence and two-stage/replay contracts."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from hrl_uav.agents.tqc import actor_value, quantile_loss, truncated_target
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.resumption import capture_rng, restore_agent, restore_rng, state_equal


ROOT = Path(__file__).resolve().parents[1]


class TQCTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = json.loads((ROOT / "configs/hrl_ppo_only_unlock.json").read_text())
        self.config["training"].update(batch_size=8, replay_capacity=64, replay_warmup=8,
                                       sac_actor_warmup_updates=0, sac_critic_steps_per_update=2)
        self.mdp = QueueMDP(self.config)
        self.state = self.mdp.reset(33000, 90)

    def build(self, algorithm):
        config = copy.deepcopy(self.config)
        config["training"]["micro_algorithm"] = algorithm
        if algorithm == "tqc":
            config["training"].update(tqc_quantiles=25, tqc_drop_per_critic=2)
        torch.manual_seed(712)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(712)
        return DirectHRL.from_mdp(self.mdp, config), config

    def fill(self, agent):
        rng = np.random.default_rng(101)
        for i in range(24):
            action = rng.uniform(-.2, .2, 18).astype(np.float32)
            agent.store(self.state, action, self.mdp.route(action), -float(i+1), self.state+.001*i,
                        .99**3 if i % 2 else .99)

    def test_pooled_truncation_drops_global_top_not_each_head(self):
        q1 = torch.tensor([[10., 20., 30.]])
        q2 = torch.tensor([[1., 2., 3.]])
        torch.testing.assert_close(truncated_target(q1, q2, 1), torch.tensor([[1., 2., 3., 10.]]))
        self.assertEqual(truncated_target(q1, q2, 0).shape, (1, 6))
        for drop in (-1, 3, True, .5):
            with self.assertRaises(ValueError):
                truncated_target(q1, q2, drop)

    def test_quantile_huber_matches_scalar_reference_and_gradient(self):
        a = torch.tensor([[.2, -1.1], [2.1, .7]], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([[1.5, -.4], [-2.2, .1]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[-2., .4, 3.], [.2, -.7, 1.]], dtype=torch.float64, requires_grad=True)
        ref = 0.
        for row in range(2):
            for head in (a, b):
                for i in range(2):
                    for j in range(3):
                        delta = float(target[row, j].detach() - head[row, i].detach())
                        huber = .5 * delta**2 if abs(delta) <= 1 else abs(delta) - .5
                        ref += abs((i+.5)/2 - float(delta < 0)) * huber / 24
        loss = quantile_loss(a, b, target)
        self.assertAlmostEqual(loss.item(), ref, places=12)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertTrue(torch.autograd.gradcheck(lambda x, y: quantile_loss(x, y, target), (a, b)))

    def test_actor_value_is_untruncated_mean(self):
        q1 = torch.tensor([[1., 3.]])
        q2 = torch.tensor([[5., 11.]])
        torch.testing.assert_close(actor_value(q1, q2), torch.tensor([[5.]]))

    def test_matching_actor_ppo_initialization_and_rng(self):
        sac, _ = self.build("sac")
        rng = capture_rng()
        tqc, _ = self.build("tqc")
        self.assertTrue(state_equal(rng, capture_rng()))
        for name in ("macro_actor", "macro_critic"):
            self.assertTrue(state_equal(getattr(sac, name).state_dict(), getattr(tqc, name).state_dict()))
        self.assertTrue(state_equal(sac.sac_actor.action_net.state_dict(), tqc.sac_actor.action_net.state_dict()))
        self.assertTrue(state_equal(sac.entropy.log_alpha, tqc.entropy.log_alpha))
        self.assertEqual(tqc.replay.memory_counter, 0)

    def test_projection_invariance_actor_gradients_clocks_and_restore(self):
        agent, config = self.build("tqc")
        self.fill(agent)
        before = copy.deepcopy(agent.sac_actor.action_net.state_dict())
        initial_macro = copy.deepcopy(agent.macro_actor.state_dict())
        state = torch.as_tensor(self.state, device=agent.entropy.log_alpha.device).unsqueeze(0)
        action = torch.full((1, 18), .1, device=state.device, requires_grad=True)
        q1, q2 = agent.sac_critic.get_v(state, action)
        r1, r2 = agent.sac_critic.get_v(state, action + .3)
        torch.testing.assert_close(q1, r1, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(q2, r2, atol=1e-6, rtol=1e-6)
        actor_value(q1, q2).sum().backward()
        self.assertGreater(action.grad.abs().max().item(), 0)
        for _ in range(4):
            old_target = copy.deepcopy(agent.sac_critic.target_critic_v.state_dict())
            agent.update_sac()
            for key, value in agent.sac_critic.target_critic_v.state_dict().items():
                torch.testing.assert_close(value, old_target[key].lerp(agent.sac_critic.critic_v.state_dict()[key], agent.cfg["tau"]), rtol=0, atol=0)
        self.assertEqual((agent.sac_updates, agent.sac_actor_updates, agent.sac_critic_gradient_steps, agent.sac_target_updates), (4, 2, 8, 4))
        self.assertFalse(state_equal(before, agent.sac_actor.action_net.state_dict()))
        self.assertTrue(state_equal(initial_macro, agent.macro_actor.state_dict()))
        saved = copy.deepcopy(agent.checkpoint(config))
        saved["run_state"] = {"rng": capture_rng()}
        agent.set_stage("joint")
        switched = agent.checkpoint(config)
        for key in ("optimizers", "sac_actor", "sac_critic", "sac_target", "replay", "updates"):
            self.assertTrue(state_equal(saved[key], switched[key]), key)
        restored, _ = self.build("tqc")
        audit = restore_agent(restored, saved, {"replay_counter": agent.replay.memory_counter})
        self.assertTrue(all(audit["exact_restore_checks"].values()))
        rng = capture_rng()
        restored.update_sac()
        expected = restored.checkpoint(config)
        restore_rng(rng)
        agent.update_sac()
        actual = agent.checkpoint(config)
        for key in ("optimizers", "sac_actor", "sac_critic", "sac_target", "replay", "updates"):
            self.assertTrue(state_equal(expected[key], actual[key]), key)

    def test_cross_algorithm_restore_rejected(self):
        sac, config = self.build("sac")
        tqc, _ = self.build("tqc")
        with self.assertRaisesRegex(ValueError, "not interchangeable"):
            restore_agent(tqc, sac.checkpoint(config), {"replay_counter": 0})


if __name__ == "__main__":
    unittest.main()
