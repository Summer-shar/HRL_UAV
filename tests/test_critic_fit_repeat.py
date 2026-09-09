"""Repeated Critic fitting must not speed up other clocks or consume RNG."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from hrl_uav.learning import DirectHRL, tensor
from hrl_uav.mdp import QueueMDP
from hrl_uav.resumption import capture_rng, restore_agent, restore_rng, state_equal, validate_resume_config


ROOT = Path(__file__).resolve().parents[1]
KEY = "sac_critic_steps_per_update"


def legacy_update(agent):
    """The pre-change one-fit update, retained here as a numerical reference."""
    batch = agent.replay.sample(agent.cfg["batch_size"])
    n, a = agent.state_dim, agent.action_dim
    state, action = tensor(batch[:, :n]), tensor(batch[:, n:n+a])
    reward = tensor(batch[:, n+a:n+a+1])
    discount = tensor(batch[:, n+a+1:n+a+2])
    next_state = tensor(batch[:, -n:])
    with torch.no_grad():
        next_action, logprob = agent.sac_actor.evaluate(next_state)
        q1, q2 = agent.sac_critic.target_critic_v(next_state, next_action)
        target = reward + discount * (torch.minimum(q1, q2) - agent.entropy.log_alpha.exp() * logprob)
    q1, q2 = agent.sac_critic.get_v(state, action)
    loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite reference loss")
    agent.sac_critic.learn(q1, q2, target, max_grad_norm=agent.cfg["grad_clip"])
    agent.sac_updates += 1
    if (agent.sac_updates-agent.stage_initial_updates > agent.sac_actor_warmup()
            and agent.sac_updates % agent.cfg["sac_policy_delay"] == 0):
        for parameter in agent.sac_critic.critic_v.parameters():
            parameter.requires_grad_(False)
        sampled, logprob = agent.sac_actor.evaluate(state)
        qa, qb = agent.sac_critic.get_v(state, sampled)
        loss = (agent.entropy.log_alpha.exp().detach()*logprob-torch.minimum(qa, qb)).mean()
        agent.sac_actor.learn(loss, max_grad_norm=agent.cfg["grad_clip"])
        for parameter in agent.sac_critic.critic_v.parameters():
            parameter.requires_grad_(True)
        agent.entropy.learn(agent.entropy.temperature_loss(logprob))
        agent.entropy.alpha = agent.entropy.log_alpha.exp()
        agent.sac_actor_updates += 1
    with torch.no_grad():
        for target_parameter, parameter in zip(agent.sac_critic.target_critic_v.parameters(), agent.sac_critic.critic_v.parameters()):
            target_parameter.lerp_(parameter, agent.cfg["tau"])


class CriticFitRepeatTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9312)
        self.config = json.loads((ROOT/"configs/hrl_ppo_only_unlock.json").read_text())
        self.config["training"].update(batch_size=8, replay_capacity=64, replay_warmup=8,
                                       sac_actor_warmup_updates=0)
        mdp = QueueMDP(self.config)
        state = mdp.reset(33500, 90)
        self.agent = DirectHRL.from_mdp(mdp, self.config)
        rng = np.random.default_rng(101)
        for i in range(24):
            scores = rng.uniform(-.3, .3, 18).astype(np.float32)
            self.agent.store(state, scores, mdp.route(scores), -float(i+1), state+.001*i, .99)

    def clone_agent(self, original):
        saved = copy.deepcopy(original.checkpoint(self.config))
        saved["run_state"] = {"rng": capture_rng()}
        mdp = QueueMDP(self.config)
        mdp.reset(33500, 90)
        agent = DirectHRL.from_mdp(mdp, self.config)
        restore_agent(agent, saved, {"replay_counter": original.replay.memory_counter})
        agent.critic_steps_per_update = original.critic_steps_per_update
        return agent

    def test_default_one_fit_is_bitwise_equal_to_legacy_updates(self):
        reference = self.clone_agent(self.agent)
        rng = capture_rng()
        for _ in range(4):
            self.agent.update_sac()
        after = capture_rng()
        restore_rng(rng)
        for _ in range(4):
            legacy_update(reference)
        left, right = self.agent.checkpoint(self.config), reference.checkpoint(self.config)
        for key in ("sac_actor", "sac_critic", "sac_target", "log_alpha", "optimizers", "replay", "macro_actor", "macro_critic"):
            self.assertTrue(state_equal(left[key], right[key]), key)
        self.assertTrue(state_equal(after, capture_rng()))
        self.assertEqual(self.agent.sac_critic_gradient_steps, 4)
        self.assertEqual(self.agent.sac_target_updates, 4)
        self.assertEqual(self.agent.sac_actor_updates, 2)

    def test_double_fit_keeps_target_actor_entropy_and_rng_clocks(self):
        control = self.clone_agent(self.agent)
        treatment = self.clone_agent(self.agent)
        treatment.critic_steps_per_update = 2
        rng = capture_rng()
        for _ in range(4):
            control.update_sac()
        after = capture_rng()
        restore_rng(rng)
        for _ in range(4):
            previous = copy.deepcopy(treatment.sac_critic.target_critic_v.state_dict())
            treatment.update_sac()
            current = treatment.sac_critic.critic_v.state_dict()
            for key, value in treatment.sac_critic.target_critic_v.state_dict().items():
                expected = previous[key].lerp(current[key], treatment.cfg["tau"])
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
        self.assertTrue(state_equal(after, capture_rng()))
        self.assertTrue(state_equal(control.replay.state_dict(), treatment.replay.state_dict()))
        self.assertEqual((control.sac_updates, treatment.sac_updates), (4, 4))
        self.assertEqual((control.sac_actor_updates, treatment.sac_actor_updates), (2, 2))
        self.assertEqual((control.sac_target_updates, treatment.sac_target_updates), (4, 4))
        self.assertEqual((control.sac_critic_gradient_steps, treatment.sac_critic_gradient_steps), (4, 8))
        for agent in (control, treatment):
            steps = {int(state["step"].item()) for state in agent.entropy.optimizer.state.values()}
            self.assertEqual(steps, {2})
            self.assertEqual(agent.ppo_updates, 0)
        self.assertFalse(state_equal(control.sac_critic.critic_v.state_dict(), treatment.sac_critic.critic_v.state_dict()))

    def test_resume_requires_explicit_narrow_override(self):
        target = copy.deepcopy(self.config)
        target["training"]["joint_windows"] += 200
        target["training"][KEY] = 1
        self.assertEqual(validate_resume_config(self.config, target, 600, "joint"), {})
        target["training"][KEY] = 2
        with self.assertRaises(ValueError):
            validate_resume_config(self.config, target, 600, "joint")
        changes = validate_resume_config(self.config, target, 600, "joint", allowed_training_overrides=(KEY,))
        self.assertEqual(changes, {KEY: {"from": 1, "to": 2}})
        target["reward"]["delay_weight"] = .8
        with self.assertRaises(ValueError):
            validate_resume_config(self.config, target, 600, "joint", allowed_training_overrides=(KEY,))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_resume_config(self.config, self.config, 600, "joint", allowed_training_overrides=("pretrain_lr",))

    def test_new_and_legacy_gradient_counters_restore(self):
        for _ in range(2):
            self.agent.update_sac()
        self.agent.critic_steps_per_update = 2
        self.agent.update_sac()
        self.assertEqual(self.agent.sac_critic_gradient_steps, 4)
        saved = copy.deepcopy(self.agent.checkpoint(self.config))
        saved["run_state"] = {"rng": capture_rng()}
        restored = self.clone_agent(self.agent)
        restored.sac_updates = restored.sac_critic_gradient_steps = restored.sac_target_updates = 0
        audit = restore_agent(restored, saved, {"replay_counter": self.agent.replay.memory_counter})
        self.assertTrue(all(audit["exact_restore_checks"].values()))
        self.assertEqual((restored.sac_updates, restored.sac_critic_gradient_steps, restored.sac_target_updates), (3, 4, 3))
        del saved["updates"]["sac_critic_gradients"]
        del saved["updates"]["sac_target"]
        restore_agent(restored, saved, {"replay_counter": self.agent.replay.memory_counter})
        self.assertEqual((restored.sac_critic_gradient_steps, restored.sac_target_updates), (3, 3))

    def test_fit_count_must_be_positive_integer(self):
        for value in (0, -1, 1.5, True):
            config = copy.deepcopy(self.config)
            config["training"][KEY] = value
            mdp = QueueMDP(config)
            mdp.reset(1, 90)
            with self.assertRaises(ValueError):
                DirectHRL.from_mdp(mdp, config)


if __name__ == "__main__":
    unittest.main()
