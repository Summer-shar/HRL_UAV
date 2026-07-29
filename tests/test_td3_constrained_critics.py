import copy
import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_centralized_td3 import MacroReplayBuffer
from train_centralized_td3 import TD3Agent
from train_centralized_td3 import build_checkpoint_payload
from train_centralized_td3 import macro_power_violation_penalty
from train_centralized_td3 import update_smoothed_energy_dual


def make_args(policy_delay=100, ema_decay=0.0):
    return SimpleNamespace(
        td3_hidden_dim=32,
        td3_actor_lr=3e-4,
        td3_critic_lr=1e-3,
        td3_gamma=0.99,
        td3_tau=0.005,
        td3_policy_noise=0.02,
        td3_noise_clip=0.05,
        td3_policy_delay=policy_delay,
        td3_actor_ema_decay=ema_decay,
        td3_reward_scale=0.01,
        objective_mode="constrained_delay",
        micro_solver="local",
    )


def clone_agent(source, args):
    target = TD3Agent(4, 6, 3, args, torch.device("cpu"))
    for name in (
        "actor",
        "actor_target",
        "critic",
        "critic_target",
        "energy_critic",
        "energy_critic_target",
    ):
        getattr(target, name).load_state_dict(
            copy.deepcopy(getattr(source, name).state_dict())
        )
    return target


class TD3ConstrainedCriticTests(unittest.TestCase):
    def make_buffer(self):
        buffer = MacroReplayBuffer(4, 6, 64)
        rng = np.random.default_rng(7)
        for _ in range(48):
            state = rng.normal(size=4).astype(np.float32)
            action = rng.uniform(-0.5, 0.5, size=6).astype(np.float32)
            next_state = rng.normal(size=4).astype(np.float32)
            buffer.add(
                state,
                action,
                reward=float(rng.normal(-0.1, 0.01)),
                next_state=next_state,
                done=False,
                energy_ratio=float(rng.uniform(0.8, 1.2)),
            )
        return buffer

    def test_critic_targets_do_not_depend_on_dual(self):
        args = make_args(policy_delay=100)
        torch.manual_seed(11)
        first = TD3Agent(4, 6, 3, args, torch.device("cpu"))
        second = clone_agent(first, args)
        buffer = self.make_buffer()

        np.random.seed(19)
        torch.manual_seed(23)
        result_zero = first.update(buffer, 16, energy_lagrange_multiplier=0.0)
        np.random.seed(19)
        torch.manual_seed(23)
        result_large = second.update(buffer, 16, energy_lagrange_multiplier=5.0)

        self.assertTrue(np.isnan(result_zero[0]))
        self.assertTrue(np.isnan(result_large[0]))
        self.assertAlmostEqual(result_zero[1], result_large[1], places=7)
        self.assertAlmostEqual(result_zero[2], result_large[2], places=7)

    def test_dual_update_uses_trailing_power_mean(self):
        env = SimpleNamespace(energy_lagrange_multiplier=0.0)
        args = SimpleNamespace(
            objective_mode="constrained_delay",
            power_budget_w=830.0,
            power_constraint_margin_w=0.0,
            energy_dual_lr=0.2,
            energy_dual_max=20.0,
        )
        history = deque(maxlen=5)
        observed = []
        for power in (840.0, 820.0, 830.0, 810.0, 850.0, 860.0):
            _, _, _, smoothed = update_smoothed_energy_dual(
                env,
                args,
                history,
                power,
            )
            observed.append(smoothed)

        self.assertAlmostEqual(observed[0], 840.0)
        self.assertAlmostEqual(observed[4], 830.0)
        self.assertAlmostEqual(observed[5], 834.0)
        self.assertEqual(list(history), [820.0, 830.0, 810.0, 850.0, 860.0])

    def test_soft_power_penalty_is_zero_inside_budget(self):
        self.assertEqual(macro_power_violation_penalty(0.98, 10.0), 0.0)
        self.assertAlmostEqual(
            macro_power_violation_penalty(1.02, 10.0),
            0.2,
        )

    def test_actor_ema_initializes_equal_and_updates_by_fixed_decay(self):
        args = make_args(ema_decay=0.5)
        agent = TD3Agent(4, 6, 3, args, torch.device("cpu"))
        self.assertIs(agent.evaluation_actor, agent.actor_ema)

        initial = [parameter.detach().clone() for parameter in agent.actor.parameters()]
        for raw_parameter, ema_parameter in zip(
            agent.actor.parameters(), agent.actor_ema.parameters()
        ):
            self.assertTrue(torch.equal(raw_parameter, ema_parameter))

        with torch.no_grad():
            for parameter in agent.actor.parameters():
                parameter.add_(2.0)
        agent._update_actor_ema()

        for before, ema_parameter in zip(initial, agent.actor_ema.parameters()):
            self.assertTrue(torch.allclose(ema_parameter, before + 1.0))

    def test_checkpoint_compatibility_actor_points_to_ema(self):
        args = make_args(ema_decay=0.5)
        agent = TD3Agent(4, 6, 3, args, torch.device("cpu"))
        with torch.no_grad():
            for parameter in agent.actor.parameters():
                parameter.add_(2.0)
        agent._update_actor_ema()

        payload = build_checkpoint_payload(args, agent, None, None, None)
        self.assertEqual(payload["macro_actor_policy"], "ema")
        self.assertEqual(payload["td3_actor_ema_decay"], 0.5)
        self.assertIn("macro_actor_ema", payload)
        for name in payload["macro_actor"]:
            self.assertTrue(
                torch.equal(payload["macro_actor"][name], payload["macro_actor_ema"][name])
            )
        self.assertTrue(
            any(
                not torch.equal(
                    payload["macro_actor"][name], payload["macro_actor_raw"][name]
                )
                for name in payload["macro_actor"]
            )
        )


if __name__ == "__main__":
    unittest.main()
