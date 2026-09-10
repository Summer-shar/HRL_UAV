import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from hrl_uav.experiment import smoke_config
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.ppo_frozen import FrozenSACUpdates, assert_sac_frozen, sac_learning_state


ROOT = Path(__file__).resolve().parents[1]


class FrozenSACPPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        base = json.loads((ROOT / "configs/sac_only_wireless6_pilot.json").read_text())
        self.config = smoke_config(base)
        self.config["training"].update(joint_windows=2, replay_capacity=64, batch_size=8, replay_warmup=8)
        env = QueueMDP(self.config)
        env.reset(1, 3)
        self.agent = DirectHRL.from_mdp(env, self.config)
        self.agent.set_stage("joint")
        self.proxy = FrozenSACUpdates(self.agent)
        self.env = env

    def test_blocked_update_does_not_change_sac_learning_state(self):
        state = self.env.micro_observation()
        action = self.agent.sac_actor.choose_action(state, deterministic=True)
        route = self.env.route(action)
        for _ in range(8):
            self.agent.store(state, action, route, -0.1, state, 0.99)
        before = sac_learning_state(self.agent, self.config)
        for _ in range(5):
            self.proxy.update_sac()
        checks = assert_sac_frozen(self.agent, self.config, before)
        self.assertTrue(all(checks.values()))
        self.assertEqual(self.proxy.blocked_update_calls, 5)

    def test_replay_collection_is_allowed_while_sac_parameters_remain_frozen(self):
        before = sac_learning_state(self.agent, self.config)
        counter = self.agent.replay.memory_counter
        state = self.env.micro_observation()
        action = np.zeros(self.env.micro_action_dim, dtype=np.float32)
        route = self.env.route(action)
        self.proxy.store(state, action, route, -0.2, state, 0.99)
        self.assertEqual(self.agent.replay.memory_counter, counter + 1)
        checks = assert_sac_frozen(self.agent, self.config, before)
        self.assertTrue(all(checks.values()))

    def test_ppo_update_can_change_macro_without_changing_sac(self):
        before_sac = sac_learning_state(self.agent, self.config)
        macro_before = copy.deepcopy(self.agent.macro_actor.state_dict())
        state = self.env.macro_observation()
        for _ in range(self.config["training"]["ppo_update_macros"]):
            action, logprob = self.proxy.macro_action(state, enabled=True, deterministic=False)
            self.agent.macro_rollout.append((state.copy(), action.copy(), logprob, -0.1, state.copy(), 1.0))
        self.proxy.update_ppo(0.5, force=True)
        checks = assert_sac_frozen(self.agent, self.config, before_sac)
        self.assertTrue(all(checks.values()))
        changed = any(not torch.equal(macro_before[k], v) for k, v in self.agent.macro_actor.state_dict().items())
        self.assertTrue(changed)
        self.assertGreater(self.agent.ppo_updates, 0)


if __name__ == "__main__":
    unittest.main()
