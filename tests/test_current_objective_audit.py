"""The audit must preserve objectives, actions, CRN and frozen state."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from hrl_uav.experiment import snapshot
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP
from hrl_uav.routing import project_rows_to_simplex
from scripts.audit_current_objective import OneWayRouting, branch, one_way_route


ROOT = Path(__file__).resolve().parents[1]


class CurrentObjectiveAuditTests(unittest.TestCase):
    def test_rule_balances_expected_work_without_reverse_exports(self):
        loads = np.array([14.375, 6.325, 4.025])
        route = one_way_route(loads)
        np.testing.assert_allclose(route[0, 0], [.573333333333, .133333333333, .293333333333])
        np.testing.assert_array_equal(route[:, 1:, :], np.stack([np.eye(3)[1:], np.eye(3)[1:]]))
        np.testing.assert_allclose(loads @ route[0], loads.mean())
        np.testing.assert_allclose(route.sum(-1), 1)
        np.testing.assert_allclose(project_rows_to_simplex(route), route, atol=1e-12)
        for order in ([2, 0, 1], [1, 2, 0]):
            expected = route[:, order][:, :, order]
            np.testing.assert_allclose(one_way_route(loads[order]), expected)

    def test_rule_rejects_unsupported_loads_and_handles_equal_loads(self):
        np.testing.assert_array_equal(one_way_route([0, 0, 0]), np.stack([np.eye(3)]*2))
        np.testing.assert_array_equal(one_way_route([2, 2, 2]), np.stack([np.eye(3)]*2))
        for loads in ([2, 2, 0], [-1, 1, 2], [np.nan, 1, 2], [1, 2]):
            with self.assertRaises(ValueError):
                one_way_route(loads)

    def test_observation_only_rule_and_copied_branches(self):
        torch.set_num_threads(1)
        config = json.loads((ROOT/"configs/hrl_ppo_only_unlock.json").read_text())
        config["training"]["replay_capacity"] = 100
        mdp = QueueMDP(config)
        mdp.reset(33500, 90)
        agent = DirectHRL.from_mdp(mdp, config)
        rule = OneWayRouting(mdp.observation_spec()["micro"]["slices"]["current_load"])
        np.testing.assert_allclose(mdp.route(rule.choose_action(mdp.micro_observation())),
                                   one_way_route(mdp.env.zone_expected_offered_load_mbps), atol=1e-6)
        mdp.start_macro(np.zeros(6))
        for _ in range(mdp.env.deployment_resource_frames):
            mdp.step_frame(np.zeros(18))
        mdp.prepare_service()
        before = mdp.micro_observation().copy()
        cfg_before = copy.deepcopy(mdp.config)
        weights = snapshot(agent.sac_actor.action_net)
        actor = branch(mdp, agent, rule, "actor", horizons=(1, 4))
        uniform = branch(mdp, agent, rule, "sustained_uniform", horizons=(1, 4))
        self.assertEqual(actor, uniform)
        one_way = branch(mdp, agent, rule, "pulse_one_way", horizons=(1, 4))
        for h in actor:
            self.assertEqual(actor[h]["exogenous_sha256"], one_way[h]["exogenous_sha256"])
        self.assertEqual(one_way, branch(mdp, agent, rule, "pulse_one_way", horizons=(1, 4)))
        np.testing.assert_array_equal(before, mdp.micro_observation())
        self.assertEqual(cfg_before, mdp.config)
        for key, value in weights.items():
            torch.testing.assert_close(value, agent.sac_actor.action_net.state_dict()[key].detach().cpu(), rtol=0, atol=0)
        self.assertEqual(agent.sac_updates, 0)
        self.assertEqual(agent.replay.memory_counter, 0)


if __name__ == "__main__":
    unittest.main()
