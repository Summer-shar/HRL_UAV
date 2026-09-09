import json
from pathlib import Path
import unittest

import numpy as np
import torch

from hrl_uav.env import ParentTask
from hrl_uav.experiment import boundary_reward, isolated_rng, rollout, smoke_config
from hrl_uav.learning import DirectHRL, gae
from hrl_uav.mdp import QueueMDP, compress, holding_energy_cost

ROOT = Path(__file__).resolve().parents[1]


class CleanMDPTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/"configs/main.json").read_text())
        self.env = QueueMDP(self.config)
        self.state = self.env.reset(1, 92)

    def test_full_simplex_includes_local_and_single_peer(self):
        local = np.tile(2*np.eye(3)-1, (2,1,1))
        np.testing.assert_allclose(self.env.route(local), np.tile(np.eye(3),(2,1,1)))
        peer = np.full((2,3,3), -1.0)
        for source in range(3):
            peer[:,source,(source+1)%3] = 1
        result = self.env.route(peer)
        np.testing.assert_allclose(result.sum(-1),1)
        self.assertEqual(np.count_nonzero(result),6)

    def test_observation_does_not_saturate(self):
        features = []
        for queue in (10,100,300):
            self.env.env.Q_loc_low[0] = queue
            features.append(self.env.observation()[3])
        self.assertTrue(features[0] < features[1] < features[2])
        self.assertTrue(np.all(np.diff(compress([10,100,1000],10))>0))

    def test_reward_unclipped_and_priority_weighted(self):
        reward = self.config["reward"]
        costs = [holding_energy_cost(n, 0, reward) for n in (10,100,1000)]
        self.assertAlmostEqual(costs[2]/costs[1],10)
        self.assertAlmostEqual(costs[1]/costs[0],10)
        self.assertGreater(costs[2],5)

    def test_upload_parent_count_and_age_are_visible(self):
        p = ParentTask(9000,0,0,"sensor","low",-25.0,2.0,2e9,remaining_upload_mbits=2.0)
        self.env.env.parent_tasks[p.task_id] = p
        s = self.env.observation()
        slices = self.env.feature_slices
        self.assertGreater(s[slices["unfinished_tasks"][0]+3],0)
        self.assertGreater(s[slices["oldest_age"][0]+3],0)
        self.assertGreater(s[slices["upload_queue"][0]+3],0)

    def test_future_schedule_is_not_an_observation(self):
        before = self.env.observation()
        self.env.schedule.phase_index_by_macro[50:] = 2
        np.testing.assert_array_equal(before,self.env.observation())

    def test_boundary_discount_includes_deployment_and_never_reset(self):
        reward, gamma = boundary_reward(-1,[-2,-3],0.99)
        self.assertAlmostEqual(reward,-1-0.99*2-0.99**2*3)
        self.assertAlmostEqual(gamma,0.99**3)
        self.assertGreater(gamma,0)

    def test_gae_truncates_trace_but_bootstraps_continuing_value(self):
        advantage, returns = gae([1,100],[0,0],[2,0],[0,0],0.5,0.95)
        np.testing.assert_allclose(advantage,[2,100])
        np.testing.assert_allclose(returns,advantage)

    def test_evaluation_restores_rng(self):
        np.random.seed(123)
        before = np.random.get_state()
        state = torch.get_rng_state().clone()
        with isolated_rng():
            np.random.seed(555)
            np.random.random(10)
            torch.rand(10)
        self.assertEqual(np.random.get_state()[2],before[2])
        np.testing.assert_array_equal(np.random.get_state()[1],before[1])
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_sampler_rng_is_independent(self):
        config = smoke_config(self.config)
        agent = DirectHRL.from_mdp(self.env, config)
        for index in range(20):
            agent.store(self.state,np.zeros(18),self.env.route(np.zeros(18)),-1,self.state,0.99)
        np.random.seed(97)
        before = np.random.get_state()
        batch = agent.replay.sample(16)
        np.testing.assert_array_equal(before[1],np.random.get_state()[1])
        np.testing.assert_allclose(batch[:,-len(self.state):],np.tile(self.state,(16,1)))

    def test_short_run_updates_both_levels_and_matches_executed_route(self):
        config = smoke_config(self.config)
        agent = DirectHRL.from_mdp(self.env, config)
        agent.set_stage("joint")
        result, _, _, _, _ = rollout(config,agent,1,2,train=True,complete_tail=True)
        self.assertEqual(result["boundary_transitions"],2)
        self.assertEqual(result["final_pending_dropped"],0)
        self.assertGreater(agent.sac_actor_updates,0)
        self.assertGreater(agent.ppo_gradient_steps,0)
        n,a = len(self.state),18
        data = agent.replay.state_dict()["transitions"]
        for row in data:
            np.testing.assert_allclose(self.env.route(row[n:n+a]).ravel(),row[n+a+2:n+2*a+2],atol=1e-7)

    def test_no_queue_reset_at_300_seconds(self):
        # Physical local-only control keeps a heavy queue; a 300s log boundary
        # must not erase it. This tests continuity without involving a learner.
        local = np.tile(2*np.eye(3)-1,(2,1,1))
        for macro in range(31):
            self.env.start_macro(np.zeros(6))
            if macro == 30:
                for task_id, task in unfinished_before.items():
                    self.assertIs(self.env.env.parent_tasks.get(task_id), task)
            for _ in range(20):
                self.env.step_frame(local)
            self.env.advance_macro_clock()
            if macro == 29:
                unfinished_before = {task_id: task for task_id, task in self.env.env.parent_tasks.items()
                                     if task.finish_time is None}
                arrival_before = self.env.env.arrived_task_count
                self.assertGreater(self.env.metrics()["unfinished"],0)
                self.assertAlmostEqual(self.env.env._current_time(),300)
        self.assertGreater(self.env.env.arrived_task_count,arrival_before)
        self.assertAlmostEqual(self.env.env._current_time(),310)


if __name__ == "__main__":
    unittest.main()
