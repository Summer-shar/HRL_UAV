import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from hrl_uav.env import ParentTask
from hrl_uav.experiment import rollout, smoke_config
from hrl_uav.learning import DirectHRL
from hrl_uav.mdp import QueueMDP, compress
from hrl_uav.observation_ablation import arm_configs


ROOT = Path(__file__).resolve().parents[1]


class ReadyObservationTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads((ROOT/"configs/main.json").read_text())
        self.control_config, self.ready_config = arm_configs(self.original)
        self.mdp = QueueMDP(self.ready_config)
        self.mdp.reset(123, 10)

    def add_task(self, task_id, source, priority, size, ready=True):
        task = ParentTask(task_id, 0, source, "sensor", priority, 0, size,
                          size*1e9, remaining_upload_mbits=0 if ready else size,
                          upload_finish_time=0 if ready else None)
        self.mdp.env.parent_tasks[task_id] = task
        if ready:
            self.mdp.env.completed_upload_tasks_by_uav[source].append(task_id)
        return task

    def test_only_one_configuration_field_changes(self):
        restored = copy.deepcopy(self.ready_config)
        restored["observation"]["routing_ready"] = False
        self.assertEqual(restored, self.control_config)
        self.assertEqual(self.original["observation"], {"layout": "split_v1", "routing_ready": False})
        self.assertEqual(self.control_config["observation"]["layout"], "shared_v1")

    def test_original_features_preserved_and_dimensions_dynamic(self):
        state = self.mdp.observation()
        self.assertEqual(len(state), 116)
        self.assertEqual(self.mdp.feature_slices["routing_ready_work"], [104, 110])
        self.assertEqual(self.mdp.feature_slices["routing_ready_count"], [110, 116])
        self.mdp.observe_routing_ready = False
        np.testing.assert_array_equal(state[:104], self.mdp.observation())

    def test_real_task_bits_counts_and_priority_source_order(self):
        self.add_task(10000, 0, "high", 1.0)
        self.add_task(10001, 0, "high", 3.0)
        self.add_task(10002, 2, "low", 4.5)
        self.add_task(10003, 1, "low", 20, ready=False)
        work, counts = self.mdp.routing_ready_aggregates()
        expected = np.array([[4., 0, 0], [0, 0, 4.5]])
        np.testing.assert_array_equal(work, expected)
        np.testing.assert_array_equal(counts, [[2, 0, 0], [0, 0, 1]])
        state = self.mdp.observation()
        np.testing.assert_allclose(state[104:110], compress(expected, 10).ravel(), rtol=1e-6)
        np.testing.assert_allclose(state[110:], compress(counts, 10).ravel(), rtol=1e-6)

    def test_reading_never_consumes_tasks_changes_rng_or_advances_time(self):
        self.add_task(10000, 1, "low", 2)
        before = copy.deepcopy(self.mdp.env.completed_upload_tasks_by_uav)
        rng = np.random.get_state()
        time_before = self.mdp.env._current_time()
        first = self.mdp.observation()
        second = self.mdp.observation()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(before, self.mdp.env.completed_upload_tasks_by_uav)
        self.assertEqual(time_before, self.mdp.env._current_time())
        np.testing.assert_array_equal(rng[1], np.random.get_state()[1])
        self.assertEqual(rng[2:], np.random.get_state()[2:])

    def test_no_future_arrivals_and_features_do_not_clip(self):
        self.add_task(10000, 0, "low", 2, ready=False)
        self.mdp.env.A_t_low[0] = 1000
        np.testing.assert_array_equal(self.mdp.observation()[104:], np.zeros(12))
        task = self.add_task(10001, 2, "low", 10)
        values = []
        for size in (10, 100, 1000):
            task.size_mbits = size
            values.append(self.mdp.observation()[109])
        self.assertTrue(np.all(np.diff(values) > 0))
        before = self.mdp.observation().copy()
        self.mdp.schedule.phase_index_by_macro[5:] = 2
        np.testing.assert_array_equal(before, self.mdp.observation())

    def test_next_routing_consumes_list_not_observation(self):
        self.add_task(10000, 0, "high", 2)
        self.mdp.observation()
        self.assertEqual(self.mdp.routing_ready_aggregates()[1].sum(), 1)
        ready = self.mdp.env._take_completed_upload_tasks()
        self.mdp.env._route_arrived_parent_tasks(ready, np.eye(3), np.eye(3))
        self.assertEqual(self.mdp.routing_ready_aggregates()[1].sum(), 0)
        self.assertEqual(self.mdp.env.Q_loc_high[0], 2)

    def test_invalid_pending_stage_is_not_silently_encoded(self):
        task = self.add_task(10000, 0, "low", 1)
        task.remaining_upload_mbits = .5
        with self.assertRaises(ValueError):
            self.mdp.observation()

    def test_same_actions_have_identical_physics_and_rewards(self):
        other = copy.deepcopy(self.mdp)
        other.observe_routing_ready = False
        actions = np.random.default_rng(11).uniform(-1, 1, size=(40, 18))
        for k, action in enumerate(actions):
            if k % 20 == 0:
                for mdp in (self.mdp, other):
                    mdp.start_macro(np.zeros(6))
            for mdp in (self.mdp, other):
                mdp.observation()
            a, b = self.mdp.step_frame(action), other.step_frame(action)
            self.assertEqual(a["reward"], b["reward"])
            np.testing.assert_array_equal(a["route"], b["route"])
            np.testing.assert_array_equal(self.mdp.env.Q_loc_low, other.env.Q_loc_low)
            if k % 20 == 19:
                for mdp in (self.mdp, other):
                    mdp.advance_macro_clock()

    def test_116_state_network_and_replay_train_without_macro_updates(self):
        torch.set_num_threads(1)
        cfg = smoke_config(self.ready_config)
        agent = DirectHRL(116, 18, 6, cfg)
        agent.set_stage("pretrain")
        summary, _, _, _, _ = rollout(cfg, agent, 1, 2, macro_enabled=False, train=True, complete_tail=True)
        self.assertEqual(summary["final_pending_dropped"], 0)
        self.assertGreater(agent.sac_actor_updates, 0)
        self.assertEqual(agent.ppo_updates, 0)
        data = agent.replay.state_dict()["transitions"]
        self.assertEqual(data.shape[1], 2*116+2*18+2)
        for row in data:
            np.testing.assert_allclose(self.mdp.route(row[116:134]).ravel(), row[136:154], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
