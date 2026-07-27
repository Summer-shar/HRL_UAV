import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hrl_uav.agents.sac_agent import Memory
from hrl_uav.hrl_wrappers import HRL_Wrapper
from train_centralized_macro import compose_smdp_boundary_transition
from train_centralized_macro import store_sac_replay_transition


def make_dynamic_env():
    env = HRL_Wrapper(
        num_uavs=3,
        num_users=39,
        arrival_dt=0.5,
        resource_frame_dt=0.5,
        deployment_dt=1.0,
        users_static=False,
        gm_mean_speed_mps=0.5,
        gm_mean_heading_rad=np.pi / 4.0,
        gm_speed_memory=0.5,
        gm_heading_memory=0.5,
        gm_speed_noise_std=0.1,
        gm_heading_noise_std=0.1,
        cpu_alloc_mode="priority_demand",
        zone_loads=[12.5, 5.5, 3.5],
        macro_action_mode="delta",
        macro_reward_mode="strict_global",
        strict_hierarchy_validation=True,
        pause_service_during_flight=True,
        objective_mode="constrained_delay",
        power_budget_w=830.0,
    )
    env.env.F_max_array[:] = 10e9
    env.macro_shaping_lambda = 0.0
    return env


class ResourceFrameProtocolTest(unittest.TestCase):
    def test_macro_contains_two_deployment_and_eighteen_service_frames(self):
        env = make_dynamic_env()
        env.reset(seed=811)
        users_before = env.env.user_coords.copy()
        env.macro_step(np.zeros((env.N, 2), dtype=np.float32))
        action = np.zeros(env.micro_action_space.shape[0], dtype=np.float32)

        deployment = [env.step_resource_frame(action) for _ in range(2)]
        self.assertTrue(all(frame["phase"] == "deployment" for frame in deployment))
        self.assertEqual(env.env.resource_frame_prepare_count, 0)
        self.assertEqual(env.env.mobility_update_count, 0)
        np.testing.assert_allclose(env.env.user_coords, users_before)

        service = []
        for _ in range(18):
            env.prepare_resource_frame()
            frame = env.step_resource_frame(action)
            service.append(frame)
            self.assertEqual(frame["substeps"], 5)
            for info in frame["infos"]:
                np.testing.assert_allclose(
                    info["split_ratios_high"],
                    frame["infos"][0]["split_ratios_high"],
                )

        self.assertTrue(all(frame["phase"] == "service" for frame in service))
        self.assertEqual(env.env.global_micro_step, 100)
        self.assertEqual(env.env.resource_frame_prepare_count, 18)
        self.assertEqual(env.env.mobility_update_count, 18)
        self.assertFalse(np.allclose(env.env.user_coords, users_before))
        offsets = env.env.user_coords - env.env.anchor_points[env.env.user_allocations]
        self.assertTrue(
            np.all(np.linalg.norm(offsets, axis=1) <= env.env.user_zone_radius + 1e-6)
        )
        env.get_macro_reward()

    def test_boundary_reward_and_discount_follow_three_frame_smdp(self):
        service = {
            "duration_s": 0.5,
            "reward_sum": -7.0,
            "primal_reward_sum": -5.0,
            "energy_constraint_ratio_sum": 2.5,
        }
        deployment = [
            {"duration_s": 0.5, "holding_reward_sum": -2.0},
            {"duration_s": 0.5, "holding_reward_sum": -3.0},
        ]
        gamma = 0.99
        transition = compose_smdp_boundary_transition(
            service,
            deployment,
            gamma=gamma,
            objective_mode="constrained_delay",
        )

        self.assertAlmostEqual(
            transition["reward"],
            -5.0 + gamma * -2.0 + gamma**2 * -3.0,
        )
        self.assertAlmostEqual(transition["bootstrap_discount"], gamma**3)
        self.assertEqual(transition["discount_exponent"], 3)
        self.assertAlmostEqual(transition["duration_s"], 1.5)
        self.assertAlmostEqual(transition["energy_constraint_ratio"], 2.5)
        self.assertAlmostEqual(transition["service_reward"], -5.0)
        self.assertAlmostEqual(
            transition["deployment_holding_reward_1"],
            -2.0,
        )
        self.assertAlmostEqual(
            transition["deployment_holding_reward_2"],
            -3.0,
        )

    def test_replay_stores_energy_ratio_and_per_transition_discount(self):
        state = np.array([1.0, 2.0], dtype=np.float32)
        action = np.array([0.25], dtype=np.float32)
        next_state = np.array([3.0, 4.0], dtype=np.float32)
        memory = Memory(capacity=4, dims=2 + 1 + 1 + 2 + 2)
        store_sac_replay_transition(
            memory,
            state,
            action,
            reward=-6.0,
            next_state=next_state,
            objective_mode="constrained_delay",
            bootstrap_discount=0.99**3,
            energy_constraint_ratio=1.25,
        )

        row = memory.mem[0]
        np.testing.assert_allclose(row[:2], state)
        self.assertAlmostEqual(row[2], action[0])
        self.assertAlmostEqual(row[3], -6.0)
        self.assertAlmostEqual(row[4], 1.25)
        self.assertAlmostEqual(row[5], 0.99**3, places=6)
        np.testing.assert_allclose(row[-2:], next_state)


if __name__ == "__main__":
    unittest.main()
