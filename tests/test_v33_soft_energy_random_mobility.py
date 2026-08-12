import json
import unittest
from pathlib import Path

import numpy as np

from hrl_uav.env import UAV_MEC_Env
from hrl_uav.hrl_wrappers import HRL_Wrapper
from scripts.train_centralized_macro import zero_macro_policy


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "tcom_v33_soft_energy_random_mobility.json"
V34_CONFIG_PATH = ROOT / "configs" / "tcom_v34_cooperative_mobility.json"
V35_CONFIG_PATH = ROOT / "configs" / "tcom_v35_reference_like_mobility.json"


class V33SoftEnergyRandomMobilityTests(unittest.TestCase):
    def test_independent_headings_are_reproducible_and_not_shared(self):
        kwargs = dict(
            num_uavs=3,
            num_users=39,
            gm_heading_mode="independent_episode_mean",
            user_zone_radius=0.0,
        )
        first = UAV_MEC_Env(**kwargs)
        second = UAV_MEC_Env(**kwargs)
        first.reset(seed=123)
        second.reset(seed=123)

        np.testing.assert_allclose(
            first.user_mean_headings,
            second.user_mean_headings,
        )
        self.assertGreater(float(np.std(first.user_mean_headings)), 1.0)
        self.assertGreater(
            np.unique(np.round(first.user_mean_headings, decimals=6)).size,
            30,
        )

    def test_shared_heading_mode_preserves_historical_initialization(self):
        env = UAV_MEC_Env(
            num_uavs=3,
            num_users=39,
            gm_mean_heading_rad=np.pi / 4.0,
            gm_heading_mode="shared_mean",
        )
        env.reset(seed=7)
        np.testing.assert_allclose(env.user_headings, np.pi / 4.0)

    def test_weighted_objective_weights_are_normalized(self):
        env = HRL_Wrapper(
            objective_mode="weighted",
            delay_objective_weight=9.0,
            energy_objective_weight=1.0,
        )
        self.assertAlmostEqual(env.J_w_delay, 0.9)
        self.assertAlmostEqual(env.J_w_energy, 0.1)

    def test_v33_configuration_removes_hard_power_control(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["objective_mode"], "weighted")
        self.assertEqual(config["cpu_alloc_mode"], "priority_demand")
        self.assertEqual(config["gm_heading_mode"], "independent_episode_mean")
        self.assertEqual(config["micro_pretrain_macro_policy"], "zero")
        self.assertEqual(config["micro_pretrain_checkpoint"], "final")
        self.assertTrue(config["resume_micro_replay"])
        self.assertTrue(config["resume_micro_optimizer_in_joint"])
        self.assertAlmostEqual(config["load_scale"], 1.0)
        self.assertEqual(config["power_budget_w"], 0.0)
        self.assertEqual(config["energy_dual_lr"], 0.0)
        self.assertFalse(config["system_dual_state"])
        self.assertAlmostEqual(config["delay_objective_weight"], 0.9)
        self.assertAlmostEqual(config["energy_objective_weight"], 0.1)

    def test_zero_macro_policy_returns_joint_zero_residual(self):
        class DummyEnv:
            N = 3

        action = zero_macro_policy(DummyEnv(), None, True)
        self.assertEqual(action.shape, (3, 2))
        np.testing.assert_array_equal(action, np.zeros((3, 2), dtype=np.float32))

    def test_reference_point_group_motion_is_curved_and_reproducible(self):
        kwargs = dict(
            num_uavs=3,
            num_users=39,
            gm_heading_mode="reference_point_group",
            user_zone_radius=0.0,
            rpgm_group_speed_mps=0.35,
            rpgm_group_turn_std_rad_sqrt_s=0.12,
        )
        first = UAV_MEC_Env(**kwargs)
        second = UAV_MEC_Env(**kwargs)
        first.reset(seed=321)
        second.reset(seed=321)
        trajectory = [first.user_coords.copy()]
        centers = [first.mobility_group_centers.copy()]
        for _ in range(600):
            for env in (first, second):
                env._update_gauss_markov_mobility()
                for _ in range(env.resource_frame_steps):
                    env._integrate_user_positions()
            trajectory.append(first.user_coords.copy())
            centers.append(first.mobility_group_centers.copy())

        np.testing.assert_allclose(first.user_coords, second.user_coords)
        np.testing.assert_allclose(
            first.mobility_group_centers,
            second.mobility_group_centers,
        )
        trajectory = np.asarray(trajectory)
        steps = np.linalg.norm(np.diff(trajectory, axis=0), axis=2)
        path_length = np.sum(steps, axis=0)
        net_displacement = np.linalg.norm(
            trajectory[-1] - trajectory[0], axis=1
        )
        straightness = net_displacement / np.maximum(path_length, 1e-12)
        self.assertLess(float(np.mean(straightness)), 0.9)
        centers = np.asarray(centers)
        center_displacement = np.linalg.norm(centers[-1] - centers[0], axis=1)
        self.assertGreater(float(np.mean(center_displacement)), 30.0)
        self.assertGreater(float(np.std(first.user_offsets)), 5.0)

    def test_v34_configuration_uses_group_mobility_and_soft_energy(self):
        config = json.loads(V34_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["gm_heading_mode"], "reference_point_group")
        self.assertEqual(config["objective_mode"], "weighted")
        self.assertEqual(config["power_budget_w"], 0.0)
        self.assertEqual(config["energy_dual_lr"], 0.0)
        self.assertEqual(config["macro_action_mode"], "centroid_waypoint")
        self.assertGreater(config["rpgm_group_speed_mps"], 0.0)
        self.assertGreater(config["rpgm_group_turn_std_rad_sqrt_s"], 0.0)

    def test_v35_reduces_group_translation_and_increases_local_wandering(self):
        v34 = json.loads(V34_CONFIG_PATH.read_text(encoding="utf-8"))
        v35 = json.loads(V35_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(v35["gm_heading_mode"], "reference_point_group")
        self.assertEqual(v35["objective_mode"], v34["objective_mode"])
        self.assertEqual(v35["load_scale"], v34["load_scale"])
        self.assertLess(v35["rpgm_group_speed_mps"], v34["rpgm_group_speed_mps"])
        self.assertLess(
            v35["rpgm_group_speed_memory"],
            v34["rpgm_group_speed_memory"],
        )
        self.assertGreater(
            v35["rpgm_group_turn_std_rad_sqrt_s"],
            v34["rpgm_group_turn_std_rad_sqrt_s"],
        )
        self.assertGreater(
            v35["rpgm_local_velocity_std_mps"],
            v34["rpgm_local_velocity_std_mps"],
        )


if __name__ == "__main__":
    unittest.main()
