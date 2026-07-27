import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hrl_uav.env import UAV_MEC_Env
from hrl_uav.hrl_wrappers import HRL_Wrapper
from train_centralized_td3 import MacroReplayBuffer
from train_centralized_td3 import TD3Agent
from train_centralized_td3 import update_smoothed_per_uav_energy_duals
from train_centralized_macro import load_checkpoint_energy_dual
from compare_sca import _sca_cache_key


def make_research_env(load_scale=1.15):
    env = HRL_Wrapper(
        num_uavs=3,
        num_users=39,
        arrival_dt=0.5,
        resource_frame_dt=0.5,
        deployment_dt=1.0,
        zone_loads=[12.5, 5.5, 3.5],
        load_scale=load_scale,
        cpu_alloc_mode="priced_maxweight",
        cpu_power_cap_w=0.0,
        pause_service_during_flight=True,
        arrivals_during_deployment=True,
        users_static=False,
        objective_mode="constrained_delay",
        power_budget_w=840.0,
        power_constraint_scope="per_uav",
        uav_power_budgets_w=[280.0, 280.0, 280.0],
        energy_lagrange_multipliers=[0.0, 0.0, 0.0],
        strict_hierarchy_validation=True,
    )
    env.env.F_max_array[:] = 10e9
    env.macro_shaping_lambda = 0.0
    return env


class LongTermResearchProtocolTests(unittest.TestCase):
    def test_zero_zone_radius_removes_circle_but_keeps_map_boundary(self):
        env = UAV_MEC_Env(
            num_uavs=3,
            num_users=39,
            users_static=False,
            user_zone_radius=0.0,
        )
        env.reset(seed=293)
        self.assertGreater(
            float(np.max(np.linalg.norm(env.user_offsets, axis=1))),
            0.0,
        )

        user_idx = 0
        anchor = env.anchor_points[env.user_allocations[user_idx]]
        env.user_coords[user_idx] = anchor + np.asarray([80.0, 0.0])
        env.user_offsets[user_idx] = env.user_coords[user_idx] - anchor
        env.user_velocities[user_idx] = np.asarray([1.0, 0.0])
        env._integrate_user_positions()
        self.assertGreater(
            float(np.linalg.norm(env.user_offsets[user_idx])),
            80.0,
        )

        boundary_idx = 1
        env.user_coords[boundary_idx] = np.asarray([999.99, 500.0])
        boundary_anchor = env.anchor_points[
            env.user_allocations[boundary_idx]
        ]
        env.user_offsets[boundary_idx] = (
            env.user_coords[boundary_idx] - boundary_anchor
        )
        env.user_velocities[boundary_idx] = np.asarray([1.0, 0.0])
        env._integrate_user_positions()
        self.assertAlmostEqual(env.user_coords[boundary_idx, 0], 1000.0)
        self.assertLess(env.user_velocities[boundary_idx, 0], 0.0)

    def test_main_load_is_locally_overloaded_but_globally_feasible(self):
        env = make_research_env(load_scale=1.15)
        audit = env.env.get_compute_capacity_audit()

        np.testing.assert_allclose(
            audit["arrival_mbits_per_macro"],
            [143.75, 63.25, 40.25],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            audit["compute_capacity_mbits_per_macro"],
            [90.0, 90.0, 90.0],
            rtol=1e-6,
        )
        np.testing.assert_array_equal(
            audit["locally_overloaded"],
            [True, False, False],
        )
        self.assertTrue(audit["globally_feasible"])
        self.assertAlmostEqual(audit["global_margin_mbits"], 22.75, places=5)

    def test_load_1_3_is_flagged_as_globally_infeasible(self):
        env = make_research_env(load_scale=1.3)
        audit = env.env.get_compute_capacity_audit()

        self.assertFalse(audit["globally_feasible"])
        self.assertAlmostEqual(audit["global_margin_mbits"], -9.5, places=5)

    def test_deployment_has_arrivals_and_mobility_but_no_service(self):
        env = make_research_env()
        env.reset(seed=301)
        # Force deterministic non-empty Poisson traffic after reset.
        env.env.user_lambda_high.fill(0.0)
        env.env.user_lambda_low.fill(0.0)
        env.env.user_lambda_high[0] = 100.0
        env.env.user_task_size_high_mbits[0] = 1.0
        users_before = env.env.user_coords.copy()

        env.macro_step(np.zeros((env.N, 2), dtype=np.float32))
        action = np.zeros(env.micro_action_space.shape[0], dtype=np.float32)
        frames = [env.step_resource_frame(action) for _ in range(2)]

        self.assertTrue(all(frame["phase"] == "deployment" for frame in frames))
        reported_arrivals = sum(
            float(np.sum(info["arrival_total_per_uav"]))
            for frame in frames
            for info in frame["infos"]
        )
        self.assertGreater(reported_arrivals, 0.0)
        self.assertEqual(env.env.deployment_arrival_prepare_count, 2)
        self.assertEqual(env.env.resource_frame_prepare_count, 0)
        self.assertEqual(env.env.mobility_update_count, 2)
        self.assertFalse(np.array_equal(env.env.user_coords, users_before))
        self.assertEqual(env.env.pending_upload_jobs, [])
        self.assertEqual(float(np.sum(env.env.Q_loc_high)), 0.0)
        self.assertEqual(float(np.sum(env.env.Q_tx_high)), 0.0)
        self.assertEqual(
            env.env.arrived_task_count,
            env.env.get_alive_task_wait_info()["task_alive_count"],
        )

    def test_priced_cpu_ignores_legacy_cap_and_uses_total_cubic_power(self):
        env = UAV_MEC_Env(
            cpu_alloc_mode="priced_maxweight",
            cpu_energy_prices=0.0,
            cpu_power_cap_w=1.0,
            users_static=True,
        )
        env.F_max_array[:] = 10e9
        env.Q_loc_high[:] = 1.0
        env.Q_loc_low[:] = 1.0

        _, _, f_total, energy_model = env._allocate_cpu_frequencies()
        power_w = float(env.kappa * np.sum(f_total**3))

        self.assertEqual(energy_model, "total_cubic")
        self.assertFalse(env.last_cpu_power_cap_active)
        self.assertGreater(power_w, 1.0)
        np.testing.assert_allclose(f_total, np.full(3, 10e9), rtol=1e-9)

    def test_cpu_frequency_decreases_monotonically_with_energy_price(self):
        env = UAV_MEC_Env(
            cpu_alloc_mode="priced_maxweight",
            cpu_energy_prices=0.1,
            users_static=True,
        )
        env.F_max_array[:] = 10e9
        env.Q_loc_high[:] = 0.5
        env.Q_loc_low[:] = 1.0
        low_price_total = env._allocate_cpu_frequencies()[2]

        env.set_cpu_energy_prices([10.0, 10.0, 10.0])
        high_price_total = env._allocate_cpu_frequencies()[2]

        self.assertTrue(np.all(high_price_total < low_price_total))
        self.assertTrue(np.all(high_price_total >= 0.0))

    def test_cpu_price_uses_the_same_normalized_dual_as_the_constraint(self):
        env = make_research_env()
        env.env.Q_loc_high[:] = 9.0
        env.env.Q_loc_low[:] = 0.0
        env.set_energy_lagrange_multipliers([1.0, 1.0, 1.0])

        f_high, f_low, _, _ = env.env._allocate_cpu_frequencies()
        queue_reference_mbits = 90.0
        marginal_gain = (
            env.env.w_D_high
            * 9.0
            / (queue_reference_mbits * 10e9)
        )
        normalized_energy_price = 1.0 / 280.0
        expected_frequency = np.sqrt(
            marginal_gain
            / (3.0 * normalized_energy_price * env.env.kappa)
        )

        np.testing.assert_allclose(
            f_high,
            np.full(3, expected_frequency),
            rtol=1e-7,
        )
        np.testing.assert_allclose(f_low, np.zeros(3), atol=1e-12)

    def test_per_uav_constraint_excludes_ground_uplink_energy(self):
        env = make_research_env()
        env.reset(seed=307)
        env.env.user_lambda_high.fill(0.0)
        env.env.user_lambda_low.fill(0.0)
        env.env.user_lambda_high[0] = 100.0
        env.env.user_task_size_high_mbits[0] = 1.0
        env.macro_step(np.zeros((env.N, 2), dtype=np.float32))
        action = np.zeros(env.micro_action_space.shape[0], dtype=np.float32)
        for _ in range(2):
            env.step_resource_frame(action)
        env.prepare_resource_frame()
        frame = env.step_resource_frame(action)
        info = frame["infos"][0]

        total_service = float(np.sum(info["energy_total_per_uav"]))
        battery_service = float(np.sum(info["uav_battery_energy_per_uav"]))
        self.assertAlmostEqual(
            total_service,
            battery_service + float(info["ground_uplink_energy"]),
            places=8,
        )
        expected_ratio = (
            np.asarray(info["uav_battery_energy_per_uav"], dtype=np.float64)
            + env.current_macro_propulsion_energy_per_uav_share
        ) / (env.uav_power_budgets_w * env.env.micro_dt)
        np.testing.assert_allclose(
            info["energy_constraint_ratio_per_uav"],
            expected_ratio,
            rtol=1e-7,
        )

    def test_dual_prices_are_part_of_both_hierarchical_states(self):
        env = make_research_env()
        env.reset(seed=311)
        env.set_energy_lagrange_multipliers([1.0, 2.0, 3.0])

        _, macro_state = env.get_macro_state()
        micro_state = env.get_sac_state()

        self.assertEqual(macro_state.size, env.macro_global_obs_dim)
        self.assertEqual(micro_state.size, env.sac_obs_dim)
        np.testing.assert_allclose(micro_state[-3:], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            env.env.cpu_energy_prices,
            np.asarray([1.0, 2.0, 3.0]) / 280.0,
        )

    def test_td3_energy_critic_and_dual_update_are_per_uav(self):
        args = SimpleNamespace(
            td3_hidden_dim=32,
            td3_actor_lr=3e-4,
            td3_critic_lr=1e-3,
            td3_gamma=0.99,
            td3_tau=0.005,
            td3_policy_noise=0.02,
            td3_noise_clip=0.05,
            td3_policy_delay=1,
            td3_actor_ema_decay=0.0,
            td3_reward_scale=0.01,
            objective_mode="constrained_delay",
            power_constraint_scope="per_uav",
            energy_dual_lr=0.2,
            energy_dual_max=20.0,
        )
        agent = TD3Agent(8, 6, 3, args, torch.device("cpu"))
        buffer = MacroReplayBuffer(8, 6, 64, energy_dim=3)
        rng = np.random.default_rng(313)
        for _ in range(48):
            buffer.add(
                rng.normal(size=8).astype(np.float32),
                rng.uniform(-0.5, 0.5, size=6).astype(np.float32),
                reward=-0.1,
                next_state=rng.normal(size=8).astype(np.float32),
                done=False,
                energy_ratio=rng.uniform(0.8, 1.2, size=3),
            )
        actor_loss, critic_loss, energy_loss = agent.update(
            buffer,
            16,
            energy_lagrange_multiplier=[0.1, 0.2, 0.3],
        )
        self.assertTrue(np.isfinite(actor_loss))
        self.assertTrue(np.isfinite(critic_loss))
        self.assertTrue(np.isfinite(energy_loss))
        with torch.no_grad():
            q1, q2 = agent.energy_critic(
                torch.zeros((2, 8)),
                torch.zeros((2, 6)),
            )
        self.assertEqual(tuple(q1.shape), (2, 3))
        self.assertEqual(tuple(q2.shape), (2, 3))

        env = make_research_env()
        history = deque(maxlen=5)
        _, after, violations, smoothed = update_smoothed_per_uav_energy_duals(
            env,
            args,
            history,
            [300.0, 270.0, 280.0],
        )
        np.testing.assert_allclose(smoothed, [300.0, 270.0, 280.0])
        self.assertGreater(after[0], 0.0)
        self.assertEqual(after[1], 0.0)
        self.assertEqual(after[2], 0.0)
        self.assertGreater(violations[0], 0.0)

    def test_micro_checkpoint_restores_the_full_per_uav_dual_vector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "micro.pt"
            torch.save(
                {
                    "args": {
                        "energy_lagrange_multiplier_current": 0.2,
                        "energy_lagrange_multipliers_current": [
                            0.1,
                            0.2,
                            0.3,
                        ],
                    }
                },
                checkpoint,
            )
            restored = load_checkpoint_energy_dual(
                str(checkpoint),
                num_uavs=3,
            )
            np.testing.assert_allclose(restored, [0.1, 0.2, 0.3])

    def test_sca_cache_and_objective_preserve_per_uav_energy_prices(self):
        env = make_research_env()
        env.reset(seed=317)
        env.set_energy_lagrange_multipliers([0.1, 0.2, 0.3])
        first_key = _sca_cache_key(env)
        env.set_energy_lagrange_multipliers([0.3, 0.2, 0.1])
        second_key = _sca_cache_key(env)

        self.assertNotEqual(first_key, second_key)
        self.assertAlmostEqual(
            env.energy_lagrange_multiplier,
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
