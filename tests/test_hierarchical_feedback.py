import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hrl_uav.hrl_wrappers import HRL_Wrapper
from scripts.compare_common import build_micro_step_metrics
from scripts.train_centralized_macro import sample_training_start_positions


def make_env(
    *,
    pause_service=True,
    capacity_state=True,
    objective_mode="weighted",
    power_budget_w=0.0,
    energy_lagrange_multiplier=0.0,
    micro_reward_scale=10.0,
    uav_initial_positions=None,
    macro_action_mode="delta",
):
    env = HRL_Wrapper(
        num_uavs=3,
        num_users=39,
        traffic_profile="normal",
        traffic_scale=0.5,
        cpu_alloc_mode="priority_demand",
        load_scale=1.0,
        zone_loads=[12.5, 5.5, 3.5],
        micro_diag_bias=1.0,
        micro_reward_scale=micro_reward_scale,
        macro_flow_state=True,
        macro_load_state=True,
        macro_capacity_state=capacity_state,
        macro_action_mode=macro_action_mode,
        macro_reward_mode="strict_global",
        strict_hierarchy_validation=True,
        pause_service_during_flight=pause_service,
        objective_mode=objective_mode,
        power_budget_w=power_budget_w,
        energy_lagrange_multiplier=energy_lagrange_multiplier,
        uav_initial_positions=uav_initial_positions,
    )
    env.env.F_max_array[:] = 10e9
    env.macro_shaping_lambda = 0.0
    env.macro_move_penalty_weight = 0.0
    env.macro_propulsion_tiebreak_weight = 0.0
    return env


def run_macro_slot(env, macro_action, micro_action=None):
    if micro_action is None:
        micro_action = np.zeros(env.micro_action_space.shape[0], dtype=np.float32)
    env.macro_step(np.asarray(macro_action, dtype=np.float32))
    rewards = []
    paused = 0
    for _ in range(env.env.micro_steps_per_macro):
        _, reward, _, _, info = env.micro_step(micro_action)
        rewards.append(float(reward))
        paused += int(bool(info.get("service_paused", False)))
    macro_reward = float(env.get_macro_reward())
    return np.asarray(rewards, dtype=np.float64), macro_reward, paused


class HierarchicalFeedbackTest(unittest.TestCase):
    def test_micro_reward_scale_changes_only_reward_units(self):
        env = make_env(micro_reward_scale=1.0)
        env.reset(seed=98)
        env.macro_step(np.zeros((env.N, 2), dtype=np.float32))
        action = np.zeros(env.micro_action_space.shape[0], dtype=np.float32)
        for _ in range(10):
            env.micro_step(action)
        _, reward, _, _, info = env.micro_step(action)

        self.assertEqual(info["micro_reward_scale"], 1.0)
        self.assertAlmostEqual(reward, -info["J_step"], places=6)

    def test_waypoint_action_reaches_target_without_cumulative_overshoot(self):
        env = make_env(
            pause_service=True,
            macro_action_mode="waypoint",
        )
        env.reset(seed=99)
        start = env.env.uav_coords.copy()
        action = np.zeros((env.N, 2), dtype=np.float32)
        action[0, 0] = 0.5

        for _ in range(6):
            run_macro_slot(env, action)

        expected = start.copy()
        expected[0, 0] += 100.0
        np.testing.assert_allclose(env.env.uav_coords, expected, atol=1e-5)

        run_macro_slot(env, action)
        np.testing.assert_allclose(env.env.uav_coords, expected, atol=1e-5)

    def test_waypoint_origin_tracks_each_episode_start(self):
        starts = np.asarray(
            [[140.0, 160.0], [420.0, 620.0], [730.0, 510.0]],
            dtype=np.float64,
        )
        env = make_env(
            uav_initial_positions=starts,
            macro_action_mode="waypoint",
        )
        env.reset(seed=123)

        np.testing.assert_allclose(env.macro_waypoint_origins, starts)

    def test_centroid_waypoint_zero_action_tracks_current_centroid(self):
        env = make_env(
            pause_service=True,
            macro_action_mode="centroid_waypoint",
        )
        env.reset(seed=124)
        raw_state = env.env.get_raw_state()
        start = np.asarray(raw_state["uav_coords"], dtype=np.float64)
        centroids = np.asarray(
            raw_state["weighted_centroids"],
            dtype=np.float64,
        )
        displacement = centroids - start
        distance = np.linalg.norm(displacement, axis=1, keepdims=True)
        expected = start + displacement / np.maximum(
            distance,
            1e-12,
        ) * np.minimum(distance, env.max_macro_displacement)

        run_macro_slot(env, np.zeros((env.N, 2), dtype=np.float32))

        np.testing.assert_allclose(env.env.uav_coords, expected, atol=1e-5)

    def test_strict_macro_reward_is_exact_micro_mean(self):
        env = make_env()
        env.reset(seed=101)
        flow_before = env.prev_macro_flow_ema_mbits.copy()
        rewards, macro_reward, _ = run_macro_slot(
            env,
            np.zeros((env.N, 2), dtype=np.float32),
        )
        info = env.get_last_macro_reward_info()

        self.assertEqual(info["micro_steps_executed"], 100)
        self.assertEqual(info["micro_steps_expected"], 100)
        self.assertAlmostEqual(info["micro_reward_sum"], rewards.sum(), places=5)
        self.assertAlmostEqual(info["micro_reward_mean"], rewards.mean(), places=6)
        self.assertAlmostEqual(macro_reward, rewards.mean(), places=6)
        self.assertAlmostEqual(info["reward_reconstruction_error"], 0.0, places=8)
        self.assertFalse(np.allclose(flow_before, env.prev_macro_flow_ema_mbits))

    def test_macro_transition_always_pauses_exactly_one_second(self):
        hover_env = make_env(pause_service=True)
        hover_env.reset(seed=202)
        _, _, hover_paused = run_macro_slot(
            hover_env,
            np.zeros((hover_env.N, 2), dtype=np.float32),
        )

        moving_env = make_env(pause_service=True)
        moving_env.reset(seed=202)
        move_action = np.zeros((moving_env.N, 2), dtype=np.float32)
        move_action[0, 0] = 1.0
        _, _, moving_paused = run_macro_slot(moving_env, move_action)

        self.assertEqual(hover_paused, 10)
        self.assertEqual(moving_paused, 10)
        self.assertEqual(
            hover_env.env.get_task_delay_stats()["task_arrived_count"],
            moving_env.env.get_task_delay_stats()["task_arrived_count"],
        )

    def test_nonzero_macro_action_moves_incrementally_before_hover(self):
        env = make_env(pause_service=True)
        env.reset(seed=2021)
        start = env.env.uav_coords.copy()
        queue_before = np.concatenate(
            [
                env.env.Q_loc_high.ravel(),
                env.env.Q_loc_low.ravel(),
                env.env.Q_tx_high.ravel(),
                env.env.Q_tx_low.ravel(),
            ]
        )
        action = np.zeros((env.N, 2), dtype=np.float32)
        action[0, 0] = 1.0
        env.macro_step(action)
        np.testing.assert_allclose(env.env.uav_coords, start)

        x_positions = []
        micro_action = np.zeros(
            env.micro_action_space.shape[0],
            dtype=np.float32,
        )
        for _ in range(10):
            _, _, _, _, info = env.micro_step(micro_action)
            self.assertTrue(info["service_paused"])
            x_positions.append(float(env.env.uav_coords[0, 0]))

        self.assertTrue(np.all(np.diff(x_positions) > 0.0))
        self.assertAlmostEqual(x_positions[-1] - start[0, 0], 20.0, places=5)
        queue_after = np.concatenate(
            [
                env.env.Q_loc_high.ravel(),
                env.env.Q_loc_low.ravel(),
                env.env.Q_tx_high.ravel(),
                env.env.Q_tx_low.ravel(),
            ]
        )
        np.testing.assert_allclose(queue_after, queue_before)

        _, _, _, _, hover_info = env.micro_step(micro_action)
        self.assertFalse(hover_info["service_paused"])

    def test_zero_displacement_uses_control_phase_without_flight(self):
        env = make_env(pause_service=True)
        env.reset(seed=2022)
        start = env.env.uav_coords.copy()
        env.macro_step(np.zeros((env.N, 2), dtype=np.float32))
        micro_action = np.zeros(
            env.micro_action_space.shape[0],
            dtype=np.float32,
        )

        for _ in range(10):
            _, _, _, _, info = env.micro_step(micro_action)
            self.assertTrue(info["service_paused"])
            self.assertTrue(info["macro_transition_phase"])
            self.assertFalse(info["flight_phase"])
            np.testing.assert_allclose(env.env.uav_coords, start)

        _, _, _, _, service_info = env.micro_step(micro_action)
        self.assertFalse(service_info["service_paused"])
        self.assertFalse(service_info["macro_transition_phase"])

    def test_transition_has_neither_task_generation_nor_service(self):
        env = make_env(pause_service=True)
        env.reset(seed=203)
        initial_arrived = env.env.get_task_delay_stats()["task_arrived_count"]
        initial_completed = env.env.get_task_delay_stats()["task_completed_count"]
        move_action = np.zeros((env.N, 2), dtype=np.float32)
        move_action[0, 0] = 1.0
        env.macro_step(move_action)
        micro_action = np.zeros(env.micro_action_space.shape[0], dtype=np.float32)
        for _ in range(10):
            _, _, _, _, info = env.micro_step(micro_action)
            self.assertTrue(info["service_paused"])
            self.assertEqual(info["total_energy"], 0.0)

        stats = env.env.get_task_delay_stats()
        self.assertEqual(stats["task_arrived_count"], initial_arrived)
        self.assertEqual(stats["task_completed_count"], initial_completed)
        self.assertEqual(
            float(np.sum(env.env.A_t_high + env.env.A_t_low)),
            0.0,
        )

    def test_existing_backlog_waits_and_is_penalized_during_transition(self):
        env = make_env(pause_service=True)
        env.reset(seed=205)
        env.env._create_parent_task(user_id=0, priority="high", size_mbits=1.0)
        arrived_before = env.env.get_task_delay_stats()["task_arrived_count"]

        env.macro_step(np.zeros((env.N, 2), dtype=np.float32))
        micro_action = np.zeros(
            env.micro_action_space.shape[0],
            dtype=np.float32,
        )
        _, reward, _, _, info = env.micro_step(micro_action)

        self.assertTrue(info["service_paused"])
        self.assertEqual(
            info["task_arrived_count"],
            arrived_before,
        )
        self.assertGreater(info["task_holding_cost_weighted"], 0.0)
        self.assertGreater(info["delay_objective_step"], 0.0)
        self.assertLess(reward, 0.0)

    def test_service_phase_arrivals_are_reported_exactly_once(self):
        totals = []
        for moving in (False, True):
            env = make_env(pause_service=True)
            env.reset(seed=204)
            action = np.zeros((env.N, 2), dtype=np.float32)
            if moving:
                action[0, 0] = 1.0
            env.macro_step(action)

            reported = np.zeros(env.N, dtype=np.float64)
            micro_action = np.zeros(
                env.micro_action_space.shape[0],
                dtype=np.float32,
            )
            for step in range(env.env.micro_steps_per_macro):
                _, _, _, _, info = env.micro_step(micro_action)
                raw_state = env.env.get_raw_state()
                metrics = build_micro_step_metrics(raw_state, info)
                reported += metrics["arrival_total_per_uav"]
                if step < 10:
                    self.assertTrue(info["service_paused"])
                    self.assertEqual(
                        float(np.sum(raw_state["A_t_high"] + raw_state["A_t_low"])),
                        0.0,
                    )
                    self.assertEqual(
                        float(np.sum(metrics["arrival_total_per_uav"])),
                        0.0,
                    )
            totals.append(reported)

        np.testing.assert_allclose(totals[0], totals[1], rtol=0.0, atol=1e-6)
        self.assertGreater(float(np.sum(totals[0])), 0.0)

    def test_incomplete_macro_slot_is_rejected(self):
        env = make_env()
        env.reset(seed=303)
        env.macro_step(np.zeros((env.N, 2), dtype=np.float32))
        env.micro_step(np.zeros(env.micro_action_space.shape[0], dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "executed 1 micro steps"):
            env.get_macro_reward()

    def test_reported_j_matches_the_optimized_micro_cost(self):
        env = make_env()
        env.reset(seed=304)
        env.max_Q_threshold = 1e-4
        env.env.B_gt_total = 1e12
        env.env.user_lambda_low[:] = 100.0
        env.env._generate_traffic()
        env.macro_step(np.zeros((env.N, 2), dtype=np.float32))
        action = np.zeros(env.micro_action_space.shape[0], dtype=np.float32)
        for _ in range(10):
            _, _, _, _, paused_info = env.micro_step(action)
            self.assertTrue(paused_info["service_paused"])
        env.micro_step(action)
        _, reward, _, _, info = env.micro_step(action)

        self.assertGreater(info["overflow_cost"], 0.0)
        self.assertAlmostEqual(
            info["J_step"],
            info["J_base_step"] + info["drift_cost"] + info["overflow_cost"],
            places=7,
        )
        self.assertAlmostEqual(
            reward,
            -info["J_step"] * info["micro_reward_scale"],
            places=6,
        )

    def test_no_cooperation_is_an_environment_constraint(self):
        env = make_env()
        env.force_local_routing = True
        env.reset(seed=404)
        env.macro_step(np.zeros((env.N, 2), dtype=np.float32))
        extreme_action = np.full(
            env.micro_action_space.shape[0],
            100.0,
            dtype=np.float32,
        )
        for _ in range(10):
            _, _, _, _, paused_info = env.micro_step(extreme_action)
            self.assertTrue(paused_info["service_paused"])
        _, _, _, _, info = env.micro_step(extreme_action)
        np.testing.assert_allclose(info["split_ratios_high"], np.eye(env.N))
        np.testing.assert_allclose(info["split_ratios_low"], np.eye(env.N))

    def test_macro_state_observes_cpu_capacity(self):
        env8 = make_env(capacity_state=True)
        env8.env.F_max_array[:] = 8e9
        _, state8 = env8.reset(seed=505)[0]

        env14 = make_env(capacity_state=True)
        env14.env.F_max_array[:] = 14e9
        _, state14 = env14.reset(seed=505)[0]

        self.assertEqual(state8.shape, state14.shape)
        self.assertFalse(np.allclose(state8, state14))

    def test_custom_uav_initial_positions_survive_reset(self):
        positions = np.asarray(
            [[250.0, 250.0], [250.0, 750.0], [750.0, 500.0]],
            dtype=np.float64,
        )
        env = make_env(uav_initial_positions=positions)
        env.reset(seed=506)
        np.testing.assert_allclose(env.env.uav_coords, positions)

    def test_training_start_randomization_is_reproducible_and_bounded(self):
        env = make_env()
        args = Namespace(
            seed=17,
            train_start_jitter_max_m=180.0,
            train_start_anchor_probability=0.0,
        )
        first = sample_training_start_positions(env, args, episode=3)
        second = sample_training_start_positions(env, args, episode=3)
        np.testing.assert_allclose(first, second)
        offsets = np.linalg.norm(first - env.env.anchor_points, axis=1)
        self.assertTrue(np.all(offsets <= 180.0 + 1e-8))
        self.assertTrue(np.all(first >= 0.0))
        self.assertTrue(np.all(first <= 1000.0))

        args.train_start_anchor_probability = 1.0
        anchored = sample_training_start_positions(env, args, episode=3)
        np.testing.assert_allclose(anchored, env.env.anchor_points)

    def test_full_energy_share_conserves_macro_and_service_energy(self):
        env = make_env()
        env.reset(seed=606)
        move_action = np.zeros((env.N, 2), dtype=np.float32)
        move_action[:, 0] = 0.5
        _, _, macro_info = env.macro_step(move_action)
        full_energy = 0.0
        service_energy = 0.0
        controllable_energy = 0.0
        micro_action = np.zeros(env.micro_action_space.shape[0], dtype=np.float32)
        for _ in range(env.env.micro_steps_per_macro):
            _, _, _, _, info = env.micro_step(micro_action)
            full_energy += float(info["full_energy_step_j"])
            service_energy += float(info["service_energy_step_j"])
            controllable_energy += float(info["controllable_energy_step_j"])

        self.assertAlmostEqual(
            full_energy,
            service_energy + float(macro_info["E_macro_total"]),
            places=4,
        )
        self.assertAlmostEqual(
            controllable_energy,
            service_energy
            + float(macro_info["macro_propulsion_delta_total"]),
            places=4,
        )

    def test_constrained_reward_uses_signed_long_term_power_residual(self):
        env = make_env(
            objective_mode="constrained_delay",
            power_budget_w=700.0,
            energy_lagrange_multiplier=0.4,
        )
        env.reset(seed=607)
        env.macro_step(np.zeros((env.N, 2), dtype=np.float32))
        zero_action = np.zeros(
            env.micro_action_space.shape[0],
            dtype=np.float32,
        )
        for _ in range(10):
            _, _, _, _, paused_info = env.micro_step(zero_action)
            self.assertTrue(paused_info["service_paused"])
        _, reward, _, _, info = env.micro_step(
            zero_action
        )

        expected_cost = (
            float(info["primal_cost_step"])
            + 0.4
            * (float(info["energy_constraint_ratio"]) - 1.0)
        )
        self.assertAlmostEqual(info["J_step"], expected_cost, places=7)
        self.assertAlmostEqual(
            reward,
            -env.micro_reward_scale * expected_cost,
            places=6,
        )
        self.assertAlmostEqual(
            info["power_budget_step_j"],
            700.0 * env.env.micro_dt,
            places=7,
        )
        self.assertAlmostEqual(
            info["service_energy_constraint_ratio"],
            info["service_energy_step_j"]
            / info["power_budget_step_j"],
            places=7,
        )
        self.assertAlmostEqual(
            info["energy_constraint_ratio"],
            info["full_energy_step_j"]
            / info["power_budget_step_j"],
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
