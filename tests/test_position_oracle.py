import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from search_position_oracle import cem_search
from search_position_oracle import capacity_ordered_targets
from search_position_oracle import checkpoint_rank
from search_position_oracle import centroid_offset_targets
from search_position_oracle import make_uav1_centroid_override_policy
from search_position_oracle import project_targets
from search_position_oracle import solve_uplink_reference_position
from search_position_oracle import target_centroid_waypoint_action
from search_position_oracle import target_delta_action
from search_position_oracle import target_waypoint_action
from search_position_oracle import uplink_reference_objective_and_gradient


class PositionOracleTests(unittest.TestCase):
    def test_checkpoint_rank_uses_system_ucb_violation(self):
        feasible = {
            "delay_objective": 1.0,
            "power_constraint_scope": "system",
            "power_ucb95_violation_w": 0.0,
            "max_battery_power_ucb95_violation_w": 300.0,
        }
        infeasible = dict(
            feasible,
            delay_objective=0.5,
            power_ucb95_violation_w=2.0,
        )

        self.assertLess(checkpoint_rank(feasible), checkpoint_rank(infeasible))

    def test_target_action_respects_displacement_limit_and_stops(self):
        current = np.array(
            [[0.0, 0.0], [10.0, 10.0], [50.0, 50.0]],
            dtype=np.float64,
        )
        target = np.array(
            [[30.0, 40.0], [10.1, 10.1], [45.0, 50.0]],
            dtype=np.float64,
        )
        action = target_delta_action(
            current,
            target,
            max_displacement_m=20.0,
            tolerance_m=0.5,
        )
        physical = action * 20.0

        self.assertAlmostEqual(np.linalg.norm(physical[0]), 20.0, places=5)
        self.assertTrue(np.allclose(physical[1], 0.0))
        self.assertTrue(np.allclose(physical[2], [-5.0, 0.0]))

    def test_waypoint_action_encodes_target_from_episode_origin(self):
        origins = np.array(
            [[350.0, 350.0], [350.0, 650.0], [650.0, 500.0]]
        )
        targets = origins + np.array(
            [[20.0, 0.0], [0.0, -40.0], [-60.0, 0.0]]
        )
        action = target_waypoint_action(
            targets,
            origins,
            waypoint_radius_m=200.0,
        )

        np.testing.assert_allclose(
            action,
            np.array(
                [[0.1, 0.0], [0.0, -0.2], [-0.3, 0.0]],
                dtype=np.float32,
            ),
            atol=1e-7,
        )

    def test_centroid_waypoint_action_uses_current_centroid(self):
        centroids = np.array(
            [[350.0, 350.0], [350.0, 650.0], [650.0, 500.0]]
        )
        targets = centroids + np.array(
            [[0.0, 0.0], [20.0, -10.0], [-60.0, 80.0]]
        )
        action = target_centroid_waypoint_action(
            targets,
            centroids,
            waypoint_radius_m=200.0,
        )

        np.testing.assert_allclose(
            action,
            np.array(
                [[0.0, 0.0], [0.1, -0.05], [-0.3, 0.4]],
                dtype=np.float32,
            ),
            atol=1e-7,
        )

    def test_target_projection_respects_anchor_radius_and_map(self):
        anchors = np.array(
            [[100.0, 100.0], [500.0, 500.0], [900.0, 900.0]],
            dtype=np.float64,
        )
        targets = np.array(
            [[-500.0, 100.0], [700.0, 500.0], [1500.0, 1500.0]],
            dtype=np.float64,
        )
        projected = project_targets(
            targets,
            anchors,
            max_offset_m=80.0,
            map_min_m=0.0,
            map_max_m=1000.0,
        )

        self.assertTrue(np.all(projected >= 0.0))
        self.assertTrue(np.all(projected <= 1000.0))
        self.assertTrue(
            np.all(np.linalg.norm(projected - anchors, axis=1) <= 80.0 + 1e-8)
        )

    def test_centroid_offset_targets_preserve_relative_offsets(self):
        centroids = np.array(
            [[350.0, 350.0], [350.0, 650.0], [650.0, 500.0]]
        )
        offsets = np.array(
            [[0.0, 0.0], [40.0, -50.0], [-70.0, -20.0]]
        )
        targets = centroid_offset_targets(centroids, offsets)

        np.testing.assert_allclose(targets, centroids + offsets)

    def test_capacity_ordered_targets_move_light_uav_farther(self):
        centroids = np.array(
            [[350.0, 350.0], [350.0, 650.0], [650.0, 500.0]]
        )
        targets = capacity_ordered_targets(
            centroids,
            medium_offset_m=60.0,
            light_offset_m=120.0,
        )
        offsets = np.linalg.norm(targets - centroids, axis=1)

        np.testing.assert_allclose(offsets, [0.0, 60.0, 120.0])
        self.assertLess(
            np.linalg.norm(targets[1] - centroids[0]),
            np.linalg.norm(centroids[1] - centroids[0]),
        )
        self.assertLess(
            np.linalg.norm(targets[2] - centroids[0]),
            np.linalg.norm(centroids[2] - centroids[0]),
        )

    def test_uplink_reference_gradient_matches_finite_difference(self):
        users = np.array(
            [[-20.0, 5.0], [10.0, 30.0], [40.0, -10.0]],
            dtype=np.float64,
        )
        loads = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        position = np.array([5.0, 8.0], dtype=np.float64)
        kwargs = {
            "altitude_m": 50.0,
            "bandwidth_per_user_hz": 5e5,
            "p_gt_power": 0.1,
            "beta_0": 1e-6,
            "noise_power": 1e-13,
        }
        _, analytic = uplink_reference_objective_and_gradient(
            position,
            users,
            loads,
            **kwargs,
        )
        step = 1e-4
        numerical = np.zeros(2, dtype=np.float64)
        for axis in range(2):
            positive = position.copy()
            negative = position.copy()
            positive[axis] += step
            negative[axis] -= step
            positive_value, _ = uplink_reference_objective_and_gradient(
                positive,
                users,
                loads,
                **kwargs,
            )
            negative_value, _ = uplink_reference_objective_and_gradient(
                negative,
                users,
                loads,
                **kwargs,
            )
            numerical[axis] = (
                positive_value - negative_value
            ) / (2.0 * step)

        np.testing.assert_allclose(
            analytic,
            numerical,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_uplink_reference_is_center_for_symmetric_users(self):
        users = np.array(
            [[-50.0, 0.0], [50.0, 0.0], [0.0, -50.0], [0.0, 50.0]],
            dtype=np.float64,
        )
        target, diagnostic = solve_uplink_reference_position(
            users,
            np.ones(4, dtype=np.float64),
            initial_position=np.array([12.0, -9.0]),
            altitude_m=50.0,
            bandwidth_per_user_hz=5e5,
            p_gt_power=0.1,
            beta_0=1e-6,
            noise_power=1e-13,
            map_min_m=-100.0,
            map_max_m=100.0,
        )

        np.testing.assert_allclose(target, [0.0, 0.0], atol=1e-3)
        self.assertLessEqual(
            diagnostic["final_objective"],
            diagnostic["initial_objective"],
        )

    def test_uplink_reference_moves_toward_heavier_user(self):
        target, _ = solve_uplink_reference_position(
            user_coords=np.array([[0.0, 0.0], [100.0, 0.0]]),
            weighted_load_mbps=np.array([1.0, 10.0]),
            initial_position=np.array([50.0, 0.0]),
            altitude_m=50.0,
            bandwidth_per_user_hz=5e5,
            p_gt_power=0.1,
            beta_0=1e-6,
            noise_power=1e-13,
            map_min_m=0.0,
            map_max_m=100.0,
        )

        self.assertGreater(target[0], 50.0)
        self.assertAlmostEqual(target[1], 0.0, places=5)

    def test_uav1_override_preserves_teammate_actions(self):
        class BaseEnv:
            def get_raw_state(self):
                return {
                    "uav_coords": np.array(
                        [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
                    ),
                    "weighted_centroids": np.array(
                        [[10.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
                    ),
                }

        class Wrapper:
            env = BaseEnv()
            macro_action_mode = "delta"
            max_macro_displacement = 20.0

        learned_action = np.array(
            [[-0.4, -0.2], [0.3, -0.1], [-0.2, 0.5]],
            dtype=np.float32,
        )

        def learned_policy(_env, _obs, _eval):
            return learned_action

        policy = make_uav1_centroid_override_policy(learned_policy)
        action = policy(Wrapper(), None, True)

        np.testing.assert_allclose(action[0], [0.5, 0.0])
        np.testing.assert_allclose(action[1:], learned_action[1:])

    def test_cem_improves_a_deterministic_quadratic(self):
        anchors = np.zeros((3, 2), dtype=np.float64)
        optimum = np.array(
            [[20.0, -10.0], [-15.0, 25.0], [5.0, 30.0]],
            dtype=np.float64,
        )

        def evaluate(targets):
            objective = float(np.sum((targets - optimum) ** 2))
            return {
                "delay_objective": objective,
                "task_delay_weighted_ms": objective,
                "max_battery_power_violation_w": 0.0,
            }

        best, _ = cem_search(
            evaluate_target=evaluate,
            anchors=anchors,
            population=40,
            iterations=8,
            elite_fraction=0.2,
            initial_std_m=40.0,
            minimum_std_m=0.5,
            max_offset_m=100.0,
            map_min_m=-100.0,
            map_max_m=100.0,
            seed=9,
        )

        self.assertLess(
            np.linalg.norm(best["targets_m"] - optimum),
            8.0,
        )


if __name__ == "__main__":
    unittest.main()
