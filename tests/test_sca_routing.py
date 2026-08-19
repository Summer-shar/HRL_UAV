import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hrl_uav.hrl_wrappers import HRL_Wrapper
from compare_sca import optimize_routing_logits_SCA
from compare_sca import solve_sca_micro_action


class RoutingOnlySCATest(unittest.TestCase):
    def test_projected_sca_returns_bounded_simplex_routes(self):
        action, info = optimize_routing_logits_SCA(
            expected_load_mbits=np.array(
                [[0.278, 0.127, 0.049], [0.972, 0.423, 0.301]],
                dtype=np.float64,
            ),
            queue_high_mbits=np.zeros(3),
            queue_low_mbits=np.zeros(3),
            service_capacity_mbits=np.ones(3),
            nominal_u2u_rate_mbps=np.array(
                [[0.0, 40.0, 35.0], [40.0, 0.0, 38.0], [35.0, 38.0, 0.0]],
                dtype=np.float64,
            ),
            f_max_hz=np.full(3, 10e9),
            kappa=1e-28,
            micro_dt=0.5,
            tx_power=1.0,
            diagonal_bias=1.0,
            action_scale=1.0,
            max_iter=3,
        )

        self.assertEqual(action.shape, (18,))
        self.assertLessEqual(float(np.max(np.abs(action))), 1.0 + 1e-7)
        rho = np.asarray(info["rho"], dtype=np.float64)
        np.testing.assert_allclose(np.sum(rho, axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.all(rho >= 0.0))
        objective = np.asarray(info["objective_history"], dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(objective)))
        self.assertLessEqual(objective[-1], objective[0] + 1e-8)

    def test_environment_executes_exact_sca_route_without_cpu_action(self):
        env = HRL_Wrapper(
            num_uavs=3,
            num_users=39,
            cpu_alloc_mode="priority_demand",
            load_scale=1.0,
            zone_loads=[12.5, 5.5, 3.5],
            micro_diag_bias=1.0,
            micro_action_scale=1.0,
            macro_flow_state=True,
            macro_load_state=True,
            macro_capacity_state=True,
            macro_action_mode="delta",
            macro_reward_mode="strict_global",
            strict_hierarchy_validation=True,
            pause_service_during_flight=True,
        )
        env.env.F_max_array[:] = 10e9
        env.sca_update_interval = 1
        env.sca_max_iter = 3
        env.sca_tol = 1e-3
        env.sca_solver = "CLARABEL"
        env.sca_delay_weight = 0.7
        env.sca_energy_weight = 0.3
        env.sca_proximal_weight = 1e-4
        env.reset(seed=701)
        env.macro_step(np.zeros((env.N, 2), dtype=np.float32))

        zero_action = np.zeros(env.micro_action_space.shape[0], dtype=np.float32)
        for _ in range(10):
            _, _, _, _, paused_info = env.micro_step(zero_action)
            self.assertTrue(paused_info["service_paused"])

        env.prepare_resource_frame()
        action = solve_sca_micro_action(env)
        expected_rho = np.asarray(env._sca_rho_warm).reshape(2, env.N, env.N)
        frame = env.step_resource_frame(action)
        step_info = frame["infos"][0]

        np.testing.assert_allclose(
            step_info["split_ratios_high"],
            expected_rho[0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            step_info["split_ratios_low"],
            expected_rho[1],
            atol=1e-6,
        )
        self.assertFalse(hasattr(env, "_sca_f_warm"))
        self.assertEqual(step_info["cpu_alloc_mode"], "priority_demand")
        self.assertEqual(getattr(env, "_sca_solve_count", 0), 1)
        self.assertGreater(getattr(env, "_sca_total_solve_seconds", 0.0), 0.0)

    def test_sca_can_reactivate_a_route_from_a_simplex_boundary(self):
        num_uavs = 3
        local_action = np.full((2 * num_uavs, num_uavs), -1.0)
        for priority_offset in (0, num_uavs):
            rows = np.arange(num_uavs) + priority_offset
            local_action[rows, np.arange(num_uavs)] = 1.0

        _, info = optimize_routing_logits_SCA(
            expected_load_mbits=np.asarray(
                [[8.0, 2.0, 1.0], [12.0, 3.0, 2.0]],
                dtype=np.float64,
            ),
            queue_high_mbits=np.asarray([100.0, 0.0, 0.0]),
            queue_low_mbits=np.asarray([100.0, 0.0, 0.0]),
            service_capacity_mbits=np.full(num_uavs, 5.0),
            nominal_u2u_rate_mbps=np.asarray(
                [
                    [0.0, 20.0, 20.0],
                    [20.0, 0.0, 20.0],
                    [20.0, 20.0, 0.0],
                ],
                dtype=np.float64,
            ),
            f_max_hz=np.full(num_uavs, 10e9),
            kappa=1e-28,
            micro_dt=0.5,
            tx_power=1.0,
            diagonal_bias=0.0,
            action_scale=1.0,
            delay_weight_high=10.0,
            delay_weight_low=1.0,
            omega_D=0.9,
            omega_E=0.1,
            delay_norm=10.0,
            energy_norm=14.0,
            action_init=local_action.reshape(-1),
            max_iter=5,
            proximal_weight=1e-4,
        )

        rho = np.asarray(info["rho"]).reshape(2, num_uavs, num_uavs)
        self.assertGreater(float(np.sum(rho[:, 0, 1:])), 0.05)
        objective = np.asarray(info["objective_history"], dtype=np.float64)
        self.assertLess(objective[-1], objective[0])


if __name__ == "__main__":
    unittest.main()
