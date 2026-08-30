import unittest

import numpy as np

from hrl_uav.xu2024_sca import Xu2024SCAInput
from hrl_uav.xu2024_sca import nearest_uav_association
from hrl_uav.xu2024_sca import solve_xu2024_sca


class Xu2024SCATest(unittest.TestCase):
    def _problem(self, task_mbits=None):
        if task_mbits is None:
            task_mbits = np.asarray([0.5, 0.5, 0.5], dtype=np.float64)
        return Xu2024SCAInput(
            task_mbits=np.asarray(task_mbits, dtype=np.float64),
            cycles_per_bit=1000.0,
            source_uav=np.asarray([0, 1, 2], dtype=np.int64),
            u2g_rate_mbps=np.full(3, 20.0, dtype=np.float64),
            u2u_rate_mbps=np.asarray(
                [
                    [0.0, 20.0, 20.0],
                    [20.0, 0.0, 20.0],
                    [20.0, 20.0, 0.0],
                ],
                dtype=np.float64,
            ),
            f_max_ghz=np.full(3, 22.0, dtype=np.float64),
            capacitance_coeff=1e-28,
            frame_seconds=0.5,
            gt_tx_power_w=0.1,
            uav_tx_power_w=1.0,
            delay_weight=0.3,
            energy_weight=0.7,
        )

    def test_nearest_uav_association_matches_paper_greedy_rule(self):
        users = np.asarray([[0.0, 0.0], [9.0, 1.0], [5.0, 8.0]])
        uavs = np.asarray([[0.0, 1.0], [10.0, 0.0], [5.0, 10.0]])
        np.testing.assert_array_equal(
            nearest_uav_association(users, uavs),
            np.asarray([0, 1, 2]),
        )

    def test_reference_solver_satisfies_simplex_cpu_and_deadline_constraints(self):
        problem = self._problem()
        solution = solve_xu2024_sca(
            problem,
            max_iterations=5,
            tolerance=0.01,
        )

        self.assertIn(solution.status, {"optimal", "optimal_inaccurate"})
        np.testing.assert_allclose(
            np.sum(solution.rho, axis=1),
            np.ones(3),
            atol=1e-6,
        )
        self.assertTrue(np.all(solution.rho >= -1e-9))
        self.assertTrue(np.all(solution.cpu_ghz >= 0.0))
        self.assertTrue(np.all(np.sum(solution.cpu_ghz, axis=0) <= 22.0 + 1e-5))
        self.assertTrue(np.all(solution.completion_time_s <= 0.5 + 1e-5))
        self.assertTrue(np.isfinite(solution.objective))

    def test_delay_focused_solution_uses_idle_uav_capacity_for_heavy_task(self):
        problem = self._problem(task_mbits=[4.0, 0.2, 0.2])
        problem = Xu2024SCAInput(
            **{
                **problem.__dict__,
                "delay_weight": 1.0,
                "energy_weight": 0.0,
                "frame_seconds": 2.0,
            }
        )
        solution = solve_xu2024_sca(
            problem,
            max_iterations=10,
            tolerance=0.01,
        )

        heavy_remote_fraction = float(np.sum(solution.rho[0, 1:]))
        self.assertGreater(heavy_remote_fraction, 0.05)

    def test_route_wait_cost_avoids_a_congested_destination(self):
        problem = self._problem(task_mbits=[4.0, 0.2, 0.2])
        route_wait = np.zeros((3, 3), dtype=np.float64)
        route_wait[0, 1] = 10.0
        congested_problem = Xu2024SCAInput(
            **{
                **problem.__dict__,
                "delay_weight": 1.0,
                "energy_weight": 0.0,
                "frame_seconds": 2.0,
                "route_wait_cost_s": route_wait,
            }
        )
        solution = solve_xu2024_sca(
            congested_problem,
            max_iterations=10,
            tolerance=0.01,
            enforce_frame_deadline=False,
        )

        self.assertLess(solution.rho[0, 1], solution.rho[0, 2])

    def test_queue_adapter_enforces_link_and_compute_service_capacity(self):
        problem = self._problem(task_mbits=[4.0, 0.2, 0.2])
        capacity = np.full((3, 3), 10.0, dtype=np.float64)
        np.fill_diagonal(capacity, 0.0)
        capacity[0, 1] = 0.0
        stable_problem = Xu2024SCAInput(
            **{
                **problem.__dict__,
                "delay_weight": 1.0,
                "energy_weight": 0.0,
                "frame_seconds": 2.0,
                "u2u_capacity_mbits": capacity,
                "enforce_service_stability": True,
            }
        )
        solution = solve_xu2024_sca(
            stable_problem,
            max_iterations=10,
            tolerance=0.01,
            enforce_frame_deadline=False,
        )

        self.assertLessEqual(solution.rho[0, 1], 1e-6)
        workload = stable_problem.task_mbits * 1000.0 * 1e-3
        for destination in range(3):
            routed_work = float(workload @ solution.rho[:, destination])
            served_work = float(
                stable_problem.frame_seconds
                * np.sum(solution.cpu_ghz[:, destination])
            )
            self.assertLessEqual(routed_work, served_work + 1e-5)


if __name__ == "__main__":
    unittest.main()
