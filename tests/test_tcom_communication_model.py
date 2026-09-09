import unittest

import numpy as np

from hrl_uav.env import UAV_MEC_Env
from hrl_uav.routing import project_rows_to_simplex


class TcomCommunicationModelTest(unittest.TestCase):


    def test_u2g_rate_matches_reference_equation(self):
        env = UAV_MEC_Env(num_uavs=3, num_users=3, users_static=True)
        users = np.asarray(
            [[100.0, 200.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float64,
        )
        uavs = np.asarray(
            [[130.0, 240.0], [500.0, 500.0], [800.0, 800.0]],
            dtype=np.float64,
        )
        rate = env.compute_u2g_rate_bps(
            user_coords=users,
            uav_coords=uavs,
            associations=np.asarray([0, 1, 2]),
        )[0]
        distance_sq = 30.0**2 + 40.0**2 + env.H**2
        expected = (env.B_gt_total / env.M) * np.log2(
            1.0
            + env.p_gt
            * (env.beta_0 / distance_sq)
            / env.noise_power
        )
        self.assertAlmostEqual(rate, expected, places=7)

    def test_new_task_uses_fixed_region_association(self):
        env = UAV_MEC_Env(num_uavs=3, num_users=3, users_static=True)
        env.reset(seed=41)
        env.uav_coords[:] = np.asarray(
            [[100.0, 100.0], [500.0, 500.0], [900.0, 900.0]]
        )
        env.user_coords[0] = np.asarray([480.0, 510.0])
        parent = env._create_parent_task(
            user_id=0,
            priority="high",
            size_mbits=0.5,
        )
        self.assertEqual(parent.source_uav, 0)
        self.assertIn(parent.task_id, env.current_arrival_tasks_by_uav[0])



    def test_one_gt_cannot_borrow_an_idle_gt_uplink_band(self):
        env = UAV_MEC_Env(
            num_uavs=3,
            num_users=2,
            users_static=True,
        )
        env.reset(seed=42)
        env.uav_coords[:] = np.asarray(
            [[350.0, 350.0], [350.0, 650.0], [650.0, 500.0]]
        )
        env.user_coords[0] = env.uav_coords[0]
        rate_mbits_per_s = env.compute_u2g_rate_bps(
            user_coords=env.user_coords[[0]],
            associations=np.asarray([0]),
        )[0] / 1e6
        task_size = 1.5 * rate_mbits_per_s * env.micro_dt
        parent = env._create_parent_task(
            user_id=0,
            priority="low",
            size_mbits=task_size,
            source_uav=0,
        )
        env._admit_current_arrival_tasks_to_uplink()
        metrics = env._serve_uplink_task_queues()

        self.assertGreater(parent.remaining_upload_mbits, 0.0)
        self.assertEqual(metrics["completed_per_uav"][0], 0)
        self.assertAlmostEqual(
            metrics["service_mbits_per_user"][0],
            rate_mbits_per_s * env.micro_dt,
            places=7,
        )
        self.assertAlmostEqual(
            metrics["energy_per_uav"][0],
            env.p_gt * env.micro_dt,
            places=7,
        )

    def test_u2u_dynamically_shares_total_bandwidth_across_active_links(self):
        env = UAV_MEC_Env(num_uavs=3, num_users=3)
        active = np.zeros((env.N, env.N), dtype=bool)
        active[0, 1] = True
        active[2, 0] = True
        expected_bandwidth = env.B_u2u_total / 2.0
        rate, bandwidth = env.compute_u2u_rate_bps(
            np.asarray(
                [[0.0, 0.0], [300.0, 0.0], [0.0, 400.0]],
                dtype=np.float64,
            ),
            active_link_mask=active,
            return_bandwidth=True,
        )

        self.assertEqual(env.u2u_bandwidth_policy, (
            "work_conserving_equal_active_directed_links"
        ))
        np.testing.assert_allclose(bandwidth[active], expected_bandwidth)
        np.testing.assert_allclose(bandwidth[~active], 0.0)
        self.assertAlmostEqual(
            float(np.sum(bandwidth)),
            env.B_u2u_total,
        )

        expected = expected_bandwidth * np.log2(
            1.0
            + env.p_u2u
            * (env.beta_0 / 300.0**2)
            / env.noise_power
        )
        self.assertAlmostEqual(rate[0, 1], expected, places=7)

        no_active = env.allocate_u2u_link_bandwidth_hz(
            np.zeros((env.N, env.N), dtype=bool)
        )
        np.testing.assert_allclose(no_active, 0.0)

    def test_routing_projection_contains_simplex_boundary(self):
        projected = project_rows_to_simplex(
            np.asarray(
                [
                    [2.0, -1.0, -1.0],
                    [-1.0, 2.0, -1.0],
                    [-1.0, -1.0, 2.0],
                ]
            )
        )
        np.testing.assert_allclose(projected, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(
            np.sum(projected, axis=1),
            np.ones(3),
            atol=1e-12,
        )



    def test_joint_flight_respects_five_meter_separation(self):
        env = UAV_MEC_Env(
            num_uavs=3,
            num_users=3,
            uav_initial_positions=[
                [100.0, 100.0],
                [110.0, 100.0],
                [800.0, 800.0],
            ],
            min_uav_separation_m=5.0,
        )
        env.reset(seed=43)
        actions = np.asarray(
            [[10.0, 0.0], [-10.0, 0.0], [0.0, 0.0]],
            dtype=np.float64,
        )
        _, _, info = env.macro_step(actions)
        self.assertAlmostEqual(
            info["separation_projection_scale"],
            0.25,
            places=7,
        )
        self.assertGreaterEqual(
            info["minimum_flight_separation_m"],
            5.0 - 1e-7,
        )

    def test_parent_completion_waits_for_slowest_parallel_branch(self):
        env = UAV_MEC_Env(num_uavs=3, num_users=3)
        env.reset(seed=44)
        parent = env._create_parent_task(
            user_id=0,
            priority="high",
            size_mbits=1.0,
            source_uav=0,
        )
        first = env._create_chunk(parent, 0.4, target_uav=0, stage="compute_queue")
        second = env._create_chunk(parent, 0.6, target_uav=1, stage="u2u_queue")
        parent.total_chunks = 2

        env._mark_chunk_completed(first, finish_time=1.25)
        self.assertIsNone(parent.finish_time)
        env._mark_chunk_completed(second, finish_time=2.75)

        self.assertEqual(env.completed_task_count, 1)
        self.assertAlmostEqual(env.completed_task_delay_sum, 2.75)
        self.assertNotIn(parent.task_id, env.parent_tasks)


if __name__ == "__main__":
    unittest.main()
