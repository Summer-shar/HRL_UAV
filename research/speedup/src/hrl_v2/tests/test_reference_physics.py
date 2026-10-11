"""Reference-informed capacity checks; nine physical steps in total.

LyDROO d1084ab1038a49eb8c97b4a9342c456920728eaf supplies the
backlog/service and shared-resource invariants, not this UAV radio model.
All arrivals here are explicit fixtures; no workload frame is sampled.
"""
import unittest

import numpy as np

from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
from hrl_v2.tests.test_physical_contract import LOCAL, SCENE, admit, step


def fixture():
    env = HotspotIndustrialEnv(SCENE, ledger_enabled=True)
    env._scene.frame = lambda frame: ()
    return env


def transfer_service(env, source, destination, priority=None):
    return sum(row["bits"] for row in env.ledger
               if row["kind"] == "u2u_service"
               and env.fragments[row["fragment_id"]].source == source
               and env.fragments[row["fragment_id"]].destination == destination
               and (priority is None or env.fragments[row["fragment_id"]].priority == priority))


class ReferenceCapacityTests(unittest.TestCase):
    def test_zero_service_preserves_all_backlogs_until_capacity_returns(self):
        env = fixture()
        uploading = admit(env, bits=1e9, cycles=1e12)
        computing = admit(env, bits=1e9, cycles=1e12, source=1, ready=True)
        transferring = admit(env, bits=1e9, cycles=1e12, source=2, ready=True)
        route = LOCAL.copy()
        route[0, 2] = [1., 0., 0.]
        upload_rate = env.physics.compute_u2g_rate_bps
        transfer_resources = env.physics.allocate_u2u_resources
        cpu_hz = env.config["resources"]["cpu_hz"]
        baseline = (env.upload_served_bits, env.tx_served_bits, env.cpu_served_cycles)

        def unavailable_links(active):
            resources = transfer_resources(active)
            resources["u2u_rate_bps"] = np.zeros((3, 3))
            return resources

        env.physics.compute_u2g_rate_bps = lambda: np.zeros(39)
        env.physics.allocate_u2u_resources = unavailable_links
        env.config["resources"]["cpu_hz"] = [0., 0., 0.]
        for _ in range(3):
            reading = step(env, route)
            self.assertEqual(uploading.upload_remaining, 1e9)
            self.assertEqual(env.fragments[computing.fragment_ids[0]].compute_remaining, 1e12)
            self.assertEqual(env.fragments[transferring.fragment_ids[0]].transfer_remaining, 1e9)
            self.assertEqual((env.upload_served_bits, env.tx_served_bits, env.cpu_served_cycles), baseline)
            self.assertTrue(all(p.finish_s is None for p in env.parents.values()))
            for name in ("upload", "u2u", "cpu"):
                self.assertEqual(reading["energy_j"][name], 0.)
        self.assertEqual(tuple(env.upload[0]), (uploading.arrival.task_id,))
        self.assertEqual(tuple(env.compute[0][1]), tuple(computing.fragment_ids))
        self.assertEqual(tuple(env.transfer[0][2][0]), tuple(transferring.fragment_ids))

        env.physics.compute_u2g_rate_bps = upload_rate
        env.physics.allocate_u2u_resources = transfer_resources
        env.config["resources"]["cpu_hz"] = cpu_hz
        reading = step(env, route)
        self.assertLess(uploading.upload_remaining, 1e9)
        self.assertLess(env.fragments[computing.fragment_ids[0]].compute_remaining, 1e12)
        self.assertLess(env.fragments[transferring.fragment_ids[0]].transfer_remaining, 1e9)
        for name in ("upload", "u2u", "cpu"):
            self.assertGreater(reading["energy_j"][name], 0.)
        self.assertEqual(env.step_index, 4)

    def test_multiple_links_share_bandwidth_with_per_link_power_and_bounded_service(self):
        env = fixture()
        env.config["resources"]["cpu_hz"] = [0., 0., 0.]
        admit(env, bits=1e9, cycles=1e12, source=0, priority=0, ready=True)
        admit(env, bits=1e9, cycles=1e12, source=0, priority=1, ready=True)
        admit(env, bits=1e9, cycles=1e12, source=1, ready=True)
        small = admit(env, bits=.125, cycles=1., source=2, ready=True)
        route = LOCAL.copy()
        route[:, 0] = [0., .5, .5]
        route[0, 1] = [0., 0., 1.]
        route[0, 2] = [1., 0., 0.]
        active = np.array([[False, True, True], [False, False, True], [True, False, False]])
        resources = env.physics.allocate_u2u_resources(active)
        np.testing.assert_array_equal(resources["u2u_link_bandwidth_hz"][active],
                                      np.full(4, env.physics.B_u2u_total / 4))
        self.assertEqual(resources["u2u_link_bandwidth_hz"].sum(), env.physics.B_u2u_total)
        # v2.4.0: UAV 0 has two outgoing links and splits its budget; UAVs 1 and 2 have one.
        np.testing.assert_array_equal(resources["u2u_link_power_w"][active],
                                      env.physics.u2u_tx_budget_w/np.array([2., 2., 1., 1.]))
        step(env, route)
        for source, destination in zip(*np.nonzero(active)):
            backlog = sum(f.input_bits for f in env.fragments.values()
                          if (f.source, f.destination) == (source, destination))
            expected = min(backlog, resources["u2u_rate_bps"][source, destination] * env.dt)
            self.assertAlmostEqual(transfer_service(env, source, destination), expected, places=8)
        for destination in (1, 2):
            # High priority consumes this link's sole budget; low priority is
            # not allocated a second copy of rate*dt.
            self.assertGreater(transfer_service(env, 0, destination, 0), 0.)
            self.assertEqual(transfer_service(env, 0, destination, 1), 0.)
        self.assertEqual(transfer_service(env, 2, 0), small.arrival.input_bits)
        self.assertEqual(env.fragments[small.fragment_ids[0]].transfer_remaining, 0.)
        self.assertEqual(env.cpu_served_cycles, 0.)

    def test_upload_position_changes_shannon_rate_and_actual_service(self):
        observations = []
        for horizontal_distance in (0., 400.):
            env = fixture()
            env.config["resources"]["cpu_hz"] = [0., 0., 0.]
            terminal = env.physics.user_coords[0].copy()
            env.physics.uav_coords[0] = terminal + [horizontal_distance, 0.]
            parent = admit(env, bits=1e9, cycles=1e12)
            bandwidth = env.physics.B_gt_total / env.physics.M
            gain = env.physics.beta_0 / (horizontal_distance**2 + env.physics.H**2)
            expected_rate = bandwidth * np.log2(1 + env.physics.p_gt * gain /
                                                  (env.physics.noise_psd_w_per_hz * bandwidth))
            actual_rate = env.physics.compute_u2g_rate_bps()[0]
            self.assertAlmostEqual(actual_rate, expected_rate, places=8)
            step(env)
            expected_service = min(parent.arrival.input_bits, expected_rate * env.dt)
            self.assertAlmostEqual(env.upload_served_bits, expected_service, places=8)
            self.assertEqual(env.cpu_served_cycles, 0.)
            observations.append(dict(rate=actual_rate, service=env.upload_served_bits,
                bandwidth=bandwidth, power=env.physics.p_gt,
                associations=env.physics.user_allocations.copy(),
                terminals=env.physics.user_coords.copy(), other_uavs=env.physics.uav_coords[1:].copy()))
        near, far = observations
        self.assertGreater(near["rate"], far["rate"])
        self.assertGreater(near["service"], far["service"])
        for key in ("bandwidth", "power", "associations", "terminals", "other_uavs"):
            np.testing.assert_array_equal(near[key], far[key])

    def test_u2u_position_changes_shannon_rate_under_the_same_active_mask(self):
        observations = []
        active = np.array([[False, True, True], [False, False, False], [False, True, False]])
        for source_x in (390., 100.):
            env = fixture()
            env.config["resources"]["cpu_hz"] = [0., 0., 0.]
            env.physics.uav_coords = np.array([[source_x, 500.], [400., 500.], [800., 800.]])
            admit(env, bits=1e9, cycles=1e12, source=0, ready=True)
            admit(env, bits=1e9, cycles=1e12, source=2, ready=True)
            route = LOCAL.copy()
            route[0, 0] = [0., .5, .5]
            route[0, 2] = [0., 1., 0.]
            resources = env.physics.allocate_u2u_resources(active)
            bandwidth = env.physics.B_u2u_total / active.sum()
            power = env.physics.u2u_tx_budget_w / active[0].sum()
            distance_sq = (400. - source_x)**2
            gain = env.physics.beta_0 / distance_sq
            expected_rate = bandwidth * np.log2(1 + power * gain /
                                                  env.physics.u2u_noise_w)
            self.assertAlmostEqual(resources["u2u_rate_bps"][0, 1], expected_rate, places=8)
            step(env, route)
            self.assertAlmostEqual(transfer_service(env, 0, 1), min(5e8, expected_rate * env.dt), places=8)
            actual_active = np.array([[any(env.transfer[p][i][j] for p in range(2))
                                       for j in range(3)] for i in range(3)])
            np.testing.assert_array_equal(actual_active, active)
            self.assertEqual(env.cpu_served_cycles, 0.)
            observations.append(dict(resources=resources, service=transfer_service(env, 0, 1),
                associations=env.physics.user_allocations.copy(),
                terminals=env.physics.user_coords.copy(), other_uavs=env.physics.uav_coords[1:].copy()))
        near, far = observations
        self.assertGreater(near["resources"]["u2u_rate_bps"][0, 1], far["resources"]["u2u_rate_bps"][0, 1])
        self.assertGreater(near["service"], far["service"])
        for key in ("u2u_link_bandwidth_hz", "u2u_link_power_w", "u2u_link_noise_w"):
            np.testing.assert_array_equal(near["resources"][key], far["resources"][key])
        self.assertEqual(near["resources"]["u2u_rate_bps"][2, 1], far["resources"]["u2u_rate_bps"][2, 1])
        for key in ("associations", "terminals", "other_uavs"):
            np.testing.assert_array_equal(near[key], far[key])


if __name__ == "__main__":
    unittest.main()
