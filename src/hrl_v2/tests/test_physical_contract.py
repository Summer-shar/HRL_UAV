"""Physical service invariants; no policy training or learned parameters."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np

from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
from hrl_uav.industrial_scene import Arrival
from hrl_v2.physics_adapter import PhysicsAdapter
from hrl_v2.timing import ClockConfig


SCENE = json.loads((Path(__file__).parents[1] / "configs/physical_scene.json").read_text())
LOCAL = np.tile(np.eye(3), (2, 1, 1)).astype(np.float64)


def quiet_scene():
    scene = copy.deepcopy(SCENE)
    for profile in scene["base_scene"]["task_types"].values():
        profile["rate_per_s"] = 0.
    for profile in scene["business_profiles"]["catalog"]:
        for slot in profile["slots"]:
            slot["parameters"]["rate_per_s"] = 0.
    return scene


def admit(env, *, bits=1e6, cycles=1e10, priority=0, source=0, ready=False):
    pid = len(env.parents)
    env._admit(Arrival(pid, source * 13, source, "sensor", priority,
                       env.time_s, bits, cycles))
    parent = env.parents[pid]
    if ready:
        env.upload[source * 13].remove(pid)
        parent.upload_remaining = 0.
        parent.upload_finish_s = env.time_s
        env.upload_served_bits += bits
        env.ready.append(pid)
    return parent


def step(env, route=LOCAL):
    if env.step_index % env.frame_steps == 0:
        env.prepare_frame()
    return env.micro_step(route)


class PhysicalContractTests(unittest.TestCase):
    def setUp(self):
        self.env = HotspotIndustrialEnv(quiet_scene(), ledger_enabled=True)

    def assert_conserved(self, env):
        for name, error in env.conservation().items():
            scale = env.generated_cycles if "cycles" in name else env.generated_bits
            self.assertLessEqual(abs(error), max(1e-12, scale * 1e-13), name)
        for parent in env.parents.values():
            if parent.route_time_s is not None:
                children = [env.fragments[f] for f in parent.fragment_ids]
                self.assertAlmostEqual(sum(f.input_bits for f in children) / parent.arrival.input_bits, 1.)
                self.assertAlmostEqual(sum(f.compute_cycles for f in children) / parent.arrival.compute_cycles, 1.)
                self.assertEqual(parent.finish_s is not None, all(f.finish_s is not None for f in children))

    def test_split_conservation_and_parent_waits_for_last_positive_fragment(self):
        env = self.env
        parent = admit(env, bits=1e8, cycles=1e10, ready=True)
        route = LOCAL.copy()
        route[0, 0] = [.3, .5, .2]
        first = step(env, route)
        self.assertEqual(len(parent.fragment_ids), 3)
        self.assertEqual(first["holding_task_s"], env.dt * 10)
        for _ in range(400):
            self.assert_conserved(env)
            children = [env.fragments[f] for f in parent.fragment_ids]
            if parent.finish_s is not None:
                break
            if any(f.finish_s is not None for f in children):
                self.assertIn(parent.arrival.task_id, env._active_parents)
            step(env, route)
        self.assertIsNotNone(parent.finish_s)
        self.assertEqual(parent.finish_s, max(f.finish_s for f in children))
        self.assertAlmostEqual(env.holding, 10 * parent.finish_s)
        self.assert_conserved(env)

    def test_exact_zero_creates_no_fragment_and_tiny_positive_gets_service(self):
        env = self.env
        parent = admit(env, cycles=1e6, ready=True)
        route = LOCAL.copy()
        route[0, 0] = [1., 1e-20, 0.]
        step(env, route)
        children = [env.fragments[f] for f in parent.fragment_ids]
        self.assertEqual([f.destination for f in children], [0, 1])
        self.assertIsNotNone(children[0].finish_s)
        self.assertIsNone(parent.finish_s)
        step(env, route)
        self.assertGreater(children[1].compute_cycles, 0.)
        self.assertEqual(children[1].compute_remaining, 0.)
        self.assertIsNotNone(parent.finish_s)
        self.assertTrue(any(row["kind"] == "cpu_service" and
                            row["fragment_id"] == children[1].fragment_id and row["cycles"] > 0
                            for row in env.ledger))
        self.assert_conserved(env)

    def test_small_upload_residual_is_not_forgiven(self):
        env = self.env
        env.physics.compute_u2g_rate_bps = lambda: np.ones(39) * 10
        parent = admit(env, bits=1. + 5e-9, cycles=1.)
        step(env)
        self.assertGreater(parent.upload_remaining, 0.)
        self.assertIsNone(parent.upload_finish_s)
        self.assertEqual(env.upload_served_bits, 1.)
        self.assert_conserved(env)
        step(env)
        self.assertEqual(parent.upload_remaining, 0.)
        self.assertEqual(env.upload_served_bits, parent.arrival.input_bits)

    def test_small_cpu_and_transfer_residuals_are_not_forgiven(self):
        scene = quiet_scene()
        scene["base_scene"]["resources"]["cpu_hz"] = [10., 10., 10.]
        env = HotspotIndustrialEnv(scene)
        parent = admit(env, cycles=1. + 5e-7, ready=True)
        step(env)
        fragment = env.fragments[parent.fragment_ids[0]]
        self.assertGreater(fragment.compute_remaining, 0.)
        self.assertIsNone(parent.finish_s)
        self.assertEqual(env.cpu_served_cycles, 1.)
        self.assert_conserved(env)
        step(env)
        self.assertIsNotNone(parent.finish_s)
        self.assertEqual(env.cpu_served_cycles, parent.arrival.compute_cycles)

        env = self.env
        env.physics.allocate_u2u_resources = lambda active: dict(
            u2u_rate_bps=np.asarray(active) * 10.,
            u2u_link_power_w=np.asarray(active) * 1.)
        parent = admit(env, bits=1. + 5e-9, cycles=1., ready=True)
        route = LOCAL.copy()
        route[0, 0] = [0., 1., 0.]
        step(env, route)
        fragment = env.fragments[parent.fragment_ids[0]]
        self.assertGreater(fragment.transfer_remaining, 0.)
        self.assertEqual(fragment.stage, "u2u")
        self.assertEqual(env.tx_served_bits, 1.)
        self.assert_conserved(env)
        step(env, route)
        self.assertEqual(fragment.stage, "cpu")
        self.assertEqual(env.tx_served_bits, parent.arrival.input_bits)

    def test_upload_fifo_and_cpu_strict_priority_then_fifo(self):
        env = self.env
        first = admit(env, bits=1e9, priority=1)
        second = admit(env, bits=1e9, priority=0)
        step(env)
        self.assertLess(first.upload_remaining, 1e9)
        self.assertEqual(second.upload_remaining, 1e9)
        env = HotspotIndustrialEnv(quiet_scene())
        low = admit(env, cycles=2e9, priority=1, ready=True)
        high1 = admit(env, cycles=2e9, ready=True)
        high2 = admit(env, cycles=2e9, ready=True)
        step(env)
        remaining = lambda parent: env.fragments[parent.fragment_ids[0]].compute_remaining
        self.assertEqual(remaining(high1), 1e9)
        self.assertEqual(remaining(high2), 2e9)
        self.assertEqual(remaining(low), 2e9)
        step(env)
        self.assertIsNotNone(high1.finish_s)
        self.assertEqual(remaining(high2), 2e9)
        step(env)
        self.assertEqual(remaining(high2), 1e9)
        self.assertEqual(remaining(low), 2e9)

    def test_capacity_and_energy_are_physical_and_cannot_be_bypassed(self):
        env = self.env
        admit(env, bits=1e9, cycles=1e13, ready=True)
        admit(env, bits=1e9, cycles=1e13, ready=True, priority=1)
        admit(env, bits=1e9, cycles=1e13, source=1)
        route = np.full((2, 3, 3), 1 / 3, dtype=np.float64)
        rates = env.physics.compute_u2g_rate_bps()
        reading = step(env, route)
        cpu = [row for row in env.ledger if row["kind"] == "cpu_service"]
        self.assertLessEqual(sum(row["cycles"] for row in cpu),
                             sum(env.config["resources"]["cpu_hz"]) * env.dt)
        self.assertLessEqual(env.upload_served_bits - 2e9, rates[13] * env.dt)
        tx = [row for row in env.ledger if row["kind"] == "u2u_service"]
        for destination in (1, 2):
            selected = [row for row in tx if env.fragments[row["fragment_id"]].destination == destination]
            self.assertLessEqual(sum(row["bits"] for row in selected), selected[0]["rate_bps"] * env.dt)
        active = np.array([[False, True, True], [False, False, False], [False, False, False]])
        resource = env.physics.allocate_u2u_resources(active)
        self.assertEqual(resource["u2u_link_bandwidth_hz"].sum(), env.physics.B_u2u_total)
        self.assertEqual(resource["u2u_link_power_w"].sum(), env.physics.u2u_tx_budget_w)
        self.assertEqual(reading["holding_task_s"], env.dt * 21)
        self.assertAlmostEqual(reading["cost"], .09 * reading["holding_task_s"] +
                               sum(reading["energy_j"].values()) / 140)
        for name in ("upload", "u2u", "cpu", "propulsion"):
            self.assertGreater(reading["energy_j"][name], 0.)
        self.assert_conserved(env)

    def test_invalid_simplex_is_rejected_before_any_service(self):
        env = self.env
        env.prepare_frame()
        for route in (np.zeros((2, 3, 3)), -LOCAL, np.full((2, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                env.micro_step(route)
            self.assertEqual(env.step_index, 0)
            self.assertEqual(env.cost_sum, 0.)

    def test_queues_survive_lower_upper_and_rollout_boundaries(self):
        scene = quiet_scene()
        timing = ClockConfig.from_scene(scene, rollout_seconds=5.)
        adapter = PhysicsAdapter(scene, timing=timing)
        adapter.reset(arrival_seed=1, layout_seed=2, load_seed=3)
        env = adapter.env
        upload = admit(env, bits=1e12)
        cpu = admit(env, cycles=1e13, ready=True, source=1)
        tx = admit(env, bits=1e12, cycles=1e13, ready=True, source=2)
        route = LOCAL.copy()
        route[0, 2] = [1., 0., 0.]
        for _ in range(timing.physical_steps + timing.physical_per_lower):
            if env.step_index % env.macro_steps == 0:
                adapter.start_macro(np.zeros(6))
            if env.step_index % env.frame_steps == 0:
                adapter.prepare()
            adapter.step(route)
        self.assertGreater(env.step_index, timing.physical_steps)
        self.assertIs(env.parents[upload.arrival.task_id], upload)
        self.assertIn(upload.arrival.task_id, env.upload[0])
        self.assertIn(cpu.fragment_ids[0], env.compute[0][1])
        self.assertIn(tx.fragment_ids[0], env.transfer[0][2][0])
        self.assertIsNone(upload.finish_s)
        self.assertIsNone(cpu.finish_s)
        self.assertIsNone(tx.finish_s)
        self.assert_conserved(env)

    def test_ready_at_upper_boundary_waits_for_next_physical_step(self):
        env = self.env
        env.start_motion(np.zeros(6))
        for _ in range(env.macro_steps - 1):
            step(env)
        env.physics.compute_u2g_rate_bps = lambda: np.ones(39)
        parent = admit(env, bits=env.dt, cycles=1e12)
        step(env)
        self.assertIn(parent.arrival.task_id, env.ready)
        env.start_motion(np.zeros(6))
        env.prepare_frame()
        self.assertIn(parent.arrival.task_id, env.ready)
        step(env)
        self.assertIsNotNone(parent.route_time_s)
        self.assertIsNone(parent.finish_s)

    def test_motion_enforces_displacement_speed_map_and_continuous_separation(self):
        env = self.env
        env.physics.uav_coords = np.array([[1., 1.], [11., 1.], [999., 999.]])
        start = env.physics.uav_coords.copy()
        motion = env.start_motion(np.array([100., 0., -100., 0., 100., 100.]))
        self.assertLess(motion["safety_scale"], 1.)
        self.assertTrue((np.linalg.norm(motion["displacement_m"], axis=1) <= 10.).all())
        self.assertTrue((np.linalg.norm(motion["velocity_mps"], axis=1) <= 10.).all())
        self.assertGreaterEqual(motion["minimum_separation_m"], 5. - 1e-9)
        for _ in range(env.macro_steps):
            step(env)
            position = env.physics.uav_coords
            self.assertTrue(((0 <= position) & (position <= env.config["map_size_m"])).all())
            distance = np.linalg.norm(position[:, None] - position[None, :], axis=2)
            self.assertGreaterEqual(distance[~np.eye(3, dtype=bool)].min(), 5. - 1e-9)
        np.testing.assert_allclose(env.physics.uav_coords, start + motion["displacement_m"])
        np.testing.assert_array_equal(env.velocity, np.zeros((3, 2)))


if __name__ == "__main__":
    unittest.main()
