import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from hrl_uav.bandwidth import MeteredMDP, summarize_meter
from hrl_uav.experiment import boundary_reward
from hrl_uav.mdp import QueueMDP


ROOT = Path(__file__).resolve().parents[1]


def config(enabled=None):
    result = json.loads((ROOT / "configs/hrl_ppo_only_unlock.json").read_text())
    if enabled is not None:
        result["environment"]["compute_during_deployment"] = enabled
    return result


def make_mdp(enabled=True, metered=False):
    mdp = (MeteredMDP if metered else QueueMDP)(config(enabled))
    mdp.reset(51, 3)
    return mdp


def queued_parent(env, priority, pieces):
    """Build uploaded work; pieces are (destination, Mbit, compute or tx)."""
    parent = env._create_parent_task(0, priority, sum(p[1] for p in pieces), source_uav=0)
    env.current_arrival_tasks_by_uav[0].remove(parent.task_id)
    parent.remaining_upload_mbits = 0.0
    parent.upload_finish_time = env._current_time()
    parent.total_chunks = len(pieces)
    chunks = []
    for destination, size, stage in pieces:
        chunk = env._create_chunk(parent, size, destination, stage)
        if stage == "compute":
            env._enqueue_compute_chunk(chunk)
        else:
            env._enqueue_tx_chunk(chunk)
        chunks.append(chunk)
    return parent, chunks


class DeploymentComputeTests(unittest.TestCase):
    def assert_nested_equal(self, left, right):
        if isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            self.assertEqual(left, right)

    def test_default_off_and_explicit_false_are_identical(self):
        missing, explicit = make_mdp(None), make_mdp(False)
        self.assertFalse(missing.env.compute_during_deployment)
        self.assertNotIn("compute_during_deployment", missing.config["environment"])
        for mdp in (missing, explicit):
            queued_parent(mdp.env, "low", [(0, 2.0, "compute")])
            mdp.start_macro(np.zeros(6))
        for index in range(22):
            if index == 20:
                for mdp in (missing, explicit):
                    mdp.advance_macro_clock()
                    mdp.start_macro(np.full(6, 0.1))
            self.assert_nested_equal(missing.step_frame(np.zeros(18)), explicit.step_frame(np.zeros(18)))
            self.assert_nested_equal(missing.env.get_raw_state(), explicit.env.get_raw_state())
            self.assert_nested_equal(missing.metrics(), explicit.metrics())
            if index < 2:
                self.assertEqual(missing.env.Q_loc_low[0], 2.0)

    def test_option_rejects_non_boolean_and_full_service_during_flight(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            QueueMDP(config("false"))
        bad = config(True)
        bad["environment"]["pause_service_during_flight"] = False
        with self.assertRaisesRegex(ValueError, "pause_service_during_flight=True"):
            QueueMDP(bad)

    def test_cpu_priority_energy_completion_and_communication_pause(self):
        mdp = make_mdp()
        env = mdp.env
        high, _ = queued_parent(env, "high", [(0, 0.6, "compute")])
        low, _ = queued_parent(env, "low", [(0, 0.8, "compute")])
        remote, chunks = queued_parent(env, "high", [(1, 1.0, "tx")])
        # A partially transferred fragment is still ineligible for computation.
        chunks[0].remaining_mbits = 0.2
        env.Q_tx_high[0, 1] = 0.2
        upload = env._create_parent_task(0, "low", 0.5, source_uav=0)
        env.current_arrival_tasks_by_uav[0].remove(upload.task_id)
        env.uplink_task_queues[0].append(upload.task_id)
        ready = env._create_parent_task(0, "low", 0.3, source_uav=0)
        env.current_arrival_tasks_by_uav[0].remove(ready.task_id)
        ready.remaining_upload_mbits = 0.0
        ready.upload_finish_time = 0.0
        env.completed_upload_tasks_by_uav[0].append(ready.task_id)
        before_tx = env.Q_tx_high.copy()
        before_flow = env.coop_flow_mbits.copy()
        env.macro_step(np.zeros((3, 2)))
        with patch.object(env, "_generate_traffic", return_value={}), \
             patch.object(env, "_route_arrived_parent_tasks", side_effect=AssertionError("routing during deployment")), \
             patch.object(env, "_serve_uplink_task_queues", side_effect=AssertionError("upload during deployment")), \
             patch.object(env, "_serve_tx_task_queues", side_effect=AssertionError("U2U during deployment")):
            state, first = env.micro_step(np.eye(3), np.eye(3))
            self.assertTrue(first["service_paused"])
            self.assertFalse(first["compute_service_paused"])
            self.assertAlmostEqual(first["compute_service_high_mbits"], 0.6)
            self.assertAlmostEqual(first["compute_service_low_mbits"], 0.4)
            np.testing.assert_allclose(first["compute_service_per_uav_mbits"], [1, 0, 0])
            np.testing.assert_allclose(first["f_high_per_uav"], [6e9, 0, 0])
            np.testing.assert_allclose(first["f_low_per_uav"], [4e9, 0, 0])
            self.assertAlmostEqual(first["total_energy"], 10.0)
            self.assertAlmostEqual(first["energy_comp"], 10.0)
            for key in ("energy_comp_per_uav", "energy_total_per_uav", "uav_battery_energy_per_uav"):
                np.testing.assert_allclose(first[key], [10, 0, 0])
            self.assertEqual(first["energy_uplink"], 0.0)
            self.assertEqual(first["energy_trans"], 0.0)
            self.assertAlmostEqual(first["task_holding_cost_weighted"], 1.3)
            self.assertAlmostEqual(state["Q_loc_low"][0], 0.4)
            self.assertAlmostEqual(high.finish_time, 0.1)
            self.assertIsNone(low.finish_time)
            self.assertEqual(first["task_completed_high_count"], 1)
            self.assertAlmostEqual(first["task_delay_high_ms"], 100.0)
            _, second = env.micro_step(np.eye(3), np.eye(3))
            self.assertAlmostEqual(second["total_energy"], 0.64)
            self.assertAlmostEqual(low.finish_time, 0.2)
            self.assertEqual(second["task_completed_low_count"], 1)
            self.assertAlmostEqual(second["task_holding_cost_weighted"], 1.2)
            for _ in range(8):
                _, info = env.micro_step(np.eye(3), np.eye(3))
                self.assertEqual(info["energy_comp"], 0)
        np.testing.assert_array_equal(env.Q_tx_high, before_tx)
        np.testing.assert_array_equal(env.coop_flow_mbits, before_flow)
        self.assertEqual(upload.remaining_upload_mbits, 0.5)
        self.assertEqual(list(env.uplink_task_queues[0]), [upload.task_id])
        self.assertEqual(env.completed_upload_tasks_by_uav[0], [ready.task_id])
        self.assertEqual(ready.total_chunks, 0)
        self.assertIsNone(remote.finish_time)
        self.assertEqual(env.completed_task_count, 2)
        self.assertAlmostEqual(env._current_time(), 1.0)
        self.assertEqual(env.macro_service_pause_steps_remaining, 0)

    def test_parent_still_waits_for_unreceived_fragment(self):
        mdp = make_mdp()
        parent, _ = queued_parent(mdp.env, "low", [(0, 0.2, "compute"), (1, 0.8, "tx")])
        mdp.start_macro(np.zeros(6))
        with patch.object(mdp.env, "_generate_traffic", return_value={}):
            frame = mdp.step_frame(np.zeros(18))
        self.assertEqual(parent.completed_chunks, 1)
        self.assertEqual(parent.chunk_finish_times, [0.1])
        self.assertIsNone(parent.finish_time)
        self.assertEqual(frame["infos"][-1]["task_completed_count"], 0)
        self.assertAlmostEqual(frame["infos"][-1]["task_holding_cost_weighted"], 0.1)

    def test_high_backlog_cannot_exceed_cpu_cap_or_serve_low_early(self):
        mdp = make_mdp()
        queued_parent(mdp.env, "high", [(0, 1.2, "compute")])
        queued_parent(mdp.env, "low", [(0, 0.8, "compute")])
        mdp.start_macro(np.zeros(6))
        with patch.object(mdp.env, "_generate_traffic", return_value={}):
            _, first = mdp.env.micro_step(np.eye(3), np.eye(3))
            _, second = mdp.env.micro_step(np.eye(3), np.eye(3))
        self.assertAlmostEqual(first["compute_service_high_mbits"], 1.0)
        self.assertEqual(first["compute_service_low_mbits"], 0)
        self.assertAlmostEqual(second["compute_service_high_mbits"], 0.2)
        self.assertAlmostEqual(second["compute_service_low_mbits"], 0.8)
        for info in (first, second):
            self.assertTrue(np.all(info["f_total_per_uav"] <= mdp.env.F_max_array))
            self.assertAlmostEqual(info["energy_comp"], 10.0)

    def test_candidate_meter_is_read_only_across_two_macro_boundaries(self):
        plain, metered = make_mdp(), make_mdp(metered=True)
        for mdp in (plain, metered):
            queued_parent(mdp.env, "low", [(0, 12.0, "compute"), (1, 1.0, "tx")])
        for _ in range(2):
            for mdp in (plain, metered):
                mdp.start_macro(np.full(6, 0.1))
            for _ in range(20):
                self.assert_nested_equal(plain.step_frame(np.zeros(18)), metered.step_frame(np.zeros(18)))
                self.assert_nested_equal(plain.metrics(), metered.metrics())
            for mdp in (plain, metered):
                mdp.advance_macro_clock()
        self.assertEqual(sum(r["compute_steps"] for r in metered.readings), 200)
        self.assertEqual(sum(r["service_steps"] for r in metered.readings), 180)

    def test_deployment_compute_uses_same_cpu_rules_as_service(self):
        for mode in ("priority_demand", "priority_kkt"):
            with self.subTest(mode=mode):
                mdp = make_mdp()
                moving = mdp.env
                moving.cpu_alloc_mode = mode
                queued_parent(moving, "high", [(0, 0.6, "compute")])
                queued_parent(moving, "low", [(0, 1.2, "compute")])
                normal = copy.deepcopy(moving)
                moving.macro_step(np.zeros((3, 2)))
                with patch.object(moving, "_generate_traffic", return_value={}), \
                     patch.object(normal, "_generate_traffic", return_value={}):
                    _, paused_info = moving.micro_step(np.eye(3), np.eye(3))
                    _, normal_info = normal.micro_step(np.eye(3), np.eye(3))
                for key in normal_info:
                    if key.startswith(("cpu_", "f_", "energy_comp", "compute_service", "task_")):
                        self.assert_nested_equal(paused_info[key], normal_info[key])

    def test_arrivals_motion_and_clocks_unchanged_while_cpu_only_advances(self):
        old, candidate = make_mdp(False), make_mdp(True)
        for mdp in (old, candidate):
            queued_parent(mdp.env, "low", [(0, 100.0, "compute")])
            mdp.start_macro(np.full(6, 0.1))
        for _ in range(2):
            old_frame = old.step_frame(np.zeros(18))
            new_frame = candidate.step_frame(np.full(18, 0.9))
            self.assertEqual(old_frame["phase"], "deployment")
            self.assertEqual(new_frame["phase"], "deployment")
            for key in ("uav_coords", "user_coords", "weighted_centroids", "A_t_high", "A_t_low"):
                np.testing.assert_array_equal(getattr(old.env, key), getattr(candidate.env, key))
            self.assertEqual(old.env.arrived_task_count, candidate.env.arrived_task_count)
            self.assertEqual(old.env.mobility_update_count, candidate.env.mobility_update_count)
            np.testing.assert_array_equal(old.env.np_random.get_state()[1], candidate.env.np_random.get_state()[1])
        self.assertGreater(candidate.env.deployment_arrival_prepare_count, 0)
        self.assertGreater(candidate.env.arrived_task_count, 1)
        self.assertEqual(candidate.env.global_micro_step, 10)
        self.assertAlmostEqual(old.env.Q_loc_low[0] - candidate.env.Q_loc_low[0], 10.0)
        self.assertEqual(candidate.observation_spec(), old.observation_spec())
        self.assertNotEqual(candidate.prepare_service().tolist(), old.prepare_service().tolist())
        for mdp in (old, candidate):
            frame = mdp.step_frame(np.zeros(18))
            self.assertEqual(frame["phase"], "service")
            self.assertFalse(frame["infos"][0]["service_paused"])
            self.assertAlmostEqual(mdp.env._current_time(), 1.5)

    def test_reward_energy_and_boundary_include_deployment_exactly_once(self):
        mdp = make_mdp()
        queued_parent(mdp.env, "high", [(0, 0.6, "compute")])
        queued_parent(mdp.env, "low", [(0, 0.8, "compute")])
        mdp.start_macro(np.full(6, 0.1))
        gamma = mdp.config["training"]["gamma_per_resource_frame"]
        with patch.object(mdp.env, "_generate_traffic", return_value={}):
            frames = [mdp.step_frame(np.zeros(18), gamma) for _ in range(2)]
        self.assertAlmostEqual(sum(i["total_energy"] for f in frames for i in f["infos"]), 10.64)
        expected_costs = [0.9 * 0.1 / 10 + 0.1 * 10 / 14, 0.1 * 0.64 / 14] + [0.0] * 8
        expected_costs = [c + 0.1 * mdp.propulsion_delta_share / 14 for c in expected_costs]
        expected_rewards = [-sum(gamma**(k / 5) * expected_costs[offset + k] for k in range(5))
                            for offset in (0, 5)]
        np.testing.assert_allclose([f["reward"] for f in frames], expected_rewards, atol=1e-12)
        self.assertAlmostEqual(mdp.cost_sum, sum(expected_costs))
        self.assertAlmostEqual(mdp.total_power_energy, 10.64 + 10 * mdp.propulsion_share)
        self.assertEqual(mdp.integration_steps, 10)
        reward, discount = boundary_reward(-2.0, [f["reward"] for f in frames], gamma)
        self.assertAlmostEqual(reward, -2 + gamma * expected_rewards[0] + gamma**2 * expected_rewards[1])
        self.assertAlmostEqual(discount, gamma**3)
        self.assertEqual(mdp.prepare_service().shape, (68,))

    def test_meter_includes_cpu_during_communication_pause(self):
        mdp = make_mdp(metered=True)
        queued_parent(mdp.env, "low", [(0, 12.0, "compute")])
        mdp.start_macro(np.zeros(6))
        with patch.object(mdp.env, "_generate_traffic", return_value={}):
            for _ in range(2):
                mdp.step_frame(np.zeros(18))
        self.assertEqual(sum(r["service_steps"] for r in mdp.readings), 0)
        self.assertEqual(sum(r["compute_steps"] for r in mdp.readings), 10)
        summary = summarize_meter(mdp.readings, 1.0)
        self.assertAlmostEqual(summary["compute_throughput_mbps_wall"], 10.0)
        self.assertAlmostEqual(summary["cpu_utilization_service"], 1 / 3)
        self.assertEqual(summary["tx_served_mbit"], 0)
        self.assertEqual(summary["mean_active_links_service"], 0)
        self.assertAlmostEqual(mdp.env.Q_loc_low[0], 2.0)
        self.assertAlmostEqual(sum(r["computed_mbit"] for r in mdp.uav_readings), 10.0)
        self.assertLess(max(abs(r["compute_balance_error"]) for r in mdp.uav_readings), 1e-10)
        self.assertLess(max(r["transfer_balance_error_max"] for r in mdp.uav_readings), 1e-10)


if __name__ == "__main__":
    unittest.main()
