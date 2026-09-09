import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from hrl_uav.bandwidth import MeteredMDP, summarize_meter
from hrl_uav.experiment import boundary_reward, rollout
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

    def test_rollout_completes_deployment_boundaries_without_tail_metric_leak(self):
        cfg = config(True)
        cfg["training"]["window_macros"] = 1
        gamma = cfg["training"]["gamma_per_resource_frame"]
        mdp = QueueMDP(cfg)
        env = mdp.env
        frames, stored, decisions, routing_times = [], [], [], []
        prepared, window_ends, parents = {}, {}, []
        sac_update_times, ppo_update_times = [], []
        reset_states = []
        local_action = np.tile(2 * np.eye(3) - 1, (2, 1, 1)).ravel()
        real_reset, real_step = mdp.reset, mdp.step_frame
        real_prepare, real_route = mdp.prepare_service, env._route_arrived_parent_tasks

        def initialize(seed, macro_count):
            real_reset(seed, macro_count)
            # At 10 Mbit/s these initial queues finish at 10.1 and 20.2 s.
            parents.append(queued_parent(env, "high", [(0, 100.6, "compute")])[0])
            parents.append(queued_parent(env, "low", [(0, 100.8, "compute")])[0])
            reset_states.append(mdp.micro_observation().copy())
            return reset_states[-1].copy()

        def observe_frame(action, gamma_frame):
            start = env._current_time()
            result = real_step(action, gamma_frame)
            frames.append({
                "start": start, "result": copy.deepcopy(result),
                "energy": mdp.total_power_energy, "cost": mdp.cost_sum,
                "steps": mdp.integration_steps,
            })
            if env.global_micro_step in (100, 200):
                window_ends[env.global_micro_step] = mdp.metrics().copy()
            return result

        def prepare():
            state = real_prepare()
            prepared[round(env._current_time(), 6)] = state.copy()
            return state

        def choose(state, deterministic=False):
            self.assertEqual(env.macro_service_pause_steps_remaining, 0)
            decisions.append((env._current_time(), state.copy()))
            return local_action.copy()

        def store(state, action, route, reward, next_state, discount):
            stored.append((state.copy(), action.copy(), route.copy(), reward,
                           next_state.copy(), discount, env._current_time()))

        def route(*args, **kwargs):
            self.assertEqual(env.macro_service_pause_steps_remaining, 0)
            routing_times.append(env._current_time())
            return real_route(*args, **kwargs)

        # Only the scalar logging interface is stubbed; no torch/network objects.
        entropy = Mock()
        entropy.log_alpha.exp.return_value.item.return_value = 0.05
        agent = SimpleNamespace(
            stage="fixed-action-test", entropy=entropy, macro_rollout=[],
            sac_updates=0, sac_actor_updates=0, sac_critic_gradient_steps=0,
            sac_target_updates=0, ppo_updates=0, last_sac_stats={},
            macro_action=Mock(return_value=(np.full(6, 0.1), 0.0)),
            sac_actor=SimpleNamespace(choose_action=Mock(side_effect=choose)),
            store=Mock(side_effect=store),
            update_sac=Mock(side_effect=lambda: sac_update_times.append(env._current_time())),
            update_ppo=Mock(side_effect=lambda progress: ppo_update_times.append(env._current_time())),
        )
        with patch("hrl_uav.experiment.QueueMDP", return_value=mdp), \
             patch("hrl_uav.experiment.DirectHRL", side_effect=AssertionError("no learning networks")), \
             patch.object(mdp, "reset", side_effect=initialize) as reset_call, \
             patch.object(mdp, "step_frame", side_effect=observe_frame), \
             patch.object(mdp, "prepare_service", side_effect=prepare), \
             patch.object(env, "_generate_traffic", return_value={}), \
             patch.object(env, "_route_arrived_parent_tasks", side_effect=route):
            summary, rows, trajectory, routes, returned_mdp = rollout(
                cfg, agent, 51, 2, macro_enabled=True, train=True, complete_tail=True,
            )

        self.assertIs(returned_mdp, mdp)
        reset_call.assert_called_once_with(51, 2)
        self.assertTrue(env.compute_during_deployment)
        self.assertAlmostEqual(parents[0].finish_time, 10.1)
        self.assertAlmostEqual(parents[1].finish_time, 20.2)
        self.assertEqual(len(frames), 42)
        self.assertEqual(sum(f["result"]["phase"] == "deployment" for f in frames), 6)
        self.assertEqual(len(stored), 36)
        self.assertEqual(summary["boundary_transitions"], 2)
        self.assertEqual(summary["final_pending_dropped"], 0)
        self.assertAlmostEqual(summary["boundary_discount_min"], gamma**3)

        service = [f for f in frames if f["result"]["phase"] == "service"]
        for index, transition in enumerate(stored):
            np.testing.assert_array_equal(transition[0], decisions[index][1])
            np.testing.assert_array_equal(transition[1], local_action)
            np.testing.assert_array_equal(transition[2], service[index]["result"]["route"])
            if index not in (17, 35):
                self.assertAlmostEqual(transition[5], gamma)
                self.assertAlmostEqual(transition[3], service[index]["result"]["reward"])
        # These are real rollout stores, not standalone boundary_reward calls.
        for index, last_frame, next_time, low_queue, low_count in (
            (17, 19, 11.0, 91.4, 1), (35, 39, 21.0, 0.0, 0),
        ):
            transition = stored[index]
            deployment = frames[last_frame + 1:last_frame + 3]
            self.assertEqual([f["result"]["phase"] for f in deployment], ["deployment"] * 2)
            expected_reward = (frames[last_frame]["result"]["reward"]
                               + gamma * deployment[0]["result"]["reward"]
                               + gamma**2 * deployment[1]["result"]["reward"])
            self.assertAlmostEqual(transition[3], expected_reward)
            self.assertAlmostEqual(transition[5], gamma**3)
            self.assertAlmostEqual(transition[6], next_time)
            self.assertEqual(sum(np.isclose(t[6], next_time) for t in stored), 1)
            np.testing.assert_array_equal(transition[4], prepared[next_time])
            self.assertFalse(np.array_equal(transition[4], reset_states[0]))
            for feature, expected in (
                ("compute_queue", [0, 0, 0, low_queue, 0, 0]),
                ("unfinished_tasks", [0, 0, 0, low_count, 0, 0]),
            ):
                start, end = mdp.micro_feature_slices[feature]
                np.testing.assert_allclose(np.expm1(transition[4][start:end]) * 10,
                                           expected, atol=2e-5)

        # The extra deployment belongs to replay continuation, not either window.
        self.assertEqual(len(rows), 2)
        self.assertEqual(trajectory, [])
        self.assertEqual(routes, [])
        for row, end_step in zip(rows, (100, 200)):
            for key, value in window_ends[end_step].items():
                self.assertEqual(row[key], value, key)
        for key, value in window_ends[200].items():
            self.assertEqual(summary[key], value, key)
        self.assertEqual(summary["time_s"], 20.0)
        self.assertEqual(summary["task_completed_high_count"], 1)
        self.assertEqual(summary["task_completed_low_count"], 0)
        self.assertAlmostEqual(summary["task_delay_weighted_ms"], 10100)
        self.assertAlmostEqual(summary["window_delay_ms"], 10100)
        self.assertEqual(summary["window_low_ms"], 0)
        self.assertAlmostEqual(summary["queue_compute_mbit"], 1.4)
        self.assertEqual(summary["unfinished"], 1)
        prefix_energy = sum(f["result"]["infos"][k]["total_energy"]
                            for f in frames[:40] for k in range(5))
        self.assertAlmostEqual(prefix_energy, 2000.0)
        self.assertAlmostEqual(summary["power_w"] * 20, frames[39]["energy"])
        self.assertAlmostEqual(summary["J"] * 200, frames[39]["cost"])
        self.assertAlmostEqual(summary["window_power_w"] * 10,
                               frames[39]["energy"] - frames[19]["energy"])
        self.assertAlmostEqual(summary["window_J"] * 100,
                               frames[39]["cost"] - frames[19]["cost"])
        tail_energy = sum(info["total_energy"] for f in frames[40:]
                          for info in f["result"]["infos"])
        self.assertAlmostEqual(tail_energy, 10.64)
        self.assertGreater(mdp.total_power_energy, frames[39]["energy"])
        self.assertGreater(mdp.cost_sum, frames[39]["cost"])
        self.assertEqual(mdp.integration_steps, 210)
        self.assertEqual(env._current_time(), 21.0)
        self.assertEqual(env.completed_task_count, 2)
        self.assertAlmostEqual(mdp.metrics()["task_delay_low_ms"], 20200)
        self.assertEqual(mdp.metrics()["unfinished"], 0)

        np.testing.assert_allclose([t for t, _ in decisions],
                                   [1 + i * .5 for i in range(18)]
                                   + [11 + i * .5 for i in range(18)])
        np.testing.assert_allclose(routing_times,
                                   [1 + i * .1 for i in range(90)]
                                   + [11 + i * .1 for i in range(90)])
        np.testing.assert_allclose(sac_update_times, [12.0])
        np.testing.assert_allclose(ppo_update_times, [10.0, 20.0])
        self.assertEqual(agent.update_sac.call_count, 36 // cfg["training"]["sac_update_frames"])
        self.assertEqual(agent.update_ppo.call_count, 2)
        self.assertEqual(agent.macro_action.call_count, 3)  # includes tail bootstrap only
        self.assertEqual(len(agent.macro_rollout), 2)
        for row in rows:
            for key in ("sac_critic_updates", "sac_actor_updates", "sac_critic_gradient_steps",
                        "sac_target_updates", "ppo_updates"):
                self.assertEqual(row[key], 0)


if __name__ == "__main__":
    unittest.main()
