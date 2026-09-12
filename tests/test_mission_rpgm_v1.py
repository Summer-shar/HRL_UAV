"""Short deterministic functionality tests; no neural learning or long evaluation."""
import copy
import json
import math
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from hrl_uav.bandwidth import MeteredMDP, UniformRouting
from hrl_uav.env import UAV_MEC_Env
from hrl_uav.experiment import rollout
from hrl_uav.mdp import QueueMDP
from hrl_uav.mission_smoke import (
    BASE, CONTROLLERS, ROOT, MissionPolicy, load_spec, mission_config, paired_records, run_controller,
)
from hrl_uav.scene_prescreen import assert_crn_equal
from tests import test_deployment_compute as deployment_tests
from tests import test_wireless_model as legacy_tests

queued_parent = deployment_tests.queued_parent


def config():
    base = json.loads((ROOT / "configs/sac_only_wireless6_pilot.json").read_text(encoding="utf-8"))
    return mission_config(base)


def make_mdp():
    mdp = QueueMDP(config())
    mdp.reset(33600, 3)
    return mdp


def scalar_power(v):
    return 78 * (1 + 3 * v * v / 110**2) + 22 * math.sqrt(math.sqrt(265 + v**4 / 4) - v*v/2) + .0092*v**3 + 28


class MissionTests(unittest.TestCase):
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
            self.assertEqual(left.dtype, right.dtype)
            self.assertEqual(left.shape, right.shape)
            self.assertEqual(left.tobytes(), right.tobytes())
        elif isinstance(left, (float, np.floating)):
            self.assertEqual(struct.pack("d", left), struct.pack("d", right))
        else:
            self.assertEqual(left, right)

    def test_legacy_bitwise_trace_against_exact_pre_extension_source(self):
        with patch.object(legacy_tests, "BASE", BASE):
            old_env = legacy_tests.baseline_module("env")
            old_mdp = legacy_tests.baseline_module("mdp", {"hrl_uav.env": old_env})
            old_meter = legacy_tests.baseline_module("bandwidth", {"hrl_uav.mdp": old_mdp})

        def trace(cls, cfg):
            mdp = cls(cfg, record_links=True)
            mdp.reset(33600, 3)
            queued_parent(mdp.env, "high", [(0, .7, "compute"), (1, .4, "tx")])
            frames, macros = [], []
            for action in (np.zeros(6), np.array([.2, -.1, -.3, .1, .1, .2])):
                macros.append(mdp.start_macro(action))
                for i in range(20):
                    frames.append(copy.deepcopy((mdp.step_frame(np.sin(np.arange(18)+i)*.5),
                                                  mdp.env.get_raw_state(), mdp.metrics(),
                                                  mdp.micro_observation(), mdp.macro_observation())))
                mdp.advance_macro_clock()
            return macros, frames, mdp.readings, mdp.link_readings, mdp.uav_readings, mdp.env.np_random.get_state()

        for wireless in ("legacy_fixed_noise", "psd_per_uav_budget"):
            for cpu in (False, True):
                cfg = json.loads((ROOT / "configs/sac_only_wireless6_pilot.json").read_text())
                cfg["environment"].update(wireless_model=wireless, compute_during_deployment=cpu)
                with self.subTest(wireless=wireless, cpu=cpu):
                    expected = trace(old_meter.MeteredMDP, cfg)
                    self.assert_nested_equal(expected, trace(MeteredMDP, cfg))
                    cfg["environment"]["motion_protocol"] = "legacy_fly_hover"
                    self.assert_nested_equal(expected, trace(MeteredMDP, cfg))

    def test_mode_validation_and_layout(self):
        cfg = config()
        mdp = make_mdp()
        self.assertEqual(mdp.observation_spec()["micro"]["dim"], 68)
        self.assertEqual(mdp.observation_spec()["macro"]["dim"], 67)
        self.assertEqual((mdp.env.deployment_resource_frames, mdp.env.service_resource_frames), (0, 20))
        for opts in ({"motion_protocol": "bad"}, {"deployment_dt": 1},
                     {"pause_service_during_flight": True}, {"users_static": True},
                     {"deployment_transient_energy_mode": "kinetic_lower_bound"},
                     {"mission_duration_s": float("nan")}, {"mission_duration_s": 0},
                     {"mission_routes_m": [[[0, 0], [1, 1]]]}, {"resource_frame_dt": .2}):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                UAV_MEC_Env(**dict(cfg["environment"], **opts))
        for bad in (np.full(6, np.nan), np.ones(5), np.full(6, np.inf)):
            with self.assertRaises(ValueError):
                mdp.start_macro(bad)
        mdp.env.global_micro_step = 1
        with self.assertRaisesRegex(ValueError, "boundary"):
            mdp.start_macro(np.zeros(6))

    def test_displacement_safety_zero_small_action_and_energy(self):
        for action in (np.zeros(6), np.full(6, .01), np.array([1, 1, -1, 1, 1, -1])):
            mdp = make_mdp()
            env = mdp.env
            before = env.uav_coords.copy()
            info = mdp.start_macro(action)
            np.testing.assert_array_equal(env.uav_coords, before)
            delta = info["actual_displacement_m"]
            self.assertTrue(np.all(np.linalg.norm(delta, axis=1) <= 20 + 1e-10))
            expected = np.array([scalar_power(np.linalg.norm(d)/10)*10 for d in delta])
            np.testing.assert_allclose(info["E_macro_per_uav"], expected, rtol=0, atol=1e-10)
            self.assertAlmostEqual(info["E_propulsion_increment"], float(expected.sum()-3*scalar_power(0)*10))
            for frame_index in range(20):
                mdp.step_frame(np.zeros(18))
                np.testing.assert_allclose(env.uav_coords, before+(frame_index+1)/20*delta, atol=1e-8, rtol=0)
            self.assertEqual(env.macro_fly_steps_remaining, 0)
            if not np.any(action):
                np.testing.assert_array_equal(env.uav_coords, before)
                self.assertEqual(info["E_propulsion_increment"], 0)
            else:
                self.assertTrue(np.any(env.uav_coords != before))
                self.assertLess(info["E_propulsion_increment"], 0)
        mdp = make_mdp()
        mdp.env.uav_coords[:] = [[400, 400], [410, 400], [700, 700]]
        info = mdp.start_macro([1, 0, -1, 0, 0, 0])
        self.assertLess(info["separation_projection_scale"], 1)
        self.assertGreaterEqual(info["minimum_flight_separation_m"], 5-1e-8)
        with self.assertRaisesRegex(ValueError, "unfinished"):
            mdp.start_macro(np.zeros(6))

    def test_physical_step_geometry_and_task_pipeline(self):
        mdp = make_mdp()
        env = mdp.env
        remote, _ = queued_parent(env, "low", [(1, .001, "tx")])
        uploaded = env._create_parent_task(0, "high", .001, source_uav=0)
        mdp.start_macro([1, 0, 0, 0, 0, 0])
        start_uav, start_users = env.uav_coords.copy(), env.user_coords.copy()
        uplink, resources = env.compute_u2g_rate_bps, env.allocate_u2u_resources
        seen = []

        def check_uplink(*args, **kwargs):
            np.testing.assert_array_equal(env.uav_coords, start_uav)
            np.testing.assert_array_equal(env.user_coords, start_users)
            seen.append("uplink")
            return uplink(*args, **kwargs)

        def check_u2u(*args, **kwargs):
            np.testing.assert_array_equal(env.uav_coords, start_uav)
            np.testing.assert_array_equal(env.user_coords, start_users)
            seen.append("u2u")
            return resources(*args, **kwargs)

        with patch.object(env, "_generate_traffic", return_value={}), \
             patch.object(env, "compute_u2g_rate_bps", side_effect=check_uplink), \
             patch.object(env, "allocate_u2u_resources", side_effect=check_u2u):
            _, first = env.micro_step(np.eye(3), np.eye(3))
        self.assertIn("uplink", seen)
        self.assertIn("u2u", seen)
        self.assertGreater(first["energy_uplink"], 0)
        self.assertGreater(first["energy_trans"], 0)
        self.assertEqual(uploaded.total_chunks, 0)
        self.assertEqual(uploaded.remaining_upload_mbits, 0)
        self.assertIsNone(uploaded.finish_time)
        self.assertIsNone(remote.finish_time)
        self.assertEqual(env.Q_tx_low.sum(), 0)
        self.assertAlmostEqual(env.Q_loc_low[1], .001)
        self.assertAlmostEqual(env.uav_coords[0, 0]-start_uav[0, 0], .2)
        self.assertAlmostEqual(first["task_holding_cost_weighted"], 1.1)
        _, second = env.micro_step(np.eye(3), np.eye(3))
        self.assertAlmostEqual(uploaded.finish_time, .2)
        self.assertAlmostEqual(remote.finish_time, .2)
        self.assertEqual(second["task_holding_cost_weighted"], 0)
        self.assertEqual(env.completed_priority_count, {"high": 1, "low": 1})

    def test_routes_rng_clock_and_readonly_rule_smoke(self):
        sac = SimpleNamespace(sac_actor=UniformRouting(), optimizers=(), assert_unchanged=lambda: {"fixed_test_double": True})
        reference, meters = None, []
        for name in CONTROLLERS:
            row, meter = run_controller(config(), sac, name, seconds=20)
            records = paired_records(meter)
            if reference is None:
                reference = records
            assert_crn_equal(reference, records)
            self.assertEqual(row["sac_decisions"], 40)
            self.assertEqual(row["replay_writes"], 0)
            for record in meter.readings:
                self.assertEqual(record["service_steps"], 5)
                self.assertEqual(record["compute_steps"], 5)
            starts = np.array([[350, 350], [350, 650], [650, 500]])
            ends = np.array([[620, 430], [620, 650], [900, 400]])
            np.testing.assert_allclose(meter.env.mobility_group_centers, starts+(ends-starts)*20/900, atol=1e-10)
            clocks = np.asarray(meter.micro_states)[:, -1]
            np.testing.assert_allclose(clocks[:20], np.arange(20)*.05, atol=1e-7)
            meters.append(meter)
        # Retaining discarded group RNG draws preserves the legacy arrival sequence.
        old_cfg = json.loads((ROOT / "configs/sac_only_wireless6_pilot.json").read_text())
        old = QueueMDP(old_cfg)
        old.reset(33600, 2)
        for _ in range(2):
            old.start_macro(np.zeros(6))
            for _ in range(20):
                old.step_frame(np.zeros(18))
            old.advance_macro_clock()
        self.assert_nested_equal(old.env.np_random.get_state(), meters[0].env.np_random.get_state())
        self.assertEqual(old.env.arrived_priority_count, meters[0].env.arrived_priority_count)

    def test_blind_patrol_cannot_read_business_and_bounds_guard(self):
        policy = MissionPolicy(SimpleNamespace(sac_actor=None), "blind_patrol")
        # No meter at all: accidental business/position access fails immediately.
        actions = [policy.macro_action(object(), deterministic=True)[0].reshape(3, 2) for _ in range(6)]
        for i, action in enumerate(actions):
            np.testing.assert_array_equal(action, np.tile([1 if i < 3 else -1, 0], (3, 1)))
        with self.assertRaises(ValueError):
            run_controller(config(), None, "hover", seed=33700)
        with self.assertRaises(ValueError):
            run_controller(config(), None, "hover", seconds=900)

    def test_rollout_no_gap_replay_macro_rewards_and_tail_conservation(self):
        cfg = config()
        cfg["training"]["window_macros"] = 1
        # Scalar progress denominator for the no-network PPO logging stub only.
        cfg["training"]["joint_windows"] = 2
        mdp = QueueMDP(cfg)
        env = mdp.env
        gamma = cfg["training"]["gamma_per_resource_frame"]
        frames, stores, decisions, parents, prepared = [], [], [], [], {}
        reset, step, prepare = mdp.reset, mdp.step_frame, mdp.prepare_service

        def initialize(seed, macros):
            reset(seed, macros)
            parents.append(queued_parent(env, "high", [(0, 100.6, "compute")])[0])
            parents.append(queued_parent(env, "low", [(0, 100.8, "compute")])[0])
            return mdp.micro_observation()

        def record_step(action, gamma_frame):
            frame = step(action, gamma_frame)
            frames.append(copy.deepcopy(frame))
            return frame

        def record_prepare():
            s = prepare()
            prepared[round(env._current_time(), 6)] = s.copy()
            return s

        def arrivals(**kwargs):
            if env.global_micro_step == 200 and not env._resource_frame_prepared:
                env._create_parent_task(0, "low", .2, source_uav=0)
            return {}

        def choose(state, deterministic=False):
            decisions.append(state.copy())
            return np.zeros(18)

        entropy = Mock()
        entropy.log_alpha.exp.return_value.item.return_value = .05
        agent = SimpleNamespace(stage="test", entropy=entropy, macro_rollout=[], last_sac_stats={},
                                sac_updates=0, sac_actor_updates=0, sac_critic_gradient_steps=0,
                                sac_target_updates=0, ppo_updates=0,
                                macro_action=Mock(return_value=(np.zeros(6), 0)),
                                sac_actor=SimpleNamespace(choose_action=choose),
                                store=lambda *args: stores.append(copy.deepcopy(args)),
                                update_sac=Mock(), update_ppo=Mock())
        with patch("hrl_uav.experiment.QueueMDP", return_value=mdp), \
             patch("hrl_uav.experiment.DirectHRL", side_effect=AssertionError("No networks")), \
             patch.object(mdp, "reset", side_effect=initialize), \
             patch.object(mdp, "step_frame", side_effect=record_step), \
             patch.object(mdp, "prepare_service", side_effect=record_prepare), \
             patch.object(env, "_generate_traffic", side_effect=arrivals):
            summary, rows, _, _, _ = rollout(cfg, agent, 33600, 2, train=True, macro_enabled=True, complete_tail=True)
        self.assertEqual(len(frames), 40)
        self.assertEqual(len(stores), 40)
        self.assertEqual(len(decisions), 40)
        self.assertEqual(summary["boundary_transitions"], 2)
        self.assertEqual(summary["final_pending_dropped"], 0)
        self.assertEqual(summary["boundary_discount_min"], gamma)
        self.assertEqual(env._current_time(), 20)
        self.assertEqual(mdp.integration_steps, 200)
        self.assertAlmostEqual(parents[0].finish_time, 10.1)
        self.assertIsNone(parents[1].finish_time)
        self.assertAlmostEqual(env.Q_loc_low[0], 1.4, places=8)
        self.assertEqual(summary["unfinished"], 1)
        self.assertEqual(mdp.metrics()["unfinished"], 2)
        for i, transition in enumerate(stores):
            np.testing.assert_array_equal(transition[0], decisions[i])
            np.testing.assert_array_equal(transition[4], prepared[round((i+1)*.5, 6)])
            self.assertAlmostEqual(transition[3], frames[i]["reward"])
            self.assertEqual(transition[5], gamma)
        self.assertEqual(agent.update_sac.call_count, 2)
        self.assertEqual(agent.update_ppo.call_count, 2)
        for i, item in enumerate(agent.macro_rollout):
            expected = sum(gamma**j * frames[20*i+j]["reward"] for j in range(20)) / 20
            self.assertAlmostEqual(item[3], expected)
        self.assertEqual([r["time_s"] for r in rows], [10, 20])
        self.assertAlmostEqual(summary["power_w"] * 20, mdp.total_power_energy)


if __name__ == "__main__":
    unittest.main()
