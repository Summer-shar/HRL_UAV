"""Short functional checks only: no learning networks, weights or evaluation jobs."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from hrl_uav.bandwidth import MeteredMDP, UniformRouting, summarize_links, summarize_meter
from hrl_uav.env import UAV_MEC_Env
from hrl_uav.experiment import rollout
from hrl_uav.mdp import QueueMDP
from hrl_uav.scene_prescreen import PairedMeter, assert_crn_equal
from tests.test_deployment_compute import queued_parent


ROOT = Path(__file__).resolve().parents[1]
BASE = "07a727d8189be4f73c0218fca47603d8cbf6956b"
PSD = {
    "wireless_model": "psd_per_uav_budget", "noise_psd_w_per_hz": 1.95e-19,
    "u2u_tx_budget_w": 1.0, "u2u_reference_bandwidth_hz": 20e6 / 39,
    "u2u_reference_power_w": 1.0,
}
GEOMETRY = np.array([[100., 100.], [400., 100.], [100., 400.]])


def config(candidate=True, cpu=False):
    result = json.loads((ROOT / "configs/hrl_ppo_only_unlock.json").read_text())
    result["environment"]["compute_during_deployment"] = cpu
    if candidate:
        result["environment"].update(PSD)
    return result


def scalar_link(distance, total_bandwidth, links, outgoing):
    """Independent scalar specification, not a call to the allocation under test."""
    bandwidth = total_bandwidth / links
    power = 1.0 / outgoing
    noise = 1.95e-19 * bandwidth
    rate = bandwidth * math.log2(1 + power * (1e-6 / distance**2) / noise)
    return bandwidth, power, noise, rate


def baseline_module(filename, imports=None):
    """Execute the pinned pre-change source in memory; never checkout/write files."""
    name = "hrl_uav._wireless_baseline_" + filename
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:hrl_uav/{filename}.py"], cwd=ROOT).decode("utf-8")
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(ROOT / "hrl_uav" / (filename + ".py"))
    with patch.dict(sys.modules, {name: module, **(imports or {})}):
        exec(compile(source, f"{BASE}:hrl_uav/{filename}.py", "exec"), module.__dict__)
    return module


class TraceMeter(PairedMeter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.frames = []
        self.propulsion_deltas = []
        self.propulsion_shares = []
        self.env._route_arrived_parent_tasks = Mock(wraps=self.env._route_arrived_parent_tasks)
        self.env.allocate_u2u_resources = Mock(wraps=self.env.allocate_u2u_resources)

    def step_frame(self, action, gamma_frame=0.99):
        frame = super().step_frame(action, gamma_frame)
        self.frames.append(frame)
        self.propulsion_deltas.append(self.propulsion_delta_share)
        self.propulsion_shares.append(self.propulsion_share)
        return frame


class WirelessModelTests(unittest.TestCase):
    def assert_nested_equal(self, left, right):
        if isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            self.assertEqual(left, right)

    def test_legacy_short_regression_against_pinned_source_both_cpu_modes(self):
        old_env = baseline_module("env")
        old_mdp = baseline_module("mdp", {"hrl_uav.env": old_env})
        old_meter = baseline_module("bandwidth", {"hrl_uav.mdp": old_mdp})

        def trace(cls, cfg):
            mdp = cls(cfg, record_links=True)
            mdp.reset(33500, 3)
            queued_parent(mdp.env, "low", [(0, 2, "compute"), (1, 0.2, "tx")])
            frames = []
            for macro in range(2):
                mdp.start_macro(np.zeros(6))
                for _ in range(20):
                    frames.append(copy.deepcopy((mdp.step_frame(np.zeros(18)),
                                                  mdp.env.get_raw_state(), mdp.metrics(),
                                                  mdp.micro_observation(), mdp.macro_observation())))
                mdp.advance_macro_clock()
            return (frames, mdp.readings, mdp.link_readings, mdp.uav_readings,
                    summarize_meter(mdp.readings, 20),
                    summarize_links(mdp.link_readings, 20, 33500, 6))

        for cpu in (False, True):
            with self.subTest(cpu=cpu):
                cfg = config(False, cpu)
                before = trace(old_meter.MeteredMDP, cfg)
                self.assert_nested_equal(before, trace(MeteredMDP, cfg))
                cfg["environment"]["wireless_model"] = "legacy_fixed_noise"
                self.assert_nested_equal(before, trace(MeteredMDP, cfg))

    def test_mode_validation_and_legacy_custom_bandwidth(self):
        for mode in (None, "unknown", False, 1, ["psd_per_uav_budget"]):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "wireless_model"):
                UAV_MEC_Env(wireless_model=mode)
        for name in PSD.keys() - {"wireless_model"}:
            for value in (0, -1, float("nan"), float("inf"), -float("inf"), True, "1", [1], None):
                options = {**PSD, name: value}
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    UAV_MEC_Env(**options)
        for name in ("uplink_bandwidth_mhz", "u2u_bandwidth_mhz", "p_gt_power", "beta_0"):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    UAV_MEC_Env(**PSD, **{name: value})
        for options in ({"u2u_bandwidth_mhz": 1e308}, {"noise_psd_w_per_hz": 1e308},
                        {"noise_psd_w_per_hz": 1e-320, "u2u_reference_bandwidth_hz": 1e-320}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                UAV_MEC_Env(**{**PSD, **options})
        matrix = (~np.eye(3, dtype=bool)) * 0.5
        with self.assertRaisesRegex(ValueError, "does not support"):
            UAV_MEC_Env(**PSD, u2u_link_bandwidth_matrix_mhz=matrix)
        implicit = UAV_MEC_Env(u2u_link_bandwidth_matrix_mhz=matrix)
        explicit = UAV_MEC_Env(wireless_model="legacy_fixed_noise", u2u_link_bandwidth_matrix_mhz=matrix)
        for mask in (np.zeros((3, 3), bool), ~np.eye(3, dtype=bool)):
            self.assert_nested_equal(implicit.compute_u2u_rate_bps(GEOMETRY, mask, True),
                                     explicit.compute_u2u_rate_bps(GEOMETRY, mask, True))
        candidate = UAV_MEC_Env(**PSD)
        self.assertFalse(candidate.compute_during_deployment)
        self.assertEqual(candidate.noise_power, 1e-13)
        self.assertEqual(candidate.p_u2u, 1.0)

    def test_scalar_allocations_and_refresh_at_fixed_geometry(self):
        env = UAV_MEC_Env(**PSD, u2u_bandwidth_mhz=3)
        env.reset(7)
        env.uav_coords[:] = GEOMETRY
        self.assertEqual(env.last_u2u_link_bandwidth_matrix_hz.sum(), 0)
        reference = env.compute_u2u_reference_quality()
        masks = [np.zeros((3, 3), bool)]
        one = np.zeros((3, 3), bool)
        one[0, 1] = True
        two = one.copy()
        two[0, 2] = True
        masks.extend([one, two, ~np.eye(3, dtype=bool), one])
        for total in (3e6, 6e6):
            env.B_u2u_total = total
            for mask in masks:
                with self.subTest(total=total, links=int(mask.sum())):
                    resources = env.allocate_u2u_resources(mask)
                    self.assertLessEqual(resources["u2u_link_bandwidth_hz"].sum(), total + 1e-8)
                    self.assertTrue(np.all(resources["u2u_link_power_w"].sum(axis=1) <= 1.0))
                    for i in range(3):
                        for j in range(3):
                            expected = scalar_link(math.dist(GEOMETRY[i], GEOMETRY[j]), total,
                                                   int(mask.sum()), int(mask[i].sum())) if mask[i, j] else (0, 0, 0, 0)
                            for name, value in zip(("u2u_link_bandwidth_hz", "u2u_link_power_w",
                                                    "u2u_link_noise_w", "u2u_rate_bps"), expected):
                                self.assertTrue(math.isclose(resources[name][i, j], value, rel_tol=1e-12, abs_tol=1e-25),
                                                (name, i, j, resources[name][i, j], value))
                    np.testing.assert_array_equal(env.compute_u2u_reference_quality(), reference)
                    np.testing.assert_array_equal(env.compute_u2u_rate_bps(active_link_mask=mask), resources["u2u_rate_bps"])
        # Self links never receive resources, even if requested in a query.
        for value in env.allocate_u2u_resources(np.eye(3, dtype=bool)).values():
            np.testing.assert_array_equal(value, np.zeros((3, 3)))
        # A changed geometry must also invalidate position-only reference results.
        env.uav_coords[1, 0] += 50
        self.assertNotEqual(reference[0, 1], env.compute_u2u_reference_quality()[0, 1])

    def test_uplink_reference_and_fixed_psd_when_user_count_or_bandwidth_changes(self):
        for users, bandwidth in ((39, 20), (39, 10), (42, 20)):
            new = UAV_MEC_Env(**PSD, num_users=users, uplink_bandwidth_mhz=bandwidth)
            old = UAV_MEC_Env(num_users=users, uplink_bandwidth_mhz=bandwidth)
            for env in (new, old):
                env.reset(7)
            actual = new.compute_u2g_rate_bps()
            bw = bandwidth * 1e6 / users
            expected = []
            for user, source in zip(new.user_coords, new.user_allocations):
                distance_sq = math.dist(user, new.uav_coords[source])**2 + new.H**2
                expected.append(bw * math.log2(1 + 0.1 * 1e-6 / distance_sq / (1.95e-19 * bw)))
            np.testing.assert_allclose(actual, expected, rtol=1e-12)
            self.assertEqual(new.noise_psd_w_per_hz, 1.95e-19)
            if (users, bandwidth) == (39, 20):
                np.testing.assert_allclose(actual, old.compute_u2g_rate_bps(), rtol=1e-12)
            else:
                self.assertFalse(np.allclose(actual, old.compute_u2g_rate_bps()))

    def test_backlog_priority_shared_budget_and_no_intra_step_reallocation(self):
        mdp = QueueMDP(config())
        env = mdp.env
        env.reset(7)
        env.uav_coords[:] = GEOMETRY
        env.B_u2u_total = 3e6
        capacity = scalar_link(300, 3e6, 2, 2)[3] * 0.1 / 1e6
        queued_parent(env, "high", [(1, capacity * 0.4, "tx"), (2, capacity * 0.05, "tx")])
        queued_parent(env, "low", [(1, capacity * 3, "tx")])
        with patch.object(env, "_generate_traffic", return_value={}):
            # Current action is local-only; old transfer backlog must still be active.
            _, info = env.micro_step(np.eye(3), np.eye(3))
            self.assertAlmostEqual(info["tx_service_matrix_mbits"][0, 1], capacity, places=12)
            self.assertAlmostEqual(env.Q_tx_low[0, 1], capacity * 2.4, places=12)
            self.assertEqual(env.Q_tx_high.sum(), 0)
            np.testing.assert_allclose(info["u2u_link_power_w"][0], [0, .5, .5])
            self.assertAlmostEqual(info["u2u_link_energy_j"][0, 1], .05, places=12)
            self.assertAlmostEqual(info["u2u_link_energy_j"][0, 2], .0025, places=12)
            self.assertAlmostEqual(info["energy_trans"], .0525, places=12)
            self.assertTrue(np.all(info["energy_trans_per_uav"] <= .1 + 1e-12))
            _, next_info = env.micro_step(np.eye(3), np.eye(3))
            np.testing.assert_allclose(next_info["u2u_link_power_w"][0], [0, 1, 0])
            np.testing.assert_allclose(next_info["u2u_link_bandwidth_hz"][0], [0, 3e6, 0])

    def test_short_send_energy_and_full_fragment_admission_before_parent_completion(self):
        mdp = QueueMDP(config())
        env = mdp.env
        env.reset(7)
        env.uav_coords[:] = GEOMETRY
        env.B_u2u_total = 3e6
        rate = scalar_link(300, 3e6, 1, 1)[3]
        capacity = rate * .1 / 1e6
        parent, chunks = queued_parent(env, "low", [(0, .01, "compute"), (1, 1.2 * capacity, "tx")])
        with patch.object(env, "_generate_traffic", return_value={}):
            _, first = env.micro_step(np.eye(3), np.eye(3))
            self.assertAlmostEqual(first["tx_service_matrix_mbits"][0, 1], capacity, places=12)
            self.assertEqual(env.Q_loc_low[1], 0)
            self.assertIsNone(parent.finish_time)
            self.assertEqual(parent.completed_chunks, 1)
            _, second = env.micro_step(np.eye(3), np.eye(3))
            self.assertAlmostEqual(second["tx_service_matrix_mbits"][0, 1], .2 * capacity, places=12)
            self.assertAlmostEqual(second["u2u_active_time_s"][0, 1], .02, places=12)
            self.assertAlmostEqual(second["energy_trans"], .02, places=12)
            self.assertAlmostEqual(env.Q_loc_low[1], 1.2 * capacity, places=12)
            self.assertIsNone(parent.finish_time)
            self.assertAlmostEqual(chunks[1].remaining_mbits, chunks[1].initial_mbits)
            for _ in range(5):
                _, info = env.micro_step(np.eye(3), np.eye(3))
                self.assertEqual(info["energy_trans"], 0)
                self.assertEqual(info["u2u_active_time_s"].sum(), 0)
            self.assertIsNotNone(parent.finish_time)
            self.assertEqual(parent.completed_chunks, 2)
            self.assertEqual(env.completed_task_count, 1)

    def test_real_rollout_meter_energy_observations_pause_and_crn(self):
        for cpu in (False, True):
            cfg = config(True, cpu)
            agent = UniformRouting()
            agent.choose_action = Mock(wraps=agent.choose_action)
            agent.macro_action = Mock(wraps=agent.macro_action)
            with patch("hrl_uav.experiment.QueueMDP", TraceMeter):
                candidate = rollout(cfg, agent, 33500, 3, macro_enabled=False, record=True)
                legacy = rollout(config(False, cpu), UniformRouting(), 33500, 3, macro_enabled=False, record=True)
            meter = candidate[-1]
            assert_crn_equal(meter.crn_frames, legacy[-1].crn_frames)
            self.assertEqual(agent.choose_action.call_count, 54)
            self.assertEqual(agent.macro_action.call_count, 3)
            self.assertEqual(meter.env._route_arrived_parent_tasks.call_count, 270)
            self.assertEqual(meter.env.allocate_u2u_resources.call_count, 270)
            self.assertEqual(meter.observation_spec(), legacy[-1].observation_spec())
            self.assertEqual(meter.observation_spec()["macro"]["dim"], 67)
            self.assertEqual(meter.observation_spec()["micro"]["dim"], 68)
            self.assertFalse(meter.observe_routing_ready)
            eta_slice = slice(*meter.micro_feature_slices["u2u_spectral_efficiency"])
            np.testing.assert_allclose(meter.micro_observation()[eta_slice],
                                       np.log1p(meter.env.compute_u2u_reference_quality()).ravel(), rtol=1e-7)
            infos = [info for frame in meter.frames for info in frame["infos"]]
            deploy = [info for info in infos if info["service_paused"]]
            self.assertEqual(len(deploy), 30)
            for info in deploy:
                for key in ("u2u_rate_bps", "u2u_link_bandwidth_hz", "u2u_link_power_w", "u2u_link_noise_w",
                            "u2u_active_time_s", "u2u_link_energy_j", "tx_service_matrix_mbits"):
                    self.assertEqual(float(info[key].sum()), 0)
                self.assertEqual(info["energy_trans"], 0)
            cpu_deploy = sum(info["energy_comp"] for info in deploy)
            self.assertGreater(cpu_deploy, 0) if cpu else self.assertEqual(cpu_deploy, 0)
            expected_energy = 0.0
            for frame, delta, share in zip(meter.frames, meter.propulsion_deltas, meter.propulsion_shares):
                costs = []
                for info in frame["infos"]:
                    p, n, b, r = (info[k] for k in ("u2u_link_power_w", "u2u_link_noise_w",
                                                   "u2u_link_bandwidth_hz", "u2u_rate_bps"))
                    self.assertLessEqual(b.sum(), meter.env.B_u2u_total + 1e-8)
                    self.assertTrue(np.all(p.sum(axis=1) <= 1 + 1e-12))
                    np.testing.assert_allclose(n, b * 1.95e-19, atol=0, rtol=1e-12)
                    x, t = info["tx_service_matrix_mbits"], info["u2u_active_time_s"]
                    np.testing.assert_allclose(x * 1e6, r * t, atol=1e-9, rtol=1e-12)
                    self.assertTrue(np.all(t <= .1 + 1e-9))
                    np.testing.assert_allclose(p * t, info["u2u_link_energy_j"], atol=0, rtol=1e-12)
                    np.testing.assert_allclose((p * t).sum(axis=1), info["energy_trans_per_uav"], atol=0, rtol=1e-12)
                    self.assertAlmostEqual((p * t).sum(), info["energy_trans"], places=12)
                    self.assertAlmostEqual(info["total_energy"], info["energy_comp"] + info["energy_uplink"] + info["energy_trans"], places=10)
                    reward_cfg = cfg["reward"]
                    costs.append(reward_cfg["delay_weight"] * info["task_holding_cost_weighted"] / reward_cfg["holding_norm_task_s"]
                                 + reward_cfg["energy_weight"] * (info["total_energy"] + delta) / reward_cfg["energy_norm_j"])
                    expected_energy += info["total_energy"] + share
                self.assertAlmostEqual(frame["cost"], sum(costs), places=12)
                gamma = cfg["training"]["gamma_per_resource_frame"] ** (1 / 5)
                self.assertAlmostEqual(frame["reward"], -sum(gamma**k * c for k, c in enumerate(costs)), places=12)
            self.assertAlmostEqual(meter.total_power_energy, expected_energy, places=9)
            self.assertAlmostEqual(meter.cost_sum, sum(f["cost"] for f in meter.frames), places=9)
            total = summarize_meter(meter.readings, 30)
            links = summarize_links(meter.link_readings, 30, 33500, 6)
            energy = sum(info["energy_trans"] for info in infos)
            self.assertAlmostEqual(total["tx_energy_j"], energy, places=10)
            self.assertAlmostEqual(sum(r["tx_energy_j"] for r in links), energy, places=10)
            self.assertAlmostEqual(sum(r["tx_energy_j"] for r in meter.uav_readings), energy, places=10)
            self.assertAlmostEqual(total["tx_served_mbit"], sum(info["tx_service_matrix_mbits"].sum() for info in infos), places=9)
            self.assertLess(max(abs(r["balance_error"]) for r in meter.link_readings), 1e-7)
            self.assertEqual(candidate[0]["time_s"], 30)
            plain = rollout(cfg, UniformRouting(), 33500, 3, macro_enabled=False, record=True)
            self.assert_nested_equal(candidate[:4], plain[:4])


if __name__ == "__main__":
    unittest.main()
