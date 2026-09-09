import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from hrl_uav.bandwidth import MeteredMDP, UniformRouting, bandwidth_config, cpu_config, summarize_meter, summarize_links
from hrl_uav.experiment import rollout


class ConstantAgent:
    def __init__(self):
        self.sac_actor = self

    def choose_action(self, state, deterministic=False):
        return np.zeros(18, dtype=np.float32)

    def macro_action(self, state, enabled, deterministic=False):
        return np.zeros(6, dtype=np.float32), 0.0


class BandwidthTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.cfg = json.loads((root/"configs/main.json").read_text())

    def test_single_configuration_change(self):
        original = copy.deepcopy(self.cfg)
        treatment = bandwidth_config(self.cfg, 6)
        self.assertEqual(self.cfg, original)
        self.assertEqual(treatment["environment"].pop("u2u_bandwidth_mhz"), 6)
        original["environment"].pop("u2u_bandwidth_mhz")
        self.assertEqual(treatment, original)

    def test_meter_is_read_only_and_counters_conserved(self):
        plain = rollout(self.cfg, ConstantAgent(), 32500, 2, macro_enabled=False, record=True)
        with patch("hrl_uav.experiment.QueueMDP", MeteredMDP):
            metered = rollout(self.cfg, ConstantAgent(), 32500, 2, macro_enabled=False, record=True)
        self.assertEqual(plain[0], metered[0])
        np.testing.assert_array_equal(np.asarray(plain[3][0][1]), np.asarray(metered[3][0][1]))
        rows = metered[-1].readings
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(row["service_steps"] for row in rows), 180)
        summary = summarize_meter(rows, 20)
        self.assertGreater(summary["tx_served_mbit"], 0)
        self.assertLessEqual(summary["allocated_tx_capacity_utilization"], 1.00001)
        self.assertLessEqual(summary["mean_active_links_service"], 6)
        detail = metered[-1].uav_readings
        self.assertEqual(len(detail), 120)
        self.assertLess(max(abs(row["compute_balance_error"]) for row in detail), 1e-7)
        self.assertAlmostEqual(sum(row["computed_mbit"] for row in detail),
                               sum(row["compute_service_mbit"] for row in rows), places=7)
        self.assertEqual({row["role"] for row in detail}, {"heavy", "medium", "light"})

    def test_cpu_changes_only_capacity_and_observation_remains_compatible(self):
        original = bandwidth_config(self.cfg, 6)
        changed = cpu_config(original, 12)
        self.assertEqual(original["cpu_ghz"], 10)
        self.assertEqual(changed.pop("cpu_ghz"), 12)
        expected = copy.deepcopy(original)
        expected.pop("cpu_ghz")
        self.assertEqual(changed, expected)
        env = MeteredMDP(cpu_config(original, 12))
        state = env.reset(32500, 2)
        np.testing.assert_array_equal(env.env.F_max_array, np.full(3, 12e9))
        reference = MeteredMDP(original)
        self.assertEqual(state.shape, reference.reset(32500, 2).shape)
        self.assertEqual(env.observation_spec()["macro"]["dim"],
                         reference.observation_spec()["macro"]["dim"])

    def test_uniform_rule_is_zero_score_simplex_not_a_teacher(self):
        env = MeteredMDP(self.cfg)
        state = env.reset(32500,2)
        agent = UniformRouting()
        action = agent.choose_action(state,True)
        np.testing.assert_allclose(env.route(action), np.full((2,3,3),1/3))
        np.testing.assert_array_equal(action,np.zeros(18))
        with self.assertRaises(ValueError):
            agent.macro_action(state,True)

    def test_restored_load_and_cycle_keep_total_load_and_radio_budget(self):
        self.assertEqual(self.cfg["environment"]["zone_loads"], [12.5, 5.5, 3.5])
        self.assertEqual(self.cfg["phase_loads"], [[12.5, 5.5, 3.5], [3.5, 12.5, 5.5], [5.5, 3.5, 12.5]])
        for phase in self.cfg["phase_loads"]:
            self.assertAlmostEqual(sum(phase)*self.cfg["environment"]["load_scale"], 24.725)
        self.assertEqual(self.cfg["environment"]["u2u_bandwidth_mhz"], 6)
        self.assertEqual(self.cfg["cpu_ghz"], 10)

    def test_per_link_meter_does_not_change_physics_and_reconciles_service(self):
        plain = rollout(self.cfg, UniformRouting(), 33500, 2, macro_enabled=False, record=True)
        with patch("hrl_uav.experiment.QueueMDP", lambda cfg: MeteredMDP(cfg, record_links=True)):
            measured = rollout(self.cfg, UniformRouting(), 33500, 2, macro_enabled=False, record=True)
        self.assertEqual(plain[0], measured[0])
        meter = measured[-1]
        self.assertEqual(len(meter.link_readings), 240)
        links = summarize_links(meter.link_readings, 20, 33500, 6)
        self.assertEqual(len(links), 6)
        totals = summarize_meter(meter.readings, 20)
        self.assertAlmostEqual(sum(r["served_mbps_wall"] for r in links), totals["tx_throughput_mbps_wall"])
        self.assertAlmostEqual(sum(r["mean_queue_mbit"] for r in links), totals["mean_queue_transfer_mbit"])
        self.assertLess(max(abs(r["balance_error"]) for r in meter.link_readings), 1e-7)
        self.assertTrue(all(0 <= r["allocated_capacity_utilization"] <= 1.00001 for r in links))


if __name__ == "__main__":
    unittest.main()
