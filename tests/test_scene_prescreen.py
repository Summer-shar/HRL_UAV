import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hrl_uav.bandwidth import MeteredMDP, UniformRouting
from hrl_uav.experiment import rollout
from hrl_uav.scene_prescreen import (
    ROOT, LocalRouting, PairedMeter, arm_configs, assert_crn_equal, role_summary, run_prescreen,
)


class ScenePrescreenTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "configs/hrl_ppo_only_unlock.json").read_text())

    def test_six_arms_only_registered_config_fields_change(self):
        before = copy.deepcopy(self.cfg)
        arms = arm_configs(self.cfg)
        self.assertEqual(self.cfg, before)
        self.assertEqual(list(arms), ["p0_local", "p0_uniform_b3", "p0_uniform_b6",
                                      "p1_local", "p1_uniform_b3", "p1_uniform_b6"])
        for name, cfg in arms.items():
            self.assertEqual(cfg["environment"].pop("compute_during_deployment"), name.startswith("p1"))
            bw = cfg["environment"].pop("u2u_bandwidth_mhz")
            self.assertEqual(bw, 3 if name.endswith("b3") else 6)
            expected = copy.deepcopy(before)
            expected["environment"].pop("u2u_bandwidth_mhz")
            self.assertEqual(cfg, expected)

    def test_local_scores_project_to_two_identity_matrices(self):
        mdp = MeteredMDP(self.cfg)
        state = mdp.reset(33500, 2)
        agent = LocalRouting()
        np.testing.assert_array_equal(mdp.route(agent.choose_action(state)), np.tile(np.eye(3), (2, 1, 1)))
        np.testing.assert_array_equal(agent.macro_action(state, False)[0], np.zeros(6))
        with self.assertRaises(ValueError):
            agent.macro_action(state, True)

    def test_pairing_instrumentation_is_read_only(self):
        cfg = arm_configs(self.cfg)["p1_uniform_b6"]
        with patch("hrl_uav.experiment.QueueMDP", MeteredMDP):
            plain = rollout(cfg, UniformRouting(), 33500, 2, macro_enabled=False, record=True)
        with patch("hrl_uav.experiment.QueueMDP", PairedMeter):
            observed = rollout(cfg, UniformRouting(), 33500, 2, macro_enabled=False, record=True)
        self.assertEqual(plain[0], observed[0])
        self.assertEqual(plain[-1].readings, observed[-1].readings)
        self.assertEqual(plain[-1].uav_readings, observed[-1].uav_readings)
        for p, q in zip(plain[2], observed[2]):
            for a, b in zip(p, q):
                np.testing.assert_array_equal(a, b)

    def test_crn_validator_rejects_each_process_mismatch(self):
        row = {"time_s": .5, "arrivals_sha256": "a", "motion_sha256": "b",
               "geometry_sha256": "c", "load_sha256": "d", "schedule_sha256": "e"}
        for key in row:
            with self.subTest(key=key):
                bad = dict(row, **{key: "changed"})
                with self.assertRaisesRegex(AssertionError, key):
                    assert_crn_equal([row], [bad])

    def test_role_summary_follows_changing_load_not_uav_id(self):
        mdp = MeteredMDP(self.cfg)
        mdp.reset(33500, 3)
        for phase in self.cfg["phase_loads"]:
            mdp.env.set_zone_target_loads(phase)
            mdp.start_macro(np.zeros(6))
            for _ in range(20):
                mdp.step_frame(np.zeros(18))
            mdp.advance_macro_clock()
        rows = role_summary(mdp.uav_readings, .5, "test", 33500)
        for row in rows:
            self.assertEqual(row["uav_ids"], "0,1,2")
            self.assertEqual(row["duration_s"], 30)
        self.assertGreater(rows[0]["expected_source_load_mbps"], rows[1]["expected_source_load_mbps"])
        self.assertGreater(rows[1]["expected_source_load_mbps"], rows[2]["expected_source_load_mbps"])
        self.assertAlmostEqual(sum(r["computed_mbit"] for r in rows),
                               sum(r["compute_service_mbit"] for r in mdp.readings))

    def test_short_six_arm_path_writes_results_without_learning(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("hrl_uav.experiment.DirectHRL", side_effect=AssertionError("no networks")):
            output = Path(directory) / "screen"
            report = run_prescreen(self.cfg, output, smoke=True)
            self.assertEqual(report["status"], "SMOKE_COMPLETE")
            self.assertEqual(len(report["rows"]), 6)
            self.assertEqual(len(report["role_summaries"]), 18)
            self.assertTrue(report["crn_equal"])
            self.assertLess(report["queue_balance_max_error"], 1e-5)
            for name in ("manifest.json", "per_seed.csv", "summary.csv", "role_summary.csv", "result.md"):
                self.assertTrue((output / name).is_file())
            for row in report["rows"]:
                self.assertEqual(row["time_s"], 20)
                if row["routing"] == "local":
                    self.assertEqual(row["tx_throughput_mbps_wall"], 0)
            for p, capacity in (("p0", 9), ("p1", 10)):
                self.assertEqual(report["budget"][p]["cpu_nominal_mbps_wall_per_uav"], [capacity] * 3)


if __name__ == "__main__":
    unittest.main()
