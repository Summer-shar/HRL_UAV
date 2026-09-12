"""Narrow no-network checks for the second-stage mission runner."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hrl_uav.mdp import QueueMDP
from hrl_uav.mission_micro_pilot import (
    SOURCE, RULES, FixedHandle, append_result, check_paired_motion, comparisons,
    evaluate_fixed, protocol, read_json, validate_case, verify_previous,
)
from hrl_uav.mission_smoke import load_spec
from hrl_uav.trajectory_check import forbidden
from hrl_uav.wireless_check import read_csv


class MissionMicroTests(unittest.TestCase):
    def setUp(self):
        for name in ("torch.autograd.backward", "torch.autograd.grad",
                     "hrl_uav.agents.sac_agent.Actor.__init__", "hrl_uav.agents.sac_agent.Critic.__init__",
                     "hrl_uav.agents.sac_agent.Entroy.__init__", "hrl_uav.experiment.DirectHRL"):
            context = patch(name, forbidden)
            context.start()
            self.addCleanup(context.stop)
        self.cfg = read_json(SOURCE / "manifest.json")["config"]

    def test_fixed_protocol_and_scene_guards(self):
        p = protocol()
        self.assertEqual(p["new_evaluations"], 8)
        self.assertEqual(p["reused_sac_evaluations"], 2)
        self.assertEqual(RULES["Fixed-A"], (2/3, 1/3))
        self.assertEqual(RULES["Fixed-B"], (1/2, 1/3))
        for s in (33610, 33611, 33612, *range(33700, 33710)):
            with self.assertRaises(ValueError):
                validate_case("Local", s)
        with self.assertRaises(ValueError):
            validate_case("SAC101", 33600)

    def test_executed_routes_match_predeclared_matrices(self):
        mdp = QueueMDP(self.cfg)
        for name, alphas in RULES.items():
            handle = FixedHandle(name)
            for obs in (np.zeros(68), np.arange(68)):
                route = mdp.route(handle.choose_action(obs, deterministic=True))
                for p, alpha in enumerate(alphas):
                    expected = np.full((3, 3), (1-alpha)/2)
                    np.fill_diagonal(expected, alpha)
                    np.testing.assert_allclose(route[p], expected, rtol=0, atol=1e-7)
            self.assertTrue(handle.assert_unchanged()["fixed_rule_unchanged"])
            self.assertEqual(handle.optimizers, ())
            with self.assertRaises(AssertionError):
                handle.choose_action(np.zeros(68), deterministic=False)

    def test_actual_source_reuse_and_mismatch_rejected(self):
        prior, rows, refs, hashes = verify_previous(self.cfg, load_spec())
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["reused"] for r in rows))
        self.assertEqual(len(refs[33600]["crn"]), 1800)
        self.assertIn(str(SOURCE / "manifest.json"), hashes)
        bad = copy.deepcopy(prior)
        bad["config"]["environment"]["u2u_bandwidth_mhz"] = 3
        original = read_json

        def substituted(path):
            return bad if Path(path) == SOURCE / "manifest.json" else original(path)

        with patch("hrl_uav.mission_micro_pilot.read_json", substituted), self.assertRaises(ValueError):
            verify_previous(self.cfg, load_spec())

    def test_twenty_seconds_real_rollout_pairs_with_saved_sac_prefix(self):
        row, meter, _, _, checks = evaluate_fixed(self.cfg, "Uniform", 33600, _short_test=True)
        reference = dict(directory=SOURCE / "predictive_centroid_33600",
                         crn=read_json(SOURCE / "predictive_centroid_33600/crn.json"))
        check_paired_motion(meter, reference, prefix=True)
        self.assertEqual(row["sac_decisions"], 0)
        self.assertEqual(row["fixed_rule_decisions"], 40)
        self.assertEqual(row["micro_decisions"], 40)
        for key in ("sac_update_increment", "ppo_updates", "replay_writes", "replay_samples"):
            self.assertEqual(row[key], 0)
        self.assertFalse(checks["networks_instantiated"])
        self.assertAlmostEqual(row["J"], row["J_holding"]+row["J_energy"], places=12)

    def test_paired_sign_and_csv_append_order(self):
        rows = [dict(policy=p, seed=s, J=.5 if p == "SAC101" else .6, J_holding=.3, J_energy=.2)
                for s in (33600, 33601) for p in (*RULES, "SAC101")]
        paired = comparisons(rows)
        self.assertEqual(len(paired), 8)
        self.assertTrue(all(r["delta_J"] < 0 for r in paired))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.csv"
            append_result(path, dict(policy="SAC101", J=.5), ["policy", "J"])
            append_result(path, dict(J=.6, policy="Local"), ["policy", "J"])
            self.assertEqual(read_csv(path)[1], {"policy": "Local", "J": "0.6"})
            with self.assertRaises(AssertionError):
                append_result(path, dict(policy="Local", extra=1), ["policy", "J"])


if __name__ == "__main__":
    unittest.main()
