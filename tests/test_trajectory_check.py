import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hrl_uav.experiment import rollout
from hrl_uav.geometry import cooperation_aligned_basis
from hrl_uav.mdp import QueueMDP
from hrl_uav.trajectory_check import (
    J_ATOL, MEASURES, OpportunityMeter, RolePolicy, candidates, choose_candidate,
    crn_records, evaluate_rule, forbidden, pair, role_action,
)
from hrl_uav.bandwidth import UniformRouting
from hrl_uav.scene_prescreen import assert_crn_equal

ROOT = Path(__file__).resolve().parents[1]


class FixedDummy:
    def __init__(self):
        self.sac_actor = UniformRouting()
        self.optimizers = ()

    def assert_unchanged(self):
        return {"fixed_dummy": True}


class TrajectoryCheckTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = json.loads((ROOT/"configs/sac_only_wireless6_pilot.json").read_text())

    def test_exact_predeclared_candidates_and_role_permutation(self):
        cs = candidates()
        self.assertEqual(len(cs), 15)
        self.assertEqual(len({tuple(np.asarray(c["offsets_m"]).ravel()) for c in cs}), 15)
        self.assertEqual(cs[0]["name"], "zero")
        np.testing.assert_allclose(role_action([3, 9, 5], cs[1]["offsets_m"], 60),
                                   [0, 0, 1/3, 0, 0, 0], rtol=0, atol=1e-8)
        np.testing.assert_allclose(role_action([9, 3, 5], cs[1]["offsets_m"], 60),
                                   [1/3, 0, 0, 0, 0, 0], rtol=0, atol=1e-8)

    def test_controller_only_needs_current_load_not_schedule_or_future(self):
        policy = RolePolicy(FixedDummy(), candidates()[1])
        policy.meter = SimpleNamespace(env=SimpleNamespace(zone_expected_offered_load_mbps=np.array([9, 3, 5])),
                                       config={"waypoint_radius_m": 60})
        raw, _ = policy.macro_action(np.zeros(67), True, deterministic=True)
        np.testing.assert_allclose(raw, [1/3, 0, 0, 0, 0, 0], atol=1e-8)
        with self.assertRaises(AssertionError):
            policy.macro_action(np.zeros(67), True, deterministic=False)

    def test_all_nonzero_actions_use_existing_geometry_and_move_limit(self):
        for candidate in candidates()[1:]:
            a, b = OpportunityMeter(self.config), QueueMDP(self.config)
            a.reset(33600, 2)
            b.reset(33600, 2)
            before = a.env.uav_coords.copy()
            raw = role_action(a.env.zone_expected_offered_load_mbps, candidate["offsets_m"], 60)
            basis, _ = cooperation_aligned_basis(a.env.weighted_centroids, before, a.env.zone_expected_offered_load_mbps)
            info_a, info_b = a.start_macro(raw), b.start_macro(raw)
            np.testing.assert_array_equal(a.env.uav_coords, b.env.uav_coords)
            self.assertTrue(all(r["distance_m"] is None for r in a.deployments))
            for _ in range(a.env.deployment_resource_frames):
                a.step_frame(np.zeros(18))
                b.step_frame(np.zeros(18))
            np.testing.assert_array_equal(a.env.uav_coords, b.env.uav_coords)
            np.testing.assert_allclose([r["distance_m"] for r in a.deployments],
                                       np.linalg.norm(a.env.uav_coords-before, axis=1))
            self.assertTrue(all(r["execution_complete"] for r in a.deployments))
            self.assertGreater(sum(r["distance_m"] for r in a.deployments), 0)
            self.assertLessEqual(max(r["distance_m"] for r in a.deployments), 20+1e-5)
            self.assertEqual(info_a["E_macro_total"], info_b["E_macro_total"])

    def test_zero_reproduces_native_rollout_and_nonzero_has_paired_exogenous_process(self):
        dummy = FixedDummy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native, _ = evaluate_rule(self.config, dummy, candidates()[0], 33600, 2, root/"native", "smoke", native_zero=True)
            zero, _ = evaluate_rule(self.config, dummy, candidates()[0], 33600, 2, root/"zero", "smoke")
            checks = pair(native, zero, root/"native", root/"zero", same_uav=True)
            self.assertTrue(checks["crn_equal"])
            other, meter = evaluate_rule(self.config, dummy, candidates()[1], 33600, 2, root/"other", "smoke")
            checks = pair(zero, other, root/"zero", root/"other")
            self.assertGreater(other["mean_position_delta_vs_zero_m"], 0)
            self.assertEqual(meter.reset_calls, 1)
            self.assertEqual(len(meter.route_readings), 2*18*6)
            with np.load(root/"other"/"trajectory.npz") as data:
                actual = np.diff(np.concatenate([data["initial_uav"][None], data["uav"]]), axis=0)
                expected_distance = float(np.linalg.norm(actual, axis=-1).sum())
            self.assertGreater(other["fleet_distance_m"], 0)
            np.testing.assert_allclose(other["fleet_distance_m"], expected_distance, rtol=0, atol=1e-9)
            np.testing.assert_allclose(other["J"], other["J_holding"]+other["J_energy"], atol=1e-12)
            paused = [r for r in meter.readings if r["service_steps"] == 0]
            self.assertEqual(len(paused), 4)
            self.assertTrue(all(r["energy_u2u_j"] == 0 and r["energy_upload_j"] == 0 for r in paused))

    def test_selection_complete_grid_ties_and_numerical_noise(self):
        rows = [{"policy": c["name"], "J": 1.0} for c in candidates()]
        self.assertIsNone(choose_candidate(rows))
        rows[1]["J"] -= J_ATOL/2
        self.assertIsNone(choose_candidate(rows))
        rows[13]["J"] = .9
        rows[1]["J"] = .9 + J_ATOL/2
        self.assertEqual(choose_candidate(rows)["name"], rows[1]["policy"])
        rows[1]["J"] = .95
        self.assertEqual(choose_candidate(rows)["name"], rows[13]["policy"])
        with self.assertRaises(ValueError):
            choose_candidate(rows[:-1])

    def test_forbidden_scenes_and_training_actions(self):
        for seed in (*range(33700, 33710), 71000):
            with self.assertRaises(ValueError):
                evaluate_rule(self.config, FixedDummy(), candidates()[0], seed, 90, ROOT/"never_created", "bad")
        with self.assertRaises(AssertionError):
            RolePolicy(FixedDummy(), candidates()[0]).store()


if __name__ == "__main__":
    unittest.main()
