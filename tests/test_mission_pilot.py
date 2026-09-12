"""No-backward tests of mission provenance, telemetry and bounded evaluation."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hrl_uav.bandwidth import UniformRouting
from hrl_uav.experiment import isolated_rng
from hrl_uav.mission_provenance import PHYSICS_BASE_COMMIT, physics_provenance, text_digest
from hrl_uav.mission_smoke import ROOT, MissionPolicy, mission_config, load_source, load_spec, run_controller
from hrl_uav.mission_pilot import (
    CONTROLLERS, ObservedPolicy, MissionPilotMeter, SCENES, crn_records, evaluate_one, protocol, validate_case,
)
from hrl_uav.mission_pilot_analysis import pair_rows
from hrl_uav.scene_prescreen import assert_crn_equal
from hrl_uav.trajectory_check import ReadOnlySAC, forbidden


def config():
    return mission_config(json.loads((ROOT / "configs/sac_only_wireless6_pilot.json").read_text()))


class MissionPilotTests(unittest.TestCase):
    def setUp(self):
        for target in ("torch.autograd.backward", "torch.autograd.grad"):
            context = patch(target, forbidden)
            context.start()
            self.addCleanup(context.stop)

    def test_physics_lock_independent_of_head_and_detects_changes(self):
        physics = physics_provenance()
        self.assertIn("hrl_uav/env.py", physics)
        self.assertEqual(PHYSICS_BASE_COMMIT, "19976306281cf213f1e7f6973b999f2dcd39d593")
        self.assertEqual(text_digest(b"a\r\nb\r\n"), text_digest(b"a\nb\n"))
        original = Path.read_bytes

        def changed(path):
            data = original(path)
            return data+b"# unauthorized change\n" if path == ROOT / "hrl_uav/env.py" else data

        with patch.object(Path, "read_bytes", changed), self.assertRaisesRegex(ValueError, "physics changed"):
            physics_provenance()

    def test_fixed_protocol_rejects_extra_rules_scenes_durations(self):
        self.assertEqual(protocol()["evaluations"], 10)
        self.assertFalse(protocol()["training"])
        self.assertEqual(len(SCENES)*len(CONTROLLERS), 10)
        for scene in (33610, 33611, 33612, *range(33700, 33710)):
            with self.assertRaises(ValueError):
                evaluate_one(None, None, "hover", scene)
        for name, seconds in (("uniform", 900), ("local", 900), ("hover", 60), ("hover", 1800)):
            with self.assertRaises(ValueError):
                validate_case(name, 33600, seconds)

    def test_logging_preserves_actions_and_detects_cap_before_projection(self):
        for name in CONTROLLERS:
            original, observed = MissionPolicy(NoneActor(), name), ObservedPolicy(NoneActor(), name)
            # Blind/Hover have no access to the mock environment at all.
            if name in ("blind_patrol", "hover"):
                meter = SimpleNamespace()
            else:
                meter = SimpleNamespace(env=SimpleNamespace(uav_coords=np.zeros((3,2)),
                         weighted_centroids=np.array([[80.,0], [0,50.], [3.,4.]])), centroid_velocity=np.ones((3,2))*.3)
            original.meter = observed.meter = meter
            for _ in range(7):
                np.testing.assert_array_equal(original.macro_action(None, deterministic=True)[0],
                                              observed.macro_action(None, deterministic=True)[0])
        meter = MissionPilotMeter(config())
        meter.reset(33600, 2)
        meter.env.weighted_centroids = meter.env.uav_coords + np.array([[80.,0], [0,50.], [3.,4.]])
        observed = ObservedPolicy(NoneActor(), "progressive_centroid")
        observed.meter = meter
        action, _ = observed.macro_action(None, deterministic=True)
        meter.start_macro(action)
        self.assertEqual([r["cap_triggered"] for r in meter.commands], [1,1,0])
        np.testing.assert_allclose([r["requested_distance_m"] for r in meter.commands], [40,25,2.5])
        np.testing.assert_allclose([r["distance_m"] for r in meter.commands], [20,20,2.5])

    def test_twenty_second_meter_equivalence_crn_and_ledgers(self):
        sac = SimpleNamespace(sac_actor=UniformRouting(), optimizers=(), assert_unchanged=lambda: {"test_double":True})
        reference = None
        for name in ("hover", "predictive_centroid"):
            old, old_meter = run_controller(config(), sac, name, seconds=20)
            row, meter, stats, macros = evaluate_one(config(), sac, name, 33600, _short_test=True)
            for key in ("J", "J_holding", "J_energy", "power_w", "task_delay_weighted_ms", "queue_total_mbit"):
                self.assertEqual(row[key], old[key], key)
            np.testing.assert_array_equal([r[1] for r in meter.positions], [r[1] for r in old_meter.positions])
            self.assertEqual(len(stats), 4)
            self.assertEqual(len(macros), 2)
            self.assertEqual(row["sac_decisions"], 40)
            self.assertEqual(row["boundary_user_hits"], old["boundary_user_hits"])
            self.assertEqual(row["group_endpoint_max_error_m"], 0)
            self.assertGreater(row["u2g_served_mbit"], 0)
            self.assertEqual(row["replay_writes"], 0)
            self.assertEqual(row["replay_samples"], 0)
            records = crn_records(meter)
            if reference is None:
                reference = records
            assert_crn_equal(reference, records)
            bad = copy.deepcopy(records)
            bad[0]["environment_rng_sha256"] = "changed"
            with self.assertRaises(AssertionError):
                assert_crn_equal(records, bad)

    def test_real_sac_frozen_minimal_twenty_second_integration(self):
        saved, _ = load_source(load_spec())
        with isolated_rng():
            sac = ReadOnlySAC(saved)
        row, _, _, _ = evaluate_one(mission_config(saved["config"]), sac, "current_centroid", 33600, _short_test=True)
        self.assertTrue(all(sac.assert_unchanged().values()))
        self.assertEqual(row["sac_update_increment"], 0)
        self.assertEqual(row["ppo_updates"], 0)

    def test_paired_signs_and_blind_risk_are_not_success_claims(self):
        rows = [dict(seed=s, policy=p, J=1., J_holding=.8, J_energy=.2, power_w=800.)
                for s in SCENES for p in CONTROLLERS]
        for row in rows:
            if row["policy"] == "blind_patrol":
                row.update(J=.99, J_holding=.81, J_energy=.18)
        pairs = pair_rows(rows)
        self.assertEqual(len(pairs), 20)
        marked = [p for p in pairs if p["energy_driven_ordering_risk"]]
        self.assertEqual(len(marked), 8)
        self.assertAlmostEqual(marked[0]["relative_J_percent"], -1)


def NoneActor():
    return SimpleNamespace(sac_actor=None)


if __name__ == "__main__":
    unittest.main()
