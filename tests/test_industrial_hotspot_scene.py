"""Hotspot contract checks; one paired rollout totals 100 physical service steps.

The other cases only generate layouts/events or reset environments. No learning.
"""
from dataclasses import asdict
from pathlib import Path
import copy
import json
import unittest

import numpy as np

from hrl_uav.industrial_scene import TYPES, workload_summary
from hrl_uav.industrial_hotspot_scene import (
    PERMUTATIONS, HotspotIndustrialScene, generate_realization, load_config,
    validate_hotspot_config, validate_realization,
)
from hrl_uav.industrial_hotspot_env import HotspotIndustrialEnv
from hrl_uav.industrial_hotspot_mdp import HotspotIndustrialMDP
from hrl_uav.industrial_rollout import IndustrialRollout


def scene_config():
    return load_config(Path(__file__).absolute().parents[1]/"configs/industrial_hotspot_scene_v4.json")


class HotspotSceneTests(unittest.TestCase):
    def test_whole_profile_validation_rejects_partial_or_unsafe_changes(self):
        original = scene_config()
        edits = [lambda c: c["business_profiles"]["catalog"][0].update(cluster_counts_by_type=[5, 4, 1]),
                 lambda c: c["business_profiles"]["catalog"][0]["slots"][0]["parameters"].update(rate_per_s=999),
                 lambda c: c["geometry"].update(random_uav_start=True),
                 lambda c: c["geometry"].update(cluster_radius_m=1000),
                 lambda c: c["business_profiles"].update(forced_rotation=True)]
        for edit in edits:
            changed = copy.deepcopy(original)
            edit(changed)
            with self.assertRaises(ValueError):
                validate_hotspot_config(changed)

    def test_all_permutations_preserve_fixed_physics_and_exact_quotas(self):
        c = scene_config()
        baseline = workload_summary(c["base_scene"])
        for permutation in PERMUTATIONS:
            record = generate_realization(c, 922101, 932101, list(permutation))
            scene = HotspotIndustrialScene(c, 912101, realization=record)
            self.assertEqual(record["profile_by_region"], list(permutation))
            self.assertEqual(record["uav_initial_positions_m"], c["base_scene"]["uav_initial_positions_m"])
            self.assertEqual(len(scene.terminals), 39)
            for region, profile_id in enumerate(permutation):
                rows = record["terminals"][13*region:13*(region+1)]
                profile = c["business_profiles"]["catalog"][profile_id]
                center = np.array(record["cluster_centers_m"][region])
                lo, hi = np.array(c["base_scene"]["regions_m"][region])
                self.assertEqual(sum(row["is_cluster"] for row in rows), 9)
                self.assertEqual([sum(r["task_type"] == t for r in rows) for t in TYPES], profile["device_counts_by_type"])
                self.assertEqual([sum(r["is_cluster"] and r["task_type"] == t for r in rows) for t in TYPES], profile["cluster_counts_by_type"])
                for row in rows:
                    self.assertEqual(row["source"], row["user_id"]//13)
                    p = np.array(row["position_m"])
                    self.assertTrue(np.all(p >= lo) and np.all(p <= hi))
                    inside = np.linalg.norm(p-center) <= c["geometry"]["cluster_radius_m"]
                    self.assertEqual(bool(inside), row["is_cluster"])
            expected = workload_summary(scene.config)
            for name in ("total_input_mbps", "total_compute_gcycles_s"):
                self.assertAlmostEqual(expected[name], baseline[name], places=10)
            self.assertEqual(sorted(round(r["compute_gcycles_s"], 10) for r in expected["regions"]),
                             sorted(round(r["compute_gcycles_s"], 10) for r in baseline["regions"]))

    def test_layout_and_load_are_independent_and_members_move_with_template(self):
        c = scene_config()
        a = generate_realization(c, 922102, 932102, [0, 1, 2])
        b = generate_realization(c, 922102, 932103, [2, 0, 1])
        self.assertEqual(a["cluster_centers_m"], b["cluster_centers_m"])
        by_logic = lambda r: {(t["profile_id"], t["slot"]): t for t in r["terminals"]}
        rows_a, rows_b = by_logic(a), by_logic(b)
        for key, first in rows_a.items():
            second = rows_b[key]
            self.assertEqual(first["is_cluster"], second["is_cluster"])
            if first["is_cluster"]:
                offset_a = np.array(first["position_m"])-a["cluster_centers_m"][first["source"]]
                offset_b = np.array(second["position_m"])-b["cluster_centers_m"][second["source"]]
                np.testing.assert_allclose(offset_a, offset_b, rtol=0, atol=1e-12)
        changed = generate_realization(c, 922103, 932102, [0, 1, 2])
        self.assertNotEqual(a["cluster_centers_m"], changed["cluster_centers_m"])

    def test_sampling_reproducible_with_all_roles_and_adjacent_repetition(self):
        c = scene_config()
        draws = [generate_realization(c, 922104, s)["profile_by_region"] for s in range(80)]
        self.assertEqual({tuple(p) for p in draws}, set(PERMUTATIONS))
        self.assertTrue(any(a == b for a, b in zip(draws, draws[1:])))
        self.assertEqual(generate_realization(c, 922104, 37), generate_realization(c, 922104, 37))

    def test_canonical_events_exact_across_all_six_load_permutations(self):
        c = scene_config()
        scenes = [HotspotIndustrialScene(c, 912105, realization=generate_realization(c, 922105, 932105, list(p)))
                  for p in PERMUTATIONS]
        for frame in range(20):
            for scene in scenes:
                scene.frame(frame)
            for scene in scenes[1:]:
                self.assertEqual(scene.last_canonical_events, scenes[0].last_canonical_events)
                self.assertEqual(scene.canonical_event_digest.hexdigest(), scenes[0].canonical_event_digest.hexdigest())
        self.assertGreater(len({s.event_digest.hexdigest() for s in scenes}), 1)
        self.assertGreater(scenes[0]._task_id, 0)

    def test_changed_layout_preserves_physical_events_for_same_load_mapping(self):
        c = scene_config()
        scenes = [HotspotIndustrialScene(c, 912106, layout_seed=layout, load_seed=932106) for layout in (922106, 922107)]
        self.assertNotEqual(scenes[0].terminals, scenes[1].terminals)
        for frame in range(10):
            self.assertEqual(scenes[0].frame(frame), scenes[1].frame(frame))
        self.assertEqual(scenes[0].event_digest.hexdigest(), scenes[1].event_digest.hexdigest())
        with self.assertRaises(ValueError):
            scenes[0].frame(9)

    def test_sealed_realization_rejects_tampering_and_conflicting_seed(self):
        c = scene_config()
        r = generate_realization(c, 922108, 932108, [1, 2, 0])
        self.assertEqual(validate_realization(c, json.loads(json.dumps(r))), r)
        changed = copy.deepcopy(r)
        changed["terminals"][0]["position_m"][0] += .01
        with self.assertRaises(ValueError):
            validate_realization(c, changed)
        with self.assertRaises(ValueError):
            HotspotIndustrialScene(c, 912108, layout_seed=0, realization=r)
        for wrong in ([0, 0, 2], [True, 1, 2], [0, 1]):
            with self.assertRaises(ValueError):
                generate_realization(c, 922108, 932108, wrong)

    def test_constructor_and_reset_never_construct_legacy_scene(self):
        from unittest.mock import patch
        c = scene_config()
        with patch("hrl_uav.industrial_env.IndustrialScene", side_effect=AssertionError("Legacy scene constructed")):
            env = HotspotIndustrialEnv(c)
            r = generate_realization(c, 922109, 932109, [2, 1, 0])
            env.reset(912109, realization=r)
        self.assertIsInstance(env._scene, HotspotIndustrialScene)
        self.assertEqual(env.config["device_counts"][0], c["base_scene"]["device_counts"][2])
        self.assertEqual(env.step_index, 0)
        self.assertFalse(env.parents)
        self.assertEqual(env.metrics()["physical_steps"], 0)
        self.assertTrue(all(t.history_duration_s == 0. for t in env.view().terminals))
        copy_of_record = env.realization
        copy_of_record["terminals"][0]["position_m"][0] = -999
        self.assertGreater(env.physics.user_coords[0, 0], 0)

    def test_public_reports_do_not_expose_generation_metadata(self):
        mdp = HotspotIndustrialMDP(scene_config())
        self.assertEqual(mdp.reset(912110, 922110, 932110).shape, (85,))
        self.assertEqual(mdp.prepare_service().shape, (86,))
        def keys(value):
            if isinstance(value, dict):
                return set(value).union(*(keys(v) for v in value.values()))
            if isinstance(value, (list, tuple)):
                return set().union(*(keys(v) for v in value))
            return set()
        forbidden = {"cluster_centers_m", "is_cluster", "profile_id", "slot", "profile_by_region",
                     "layout_seed", "load_seed", "busy", "rate_per_s"}
        self.assertFalse(keys(asdict(mdp.controller_information())) & forbidden)
        self.assertEqual(mdp.observation_spec()["macro"]["dim"], 85)
        self.assertEqual(mdp.observation_spec()["micro"]["dim"], 86)

    def test_paired_motion_preserves_events_and_frozen_cutoff_metrics(self):
        # 2 x 1 macro x 50 physical steps = 100 physical service steps.
        c = scene_config()
        r = generate_realization(c, 922111, 932111, [1, 0, 2])
        records = []
        for action in (np.zeros(6), np.array([1., 0., 0., -1., -.5, .5])):
            mdp = HotspotIndustrialMDP(c, ledger_enabled=True)
            mdp.reset(912111, realization=r)
            start = mdp.env.physics.uav_coords.copy()
            collect = IndustrialRollout(mdp, lambda view: action, lambda view: np.tile(np.eye(3), (2, 1, 1)).ravel())
            result = collect.run(1)
            metrics = result["metrics"]
            self.assertEqual(result["physical_steps"], 50)
            self.assertEqual(metrics["physical_steps"], 50)
            self.assertEqual(mdp.env.step_index, 50)
            self.assertTrue(result["next_boundary_primed"])
            successor = mdp.encoder.encode_micro(mdp.controller_information())
            np.testing.assert_array_equal(result["micro"][-1].next_state, successor)
            self.assertTrue(result["micro"][-1].truncated)
            self.assertEqual(result["micro"][-1].trace_continue, 0.)
            self.assertEqual(result["micro"][-1].discount, mdp.gamma_frame)
            self.assertEqual(result["macro"][-1].discount, mdp.gamma_macro)
            for error in metrics["conservation"].values():
                self.assertLess(abs(error), 1e-3)
            arrived = sum(row["tasks"] for row in metrics["arrivals_by_region"])
            self.assertEqual(arrived, sum(row["arrived"] for row in metrics["priority"]))
            self.assertEqual(metrics["realization_sha256"], r["realization_sha256"])
            integral = np.asarray(metrics["unfinished_parent_time_s_by_priority"])
            self.assertAlmostEqual(metrics["holding_task_s"],
                np.dot(c["base_scene"]["objective"]["priority_weights"], integral), places=8)
            self.assertAlmostEqual(metrics["average_unfinished_parents"], integral.sum()/5., places=10)
            # Parent lifetimes provide an independent integral oracle. Tail
            # priming births at the cutoff contribute exactly zero time.
            oracle, alive = np.zeros(2), np.zeros(2, dtype=np.int64)
            for parent in mdp.env.parents.values():
                p = parent.arrival.priority
                finish = 5. if parent.finish_s is None else min(5., parent.finish_s)
                oracle[p] += max(0., finish-parent.arrival.birth_s)
                alive[p] += parent.finish_s is None
            np.testing.assert_allclose(integral, oracle, rtol=0., atol=1e-9)
            np.testing.assert_array_equal(mdp.env._alive_counts, alive)
            if np.any(action):
                self.assertGreater(np.linalg.norm(mdp.env.physics.uav_coords-start), 0.)
            else:
                np.testing.assert_array_equal(mdp.env.physics.uav_coords, start)
            mdp.reset(912112, 922112, 932112)
            self.assertEqual(mdp.env.step_index, 0)
            self.assertFalse(mdp.env.parents)
            np.testing.assert_array_equal(mdp.env._alive_counts, [0, 0])
            np.testing.assert_array_equal(mdp.env.unfinished_parent_time_s_by_priority, [0., 0.])
            np.testing.assert_array_equal(mdp.env.physics.uav_coords, start)
            records.append(metrics)
        for name in ("event_sha256", "canonical_event_sha256", "arrivals_by_region"):
            self.assertEqual(records[0][name], records[1][name])


if __name__ == "__main__":
    unittest.main()
