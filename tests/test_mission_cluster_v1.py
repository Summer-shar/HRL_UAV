"""Deterministic cluster-rotation scenario tests; no neural learning."""
import copy
import json
import unittest

import numpy as np

from hrl_uav.load_switching import (
    build_cluster_share_schedule,
    build_load_switch_schedule,
    parse_cluster_share_phases,
)
from hrl_uav.mdp import QueueMDP
from hrl_uav.mission_smoke import ROOT, load_spec, mission_config


def base_config():
    base = json.loads(
        (ROOT / "configs/sac_only_wireless6_pilot.json").read_text(encoding="utf-8")
    )
    return mission_config(base)


def cluster_config():
    base = json.loads(
        (ROOT / "configs/sac_only_wireless6_pilot.json").read_text(encoding="utf-8")
    )
    spec = json.loads(
        (ROOT / "configs/mission_cluster_v1.json").read_text(encoding="utf-8")
    )
    return mission_config(base, spec)


def make_mdp(cfg, seed=33600, macros=3):
    mdp = QueueMDP(cfg)
    mdp.reset(seed, macros)
    return mdp


def user_offered_loads(env, users):
    return (
        env.user_lambda_high[users] * env.user_task_size_high_mbits[users]
        + env.user_lambda_low[users] * env.user_task_size_low_mbits[users]
    )


def zone_load_weighted_centroid(env, zone):
    users = np.where(env.user_allocations == zone)[0]
    weights = user_offered_loads(env, users).astype(np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return env.user_coords[users].mean(axis=0)
    return (env.user_coords[users].astype(np.float64) * weights[:, None]).sum(axis=0) / total


class ClusterScenarioTests(unittest.TestCase):
    def test_cluster_assignment_deterministic_complete_and_spatial(self):
        mdp = make_mdp(cluster_config())
        env = mdp.env
        self.assertEqual(env.cluster_count, 3)
        first = env.user_cluster_ids.copy()
        for zone in range(env.N):
            zone_users = np.where(env.user_allocations == zone)[0]
            self.assertEqual(zone_users.size, 13)
            used = set(env.user_cluster_ids[zone_users].tolist())
            self.assertEqual(used, {0, 1, 2})
        mdp.reset(33600, 3)
        self.assertEqual(first.tobytes(), env.user_cluster_ids.tobytes())
        other = make_mdp(cluster_config())
        self.assertEqual(first.tobytes(), other.env.user_cluster_ids.tobytes())

    def test_zone_total_load_conserved_across_share_changes(self):
        mdp = make_mdp(cluster_config())
        env = mdp.env
        before = env.zone_expected_offered_load_mbps.astype(np.float64).copy()
        skewed = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.1, 0.8],
        ])
        env.set_cluster_target_shares(skewed)
        after = env.zone_expected_offered_load_mbps.astype(np.float64)
        self.assertTrue(np.allclose(before, after, rtol=0.0, atol=1e-4))
        cluster = env.cluster_expected_offered_load_mbps.astype(np.float64)
        for zone in range(env.N):
            self.assertTrue(
                np.allclose(cluster[zone], after[zone] * skewed[zone], rtol=0.0, atol=1e-4)
            )

    def test_cluster_operations_consume_no_env_rng(self):
        mdp = make_mdp(cluster_config())
        env = mdp.env
        state = env.np_random.get_state()
        env.set_cluster_target_shares(np.full((3, 3), 1.0 / 3.0))
        build_cluster_share_schedule("cyclic_random", np.full((2, 3, 3), 1.0 / 3.0), 10, 33600)
        self.assertEqual(
            np.asarray(state[1]).tobytes(), np.asarray(env.np_random.get_state()[1]).tobytes()
        )

    def test_macro_observation_extended_micro_unchanged(self):
        legacy = make_mdp(base_config())
        clustered = make_mdp(cluster_config())
        legacy_micro = legacy.micro_observation()
        cluster_micro = clustered.micro_observation()
        self.assertEqual(legacy_micro.size, cluster_micro.size)
        self.assertEqual(
            set(legacy.micro_feature_slices), set(clustered.micro_feature_slices)
        )
        legacy_macro = legacy.macro_observation()
        cluster_macro = clustered.macro_observation()
        self.assertEqual(legacy_macro.size, 67)
        self.assertEqual(cluster_macro.size, legacy_macro.size + 18)
        self.assertIn("cluster_load_share", clustered.macro_feature_slices)
        self.assertIn("cluster_backlog", clustered.macro_feature_slices)
        self.assertTrue(np.all(np.isfinite(cluster_macro)))
        self.assertTrue(np.all(np.isfinite(cluster_micro)))

    def test_centroid_moves_when_shares_rotate(self):
        cfg = cluster_config()
        mdp = make_mdp(cfg)
        env = mdp.env
        phases = np.asarray(cfg["phase_cluster_shares"], dtype=np.float64)

        def zone_centroids():
            return np.stack([
                zone_load_weighted_centroid(env, zone) for zone in range(env.N)
            ])

        cents = []
        for phase in phases:
            env.set_cluster_target_shares(phase)
            cents.append(zone_centroids())
        cents = np.stack(cents)
        jumps = np.stack([
            np.linalg.norm(cents[i] - cents[(i + 1) % phases.shape[0]], axis=1)
            for i in range(phases.shape[0])
        ])
        self.assertGreater(float(jumps.mean()), 10.0)
        self.assertTrue(np.all(jumps.max(axis=0) > 10.0), f"jumps: {jumps}")

    def test_short_rollout_applies_cluster_schedule(self):
        mdp = make_mdp(cluster_config(), macros=4)
        mdp.start_macro(np.zeros(6))
        action = np.zeros(18)
        for _ in range(20):
            mdp.step_frame(action)
        mdp.advance_macro_clock()
        macro = mdp.macro_observation()
        self.assertTrue(np.all(np.isfinite(macro)))
        slices = mdp.macro_feature_slices["cluster_load_share"]
        share = macro[slices[0]:slices[1]].reshape(3, 3)
        totals = share.sum(axis=1)
        self.assertTrue(np.allclose(totals, 1.0, rtol=0.0, atol=1e-5))

    def test_parse_cluster_share_phases_validation(self):
        good = [[[0.5, 0.3, 0.2]] * 3] * 3
        parsed = parse_cluster_share_phases(good, 3, 3)
        self.assertEqual(parsed.shape, (3, 3, 3))
        with self.assertRaises(ValueError):
            parse_cluster_share_phases([[[0.9, 0.2, 0.1]] * 3], 3, 3)
        with self.assertRaises(ValueError):
            parse_cluster_share_phases([[[0.5, 0.5]] * 3], 3, 3)
        text = ";".join(["0.5,0.3,0.2,0.5,0.3,0.2,0.5,0.3,0.2"] * 2)
        parsed_text = parse_cluster_share_phases(text, 3, 3)
        self.assertEqual(parsed_text.shape, (2, 3, 3))

    def test_cluster_schedule_reproducible_and_independent(self):
        phases = np.array([
            [[0.5, 0.3, 0.2], [0.3, 0.2, 0.5], [0.2, 0.5, 0.3]],
            [[0.2, 0.5, 0.3], [0.5, 0.3, 0.2], [0.3, 0.2, 0.5]],
        ])
        a = build_cluster_share_schedule("cyclic_random", phases, 120, 33600)
        b = build_cluster_share_schedule("cyclic_random", phases, 120, 33600)
        self.assertEqual(a.switch_steps, b.switch_steps)
        self.assertEqual(a.phase_index_by_macro.tobytes(), b.phase_index_by_macro.tobytes())
        zone = build_load_switch_schedule(
            "cyclic_random", np.array([[12.5, 5.5, 3.5], [3.5, 12.5, 5.5]]), 120, 33600
        )
        self.assertNotEqual(a.switch_steps, zone.switch_steps)
        c = build_cluster_share_schedule("cyclic_random", phases, 120, 33601)
        self.assertNotEqual(a.switch_steps, c.switch_steps)

    def test_default_spec_unchanged_by_cluster_extension(self):
        spec = load_spec()
        self.assertNotIn("config_overrides", spec)
        self.assertNotIn("cluster_count", spec["environment_overrides"])


if __name__ == "__main__":
    unittest.main()
