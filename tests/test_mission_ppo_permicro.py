"""Unit tests for the per-micro PPO qualification runner; no environment rollout."""

import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from hrl_uav.experiment import isolated_rng
from hrl_uav.micro_controllers import CVXMicro, FixedRuleMicro, RandomMicro
from hrl_uav.mission_ppo_permicro import (
    DEPENDENCIES, METRICS, MICRO_LABELS, MICROS, HoverMacro, aggregate,
    boundary_payload, make_controller, restore_boundary, stage_name, window0_gate,
)
from hrl_uav.mission_ppo_stage4a import MissionMacroPPO, stage_config
from hrl_uav.mission_smoke import load_source, load_spec
from hrl_uav.trajectory_check import ReadOnlySAC
from hrl_uav.xu_cvx_r import XuCVXRouting


def metric_row(seed, ppo_seed, window, base):
    return dict(policy="direct_ppo", seed=seed, ppo_seed=ppo_seed, window=window,
                **{k: base + i for i, k in enumerate(METRICS)})


class StructureTests(unittest.TestCase):
    def test_micro_registry(self):
        self.assertEqual(MICROS, ("cvx", "random", "local"))
        self.assertEqual(set(MICRO_LABELS), set(MICROS))
        for micro in MICROS:
            self.assertIn(micro, stage_name(micro))
        with self.assertRaises(ValueError):
            stage_name("td3")
        with self.assertRaises(ValueError):
            stage_name("uniform")

    def test_dependencies_cover_new_sources(self):
        for path in ("hrl_uav/micro_controllers.py", "hrl_uav/mission_ppo_permicro.py",
                     "scripts/run_mission_ppo_permicro.py", "hrl_uav/xu_cvx_r.py"):
            self.assertIn(path, DEPENDENCIES)

    def test_random_controller_reproducible_per_scene(self):
        first = make_controller("random", None, 72000)
        again = make_controller("random", None, 72000)
        np.testing.assert_array_equal(first.choose_action(None), again.choose_action(None))

    def test_local_controller(self):
        controller = make_controller("local", None, 33600)
        self.assertIsInstance(controller, FixedRuleMicro)
        self.assertEqual(controller.assert_unchanged()["fixed_rule"], "Local")

    def test_window0_gate_rejects_nonzero_actions_before_io(self):
        observations = [(np.zeros(67), np.full(6, .1), 0., 0., np.zeros(67), 1.)] * 90
        meter = SimpleNamespace(positions=[None]*9001, micro_states=[None]*1800,
                                readings=[None]*1800, commands=[None]*270,
                                route_readings=[None]*10800)
        result = (dict(time_s=900), meter, None, observations, None, None)
        with self.assertRaises(AssertionError):
            window0_gate(result, {}, "unused")
        short = (dict(time_s=20), meter, None, observations, None, None)
        with self.assertRaises(AssertionError):
            window0_gate(short, {}, "unused")

    def test_aggregate(self):
        final_rows = [metric_row(scene, seed, 120, 0.5 + 0.01*i)
                      for i, (seed, scene) in enumerate(
                          (s, e) for s in (201, 202, 203) for e in (33600, 33601))]
        baselines = [dict(metric_row(scene, "none", 0, 0.6), policy="hover")
                     for scene in (33600, 33601)]
        per_seed, overall, pairs = aggregate(final_rows, baselines)
        self.assertEqual(len(per_seed), 3)
        self.assertEqual(len(pairs), 6)
        j = next(r for r in overall if r["policy"] == "direct_ppo" and r["metric"] == "J")
        expected = [np.mean([r["J"] for r in final_rows if r["ppo_seed"] == s]) for s in (201, 202, 203)]
        self.assertAlmostEqual(j["mean"], float(np.mean(expected)), places=12)
        self.assertAlmostEqual(j["std"], float(np.std(expected, ddof=1)), places=12)
        hover = next(r for r in overall if r["policy"] == "hover" and r["metric"] == "J")
        self.assertIsNone(hover["std"])
        with self.assertRaises(ValueError):
            aggregate(final_rows[:5], baselines)
        with self.assertRaises(ValueError):
            aggregate([dict(r, window=0) for r in final_rows], baselines)
        with self.assertRaises(ValueError):
            aggregate(final_rows, baselines[:1])


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.saved, _ = load_source(load_spec())
        with isolated_rng():
            cls.sac = ReadOnlySAC(cls.saved)
        cls.cfg = stage_config(cls.saved["config"], 201)

    def setUp(self):
        self.policy = MissionMacroPPO(self.cfg, 201, self.sac)
        self.lock = {"probe": "per-micro-test"}
        self.stage = stage_name("random")

    def test_payload_validate_restore_roundtrip(self):
        payload = boundary_payload(self.policy, 0, self.lock, self.stage)
        self.assertEqual(payload["stage"], self.stage)
        self.assertIsNone(payload["replay"])
        index = restore_boundary(self.policy, copy.deepcopy(payload), self.lock, self.stage)
        self.assertEqual(index, 0)
        self.assertTrue(self.sac.assert_unchanged().values())

    def test_stage_mismatch_rejected(self):
        payload = boundary_payload(self.policy, 0, self.lock, self.stage)
        with self.assertRaises(ValueError):
            restore_boundary(self.policy, payload, self.lock, stage_name("local"))
        with self.assertRaises(ValueError):
            restore_boundary(self.policy, payload, {"other": 1}, self.stage)

    def test_hover_macro_zero_action(self):
        hover = HoverMacro(self.cfg, self.sac)
        action, logprob = hover.macro_action(np.zeros(67), enabled=True, deterministic=True)
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertEqual(logprob, 0.0)
        with self.assertRaises(ValueError):
            hover.macro_action(np.zeros(67), deterministic=False)
        with self.assertRaises(ValueError):
            hover.macro_action(np.zeros(68), deterministic=True)
        self.assertEqual(hover.ppo_updates, 0)

    def test_cvx_controller_factory(self):
        controller = make_controller("cvx", self.cfg, 72000)
        self.assertIsInstance(controller, CVXMicro)
        self.assertIsInstance(controller.actor, XuCVXRouting)
        audit = controller.assert_unchanged()
        self.assertTrue(audit["convex_lp"])
        self.assertEqual(audit["solver_calls"], 0)


if __name__ == "__main__":
    unittest.main()
