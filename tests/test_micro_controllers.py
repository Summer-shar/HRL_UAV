"""Micro-controller interface tests; clock doubles only, no environment rollout."""

import unittest
from types import SimpleNamespace

import numpy as np

from hrl_uav.mdp import compress
from hrl_uav.micro_controllers import (
    CVXMicro, DEFAULT_ALLOWED_SCENES, FixedRuleMicro, RandomMicro, SACMicro, cvx_micro,
)
from hrl_uav.mission_ppo_stage4a import collect_stream
from hrl_uav.xu_cvx_r import SLICES, StaticModel, XuCVXRouting

from tests.test_mission_ppo_stage4a import ClockMeter, FixedMacro

CFG = {"training": {"gamma_per_resource_frame": .99}}


class FixedRuleTests(unittest.TestCase):
    def test_local_routes_everything_to_source(self):
        micro = FixedRuleMicro("Local")
        action = np.asarray(micro.choose_action(None), dtype=np.float64).reshape(2, 3, 3)
        self.assertEqual(action.shape, (2, 3, 3))
        for priority in range(2):
            for source in range(3):
                row = action[priority, source]
                self.assertEqual(row[source], row.max())
        self.assertEqual(micro.assert_unchanged(), {"fixed_rule": "Local", "calls": 1})

    def test_uniform_is_zero_scores(self):
        micro = FixedRuleMicro("Uniform")
        np.testing.assert_array_equal(micro.choose_action(None), np.zeros(18, dtype=np.float32))

    def test_unknown_rule_rejected(self):
        with self.assertRaises(ValueError):
            FixedRuleMicro("Fixed-A")


class RandomMicroTests(unittest.TestCase):
    def test_reproducible_per_scene_and_distinct_across_scenes(self):
        first = [RandomMicro(33600).choose_action(None) for _ in range(1)]
        again = RandomMicro(33600).choose_action(None)
        other = RandomMicro(33601).choose_action(None)
        np.testing.assert_array_equal(first[0], again)
        self.assertFalse(np.allclose(again, other))

    def test_rows_are_probability_vectors(self):
        micro = RandomMicro(72000)
        action = micro.choose_action(None).reshape(2, 3, 3)
        np.testing.assert_allclose(action.sum(axis=2), np.ones((2, 3)), rtol=1e-12)
        self.assertTrue(np.all(action >= 0))
        self.assertEqual(micro.calls, 1)

    def test_scene_whitelist(self):
        for scene in (33610, 33700, 72040):
            with self.assertRaises(ValueError):
                RandomMicro(scene)
        for scene in DEFAULT_ALLOWED_SCENES:
            RandomMicro(scene)  # All declared training/development scenes pass.


class SACMicroTests(unittest.TestCase):
    def test_delegates_and_freezes(self):
        calls = []

        class StubActor:
            def choose_action(self, state, deterministic=False):
                calls.append((state, deterministic))
                return np.ones(18, dtype=np.float32)

        stub = SimpleNamespace(sac_actor=StubActor(), assert_unchanged=lambda: {"frozen": True})
        micro = SACMicro(stub)
        np.testing.assert_array_equal(micro.choose_action("s", deterministic=True), np.ones(18))
        self.assertEqual(calls, [("s", True)])
        self.assertEqual(micro.assert_unchanged(), {"frozen": True})


def synthetic_micro_observation():
    parts = dict.fromkeys(SLICES, None)
    for name, (lo, hi) in SLICES.items():
        size = hi - lo
        if name == "cpu_capacity":
            parts[name] = np.ones(size)
        elif name == "u2u_spectral_efficiency":
            parts[name] = compress(np.full(size, 5.0), 1.0)
        else:
            parts[name] = np.zeros(size)
    state = np.concatenate([parts[name] for name, _ in SLICES.items()]).astype(np.float32)
    assert state.shape == (68,)
    return state


class CVXMicroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = StaticModel(bandwidth_hz=6e6, noise_psd=1.95e-19, tx_budget_w=1.,
                            reference_bandwidth_hz=6e6/11.7, reference_power_w=1.,
                            cycles_per_bit=1000., kappa=1e-28, priority_high=10.,
                            priority_low=1., delay_weight=.9, energy_weight=.1,
                            holding_norm=10., energy_norm=14.)
        cls.micro = CVXMicro(XuCVXRouting(model))

    def test_rejects_non_xu_actor(self):
        with self.assertRaises(TypeError):
            CVXMicro(object())

    def test_empty_arrivals_keep_previous_local_route(self):
        action = np.asarray(self.micro.choose_action(synthetic_micro_observation()),
                            dtype=np.float64).reshape(2, 3, 3)
        np.testing.assert_allclose(action.sum(axis=2), np.ones((2, 3)), rtol=1e-7)
        np.testing.assert_allclose(action, np.tile(np.eye(3), (2, 1, 1)), atol=1e-7)
        self.assertEqual(self.micro.assert_unchanged()["solver_calls"], 1)


class RecordingMicro:
    def __init__(self):
        self.calls = []

    def choose_action(self, state, deterministic=True):
        self.calls.append(deterministic)
        return np.zeros(18, dtype=np.float32)


class InjectionTests(unittest.TestCase):
    """collect_stream wiring: injected controller replaces policy.sac; default is bit-exact."""

    def test_injected_controller_replaces_sac(self):
        policy = FixedMacro(CFG)
        injected = RecordingMicro()
        collect_stream(CFG, policy, 33600, train=False, macros=2,
                       meter_factory=ClockMeter, micro_controller=injected)
        self.assertEqual(len(injected.calls), 40)
        self.assertTrue(all(injected.calls))
        self.assertEqual(policy.micro_flags, [])  # policy.sac.sac_actor never called

    def test_default_path_uses_policy_sac(self):
        policy = FixedMacro(CFG)
        collect_stream(CFG, policy, 33600, train=False, macros=2, meter_factory=ClockMeter)
        self.assertEqual(len(policy.micro_flags), 40)
        self.assertTrue(all(policy.micro_flags))


if __name__ == "__main__":
    unittest.main()
