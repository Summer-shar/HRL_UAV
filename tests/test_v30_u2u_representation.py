import unittest

import numpy as np
import torch

from hrl_uav.agents.sac_agent import Memory
from hrl_uav.env import UAV_MEC_Env
from hrl_uav.hrl_wrappers import HRL_Wrapper
from scripts.train_centralized_td3 import MacroReplayBuffer


def make_wrapper(**kwargs):
    defaults = {
        "num_uavs": 3,
        "num_users": 9,
        "users_static": True,
        "macro_action_mode": "centroid_waypoint",
        "macro_reward_mode": "strict_global",
        "pause_service_during_flight": True,
    }
    defaults.update(kwargs)
    return HRL_Wrapper(**defaults)


class V30U2URepresentationTest(unittest.TestCase):
    def test_public_spectral_efficiency_exactly_reconstructs_rate(self):
        env = UAV_MEC_Env(num_uavs=3, num_users=3)
        coords = np.asarray(
            [[100.0, 100.0], [400.0, 100.0], [300.0, 500.0]],
            dtype=np.float64,
        )
        active = np.zeros((3, 3), dtype=bool)
        active[0, 1] = True
        active[2, 0] = True
        eta = env.compute_u2u_spectral_efficiency(coords)
        rate, bandwidth = env.compute_u2u_rate_bps(
            coords,
            active_link_mask=active,
            return_bandwidth=True,
        )
        np.testing.assert_allclose(rate, bandwidth * eta, rtol=0.0, atol=0.0)
        self.assertTrue(np.all(np.diff([
            env.compute_u2u_spectral_efficiency(
                np.asarray([[0.0, 0.0], [d, 0.0], [0.0, 500.0]])
            )[0, 1]
            for d in (100.0, 200.0, 400.0)
        ]) < 0.0))

    def test_old_and_eta_features_are_constant_rescalings(self):
        control = make_wrapper(u2u_state_representation="nominal_rate")
        treatment = make_wrapper(
            u2u_state_representation="spectral_efficiency"
        )
        control.reset(seed=30)
        treatment.reset(seed=30)
        coords = control.env.uav_coords.copy()
        _, nominal, _ = control._compute_u2u_topology_metrics(
            coords,
            np.zeros((3, 3), dtype=np.float32),
        )
        old = control._normalized_u2u_observation(coords, nominal)
        new = treatment._normalized_u2u_observation(coords)
        mask = ~np.eye(3, dtype=bool)
        expected_ratio = 120.0 / treatment.u2u_eta_reference
        np.testing.assert_allclose(
            new[mask] / old[mask],
            expected_ratio,
            rtol=1e-6,
        )

    def test_default_control_state_is_unchanged(self):
        implicit = make_wrapper()
        explicit = make_wrapper(
            u2u_state_representation="nominal_rate",
            u2u_prev_active_state=False,
        )
        implicit.reset(seed=31)
        explicit.reset(seed=31)
        np.testing.assert_array_equal(
            implicit.get_sac_state(),
            explicit.get_sac_state(),
        )
        np.testing.assert_array_equal(
            implicit.get_macro_state()[1],
            explicit.get_macro_state()[1],
        )

    def test_previous_contention_is_causal_and_has_conservative_fallback(self):
        env = make_wrapper(
            u2u_state_representation="spectral_efficiency",
            u2u_prev_active_state=True,
        )
        env.reset(seed=32)
        self.assertAlmostEqual(env.get_sac_state()[-1], 1.0)
        bandwidth_two = np.zeros((3, 3), dtype=np.float64)
        bandwidth_two[0, 1] = 1.0
        bandwidth_two[1, 0] = 1.0
        bandwidth_four = bandwidth_two.copy()
        bandwidth_four[0, 2] = 1.0
        bandwidth_four[2, 0] = 1.0
        env._update_previous_u2u_contention(
            [
                {"u2u_link_bandwidth_hz": bandwidth_two},
                {"u2u_link_bandwidth_hz": bandwidth_four},
            ]
        )
        self.assertAlmostEqual(env.prev_u2u_active_link_count, 3.0)
        self.assertAlmostEqual(env.get_sac_state()[-1], 0.5)
        env._update_previous_u2u_contention(
            [{"u2u_link_bandwidth_hz": np.zeros((3, 3))}]
        )
        self.assertAlmostEqual(env.prev_u2u_active_link_count, 6.0)
        self.assertAlmostEqual(env.get_sac_state()[-1], 1.0)

    def test_previous_contention_controls_only_estimated_pressure_capacity(self):
        env = make_wrapper(
            u2u_state_representation="spectral_efficiency",
            u2u_prev_active_state=True,
        )
        env.reset(seed=33)
        coords = env.env.uav_coords.copy()
        env.prev_u2u_active_link_count = 2.0
        estimated = env._estimated_u2u_rate_mbps(coords)
        eta = env.env.compute_u2u_spectral_efficiency(coords)
        expected = env.env.B_u2u_total / 2.0 * eta / 1e6
        np.testing.assert_allclose(estimated, expected, rtol=1e-6)

    def test_eta_observation_can_be_decoupled_from_goal_and_pressure(self):
        control = make_wrapper(
            u2u_state_representation="nominal_rate",
            u2u_goal_representation="match_observation",
            u2u_contention_aware_pressure=False,
            u2u_prev_active_state=False,
        )
        eta_only = make_wrapper(
            u2u_state_representation="spectral_efficiency",
            u2u_goal_representation="nominal_rate",
            u2u_contention_aware_pressure=False,
            u2u_prev_active_state=False,
        )
        control.reset(seed=34)
        eta_only.reset(seed=34)
        control.prev_u2u_active_link_count = 2.0
        eta_only.prev_u2u_active_link_count = 2.0

        control_state = control.get_sac_state()
        eta_state = eta_only.get_sac_state()
        self.assertEqual(control_state.shape, eta_state.shape)
        np.testing.assert_allclose(control_state[6:30], eta_state[6:30])
        self.assertFalse(np.allclose(control_state[30:39], eta_state[30:39]))
        np.testing.assert_allclose(control_state[39:48], eta_state[39:48])

    def test_replay_rng_isolated_from_global_numpy_stream(self):
        width = 4
        memory_a = Memory(
            32,
            width,
            recent_ratio=0.0,
            rng=np.random.default_rng(34),
        )
        memory_b = Memory(
            32,
            width,
            recent_ratio=0.0,
            rng=np.random.default_rng(34),
        )
        for index in range(20):
            row = np.full(width, index, dtype=np.float32)
            memory_a.mem[index] = row
            memory_b.mem[index] = row
        memory_a.memory_counter = memory_b.memory_counter = 20
        sample_a = memory_a.sample(12)
        np.random.normal(size=1000)
        sample_b = memory_b.sample(12)
        np.testing.assert_array_equal(sample_a, sample_b)

        macro_a = MacroReplayBuffer(2, 2, 32, rng=np.random.default_rng(35))
        macro_b = MacroReplayBuffer(2, 2, 32, rng=np.random.default_rng(35))
        for index in range(20):
            for replay in (macro_a, macro_b):
                replay.add(
                    np.full(2, index, dtype=np.float32),
                    np.zeros(2, dtype=np.float32),
                    float(index),
                    np.full(2, index + 1, dtype=np.float32),
                    False,
                )
        batch_a = macro_a.sample(12, torch.device("cpu"))
        np.random.normal(size=1000)
        batch_b = macro_b.sample(12, torch.device("cpu"))
        for left, right in zip(batch_a, batch_b):
            torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
