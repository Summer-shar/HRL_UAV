import argparse
import unittest

import numpy as np

from hrl_uav.agents.sac_agent import Actor as SACActor
from scripts.train_centralized_macro import (
    compose_support_preserving_sca_residual,
    initialize_sac_residual_actor,
)


class DummyResidualEnv:
    N = 3
    micro_action_encoding = "priority_gated"
    micro_priority_weight_ratio = 10.0
    micro_stick_breaking_scale = 4.0
    micro_diag_bias = 0.0
    micro_action_scale = 1.0


def residual_args(scale=0.05):
    return argparse.Namespace(
        micro_sca_residual_scale=scale,
        micro_sca_support_threshold=1e-6,
        micro_sca_residual_update_interval=1,
        micro_sca_residual_max_iter=3,
        micro_sca_residual_tol=1e-3,
        micro_sca_residual_solver="CLARABEL",
        micro_sca_residual_proximal_weight=1e-3,
        delay_objective_weight=0.9,
        energy_objective_weight=0.1,
        sac_zero_mean_output_init=1,
    )


class V49SupportResidualTest(unittest.TestCase):
    def setUp(self):
        self.env = DummyResidualEnv()
        self.baseline = np.asarray(
            [
                [[0.8, 0.2, 0.0], [0.0, 0.7, 0.3], [0.1, 0.0, 0.9]],
                [[0.6, 0.4, 0.0], [0.2, 0.5, 0.3], [0.0, 0.25, 0.75]],
            ],
            dtype=np.float32,
        )

    def sca_policy(self, _env, deterministic=True):
        self.assertTrue(deterministic)
        return self.baseline.reshape(-1)

    def test_zero_latent_reproduces_sca_baseline(self):
        output = compose_support_preserving_sca_residual(
            self.env,
            np.zeros(12, dtype=np.float32),
            residual_args(),
            sca_policy_fn=self.sca_policy,
        ).reshape(2, 3, 3)
        np.testing.assert_allclose(output, self.baseline, atol=1e-6)

    def test_residual_preserves_support_and_simplex(self):
        output = compose_support_preserving_sca_residual(
            self.env,
            np.linspace(-1.0, 1.0, 12, dtype=np.float32),
            residual_args(scale=0.25),
            sca_policy_fn=self.sca_policy,
        ).reshape(2, 3, 3)
        np.testing.assert_array_equal(output[self.baseline == 0.0], 0.0)
        np.testing.assert_allclose(np.sum(output, axis=-1), 1.0, atol=1e-6)
        self.assertTrue(np.all(output[self.baseline > 0.0] > 0.0))

    def test_zero_scale_returns_policy_latent_unchanged(self):
        action = np.linspace(-0.8, 0.8, 12, dtype=np.float32)
        output = compose_support_preserving_sca_residual(
            self.env,
            action,
            residual_args(scale=0.0),
            sca_policy_fn=self.sca_policy,
        )
        np.testing.assert_array_equal(output, action)

    def test_zero_mean_initialization_sets_deterministic_residual_to_zero(self):
        actor = SACActor(5, 12)
        initialize_sac_residual_actor(actor, residual_args())
        action = actor.choose_action(np.ones(5, dtype=np.float32), deterministic=True)
        np.testing.assert_allclose(action, 0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
