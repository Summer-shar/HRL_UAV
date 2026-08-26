import argparse
import unittest

import numpy as np
import torch

from hrl_uav.agents.sac_agent import Actor as SACActor
from hrl_uav.agents.sac_agent import Critic as SACCritic
from scripts.train_centralized_macro import (
    compose_support_preserving_sca_residual,
    split_autonomous_residual_action,
)
from scripts.evaluate_v51_actor_autonomous_gate import factorial_effects


class DummyResidualEnv:
    N = 3
    micro_action_encoding = "priority_gated"
    micro_priority_weight_ratio = 10.0
    micro_stick_breaking_scale = 4.0
    micro_diag_bias = 0.0
    micro_action_scale = 1.0


def residual_args(gated):
    return argparse.Namespace(
        micro_sca_residual_scale=0.05,
        micro_sca_support_threshold=1e-6,
        micro_sca_residual_update_interval=1,
        micro_sca_residual_max_iter=3,
        micro_sca_residual_tol=1e-3,
        micro_sca_residual_solver="CLARABEL",
        micro_sca_residual_proximal_weight=1e-3,
        micro_autonomous_residual_gate=int(gated),
        delay_objective_weight=0.9,
        energy_objective_weight=0.1,
    )


class V51ActorAutonomousGateTest(unittest.TestCase):
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

    def test_gate_latent_maps_continuously_to_effective_residual(self):
        direction = np.linspace(-0.8, 0.8, 12, dtype=np.float32)
        cases = ((-1.0, 0.0), (0.0, 0.5), (1.0, 1.0))
        for gate_latent, expected_gate in cases:
            components = split_autonomous_residual_action(
                np.concatenate([[gate_latent], direction]),
                12,
                enabled=True,
            )
            self.assertAlmostEqual(components["gate"], expected_gate)
            np.testing.assert_allclose(
                components["effective"],
                expected_gate * direction,
                atol=1e-7,
            )

    def test_gate_minus_one_exactly_reproduces_sca(self):
        latent = np.concatenate(
            [[-1.0], np.linspace(-1.0, 1.0, 12, dtype=np.float32)]
        )
        output = compose_support_preserving_sca_residual(
            self.env,
            latent,
            residual_args(gated=True),
            sca_policy_fn=self.sca_policy,
        ).reshape(2, 3, 3)
        np.testing.assert_allclose(output, self.baseline, atol=1e-6)

    def test_gate_plus_one_matches_legacy_residual_mapping(self):
        direction = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
        gated = compose_support_preserving_sca_residual(
            self.env,
            np.concatenate([[1.0], direction]),
            residual_args(gated=True),
            sca_policy_fn=self.sca_policy,
        )
        legacy = compose_support_preserving_sca_residual(
            self.env,
            direction,
            residual_args(gated=False),
            sca_policy_fn=self.sca_policy,
        )
        np.testing.assert_allclose(gated, legacy, atol=1e-7)

    def test_thirteen_dimensional_sac_action_and_logprob_are_finite(self):
        actor = SACActor(7, 13)
        critic = SACCritic(7, 13)
        actor_device = next(actor.action_net.parameters()).device
        states = torch.zeros((4, 7), dtype=torch.float32, device=actor_device)
        actions, log_prob = actor.evaluate(states)
        self.assertEqual(tuple(actions.shape), (4, 13))
        self.assertEqual(tuple(log_prob.shape), (4, 1))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertTrue(torch.all(actions <= 1.0))
        self.assertTrue(torch.all(actions >= -1.0))
        q1, q2 = critic.get_v(states, actions)
        self.assertEqual(tuple(q1.shape), (4, 1))
        self.assertEqual(tuple(q2.shape), (4, 1))

    def test_v51_config_builds_four_independent_methods(self):
        from scripts import run_experiment

        config_path = (
            run_experiment.PROJECT_ROOT
            + "/configs/tcom_v51_actor_autonomous_gate.json"
        )
        import json

        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual(
            config["enabled_methods"],
            [
                "cppo_sca",
                "joint_ppo_sac",
                "centroid_gated_sac",
                "joint_ppo_gated_sac",
            ],
        )
        self.assertTrue(config["independent_micro_pretraining"])
        self.assertFalse(config["freeze_micro_in_joint"])

        original = (
            run_experiment.ENABLE_AUTONOMOUS_RESIDUAL_GATE_METHODS,
            run_experiment.ENABLED_METHODS,
            run_experiment.INDEPENDENT_MICRO_PRETRAINING,
        )
        try:
            run_experiment.ENABLE_AUTONOMOUS_RESIDUAL_GATE_METHODS = True
            run_experiment.ENABLED_METHODS = tuple(config["enabled_methods"])
            run_experiment.INDEPENDENT_MICRO_PRETRAINING = True
            experiments = run_experiment.build_experiments(
                "smoke",
                seeds=[1],
                config=run_experiment.stage_config("smoke"),
                device="cpu",
            )
        finally:
            (
                run_experiment.ENABLE_AUTONOMOUS_RESIDUAL_GATE_METHODS,
                run_experiment.ENABLED_METHODS,
                run_experiment.INDEPENDENT_MICRO_PRETRAINING,
            ) = original

        by_method = {item["method"]: item for item in experiments}
        self.assertEqual(set(by_method), set(config["enabled_methods"]))
        for method in (
            "joint_ppo_sac",
            "centroid_gated_sac",
            "joint_ppo_gated_sac",
        ):
            checkpoint = by_method[method]["args"]
            checkpoint = checkpoint[checkpoint.index("--micro_ckpt") + 1]
            self.assertIn(f"micro_pretrain_{method}", checkpoint)
        for method in ("centroid_gated_sac", "joint_ppo_gated_sac"):
            self.assertIn(
                "--micro_autonomous_residual_gate",
                by_method[method]["args"],
            )

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            split_autonomous_residual_action(
                np.zeros(12, dtype=np.float32),
                12,
                enabled=True,
            )

    def test_factorial_effects_use_same_seed_four_cell_contrasts(self):
        rows = []
        methods = {
            "joint_ppo_gated_sac": (10.0, 20.0),
            "complete_checkpoint_forced_sca": (13.0, 24.0),
            "complete_checkpoint_centroid_macro": (9.0, 18.0),
            "complete_checkpoint_centroid_macro_forced_sca": (11.0, 21.0),
        }
        for method, values in methods.items():
            for seed, value in enumerate(values, start=1):
                rows.append({"seed": seed, "method": method, "delay": value})

        result = factorial_effects(rows, "delay")
        self.assertEqual(result["paired_seeds"], [1, 2])
        self.assertAlmostEqual(result["gate_effect_under_ppo"]["mean"], -3.5)
        self.assertAlmostEqual(
            result["gate_effect_under_centroid"]["mean"], -2.5
        )
        self.assertAlmostEqual(result["macro_effect_with_gate"]["mean"], 1.5)
        self.assertAlmostEqual(result["macro_effect_with_sca"]["mean"], 2.5)
        self.assertAlmostEqual(result["interaction"]["mean"], -1.0)


if __name__ == "__main__":
    unittest.main()
