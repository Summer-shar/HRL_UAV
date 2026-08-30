import argparse
import json
import os
import unittest

import numpy as np

from hrl_uav.hrl_wrappers import HRL_Wrapper
from hrl_uav.routing import project_rows_to_simplex
from hrl_uav.agents.sac_agent import Actor as SACActor
from scripts.train_centralized_macro import (
    compose_support_preserving_sca_residual,
    configure_sac_actor_log_std,
    initialize_sac_residual_actor,
    validate_micro_policy_mode,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def policy_args(mode="direct", residual_scale=0.0, gated=0):
    return argparse.Namespace(
        micro_policy_mode=mode,
        micro_sca_residual_scale=residual_scale,
        micro_autonomous_residual_gate=gated,
    )


class DummyDirectEnv:
    N = 3


class V54DirectSacRepairTest(unittest.TestCase):
    def test_direct_mode_rejects_hidden_sca_residual_or_gate(self):
        self.assertEqual(
            validate_micro_policy_mode(policy_args(), use_sac=True),
            "direct",
        )
        with self.assertRaises(ValueError):
            validate_micro_policy_mode(
                policy_args(residual_scale=0.05),
                use_sac=True,
            )
        with self.assertRaises(ValueError):
            validate_micro_policy_mode(
                policy_args(gated=1),
                use_sac=True,
            )

    def test_direct_action_bypasses_sca_completely(self):
        def forbidden_sca(*_args, **_kwargs):
            raise AssertionError("Direct SAC must not invoke SCA")

        for action_dim in (12, 18):
            action = np.linspace(-0.9, 0.9, action_dim, dtype=np.float32)
            output = compose_support_preserving_sca_residual(
                DummyDirectEnv(),
                action,
                policy_args(),
                sca_policy_fn=forbidden_sca,
            )
            np.testing.assert_array_equal(output, action)

    def test_direct_state_contains_causal_load_and_link_identity(self):
        env = HRL_Wrapper(
            num_uavs=3,
            num_users=39,
            zone_loads=[18.5, 2.0, 1.0],
            load_scale=1.15,
            micro_action_encoding="priority_gated",
            micro_state_mode="pressure",
            micro_expected_load_state=True,
            u2u_active_link_mask_state=True,
        )
        env.reset(seed=7)
        raw = env.env.get_raw_state()
        state = env.get_sac_state(raw)
        self.assertEqual(env.sac_obs_dim, 60)
        self.assertEqual(state.size, env.sac_obs_dim)

        expected_load = state[48:54].reshape(2, 3)
        self.assertGreater(expected_load[:, 0].sum(), expected_load[:, 1].sum())
        self.assertGreater(expected_load[:, 1].sum(), expected_load[:, 2].sum())

        raw["Q_tx_high"][0, 1] = 1.0
        active_mask = env._current_u2u_active_link_mask(raw)
        self.assertEqual(active_mask.shape, (6,))
        self.assertEqual(int(np.sum(active_mask)), 1)

    def test_v54_config_and_named_methods_are_explicitly_direct(self):
        config_path = os.path.join(
            ROOT,
            "configs",
            "tcom_v54_direct_sac_hrl.json",
        )
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual(config["micro_policy_mode"], "direct")
        self.assertEqual(config["micro_sca_residual_scale"], 0.0)
        self.assertFalse(config["enable_autonomous_residual_gate_methods"])
        self.assertTrue(config["micro_expected_load_state"])
        self.assertTrue(config["u2u_active_link_mask_state"])
        self.assertIn("joint_ppo_direct_sac", config["enabled_methods"])

        from scripts import run_experiment

        original_enabled = run_experiment.ENABLED_METHODS
        try:
            run_experiment.ENABLED_METHODS = tuple(config["enabled_methods"])
            specs = {
                item["method"]: item
                for item in run_experiment.method_specs()
            }
        finally:
            run_experiment.ENABLED_METHODS = original_enabled
        for method in (
            "joint_ppo_direct_sac",
            "centroid_direct_sac",
            "fixed_direct_sac",
        ):
            args = specs[method]["args"]
            self.assertEqual(args[args.index("--micro_policy_mode") + 1], "direct")
            self.assertEqual(args[args.index("--micro_sca_residual_scale") + 1], "0")

    def test_sparse_direct_zero_action_is_exactly_local(self):
        action = np.zeros((2, 3, 3), dtype=np.float64)
        action += np.eye(3, dtype=np.float64)[None, ...]
        routed = project_rows_to_simplex(action)
        expected = np.repeat(np.eye(3)[None, ...], 2, axis=0)
        np.testing.assert_allclose(routed, expected, atol=0.0, rtol=0.0)
        self.assertEqual(int(np.count_nonzero(routed)), 6)

    def test_direct_actor_can_start_with_low_variance_local_policy(self):
        actor = SACActor(60, 18)
        args = argparse.Namespace(
            sac_zero_mean_output_init=1,
            sac_initial_log_std=-2.0,
        )
        initialize_sac_residual_actor(actor, args)
        state = np.zeros((1, 60), dtype=np.float32)
        import torch

        with torch.no_grad():
            device = next(actor.action_net.parameters()).device
            mean, std = actor.action_net(torch.as_tensor(state, device=device))
        np.testing.assert_allclose(mean.cpu().numpy(), 0.0, atol=0.0)
        np.testing.assert_allclose(
            std.cpu().numpy(),
            np.exp(-2.0),
            atol=1e-7,
        )

    def test_log_std_override_preserves_warm_started_mean_policy(self):
        import torch

        actor = SACActor(48, 18)
        device = next(actor.action_net.parameters()).device
        state = torch.randn(7, 48, device=device)
        with torch.no_grad():
            mean_before, _ = actor.action_net(state)
            mean_weight_before = actor.action_net.out.weight.detach().clone()
            mean_bias_before = actor.action_net.out.bias.detach().clone()

        configure_sac_actor_log_std(actor, -3.0)

        with torch.no_grad():
            mean_after, std_after = actor.action_net(state)
        torch.testing.assert_close(mean_after, mean_before)
        torch.testing.assert_close(
            actor.action_net.out.weight,
            mean_weight_before,
        )
        torch.testing.assert_close(actor.action_net.out.bias, mean_bias_before)
        np.testing.assert_allclose(
            std_after.cpu().numpy(),
            np.exp(-3.0),
            atol=1e-7,
        )

    def test_v60_is_micro_pretrain_then_joint_two_layer_training(self):
        config_path = os.path.join(
            ROOT,
            "configs",
            "tcom_v60_direct_sac_two_stage.json",
        )
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)

        self.assertEqual(config["micro_pretrain_macro_policy"], "zero")
        self.assertEqual(
            config["micro_pretrain_macro_policies"]["joint_ppo_direct_sac"],
            "zero",
        )
        self.assertFalse(config["freeze_micro_in_joint"])
        self.assertFalse(config["resume_micro_replay"])
        self.assertFalse(config["resume_micro_optimizer_in_joint"])
        self.assertEqual(config["micro_policy_mode"], "direct")
        self.assertEqual(config["micro_sca_residual_scale"], 0.0)
        self.assertEqual(config["micro_pretrain_sac_policy_delay"], 20)
        self.assertEqual(config["joint_sac_policy_delay"], 20)
        self.assertEqual(config["joint_sac_actor_warmup_updates"], 500)

    def test_v55_sparse_direct_config_has_no_sca_ownership(self):
        config_path = os.path.join(
            ROOT,
            "configs",
            "tcom_v55_direct_sparse_sac_hrl.json",
        )
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual(config["micro_policy_mode"], "direct")
        self.assertEqual(config["micro_action_encoding"], "projected_full")
        self.assertEqual(config["micro_diag_bias"], 1.0)
        self.assertEqual(config["micro_sca_residual_scale"], 0.0)
        self.assertEqual(config["sac_initial_log_std"], -2.0)

    def test_v56_distillation_is_actor_only_and_direct(self):
        config_path = os.path.join(
            ROOT,
            "configs",
            "tcom_v56_distilled_direct_sac_hrl.json",
        )
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual(config["micro_policy_mode"], "direct")
        self.assertEqual(config["micro_action_encoding"], "projected_full")
        self.assertEqual(config["micro_sca_residual_scale"], 0.0)
        self.assertEqual(config["micro_pretrain_prefill_episodes"], 5)
        self.assertEqual(config["micro_pretrain_actor_freeze_episodes"], 10)
        self.assertIn(
            "v56_direct_sac_sca_distillation/clone_actor.pt",
            config["micro_pretrain_actor_init_checkpoint"],
        )


if __name__ == "__main__":
    unittest.main()
