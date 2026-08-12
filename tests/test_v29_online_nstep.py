import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_centralized_td3 import MacroReplayBuffer
from train_centralized_td3 import TD3Agent
from train_centralized_td3 import discounted_sequence_target
from audit_v29_fixed_probe import aggregate_episode_checkpoint
from audit_v29_fixed_probe import aggregate_main_window
from audit_v29_fixed_probe import paired_mechanism_summary


def make_args(delay_n_step):
    return SimpleNamespace(
        td3_hidden_dim=32,
        td3_actor_lr=3e-4,
        td3_critic_lr=1e-3,
        td3_gamma=0.99,
        td3_tau=0.005,
        td3_policy_noise=0.02,
        td3_noise_clip=0.05,
        td3_policy_delay=100,
        td3_actor_warmup_updates=0,
        td3_actor_delay_aggregation="q1",
        td3_actor_ema_decay=0.0,
        td3_reward_scale=0.01,
        td3_delay_n_step=delay_n_step,
        td3_replay_sequence_span=5,
        macro_steps=6,
        objective_mode="constrained_delay",
        power_constraint_scope="system",
        micro_solver="local",
    )


def populate_replay(buffer, episodes=5, horizon=6):
    rng = np.random.default_rng(90210)
    for episode in range(1, episodes + 1):
        for macro in range(horizon):
            state = rng.normal(size=4).astype(np.float32)
            action = rng.uniform(-0.4, 0.4, size=6).astype(np.float32)
            next_state = rng.normal(size=4).astype(np.float32)
            buffer.add(
                state,
                action,
                reward=-0.01 * (1 + macro),
                next_state=next_state,
                done=False,
                energy_ratio=0.8 + 0.01 * macro,
                episode_index=episode,
                macro_index=macro,
            )


class V29OnlineNStepTests(unittest.TestCase):
    def test_v32a_configs_are_a_frozen_micro_single_factor_pair(self):
        config_dir = ROOT / "configs"
        with (config_dir / "tcom_dynamic_u2u_sparse_v32a_control_n1.json").open(
            "r", encoding="utf-8"
        ) as handle:
            control = json.load(handle)
        with (
            config_dir / "tcom_dynamic_u2u_sparse_v32a_treatment_n5.json"
        ).open("r", encoding="utf-8") as handle:
            treatment = json.load(handle)

        ignored = {"results_version", "td3_delay_n_step"}
        self.assertEqual(
            {key: value for key, value in control.items() if key not in ignored},
            {
                key: value
                for key, value in treatment.items()
                if key not in ignored
            },
        )
        self.assertEqual(control["td3_delay_n_step"], 1)
        self.assertEqual(treatment["td3_delay_n_step"], 5)
        self.assertEqual(control["td3_replay_sequence_span"], 5)
        self.assertEqual(control["micro_pretrain_source_seed"], 1)
        self.assertTrue(control["freeze_micro_in_joint"])
        self.assertEqual(control["formal_seeds"], [1, 42, 123])
        self.assertEqual(control["u2u_state_representation"], "nominal_rate")
        self.assertFalse(control["u2u_contention_aware_pressure"])
        self.assertFalse(control["u2u_prev_active_state"])

    def test_v29_configs_are_a_single_factor_pair(self):
        with (ROOT / "configs" / "tcom_dynamic_u2u_sparse_v29_control_n1.json").open(
            "r", encoding="utf-8"
        ) as handle:
            control = json.load(handle)
        with (
            ROOT / "configs" / "tcom_dynamic_u2u_sparse_v29_treatment_n5.json"
        ).open("r", encoding="utf-8") as handle:
            treatment = json.load(handle)

        self.assertEqual(control["td3_delay_n_step"], 1)
        self.assertEqual(treatment["td3_delay_n_step"], 5)
        self.assertEqual(control["td3_replay_sequence_span"], 5)
        self.assertEqual(treatment["td3_replay_sequence_span"], 5)
        self.assertEqual(control["pilot_eval_interval"], 10)
        self.assertEqual(treatment["pilot_eval_interval"], 10)
        ignored = {"results_version", "td3_delay_n_step"}
        self.assertEqual(
            {key: value for key, value in control.items() if key not in ignored},
            {
                key: value
                for key, value in treatment.items()
                if key not in ignored
            },
        )

    def test_sequence_sampler_never_crosses_episode_and_shortens_tail(self):
        buffer = MacroReplayBuffer(2, 2, 20)
        for macro in range(5):
            buffer.add(
                np.full(2, macro, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                reward=float(macro),
                next_state=np.full(2, macro + 1, dtype=np.float32),
                done=False,
                episode_index=1,
                macro_index=macro,
            )
        for macro in range(3):
            buffer.add(
                np.full(2, 10 + macro, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                reward=float(macro),
                next_state=np.full(2, 11 + macro, dtype=np.float32),
                done=False,
                episode_index=2,
                macro_index=macro,
            )

        np.testing.assert_array_equal(
            buffer.sequence_indices(0, sequence_span=5, episode_horizon=5),
            np.arange(5),
        )
        np.testing.assert_array_equal(
            buffer.sequence_indices(3, sequence_span=5, episode_horizon=5),
            np.asarray([3, 4]),
        )
        self.assertIsNone(
            buffer.sequence_indices(5, sequence_span=5, episode_horizon=5)
        )

    def test_sequence_sampler_rng_isolated_and_indices_reproducible(self):
        buffers = [
            MacroReplayBuffer(
                4,
                6,
                128,
                rng=np.random.default_rng(731),
            )
            for _ in range(2)
        ]
        for buffer in buffers:
            populate_replay(buffer)

        first = buffers[0].sample_sequence_batch(
            16,
            torch.device("cpu"),
            delay_n_step=1,
            sequence_span=5,
            episode_horizon=5,
            current_episode_index=6,
        )
        np.random.normal(size=1000)
        second = buffers[1].sample_sequence_batch(
            16,
            torch.device("cpu"),
            delay_n_step=5,
            sequence_span=5,
            episode_horizon=5,
            current_episode_index=6,
        )
        np.testing.assert_array_equal(
            first["start_indices"],
            second["start_indices"],
        )
        torch.testing.assert_close(first["state"], second["state"])
        torch.testing.assert_close(
            first["reward_sequence"],
            second["reward_sequence"],
        )

    def test_n1_sequence_target_equals_historical_one_step_formula(self):
        rewards = torch.tensor([[[-0.4]], [[0.3]]], dtype=torch.float32)
        lengths = torch.ones((2, 1), dtype=torch.long)
        bootstrap = torch.tensor([[0.7], [-0.2]], dtype=torch.float32)
        done = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        gamma = 0.99
        sequence_target = discounted_sequence_target(
            rewards,
            lengths,
            bootstrap,
            done,
            gamma,
        )
        historical_target = rewards[:, 0] + (
            (1.0 - done) * gamma * bootstrap
        )
        torch.testing.assert_close(
            sequence_target,
            historical_target,
            rtol=0.0,
            atol=0.0,
        )

    def test_discounted_target_uses_variable_horizon_bootstrap(self):
        rewards = torch.as_tensor(
            [
                [[1.0], [2.0], [3.0], [4.0], [5.0]],
                [[1.0], [2.0], [3.0], [4.0], [5.0]],
            ]
        )
        target = discounted_sequence_target(
            rewards,
            torch.as_tensor([[1], [5]]),
            torch.as_tensor([[10.0], [10.0]]),
            torch.zeros((2, 1)),
            gamma=0.5,
        )
        np.testing.assert_allclose(
            target.numpy().reshape(-1),
            np.asarray([6.0, 3.875]),
        )

    def test_nstep_changes_only_delay_critic_target(self):
        control_args = make_args(delay_n_step=1)
        treatment_args = make_args(delay_n_step=5)
        torch.manual_seed(17)
        control = TD3Agent(4, 6, 3, control_args, torch.device("cpu"))
        treatment = TD3Agent(4, 6, 3, treatment_args, torch.device("cpu"))
        for name in (
            "actor",
            "actor_target",
            "critic",
            "critic_target",
            "energy_critic",
            "energy_critic_target",
        ):
            getattr(treatment, name).load_state_dict(
                copy.deepcopy(getattr(control, name).state_dict())
            )

        buffer = MacroReplayBuffer(4, 6, 64)
        populate_replay(buffer)
        np.random.seed(23)
        torch.manual_seed(29)
        control_result = control.update(
            buffer,
            batch_size=16,
            energy_lagrange_multiplier=0.0,
            current_episode_index=6,
        )
        np.random.seed(23)
        torch.manual_seed(29)
        treatment_result = treatment.update(
            buffer,
            batch_size=16,
            energy_lagrange_multiplier=0.0,
            current_episode_index=6,
        )

        self.assertAlmostEqual(control_result[2], treatment_result[2], places=7)
        self.assertNotAlmostEqual(control_result[1], treatment_result[1], places=7)
        self.assertIsNotNone(control.last_nstep_diagnostics)
        self.assertIsNotNone(treatment.last_nstep_diagnostics)
        np.testing.assert_allclose(
            control.last_nstep_diagnostics["y5_minus_y1"],
            treatment.last_nstep_diagnostics["y5_minus_y1"],
        )
        for control_parameter, treatment_parameter in zip(
            control.energy_critic.parameters(),
            treatment.energy_critic.parameters(),
        ):
            self.assertTrue(
                torch.equal(control_parameter, treatment_parameter)
            )

    def test_probe_ci_uses_episode_seed_after_fork_and_checkpoint_aggregation(self):
        raw = []
        for arm, agreement in (("control_n1", 0.25), ("treatment_n5", 0.75)):
            for checkpoint in (60, 70, 80, 90, 100):
                for seed in range(64000, 64020):
                    for policy in ("actor", "centroid"):
                        for fork in (0, 5, 10, 15, 20, 25):
                            raw.append(
                                {
                                    "arm": arm,
                                    "checkpoint_episode": checkpoint,
                                    "episode_seed": seed,
                                    "snapshot_policy": policy,
                                    "fork_macro_index": fork,
                                    "q1_delta": 0.1,
                                    "q2_delta": 0.1,
                                    "qmin_delta": 0.1,
                                    "true_delta_g30": 0.1,
                                    "q1_agreement": agreement,
                                    "q2_agreement": agreement,
                                    "qmin_agreement": agreement,
                                    "twin_gap_actor_minus_zero": 0.0,
                                }
                            )
        episode_rows = aggregate_episode_checkpoint(raw)
        window_rows = aggregate_main_window(episode_rows)
        paired, summary = paired_mechanism_summary(window_rows)

        self.assertEqual(len(episode_rows), 2 * 5 * 20 * 2)
        self.assertEqual(len(window_rows), 2 * 20 * 2)
        self.assertEqual(len(paired), 20 * 2)
        self.assertEqual(summary["centroid"]["n_episode_seeds"], 20)
        np.testing.assert_allclose(
            summary["centroid"]["q1_agreement_n5_minus_n1_ci95"],
            [0.5, 0.5, 0.5],
        )
        self.assertTrue(summary["mechanism_pass"])


if __name__ == "__main__":
    unittest.main()
