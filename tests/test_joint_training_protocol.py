import sys
import json
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_experiment import build_experiments
from run_experiment import build_micro_pretrain
from run_experiment import common_args
from run_experiment import normalized_macro_distance
from run_experiment import stage_config
from train_centralized_td3 import sample_training_exploration
from train_centralized_td3 import sample_neutral_episode_schedule
from train_centralized_td3 import joint_sac_training_enabled
from train_centralized_td3 import td3_exploration_scope


class JointTrainingProtocolTest(unittest.TestCase):
    def test_v27_min_actor_is_a_single_factor_change_from_v24(self):
        config_dir = PROJECT_ROOT / "configs"
        with (config_dir / "tcom_dynamic_u2u_sparse_v24_dual_on.json").open(
            encoding="utf-8"
        ) as file:
            control = json.load(file)
        with (config_dir / "tcom_dynamic_u2u_sparse_v27_min_actor.json").open(
            encoding="utf-8"
        ) as file:
            experiment = json.load(file)

        ignored = {"results_version", "td3_actor_delay_aggregation"}
        self.assertEqual(
            {key: value for key, value in control.items() if key not in ignored},
            {
                key: value
                for key, value in experiment.items()
                if key not in ignored
            },
        )
        self.assertNotIn("td3_actor_delay_aggregation", control)
        self.assertEqual(experiment["td3_actor_delay_aggregation"], "min")
        self.assertNotIn("td3_neutral_episode_probability", experiment)

    def test_v26_joint_support_audit_only_enables_replay_export(self):
        config_dir = PROJECT_ROOT / "configs"
        with (
            config_dir
            / "tcom_dynamic_u2u_sparse_v25_dual_neutral_coverage.json"
        ).open(encoding="utf-8") as file:
            control = json.load(file)
        with (
            config_dir
            / "tcom_dynamic_u2u_sparse_v26_joint_support_audit.json"
        ).open(encoding="utf-8") as file:
            audit = json.load(file)

        ignored = {"results_version", "save_macro_replay"}
        self.assertEqual(
            {key: value for key, value in control.items() if key not in ignored},
            {key: value for key, value in audit.items() if key not in ignored},
        )
        self.assertNotIn("save_macro_replay", control)
        self.assertTrue(audit["save_macro_replay"])

    def test_v25_neutral_coverage_is_a_single_factor_change(self):
        config_dir = PROJECT_ROOT / "configs"
        with (config_dir / "tcom_dynamic_u2u_sparse_v24_dual_on.json").open(
            encoding="utf-8"
        ) as file:
            control = json.load(file)
        with (
            config_dir
            / "tcom_dynamic_u2u_sparse_v25_dual_neutral_coverage.json"
        ).open(encoding="utf-8") as file:
            coverage = json.load(file)

        ignored = {
            "results_version",
            "td3_neutral_episode_probability",
        }
        self.assertEqual(
            {key: value for key, value in control.items() if key not in ignored},
            {
                key: value
                for key, value in coverage.items()
                if key not in ignored
            },
        )
        self.assertEqual(coverage["td3_neutral_episode_probability"], 0.25)
        self.assertEqual(
            coverage["micro_pretrain_source_results_version"],
            control["results_version"],
        )
        self.assertEqual(coverage["constraint_form"], "pure_lagrangian")
        self.assertEqual(coverage["macro_power_violation_penalty_weight"], 0.0)

    def test_neutral_episode_schedule_is_deterministic_and_rng_isolated(self):
        np.random.seed(1234)
        expected = np.random.random(8)
        np.random.seed(1234)
        first = sample_neutral_episode_schedule(100, 0.25, seed=1)
        observed = np.random.random(8)
        second = sample_neutral_episode_schedule(100, 0.25, seed=1)

        self.assertTrue(np.array_equal(observed, expected))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.dtype, np.dtype(bool))
        self.assertTrue(np.array_equal(
            sample_neutral_episode_schedule(5, 0.0, seed=1),
            np.zeros(5, dtype=bool),
        ))
        self.assertTrue(np.array_equal(
            sample_neutral_episode_schedule(5, 1.0, seed=1),
            np.ones(5, dtype=bool),
        ))

    def test_runner_forwards_neutral_episode_probability(self):
        import run_experiment

        original = run_experiment.TD3_NEUTRAL_EPISODE_PROB
        try:
            run_experiment.TD3_NEUTRAL_EPISODE_PROB = 0.25
            experiments = build_experiments(
                "smoke",
                seeds=[1],
                config=stage_config("smoke"),
                device="cpu",
            )
        finally:
            run_experiment.TD3_NEUTRAL_EPISODE_PROB = original

        experiment = next(
            item
            for item in experiments
            if item["method"] == "joint_td3_sac"
        )
        index = experiment["args"].index("--td3_neutral_episode_prob")
        self.assertEqual(float(experiment["args"][index + 1]), 0.25)

    def test_runner_forwards_macro_replay_export_flag(self):
        import run_experiment

        original = run_experiment.SAVE_MACRO_REPLAY
        try:
            run_experiment.SAVE_MACRO_REPLAY = True
            experiments = build_experiments(
                "smoke",
                seeds=[1],
                config=stage_config("smoke"),
                device="cpu",
            )
        finally:
            run_experiment.SAVE_MACRO_REPLAY = original

        experiment = next(
            item
            for item in experiments
            if item["method"] == "joint_td3_sac"
        )
        index = experiment["args"].index("--save_macro_replay")
        self.assertEqual(experiment["args"][index + 1], "1")

    def test_runner_forwards_actor_delay_aggregation(self):
        import run_experiment

        original = run_experiment.TD3_ACTOR_DELAY_AGGREGATION
        try:
            run_experiment.TD3_ACTOR_DELAY_AGGREGATION = "min"
            experiments = build_experiments(
                "smoke",
                seeds=[1],
                config=stage_config("smoke"),
                device="cpu",
            )
        finally:
            run_experiment.TD3_ACTOR_DELAY_AGGREGATION = original

        experiment = next(
            item
            for item in experiments
            if item["method"] == "joint_td3_sac"
        )
        index = experiment["args"].index("--td3_actor_delay_aggregation")
        self.assertEqual(experiment["args"][index + 1], "min")

    def test_v24_dual_pilot_is_a_single_factor_pure_lagrangian_pair(self):
        config_dir = PROJECT_ROOT / "configs"
        with (config_dir / "tcom_dynamic_u2u_sparse_v24_dual_off.json").open(
            encoding="utf-8"
        ) as file:
            dual_off = json.load(file)
        with (config_dir / "tcom_dynamic_u2u_sparse_v24_dual_on.json").open(
            encoding="utf-8"
        ) as file:
            dual_on = json.load(file)

        ignored = {
            "results_version",
            "micro_pretrain_source_results_version",
            "energy_dual_lr",
        }
        self.assertEqual(
            {key: value for key, value in dual_off.items() if key not in ignored},
            {key: value for key, value in dual_on.items() if key not in ignored},
        )
        self.assertEqual(dual_off["energy_dual_lr"], 0.0)
        self.assertGreater(dual_on["energy_dual_lr"], 0.0)
        for config in (dual_off, dual_on):
            self.assertEqual(config["objective_mode"], "constrained_delay")
            self.assertEqual(config["constraint_form"], "pure_lagrangian")
            self.assertEqual(config["macro_power_violation_penalty_weight"], 0.0)
            self.assertTrue(config["system_dual_state"])

    def test_joint_sac_starts_after_macro_replay_warmup(self):
        self.assertFalse(joint_sac_training_enabled(True, False, True))
        self.assertTrue(joint_sac_training_enabled(True, False, False))
        self.assertFalse(joint_sac_training_enabled(True, True, False))
        self.assertFalse(joint_sac_training_enabled(False, False, False))

    def test_main_stages_use_thirty_macro_steps_without_zone_confinement(self):
        config = stage_config("pilot")
        self.assertEqual(config["macro_steps"], 30)
        experiment = build_micro_pretrain(
            "pilot",
            seed=1,
            config=config,
            device="cpu",
        )
        radius_index = experiment["args"].index("--user_zone_radius")
        self.assertEqual(float(experiment["args"][radius_index + 1]), 0.0)

    def test_micro_pretraining_freezes_only_the_macro_layer(self):
        import run_experiment

        original = (
            run_experiment.MICRO_REWARD_SCALE,
            run_experiment.SAC_GRAD_CLIP_NORM,
        )
        try:
            run_experiment.MICRO_REWARD_SCALE = 1.0
            run_experiment.SAC_GRAD_CLIP_NORM = 1.0
            config = stage_config("smoke")
            config["micro_pretrain_macro_policy"] = "zero"
            experiment = build_micro_pretrain(
                "smoke",
                seed=1,
                config=config,
                device="cpu",
            )
        finally:
            (
                run_experiment.MICRO_REWARD_SCALE,
                run_experiment.SAC_GRAD_CLIP_NORM,
            ) = original

        self.assertIn("--freeze_macro", experiment["args"])
        self.assertNotIn("--freeze_micro", experiment["args"])
        frozen_policy_index = experiment["args"].index("--frozen_macro_policy")
        self.assertEqual(experiment["args"][frozen_policy_index + 1], "zero")
        scale_index = experiment["args"].index("--micro_reward_scale")
        clip_index = experiment["args"].index("--sac_grad_clip_norm")
        self.assertEqual(float(experiment["args"][scale_index + 1]), 1.0)
        self.assertEqual(float(experiment["args"][clip_index + 1]), 1.0)

    def test_main_method_continues_training_the_micro_layer(self):
        config = stage_config("smoke")
        experiments = build_experiments(
            "smoke",
            seeds=[1],
            config=config,
            device="cpu",
        )
        experiment = next(
            item
            for item in experiments
            if item["method"] == "joint_td3_sac"
        )

        self.assertIn("--micro_ckpt", experiment["args"])
        self.assertNotIn("--freeze_micro", experiment["args"])
        optimizer_index = experiment["args"].index(
            "--resume_micro_optimizer"
        )
        self.assertEqual(experiment["args"][optimizer_index + 1], "0")
        capacity_index = experiment["args"].index("--td3_replay_capacity")
        self.assertEqual(experiment["args"][capacity_index + 1], "12000")
        warmup_index = experiment["args"].index("--td3_warmup_steps")
        self.assertEqual(experiment["args"][warmup_index + 1], "300")
        updates_index = experiment["args"].index(
            "--td3_updates_per_step"
        )
        delay_index = experiment["args"].index("--td3_policy_delay")
        self.assertEqual(experiment["args"][updates_index + 1], "1")
        self.assertEqual(experiment["args"][delay_index + 1], "2")
        hover_index = experiment["args"].index(
            "--td3_hover_exploration_prob"
        )
        self.assertEqual(experiment["args"][hover_index + 1], "0.0")
        neutral_index = experiment["args"].index(
            "--td3_neutral_episode_prob"
        )
        self.assertEqual(experiment["args"][neutral_index + 1], "0.0")
        persistence_index = experiment["args"].index(
            "--td3_exploration_persistence"
        )
        self.assertEqual(
            experiment["args"][persistence_index + 1],
            "episode",
        )
        initial_index = experiment["args"].index(
            "--td3_exploration_initial"
        )
        self.assertAlmostEqual(
            float(experiment["args"][initial_index + 1]),
            normalized_macro_distance(4.0),
        )

    def test_joint_methods_receive_configured_micro_learning_rates(self):
        import run_experiment

        original = (
            run_experiment.JOINT_SAC_ACTOR_LR,
            run_experiment.JOINT_SAC_CRITIC_LR,
            run_experiment.JOINT_SAC_ALPHA_LR,
        )
        try:
            run_experiment.JOINT_SAC_ACTOR_LR = 1e-5
            run_experiment.JOINT_SAC_CRITIC_LR = 2e-5
            run_experiment.JOINT_SAC_ALPHA_LR = 3e-5
            experiments = build_experiments(
                "smoke",
                seeds=[1],
                config=stage_config("smoke"),
                device="cpu",
            )
        finally:
            (
                run_experiment.JOINT_SAC_ACTOR_LR,
                run_experiment.JOINT_SAC_CRITIC_LR,
                run_experiment.JOINT_SAC_ALPHA_LR,
            ) = original

        for method in ("joint_td3_sac", "joint_ppo_sac"):
            experiment = next(
                item for item in experiments if item["method"] == method
            )
            for flag, expected in (
                ("--sac_actor_lr", 1e-5),
                ("--sac_critic_lr", 2e-5),
                ("--sac_alpha_lr", 3e-5),
            ):
                index = experiment["args"].index(flag)
                self.assertEqual(
                    float(experiment["args"][index + 1]),
                    expected,
                )

    def test_joint_run_can_reuse_an_explicit_pretraining_version(self):
        import run_experiment

        original = run_experiment.MICRO_PRETRAIN_SOURCE_VERSION
        try:
            run_experiment.MICRO_PRETRAIN_SOURCE_VERSION = "shared_micro_v1"
            experiments = build_experiments(
                "pilot",
                seeds=[42],
                config=stage_config("pilot"),
                device="cpu",
            )
        finally:
            run_experiment.MICRO_PRETRAIN_SOURCE_VERSION = original

        experiment = next(
            item
            for item in experiments
            if item["method"] == "joint_ppo_sac"
        )
        index = experiment["args"].index("--micro_ckpt")
        self.assertEqual(
            experiment["args"][index + 1],
            str(
                Path(
                    "results",
                    "shared_micro_v1",
                    "pilot",
                    "seed42",
                    "micro_pretrain",
                    "checkpoints",
                    "best.pt",
                )
            ),
        )

    def test_joint_run_can_freeze_one_shared_micro_checkpoint(self):
        import run_experiment

        original = (
            run_experiment.MICRO_PRETRAIN_SOURCE_VERSION,
            run_experiment.MICRO_PRETRAIN_SOURCE_SEED,
            run_experiment.FREEZE_MICRO_IN_JOINT,
        )
        try:
            run_experiment.MICRO_PRETRAIN_SOURCE_VERSION = "shared_micro_v2"
            run_experiment.MICRO_PRETRAIN_SOURCE_SEED = 1
            run_experiment.FREEZE_MICRO_IN_JOINT = True
            experiments = build_experiments(
                "pilot",
                seeds=[42],
                config=stage_config("pilot"),
                device="cpu",
            )
        finally:
            (
                run_experiment.MICRO_PRETRAIN_SOURCE_VERSION,
                run_experiment.MICRO_PRETRAIN_SOURCE_SEED,
                run_experiment.FREEZE_MICRO_IN_JOINT,
            ) = original

        experiment = next(
            item
            for item in experiments
            if item["method"] == "joint_td3_sac"
        )
        index = experiment["args"].index("--micro_ckpt")
        self.assertEqual(
            experiment["args"][index + 1],
            str(
                Path(
                    "results",
                    "shared_micro_v2",
                    "pilot",
                    "seed1",
                    "micro_pretrain",
                    "checkpoints",
                    "best.pt",
                )
            ),
        )
        self.assertIn("--freeze_micro", experiment["args"])

    def test_delta_exploration_can_persist_for_a_whole_episode(self):
        episode_noise = np.array(
            [[0.2, -0.1], [0.1, 0.0], [-0.3, 0.2]],
            dtype=np.float32,
        )
        self.assertEqual(
            td3_exploration_scope("delta", "episode"),
            "episode",
        )
        sampled, hover = sample_training_exploration(
            num_agents=3,
            noise_std=0.2,
            hover_probability=0.0,
            macro_action_mode="delta",
            exploration_persistence="episode",
            episode_noise=episode_noise,
            episode_hover=False,
        )
        self.assertFalse(hover)
        self.assertTrue(np.array_equal(sampled, episode_noise))

    def test_waypoint_distance_uses_waypoint_radius(self):
        self.assertAlmostEqual(
            normalized_macro_distance(40.0, action_mode="waypoint"),
            0.2,
        )

    def test_ppo_waypoint_exploration_uses_physical_target_scale(self):
        import run_experiment

        original = (
            run_experiment.MACRO_ACTION_MODE,
            run_experiment.MACRO_WAYPOINT_RADIUS_M,
            run_experiment.PPO_EXPLORATION_INITIAL_M,
            run_experiment.PPO_EXPLORATION_FINAL_M,
        )
        try:
            run_experiment.MACRO_ACTION_MODE = "waypoint"
            run_experiment.MACRO_WAYPOINT_RADIUS_M = 200.0
            run_experiment.PPO_EXPLORATION_INITIAL_M = 60.0
            run_experiment.PPO_EXPLORATION_FINAL_M = 5.0
            experiments = build_experiments(
                "smoke",
                seeds=[1],
                config=stage_config("smoke"),
                device="cpu",
            )
        finally:
            (
                run_experiment.MACRO_ACTION_MODE,
                run_experiment.MACRO_WAYPOINT_RADIUS_M,
                run_experiment.PPO_EXPLORATION_INITIAL_M,
                run_experiment.PPO_EXPLORATION_FINAL_M,
            ) = original

        experiment = next(
            item
            for item in experiments
            if item["method"] == "joint_ppo_sac"
        )
        initial_index = experiment["args"].index("--ppo_initial_log_std")
        final_index = experiment["args"].index("--ppo_final_log_std")
        self.assertAlmostEqual(
            float(experiment["args"][initial_index + 1]),
            np.log(60.0 / 200.0),
        )
        self.assertAlmostEqual(
            float(experiment["args"][final_index + 1]),
            np.log(5.0 / 200.0),
        )

    def test_common_args_expose_configured_waypoint_controls(self):
        import run_experiment

        original = (
            run_experiment.MACRO_ACTION_MODE,
            run_experiment.MACRO_WAYPOINT_RADIUS_M,
            run_experiment.MACRO_WAYPOINT_TOLERANCE_M,
        )
        try:
            run_experiment.MACRO_ACTION_MODE = "waypoint"
            run_experiment.MACRO_WAYPOINT_RADIUS_M = 200.0
            run_experiment.MACRO_WAYPOINT_TOLERANCE_M = 0.5
            args = common_args(stage_config("smoke"), device="cpu")
        finally:
            (
                run_experiment.MACRO_ACTION_MODE,
                run_experiment.MACRO_WAYPOINT_RADIUS_M,
                run_experiment.MACRO_WAYPOINT_TOLERANCE_M,
            ) = original

        mode_index = args.index("--macro_action_mode")
        radius_index = args.index("--macro_waypoint_radius_m")
        tolerance_index = args.index("--macro_waypoint_tolerance_m")
        self.assertEqual(args[mode_index + 1], "waypoint")
        self.assertEqual(float(args[radius_index + 1]), 200.0)
        self.assertEqual(float(args[tolerance_index + 1]), 0.5)

    def test_common_args_support_one_system_average_power_constraint(self):
        import run_experiment

        original = (
            run_experiment.POWER_CONSTRAINT_SCOPE,
            run_experiment.SYSTEM_POWER_BUDGET_W,
        )
        try:
            run_experiment.POWER_CONSTRAINT_SCOPE = "system"
            run_experiment.SYSTEM_POWER_BUDGET_W = 840.0
            args = common_args(stage_config("smoke"), device="cpu")
        finally:
            (
                run_experiment.POWER_CONSTRAINT_SCOPE,
                run_experiment.SYSTEM_POWER_BUDGET_W,
            ) = original

        scope_index = args.index("--power_constraint_scope")
        budget_index = args.index("--power_budget_w")
        self.assertEqual(args[scope_index + 1], "system")
        self.assertEqual(float(args[budget_index + 1]), 840.0)
        self.assertNotIn("--uav_power_budgets_w", args)
        self.assertNotIn("--energy_lagrange_multipliers", args)


if __name__ == "__main__":
    unittest.main()
