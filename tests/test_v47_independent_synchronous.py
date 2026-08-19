import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_experiment
from compare_sca import _build_zone_priority_task_bits
from train_centralized_macro import fixed_macro_action
from train_centralized_macro import optimizer_update_count
from train_centralized_td3 import joint_sac_training_enabled


class V47IndependentSynchronousProtocolTest(unittest.TestCase):
    def test_frozen_config_declares_independent_full_training(self):
        path = PROJECT_ROOT / "configs" / "tcom_v47_independent_synchronous.json"
        config = json.loads(path.read_text(encoding="utf-8"))

        self.assertTrue(config["independent_micro_pretraining"])
        self.assertTrue(config["resume_micro_replay"])
        self.assertTrue(config["resume_micro_optimizer_in_joint"])
        self.assertFalse(config["freeze_micro_in_joint"])
        self.assertTrue(config["joint_train_micro_during_macro_warmup"])
        self.assertEqual(config["micro_pretrain_episodes"], 200)
        self.assertEqual(config["training_episodes"], 800)
        self.assertEqual(config["formal_seeds"], [1, 7, 42, 99, 123])

    def test_each_sac_method_gets_a_distinct_pretraining_artifact(self):
        original = (
            run_experiment.INDEPENDENT_MICRO_PRETRAINING,
            run_experiment.MICRO_PRETRAIN_MACRO_POLICIES,
        )
        try:
            run_experiment.INDEPENDENT_MICRO_PRETRAINING = True
            run_experiment.MICRO_PRETRAIN_MACRO_POLICIES = {
                "fixed_sac": "fixed",
            }
            config = run_experiment.stage_config("smoke")
            artifacts = {
                method: run_experiment.build_micro_pretrain(
                    "smoke",
                    seed=1,
                    config=config,
                    device="cpu",
                    target_method=method,
                )
                for method in (
                    "joint_td3_sac",
                    "joint_ppo_sac",
                    "centroid_sac",
                    "fixed_sac",
                )
            }
        finally:
            (
                run_experiment.INDEPENDENT_MICRO_PRETRAINING,
                run_experiment.MICRO_PRETRAIN_MACRO_POLICIES,
            ) = original

        subdirs = {item["results_subdir"] for item in artifacts.values()}
        self.assertEqual(len(subdirs), 4)
        for method, artifact in artifacts.items():
            self.assertIn(f"micro_pretrain_{method}", artifact["results_subdir"])
        fixed_args = artifacts["fixed_sac"]["args"]
        index = fixed_args.index("--frozen_macro_policy")
        self.assertEqual(fixed_args[index + 1], "fixed")

    def test_all_formal_baselines_are_built_and_sac_is_not_frozen(self):
        original = (
            run_experiment.INDEPENDENT_MICRO_PRETRAINING,
            run_experiment.JOINT_TRAIN_MICRO_DURING_MACRO_WARMUP,
        )
        try:
            run_experiment.INDEPENDENT_MICRO_PRETRAINING = True
            run_experiment.JOINT_TRAIN_MICRO_DURING_MACRO_WARMUP = True
            experiments = run_experiment.build_experiments(
                "smoke",
                seeds=[1],
                config=run_experiment.stage_config("smoke"),
                device="cpu",
            )
        finally:
            (
                run_experiment.INDEPENDENT_MICRO_PRETRAINING,
                run_experiment.JOINT_TRAIN_MICRO_DURING_MACRO_WARMUP,
            ) = original

        by_method = {item["method"]: item for item in experiments}
        self.assertEqual(
            set(by_method),
            {
                "joint_td3_sac",
                "joint_ppo_sac",
                "cppo_sca",
                "no_coop",
                "centroid_sac",
                "fixed_sac",
            },
        )
        for method in (
            "joint_td3_sac",
            "joint_ppo_sac",
            "centroid_sac",
            "fixed_sac",
        ):
            args = by_method[method]["args"]
            self.assertIn("--micro_ckpt", args)
            self.assertNotIn("--freeze_micro", args)
            checkpoint = args[args.index("--micro_ckpt") + 1]
            self.assertIn(f"micro_pretrain_{method}", checkpoint)
        self.assertIn(
            "--train_micro_during_macro_warmup",
            by_method["joint_td3_sac"]["args"],
        )

    def test_synchronous_flag_keeps_micro_training_during_macro_warmup(self):
        self.assertFalse(joint_sac_training_enabled(True, False, True))
        self.assertTrue(
            joint_sac_training_enabled(
                True,
                False,
                True,
                train_during_macro_warmup=True,
            )
        )
        self.assertFalse(
            joint_sac_training_enabled(
                True,
                True,
                True,
                train_during_macro_warmup=True,
            )
        )

    def test_sca_uses_the_same_soft_objective_as_all_other_methods(self):
        experiments = run_experiment.build_experiments(
            "smoke",
            seeds=[1],
            config=run_experiment.stage_config("smoke"),
            device="cpu",
        )
        sca_args = next(
            item["args"]
            for item in experiments
            if item["method"] == "cppo_sca"
        )
        delay_index = sca_args.index("--sca_delay_weight")
        energy_index = sca_args.index("--sca_energy_weight")
        self.assertAlmostEqual(
            float(sca_args[delay_index + 1]),
            run_experiment.DELAY_OBJECTIVE_WEIGHT,
        )
        self.assertAlmostEqual(
            float(sca_args[energy_index + 1]),
            run_experiment.ENERGY_OBJECTIVE_WEIGHT,
        )

    def test_sca_uses_fixed_region_task_ownership(self):
        base_env = SimpleNamespace(
            N=2,
            user_lambda_high=np.asarray([1.0, 2.0, 3.0, 4.0]),
            user_task_size_high_mbits=np.ones(4),
            user_lambda_low=np.asarray([10.0, 20.0, 30.0, 40.0]),
            user_task_size_low_mbits=np.ones(4),
            arrival_dt=1.0,
            user_allocations=np.asarray([0, 0, 1, 1]),
        )
        base_env.get_nearest_uav_associations = lambda: np.asarray(
            [1, 1, 0, 0]
        )

        task_bits = _build_zone_priority_task_bits(
            SimpleNamespace(env=base_env)
        )
        expected = np.asarray([3.0, 7.0, 30.0, 70.0]) * 1e6
        self.assertTrue(np.array_equal(task_bits, expected))

    def test_optimizer_update_count_survives_state_restore(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        (parameter.square().sum()).backward()
        optimizer.step()

        restored_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        restored = torch.optim.Adam([restored_parameter], lr=1e-3)
        restored.load_state_dict(optimizer.state_dict())
        self.assertEqual(optimizer_update_count(restored), 1)

    def test_fixed_policy_targets_episode_start_positions(self):
        base_env = SimpleNamespace(
            uav_initial_positions=np.asarray(
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
                dtype=np.float64,
            ),
            zone_expected_offered_load_mbps=np.asarray([1.0, 1.0, 1.0]),
        )
        base_env.get_raw_state = lambda: {
            "weighted_centroids": np.asarray(
                [[4.0, 8.0], [24.0, 28.0], [44.0, 48.0]],
                dtype=np.float64,
            ),
            "uav_coords": base_env.uav_initial_positions.copy(),
        }
        env = SimpleNamespace(
            env=base_env,
            macro_waypoint_radius_m=60.0,
            macro_action_mode="centroid_waypoint",
        )

        action = fixed_macro_action(env)
        expected = np.asarray(
            [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(action, expected))


if __name__ == "__main__":
    unittest.main()
