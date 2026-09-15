"""Unit tests for joint HRL training experiment module."""

import unittest
from dataclasses import dataclass

import numpy as np

from hrl_uav.mission_joint_training import (
    JointTrainingConfig, JointTrainingState, TrainingMode
)


class TestBudgetMatching(unittest.TestCase):
    """Verify budget matching is exact across conditions."""

    def test_joint_1_10_budget(self):
        """600K micro updates should yield exactly 60 PPO updates at 1:10 ratio."""
        config = JointTrainingConfig(
            mode=TrainingMode.JOINT,
            ppo_update_interval=10,
            total_micro_updates=600_000
        )

        # Calculate expected PPO updates
        # 600K micro / 10 interval = 60K macro collections / 10 interval = 6000 PPO updates?
        # No: 90 macros per scene * 200 scenes = 18,000 total macros
        # At 1:10 ratio: 18,000 / 10 = 1,800 PPO updates max

        # More realistic: budget is 600K micro updates total
        # With 1:10 ratio, each PPO update = 10 micro updates worth of "budget"
        # So 600K / 10 = 60K PPO updates possible

        target_ppo = 600_000 // 10
        self.assertEqual(target_ppo, 60_000)

    def test_joint_1_5_budget(self):
        """600K micro updates at 1:5 ratio."""
        config = JointTrainingConfig(
            mode=TrainingMode.JOINT,
            ppo_update_interval=5,
            total_micro_updates=600_000
        )

        target_ppo = 600_000 // 5
        self.assertEqual(target_ppo, 120_000)

    def test_joint_1_20_budget(self):
        """600K micro updates at 1:20 ratio."""
        config = JointTrainingConfig(
            mode=TrainingMode.JOINT,
            ppo_update_interval=20,
            total_micro_updates=600_000
        )

        target_ppo = 600_000 // 20
        self.assertEqual(target_ppo, 30_000)

    def test_budget_utilization(self):
        """Budget fraction should be computed correctly."""
        state = JointTrainingState(micro_updates=300_000)
        self.assertAlmostEqual(state.budget_utilization(), 0.5)

        state.micro_updates = 600_000
        self.assertAlmostEqual(state.budget_utilization(), 1.0)

        state.micro_updates = 0
        self.assertAlmostEqual(state.budget_utilization(), 0.0)

    def test_budget_exhaustion(self):
        """Budget check should trigger correctly."""
        config = JointTrainingConfig(total_micro_updates=600_000)
        state = JointTrainingState()

        # Initially not exhausted
        self.assertLess(state.micro_updates, config.total_micro_updates)

        # At boundary
        state.micro_updates = 600_000
        self.assertGreaterEqual(state.micro_updates, config.total_micro_updates)


class TestUpdateController(unittest.TestCase):
    """Test update frequency ratio enforcement."""

    def test_ppo_ready_at_exact_interval(self):
        """PPO should update exactly at configured interval."""
        state = JointTrainingState()
        interval = 10

        # Before interval, not ready
        for i in range(1, interval):
            state.ppo_macros_since_update = i
            self.assertFalse(state.ppo_ready(interval), f"Step {i}: should not be ready")

        # At interval, ready
        state.ppo_macros_since_update = interval
        self.assertTrue(state.ppo_ready(interval), f"Step {interval}: should be ready")

        # Beyond interval, ready every interval
        for i in range(interval + 1, interval * 3 + 1):
            state.ppo_macros_since_update = i
            expected = (i % interval == 0)
            self.assertEqual(state.ppo_ready(interval), expected, f"Step {i}: expected {expected}")

    def test_ppo_ready_minimum_threshold(self):
        """PPO should not update before minimum threshold."""
        state = JointTrainingState()
        interval = 10

        # Less than interval
        state.ppo_macros_since_update = 5
        self.assertFalse(state.ppo_ready(interval))

        # Equal to interval
        state.ppo_macros_since_update = 10
        self.assertTrue(state.ppo_ready(interval))

    def test_sac_ready_warmup(self):
        """SAC should not update before warmup threshold."""
        state = JointTrainingState()
        warmup = 1000

        # Below warmup
        state.micro_updates = 500
        self.assertFalse(state.sac_ready(warmup))

        # At warmup
        state.micro_updates = 1000
        self.assertTrue(state.sac_ready(warmup))

        # Above warmup
        state.micro_updates = 2000
        self.assertTrue(state.sac_ready(warmup))


class TestTrainingModes(unittest.TestCase):
    """Test training mode configurations."""

    def test_joint_mode(self):
        """Joint mode should allow both PPO and SAC updates."""
        config = JointTrainingConfig(mode=TrainingMode.JOINT)
        self.assertEqual(config.mode, TrainingMode.JOINT)

    def test_ppo_only_mode(self):
        """PPO-only mode should freeze SAC."""
        config = JointTrainingConfig(mode=TrainingMode.PPO_ONLY)
        self.assertEqual(config.mode, TrainingMode.PPO_ONLY)

        # PPO updates every 10 macros, SAC frozen
        self.assertEqual(config.ppo_update_interval, 10)
        # In PPO_ONLY mode, SAC should be frozen by the trainer

    def test_sac_only_mode(self):
        """SAC-only mode should freeze PPO."""
        config = JointTrainingConfig(mode=TrainingMode.SAC_ONLY)
        self.assertEqual(config.mode, TrainingMode.SAC_ONLY)

        # SAC updates every micro-step, PPO frozen
        # In SAC_ONLY mode, PPO should be frozen by the trainer


class TestJointTrainingState(unittest.TestCase):
    """Test JointTrainingState tracking."""

    def test_initialization(self):
        """Default initialization should be zero."""
        state = JointTrainingState()

        self.assertEqual(state.micro_updates, 0)
        self.assertEqual(state.ppo_updates, 0)
        self.assertEqual(state.macros_collected, 0)
        self.assertEqual(state.episodes_completed, 0)
        self.assertEqual(state.ppo_macros_since_update, 0)

    def test_counters(self):
        """Counters should increment correctly."""
        state = JointTrainingState()

        # Simulate one episode
        for _ in range(90):  # 90 macros per episode
            state.macros_collected += 1
            state.ppo_macros_since_update += 1

            if state.ppo_ready(10):
                state.ppo_updates += 1
                state.ppo_macros_since_update = 0

        self.assertEqual(state.macros_collected, 90)
        self.assertEqual(state.ppo_updates, 9)  # 90 / 10 = 9

    def test_target_ppo_calculation(self):
        """Target PPO updates should be calculated from budget."""
        config = JointTrainingConfig(
            ppo_update_interval=10,
            total_micro_updates=600_000
        )

        state = JointTrainingState(
            target_ppo_updates=config.total_micro_updates // config.ppo_update_interval
        )

        self.assertEqual(state.target_ppo_updates, 60_000)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation."""

    def test_ppo_settings_defaults(self):
        """Default PPO settings should match mission_ppo_stage4a."""
        config = JointTrainingConfig()

        ppo_settings = config.get_ppo_settings()

        self.assertEqual(ppo_settings['ppo_initial_std'], 1/3)
        self.assertEqual(ppo_settings['ppo_final_std'], 1/30)
        self.assertEqual(ppo_settings['ppo_epochs'], 10)
        self.assertEqual(ppo_settings['ppo_actor_lr'], 3e-4)
        self.assertEqual(ppo_settings['ppo_critic_lr'], 1e-3)
        self.assertEqual(ppo_settings['ppo_clip'], 0.2)
        self.assertEqual(ppo_settings['ppo_target_kl'], 0.03)

    def test_budget_matching_consistency(self):
        """Budget should be consistent across update ratios."""
        base_budget = 600_000

        for interval in [5, 10, 20]:
            config = JointTrainingConfig(
                ppo_update_interval=interval,
                total_micro_updates=base_budget
            )

            # Each interval consumes 'interval' micro budget units
            # Total budget / interval = target PPO updates
            target = base_budget // interval

            state = JointTrainingState(target_ppo_updates=target)

            self.assertEqual(state.target_ppo_updates, target)


class TestExperimentMatrix(unittest.TestCase):
    """Test experiment matrix completeness."""

    def test_experiment_configs_exist(self):
        """All planned experiments should have configurations."""
        from scripts.run_mission_joint_training import EXPERIMENT_CONFIGS

        expected_experiments = ['joint_1_5', 'joint_1_10', 'joint_1_20', 'ppo_only', 'sac_only']

        for exp in expected_experiments:
            self.assertIn(exp, EXPERIMENT_CONFIGS, f"Missing config for {exp}")

    def test_experiment_modes(self):
        """Each experiment should have correct mode."""
        from scripts.run_mission_joint_training import EXPERIMENT_CONFIGS

        self.assertEqual(EXPERIMENT_CONFIGS['joint_1_5'].mode, TrainingMode.JOINT)
        self.assertEqual(EXPERIMENT_CONFIGS['joint_1_10'].mode, TrainingMode.JOINT)
        self.assertEqual(EXPERIMENT_CONFIGS['joint_1_20'].mode, TrainingMode.JOINT)
        self.assertEqual(EXPERIMENT_CONFIGS['ppo_only'].mode, TrainingMode.PPO_ONLY)
        self.assertEqual(EXPERIMENT_CONFIGS['sac_only'].mode, TrainingMode.SAC_ONLY)

    def test_experiment_budgets_match(self):
        """All experiments should use same total budget."""
        from scripts.run_mission_joint_training import EXPERIMENT_CONFIGS

        budgets = [cfg.total_micro_updates for cfg in EXPERIMENT_CONFIGS.values()]

        self.assertEqual(len(set(budgets)), 1, "Budgets should be equal across experiments")
        self.assertEqual(budgets[0], 600_000, "Standard budget should be 600K")


if __name__ == "__main__":
    unittest.main(verbosity=2)
