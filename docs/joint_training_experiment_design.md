# Joint Training Experiment Design: HRL_UAV Hierarchical Reinforcement Learning

**Version**: 1.0  
**Date**: 2026-09-14  
**Author**: Claude Fable 5  
**Status**: Design Draft for Review

---

## 1. Problem Statement and Experimental Goals

### 1.1 Current Bottlenecks (Confirmed by Diagnostics)

| Bottleneck | Evidence | Impact |
|------------|----------|--------|
| PPO trajectory tracking | centroid distance 90m vs 5m (Predictive oracle) | Macro policy fails to position UAVs for load balancing |
| Joint training value | No controlled comparison exists | Cannot justify hierarchical complexity |
| Budget matching | Different experiments have different update counts | Invalid comparison |

### 1.2 Experimental Questions

1. **Q1**: Can joint training improve PPO centroid tracking from 90m to <30m?
2. **Q2**: What is the optimal update frequency ratio between macro (PPO) and micro (SAC)?
3. **Q3**: Does joint training provide significant J improvement over single-layer trained baselines?
4. **Q4**: Is the HRL overhead justified by performance gains?

---

## 2. Hierarchical Architecture Specification

### 2.1 Two-Timescale Update Mechanism

```
PPO (Macro): Slow update, long-horizon value estimation
  - Action: UAV displacement (6D: dx, dy per UAV)
  - State: Global load distribution, queue backlog, centroid position
  - Update frequency: Every K_macro macro-steps (K_macro ∈ {5, 10, 20})
  - Value function: GAE(λ=0.95) over macro-steps

SAC (Micro): Fast update, immediate reward response
  - Action: Routing matrix (18D: 2 UAVs × 3 priorities × 3 destinations)
  - State: Local queues, link qualities, macro goal embedding
  - Update frequency: Every micro-step (after each 0.5s frame)
  - Off-policy: Experience replay with batch_size=256
```

### 2.2 Update Frequency Ratio Design

| Configuration | Ratio (PPO:SAC) | PPO Update Interval | Rationale |
|--------------|-----------------|---------------------|-----------|
| **J-1:5** | 1:5 | 5 macros (~50s) | Aggressive macro learning |
| **J-1:10** | 1:10 | 10 macros (~100s) | Balanced (recommended) |
| **J-1:20** | 1:20 | 20 macros (~200s) | Conservative macro |
| **J-1:30** | 1:30 | 30 macros (~300s) | Matches window boundary |

**Recommended Default**: J-1:10 (10 macros per PPO update)

### 2.3 Synchronization Strategy

```python
class JointUpdateController:
    """Controls update frequency ratio between PPO and SAC."""
    
    def __init__(self, macro_interval: int = 10):
        self.macro_interval = macro_interval
        self.macro_step_counter = 0
        self.ppo_update_threshold = macro_interval  # PPO updates every N macros
    
    def step(self, macro_completed: bool) -> tuple[bool, bool]:
        """Returns (should_update_ppo, should_update_sac)."""
        if macro_completed:
            self.macro_step_counter += 1
        
        update_ppo = (self.macro_step_counter % self.ppo_update_threshold == 0 
                      and self.macro_step_counter >= self.ppo_update_threshold)
        update_sac = True  # SAC updates every micro-step
        
        return update_ppo, update_sac
```

---

## 3. Experimental Matrix

### 3.1 Complete Factorial Design

| ID | Name | PPO Training | SAC Training | Update Ratio | Total Updates | Duration Est. |
|----|------|--------------|--------------|--------------|---------------|---------------|
| **E0** | Hover Baseline | None | None | N/A | 0 | 0.5h |
| **E1** | Fixed PPO + Train SAC | Frozen | Train | N/A (SAC only) | 60K | 4h |
| **E2** | Train PPO + Fixed SAC | Train | Frozen | N/A (PPO only) | 60 | 4h |
| **E3** | Joint-1:5 | Train | Train | 1:5 | 60 PPOs + 300K SACs | 6h |
| **E4** | **Joint-1:10** | Train | Train | 1:10 | 60 PPOs + 600K SACs | 8h |
| **E5** | Joint-1:20 | Train | Train | 1:20 | 60 PPOs + 1.2M SACs | 12h |
| **E6** | Predictive Oracle | Rule-based | Rule-based | N/A | 0 | 0.5h |
| **E7** | Xu-CVX-R | CVX solver | CVX solver | N/A | 0 | 1h |
| **E8** | Single-layer SAC (budget-matched) | None | Train | N/A | 600K | 8h |

### 3.2 Budget Matching Protocol

To ensure fair comparison, all training experiments use **equivalent total gradient steps**:

```
Budget Unit (BU) = 600,000 micro-updates

E1: 60K SAC updates (10% BU) + PPO frozen
E2: 60 PPO updates + SAC frozen
E4: 60 PPO + 600K SAC = 1.0 BU (reference)
E8: 600K SAC = 1.0 BU (single-layer baseline)
```

**Validation Scenes**: (33600, 33601) - same as existing experiments  
**Test Scenes**: (33610, 33611, 33612, range(33700, 33710)) - held-out  
**Training Scenes**: (72000, 72200) - same as PPO continuation

---

## 4. Evaluation Metrics Framework

### 4.1 Primary Metrics

| Metric | Unit | Weight in J | Target |
|--------|------|-------------|--------|
| **J (weighted cost)** | - | Primary | Minimize |
| J_holding | - | 0.5 | Minimize |
| J_energy | - | 0.5 | Minimize |

### 4.2 Decomposition Metrics

| Category | Metrics |
|----------|---------|
| **Delay** | task_delay_weighted_ms, task_delay_high_ms, task_delay_low_ms |
| **Queue** | queue_total_mbit, mean_queue_total_mbit, late_queue_slope_mbit_s |
| **Energy** | energy_total_j, power_w_mean |
| **Completion** | task_completion_rate, unfinished |

### 4.3 Trajectory Quality Metrics

| Metric | Unit | Target | Diagnostic Value |
|--------|------|--------|------------------|
| **centroid_distance_mean** | m | <30m | PPO tracking quality |
| centroid_distance_p90 | m | <60m | Worst-case tracking |
| speed_mps_mean | m/s | 5-15 | Movement efficiency |
| actual_at_20m_row_fraction | % | >95% | Safety compliance |

### 4.4 Fairness Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **High-priority delay gap** | task_delay_high_ms - task_delay_low_ms | <50ms |
| **Priority completion gap** | completion_rate_high - completion_rate_low | <5% |

### 4.5 Statistical Reporting

- **Primary**: Mean ± std over 3 PPO seeds (201, 202, 203)
- **Within-seed**: Mean over validation scenes (33600, 33601)
- **Significance**: Paired t-test vs E4, p<0.05

---

## 5. Implementation Architecture

### 5.1 New Module: `mission_joint_training.py`

```python
"""
Joint HRL Training Experiment Module

Design Principles:
1. Strict budget matching across experiments
2. Deterministic evaluation with fixed RNG
3. Incremental checkpointing every 10 PPO updates
4. Graceful interruption recovery
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
import numpy as np
import torch

class TrainingMode(Enum):
    JOINT = "joint"
    PPO_ONLY = "ppo_only"
    SAC_ONLY = "sac_only"
    BASELINE_HOVER = "hover"
    BASELINE_PREDICTIVE = "predictive"
    BASELINE_CVX = "cvx"

@dataclass
class JointTrainingConfig:
    # Budget matching
    total_micro_updates: int = 600_000
    ppo_update_interval: int = 10  # macros between PPO updates
    
    # PPO settings (from PPO_SETTINGS in mission_ppo_stage4a)
    ppo_initial_std: float = 1/3
    ppo_final_std: float = 1/30
    ppo_std_decay_windows: int = 200
    ppo_update_macros: int = 300
    ppo_epochs: int = 10
    ppo_actor_lr: float = 3e-4
    ppo_critic_lr: float = 1e-3
    ppo_clip: float = 0.2
    ppo_target_kl: float = 0.03
    
    # SAC settings (from learning.py defaults)
    sac_actor_lr: float = 3e-4
    sac_critic_lr: float = 3e-4
    sac_entropy_lr: float = 3e-4
    tau: float = 0.005
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    replay_warmup: int = 1000
    
    # Training control
    mode: TrainingMode = TrainingMode.JOINT
    checkpoint_interval: int = 10  # PPO updates
    evaluation_interval: int = 5   # PPO updates
    early_stopping_patience: int = 20  # PPO updates
    
    # Random seeds
    ppo_seeds: tuple = (201, 202, 203)
    sac_seed: int = 101


@dataclass
class JointTrainingState:
    """Tracks training progress for budget matching."""
    micro_updates: int = 0
    ppo_updates: int = 0
    macros_collected: int = 0
    episodes_completed: int = 0
    
    # For budget matching
    ppo_macros_since_update: int = 0
    target_ppo_updates: int = 60  # For 600K micro budget with 1:10 ratio
    
    def budget_utilization(self) -> float:
        """Fraction of target micro-updates completed."""
        return self.micro_updates / 600_000
    
    def ppo_ready(self, interval: int) -> bool:
        """Check if PPO should update based on interval."""
        return self.ppo_macros_since_update >= interval
```

### 5.2 Core Training Loop

```python
class JointTrainer:
    """Joint HRL trainer with configurable update frequencies."""
    
    def __init__(self, config: JointTrainingConfig, env, policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.state = JointTrainingState()
        self.best_J = float('inf')
        self.patience_counter = 0
        self.checkpoints = []
    
    def train_epoch(self, scene: int, macros: int = 90) -> dict:
        """Train for one scene (900s simulation)."""
        self.policy.macro_rollout.clear()
        episode_metrics = []
        
        for macro in range(macros):
            # === MICRO STEP ===
            macro_state = self.env.macro_observation()
            macro_action, logprob = self.policy.macro_action(
                macro_state, 
                deterministic=False,
                enabled=(self.config.mode in [TrainingMode.JOINT, TrainingMode.PPO_ONLY])
            )
            
            self.env.start_macro(macro_action)
            micro_rewards = []
            
            for frame in range(20):  # 20 service frames per macro
                micro_state = self.env.prepare_service()
                micro_action = self.policy.sac.sac_actor.choose_action(
                    micro_state, deterministic=False
                )
                
                reading = self.env.step_frame(micro_action)
                micro_rewards.append(reading['reward'])
                
                # Store transition for SAC (off-policy)
                if self.config.mode in [TrainingMode.JOINT, TrainingMode.SAC_ONLY]:
                    self.policy.store(
                        micro_state, micro_action, 
                        reading['route'], reading['reward'],
                        reading['next_state'],
                        reading.get('discount', 0.99)
                    )
                    self.state.micro_updates += 1
                
                # SAC update (every micro-step for joint mode)
                if (self.config.mode == TrainingMode.JOINT and 
                    self.state.micro_updates >= self.config.replay_warmup):
                    self.policy.update_sac()
                
                if frame < 19:
                    self.env.prepare_service()
            
            # Compute discounted macro reward
            gamma = self.config.get('gamma_per_resource_frame', 0.99)
            macro_reward = sum(gamma**i * r for i, r in enumerate(micro_rewards)) / 20
            
            # Advance macro clock
            self.env.advance_macro_clock()
            next_macro_state = self.env.macro_observation()
            
            # Store macro transition for PPO (on-policy)
            if self.config.mode in [TrainingMode.JOINT, TrainingMode.PPO_ONLY]:
                self.policy.macro_rollout.append((
                    macro_state.copy(), macro_action.copy(), logprob,
                    macro_reward, next_macro_state.copy(), 1.0
                ))
                self.state.ppo_macros_since_update += 1
            
            # === PPO UPDATE (conditional on interval) ===
            if (self.config.mode == TrainingMode.JOINT and 
                self.state.ppo_ready(self.config.ppo_update_interval)):
                progress = self.state.ppo_updates / self.config.target_ppo_updates
                self.policy.update_ppo(progress=progress)
                self.state.ppo_updates += 1
                self.state.ppo_macros_since_update = 0
            
            episode_metrics.append(self.env.metrics())
        
        self.state.episodes_completed += 1
        return self._aggregate_metrics(episode_metrics)
    
    def evaluate(self, scenes: tuple, deterministic: bool = True) -> dict:
        """Evaluate on validation scenes."""
        results = []
        for scene in scenes:
            metrics = self._evaluate_single(scene, deterministic)
            results.append(metrics)
        return self._average_results(results)
```

### 5.3 Budget-Matched Baseline Implementations

```python
class PPOTrainingOnly(JointTrainer):
    """E2: Train PPO with frozen SAC - validates PPO tracking improvement."""
    
    def __init__(self, config, env, policy):
        config.mode = TrainingMode.PPO_ONLY
        super().__init__(config, env, policy)
        # Freeze SAC
        for param in self.policy.sac.sac_actor.parameters():
            param.requires_grad_(False)
        for param in self.policy.sac_critic.parameters():
            param.requires_grad_(False)


class SACTrainingOnly(JointTrainer):
    """E1: Train SAC with frozen PPO - validates SAC routing improvement."""
    
    def __init__(self, config, env, policy):
        config.mode = TrainingMode.SAC_ONLY
        super().__init__(config, env, policy)
        # Freeze PPO
        for param in self.policy.macro_actor.parameters():
            param.requires_grad_(False)
        for param in self.policy.macro_critic.parameters():
            param.requires_grad_(False)
    
    def train_epoch(self, scene, macros=90):
        """Override: no macro rollout, only micro updates."""
        self.state.micro_updates = 0
        episode_metrics = []
        
        for macro in range(macros):
            # Fixed macro action (e.g., hover or current centroid)
            macro_state = self.env.macro_observation()
            macro_action, _ = self.policy.macro_action(
                macro_state, deterministic=True, enabled=False
            )
            self.env.start_macro(macro_action)
            
            for frame in range(20):
                micro_state = self.env.prepare_service()
                micro_action = self.policy.sac.sac_actor.choose_action(
                    micro_state, deterministic=False
                )
                reading = self.env.step_frame(micro_action)
                
                self.policy.store(...)
                self.state.micro_updates += 1
                self.policy.update_sac()
                
                if frame < 19:
                    self.env.prepare_service()
            
            self.env.advance_macro_clock()
            episode_metrics.append(self.env.metrics())
        
        return self._aggregate_metrics(episode_metrics)
```

---

## 6. Experiment Runner Scripts

### 6.1 Main Runner: `run_mission_joint_training.py`

```python
"""
Joint Training Experiment Runner

Usage:
    # Joint training (recommended)
    python scripts/run_mission_joint_training.py --output results/joint_training --experiment joint --run-authorized
    
    # PPO-only baseline
    python scripts/run_mission_joint_training.py --output results/ppo_only --experiment ppo_only --run-authorized
    
    # Resume interrupted run
    python scripts/run_mission_joint_training.py --output results/joint_training --experiment joint --run-authorized --resume
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from hrl_uav.mission_joint_training import (
    JointTrainingConfig, JointTrainer, PPOTrainingOnly, SACTrainingOnly,
    TrainingMode, BASELINES, TRAINING_SCENES, VALIDATION_SCENES
)
from hrl_uav.mission_smoke import ROOT, mission_config
from hrl_uav.mission_ppo_stage4a import load_sac_checkpoint, load_ppo_checkpoint


def run(output_dir: Path, experiment: str, resume: bool = False):
    """Execute joint training experiment."""
    
    # Load base configuration from SAC checkpoint
    sac_source = ROOT / "results/mission_rpgm_v1_pilot"
    ppo_source = ROOT / "results/mission_ppo_stage4a_qualification"
    
    # Experiment configurations
    configs = {
        'joint': JointTrainingConfig(
            mode=TrainingMode.JOINT,
            ppo_update_interval=10,
            total_micro_updates=600_000,
        ),
        'joint_1_5': JointTrainingConfig(
            mode=TrainingMode.JOINT,
            ppo_update_interval=5,
            total_micro_updates=600_000,
        ),
        'joint_1_20': JointTrainingConfig(
            mode=TrainingMode.JOINT,
            ppo_update_interval=20,
            total_micro_updates=600_000,
        ),
        'ppo_only': JointTrainingConfig(
            mode=TrainingMode.PPO_ONLY,
            ppo_update_interval=10,
            total_micro_updates=600_000,
        ),
        'sac_only': JointTrainingConfig(
            mode=TrainingMode.SAC_ONLY,
            ppo_update_interval=10,
            total_micro_updates=600_000,
        ),
    }
    
    config = configs[experiment]
    
    # Training loop with budget matching
    for seed in config.ppo_seeds:
        print(f"\n=== Training seed {seed} ===")
        
        # Initialize policy from checkpoints
        policy = load_joint_policy(
            sac_source, ppo_source, seed, config
        )
        
        trainer = create_trainer(config.mode, config, env, policy)
        
        # Main training loop
        for scene_idx, scene in enumerate(TRAINING_SCENES):
            print(f"Scene {scene} ({scene_idx+1}/{len(TRAINING_SCENES)})")
            
            # Train epoch
            train_metrics = trainer.train_epoch(scene)
            
            # Periodic evaluation
            if trainer.state.ppo_updates % config.evaluation_interval == 0:
                eval_metrics = trainer.evaluate(VALIDATION_SCENES)
                print(f"  J: {eval_metrics['J_mean']:.4f} ± {eval_metrics['J_std']:.4f}")
                print(f"  Centroid: {eval_metrics['centroid_distance_mean']:.1f}m")
                
                # Checkpointing
                if eval_metrics['J_mean'] < trainer.best_J:
                    trainer.best_J = eval_metrics['J_mean']
                    trainer.patience_counter = 0
                    save_checkpoint(output_dir, seed, trainer, 'best')
                else:
                    trainer.patience_counter += 1
            
            # Budget-matched checkpointing
            if trainer.state.ppo_updates % config.checkpoint_interval == 0:
                save_checkpoint(output_dir, seed, trainer, f'update{trainer.state.ppo_updates}')
            
            # Early stopping
            if trainer.patience_counter >= config.early_stopping_patience:
                print(f"  Early stopping at update {trainer.state.ppo_updates}")
                break
        
        # Final evaluation
        final_metrics = trainer.evaluate(VALIDATION_SCENES)
        save_results(output_dir, seed, final_metrics)
    
    # Aggregate and compare
    aggregate_results(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", choices=['joint', 'joint_1_5', 'joint_1_20', 'ppo_only', 'sac_only'])
    parser.add_argument("--run-authorized", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    if not args.run_authorized:
        parser.error("Explicit --run-authorized is required")
    
    run(args.output, args.experiment, args.resume)
```

### 6.2 Comparison Script: `run_mission_joint_comparison.py`

```python
"""
Comparative analysis across all experimental conditions.

Generates:
1. Main comparison table (J, decomposition, trajectory quality)
2. Statistical significance tests
3. Learning curve plots
4. Ablation analysis
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

from hrl_uav.mission_joint_training import load_all_results, compare_experiments


def main(output_dir: Path):
    # Load all experiment results
    experiments = {
        'E1_FixedPPO_TrainSAC': load_results('results/sac_only'),
        'E2_TrainPPO_FixedSAC': load_results('results/ppo_only'),
        'E4_Joint_1_10': load_results('results/joint_training'),
        'E6_Predictive': load_results('results/baseline_predictive'),
        'E7_XuCVXR': load_results('results/xu_cvx_r'),
    }
    
    # Generate comparison table
    table = compare_experiments(experiments)
    print("\n" + "="*80)
    print("MAIN COMPARISON TABLE")
    print("="*80)
    print(table.to_string(index=False))
    
    # Statistical tests
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE (vs Joint-1:10)")
    print("="*80)
    joint_results = experiments['E4_Joint_1_10']
    
    for name, results in experiments.items():
        if name == 'E4_Joint_1_10':
            continue
        
        t_stat, p_val = stats.ttest_rel(
            joint_results['J_mean'], results['J_mean']
        )
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        
        print(f"{name}: ΔJ = {results['J_mean'] - joint_results['J_mean']:+.4f}, "
              f"p = {p_val:.4f} {significance}")
    
    # Save comparison
    table.to_csv(output_dir / 'comparison_table.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
```

---

## 7. Validation and Testing

### 7.1 Unit Tests: `test_mission_joint_training.py`

```python
"""Test suite for joint training experiment module."""

import unittest
from hrl_uav.mission_joint_training import (
    JointTrainingConfig, JointTrainingState, TrainingMode
)

class TestBudgetMatching(unittest.TestCase):
    """Verify budget matching is exact across conditions."""
    
    def test_joint_1_10_budget(self):
        """600K micro updates should yield exactly 60 PPO updates."""
        config = JointTrainingConfig(
            mode=TrainingMode.JOINT,
            ppo_update_interval=10,
            total_micro_updates=600_000
        )
        state = JointTrainingState()
        
        # Simulate training
        target_ppo = 600_000 // 10  # 60,000 macros / 10 interval
        self.assertEqual(target_ppo, 60)
    
    def test_budget_utilization(self):
        """Budget fraction should be computed correctly."""
        state = JointTrainingState(micro_updates=300_000)
        self.assertAlmostEqual(state.budget_utilization(), 0.5)


class TestUpdateController(unittest.TestCase):
    """Test update frequency ratio enforcement."""
    
    def test_ppo_ready_at_interval(self):
        """PPO should update exactly at configured interval."""
        state = JointTrainingState()
        interval = 10
        
        for i in range(1, 21):
            state.ppo_macros_since_update = i
            ready = state.ppo_ready(interval)
            expected = (i % interval == 0 and i >= interval)
            self.assertEqual(ready, expected, f"Step {i}: expected {expected}, got {ready}")


if __name__ == "__main__":
    unittest.main()
```

### 7.2 Smoke Test Protocol

```python
def smoke_test():
    """Quick verification before full run."""
    print("Running smoke test...")
    
    # 1. Verify checkpoint loading
    policy = load_joint_policy(sac_source, ppo_source, seed=201, config=default_config)
    assert policy is not None
    
    # 2. Verify single epoch runs
    trainer = JointTrainer(default_config, env, policy)
    metrics = trainer.train_epoch(scene=72000, macros=2)  # Only 2 macros
    assert 'J' in metrics
    
    # 3. Verify update ratio
    assert state.ppo_updates == 0  # 2 macros < 10 interval
    state.ppo_macros_since_update = 10
    assert state.ppo_ready(10)  # Now ready
    
    print("Smoke test PASSED")
```

---

## 8. Execution Plan

### 8.1 Phase 1: Module Implementation (1 day)
- [ ] Create `mission_joint_training.py`
- [ ] Create `run_mission_joint_training.py`
- [ ] Create `test_mission_joint_training.py`
- [ ] Implement smoke test

### 8.2 Phase 2: Baseline Execution (2 days)
- [ ] Run E1: Fixed PPO + Train SAC (sac_only)
- [ ] Run E2: Train PPO + Fixed SAC (ppo_only)
- [ ] Run E8: Single-layer SAC (budget-matched)

### 8.3 Phase 3: Joint Training (3 days)
- [ ] Run E4: Joint-1:10 (primary)
- [ ] Run E3: Joint-1:5 (ablation)
- [ ] Run E5: Joint-1:20 (ablation)

### 8.4 Phase 4: Comparison (1 day)
- [ ] Run E6: Predictive Oracle
- [ ] Run E7: Xu-CVX-R
- [ ] Execute comparison script
- [ ] Generate tables and plots

### 8.5 Total Estimated Runtime
- **Single GPU/CPU**: ~80 hours (3.3 days)
- **Recommended**: Run in parallel on separate machines

---

## 9. Expected Outcomes and Hypotheses

### 9.1 Hypotheses

| Hypothesis | Expected Outcome | Evidence Needed |
|------------|------------------|-----------------|
| H1 | Joint-1:10 improves centroid tracking | centroid_distance < 30m |
| H2 | Joint > independent layers | J_joint < min(J_ppo_only, J_sac_only) |
| H3 | 1:10 ratio is optimal | J_1:10 < J_1:5 and J_1:10 < J_1:20 |
| H4 | PPO learns meaningful trajectories | centroid_distance_p90 < 60m |

### 9.2 Interpretation Guidelines

| Finding | Implication |
|---------|-------------|
| Joint-1:10 significantly better than all baselines | HRL architecture validated |
| Joint-1:10 ≈ Single-layer SAC | Hierarchical overhead not justified |
| PPO-only ≈ Hover baseline | PPO provides no value without learned micro |
| SAC-only < Joint | Joint provides value beyond SAC alone |

---

## 10. Integration with Existing Codebase

### 10.1 Reused Components

| Component | Source | Usage |
|-----------|--------|-------|
| `DirectHRL` | `learning.py` | Base policy class |
| `Stage4Meter` | `mission_ppo_stage4a.py` | Evaluation interface |
| `SAC checkpoint` | `mission_rpgm_v1_pilot` | Frozen micro policy |
| `PPO checkpoint` | `mission_ppo_stage4a_qualification` | Initial macro policy |
| `METRICS` | `mission_ppo_qualification.py` | Standard metric set |

### 10.2 Key Modifications Required

1. **`learning.py`**: Add optional `ppo_update_interval` parameter
2. **`mission_ppo_stage4a.py`**: Extend to support joint mode
3. **`env.py`**: No changes required (observation/action unchanged)

### 10.3 Backward Compatibility

- All existing checkpoints remain valid
- New experiments use separate output directories
- No modification to evaluation protocols

---

## 11. Appendix: Hyperparameter Sensitivity Analysis

### 11.1 PPO Sensitivity (for future exploration)

| Parameter | Current | Tested Range | Notes |
|-----------|---------|--------------|-------|
| ppo_epochs | 10 | 5-20 | More epochs may overfit |
| ppo_clip | 0.2 | 0.1-0.3 | Standard range |
| ppo_target_kl | 0.03 | 0.01-0.1 | KL early stopping |
| ppo_actor_lr | 3e-4 | 1e-4-1e-3 | Critical for stability |

### 11.2 SAC Sensitivity (for future exploration)

| Parameter | Current | Tested Range | Notes |
|-----------|---------|--------------|-------|
| batch_size | 256 | 128-512 | Memory vs stability |
| tau | 0.005 | 0.001-0.01 | Target network speed |
| entropy_lr | 3e-4 | 1e-4-1e-3 | Exploration-exploitation |

---

**Document Version**: 1.0  
**Next Review**: After Phase 1 implementation
