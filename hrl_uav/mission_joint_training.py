"""
Joint HRL Training Experiment Module

Design Principles:
1. Strict budget matching across experiments
2. Deterministic evaluation with fixed RNG
3. Incremental checkpointing every N PPO updates
4. Graceful interruption recovery

This module implements the experiment design documented in:
    docs/joint_training_experiment_design.md
"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import learning
from .agents.ppo_agent import ActorNet as MacroActorNet, CriticNet as MacroCriticNet
from .agents.sac_agent import Actor, Critic, Entroy, Memory, device
from .experiment import append_csv, isolated_rng, write_json
from .mission_ppo_stage4a import (
    BASELINES, MONITORS, PPO_SETTINGS, PPO_SEEDS, TRAIN_SCENES,
    VALIDATION_SCENES, MissionMacroPPO, Stage4Meter, collect_stream,
    frozen_sac_guard, stage_config
)
from .mission_ppo_qualification import METRICS, read_json, transaction
from .mission_smoke import ROOT, load_source, load_spec, mission_config
from .priority_grid import file_hash
from .resumption import capture_rng, restore_rng, state_equal
from .scene_prescreen import assert_crn_equal
from .trajectory_check import ReadOnlySAC, forbidden, no_learning
from .wireless_check import ENERGY, PRIORITY_MEASURES


# =============================================================================
# Configuration and State
# =============================================================================

TRAIN_SCENES_JT = tuple(range(72000, 72200))  # 200 scenes for joint training
MONITORS_JT = (0, 30, 60, 90, 120, 180, 240, 300, 360, 420, 480, 540, 600)
BASELINES_JT = ("hover", "current_centroid", "progressive_centroid", "predictive_centroid")


class TrainingMode(Enum):
    """Training mode for joint HRL experiments."""
    JOINT = "joint"                    # Both PPO and SAC train
    PPO_ONLY = "ppo_only"             # Train PPO, freeze SAC
    SAC_ONLY = "sac_only"             # Train SAC, freeze PPO
    BASELINE_HOVER = "hover"          # Fixed hover policy
    BASELINE_PREDICTIVE = "predictive"  # Oracle centroid tracking
    BASELINE_CVX = "cvx"              # CVX optimization


@dataclass
class JointTrainingConfig:
    """Configuration for joint HRL training experiments."""

    # Budget matching
    total_micro_updates: int = 600_000
    ppo_update_interval: int = 10  # macros between PPO updates

    # PPO settings (from mission_ppo_stage4a)
    ppo_initial_std: float = 1/3
    ppo_final_std: float = 1/30
    ppo_std_decay_windows: int = 200
    ppo_update_macros: int = 300
    ppo_epochs: int = 10
    ppo_actor_lr: float = 3e-4
    ppo_critic_lr: float = 1e-3
    ppo_clip: float = 0.2
    ppo_target_kl: float = 0.03

    # SAC settings
    sac_actor_lr: float = 3e-4
    sac_critic_lr: float = 3e-4
    sac_entropy_lr: float = 3e-4
    tau: float = 0.005
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    replay_warmup: int = 1000
    initial_alpha: float = 0.2
    initial_sac_log_std: float = -0.5
    grad_clip: float = 10.0
    gamma: float = 0.99
    sac_policy_delay: int = 2

    # Training control
    mode: TrainingMode = TrainingMode.JOINT
    checkpoint_interval: int = 10  # PPO updates
    evaluation_interval: int = 5   # PPO updates
    early_stopping_patience: int = 20  # PPO updates

    # Random seeds
    ppo_seeds: tuple = (201, 202, 203)
    sac_seed: int = 101

    # Scene configuration
    macros_per_scene: int = 90  # 900s / 10s per macro
    frames_per_macro: int = 20  # 10s / 0.5s per frame

    def get_ppo_settings(self):
        """Return PPO settings dict matching mission_ppo_stage4a format."""
        return dict(
            ppo_initial_std=self.ppo_initial_std,
            ppo_final_std=self.ppo_final_std,
            ppo_std_decay_windows=self.ppo_std_decay_windows,
            ppo_update_macros=self.ppo_update_macros,
            ppo_epochs=self.ppo_epochs,
            ppo_actor_lr=self.ppo_actor_lr,
            ppo_critic_lr=self.ppo_critic_lr,
            ppo_clip=self.ppo_clip,
            ppo_target_kl=self.ppo_target_kl,
        )


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
        return (self.ppo_macros_since_update >= interval
                and self.ppo_macros_since_update % interval == 0)

    def sac_ready(self, warmup: int) -> bool:
        """Check if SAC should update based on replay warmup."""
        return self.micro_updates >= warmup


# =============================================================================
# Joint Policy (PPO + SAC)
# =============================================================================

class JointPolicy(learning.DirectHRL):
    """
    Extended DirectHRL with configurable update frequency control.

    Key differences from DirectHRL:
    - External control over when PPO/SAC updates occur
    - Support for frozen modes (PPO_ONLY, SAC_ONLY)
    - Budget tracking for comparison fairness
    """

    def __init__(self, config, sac_checkpoint, ppo_checkpoint=None, *, seed=101):
        # Load checkpoint to get config
        path = Path(sac_checkpoint)
        if path.suffix == '.gz':
            import gzip, pickle
            with gzip.open(path, 'rb') as f:
                saved = pickle.load(f)
        else:
            saved = torch.load(path, map_location='cpu', weights_only=False)

        # Get base config from checkpoint
        base_cfg = mission_config(saved["config"])

        # Initialize as DirectHRL
        super().__init__(
            state_dim=68,  # micro state dim
            micro_action_dim=18,
            macro_action_dim=6,
            config=base_cfg,
            macro_state_dim=67
        )

        # Load SAC from checkpoint
        self._load_sac(sac_checkpoint)

        # Load PPO from checkpoint if provided
        if ppo_checkpoint is not None:
            self._load_ppo(ppo_checkpoint)

        # Override with joint training config
        self.cfg.update(config.get_ppo_settings())
        self.cfg['seed'] = seed

        # Joint training state
        self.joint_state = JointTrainingState()
        self._ppo_frozen = False
        self._sac_frozen = False
        self.seed = seed  # For summarize() compatibility

    def _load_sac(self, checkpoint_path):
        """Load SAC from checkpoint."""
        import gzip
        import pickle

        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"SAC checkpoint not found: {checkpoint_path}")

        # Support both .pkl.gz and .pt formats
        if path.suffix == '.gz':
            with gzip.open(path, 'rb') as f:
                ckpt = pickle.load(f)
        else:
            # .pt format (PyTorch checkpoint)
            ckpt = torch.load(path, map_location='cpu', weights_only=False)

        self.sac_actor.action_net.load_state_dict(ckpt['sac_actor'])
        self.sac_critic.critic_v.load_state_dict(ckpt['sac_critic'])
        self.sac_critic.target_critic_v.load_state_dict(ckpt['sac_target'])

        # Handle log_alpha - can be tensor or float
        log_alpha_val = ckpt['log_alpha']
        if isinstance(log_alpha_val, torch.Tensor):
            log_alpha_val = log_alpha_val.item()
        self.entropy.log_alpha.data.fill_(log_alpha_val)

    def _load_ppo(self, checkpoint_path):
        """Load PPO from checkpoint (for continuation)."""
        # For fresh PPO, we initialize from scratch (matches mission_ppo_stage4a)
        pass

    def set_mode(self, mode: TrainingMode):
        """Set training mode and freeze appropriate networks."""
        self.mode = mode
        self._ppo_frozen = mode in (TrainingMode.SAC_ONLY, TrainingMode.BASELINE_HOVER,
                                      TrainingMode.BASELINE_PREDICTIVE, TrainingMode.BASELINE_CVX)
        self._sac_frozen = mode in (TrainingMode.PPO_ONLY, TrainingMode.BASELINE_HOVER,
                                     TrainingMode.BASELINE_PREDICTIVE, TrainingMode.BASELINE_CVX)

        # Freeze gradients
        for param in self.macro_actor.parameters():
            param.requires_grad_(not self._ppo_frozen)
        for param in self.macro_critic.parameters():
            param.requires_grad_(not self._ppo_frozen)

        if self._sac_frozen:
            for param in self.sac_actor.action_net.parameters():
                param.requires_grad_(False)
            for param in self.sac_critic.critic_v.parameters():
                param.requires_grad_(False)
        else:
            for param in self.sac_actor.action_net.parameters():
                param.requires_grad_(True)
            for param in self.sac_critic.critic_v.parameters():
                param.requires_grad_(True)

    def macro_action(self, state, enabled=True, deterministic=False):
        """Get macro (PPO) action. Returns zero if PPO disabled/frozen."""
        if not enabled or self._ppo_frozen:
            return np.zeros(6, dtype=np.float32), 0.0
        return super().macro_action(state, deterministic=deterministic)

    def update_ppo_if_ready(self, interval: int, progress: float = 0.0):
        """Conditionally update PPO based on interval."""
        if not self.joint_state.ppo_ready(interval):
            return

        if not self._ppo_frozen:
            self.update_ppo(progress=progress)
            self.joint_state.ppo_updates += 1

        self.joint_state.ppo_macros_since_update = 0

    def update_sac_if_ready(self, warmup: int):
        """Conditionally update SAC based on replay."""
        if self._sac_frozen or not self.joint_state.sac_ready(warmup):
            return

        self.update_sac()
        self.joint_state.micro_updates += 1


# =============================================================================
# Frozen Baseline Policies
# =============================================================================

class HoverPolicy:
    """Fixed hover policy for baseline comparison."""

    def __init__(self, config):
        self.config = config

    def macro_action(self, state, enabled=True, deterministic=True):
        return np.zeros(6, dtype=np.float32), 0.0


class PredictiveCentroidPolicy:
    """Oracle centroid tracking policy for upper bound."""

    def __init__(self, config):
        self.config = config
        self.meter = None

    def set_meter(self, meter):
        self.meter = meter

    def macro_action(self, state, enabled=True, deterministic=True):
        """Compute centroid from actual state (oracle)."""
        if self.meter is None:
            return np.zeros(6, dtype=np.float32), 0.0

        # Get current positions
        positions = self.meter.env.uav_positions  # Shape: (3, 2)

        # Get task positions to compute true centroid
        tasks = self.meter.env.tasks
        if len(tasks) == 0:
            return np.zeros(6, dtype=np.float32), 0.0

        # Weighted centroid based on task priorities
        total_weight = sum(t.priority for t in tasks)
        if total_weight == 0:
            return np.zeros(6, dtype=np.float32), 0.0

        centroid = np.zeros(2)
        for task in tasks:
            centroid += task.position * (task.priority / total_weight)

        # Compute displacement for each UAV
        actions = np.zeros(6, dtype=np.float32)
        for i, uav_pos in enumerate(positions):
            direction = centroid - uav_pos
            distance = np.linalg.norm(direction)

            if distance > 1.0:  # Only move if far from centroid
                max_step = 20.0  # Max 20m per macro step
                direction = direction / distance * min(distance, max_step)
                actions[i*2:(i+1)*2] = direction

        return actions, 0.0


# =============================================================================
# Evaluation Interface
# =============================================================================

class ObservedPolicy:
    """Wraps a policy for evaluation with trajectory tracking."""

    def __init__(self, policy, baseline_type=None):
        self.policy = policy
        self.baseline_type = baseline_type
        self.meter = None

    def macro_action(self, state, enabled=True, deterministic=True):
        if self.baseline_type == "predictive_centroid":
            return self.policy.macro_action(state, enabled, deterministic)
        elif self.baseline_type == "current_centroid":
            return self.policy.macro_action(state, enabled, deterministic)
        else:
            return self.policy.macro_action(state, enabled, deterministic)

    def set_meter(self, meter):
        self.policy.meter = meter
        self.meter = meter


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run(output_dir: Path, config: JointTrainingConfig, resume: bool = False):
    """
    Execute joint training experiment.

    Args:
        output_dir: Output directory for results
        config: JointTrainingConfig with experiment parameters
        resume: Whether to resume from interruption
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source checkpoints - use existing trained SAC at window2000
    sac_source = ROOT / "results/sac_only_wireless6_long_seed101/checkpoints"
    ppo_source = ROOT / "results/mission_ppo_stage4a_qualification"

    # Write protocol
    config_dict = vars(config).copy()
    config_dict['mode'] = config.mode.value  # Convert enum to string for JSON
    protocol_data = dict(
        stage=f"joint_training_{config.mode.value}",
        config=config_dict,
        source_sac=str(sac_source),
        source_ppo=str(ppo_source) if ppo_source.exists() else None,
        scenes=list(TRAIN_SCENES_JT),
        validation_scenes=list(VALIDATION_SCENES),
        monitors=list(MONITORS_JT),
        baselines=list(BASELINES_JT),
    )
    write_json(output_dir / "protocol.json", protocol_data)

    results = []

    for seed in config.ppo_seeds:
        print(f"\n{'='*60}")
        print(f"Training seed {seed}")
        print(f"{'='*60}")

        seed_dir = output_dir / f"seed{seed}"
        seed_dir.mkdir(exist_ok=True)

        # Initialize policy
        policy = _initialize_policy(config, sac_source, ppo_source, seed)
        policy.set_mode(config.mode)

        # Training loop
        state = JointTrainingState(target_ppo_updates=config.total_micro_updates // config.ppo_update_interval)
        best_J = float('inf')
        patience = 0

        for scene_idx, scene in enumerate(TRAIN_SCENES_JT):
            print(f"\nScene {scene} ({scene_idx+1}/{len(TRAIN_SCENES_JT)})")
            print(f"  Micro updates: {state.micro_updates}/{config.total_micro_updates}")
            print(f"  PPO updates: {state.ppo_updates}/{state.target_ppo_updates}")

            # Train one scene
            scene_metrics = _train_scene(
                config, policy, scene, state
            )

            # Periodic evaluation
            if state.ppo_updates > 0 and state.ppo_updates % config.evaluation_interval == 0:
                eval_metrics = _evaluate(config, policy, VALIDATION_SCENES)

                J_mean = eval_metrics['J_mean']
                centroid = eval_metrics.get('centroid_distance_mean', float('nan'))

                print(f"  Eval J: {J_mean:.4f} ± {eval_metrics['J_std']:.4f}")
                print(f"  Centroid dist: {centroid:.1f}m")

                # Track best
                if J_mean < best_J:
                    best_J = J_mean
                    patience = 0
                    _save_checkpoint(seed_dir, policy, state, 'best')
                else:
                    patience += 1

                # Early stopping
                if patience >= config.early_stopping_patience:
                    print(f"  Early stopping at PPO update {state.ppo_updates}")
                    break

                # Regular checkpoint
                if state.ppo_updates % config.checkpoint_interval == 0:
                    _save_checkpoint(seed_dir, policy, state, f'update{state.ppo_updates}')

            # Budget exhaustion check
            if state.micro_updates >= config.total_micro_updates:
                print(f"  Budget exhausted at {state.micro_updates} micro updates")
                break

        # Final evaluation
        final_metrics = _evaluate(config, policy, VALIDATION_SCENES)
        final_metrics['seed'] = seed
        final_metrics['best_J'] = best_J
        results.append(final_metrics)

        # Save final state
        _save_checkpoint(seed_dir, policy, state, 'final')
        _save_results(seed_dir, final_metrics)

    # Aggregate results
    agg = _aggregate_results(results)
    write_json(output_dir / "results.json", agg)

    return agg


def _initialize_policy(config, sac_source, ppo_source, seed):
    """Initialize joint policy from checkpoints."""
    # Find SAC checkpoint - try multiple patterns
    sac_window = 2000
    sac_ckpt = None

    # First try .pt files (new format)
    for pattern in (f"window{sac_window}.pt", "window2000.pt"):
        for path in (sac_source).glob(f"checkpoints/{pattern}"):
            sac_ckpt = path
            break
        if sac_ckpt:
            break

    # Then try .pkl.gz files (legacy format)
    if sac_ckpt is None:
        for path in (sac_source).glob("*.pkl.gz"):
            if "window2000" in str(path) or f"window{sac_window}" in str(path):
                sac_ckpt = path
                break

    # Try flat directory structure
    if sac_ckpt is None:
        for path in (sac_source).glob("*.pt"):
            if "window2000" in str(path) or f"window{sac_window}" in str(path):
                sac_ckpt = path
                break

    if sac_ckpt is None:
        raise FileNotFoundError(f"No SAC window2000 checkpoint found in {sac_source}")

    # Create policy (PPO initialized fresh for joint mode)
    policy = JointPolicy(
        config=config,
        sac_checkpoint=sac_ckpt,
        ppo_checkpoint=None,  # Fresh PPO initialization
        seed=seed
    )

    return policy


def _train_scene(config, policy, scene, state):
    """Train for one scene (900s simulation)."""
    # Load config from checkpoint for meter
    sac_dir = sac_source_dir()
    ckpt_path = sac_dir / "checkpoints" / "window2000.pt"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = mission_config(ckpt['config'])
    meter = Stage4Meter(cfg)
    meter.reset(scene, config.macros_per_scene)

    episode_rewards = []

    for macro in range(config.macros_per_scene):
        # Get macro observation
        macro_state = meter.macro_observation()

        # Sample macro action
        if config.mode == TrainingMode.JOINT:
            macro_action, logprob = policy.macro_action(macro_state, enabled=True, deterministic=False)
        elif config.mode == TrainingMode.PPO_ONLY:
            macro_action, logprob = policy.macro_action(macro_state, enabled=True, deterministic=False)
        elif config.mode == TrainingMode.SAC_ONLY:
            macro_action, _ = policy.macro_action(macro_state, enabled=False, deterministic=True)
        else:
            macro_action, _ = policy.macro_action(macro_state, enabled=False, deterministic=True)

        meter.start_macro(macro_action)
        state.macros_collected += 1
        policy.joint_state.ppo_macros_since_update += 1

        micro_rewards = []
        next_micro_state = None

        for frame in range(config.frames_per_macro):
            micro_state = meter.prepare_service() if frame == 0 else next_micro_state

            # Sample micro action
            if config.mode in (TrainingMode.JOINT, TrainingMode.SAC_ONLY):
                micro_action = policy.sac_actor.choose_action(micro_state, deterministic=False)
            else:
                micro_action = policy.sac_actor.choose_action(micro_state, deterministic=True)

            reading = meter.step_frame(micro_action, cfg["training"]["gamma_per_resource_frame"])
            micro_rewards.append(reading['reward'])

            # Get next state after the frame (prepare_service for next frame)
            if frame < config.frames_per_macro - 1:
                next_micro_state = meter.prepare_service()

            # Store transition for SAC
            if config.mode in (TrainingMode.JOINT, TrainingMode.SAC_ONLY):
                # Use prepare_service state as next_state
                store_next_state = next_micro_state if frame < config.frames_per_macro - 1 else meter.prepare_service()
                policy.store(
                    micro_state, micro_action,
                    reading['route'], reading['reward'],
                    store_next_state,
                    reading.get('discount', cfg["training"]["gamma_per_resource_frame"])
                )

                # Conditional SAC update
                policy.update_sac_if_ready(config.replay_warmup)

        # Store macro transition for PPO
        if config.mode in (TrainingMode.JOINT, TrainingMode.PPO_ONLY):
            gamma = cfg["training"]["gamma_per_resource_frame"]
            macro_reward = sum(gamma**i * r for i, r in enumerate(micro_rewards)) / len(micro_rewards)

            next_macro_state = meter.macro_observation()
            policy.macro_rollout.append((
                macro_state.copy(), macro_action.copy(), logprob,
                macro_reward, next_macro_state.copy(), 1.0
            ))

        # Conditional PPO update
        progress = state.ppo_updates / state.target_ppo_updates if state.target_ppo_updates > 0 else 0
        policy.update_ppo_if_ready(config.ppo_update_interval, progress)

        # Advance macro clock
        meter.advance_macro_clock()
        episode_rewards.extend(micro_rewards)

        # Budget check
        if state.micro_updates >= config.total_micro_updates:
            break

    state.episodes_completed += 1

    return {
        'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
        'total_reward': np.sum(episode_rewards) if episode_rewards else 0,
    }


def _evaluate(config, policy, scenes):
    """Evaluate on validation scenes."""
    # Load config from checkpoint for meter
    sac_dir = sac_source_dir()
    ckpt_path = sac_dir / "checkpoints" / "window2000.pt"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = mission_config(ckpt['config'])

    results = []
    for scene in scenes:
        meter = Stage4Meter(cfg)
        meter.reset(scene, MONITORS_JT[-1])  # Full 600s evaluation

        for macro in range(90):
            macro_state = meter.macro_observation()
            macro_action, _ = policy.macro_action(macro_state, deterministic=True, enabled=True)
            meter.start_macro(macro_action)

            for frame in range(config.frames_per_macro):
                micro_state = meter.prepare_service()
                micro_action = policy.sac_actor.choose_action(micro_state, deterministic=True)
                meter.step_frame(micro_action, cfg["training"]["gamma_per_resource_frame"])

                if frame < config.frames_per_macro - 1:
                    meter.prepare_service()

            meter.advance_macro_clock()

        # Build row dict directly from meter readings
        time_s = meter.readings[-1]["time_s"] if meter.readings else 600.0
        row = dict(
            policy="joint_hrl",
            seed=scene,
            duration_s=time_s,
        )

        # Add metrics from final reading
        if meter.readings:
            last = meter.readings[-1]
            row.update({k: last[k] for k in last.keys()})

        # Add uplink metrics
        if meter.uplink_readings:
            row["u2g_served_mbit"] = sum(r["served_mbit"] for r in meter.uplink_readings)
            row["u2g_tx_time_s"] = sum(r["tx_time_s"] for r in meter.uplink_readings)
            row["u2g_rate_time_weighted_mbps"] = row["u2g_served_mbit"] / max(row["u2g_tx_time_s"], 1e-12)

        # Add cost-based metrics (J_holding, J_energy) from readings
        if meter.readings and meter.integration_steps > 0:
            row["J_holding"] = sum(r["holding_cost"] for r in meter.readings) / meter.integration_steps
            row["J_energy"] = sum(r["energy_cost"] for r in meter.readings) / meter.integration_steps
            row["J"] = row["J_holding"] + row["J_energy"]

        results.append(row)

    # Aggregate only numeric metrics
    if not results:
        return {}

    numeric_keys = [k for k in results[0].keys()
                    if isinstance(results[0].get(k), (int, float, np.number)) and results[0].get(k) is not None]

    return {
        k: (np.mean([r[k] for r in results]), np.std([r[k] for r in results]))
        for k in numeric_keys
    }


def _save_checkpoint(directory, policy, state, tag):
    """Save training checkpoint."""
    import gzip
    import pickle

    checkpoint = {
        'policy_state': {
            'macro_actor': policy.macro_actor.state_dict(),
            'macro_critic': policy.macro_critic.state_dict(),
            'sac_actor': policy.sac_actor.action_net.state_dict(),
            'sac_critic': policy.sac_critic.critic_v.state_dict(),
            'target': policy.sac_critic.target_critic_v.state_dict(),
            'log_alpha': policy.entropy.log_alpha.detach(),
        },
        'optimizer_state': {
            'ppo_actor': policy.ppo_actor_optimizer.state_dict(),
            'ppo_critic': policy.ppo_critic_optimizer.state_dict(),
            'sac_actor': policy.sac_actor.optimizer.state_dict(),
            'sac_critic': policy.sac_critic.optimizer.state_dict(),
        },
        'training_state': state,
        'tag': tag,
    }

    path = Path(directory) / f"checkpoint_{tag}.pkl.gz"
    with gzip.open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


def _save_results(directory, metrics):
    """Save evaluation results."""
    # Convert metrics dict to list of dicts for CSV
    # All rows must have the same keys for append_csv
    rows = []
    for key, val in metrics.items():
        row = {'metric': key, 'mean': None, 'std': None, 'value': None}
        if isinstance(val, tuple) and len(val) == 2:
            row['mean'], row['std'] = val
        else:
            row['value'] = val
        rows.append(row)
    append_csv(Path(directory) / "results.csv", rows)


def _aggregate_results(all_results):
    """Aggregate results across seeds."""
    keys = [k for k in all_results[0].keys() if isinstance(all_results[0][k], (int, float, np.number))]

    agg = {}
    for key in keys:
        values = [r[key] for r in all_results if r.get(key) is not None]
        if len(values) < 2:
            agg[key] = {'mean': values[0] if values else np.nan, 'std': 0.0, 'values': values}
        else:
            std_val = np.std(values)
            # Replace inf/nan with 0.0
            if not np.isfinite(std_val):
                std_val = 0.0
            mean_val = np.mean(values)
            agg[key] = {'mean': mean_val, 'std': std_val, 'values': values}

    return agg


def sac_source_dir():
    """Get SAC source directory."""
    return ROOT / "results/sac_only_wireless6_long_seed101"


# =============================================================================
# Protocol Export
# =============================================================================

def protocol():
    """Export experiment protocol for documentation."""
    return dict(
        stage="joint_hrl_training_experiment",
        purpose="Validate hierarchical training with controlled update frequency",
        experiments=[
            dict(id="E1", name="sac_only", description="Train SAC with frozen PPO"),
            dict(id="E2", name="ppo_only", description="Train PPO with frozen SAC"),
            dict(id="E3", name="joint_1_5", description="Joint 1:5 ratio"),
            dict(id="E4", name="joint_1_10", description="Joint 1:10 ratio (recommended)"),
            dict(id="E5", name="joint_1_20", description="Joint 1:20 ratio"),
        ],
        validation_scenes=list(VALIDATION_SCENES),
        training_scenes=list(TRAIN_SCENES_JT),
        metrics=list(METRICS),
        budget_matching="600K micro-updates per condition",
    )
