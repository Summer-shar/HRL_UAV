import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

PROJECT_ROOT = bootstrap()

from compare_common import EvaluationHistory
from compare_common import add_common_args
from compare_common import advance_load_switch_before_macro
from compare_common import append_load_switch_trace
from compare_common import evaluate_policy
from compare_common import initialize_episode_load_switch
from compare_common import initialize_load_switch_trace
from compare_common import make_env_from_args
from compare_common import prepare_results_dirs
from compare_common import print_evaluation_summary
from compare_common import save_scenario_metadata
from hrl_uav.agents.ppo_agent import A_UPDATE_STEPS
from hrl_uav.agents.ppo_agent import CLIP_EPSILON
from hrl_uav.agents.ppo_agent import Actor as PPO_Actor
from hrl_uav.agents.ppo_agent import Critic as PPO_Critic
from hrl_uav.agents.sac_agent import Actor as SAC_Actor
from hrl_uav.agents.sac_agent import Critic as SAC_Critic
from hrl_uav.agents.sac_agent import Entroy as SAC_Entropy
from hrl_uav.agents.sac_agent import Memory as SAC_Memory
from hrl_uav.hrl_wrappers import normalized_power_residual
from hrl_uav.load_switching import apply_load_switch_phase
import hrl_uav.agents.ppo_agent as ppo_agent_mod
import hrl_uav.agents.sac_agent as sac_agent_mod


SAC_GAMMA = 0.99
SAC_MEMORY_WARMUP = 1000
SAC_MEMORY_CAPACITY = 500000
SAC_RECENT_WINDOW = 500000
SAC_RECENT_RATIO = 0.0

TRAIN_LOG_FIELDS = [
    "episode",
    "macro_reward",
    "micro_reward",
    "J_train",
    "delay_objective_train",
    "micro_cost_delay_mean",
    "micro_lagrangian_energy_cost_mean",
    "micro_drift_cost_mean",
    "micro_overflow_cost_mean",
    "micro_primal_cost_mean",
    "micro_j_base_mean",
    "micro_energy_ratio_mean",
    "micro_q_total_mean",
    "micro_q_total_final",
    "micro_upload_delay_cost_mean",
    "micro_route_entropy_mean",
    "micro_route_boundary_fraction",
    "micro_route_max_ratio",
    "micro_reward_reconstruction_max_error",
    "service_energy_episode_j",
    "total_energy_episode_j",
    "average_power_w",
    "power_budget_w",
    "power_target_w",
    "power_violation_ratio",
    "energy_dual_before",
    "energy_dual_after",
    "dual_update_power_w",
    "dual_power_window",
    "battery_power_per_uav_w",
    "battery_power_budget_per_uav_w",
    "energy_dual_before_per_uav",
    "energy_dual_after_per_uav",
    "dual_update_power_per_uav_w",
    "cpu_energy_price_per_uav",
    "comm_cost",
    "u2g_m",
    "move_m",
    "macro_all_hover_ratio",
    "macro_uav_hover_ratio",
    "start_offset_mean_m",
    "start_offset_max_m",
    "load_scale",
    "cpu_ghz",
    "load_switch_mode",
    "load_switch_count",
    "load_switch_steps",
    "load_switch_dwell_steps",
    "advisor_weight",
    "macro_shaping_weight",
    "macro_move_penalty_weight",
    "macro_propulsion_tiebreak_weight",
    "macro_propulsion_delta_energy",
    "deployment_transient_energy_episode_j",
    "macro_energy_episode",
    "macro_feedback_mean",
    "macro_primal_cost_mean",
    "macro_energy_ratio_mean",
    "macro_global_micro_signal_mean",
    "macro_shaping_signal_mean",
    "macro_positioning_improvement_mean",
    "macro_movement_penalty_mean",
    "macro_propulsion_penalty_mean",
    "macro_action_norm_mean",
    "projected_uplink_time_s_mean",
    "projected_u2u_time_s_mean",
    "projected_u2u_rate_mbps_mean",
    "macro_micro_steps_min",
    "macro_micro_steps_max",
    "macro_reward_reconstruction_max_error",
    "ppo_actor_lr",
    "ppo_action_std",
    "ppo_actor_loss",
    "ppo_policy_loss",
    "ppo_bc_loss",
    "ppo_approx_kl",
    "ppo_clip_fraction",
    "ppo_update_epochs",
    "ppo_critic_loss",
    "ppo_energy_critic_loss",
    "ppo_delay_advantage_mean",
    "ppo_energy_advantage_mean",
    "ppo_combined_advantage_mean",
    "sac_trainable_micro_steps",
    "sac_actor_loss",
    "sac_critic_loss",
    "sac_actor_grad_norm",
    "sac_critic_grad_norm",
    "sac_alpha",
    "sac_critic_updates_total",
    "sac_actor_updates_total",
    "service_resource_decisions",
    "smdp_boundary_transitions",
    "smdp_boundary_duration_s",
    "smdp_boundary_discount_exponent",
    "smdp_boundary_service_reward",
    "smdp_boundary_deployment_holding_reward_1",
    "smdp_boundary_deployment_holding_reward_2",
    "smdp_boundary_composed_reward",
    "smdp_boundary_bootstrap_discount",
    "sac_replay_size",
    "sca_solve_count",
    "sca_solve_seconds",
    "sca_solve_mean_ms",
    "episode_seconds",
]

MICRO_REWARD_DIAGNOSTIC_SOURCES = {
    "micro_cost_delay_mean": "cost_delay",
    "micro_lagrangian_energy_cost_mean": "lagrangian_energy_cost",
    "micro_drift_cost_mean": "drift_cost",
    "micro_overflow_cost_mean": "overflow_cost",
    "micro_primal_cost_mean": "primal_cost_step",
    "micro_j_base_mean": "J_base_step",
    "micro_energy_ratio_mean": "energy_constraint_ratio",
    "micro_q_total_mean": "q_total_current",
    "micro_upload_delay_cost_mean": "upload_delay_cost",
    "micro_route_entropy_mean": "micro_route_entropy_mean",
    "micro_route_boundary_fraction": "micro_route_boundary_fraction",
    "micro_route_max_ratio": "micro_route_max_ratio",
}

MACRO_REWARD_DIAGNOSTIC_SOURCES = {
    "macro_primal_cost_mean": "primal_cost_mean",
    "macro_energy_ratio_mean": "energy_constraint_ratio_mean",
    "macro_global_micro_signal_mean": "global_micro_signal",
    "macro_shaping_signal_mean": "shaping_signal",
    "macro_positioning_improvement_mean": "positioning_improvement",
    "macro_movement_penalty_mean": "movement_penalty",
    "macro_propulsion_penalty_mean": "propulsion_penalty",
    "macro_action_norm_mean": "macro_action_norm_mean",
}


def initialize_training_log(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        csv.DictWriter(fp, fieldnames=TRAIN_LOG_FIELDS).writeheader()


def append_training_log(path, row):
    with open(path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=TRAIN_LOG_FIELDS)
        writer.writerow({field: row.get(field, "") for field in TRAIN_LOG_FIELDS})


def initialize_reward_diagnostics(source_map):
    return {output_key: [] for output_key in source_map}


def initialize_micro_reward_diagnostics():
    storage = initialize_reward_diagnostics(
        MICRO_REWARD_DIAGNOSTIC_SOURCES
    )
    storage["micro_reward_reconstruction_error"] = []
    return storage


def record_reward_diagnostics(storage, info, source_map):
    for output_key, source_key in source_map.items():
        storage[output_key].append(float(info.get(source_key, 0.0)))


def summarize_reward_diagnostics(storage):
    return {
        output_key: (
            float(np.mean(values))
            if values
            else 0.0
        )
        for output_key, values in storage.items()
    }


def record_micro_reward_diagnostics(storage, info):
    record_reward_diagnostics(
        storage,
        info,
        MICRO_REWARD_DIAGNOSTIC_SOURCES,
    )
    expected_reward = (
        -float(info.get("J_step", 0.0))
        * float(info.get("micro_reward_scale", 1.0))
    )
    storage["micro_reward_reconstruction_error"].append(
        abs(float(info.get("micro_reward_scaled", 0.0)) - expected_reward)
    )


def summarize_micro_reward_diagnostics(storage):
    summary = summarize_reward_diagnostics(
        {
            key: values
            for key, values in storage.items()
            if key != "micro_reward_reconstruction_error"
        }
    )
    q_values = storage.get("micro_q_total_mean", [])
    summary["micro_q_total_final"] = (
        float(q_values[-1]) if q_values else 0.0
    )
    reconstruction_errors = storage.get(
        "micro_reward_reconstruction_error",
        [],
    )
    summary["micro_reward_reconstruction_max_error"] = (
        float(np.max(reconstruction_errors))
        if reconstruction_errors
        else 0.0
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Centralized macro PPO + SAC micro training"
    )
    add_common_args(parser, default_results_subdir="central_macro")
    parser.add_argument(
        "--frozen_macro_policy",
        choices=["actor", "zero"],
        default="actor",
        help=(
            "Macro action source while --freeze_macro is active: preserve the "
            "historical deterministic actor behavior or force zero residual "
            "actions for centroid tracking"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compute device",
    )
    parser.add_argument(
        "--micro_solver",
        choices=["sac", "sca", "local"],
        default="sac",
        help="Micro layer: learned SAC, model-based SCA, or local-only routing",
    )
    parser.add_argument(
        "--sac_actor_lr",
        type=float,
        default=3e-4,
        help="Learning rate of the SAC routing actor",
    )
    parser.add_argument(
        "--sac_critic_lr",
        type=float,
        default=3e-4,
        help="Learning rate of the SAC twin critic",
    )
    parser.add_argument(
        "--sac_alpha_lr",
        type=float,
        default=3e-4,
        help="Learning rate of the SAC entropy temperature",
    )
    parser.add_argument(
        "--sac_initial_alpha",
        type=float,
        default=1.0,
        help="Initial SAC entropy temperature",
    )
    parser.add_argument(
        "--resume_micro_optimizer",
        type=int,
        choices=[0, 1],
        default=1,
        help="Restore SAC optimizer and entropy states when available",
    )
    parser.add_argument(
        "--sca_update_interval",
        type=int,
        default=1,
        help="Resource frames between online SCA updates",
    )
    parser.add_argument(
        "--sca_max_iter",
        type=int,
        default=3,
        help="Convex approximation steps per online SCA update",
    )
    parser.add_argument("--sca_tol", type=float, default=1e-3)
    parser.add_argument("--sca_solver", type=str, default="CLARABEL")
    parser.add_argument("--sca_delay_weight", type=float, default=0.7)
    parser.add_argument("--sca_energy_weight", type=float, default=0.3)
    parser.add_argument(
        "--sca_proximal_weight",
        type=float,
        default=0.1,
        help="SCA trust-region weight around the previous feasible routing",
    )
    parser.add_argument(
        "--ppo_update_episode_window",
        type=int,
        default=20,
        help="Number of episodes worth of macro steps per PPO update",
    )
    parser.add_argument(
        "--train_start_jitter_max_m",
        type=float,
        default=0.0,
        help=(
            "Maximum random UAV start offset from each service anchor during "
            "training; <=0 keeps the configured fixed starts"
        ),
    )
    parser.add_argument(
        "--train_start_anchor_probability",
        type=float,
        default=0.2,
        help="Per-UAV probability of starting exactly at its service anchor",
    )
    parser.add_argument(
        "--ppo_zero_inflated_actions",
        type=int,
        choices=[0, 1],
        default=1,
        help="Give each UAV macro policy an explicit probability mass at dx=dy=0",
    )
    parser.add_argument(
        "--ppo_hover_gate_threshold",
        type=float,
        default=0.05,
        help="Deterministic mean-action norm below which the UAV hovers",
    )
    parser.add_argument(
        "--ppo_hover_gate_temperature",
        type=float,
        default=10.0,
        help="Sharpness of the stochastic move/hover mixture probability",
    )
    parser.add_argument(
        "--ppo_actor_lr",
        type=float,
        default=3e-4,
        help="Initial learning rate of the centralized PPO actor",
    )
    parser.add_argument(
        "--ppo_initial_log_std",
        type=float,
        default=-0.8,
        help="Initial log standard deviation of each macro action component",
    )
    parser.add_argument(
        "--ppo_final_log_std",
        type=float,
        default=None,
        help=(
            "Optional final log standard deviation for a fixed linear "
            "exploration schedule; omitted keeps log_std trainable"
        ),
    )
    parser.add_argument(
        "--ppo_log_std_decay_episodes",
        type=int,
        default=0,
        help="Linear PPO action-std decay horizon; <=0 uses the initial value",
    )
    parser.add_argument(
        "--ppo_gamma",
        type=float,
        default=0.99,
        help="Macro-step discount factor for long-term trajectory returns",
    )
    parser.add_argument(
        "--ppo_gae_lambda",
        type=float,
        default=0.95,
        help="GAE trace parameter for the macro PPO critic",
    )
    parser.add_argument(
        "--ppo_target_kl",
        type=float,
        default=0.015,
        help=(
            "Stop PPO epochs early when sampled approximate KL exceeds "
            "this value; <=0 disables"
        ),
    )
    parser.add_argument(
        "--ppo_actor_final_lr",
        type=float,
        default=3e-4,
        help="Final PPO actor learning rate after linear decay",
    )
    parser.add_argument(
        "--ppo_lr_decay_episodes",
        type=int,
        default=0,
        help="Linear PPO actor LR decay horizon; <=0 disables decay",
    )
    parser.add_argument(
        "--energy_dual_power_window",
        type=int,
        default=1,
        help="Trailing episode-power window used by dual ascent",
    )
    parser.add_argument(
        "--advisor_weight",
        type=float,
        default=0.0,
        help="Initial imitation weight for the communication advisor",
    )
    parser.add_argument(
        "--advisor_final_weight",
        type=float,
        default=0.0,
        help="Final advisor imitation weight after annealing",
    )
    parser.add_argument(
        "--advisor_decay_episodes",
        type=int,
        default=200,
        help="Episodes over which advisor imitation weight is annealed",
    )
    parser.add_argument(
        "--advisor_steps",
        type=int,
        default=8,
        help="Projected gradient steps used by the communication advisor",
    )
    parser.add_argument(
        "--advisor_lr",
        type=float,
        default=0.35,
        help="Adam learning rate for advisor action optimization",
    )
    parser.add_argument(
        "--advisor_min_separation_m",
        type=float,
        default=80.0,
        help="Minimum UAV separation encouraged by advisor penalty",
    )
    parser.add_argument(
        "--advisor_move_penalty",
        type=float,
        default=0.02,
        help="Small advisor penalty on large macro displacements",
    )
    parser.add_argument(
        "--advisor_sep_penalty",
        type=float,
        default=10.0,
        help="Advisor penalty weight for violating UAV separation",
    )
    parser.add_argument(
        "--advisor_boundary_penalty",
        type=float,
        default=10.0,
        help="Advisor penalty weight for leaving the 1000m x 1000m area",
    )
    parser.add_argument(
        "--micro_ckpt",
        type=str,
        default="",
        help="Checkpoint used to initialize the lower SAC policy",
    )
    parser.add_argument(
        "--micro_replay",
        type=str,
        default="",
        help="Replay state used to continue lower-layer SAC training",
    )
    parser.add_argument(
        "--freeze_micro",
        action="store_true",
        help="Keep the loaded lower SAC fixed while training macro PPO",
    )
    parser.add_argument(
        "--frozen_micro_stochastic",
        action="store_true",
        help="Sample lower SAC actions while frozen; deterministic is the default",
    )
    parser.add_argument("--macro_shaping_initial", type=float, default=0.0)
    parser.add_argument("--macro_shaping_final", type=float, default=0.0)
    parser.add_argument(
        "--macro_shaping_decay_episodes",
        type=int,
        default=0,
        help="Linear shaping decay horizon; <=0 keeps the initial value",
    )
    parser.add_argument(
        "--macro_move_penalty_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--macro_propulsion_tiebreak_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--macro_propulsion_energy_norm",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--checkpoint_epsilon_relative",
        type=float,
        default=0.0,
        help=(
            "Select the lowest-macro-energy checkpoint among policies within "
            "this relative J gap; 0 keeps strict best-J selection"
        ),
    )
    parser.add_argument(
        "--train_load_min",
        type=float,
        default=0.0,
        help="Minimum episode load_scale for randomized training; <=0 disables",
    )
    parser.add_argument(
        "--train_load_max",
        type=float,
        default=0.0,
        help="Maximum episode load_scale for randomized training; <=0 disables",
    )
    args = parser.parse_args()

    if "--macro_action_mode" not in sys.argv:
        args.macro_action_mode = "delta"
    if "--macro_comm_weight" not in sys.argv:
        args.macro_comm_weight = 0.0
    if args.freeze_micro and not args.micro_ckpt:
        parser.error("--freeze_micro requires --micro_ckpt")
    if args.micro_replay and args.micro_solver != "sac":
        parser.error("--micro_replay is only valid with --micro_solver sac")
    if args.freeze_macro and (
        args.advisor_weight > 0.0 or args.advisor_final_weight > 0.0
    ):
        parser.error("--freeze_macro is incompatible with advisor imitation")
    if not args.freeze_macro and args.frozen_macro_policy != "actor":
        parser.error("--frozen_macro_policy zero requires --freeze_macro")
    if args.macro_propulsion_energy_norm <= 0.0:
        parser.error("--macro_propulsion_energy_norm must be positive")
    if args.checkpoint_epsilon_relative < 0.0:
        parser.error("--checkpoint_epsilon_relative must be non-negative")
    if args.objective_mode == "constrained_delay" and args.power_budget_w <= 0.0:
        parser.error("--constrained_delay requires --power_budget_w > 0")
    if (
        args.objective_mode == "constrained_delay"
        and args.checkpoint_epsilon_relative > 0.0
    ):
        parser.error(
            "--checkpoint_epsilon_relative is only valid for weighted mode"
        )
    if args.energy_dual_lr < 0.0 or args.energy_dual_max <= 0.0:
        parser.error("Energy dual parameters must be non-negative with max > 0")
    if args.energy_dual_power_window <= 0:
        parser.error("--energy_dual_power_window must be positive")
    if args.power_constraint_margin_w < 0.0 or (
        args.power_constraint_margin_w > 0.0
        and (
            args.power_budget_w <= 0.0
            or args.power_constraint_margin_w >= args.power_budget_w
        )
    ):
        parser.error(
            "--power_constraint_margin_w must be in [0, power_budget_w)"
        )
    if not 0.0 < args.ppo_gamma <= 1.0:
        parser.error("--ppo_gamma must be in (0, 1]")
    if not 0.0 <= args.ppo_gae_lambda <= 1.0:
        parser.error("--ppo_gae_lambda must be in [0, 1]")
    if not np.isfinite(args.ppo_target_kl):
        parser.error("--ppo_target_kl must be finite")
    if min(
        args.sac_actor_lr,
        args.sac_critic_lr,
        args.sac_alpha_lr,
        args.sac_initial_alpha,
    ) <= 0.0:
        parser.error("SAC learning rates and initial alpha must be positive")
    if args.micro_reward_scale <= 0.0:
        parser.error("--micro_reward_scale must be positive")
    if args.sac_grad_clip_norm < 0.0:
        parser.error("--sac_grad_clip_norm must be non-negative")
    if args.train_start_jitter_max_m < 0.0:
        parser.error("--train_start_jitter_max_m must be non-negative")
    if not 0.0 <= args.train_start_anchor_probability <= 1.0:
        parser.error("--train_start_anchor_probability must be in [0, 1]")
    return args


def set_wrapper_load_scale(env, load_scale):
    env.env.load_scale = float(load_scale)


def set_wrapper_cpu_ghz(env, cpu_ghz):
    env.env.F_max_array[:] = float(cpu_ghz) * 1e9


def sample_training_start_positions(env, args, episode):
    max_offset = float(args.train_start_jitter_max_m)
    if max_offset <= 0.0:
        return np.asarray(env.env.uav_initial_positions, dtype=np.float64).copy()

    rng = np.random.default_rng(int(args.seed) * 1000003 + int(episode))
    anchors = np.asarray(env.env.anchor_points, dtype=np.float64)
    angles = rng.uniform(-np.pi, np.pi, size=env.N)
    radii = rng.uniform(0.0, max_offset, size=env.N)
    anchor_mask = rng.random(env.N) < float(
        args.train_start_anchor_probability
    )
    radii[anchor_mask] = 0.0
    offsets = np.stack(
        [radii * np.cos(angles), radii * np.sin(angles)],
        axis=1,
    )
    return np.clip(anchors + offsets, 0.0, 1000.0)


def sample_training_load(args):
    lo = float(getattr(args, "train_load_min", 0.0))
    hi = float(getattr(args, "train_load_max", 0.0))
    if lo > 0.0 and hi >= lo:
        return float(np.random.uniform(lo, hi))
    return float(getattr(args, "load_scale", 1.0))


def sample_training_cpu(args):
    lo = float(getattr(args, "train_cpu_min", 0.0))
    hi = float(getattr(args, "train_cpu_max", 0.0))
    if lo > 0.0 and hi >= lo:
        return float(np.random.uniform(lo, hi))
    return float(getattr(args, "cpu_ghz", 16.0))


def sample_training_scenario(args):
    if str(getattr(args, "train_scenario_mode", "independent")) != "axes":
        return sample_training_load(args), sample_training_cpu(args)
    if np.random.random() < 0.5:
        return float(args.load_scale), sample_training_cpu(args)
    return sample_training_load(args), float(args.cpu_ghz)


def power_constraint_target_w(args):
    budget = float(args.power_budget_w)
    margin = max(float(getattr(args, "power_constraint_margin_w", 0.0)), 0.0)
    return max(budget - margin, 1e-8)


def update_energy_dual(env, args, average_power_w):
    current = float(getattr(env, "energy_lagrange_multiplier", 0.0))
    if str(getattr(args, "objective_mode", "weighted")) != "constrained_delay":
        return current, current, 0.0
    target = power_constraint_target_w(args)
    normalized_violation = (float(average_power_w) - target) / target
    updated = np.clip(
        current + float(args.energy_dual_lr) * normalized_violation,
        0.0,
        float(args.energy_dual_max),
    )
    setter = getattr(env, "set_energy_lagrange_multipliers", None)
    if callable(setter):
        setter(float(updated))
    else:
        env.energy_lagrange_multiplier = float(updated)
    return current, float(updated), float(normalized_violation)


def update_per_uav_energy_duals(env, args, average_power_per_uav_w):
    powers = np.asarray(average_power_per_uav_w, dtype=np.float64)
    if powers.shape != (env.N,):
        raise ValueError(
            f"Expected per-UAV powers with shape ({env.N},), got {powers.shape}."
        )
    current = np.asarray(
        env.energy_lagrange_multipliers,
        dtype=np.float64,
    ).copy()
    if str(getattr(args, "objective_mode", "weighted")) != "constrained_delay":
        return current, current, np.zeros_like(current)
    targets = np.asarray(env.uav_power_budgets_w, dtype=np.float64)
    normalized_violation = (powers - targets) / np.maximum(targets, 1e-8)
    updated = np.clip(
        current + float(args.energy_dual_lr) * normalized_violation,
        0.0,
        float(args.energy_dual_max),
    )
    env.set_energy_lagrange_multipliers(updated)
    return current, updated, normalized_violation


def split_macro_lagrangian_reward(macro_reward, reward_info, env):
    """Remove the current dual term and return a stationary reward/cost pair."""
    if str(getattr(env, "power_constraint_scope", "system")) == "per_uav":
        ratios = np.asarray(
            reward_info.get(
                "energy_constraint_ratio_per_uav_mean",
                np.zeros(env.N),
            ),
            dtype=np.float32,
        )
        energy_cost = (
            float(env.macro_feudal_weight)
            * float(env.micro_reward_scale)
            * ratios
        )
        if str(getattr(env, "objective_mode", "weighted")) == "constrained_delay":
            delay_reward = float(macro_reward) + float(
                np.dot(env.energy_lagrange_multipliers, energy_cost)
            )
        else:
            delay_reward = float(macro_reward)
            energy_cost = np.zeros(env.N, dtype=np.float32)
        return float(delay_reward), energy_cost.astype(np.float32)

    energy_ratio = float(reward_info.get("energy_constraint_ratio_mean", 0.0))
    energy_residual = normalized_power_residual(energy_ratio)
    cost_scale = (
        float(env.macro_feudal_weight)
        * float(env.micro_reward_scale)
        * energy_residual
    )
    if str(getattr(env, "objective_mode", "weighted")) == "constrained_delay":
        delay_reward = float(macro_reward) + (
            float(env.energy_lagrange_multiplier) * cost_scale
        )
        energy_cost = cost_scale
    else:
        delay_reward = float(macro_reward)
        energy_cost = 0.0
    return float(delay_reward), float(energy_cost)


def generalized_advantage_estimate(
    rewards,
    values,
    next_value,
    gamma,
    gae_lambda,
):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if rewards.shape != values.shape:
        raise ValueError("rewards and values must have the same shape")
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros_like(values[0], dtype=np.float32)
    next_value = np.asarray(next_value, dtype=np.float32)
    for index in range(len(rewards) - 1, -1, -1):
        following_value = (
            next_value if index == len(rewards) - 1 else values[index + 1]
        )
        delta = rewards[index] + float(gamma) * following_value - values[index]
        last_gae = delta + float(gamma) * float(gae_lambda) * last_gae
        advantages[index] = last_gae
    return advantages, advantages + values


def combine_constrained_advantages(
    delay_advantages,
    energy_advantages,
    energy_lagrange_multiplier,
):
    delay = np.asarray(delay_advantages, dtype=np.float32)
    energy = np.asarray(energy_advantages, dtype=np.float32)
    multipliers = np.asarray(
        energy_lagrange_multiplier,
        dtype=np.float32,
    ).reshape(-1)
    if multipliers.size == 1:
        if delay.shape != energy.shape:
            raise ValueError("delay and energy advantages must have the same shape")
        return delay - float(multipliers[0]) * energy

    if energy.ndim != 2 or energy.shape[1] != multipliers.size:
        raise ValueError(
            "Vector energy advantages must have shape (T, number_of_constraints)."
        )
    delay_column = delay.reshape(energy.shape[0], -1)
    if delay_column.shape[1] != 1:
        raise ValueError("Delay advantages must contain one scalar per transition.")
    energy_penalty = np.sum(
        energy * multipliers.reshape(1, -1),
        axis=1,
        keepdims=True,
    )
    return delay_column - energy_penalty


def advisor_weight_for_episode(args, episode):
    initial = float(args.advisor_weight)
    final = float(args.advisor_final_weight)
    decay = max(int(args.advisor_decay_episodes), 1)
    progress = min(max(float(episode) / float(decay), 0.0), 1.0)
    return final + (initial - final) * (1.0 - progress)


def ppo_actor_lr_for_episode(args, episode):
    initial = float(args.ppo_actor_lr)
    final = float(args.ppo_actor_final_lr)
    decay = int(args.ppo_lr_decay_episodes)
    if decay <= 0:
        return initial
    progress = min(max(float(episode) / max(decay - 1, 1), 0.0), 1.0)
    return initial + progress * (final - initial)


def ppo_log_std_for_episode(args, episode):
    final = getattr(args, "ppo_final_log_std", None)
    if final is None:
        return None
    initial = float(args.ppo_initial_log_std)
    decay = int(getattr(args, "ppo_log_std_decay_episodes", 0))
    if decay <= 0:
        return initial
    progress = min(max(float(episode) / max(decay - 1, 1), 0.0), 1.0)
    return initial + progress * (float(final) - initial)


def configure_ppo_log_std_schedule(actor, args):
    """Use scheduled exploration noise rather than learning its physical scale."""
    scheduled = ppo_log_std_for_episode(args, 0)
    if scheduled is None:
        return None
    for network in (actor.old_pi, actor.new_pi):
        network.log_std.requires_grad_(False)
        with torch.no_grad():
            network.log_std.fill_(float(scheduled))
    return float(scheduled)


def set_ppo_log_std(actor, log_std):
    if log_std is None:
        return
    for network in (actor.old_pi, actor.new_pi):
        with torch.no_grad():
            network.log_std.fill_(float(log_std))


def macro_shaping_weight_for_episode(args, episode):
    initial = float(args.macro_shaping_initial)
    final = float(args.macro_shaping_final)
    decay = int(args.macro_shaping_decay_episodes)
    if decay <= 0:
        return initial
    progress = min(max(float(episode) / max(decay - 1, 1), 0.0), 1.0)
    return initial + progress * (final - initial)


def configure_macro_objective(env, args, shaping_weight=None):
    if shaping_weight is None:
        shaping_weight = float(args.macro_shaping_initial)
    env.macro_shaping_lambda = float(shaping_weight)
    env.macro_move_penalty_weight = float(args.macro_move_penalty_weight)
    env.macro_propulsion_tiebreak_weight = float(
        args.macro_propulsion_tiebreak_weight
    )
    env.macro_propulsion_energy_norm = float(args.macro_propulsion_energy_norm)


def torch_load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_micro_checkpoint(
    path,
    sac_actor,
    sac_critic,
    sac_entropy=None,
    load_optimizers=True,
):
    resolved = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Micro checkpoint not found: {resolved}")
    checkpoint = torch_load_checkpoint(resolved)
    required = ("sac_actor", "sac_critic")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Micro checkpoint is missing keys: {missing}")
    sac_actor.action_net.load_state_dict(checkpoint["sac_actor"])
    sac_critic.critic_v.load_state_dict(checkpoint["sac_critic"])
    target_state = checkpoint.get("sac_target_critic", checkpoint["sac_critic"])
    sac_critic.target_critic_v.load_state_dict(target_state)
    if sac_entropy is not None and "sac_log_alpha" in checkpoint:
        alpha_state = torch.as_tensor(
            checkpoint["sac_log_alpha"],
            dtype=sac_entropy.log_alpha.dtype,
            device=sac_entropy.log_alpha.device,
        )
        sac_entropy.log_alpha.data.copy_(alpha_state.reshape_as(sac_entropy.log_alpha))
        sac_entropy.alpha = sac_entropy.log_alpha.exp()
    if load_optimizers:
        optimizer_states = (
            (sac_actor.optimizer, "sac_actor_optimizer"),
            (sac_critic.optimizer, "sac_critic_optimizer"),
            (
                sac_entropy.optimizer if sac_entropy is not None else None,
                "sac_entropy_optimizer",
            ),
        )
        for optimizer, key in optimizer_states:
            if optimizer is not None and key in checkpoint:
                optimizer.load_state_dict(checkpoint[key])
    return resolved


def load_checkpoint_energy_dual(path, fallback=0.0, num_uavs=None):
    resolved = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    checkpoint = torch_load_checkpoint(resolved)
    saved_args = checkpoint.get("args", {})
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    if not isinstance(saved_args, dict):
        saved_args = {}
    value = saved_args.get(
        "energy_lagrange_multiplier_current",
        saved_args.get("energy_dual_initial", fallback),
    )
    if num_uavs is not None:
        vector = saved_args.get(
            "energy_lagrange_multipliers_current",
            value,
        )
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim == 0 or vector.shape == (1,):
            vector = np.full(int(num_uavs), float(vector.reshape(-1)[0]))
        if vector.shape != (int(num_uavs),):
            raise ValueError(
                "Checkpoint energy dual vector has shape "
                f"{vector.shape}; expected ({int(num_uavs)},)."
            )
        if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
            raise ValueError(
                "Checkpoint energy duals must be finite and non-negative."
            )
        return vector
    return max(float(value), 0.0)


def configure_sac_learning_rates(sac_actor, sac_critic, sac_entropy, args):
    if sac_actor is None:
        return
    for param_group in sac_actor.optimizer.param_groups:
        param_group["lr"] = float(args.sac_actor_lr)
    for param_group in sac_critic.optimizer.param_groups:
        param_group["lr"] = float(args.sac_critic_lr)
    for param_group in sac_entropy.optimizer.param_groups:
        param_group["lr"] = float(args.sac_alpha_lr)


def save_sac_replay(path, sac_memory):
    if sac_memory is None:
        return ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(sac_memory.state_dict(), path)
    return path


def load_sac_replay(path, sac_memory):
    if sac_memory is None:
        raise ValueError("Cannot restore replay without a trainable SAC memory.")
    resolved = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Micro replay state not found: {resolved}")
    state = torch_load_checkpoint(resolved)
    restored = sac_memory.load_state_dict(state)
    return resolved, restored


def load_central_macro_checkpoint(
    path,
    macro_actor,
    macro_critic,
    macro_energy_critic=None,
):
    resolved = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Central macro checkpoint not found: {resolved}")
    checkpoint = torch_load_checkpoint(resolved)
    macro_solver = str(checkpoint.get("macro_solver", "ppo")).lower()
    if macro_solver != "ppo":
        raise ValueError(f"Expected PPO macro checkpoint, got: {macro_solver}")
    if "macro_actor" not in checkpoint:
        raise KeyError("Central macro checkpoint is missing 'macro_actor'")
    macro_actor.old_pi.load_state_dict(checkpoint["macro_actor"])
    macro_actor.new_pi.load_state_dict(checkpoint["macro_actor"])
    delay_critic_state = checkpoint.get(
        "macro_delay_critic",
        checkpoint.get("macro_critic"),
    )
    if delay_critic_state is not None:
        macro_critic.critic_v.load_state_dict(delay_critic_state)
    if macro_energy_critic is not None and "macro_energy_critic" in checkpoint:
        macro_energy_critic.critic_v.load_state_dict(
            checkpoint["macro_energy_critic"]
        )
    return resolved


def macro_energy_from_metrics(metrics):
    return float(
        np.sum(
            np.asarray(
                metrics.get("energy_macro_episode_per_uav", 0.0),
                dtype=np.float64,
            )
        )
    )


def save_eval_candidate(
    ckpt_dir,
    episode,
    metrics,
    payload,
    records,
):
    candidate_dir = os.path.join(ckpt_dir, "eval_candidates")
    os.makedirs(candidate_dir, exist_ok=True)
    candidate_path = os.path.join(candidate_dir, f"eval_ep_{int(episode):04d}.pt")
    torch.save(payload, candidate_path)
    records.append(
        {
            "episode": int(episode),
            "J": float(metrics.get("J", np.inf)),
            "macro_energy_episode": macro_energy_from_metrics(metrics),
            "u2g_m": float(metrics.get("u2g", 0.0)),
            "task_delay_high_ms": float(
                metrics.get("task_delay_high_ms", 0.0)
            ),
            "task_delay_low_ms": float(metrics.get("task_delay_low_ms", 0.0)),
            "checkpoint": candidate_path,
        }
    )


def select_epsilon_checkpoint(ckpt_dir, records, epsilon_relative):
    if not records:
        return None
    best_j = min(record["J"] for record in records)
    threshold = best_j * (1.0 + float(epsilon_relative))
    eligible = [record for record in records if record["J"] <= threshold]
    selected = min(
        eligible,
        key=lambda record: (record["macro_energy_episode"], record["J"]),
    )
    shutil.copyfile(selected["checkpoint"], os.path.join(ckpt_dir, "best.pt"))
    payload = {
        "epsilon_relative": float(epsilon_relative),
        "best_j": float(best_j),
        "j_threshold": float(threshold),
        "selected": selected,
        "eligible": eligible,
        "all_candidates": records,
    }
    with open(
        os.path.join(ckpt_dir, "epsilon_selection.json"),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return selected


def checkpoint_rank(metrics, args):
    if str(getattr(args, "objective_mode", "weighted")) != "constrained_delay":
        return (0, float(metrics.get("J", np.inf)))
    objective_value = float(
        metrics.get(
            "J",
            metrics.get("delay_objective", np.inf),
        )
    )
    checkpoint_power_rule = str(
        getattr(args, "checkpoint_power_rule", "mean_target")
    )
    if str(getattr(args, "power_constraint_scope", "system")) == "per_uav":
        violation_key = (
            "max_battery_power_ucb95_violation_w"
            if checkpoint_power_rule == "ucb95"
            else "max_battery_power_violation_w"
        )
        excess_w = max(float(metrics.get(violation_key, np.inf)), 0.0)
        tolerance_w = float(
            np.mean(
                np.asarray(
                    metrics.get("uav_power_budgets_w", [0.0]),
                    dtype=np.float64,
                )
            )
        ) * float(args.power_budget_tolerance)
        excess_w = max(excess_w - tolerance_w, 0.0)
        if excess_w <= 0.0:
            return (0, objective_value)
        return (1, excess_w, objective_value)
    if checkpoint_power_rule == "ucb95":
        budget = float(args.power_budget_w)
        observed_power_w = float(
            metrics.get("average_power_ucb95_w", np.inf)
        )
    else:
        budget = power_constraint_target_w(args)
        observed_power_w = float(metrics.get("average_power_w", np.inf))
    tolerance_w = float(args.power_budget_w) * float(args.power_budget_tolerance)
    excess_w = max(
        observed_power_w - budget - tolerance_w,
        0.0,
    )
    if excess_w <= 0.0:
        return (0, objective_value)
    return (1, excess_w, objective_value)


def project_unit_disk_np(action):
    action = np.asarray(action, dtype=np.float32).reshape(-1, 2)
    norm = np.linalg.norm(action, axis=1, keepdims=True)
    scale = np.maximum(norm, 1.0)
    return (action / scale).astype(np.float32)


def prepare_macro_policy_action(action_flat, num_uavs):
    """Keep PPO's sampled action separate from the physical action projection."""
    policy_action = np.asarray(action_flat, dtype=np.float32).reshape(
        int(num_uavs) * 2
    ).copy()
    environment_action = project_unit_disk_np(
        policy_action.reshape(int(num_uavs), 2)
    )
    return policy_action, environment_action


def centralized_macro_policy(actor, deterministic=False):
    def _policy(env, _local_obs_batch, eval_mode):
        _, global_state = env.get_macro_state()
        if deterministic or eval_mode:
            action = actor.choose_action_deterministic(global_state)
        else:
            action, _ = actor.choose_action(global_state)
        return project_unit_disk_np(action.reshape(env.N, 2))

    return _policy


def zero_macro_policy(env, _local_obs_batch, _eval_mode):
    """Return the centroid-waypoint residual used during v33 micro training."""
    return np.zeros((env.N, 2), dtype=np.float32)


def evaluation_macro_policy(args, actor):
    if (
        bool(getattr(args, "freeze_macro", False))
        and str(getattr(args, "frozen_macro_policy", "actor")) == "zero"
    ):
        return zero_macro_policy
    return centralized_macro_policy(actor, deterministic=True)


def configure_ppo_hover_policy(actor, args, initialize_hover=False):
    actor.configure_zero_inflated_actions(
        enabled=bool(int(getattr(args, "ppo_zero_inflated_actions", 0))),
        threshold=float(getattr(args, "ppo_hover_gate_threshold", 0.05)),
        temperature=float(
            getattr(args, "ppo_hover_gate_temperature", 10.0)
        ),
        initialize_hover=bool(initialize_hover),
        initial_log_std=float(getattr(args, "ppo_initial_log_std", -0.8)),
    )
    return actor


def sac_micro_policy(actor):
    def _policy(env, eval_mode):
        return actor.choose_action(
            env.get_sac_state(),
            deterministic=bool(eval_mode),
        )

    return _policy


def configure_sca_wrapper(env, args):
    env.sca_update_interval = int(args.sca_update_interval)
    env.sca_max_iter = int(args.sca_max_iter)
    env.sca_tol = float(args.sca_tol)
    env.sca_solver = str(args.sca_solver)
    env.sca_delay_weight = float(args.sca_delay_weight)
    env.sca_energy_weight = float(args.sca_energy_weight)
    env.sca_proximal_weight = float(args.sca_proximal_weight)


def sca_micro_policy(policy_fn, args):
    def _policy(env, eval_mode):
        configure_sca_wrapper(env, args)
        return policy_fn(env, deterministic=bool(eval_mode))

    return _policy


def save_trajectory_with_verification(env, save_path):
    traj_env = getattr(env, "env", env)
    uav_traj = np.asarray(getattr(traj_env, "uav_trajectory", []), dtype=np.float32)
    centroid_traj = np.asarray(
        getattr(traj_env, "centroid_trajectory", []), dtype=np.float32
    )
    users_traj = np.asarray(
        getattr(traj_env, "users_trajectory", []), dtype=np.float32
    )
    if uav_traj.size == 0 or centroid_traj.size == 0 or users_traj.size == 0:
        print(f"  [SKIP] No trajectory samples available for {save_path}")
        return

    anchor_points = np.asarray(
        getattr(traj_env, "anchor_points", []), dtype=np.float32
    )
    user_allocations = np.asarray(
        getattr(traj_env, "user_allocations", []), dtype=np.int32
    )
    zone_radius = float(getattr(traj_env, "user_zone_radius", 0.0))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        uav_traj=uav_traj,
        centroid_traj=centroid_traj,
        users_traj=users_traj,
        anchor_points=anchor_points,
        user_allocations=user_allocations,
        zone_radius=np.asarray(zone_radius, dtype=np.float32),
    )

    if os.path.exists(save_path):
        file_size_kb = os.path.getsize(save_path) / 1024.0
        print(f"  Saved trajectory data -> {save_path} ({file_size_kb:.1f} KB)")
    else:
        print(f"  [WARN] Trajectory data was not found after saving: {save_path}")

    png_path = os.path.splitext(save_path)[0] + ".png"
    try:
        traj_env.render_trajectory(save_path=png_path, show=False)
    except Exception as exc:
        print(f"  [WARN] Failed to render trajectory PNG: {exc}")
        return

    if os.path.exists(png_path):
        file_size_kb = os.path.getsize(png_path) / 1024.0
        print(f"  Saved trajectory figure -> {png_path} ({file_size_kb:.1f} KB)")
    else:
        print(f"  [WARN] Trajectory figure was not found after saving: {png_path}")


def save_final_policy_trajectory(results_dirs, args, macro_actor, micro_policy):
    if int(getattr(args, "trajectory_plot_interval", 0)) <= 0:
        return

    traj_env = make_env_from_args(args, record_trajectory=True)
    traj_args = argparse.Namespace(**vars(args))
    traj_args.eval_episodes = 1
    evaluate_policy(
        traj_env,
        traj_args,
        evaluation_macro_policy(args, macro_actor),
        micro_policy,
    )
    final_path = os.path.join(results_dirs["trajectory_dir"], "trajectory_final.npz")
    save_trajectory_with_verification(traj_env, final_path)


def build_checkpoint_payload(
    args,
    macro_actor,
    macro_critic,
    sac_actor=None,
    sac_critic=None,
    sac_entropy=None,
    macro_energy_critic=None,
):
    payload = {
        "macro_solver": "ppo",
        "constrained_critic_structure": "separate_delay_energy",
        "macro_actor": macro_actor.old_pi.state_dict(),
        "macro_critic": macro_critic.critic_v.state_dict(),
        "macro_delay_critic": macro_critic.critic_v.state_dict(),
        "micro_solver": str(args.micro_solver),
        "args": vars(args),
    }
    if macro_energy_critic is not None:
        payload["macro_energy_critic"] = (
            macro_energy_critic.critic_v.state_dict()
        )
    if sac_actor is not None and sac_critic is not None:
        payload.update(
            {
                "sac_actor": sac_actor.action_net.state_dict(),
                "sac_critic": sac_critic.critic_v.state_dict(),
                "sac_target_critic": sac_critic.target_critic_v.state_dict(),
                "sac_actor_optimizer": sac_actor.optimizer.state_dict(),
                "sac_critic_optimizer": sac_critic.optimizer.state_dict(),
            }
        )
        if sac_entropy is not None:
            payload.update(
                {
                    "sac_log_alpha": sac_entropy.log_alpha.detach().cpu(),
                    "sac_entropy_optimizer": sac_entropy.optimizer.state_dict(),
                }
            )
    return payload


def compute_advisor_action(env, args):
    raw_state = env.env.get_raw_state()
    flow = env._estimate_macro_flow_reference(raw_state)

    device = ppo_agent_mod.device
    positions = torch.as_tensor(
        raw_state["uav_coords"],
        dtype=torch.float32,
        device=device,
    )
    user_coords = torch.as_tensor(
        raw_state["user_coords"],
        dtype=torch.float32,
        device=device,
    )
    user_alloc = torch.as_tensor(
        env.env.user_allocations,
        dtype=torch.long,
        device=device,
    ).clamp(0, env.N - 1)
    flow_t = torch.as_tensor(flow, dtype=torch.float32, device=device)

    action = torch.zeros((env.N, 2), dtype=torch.float32, device=device)
    action.requires_grad_(True)
    optimizer = torch.optim.Adam([action], lr=float(args.advisor_lr))

    max_disp = float(env.max_macro_displacement)
    macro_dt = max(float(env.env.macro_dt), 1e-8)
    min_sep = float(args.advisor_min_separation_m)

    for _ in range(max(int(args.advisor_steps), 1)):
        norm = torch.linalg.norm(action, dim=1, keepdim=True).clamp_min(1.0)
        action_proj = action / norm
        proposed = positions + action_proj * max_disp
        eval_pos = proposed.clamp(0.0, 1000.0)

        assigned_pos = eval_pos[user_alloc]
        diff_user = user_coords - assigned_pos
        dist_sq_user = (diff_user * diff_user).sum(dim=1) + float(env.env.H) ** 2
        gain_user = float(env.env.beta_0) / dist_sq_user.clamp_min(1e-8)
        user_bw_hz = float(env.env.B_gt_total) / max(int(env.env.M), 1)
        user_rate_mbps = (
            user_bw_hz
            * torch.log2(
                1.0
                + (float(env.env.p_gt) * gain_user)
                / float(env.env.noise_power)
            )
            / 1e6
        )
        zone_rates = torch.zeros(env.N, dtype=torch.float32, device=device)
        zone_rates.scatter_add_(0, user_alloc, user_rate_mbps)
        upload_volume = flow_t.sum(dim=1)
        uplink_time = (upload_volume / zone_rates.clamp_min(1e-6)).sum()

        active = flow_t > 1e-9
        active = active & (~torch.eye(env.N, dtype=torch.bool, device=device))
        active_count = int(active.sum().detach().cpu().item())
        if active_count > 0:
            diff_uav = eval_pos[:, None, :] - eval_pos[None, :, :]
            dist_sq_uav = (diff_uav * diff_uav).sum(dim=-1).clamp_min(1e-8)
            gain_uav = float(env.env.beta_0) / dist_sq_uav
            bw_u2u_hz = float(env.env.B_u2u_total) / float(active_count)
            rates_u2u_mbps = (
                bw_u2u_hz
                * torch.log2(
                    1.0
                    + (float(env.env.p_u2u) * gain_uav)
                    / float(env.env.noise_power)
                )
                / 1e6
            )
            u2u_time = (
                flow_t[active] / rates_u2u_mbps[active].clamp_min(1e-6)
            ).sum()
        else:
            u2u_time = torch.zeros((), dtype=torch.float32, device=device)

        pair_diff = proposed[:, None, :] - proposed[None, :, :]
        pair_dist = torch.linalg.norm(pair_diff + 1e-9, dim=-1)
        offdiag = ~torch.eye(env.N, dtype=torch.bool, device=device)
        sep_violation = F.relu(min_sep - pair_dist[offdiag]) / max(min_sep, 1e-8)
        sep_penalty = (sep_violation * sep_violation).mean()

        lower = F.relu(-proposed / 1000.0)
        upper = F.relu((proposed - 1000.0) / 1000.0)
        boundary_penalty = (lower * lower + upper * upper).mean()
        move_penalty = (action_proj * action_proj).sum(dim=1).mean()

        loss = (
            (uplink_time + u2u_time) / macro_dt
            + float(args.advisor_move_penalty) * move_penalty
            + float(args.advisor_sep_penalty) * sep_penalty
            + float(args.advisor_boundary_penalty) * boundary_penalty
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            action.clamp_(-1.0, 1.0)

    with torch.no_grad():
        norm = torch.linalg.norm(action, dim=1, keepdim=True).clamp_min(1.0)
        action_proj = action / norm
    return action_proj.detach().cpu().numpy().astype(np.float32)


def learn_central_actor(
    actor,
    states,
    actions,
    advantages,
    old_logprobs,
    advisor_actions,
    advisor_weights,
    target_kl=0.0,
):
    device = ppo_agent_mod.device
    bs = torch.as_tensor(states, dtype=torch.float32, device=device)
    ba = torch.as_tensor(actions, dtype=torch.float32, device=device)
    adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    old_lp = torch.as_tensor(old_logprobs, dtype=torch.float32, device=device)
    advisor = torch.as_tensor(advisor_actions, dtype=torch.float32, device=device)
    bc_w = torch.as_tensor(advisor_weights, dtype=torch.float32, device=device)

    if adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    else:
        adv = adv - adv.mean()

    last_loss = None
    last_ppo = None
    last_bc = None
    last_kl = 0.0
    last_clip_fraction = 0.0
    epochs_run = 0
    for _ in range(A_UPDATE_STEPS):
        new_lp, mean = actor.action_log_prob(actor.new_pi, bs, ba)
        log_ratio = new_lp - old_lp.detach()
        ratio = torch.exp(log_ratio)
        surr1 = ratio * adv
        surr2 = torch.clamp(
            ratio,
            1.0 - CLIP_EPSILON,
            1.0 + CLIP_EPSILON,
        ) * adv
        ppo_loss = -torch.min(surr1, surr2).mean()

        bc_per_sample = ((mean - advisor) ** 2).mean(dim=1, keepdim=True)
        bc_loss = (bc_w * bc_per_sample).mean()
        loss = ppo_loss + bc_loss

        actor.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.new_pi.parameters(), 0.5)
        actor.optimizer.step()

        with torch.no_grad():
            updated_lp, _ = actor.action_log_prob(actor.new_pi, bs, ba)
            updated_log_ratio = updated_lp - old_lp.detach()
            updated_ratio = torch.exp(updated_log_ratio)
            approx_kl = torch.mean(
                (updated_ratio - 1.0) - updated_log_ratio
            )
            clip_fraction = torch.mean(
                (torch.abs(updated_ratio - 1.0) > CLIP_EPSILON).to(
                    torch.float32
                )
            )

        last_loss = loss.item()
        last_ppo = ppo_loss.item()
        last_bc = bc_loss.item()
        last_kl = float(approx_kl.item())
        last_clip_fraction = float(clip_fraction.item())
        epochs_run += 1
        if float(target_kl) > 0.0 and last_kl > float(target_kl):
            break

    return (
        float(last_loss),
        float(last_ppo),
        float(last_bc),
        float(last_kl),
        float(last_clip_fraction),
        int(epochs_run),
    )


def update_sac(
    sac_actor,
    sac_critic,
    sac_entropy,
    sac_memory,
    sac_state_dim,
    sac_action_dim,
    args,
    energy_lagrange_multiplier=0.0,
    update_actor=True,
):
    device = sac_agent_mod.device
    actor_losses = []
    critic_losses = []
    actor_grad_norms = []
    critic_grad_norms = []
    for _ in range(args.sac_gradient_steps):
        batch = sac_memory.sample(args.sac_batch_size)
        b_s = torch.as_tensor(batch[:, :sac_state_dim], dtype=torch.float32, device=device)
        b_a = torch.as_tensor(
            batch[:, sac_state_dim:sac_state_dim + sac_action_dim],
            dtype=torch.float32,
            device=device,
        )
        reward_col = sac_state_dim + sac_action_dim
        b_r = torch.as_tensor(
            batch[:, reward_col:reward_col + 1],
            dtype=torch.float32,
            device=device,
        )
        extras_col = reward_col + 1
        if str(getattr(args, "objective_mode", "weighted")) == "constrained_delay":
            b_controllable_energy_cost = torch.as_tensor(
                batch[:, extras_col:extras_col + 1],
                dtype=torch.float32,
                device=device,
            )
            if str(
                getattr(args, "power_constraint_scope", "system")
            ) != "per_uav":
                b_r = b_r - (
                    float(args.micro_reward_scale)
                    * float(energy_lagrange_multiplier)
                    * b_controllable_energy_cost
                )
            extras_col += 1
        b_discount = torch.as_tensor(
            batch[:, extras_col:extras_col + 1],
            dtype=torch.float32,
            device=device,
        )
        b_s_ = torch.as_tensor(batch[:, -sac_state_dim:], dtype=torch.float32, device=device)

        new_action, log_prob_ = sac_actor.evaluate(b_s_)
        target_q1, target_q2 = sac_critic.target_critic_v(b_s_, new_action)
        target_q = b_r + b_discount * (
            torch.min(target_q1, target_q2)
            - sac_entropy.alpha.detach().to(device) * log_prob_
        )

        current_q1, current_q2 = sac_critic.get_v(b_s, b_a)
        critic_grad_norm = sac_critic.learn(
            current_q1,
            current_q2,
            target_q.detach(),
            max_grad_norm=float(args.sac_grad_clip_norm),
        )

        if update_actor:
            a_eval, log_prob = sac_actor.evaluate(b_s)
            q1_eval, q2_eval = sac_critic.get_v(b_s, a_eval)
            actor_loss = (
                sac_entropy.alpha.detach().to(device) * log_prob
                - torch.min(q1_eval, q2_eval)
            ).mean()
            actor_grad_norm = sac_actor.learn(
                actor_loss,
                max_grad_norm=float(args.sac_grad_clip_norm),
            )

            alpha_loss = sac_entropy.temperature_loss(log_prob)
            sac_entropy.learn(alpha_loss)
            sac_entropy.alpha = sac_entropy.log_alpha.exp().to(device)
            actor_losses.append(float(actor_loss.item()))
            actor_grad_norms.append(float(actor_grad_norm))
        sac_critic.soft_update()

        critic_loss = (
            F.mse_loss(current_q1, target_q.detach())
            + F.mse_loss(current_q2, target_q.detach())
        )
        critic_losses.append(float(critic_loss.item()))
        critic_grad_norms.append(float(critic_grad_norm))

    return actor_losses, critic_losses, actor_grad_norms, critic_grad_norms


def compose_smdp_boundary_transition(
    service_frame,
    deployment_frames,
    gamma=SAC_GAMMA,
    objective_mode="constrained_delay",
    power_constraint_scope="system",
):
    """Attach the next period's two deployment halves to the last service frame."""
    if len(deployment_frames) != 2:
        raise ValueError("Strict 1 s deployment requires exactly two 0.5 s frames.")
    use_conditional_lagrangian = (
        str(objective_mode) == "constrained_delay"
        and str(power_constraint_scope) == "per_uav"
    )
    reward_key = (
        "primal_reward_sum"
        if str(objective_mode) == "constrained_delay"
        and not use_conditional_lagrangian
        else "reward_sum"
    )
    service_reward = float(service_frame[reward_key])
    deployment_holding_rewards = [
        float(
            frame["reward_sum"]
            if use_conditional_lagrangian
            else frame["holding_reward_sum"]
        )
        for frame in deployment_frames
    ]
    reward = service_reward
    for offset, holding_reward in enumerate(
        deployment_holding_rewards,
        start=1,
    ):
        reward += float(gamma) ** offset * holding_reward
    if use_conditional_lagrangian:
        if "energy_constraint_ratio_per_uav_mean" not in service_frame:
            raise KeyError(
                "Per-UAV SMDP transition is missing "
                "energy_constraint_ratio_per_uav_mean."
            )
        energy_ratio_per_uav = np.asarray(
            service_frame["energy_constraint_ratio_per_uav_mean"],
            dtype=np.float64,
        ).reshape(-1)
    else:
        energy_ratio_per_uav = np.asarray(
            [service_frame.get("energy_constraint_ratio_mean", 0.0)],
            dtype=np.float64,
        )
    return {
        "reward": float(reward),
        "service_reward": service_reward,
        "deployment_holding_reward_1": deployment_holding_rewards[0],
        "deployment_holding_reward_2": deployment_holding_rewards[1],
        "energy_constraint_ratio": float(
            service_frame.get("energy_constraint_ratio_mean", 0.0)
        ),
        "controllable_energy_cost": float(
            service_frame.get(
                "service_energy_constraint_ratio_sum",
                0.0,
            )
        ),
        "energy_constraint_ratio_per_uav": energy_ratio_per_uav.copy(),
        "bootstrap_discount": float(gamma) ** 3,
        "discount_exponent": 3,
        "duration_s": float(service_frame["duration_s"])
        + sum(float(frame["duration_s"]) for frame in deployment_frames),
    }


def store_sac_replay_transition(
    memory,
    state,
    action,
    reward,
    next_state,
    objective_mode,
    bootstrap_discount,
    controllable_energy_cost=0.0,
):
    extras = []
    if str(objective_mode) == "constrained_delay":
        extras.append(float(controllable_energy_cost))
    extras.append(float(bootstrap_discount))
    memory.store_transition(
        state,
        action,
        float(reward),
        next_state,
        extras=extras,
    )


def main():
    args = parse_args()
    use_sac = args.micro_solver == "sac"
    use_sca = args.micro_solver == "sca"
    use_local = args.micro_solver == "local"
    sca_policy_fn = None
    if use_sca:
        from compare_sca import solve_sca_micro_action

        sca_policy_fn = solve_sca_micro_action

    requested = args.device
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU")
        requested = "cpu"
    device = torch.device(requested)
    ppo_agent_mod.device = device
    sac_agent_mod.device = device
    print(f"[INFO] Training on {str(device).upper()}")
    print("[INFO] Macro policy: centralized dx/dy actor")
    print(f"[INFO] Micro solver: {args.micro_solver.upper()}")
    print(f"[INFO] Macro reward mode: {args.macro_reward_mode}")
    print(
        f"[INFO] Objective: {args.objective_mode} "
        f"power_budget={args.power_budget_w:.2f}W"
    )
    print(f"[INFO] macro_comm_weight={args.macro_comm_weight}")
    print(
        f"[INFO] shaping={args.macro_shaping_initial}->"
        f"{args.macro_shaping_final} over "
        f"{args.macro_shaping_decay_episodes} episodes"
    )
    print(
        f"[INFO] move_penalty={args.macro_move_penalty_weight} "
        f"propulsion_tiebreak={args.macro_propulsion_tiebreak_weight}"
    )
    if float(args.train_load_min) > 0.0 and float(args.train_load_max) >= float(
        args.train_load_min
    ):
        print(
            f"[INFO] randomized train load_scale in "
            f"[{args.train_load_min}, {args.train_load_max}]"
        )
    if float(args.train_cpu_min) > 0.0 and float(args.train_cpu_max) >= float(
        args.train_cpu_min
    ):
        print(
            f"[INFO] randomized train cpu_ghz in "
            f"[{args.train_cpu_min}, {args.train_cpu_max}]"
        )
    print(f"[INFO] training scenario mode: {args.train_scenario_mode}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    env = make_env_from_args(args, record_trajectory=False)
    eval_env = make_env_from_args(args, record_trajectory=False)
    env.force_local_routing = bool(use_local)
    eval_env.force_local_routing = bool(use_local)
    configure_macro_objective(env, args)
    configure_macro_objective(eval_env, args, args.macro_shaping_final)
    if use_sca:
        configure_sca_wrapper(env, args)
        configure_sca_wrapper(eval_env, args)
    results_dirs = prepare_results_dirs(args.results_subdir)
    save_scenario_metadata(
        results_dirs,
        args,
        {
            "sac": "CentralizedMacroSAC",
            "sca": "CentralizedMacroSCA",
            "local": "CentralizedMacroNoCoop",
        }[args.micro_solver],
    )

    sac_action_dim = env.micro_action_space.shape[0]
    sac_state_dim = env.sac_obs_dim if use_sac else None
    sac_actor = SAC_Actor(sac_state_dim, sac_action_dim) if use_sac else None
    sac_critic = SAC_Critic(sac_state_dim, sac_action_dim) if use_sac else None
    sac_entropy = (
        SAC_Entropy(
            sac_action_dim,
            initial_alpha=args.sac_initial_alpha,
        )
        if use_sac
        else None
    )
    configure_sac_learning_rates(sac_actor, sac_critic, sac_entropy, args)
    sac_memory = (
        SAC_Memory(
            SAC_MEMORY_CAPACITY,
            2 * sac_state_dim
            + sac_action_dim
            + 1
            + int(args.objective_mode == "constrained_delay")
            + 1,
            recent_window=SAC_RECENT_WINDOW,
            recent_ratio=SAC_RECENT_RATIO,
            rng=np.random.default_rng(
                np.random.SeedSequence([int(args.seed), 0x534143])
            ),
        )
        if use_sac and not args.freeze_micro
        else None
    )
    if use_sac and args.micro_ckpt:
        loaded_path = load_micro_checkpoint(
            args.micro_ckpt,
            sac_actor,
            sac_critic,
            sac_entropy,
            load_optimizers=bool(args.resume_micro_optimizer),
        )
        configure_sac_learning_rates(sac_actor, sac_critic, sac_entropy, args)
        print(f"[INFO] Loaded lower SAC -> {loaded_path}")
        if args.objective_mode == "constrained_delay":
            inherited_dual = load_checkpoint_energy_dual(
                loaded_path,
                fallback=args.energy_dual_initial,
                num_uavs=(
                    env.N
                    if args.power_constraint_scope == "per_uav"
                    else None
                ),
            )
            if args.power_constraint_scope == "per_uav":
                env.set_energy_lagrange_multipliers(inherited_dual)
                eval_env.set_energy_lagrange_multipliers(inherited_dual)
            else:
                env.set_energy_lagrange_multipliers(inherited_dual)
                eval_env.set_energy_lagrange_multipliers(inherited_dual)
            args.energy_lagrange_multiplier_current = inherited_dual
            print(
                "[INFO] Inherited energy dual -> "
                f"{np.asarray(inherited_dual)}"
            )
    if use_sac and args.micro_replay:
        replay_path, replay_count = load_sac_replay(
            args.micro_replay,
            sac_memory,
        )
        print(
            f"[INFO] Loaded lower SAC replay ({replay_count} transitions) "
            f"-> {replay_path}"
        )
    if use_sac and args.freeze_micro:
        sac_actor.action_net.eval()
        sac_critic.critic_v.eval()
        sac_critic.target_critic_v.eval()
        for module in (
            sac_actor.action_net,
            sac_critic.critic_v,
            sac_critic.target_critic_v,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        mode = "stochastic" if args.frozen_micro_stochastic else "deterministic"
        print(f"[INFO] Lower SAC frozen; training actions are {mode}")
    eval_micro_policy = (
        sac_micro_policy(sac_actor)
        if use_sac
        else (
            sca_micro_policy(sca_policy_fn, args)
            if use_sca
            else lambda policy_env, _eval_mode: np.zeros(
                policy_env.micro_action_space.shape[0],
                dtype=np.float32,
            )
        )
    )

    if int(args.micro_steps) != int(env.env.micro_steps_per_macro):
        raise ValueError(
            f"micro_steps={args.micro_steps} does not match physical macro slot "
            f"length {env.env.micro_steps_per_macro}."
        )

    _, global_state0 = env.reset(seed=args.seed)[0]
    global_state_dim = int(np.asarray(global_state0).size)
    central_action_dim = int(args.num_agents) * 2
    macro_actor = PPO_Actor(global_state_dim, central_action_dim)
    configure_ppo_hover_policy(
        macro_actor,
        args,
        initialize_hover=not bool(args.macro_ckpt),
    )
    configure_ppo_log_std_schedule(macro_actor, args)
    macro_critic = PPO_Critic(global_state_dim, 1)
    energy_value_dim = env.N if args.power_constraint_scope == "per_uav" else 1
    macro_energy_critic = PPO_Critic(global_state_dim, energy_value_dim)
    if args.macro_ckpt:
        loaded_macro_path = load_central_macro_checkpoint(
            args.macro_ckpt,
            macro_actor,
            macro_critic,
            macro_energy_critic,
        )
        print(f"[INFO] Loaded centralized macro PPO -> {loaded_macro_path}")
    if args.freeze_macro:
        if args.frozen_macro_policy == "zero":
            print(
                "[INFO] Macro PPO frozen; zero residual centroid tracking is "
                "used and only the micro layer is trained"
            )
        else:
            print(
                "[INFO] Macro PPO frozen; deterministic actor actions are used "
                "and only the micro layer is trained"
            )
    for param_group in macro_actor.optimizer.param_groups:
        param_group["lr"] = float(args.ppo_actor_lr)

    eval_history = EvaluationHistory()
    dual_power_history = deque(maxlen=int(args.energy_dual_power_window))
    ppo_states = []
    ppo_actions = []
    ppo_old_logprobs = []
    ppo_delay_returns = []
    ppo_energy_returns = []
    ppo_delay_advantages = []
    ppo_energy_advantages = []
    ppo_advisor_actions = []
    ppo_advisor_weights = []
    sac_trainable_micro_steps = 0
    sac_critic_update_count = 0
    sac_actor_update_count = 0

    best_eval_rank = None
    eval_checkpoint_records = []
    ckpt_dir = results_dirs["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    train_csv = os.path.join(results_dirs["results_dir"], "train_centralized.csv")
    eval_stem = {
        "sac": "eval_central_macro",
        "sca": "eval_central_sca",
        "local": "eval_central_no_coop",
    }[args.micro_solver]
    eval_csv = os.path.join(results_dirs["results_dir"], f"{eval_stem}.csv")
    load_switch_trace_csv = os.path.join(
        results_dirs["results_dir"],
        "load_switch_macro_trace.csv",
    )
    initialize_training_log(train_csv)
    if str(getattr(args, "load_switch_mode", "off")) != "off":
        initialize_load_switch_trace(load_switch_trace_csv)
    eval_history.save_csv(eval_csv)

    for episode in range(args.episodes):
        episode_started = time.perf_counter()
        actor_lr = ppo_actor_lr_for_episode(args, episode)
        scheduled_log_std = ppo_log_std_for_episode(args, episode)
        set_ppo_log_std(macro_actor, scheduled_log_std)
        ppo_action_std = float(
            torch.exp(macro_actor.old_pi.log_std.detach()).mean().cpu().item()
        )
        shaping_weight = macro_shaping_weight_for_episode(args, episode)
        configure_macro_objective(env, args, shaping_weight)
        for param_group in macro_actor.optimizer.param_groups:
            param_group["lr"] = actor_lr
        sca_count_started = int(getattr(env, "_sca_solve_count", 0))
        sca_seconds_started = float(
            getattr(env, "_sca_total_solve_seconds", 0.0)
        )
        should_record = (
            args.trajectory_plot_interval > 0
            and (episode + 1) % args.trajectory_plot_interval == 0
        )
        env.env.record_trajectory = should_record
        episode_load, episode_cpu_ghz = sample_training_scenario(args)
        set_wrapper_load_scale(env, episode_load)
        set_wrapper_cpu_ghz(env, episode_cpu_ghz)
        training_start_positions = sample_training_start_positions(
            env,
            args,
            episode,
        )
        env.env.uav_initial_positions = training_start_positions.copy()
        start_offsets = np.linalg.norm(
            training_start_positions
            - np.asarray(env.env.anchor_points, dtype=np.float64),
            axis=1,
        )
        episode_seed = args.seed + episode
        load_switch_schedule = initialize_episode_load_switch(
            env,
            args,
            episode_seed,
        )
        _, info = env.reset(seed=episode_seed)
        if load_switch_schedule is not None:
            apply_load_switch_phase(
                env,
                load_switch_schedule,
                0,
                force=True,
            )
        _local_state, global_state = env.get_macro_state()

        ep_states = []
        ep_actions = []
        ep_logprobs = []
        ep_advisor_actions = []
        ep_advisor_weights = []
        ep_macro_reward = 0.0
        ep_micro_reward = 0.0
        ep_j = []
        ep_delay_objective = []
        ep_service_energy = []
        ep_comm = []
        ep_u2g = []
        ep_move = []
        ep_all_hover = []
        ep_uav_hover = []
        ep_macro_energy = []
        ep_service_battery_energy_per_uav = np.zeros(
            env.N,
            dtype=np.float64,
        )
        ep_macro_energy_per_uav = np.zeros(env.N, dtype=np.float64)
        ep_propulsion_delta = []
        ep_deployment_transient_energy = []
        ep_macro_feedback = []
        ep_macro_micro_steps = []
        ep_macro_reconstruction_error = []
        ep_micro_diagnostics = initialize_micro_reward_diagnostics()
        ep_macro_diagnostics = initialize_reward_diagnostics(
            MACRO_REWARD_DIAGNOSTIC_SOURCES
        )
        ep_projected_uplink_time = []
        ep_projected_u2u_time = []
        ep_projected_u2u_rate = []
        ep_delay_rewards = []
        ep_energy_costs = []
        done = False
        energy_dual_before = (
            env.energy_lagrange_multipliers.copy()
            if args.power_constraint_scope == "per_uav"
            else float(env.energy_lagrange_multiplier)
        )
        energy_dual_after = energy_dual_before
        pending_sac_transition = None
        ep_service_resource_decisions = 0
        ep_smdp_boundary_transitions = 0
        ep_smdp_boundary_audit = None
        ep_sac_actor_losses = []
        ep_sac_critic_losses = []
        ep_sac_actor_grad_norms = []
        ep_sac_critic_grad_norms = []

        bc_weight = advisor_weight_for_episode(args, episode)

        for macro_idx in range(args.macro_steps):
            switched = advance_load_switch_before_macro(
                env,
                load_switch_schedule,
                macro_idx,
            )
            if switched:
                _local_state, global_state = env.get_macro_state()
            macro_service_energy_start = len(ep_service_energy)
            macro_delay_start = len(ep_delay_objective)
            macro_energy_current = 0.0
            macro_move_current = 0.0
            advisor_action = (
                compute_advisor_action(env, args)
                if bc_weight > 0.0
                else np.zeros((env.N, 2), dtype=np.float32)
            )
            if args.freeze_macro and args.frozen_macro_policy == "zero":
                action_flat = np.zeros((env.N * 2,), dtype=np.float32)
                logprob = np.zeros((1,), dtype=np.float32)
            elif args.freeze_macro or args.macro_deterministic:
                action_flat = macro_actor.choose_action_deterministic(global_state)
                logprob = np.zeros((1,), dtype=np.float32)
            else:
                action_flat, logprob = macro_actor.choose_action(global_state)
            policy_action_flat, action = prepare_macro_policy_action(
                action_flat,
                env.N,
            )
            hover_mask = np.linalg.norm(action, axis=1) <= 1e-8
            ep_all_hover.append(float(np.all(hover_mask)))
            ep_uav_hover.append(float(np.mean(hover_mask)))

            _, _, mac_info = env.macro_step(action)
            if isinstance(mac_info, dict):
                ep_comm.append(float(mac_info.get("projected_macro_comm_cost", 0.0)))
                ep_u2g.append(float(mac_info.get("projected_mean_u2g_distance_m", 0.0)))
                macro_move_current = float(
                    mac_info.get("macro_move_distance_mean", 0.0)
                )
                ep_move.append(macro_move_current)
                macro_energy_current = float(
                    mac_info.get("E_macro_total", 0.0)
                )
                ep_macro_energy.append(macro_energy_current)
                ep_macro_energy_per_uav += np.asarray(
                    mac_info.get("E_macro_per_uav", np.zeros(env.N)),
                    dtype=np.float64,
                )
                ep_deployment_transient_energy.append(
                    float(mac_info.get("E_deployment_transient", 0.0))
                )
                ep_propulsion_delta.append(
                    float(mac_info.get("macro_propulsion_delta_total", 0.0))
                )
                ep_projected_uplink_time.append(
                    float(mac_info.get("projected_comm_uplink_time_s", 0.0))
                )
                ep_projected_u2u_time.append(
                    float(mac_info.get("projected_comm_u2u_time_s", 0.0))
                )
                ep_projected_u2u_rate.append(
                    float(
                        mac_info.get(
                            "projected_mean_u2u_rate_mbps",
                            0.0,
                        )
                    )
                )

            deployment_frames = []
            zero_micro_action = np.zeros(sac_action_dim, dtype=np.float32)
            for _ in range(env.env.deployment_resource_frames):
                frame = env.step_resource_frame(zero_micro_action)
                deployment_frames.append(frame)
                ep_micro_reward += float(frame["reward_sum"])
                for mic_info in frame["infos"]:
                    record_micro_reward_diagnostics(
                        ep_micro_diagnostics,
                        mic_info,
                    )
                    ep_j.append(float(mic_info.get("J_step", 0.0)))
                    ep_delay_objective.append(
                        float(mic_info.get("delay_objective_step", 0.0))
                    )
                    ep_service_energy.append(
                        float(mic_info.get("service_energy_step_j", 0.0))
                    )
                    ep_service_battery_energy_per_uav += np.asarray(
                        mic_info.get(
                            "uav_battery_energy_per_uav",
                            np.zeros(env.N),
                        ),
                        dtype=np.float64,
                    )
                done = done or bool(frame["done"])

            current_service_state = env.prepare_resource_frame()
            if pending_sac_transition is not None:
                boundary = compose_smdp_boundary_transition(
                    pending_sac_transition["frame"],
                    deployment_frames,
                    gamma=SAC_GAMMA,
                    objective_mode=args.objective_mode,
                    power_constraint_scope=args.power_constraint_scope,
                )
                if ep_smdp_boundary_audit is None:
                    ep_smdp_boundary_audit = dict(boundary)
                store_sac_replay_transition(
                    sac_memory,
                    pending_sac_transition["state"],
                    pending_sac_transition["action"],
                    boundary["reward"],
                    current_service_state,
                    args.objective_mode,
                    boundary["bootstrap_discount"],
                    boundary["controllable_energy_cost"],
                )
                ep_smdp_boundary_transitions += 1
                pending_sac_transition = None

            for service_frame_index in range(env.env.service_resource_frames):
                ep_service_resource_decisions += 1
                sac_obs = np.asarray(current_service_state, dtype=np.float32)
                if use_sac:
                    frozen_deterministic = (
                        args.freeze_micro
                        and not args.frozen_micro_stochastic
                    )
                    micro_action = sac_actor.choose_action(
                        sac_obs,
                        deterministic=frozen_deterministic,
                    )
                elif use_sca:
                    micro_action = eval_micro_policy(env, False)
                else:
                    micro_action = zero_micro_action

                frame = env.step_resource_frame(micro_action)
                ep_micro_reward += float(frame["reward_sum"])
                for mic_info in frame["infos"]:
                    record_micro_reward_diagnostics(
                        ep_micro_diagnostics,
                        mic_info,
                    )
                    ep_j.append(float(mic_info.get("J_step", 0.0)))
                    ep_delay_objective.append(
                        float(mic_info.get("delay_objective_step", 0.0))
                    )
                    ep_service_energy.append(
                        float(mic_info.get("service_energy_step_j", 0.0))
                    )
                    ep_service_battery_energy_per_uav += np.asarray(
                        mic_info.get(
                            "uav_battery_energy_per_uav",
                            np.zeros(env.N),
                        ),
                        dtype=np.float64,
                    )
                done = done or bool(frame["done"])

                should_train_sac = use_sac and not args.freeze_micro
                is_last_service_frame = (
                    service_frame_index == env.env.service_resource_frames - 1
                )
                if should_train_sac:
                    sac_trainable_micro_steps += int(frame["substeps"])
                    if is_last_service_frame:
                        pending_sac_transition = {
                            "state": sac_obs.copy(),
                            "action": np.asarray(
                                micro_action,
                                dtype=np.float32,
                            ).copy(),
                            "frame": frame,
                        }
                    else:
                        next_service_state = env.prepare_resource_frame()
                        reward_key = (
                            "primal_reward_sum"
                            if args.objective_mode == "constrained_delay"
                            and args.power_constraint_scope != "per_uav"
                            else "reward_sum"
                        )
                        store_sac_replay_transition(
                            sac_memory,
                            sac_obs,
                            micro_action,
                            frame[reward_key],
                            next_service_state,
                            args.objective_mode,
                            SAC_GAMMA,
                            frame["service_energy_constraint_ratio_sum"],
                        )
                        current_service_state = next_service_state
                elif not is_last_service_frame:
                    current_service_state = env.prepare_resource_frame()

                if (
                    should_train_sac
                    and sac_memory.memory_counter > SAC_MEMORY_WARMUP
                    and sac_trainable_micro_steps % args.sac_update_freq == 0
                ):
                    next_sac_critic_update_count = (
                        sac_critic_update_count
                        + max(int(args.sac_gradient_steps), 1)
                    )
                    sac_actor_ready = (
                        next_sac_critic_update_count
                        > int(args.sac_actor_warmup_updates)
                    )
                    sac_actor_due = (
                        next_sac_critic_update_count
                        % max(int(args.sac_policy_delay), 1)
                        == 0
                    )
                    (
                        sac_actor_losses,
                        sac_critic_losses,
                        sac_actor_grad_norms,
                        sac_critic_grad_norms,
                    ) = update_sac(
                        sac_actor,
                        sac_critic,
                        sac_entropy,
                        sac_memory,
                        sac_state_dim,
                        sac_action_dim,
                        args,
                        energy_lagrange_multiplier=(
                            env.energy_lagrange_multiplier
                        ),
                        update_actor=(
                            sac_actor_ready and sac_actor_due
                        ),
                    )
                    sac_critic_update_count = next_sac_critic_update_count
                    sac_actor_update_count += len(sac_actor_losses)
                    ep_sac_actor_losses.extend(sac_actor_losses)
                    ep_sac_critic_losses.extend(sac_critic_losses)
                    ep_sac_actor_grad_norms.extend(sac_actor_grad_norms)
                    ep_sac_critic_grad_norms.extend(sac_critic_grad_norms)
                if done:
                    break

            macro_reward = float(env.get_macro_reward())
            macro_reward_info = env.get_last_macro_reward_info()
            record_reward_diagnostics(
                ep_macro_diagnostics,
                macro_reward_info,
                MACRO_REWARD_DIAGNOSTIC_SOURCES,
            )
            delay_reward, energy_cost = split_macro_lagrangian_reward(
                macro_reward,
                macro_reward_info,
                env,
            )
            if load_switch_schedule is not None:
                macro_service_energy_current = float(
                    np.sum(ep_service_energy[macro_service_energy_start:])
                )
                macro_delay_values = ep_delay_objective[macro_delay_start:]
                append_load_switch_trace(
                    load_switch_trace_csv,
                    {
                        "episode": episode + 1,
                        "episode_seed": episode_seed,
                        "macro_index": macro_idx,
                        "phase_index": load_switch_schedule.phase_index(
                            macro_idx
                        ),
                        "switch_flag": int(switched),
                        "phase_loads_mbps": json.dumps(
                            load_switch_schedule.loads(macro_idx).tolist(),
                            separators=(",", ":"),
                        ),
                        "energy_dual": float(
                            np.mean(
                                np.atleast_1d(
                                    env.energy_lagrange_multiplier
                                )
                            )
                        ),
                        "macro_energy_j": macro_energy_current,
                        "service_energy_j": macro_service_energy_current,
                        "macro_average_power_w": (
                            macro_energy_current
                            + macro_service_energy_current
                        )
                        / max(float(env.env.macro_dt), 1e-8),
                        "macro_reward": macro_reward,
                        "delay_objective_mean": float(
                            np.mean(macro_delay_values)
                        )
                        if macro_delay_values
                        else 0.0,
                        "movement_mean_m": macro_move_current,
                    },
                )
            ep_macro_feedback.append(
                float(macro_reward_info.get("global_micro_signal", 0.0))
            )
            ep_macro_micro_steps.append(
                int(macro_reward_info.get("micro_steps_executed", 0))
            )
            ep_macro_reconstruction_error.append(
                abs(
                    float(
                        macro_reward_info.get(
                            "reward_reconstruction_error",
                            0.0,
                        )
                    )
                )
            )
            next_local_state, next_global_state = env.get_macro_state()
            ep_macro_reward += macro_reward

            ep_states.append(np.asarray(global_state, dtype=np.float32))
            ep_actions.append(policy_action_flat.copy())
            ep_logprobs.append(
                np.asarray(logprob, dtype=np.float32).reshape(1)
            )
            ep_delay_rewards.append(delay_reward)
            ep_energy_costs.append(energy_cost)
            ep_advisor_actions.append(advisor_action.reshape(-1).astype(np.float32))
            ep_advisor_weights.append(np.asarray([bc_weight], dtype=np.float32))

            global_state = next_global_state
            if done:
                break

        if pending_sac_transition is not None:
            terminal_frame = pending_sac_transition["frame"]
            reward_key = (
                "primal_reward_sum"
                if args.objective_mode == "constrained_delay"
                and args.power_constraint_scope != "per_uav"
                else "reward_sum"
            )
            store_sac_replay_transition(
                sac_memory,
                pending_sac_transition["state"],
                pending_sac_transition["action"],
                terminal_frame[reward_key],
                terminal_frame["next_state"],
                args.objective_mode,
                0.0,
                terminal_frame["service_energy_constraint_ratio_sum"],
            )
            pending_sac_transition = None

        episode_duration_s = max(
            len(ep_macro_energy) * float(env.env.macro_dt),
            float(env.env.micro_dt),
        )
        service_energy_episode = float(np.sum(ep_service_energy))
        total_energy_episode = (
            service_energy_episode + float(np.sum(ep_macro_energy))
        )
        average_power_w = total_energy_episode / episode_duration_s
        average_battery_power_per_uav_w = (
            ep_service_battery_energy_per_uav + ep_macro_energy_per_uav
        ) / episode_duration_s
        if args.power_constraint_scope == "per_uav":
            dual_power_history.append(
                average_battery_power_per_uav_w.copy()
            )
            dual_update_power_w = np.mean(
                np.stack(tuple(dual_power_history), axis=0),
                axis=0,
            )
            _, energy_dual_after, power_violation_ratio = (
                update_per_uav_energy_duals(
                    env,
                    args,
                    dual_update_power_w,
                )
            )
            args.energy_lagrange_multiplier_current = float(
                np.mean(energy_dual_after)
            )
            args.energy_lagrange_multipliers_current = (
                np.asarray(energy_dual_after).tolist()
            )
        else:
            if args.objective_mode == "constrained_delay":
                power_violation_ratio = (
                    average_power_w - float(args.power_budget_w)
                ) / max(float(args.power_budget_w), 1e-8)
            else:
                power_violation_ratio = 0.0
            dual_power_history.append(float(average_power_w))
            dual_update_power_w = float(np.mean(dual_power_history))
            _, energy_dual_after, _ = update_energy_dual(
                env,
                args,
                dual_update_power_w,
            )
            args.energy_lagrange_multiplier_current = energy_dual_after

        if ep_states and not args.freeze_macro:
            T = len(ep_states)
            state_arr = np.vstack(ep_states).astype(np.float32)
            delay_values = (
                macro_critic.get_v(state_arr)
                .detach()
                .cpu()
                .numpy()
                .reshape(T)
            )
            energy_values_raw = (
                macro_energy_critic.get_v(state_arr)
                .detach()
                .cpu()
                .numpy()
            )
            energy_values = (
                energy_values_raw.reshape(T, env.N)
                if args.power_constraint_scope == "per_uav"
                else energy_values_raw.reshape(T)
            )
            delay_next_value = float(
                macro_critic.get_v(global_state)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)[0]
            )
            energy_next_raw = (
                macro_energy_critic.get_v(global_state)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            energy_next_value = (
                energy_next_raw.copy()
                if args.power_constraint_scope == "per_uav"
                else float(energy_next_raw[0])
            )
            delay_adv, delay_returns = generalized_advantage_estimate(
                ep_delay_rewards,
                delay_values,
                delay_next_value,
                args.ppo_gamma,
                args.ppo_gae_lambda,
            )
            energy_adv, energy_returns = generalized_advantage_estimate(
                ep_energy_costs,
                energy_values,
                energy_next_value,
                args.ppo_gamma,
                args.ppo_gae_lambda,
            )

            ppo_states.extend(ep_states)
            ppo_actions.extend(ep_actions)
            ppo_old_logprobs.extend(ep_logprobs)
            ppo_delay_returns.extend(list(delay_returns.reshape(-1, 1)))
            if args.power_constraint_scope == "per_uav":
                ppo_energy_returns.extend(list(energy_returns.reshape(T, env.N)))
            else:
                ppo_energy_returns.extend(list(energy_returns.reshape(-1, 1)))
            ppo_delay_advantages.extend(list(delay_adv.reshape(-1, 1)))
            if args.power_constraint_scope == "per_uav":
                ppo_energy_advantages.extend(list(energy_adv.reshape(T, env.N)))
            else:
                ppo_energy_advantages.extend(list(energy_adv.reshape(-1, 1)))
            ppo_advisor_actions.extend(ep_advisor_actions)
            ppo_advisor_weights.extend(ep_advisor_weights)

        update_threshold = args.macro_steps * max(
            int(args.ppo_update_episode_window),
            1,
        )
        actor_loss = np.nan
        ppo_loss = np.nan
        bc_loss = np.nan
        approx_kl = np.nan
        clip_fraction = np.nan
        update_epochs = 0
        critic_loss = np.nan
        energy_critic_loss = np.nan
        delay_advantage_mean = np.nan
        energy_advantage_mean = np.nan
        combined_advantage_mean = np.nan
        if not args.freeze_macro and (
            len(ppo_states) >= update_threshold
            or (episode == args.episodes - 1 and ppo_states)
        ):
            bs = np.vstack(ppo_states).astype(np.float32)
            b_delay_returns = np.vstack(ppo_delay_returns).astype(np.float32)
            b_energy_returns = np.vstack(ppo_energy_returns).astype(np.float32)
            ba = np.vstack(ppo_actions).astype(np.float32)
            bap = np.vstack(ppo_old_logprobs).astype(np.float32)
            b_delay_adv = np.vstack(ppo_delay_advantages).astype(np.float32)
            b_energy_adv = np.vstack(ppo_energy_advantages).astype(np.float32)
            badv = combine_constrained_advantages(
                b_delay_adv,
                b_energy_adv,
                (
                    env.energy_lagrange_multipliers
                    if args.power_constraint_scope == "per_uav"
                    else env.energy_lagrange_multiplier
                ),
            )
            badvisor = np.vstack(ppo_advisor_actions).astype(np.float32)
            bw = np.vstack(ppo_advisor_weights).astype(np.float32)

            _, critic_loss = macro_critic.learn(bs, b_delay_returns)
            _, energy_critic_loss = macro_energy_critic.learn(
                bs,
                b_energy_returns,
            )
            delay_advantage_mean = float(np.mean(b_delay_adv))
            energy_advantage_mean = float(np.mean(b_energy_adv))
            combined_advantage_mean = float(np.mean(badv))
            (
                actor_loss,
                ppo_loss,
                bc_loss,
                approx_kl,
                clip_fraction,
                update_epochs,
            ) = learn_central_actor(
                macro_actor,
                bs,
                ba,
                badv,
                bap,
                badvisor,
                bw,
                target_kl=args.ppo_target_kl,
            )
            macro_actor.update_oldpi()

            print(
                f"  PPO update | actor={actor_loss:.4f} "
                f"ppo={ppo_loss:.4f} bc={bc_loss:.4f} "
                f"kl={approx_kl:.5f} clip={clip_fraction:.3f} "
                f"epochs={update_epochs} Dcritic={critic_loss:.4f} "
                f"Ecritic={energy_critic_loss:.4f}"
            )

            ppo_states.clear()
            ppo_actions.clear()
            ppo_old_logprobs.clear()
            ppo_delay_returns.clear()
            ppo_energy_returns.clear()
            ppo_delay_advantages.clear()
            ppo_energy_advantages.clear()
            ppo_advisor_actions.clear()
            ppo_advisor_weights.clear()

        if args.power_constraint_scope == "per_uav":
            energy_dual_after_values = env.energy_lagrange_multipliers.copy()
            energy_dual_before_log = float(np.mean(energy_dual_before))
            energy_dual_after_log = float(np.mean(energy_dual_after_values))
            power_violation_ratio_log = float(np.max(power_violation_ratio))
            dual_update_power_w_log = float(np.max(dual_update_power_w))
            constrained_power_display = np.array2string(
                average_battery_power_per_uav_w,
                precision=1,
                separator=",",
            )
            dual_display = np.array2string(
                energy_dual_after_values,
                precision=4,
                separator=",",
            )
        else:
            energy_dual_after_values = float(env.energy_lagrange_multiplier)
            energy_dual_before_log = float(energy_dual_before)
            energy_dual_after_log = float(energy_dual_after_values)
            power_violation_ratio_log = float(power_violation_ratio)
            dual_update_power_w_log = float(dual_update_power_w)
            constrained_power_display = f"{average_power_w:.1f}"
            dual_display = f"{energy_dual_after_log:.4f}"
        args.energy_lagrange_multiplier_current = energy_dual_after_log
        print(
            f"Episode {episode} | Macro R {ep_macro_reward:.2f} | "
            f"Micro R {ep_micro_reward:.2f} | "
            f"J {np.mean(ep_j) if ep_j else 0.0:.4f} | "
            f"Comm {np.mean(ep_comm) if ep_comm else 0.0:.3f} | "
            f"U2G {np.mean(ep_u2g) if ep_u2g else 0.0:.1f} | "
            f"Move {np.mean(ep_move) if ep_move else 0.0:.1f} | "
            f"Start {np.mean(start_offsets):.1f}m | "
            f"Load {episode_load:.2f} | "
            f"CPU {episode_cpu_ghz:.2f}GHz | "
            f"Psys {average_power_w:.1f}W | Pcon {constrained_power_display}W | "
            f"Dual {dual_display} | "
            f"BCw {bc_weight:.3f} | Shape {shaping_weight:.3f}"
        )
        append_training_log(
            train_csv,
            {
                "episode": episode + 1,
                "macro_reward": ep_macro_reward,
                "micro_reward": ep_micro_reward,
                "J_train": float(np.mean(ep_j)) if ep_j else 0.0,
                "delay_objective_train": float(
                    np.mean(ep_delay_objective)
                )
                if ep_delay_objective
                else 0.0,
                **summarize_micro_reward_diagnostics(
                    ep_micro_diagnostics
                ),
                "service_energy_episode_j": service_energy_episode,
                "total_energy_episode_j": total_energy_episode,
                "average_power_w": average_power_w,
                "power_budget_w": float(args.power_budget_w),
                "power_target_w": (
                    float(np.mean(env.uav_power_budgets_w))
                    if args.power_constraint_scope == "per_uav"
                    else power_constraint_target_w(args)
                ),
                "power_violation_ratio": power_violation_ratio_log,
                "energy_dual_before": energy_dual_before_log,
                "energy_dual_after": energy_dual_after_log,
                "dual_update_power_w": dual_update_power_w_log,
                "dual_power_window": len(dual_power_history),
                "battery_power_per_uav_w": json.dumps(
                    np.atleast_1d(
                        average_battery_power_per_uav_w
                    ).astype(float).tolist(),
                    separators=(",", ":"),
                ),
                "battery_power_budget_per_uav_w": json.dumps(
                    np.atleast_1d(env.uav_power_budgets_w)
                    .astype(float)
                    .tolist(),
                    separators=(",", ":"),
                ),
                "energy_dual_before_per_uav": json.dumps(
                    np.atleast_1d(energy_dual_before)
                    .astype(float)
                    .tolist(),
                    separators=(",", ":"),
                ),
                "energy_dual_after_per_uav": json.dumps(
                    np.atleast_1d(energy_dual_after_values)
                    .astype(float)
                    .tolist(),
                    separators=(",", ":"),
                ),
                "dual_update_power_per_uav_w": json.dumps(
                    np.atleast_1d(dual_update_power_w)
                    .astype(float)
                    .tolist(),
                    separators=(",", ":"),
                ),
                "cpu_energy_price_per_uav": json.dumps(
                    np.atleast_1d(env.env.cpu_energy_prices)
                    .astype(float)
                    .tolist(),
                    separators=(",", ":"),
                ),
                "comm_cost": float(np.mean(ep_comm)) if ep_comm else 0.0,
                "u2g_m": float(np.mean(ep_u2g)) if ep_u2g else 0.0,
                "move_m": float(np.mean(ep_move)) if ep_move else 0.0,
                "macro_all_hover_ratio": float(np.mean(ep_all_hover))
                if ep_all_hover
                else 0.0,
                "macro_uav_hover_ratio": float(np.mean(ep_uav_hover))
                if ep_uav_hover
                else 0.0,
                "start_offset_mean_m": float(np.mean(start_offsets)),
                "start_offset_max_m": float(np.max(start_offsets)),
                "load_scale": episode_load,
                "cpu_ghz": episode_cpu_ghz,
                "load_switch_mode": str(
                    getattr(args, "load_switch_mode", "off")
                ),
                "load_switch_count": (
                    len(load_switch_schedule.switch_steps)
                    if load_switch_schedule is not None
                    else 0
                ),
                "load_switch_steps": json.dumps(
                    list(load_switch_schedule.switch_steps)
                    if load_switch_schedule is not None
                    else [],
                    separators=(",", ":"),
                ),
                "load_switch_dwell_steps": json.dumps(
                    list(load_switch_schedule.dwell_steps)
                    if load_switch_schedule is not None
                    else [int(args.macro_steps)],
                    separators=(",", ":"),
                ),
                "advisor_weight": bc_weight,
                "macro_shaping_weight": shaping_weight,
                "macro_move_penalty_weight": args.macro_move_penalty_weight,
                "macro_propulsion_tiebreak_weight": (
                    args.macro_propulsion_tiebreak_weight
                ),
                "macro_propulsion_delta_energy": float(
                    np.sum(ep_propulsion_delta)
                ),
                "deployment_transient_energy_episode_j": float(
                    np.sum(ep_deployment_transient_energy)
                ),
                "macro_energy_episode": float(np.sum(ep_macro_energy)),
                "macro_feedback_mean": float(np.mean(ep_macro_feedback))
                if ep_macro_feedback
                else 0.0,
                **summarize_reward_diagnostics(
                    ep_macro_diagnostics
                ),
                "projected_uplink_time_s_mean": float(
                    np.mean(ep_projected_uplink_time)
                )
                if ep_projected_uplink_time
                else 0.0,
                "projected_u2u_time_s_mean": float(
                    np.mean(ep_projected_u2u_time)
                )
                if ep_projected_u2u_time
                else 0.0,
                "projected_u2u_rate_mbps_mean": float(
                    np.mean(ep_projected_u2u_rate)
                )
                if ep_projected_u2u_rate
                else 0.0,
                "macro_micro_steps_min": int(np.min(ep_macro_micro_steps))
                if ep_macro_micro_steps
                else 0,
                "macro_micro_steps_max": int(np.max(ep_macro_micro_steps))
                if ep_macro_micro_steps
                else 0,
                "macro_reward_reconstruction_max_error": float(
                    np.max(ep_macro_reconstruction_error)
                )
                if ep_macro_reconstruction_error
                else 0.0,
                "ppo_actor_lr": actor_lr,
                "ppo_action_std": ppo_action_std,
                "ppo_actor_loss": actor_loss,
                "ppo_policy_loss": ppo_loss,
                "ppo_bc_loss": bc_loss,
                "ppo_approx_kl": approx_kl,
                "ppo_clip_fraction": clip_fraction,
                "ppo_update_epochs": update_epochs,
                "ppo_critic_loss": critic_loss,
                "ppo_energy_critic_loss": energy_critic_loss,
                "ppo_delay_advantage_mean": delay_advantage_mean,
                "ppo_energy_advantage_mean": energy_advantage_mean,
                "ppo_combined_advantage_mean": combined_advantage_mean,
                "sac_trainable_micro_steps": sac_trainable_micro_steps,
                "sac_actor_loss": (
                    float(np.mean(ep_sac_actor_losses))
                    if ep_sac_actor_losses
                    else ""
                ),
                "sac_critic_loss": (
                    float(np.mean(ep_sac_critic_losses))
                    if ep_sac_critic_losses
                    else ""
                ),
                "sac_actor_grad_norm": (
                    float(np.mean(ep_sac_actor_grad_norms))
                    if ep_sac_actor_grad_norms
                    else ""
                ),
                "sac_critic_grad_norm": (
                    float(np.mean(ep_sac_critic_grad_norms))
                    if ep_sac_critic_grad_norms
                    else ""
                ),
                "sac_alpha": (
                    float(sac_entropy.alpha.detach().cpu().item())
                    if sac_entropy is not None
                    else ""
                ),
                "sac_critic_updates_total": sac_critic_update_count,
                "sac_actor_updates_total": sac_actor_update_count,
                "service_resource_decisions": ep_service_resource_decisions,
                "smdp_boundary_transitions": ep_smdp_boundary_transitions,
                "smdp_boundary_duration_s": (
                    ep_smdp_boundary_audit["duration_s"]
                    if ep_smdp_boundary_audit is not None
                    else ""
                ),
                "smdp_boundary_discount_exponent": (
                    ep_smdp_boundary_audit["discount_exponent"]
                    if ep_smdp_boundary_audit is not None
                    else ""
                ),
                "smdp_boundary_service_reward": (
                    ep_smdp_boundary_audit["service_reward"]
                    if ep_smdp_boundary_audit is not None
                    else ""
                ),
                "smdp_boundary_deployment_holding_reward_1": (
                    ep_smdp_boundary_audit[
                        "deployment_holding_reward_1"
                    ]
                    if ep_smdp_boundary_audit is not None
                    else ""
                ),
                "smdp_boundary_deployment_holding_reward_2": (
                    ep_smdp_boundary_audit[
                        "deployment_holding_reward_2"
                    ]
                    if ep_smdp_boundary_audit is not None
                    else ""
                ),
                "smdp_boundary_composed_reward": (
                    ep_smdp_boundary_audit["reward"]
                    if ep_smdp_boundary_audit is not None
                    else ""
                ),
                "smdp_boundary_bootstrap_discount": (
                    ep_smdp_boundary_audit["bootstrap_discount"]
                    if ep_smdp_boundary_audit is not None
                    else ""
                ),
                "sac_replay_size": (
                    sac_memory.memory_counter if sac_memory is not None else 0
                ),
                "sca_solve_count": int(getattr(env, "_sca_solve_count", 0))
                - sca_count_started,
                "sca_solve_seconds": float(
                    getattr(env, "_sca_total_solve_seconds", 0.0)
                )
                - sca_seconds_started,
                "sca_solve_mean_ms": 1000.0
                * (
                    float(getattr(env, "_sca_total_solve_seconds", 0.0))
                    - sca_seconds_started
                )
                / max(
                    int(getattr(env, "_sca_solve_count", 0))
                    - sca_count_started,
                    1,
                ),
                "episode_seconds": time.perf_counter() - episode_started,
            },
        )

        if should_record:
            trajectory_path = os.path.join(
                results_dirs["trajectory_dir"],
                f"trajectory_ep_{episode + 1:04d}.npz",
            )
            save_trajectory_with_verification(env, trajectory_path)

        if args.eval_interval > 0 and (episode + 1) % args.eval_interval == 0:
            if args.power_constraint_scope == "per_uav":
                eval_env.set_energy_lagrange_multipliers(
                    env.energy_lagrange_multipliers
                )
            else:
                eval_env.set_energy_lagrange_multipliers(
                    env.energy_lagrange_multiplier
                )
            metrics = evaluate_policy(
                eval_env,
                args,
                evaluation_macro_policy(args, macro_actor),
                eval_micro_policy,
            )
            eval_history.append(episode + 1, metrics)
            eval_history.save_csv(eval_csv)
            print_evaluation_summary(episode + 1, metrics)
            checkpoint_payload = build_checkpoint_payload(
                args,
                macro_actor,
                macro_critic,
                sac_actor,
                sac_critic,
                sac_entropy,
                macro_energy_critic=macro_energy_critic,
            )
            save_eval_candidate(
                ckpt_dir,
                episode + 1,
                metrics,
                checkpoint_payload,
                eval_checkpoint_records,
            )
            candidate_rank = checkpoint_rank(metrics, args)
            if best_eval_rank is None or candidate_rank < best_eval_rank:
                best_eval_rank = candidate_rank
                torch.save(
                    checkpoint_payload,
                    os.path.join(
                        ckpt_dir,
                        "best_j.pt"
                        if args.checkpoint_epsilon_relative > 0.0
                        else "best.pt",
                    ),
                )

    if not eval_history.steps or eval_history.steps[-1] != args.episodes:
        if args.power_constraint_scope == "per_uav":
            eval_env.set_energy_lagrange_multipliers(
                env.energy_lagrange_multipliers
            )
        else:
            eval_env.set_energy_lagrange_multipliers(
                env.energy_lagrange_multiplier
            )
        metrics = evaluate_policy(
            eval_env,
            args,
            evaluation_macro_policy(args, macro_actor),
            eval_micro_policy,
        )
        eval_history.append(args.episodes, metrics)
        print_evaluation_summary(args.episodes, metrics)
        checkpoint_payload = build_checkpoint_payload(
            args,
            macro_actor,
            macro_critic,
            sac_actor,
            sac_critic,
            sac_entropy,
            macro_energy_critic=macro_energy_critic,
        )
        save_eval_candidate(
            ckpt_dir,
            args.episodes,
            metrics,
            checkpoint_payload,
            eval_checkpoint_records,
        )
        candidate_rank = checkpoint_rank(metrics, args)
        if best_eval_rank is None or candidate_rank < best_eval_rank:
            best_eval_rank = candidate_rank
            torch.save(
                checkpoint_payload,
                os.path.join(
                    ckpt_dir,
                    "best_j.pt"
                    if args.checkpoint_epsilon_relative > 0.0
                    else "best.pt",
                ),
            )

    eval_history.save_csv(eval_csv)
    eval_history.save_detail_npz(
        os.path.join(results_dirs["results_dir"], f"{eval_stem}_detail.npz")
    )
    torch.save(
        build_checkpoint_payload(
            args,
            macro_actor,
            macro_critic,
            sac_actor,
            sac_critic,
            sac_entropy,
            macro_energy_critic=macro_energy_critic,
        ),
        os.path.join(ckpt_dir, "final.pt"),
    )
    replay_path = save_sac_replay(
        os.path.join(ckpt_dir, "micro_replay.pt"),
        sac_memory,
    )
    if replay_path:
        print(f"Saved lower SAC replay -> {replay_path}")
    if args.checkpoint_epsilon_relative > 0.0:
        selected = select_epsilon_checkpoint(
            ckpt_dir,
            eval_checkpoint_records,
            args.checkpoint_epsilon_relative,
        )
        if selected is not None:
            print(
                f"[EPSILON SELECT] episode={selected['episode']} "
                f"J={selected['J']:.6f} "
                f"macro_energy={selected['macro_energy_episode']:.2f}"
            )
    save_final_policy_trajectory(
        results_dirs, args, macro_actor, eval_micro_policy
    )
    if use_sca:
        train_sca_count = int(getattr(env, "_sca_solve_count", 0))
        train_sca_seconds = float(
            getattr(env, "_sca_total_solve_seconds", 0.0)
        )
        eval_sca_count = int(getattr(eval_env, "_sca_solve_count", 0))
        eval_sca_seconds = float(
            getattr(eval_env, "_sca_total_solve_seconds", 0.0)
        )
        print(
            f"[SCA] train solves={train_sca_count} "
            f"seconds={train_sca_seconds:.2f} "
            f"mean={1000.0 * train_sca_seconds / max(train_sca_count, 1):.2f}ms | "
            f"eval solves={eval_sca_count} "
            f"seconds={eval_sca_seconds:.2f} "
            f"mean={1000.0 * eval_sca_seconds / max(eval_sca_count, 1):.2f}ms"
        )
    print(f"Saved evaluation CSV -> {eval_csv}")
    print(f"Saved per-episode training CSV -> {train_csv}")
    print("Training complete.")


if __name__ == "__main__":
    main()
