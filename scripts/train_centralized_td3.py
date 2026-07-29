import argparse
import copy
import csv
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

PROJECT_ROOT = bootstrap()

from compare_common import EvaluationHistory
from compare_common import add_common_args
from compare_common import evaluate_policy
from compare_common import make_env_from_args
from compare_common import prepare_results_dirs
from compare_common import print_evaluation_summary
from compare_common import resolve_project_path
from compare_common import save_scenario_metadata
from compare_common import torch_load_checkpoint
from hrl_uav.agents.sac_agent import Actor as SAC_Actor
from hrl_uav.agents.sac_agent import Critic as SAC_Critic
from hrl_uav.agents.sac_agent import Entroy as SAC_Entropy
from hrl_uav.agents.sac_agent import Memory as SAC_Memory
import hrl_uav.agents.sac_agent as sac_agent_mod
from train_centralized_macro import SAC_MEMORY_CAPACITY
from train_centralized_macro import SAC_MEMORY_WARMUP
from train_centralized_macro import SAC_RECENT_RATIO
from train_centralized_macro import SAC_RECENT_WINDOW
from train_centralized_macro import SAC_GAMMA
from train_centralized_macro import configure_macro_objective
from train_centralized_macro import compose_smdp_boundary_transition
from train_centralized_macro import checkpoint_rank
from train_centralized_macro import configure_sac_learning_rates
from train_centralized_macro import load_checkpoint_energy_dual
from train_centralized_macro import load_sac_replay
from train_centralized_macro import MACRO_REWARD_DIAGNOSTIC_SOURCES
from train_centralized_macro import MICRO_REWARD_DIAGNOSTIC_SOURCES
from train_centralized_macro import project_unit_disk_np
from train_centralized_macro import initialize_reward_diagnostics
from train_centralized_macro import initialize_micro_reward_diagnostics
from train_centralized_macro import record_micro_reward_diagnostics
from train_centralized_macro import record_reward_diagnostics
from train_centralized_macro import power_constraint_target_w
from train_centralized_macro import sac_micro_policy
from train_centralized_macro import sample_training_start_positions
from train_centralized_macro import sample_training_scenario
from train_centralized_macro import save_eval_candidate
from train_centralized_macro import save_sac_replay
from train_centralized_macro import save_trajectory_with_verification
from train_centralized_macro import set_wrapper_load_scale
from train_centralized_macro import set_wrapper_cpu_ghz
from train_centralized_macro import store_sac_replay_transition
from train_centralized_macro import summarize_micro_reward_diagnostics
from train_centralized_macro import summarize_reward_diagnostics
from train_centralized_macro import update_sac
from train_centralized_macro import update_energy_dual


TRAIN_LOG_FIELDS = [
    "episode",
    "macro_reward",
    "macro_actor_reward",
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
    "deployment_transient_energy_episode_j",
    "exploration_noise",
    "td3_actor_loss",
    "td3_critic_loss",
    "td3_delay_critic_loss",
    "td3_energy_critic_loss",
    "sac_actor_loss",
    "sac_critic_loss",
    "sac_actor_grad_norm",
    "sac_critic_grad_norm",
    "sac_alpha",
    "sac_updates_episode",
    "sac_critic_updates_total",
    "sac_actor_updates_total",
    "td3_buffer_size",
    "td3_updates",
    "sac_trainable_micro_steps",
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
    "macro_feedback_mean",
    "macro_primal_cost_mean",
    "macro_energy_ratio_mean",
    "macro_global_micro_signal_mean",
    "macro_shaping_signal_mean",
    "macro_positioning_improvement_mean",
    "macro_movement_penalty_mean",
    "macro_propulsion_penalty_mean",
    "macro_action_norm_mean",
    "macro_power_violation_penalty_mean",
    "projected_uplink_time_s_mean",
    "projected_u2u_time_s_mean",
    "projected_u2u_rate_mbps_mean",
    "macro_reward_reconstruction_max_error",
    "episode_seconds",
]


def initialize_training_log(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        csv.DictWriter(fp, fieldnames=TRAIN_LOG_FIELDS).writeheader()


def append_training_log(path, row):
    with open(path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=TRAIN_LOG_FIELDS)
        writer.writerow({field: row.get(field, "") for field in TRAIN_LOG_FIELDS})


def project_unit_disk_torch(action, num_agents):
    original_shape = action.shape
    action_pairs = action.reshape(-1, int(num_agents), 2)
    norm = torch.linalg.norm(action_pairs, dim=-1, keepdim=True).clamp_min(1.0)
    return (action_pairs / norm).reshape(original_shape)


class MacroActor(nn.Module):
    """Centralized deterministic actor for all UAV dx/dy actions."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.num_agents = int(action_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(action_dim)),
            nn.Tanh(),
        )
        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-2].bias, -3e-3, 3e-3)

    def forward(self, state):
        return project_unit_disk_torch(self.net(state), self.num_agents)

    def choose_action(self, state):
        device = next(self.parameters()).device
        state_t = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        ).reshape(1, -1)
        with torch.no_grad():
            action = self(state_t)
        return action.squeeze(0).cpu().numpy().astype(np.float32)


class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, output_dim=1):
        super().__init__()
        input_dim = int(state_dim) + int(action_dim)
        self.q1_net = self._build_q(input_dim, hidden_dim, output_dim)
        self.q2_net = self._build_q(input_dim, hidden_dim, output_dim)

    @staticmethod
    def _build_q(input_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=-1)
        return self.q1_net(state_action), self.q2_net(state_action)

    def q1(self, state, action):
        return self.q1_net(torch.cat((state, action), dim=-1))


class MacroReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, energy_dim=1):
        self.capacity = int(capacity)
        self.state = np.zeros((self.capacity, int(state_dim)), dtype=np.float32)
        self.action = np.zeros((self.capacity, int(action_dim)), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.energy_dim = int(energy_dim)
        self.energy_ratio = np.zeros(
            (self.capacity, self.energy_dim),
            dtype=np.float32,
        )
        self.next_state = np.zeros(
            (self.capacity, int(state_dim)), dtype=np.float32
        )
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, energy_ratio=0.0):
        index = self.position
        self.state[index] = np.asarray(state, dtype=np.float32).reshape(-1)
        self.action[index] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.reward[index, 0] = float(reward)
        energy_values = np.asarray(energy_ratio, dtype=np.float32).reshape(-1)
        if energy_values.size == 1 and self.energy_dim > 1:
            energy_values = np.full(self.energy_dim, float(energy_values[0]))
        if energy_values.size != self.energy_dim:
            raise ValueError(
                f"Expected {self.energy_dim} energy ratios, got "
                f"{energy_values.size}."
            )
        self.energy_ratio[index] = energy_values
        self.next_state[index] = np.asarray(next_state, dtype=np.float32).reshape(-1)
        self.done[index, 0] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        indices = np.random.randint(0, self.size, size=int(batch_size))
        return tuple(
            torch.as_tensor(array[indices], dtype=torch.float32, device=device)
            for array in (
                self.state,
                self.action,
                self.reward,
                self.energy_ratio,
                self.next_state,
                self.done,
            )
        )

    def __len__(self):
        return self.size


class TD3Agent:
    def __init__(self, state_dim, action_dim, num_agents, args, device):
        self.device = device
        self.num_agents = int(num_agents)
        self.energy_dim = (
            self.num_agents
            if str(getattr(args, "power_constraint_scope", "system"))
            == "per_uav"
            else 1
        )
        self.actor = MacroActor(
            state_dim,
            action_dim,
            hidden_dim=args.td3_hidden_dim,
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_ema_decay = float(getattr(args, "td3_actor_ema_decay", 0.0))
        self.actor_ema = None
        if self.actor_ema_decay > 0.0:
            self.actor_ema = copy.deepcopy(self.actor).to(device)
            self.actor_ema.eval()
            for parameter in self.actor_ema.parameters():
                parameter.requires_grad_(False)
        self.critic = TwinCritic(
            state_dim,
            action_dim,
            hidden_dim=args.td3_hidden_dim,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.energy_critic = TwinCritic(
            state_dim,
            action_dim,
            hidden_dim=args.td3_hidden_dim,
            output_dim=self.energy_dim,
        ).to(device)
        self.energy_critic_target = copy.deepcopy(self.energy_critic).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(args.td3_actor_lr)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=float(args.td3_critic_lr)
        )
        self.energy_critic_optimizer = torch.optim.Adam(
            self.energy_critic.parameters(), lr=float(args.td3_critic_lr)
        )
        self.gamma = float(args.td3_gamma)
        self.tau = float(args.td3_tau)
        self.policy_noise = float(args.td3_policy_noise)
        self.noise_clip = float(args.td3_noise_clip)
        self.policy_delay = max(int(args.td3_policy_delay), 1)
        self.actor_warmup_updates = max(
            int(getattr(args, "td3_actor_warmup_updates", 0)),
            0,
        )
        self.objective_mode = str(args.objective_mode)
        self.reward_scale = float(args.td3_reward_scale)
        self.constraint_cost_scale = 10.0 * self.reward_scale
        self.update_count = 0

    def choose_action(self, state):
        return self.actor.choose_action(state)

    def update(
        self,
        replay_buffer,
        batch_size,
        energy_lagrange_multiplier=0.0,
    ):
        state, action, reward, energy_ratio, next_state, done = replay_buffer.sample(
            batch_size,
            self.device,
        )
        energy_cost = self.constraint_cost_scale * (energy_ratio - 1.0)
        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)
            next_action = project_unit_disk_torch(next_action, self.num_agents)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + (1.0 - done) * self.gamma * torch.minimum(
                target_q1,
                target_q2,
            )
            target_energy_q1, target_energy_q2 = self.energy_critic_target(
                next_state,
                next_action,
            )
            target_energy_q = energy_cost + (
                (1.0 - done)
                * self.gamma
                * torch.maximum(target_energy_q1, target_energy_q2)
            )

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2,
            target_q,
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        current_energy_q1, current_energy_q2 = self.energy_critic(state, action)
        energy_critic_loss = F.mse_loss(
            current_energy_q1,
            target_energy_q,
        ) + F.mse_loss(current_energy_q2, target_energy_q)
        self.energy_critic_optimizer.zero_grad()
        energy_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.energy_critic.parameters(), 10.0)
        self.energy_critic_optimizer.step()

        self.update_count += 1
        actor_loss_value = np.nan
        if self.update_count % self.policy_delay == 0:
            if self.update_count > self.actor_warmup_updates:
                actor_action = self.actor(state)
                delay_value = self.critic.q1(state, actor_action)
                energy_q1, energy_q2 = self.energy_critic(state, actor_action)
                conservative_energy_value = torch.maximum(energy_q1, energy_q2)
                multipliers = torch.as_tensor(
                    np.asarray(energy_lagrange_multiplier, dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(-1)
                if multipliers.numel() == 1 and self.energy_dim > 1:
                    multipliers = multipliers.repeat(self.energy_dim)
                if multipliers.numel() != self.energy_dim:
                    raise ValueError(
                        f"Expected {self.energy_dim} energy multipliers, got "
                        f"{multipliers.numel()}."
                    )
                constrained_energy_value = torch.sum(
                    conservative_energy_value * multipliers.reshape(1, -1),
                    dim=1,
                    keepdim=True,
                )
                actor_loss = (
                    -delay_value + constrained_energy_value
                ).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    10.0,
                )
                self.actor_optimizer.step()
                self._update_actor_ema()
                self._soft_update(self.actor_target, self.actor)
                actor_loss_value = float(actor_loss.item())
            self._soft_update(self.critic_target, self.critic)
            self._soft_update(self.energy_critic_target, self.energy_critic)

        return (
            actor_loss_value,
            float(critic_loss.item()),
            float(energy_critic_loss.item()),
        )

    def _soft_update(self, target, source):
        with torch.no_grad():
            for target_param, source_param in zip(
                target.parameters(), source.parameters()
            ):
                target_param.mul_(1.0 - self.tau)
                target_param.add_(source_param, alpha=self.tau)

    @property
    def evaluation_actor(self):
        return self.actor_ema if self.actor_ema is not None else self.actor

    def _update_actor_ema(self):
        if self.actor_ema is None:
            return
        with torch.no_grad():
            for ema_param, actor_param in zip(
                self.actor_ema.parameters(), self.actor.parameters()
            ):
                ema_param.mul_(self.actor_ema_decay)
                ema_param.add_(actor_param, alpha=1.0 - self.actor_ema_decay)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Centralized macro TD3 + SAC micro training"
    )
    add_common_args(parser, default_results_subdir="central_td3")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compute device",
    )
    parser.add_argument(
        "--micro_solver",
        choices=["sac", "local"],
        default="sac",
        help="Frozen SAC routing or local-only no-cooperation routing",
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
        help="Keep the loaded lower SAC fixed while training TD3",
    )
    parser.add_argument(
        "--frozen_micro_stochastic",
        action="store_true",
        help="Sample lower SAC actions while frozen; deterministic is the default",
    )
    parser.add_argument("--sac_actor_lr", type=float, default=3e-4)
    parser.add_argument("--sac_critic_lr", type=float, default=3e-4)
    parser.add_argument("--sac_alpha_lr", type=float, default=3e-4)
    parser.add_argument("--sac_initial_alpha", type=float, default=1.0)
    parser.add_argument(
        "--resume_micro_optimizer",
        type=int,
        choices=[0, 1],
        default=1,
        help="Restore lower SAC optimizer and entropy states",
    )
    parser.add_argument("--td3_hidden_dim", type=int, default=256)
    parser.add_argument("--td3_actor_lr", type=float, default=3e-4)
    parser.add_argument("--td3_critic_lr", type=float, default=1e-3)
    parser.add_argument("--td3_gamma", type=float, default=0.99)
    parser.add_argument("--td3_tau", type=float, default=0.005)
    parser.add_argument("--td3_batch_size", type=int, default=128)
    parser.add_argument("--td3_replay_capacity", type=int, default=50000)
    parser.add_argument("--td3_warmup_steps", type=int, default=300)
    parser.add_argument("--td3_updates_per_step", type=int, default=1)
    parser.add_argument("--td3_policy_noise", type=float, default=0.2)
    parser.add_argument("--td3_noise_clip", type=float, default=0.5)
    parser.add_argument("--td3_policy_delay", type=int, default=2)
    parser.add_argument("--td3_actor_warmup_updates", type=int, default=0)
    parser.add_argument(
        "--td3_actor_ema_decay",
        type=float,
        default=0.0,
        help=(
            "EMA decay for the deployment actor. Zero disables EMA; the raw "
            "actor remains responsible for training and exploration."
        ),
    )
    parser.add_argument("--td3_exploration_initial", type=float, default=0.30)
    parser.add_argument("--td3_exploration_final", type=float, default=0.05)
    parser.add_argument(
        "--td3_exploration_persistence",
        choices=["auto", "macro_step", "episode"],
        default="auto",
        help=(
            "Temporal persistence of action-space exploration. Episode-level "
            "persistence lets delta actions explore coherent relocations."
        ),
    )
    parser.add_argument(
        "--td3_warmup_action_radius",
        type=float,
        default=1.0,
        help=(
            "Maximum normalized radius used by warm-up actions. For waypoint "
            "control this should reflect a physical exploration distance."
        ),
    )
    parser.add_argument(
        "--td3_hover_exploration_prob",
        type=float,
        default=0.25,
        help="Probability of adding an exact joint-hover action to off-policy exploration",
    )
    parser.add_argument("--td3_exploration_decay_episodes", type=int, default=300)
    parser.add_argument(
        "--td3_reward_scale",
        type=float,
        default=0.01,
        help="Positive numerical scale applied only inside the TD3 critic target",
    )
    parser.add_argument(
        "--macro_power_violation_penalty_weight",
        type=float,
        default=0.0,
        help=(
            "One-sided penalty weight for macro-period average power above "
            "the configured budget; zero disables the soft penalty."
        ),
    )
    parser.add_argument(
        "--td3_dual_power_window",
        type=int,
        default=5,
        help="Number of recent episode powers averaged before dual ascent",
    )
    parser.add_argument(
        "--train_load_min",
        type=float,
        default=0.0,
        help="Minimum randomized episode load_scale; <=0 disables randomization",
    )
    parser.add_argument(
        "--train_load_max",
        type=float,
        default=0.0,
        help="Maximum randomized episode load_scale; <=0 disables randomization",
    )
    parser.add_argument(
        "--train_start_jitter_max_m",
        type=float,
        default=0.0,
        help="Maximum random UAV start offset from each service anchor",
    )
    parser.add_argument(
        "--train_start_anchor_probability",
        type=float,
        default=0.2,
        help="Per-UAV probability of starting exactly at its service anchor",
    )
    parser.add_argument("--macro_shaping_initial", type=float, default=0.0)
    parser.add_argument("--macro_shaping_final", type=float, default=0.0)
    parser.add_argument("--macro_shaping_decay_episodes", type=int, default=0)
    parser.add_argument("--macro_move_penalty_weight", type=float, default=0.0)
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
    args = parser.parse_args()
    if "--macro_action_mode" not in sys.argv:
        args.macro_action_mode = "delta"
    if "--macro_comm_weight" not in sys.argv:
        args.macro_comm_weight = 0.0
    if args.freeze_micro and args.micro_solver != "sac":
        parser.error("--freeze_micro is only valid with --micro_solver sac")
    if args.micro_ckpt and args.micro_solver != "sac":
        parser.error("--micro_ckpt is only valid with --micro_solver sac")
    if args.micro_replay and args.micro_solver != "sac":
        parser.error("--micro_replay is only valid with --micro_solver sac")
    if args.freeze_micro and not args.micro_ckpt:
        parser.error("--freeze_micro requires --micro_ckpt")
    if args.td3_reward_scale <= 0.0:
        parser.error("--td3_reward_scale must be positive")
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
    if args.td3_dual_power_window <= 0:
        parser.error("--td3_dual_power_window must be positive")
    if not 0.0 <= args.td3_actor_ema_decay < 1.0:
        parser.error("--td3_actor_ema_decay must be in [0, 1)")
    if not 0.0 <= args.td3_hover_exploration_prob <= 1.0:
        parser.error("--td3_hover_exploration_prob must be in [0, 1]")
    if not 0.0 < args.td3_warmup_action_radius <= 1.0:
        parser.error("--td3_warmup_action_radius must be in (0, 1]")
    if args.objective_mode == "constrained_delay" and args.power_budget_w <= 0.0:
        parser.error("constrained_delay requires --power_budget_w > 0")
    if args.energy_dual_lr < 0.0 or args.energy_dual_max <= 0.0:
        parser.error("Energy dual parameters must be non-negative with max > 0")
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
    if args.train_start_jitter_max_m < 0.0:
        parser.error("--train_start_jitter_max_m must be non-negative")
    if not 0.0 <= args.train_start_anchor_probability <= 1.0:
        parser.error("--train_start_anchor_probability must be in [0, 1]")
    return args


def exploration_noise_for_episode(args, episode):
    initial = float(args.td3_exploration_initial)
    final = float(args.td3_exploration_final)
    decay = max(int(args.td3_exploration_decay_episodes), 1)
    progress = min(max(float(episode) / float(decay), 0.0), 1.0)
    return initial + progress * (final - initial)


def update_smoothed_energy_dual(env, args, power_history, average_power_w):
    """Update the multiplier from a trailing mean without changing its rule."""
    power_history.append(float(average_power_w))
    smoothed_power_w = float(np.mean(power_history))
    before, after, violation = update_energy_dual(
        env,
        args,
        smoothed_power_w,
    )
    return before, after, violation, smoothed_power_w


def update_smoothed_per_uav_energy_duals(
    env,
    args,
    power_history,
    average_power_per_uav_w,
):
    """Episode-level projected dual ascent for independent UAV batteries."""
    powers = np.asarray(average_power_per_uav_w, dtype=np.float64)
    if powers.shape != (env.N,):
        raise ValueError(
            f"Expected per-UAV powers with shape ({env.N},), got {powers.shape}."
        )
    power_history.append(powers.copy())
    smoothed = np.mean(np.stack(tuple(power_history), axis=0), axis=0)
    before = np.asarray(
        env.energy_lagrange_multipliers,
        dtype=np.float64,
    ).copy()
    targets = np.asarray(env.uav_power_budgets_w, dtype=np.float64)
    violations = (smoothed - targets) / np.maximum(targets, 1e-8)
    if str(getattr(args, "objective_mode", "weighted")) == "constrained_delay":
        after = np.clip(
            before + float(args.energy_dual_lr) * violations,
            0.0,
            float(args.energy_dual_max),
        )
    else:
        after = before
        violations = np.zeros_like(before)
    env.set_energy_lagrange_multipliers(after)
    return before, after, violations, smoothed


def copy_energy_constraint_state(source_env, target_env, args):
    if str(getattr(args, "power_constraint_scope", "system")) == "per_uav":
        target_env.set_energy_lagrange_multipliers(
            source_env.energy_lagrange_multipliers
        )
    else:
        target_env.energy_lagrange_multiplier = float(
            source_env.energy_lagrange_multiplier
        )
        target_env._sync_cpu_energy_prices()


def sample_random_macro_action(
    num_agents,
    hover_probability=0.0,
    max_radius=1.0,
):
    if np.random.random() < float(hover_probability):
        return np.zeros((int(num_agents), 2), dtype=np.float32)
    theta = np.random.uniform(0.0, 2.0 * np.pi, size=int(num_agents))
    radius = float(max_radius) * np.sqrt(
        np.random.uniform(0.0, 1.0, size=int(num_agents))
    )
    action = np.stack(
        (radius * np.cos(theta), radius * np.sin(theta)),
        axis=1,
    )
    return action.astype(np.float32)


def noisy_macro_action(
    agent,
    state,
    num_agents,
    episode_noise=None,
    force_hover=False,
):
    if force_hover:
        return np.zeros((int(num_agents), 2), dtype=np.float32)
    action = agent.choose_action(state).reshape(int(num_agents), 2)
    if episode_noise is not None:
        action = action + np.asarray(
            episode_noise,
            dtype=np.float32,
        ).reshape(action.shape)
    action = np.clip(action, -1.0, 1.0)
    return project_unit_disk_np(action)


def sample_episode_exploration(num_agents, noise_std, hover_probability=0.0):
    """Sample one waypoint perturbation and hold it for the whole episode."""
    if np.random.random() < float(hover_probability):
        return np.zeros((int(num_agents), 2), dtype=np.float32), True
    noise = np.random.normal(
        0.0,
        max(float(noise_std), 0.0),
        size=(int(num_agents), 2),
    ).astype(np.float32)
    return noise, False


def td3_exploration_scope(macro_action_mode, persistence="auto"):
    """Match exploration persistence to the semantics of the macro action."""
    persistence = str(persistence).strip().lower()
    if persistence in ("macro_step", "episode"):
        return persistence
    if persistence != "auto":
        raise ValueError(f"Unsupported exploration persistence: {persistence}")
    mode = str(macro_action_mode).strip().lower()
    if mode in ("waypoint", "centroid_waypoint"):
        return "episode"
    if mode == "delta":
        return "macro_step"
    raise ValueError(f"Unsupported macro action mode: {macro_action_mode}")


def joint_sac_training_enabled(use_sac, freeze_micro, macro_warmup):
    """Fine-tune SAC only after the macro replay warm-up is complete."""
    return bool(use_sac and not freeze_micro and not macro_warmup)


def macro_power_violation_penalty(energy_ratio, weight):
    violation = np.maximum(
        np.asarray(energy_ratio, dtype=np.float64) - 1.0,
        0.0,
    )
    return float(weight) * float(np.mean(violation))


def sample_training_exploration(
    num_agents,
    noise_std,
    hover_probability,
    macro_action_mode,
    exploration_persistence="auto",
    episode_noise=None,
    episode_hover=False,
):
    """Return exploration noise with the persistence required by the action mode."""
    if (
        td3_exploration_scope(
            macro_action_mode,
            exploration_persistence,
        )
        == "episode"
    ):
        if episode_noise is None:
            raise ValueError("Waypoint exploration requires an episode perturbation")
        return np.asarray(episode_noise, dtype=np.float32).copy(), bool(episode_hover)
    return sample_episode_exploration(
        num_agents,
        noise_std,
        hover_probability=hover_probability,
    )


def sample_training_warmup_action(
    num_agents,
    hover_probability,
    max_radius,
    macro_action_mode,
    exploration_persistence="auto",
    episode_action=None,
):
    """Use a fixed waypoint target or an independent delta at each macro step."""
    if (
        td3_exploration_scope(
            macro_action_mode,
            exploration_persistence,
        )
        == "episode"
    ):
        if episode_action is None:
            raise ValueError("Waypoint warm-up requires an episode action")
        return np.asarray(episode_action, dtype=np.float32).copy()
    return sample_random_macro_action(
        num_agents,
        hover_probability=hover_probability,
        max_radius=max_radius,
    )


def td3_macro_policy(actor):
    def _policy(env, _local_obs_batch, _eval_mode):
        _, global_state = env.get_macro_state()
        action = actor.choose_action(global_state)
        return project_unit_disk_np(action.reshape(env.N, 2))

    return _policy


def load_micro_checkpoint(
    path,
    sac_actor,
    sac_critic,
    sac_entropy=None,
    load_optimizers=True,
):
    resolved = resolve_project_path(path)
    if not resolved or not os.path.isfile(resolved):
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
        sac_entropy.log_alpha.data.copy_(
            alpha_state.reshape_as(sac_entropy.log_alpha)
        )
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


def build_checkpoint_payload(args, agent, sac_actor, sac_critic, sac_entropy):
    deployment_actor = agent.evaluation_actor
    using_ema = agent.actor_ema is not None
    payload = {
        "macro_solver": "td3",
        "constrained_critic_structure": "separate_delay_energy",
        "macro_actor": deployment_actor.state_dict(),
        "macro_actor_raw": agent.actor.state_dict(),
        "macro_actor_target": agent.actor_target.state_dict(),
        "macro_delay_critic": agent.critic.state_dict(),
        "macro_delay_critic_target": agent.critic_target.state_dict(),
        "macro_energy_critic": agent.energy_critic.state_dict(),
        "macro_energy_critic_target": agent.energy_critic_target.state_dict(),
        "macro_critic": agent.critic.state_dict(),
        "macro_critic_target": agent.critic_target.state_dict(),
        "macro_actor_optimizer": agent.actor_optimizer.state_dict(),
        "macro_delay_critic_optimizer": agent.critic_optimizer.state_dict(),
        "macro_energy_critic_optimizer": (
            agent.energy_critic_optimizer.state_dict()
        ),
        "macro_critic_optimizer": agent.critic_optimizer.state_dict(),
        "macro_update_count": int(agent.update_count),
        "macro_actor_policy": "ema" if using_ema else "raw",
        "td3_actor_ema_decay": float(agent.actor_ema_decay),
        "micro_solver": str(args.micro_solver),
        "args": vars(args),
    }
    if using_ema:
        payload["macro_actor_ema"] = agent.actor_ema.state_dict()
    if str(args.micro_solver) == "sac":
        payload.update(
            {
                "sac_actor": sac_actor.action_net.state_dict(),
                "sac_critic": sac_critic.critic_v.state_dict(),
                "sac_target_critic": (
                    sac_critic.target_critic_v.state_dict()
                ),
                "sac_log_alpha": sac_entropy.log_alpha.detach().cpu(),
                "sac_actor_optimizer": sac_actor.optimizer.state_dict(),
                "sac_critic_optimizer": sac_critic.optimizer.state_dict(),
                "sac_entropy_optimizer": sac_entropy.optimizer.state_dict(),
            }
        )
    return payload


def save_final_policy_trajectory(results_dirs, args, actor, micro_policy):
    if int(getattr(args, "trajectory_plot_interval", 0)) <= 0:
        return
    traj_env = make_env_from_args(args, record_trajectory=True)
    if str(getattr(args, "power_constraint_scope", "system")) == "per_uav":
        saved_duals = getattr(
            args,
            "energy_lagrange_multipliers_current",
            None,
        )
        if saved_duals is not None:
            traj_env.set_energy_lagrange_multipliers(saved_duals)
    traj_args = argparse.Namespace(**vars(args))
    traj_args.eval_episodes = 1
    evaluate_policy(
        traj_env,
        traj_args,
        td3_macro_policy(actor),
        micro_policy,
    )
    final_path = os.path.join(results_dirs["trajectory_dir"], "trajectory_final.npz")
    save_trajectory_with_verification(traj_env, final_path)


def render_td3_checkpoint_trajectory(
    checkpoint_path,
    output_path,
    device=None,
):
    checkpoint = torch_load_checkpoint(resolve_project_path(checkpoint_path))
    if "macro_actor" not in checkpoint or "args" not in checkpoint:
        raise KeyError("Checkpoint must contain macro_actor and args")

    checkpoint_args = argparse.Namespace(**checkpoint["args"])
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)
    sac_agent_mod.device = device

    traj_env = make_env_from_args(checkpoint_args, record_trajectory=True)
    configure_macro_objective(
        traj_env,
        checkpoint_args,
        checkpoint_args.macro_shaping_final,
    )
    if (
        str(getattr(checkpoint_args, "power_constraint_scope", "system"))
        == "per_uav"
    ):
        saved_duals = getattr(
            checkpoint_args,
            "energy_lagrange_multipliers_current",
            None,
        )
        if saved_duals is not None:
            traj_env.set_energy_lagrange_multipliers(saved_duals)

    _, global_state = traj_env.reset(seed=checkpoint_args.seed)[0]
    actor = MacroActor(
        int(np.asarray(global_state).size),
        int(checkpoint_args.num_agents) * 2,
        hidden_dim=checkpoint_args.td3_hidden_dim,
    ).to(device)
    actor.load_state_dict(checkpoint["macro_actor"])
    actor.eval()

    if str(checkpoint.get("micro_solver", "")) == "sac":
        micro_actor = SAC_Actor(
            int(traj_env.sac_obs_dim),
            int(traj_env.micro_action_space.shape[0]),
        )
        micro_actor.action_net.load_state_dict(checkpoint["sac_actor"])
        micro_actor.action_net.eval()
        micro_policy = sac_micro_policy(micro_actor)
    else:
        micro_policy = lambda policy_env, _eval_mode: np.zeros(
            policy_env.micro_action_space.shape[0],
            dtype=np.float32,
        )

    eval_args = argparse.Namespace(**vars(checkpoint_args))
    eval_args.eval_episodes = 1
    metrics = evaluate_policy(
        traj_env,
        eval_args,
        td3_macro_policy(actor),
        micro_policy,
    )
    save_trajectory_with_verification(traj_env, output_path)
    return metrics


def main():
    args = parse_args()
    use_sac = args.micro_solver == "sac"
    use_local = args.micro_solver == "local"
    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)
    sac_agent_mod.device = device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"[INFO] Training on {str(device).upper()}")
    print(
        "[INFO] Macro solver: centralized TD3, joint 6D "
        f"{args.macro_action_mode} action"
    )
    print(f"[INFO] Micro solver: {args.micro_solver.upper()}")
    print(
        "[INFO] TD3 exploration scope: "
        f"{td3_exploration_scope(args.macro_action_mode, args.td3_exploration_persistence)}"
    )
    print("[INFO] PPO trainer remains available as the macro-layer comparator")
    print(f"[INFO] macro_comm_weight={args.macro_comm_weight}")
    print(f"[INFO] Macro reward mode: {args.macro_reward_mode}")
    print(
        f"[INFO] Objective: {args.objective_mode} "
        f"power_budget={args.power_budget_w:.2f}W"
    )
    print(f"[INFO] training scenario mode: {args.train_scenario_mode}")

    env = make_env_from_args(args, record_trajectory=False)
    eval_env = make_env_from_args(args, record_trajectory=False)
    env.force_local_routing = bool(use_local)
    eval_env.force_local_routing = bool(use_local)
    configure_macro_objective(env, args, args.macro_shaping_initial)
    configure_macro_objective(eval_env, args, args.macro_shaping_final)
    if int(args.micro_steps) != int(env.env.micro_steps_per_macro):
        raise ValueError(
            f"micro_steps={args.micro_steps} does not match physical macro slot "
            f"length {env.env.micro_steps_per_macro}."
        )
    results_dirs = prepare_results_dirs(args.results_subdir)
    save_scenario_metadata(
        results_dirs,
        args,
        "CentralizedTD3SAC" if use_sac else "CentralizedTD3NoCoop",
    )

    sac_state_dim = int(env.sac_obs_dim)
    sac_action_dim = int(env.micro_action_space.shape[0])
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
    sac_memory = None
    if use_sac and not args.freeze_micro:
        sac_memory = SAC_Memory(
            SAC_MEMORY_CAPACITY,
            2 * sac_state_dim
            + sac_action_dim
            + 1
            + int(args.objective_mode == "constrained_delay")
            + 1,
            recent_window=SAC_RECENT_WINDOW,
            recent_ratio=SAC_RECENT_RATIO,
        )
    if use_sac and args.micro_ckpt:
        loaded_path = load_micro_checkpoint(
            args.micro_ckpt,
            sac_actor,
            sac_critic,
            sac_entropy,
            load_optimizers=(
                not args.freeze_micro and bool(args.resume_micro_optimizer)
            ),
        )
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
                env.energy_lagrange_multiplier = inherited_dual
                eval_env.energy_lagrange_multiplier = inherited_dual
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
    if use_sac and not args.freeze_micro:
        configure_sac_learning_rates(
            sac_actor,
            sac_critic,
            sac_entropy,
            args,
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

    _, global_state0 = env.reset(seed=args.seed)[0]
    global_state_dim = int(np.asarray(global_state0).size)
    macro_action_dim = int(args.num_agents) * 2
    agent = TD3Agent(
        global_state_dim,
        macro_action_dim,
        args.num_agents,
        args,
        device,
    )
    replay_buffer = MacroReplayBuffer(
        global_state_dim,
        macro_action_dim,
        args.td3_replay_capacity,
        energy_dim=(env.N if args.power_constraint_scope == "per_uav" else 1),
    )
    eval_micro_policy = (
        sac_micro_policy(sac_actor)
        if use_sac
        else lambda policy_env, _eval_mode: np.zeros(
            policy_env.micro_action_space.shape[0],
            dtype=np.float32,
        )
    )

    eval_history = EvaluationHistory()
    best_eval_rank = None
    eval_checkpoint_records = []
    dual_power_history = deque(maxlen=int(args.td3_dual_power_window))
    total_macro_steps = 0
    sac_trainable_micro_steps = 0
    sac_critic_update_count = 0
    sac_actor_update_count = 0
    ckpt_dir = results_dirs["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    train_csv = os.path.join(results_dirs["results_dir"], "train_td3.csv")
    eval_csv = os.path.join(results_dirs["results_dir"], "eval_central_td3.csv")
    initialize_training_log(train_csv)
    eval_history.save_csv(eval_csv)

    for episode in range(args.episodes):
        episode_started = time.perf_counter()
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
        env.reset(seed=args.seed + episode)
        _, global_state = env.get_macro_state()

        ep_macro_reward = 0.0
        ep_macro_actor_reward = 0.0
        ep_micro_reward = 0.0
        ep_j = []
        ep_delay_objective = []
        ep_service_energy = []
        ep_macro_energy = []
        ep_service_battery_energy_per_uav = np.zeros(
            env.N,
            dtype=np.float64,
        )
        ep_macro_energy_per_uav = np.zeros(env.N, dtype=np.float64)
        ep_deployment_transient_energy = []
        ep_comm = []
        ep_u2g = []
        ep_move = []
        ep_all_hover = []
        ep_uav_hover = []
        ep_actor_losses = []
        ep_critic_losses = []
        ep_energy_critic_losses = []
        ep_sac_actor_losses = []
        ep_sac_critic_losses = []
        ep_sac_actor_grad_norms = []
        ep_sac_critic_grad_norms = []
        ep_macro_feedback = []
        ep_macro_power_violation_penalties = []
        ep_macro_reconstruction_error = []
        ep_micro_diagnostics = initialize_micro_reward_diagnostics()
        ep_macro_diagnostics = initialize_reward_diagnostics(
            MACRO_REWARD_DIAGNOSTIC_SOURCES
        )
        ep_projected_uplink_time = []
        ep_projected_u2u_time = []
        ep_projected_u2u_rate = []
        done = False
        energy_dual_before = (
            env.energy_lagrange_multipliers.copy()
            if args.power_constraint_scope == "per_uav"
            else float(env.energy_lagrange_multiplier)
        )
        exploration_noise = exploration_noise_for_episode(args, episode)
        warmup_episode = total_macro_steps < int(args.td3_warmup_steps)
        exploration_scope = td3_exploration_scope(
            args.macro_action_mode,
            args.td3_exploration_persistence,
        )
        if warmup_episode:
            warmup_action = None
            episode_noise = None
            episode_hover = False
        else:
            warmup_action = None
            if exploration_scope == "episode":
                episode_noise, episode_hover = sample_episode_exploration(
                    env.N,
                    0.0 if args.macro_deterministic else exploration_noise,
                    hover_probability=(
                        0.0
                        if args.macro_deterministic
                        else args.td3_hover_exploration_prob
                    ),
                )
            else:
                episode_noise = None
                episode_hover = False
        pending_sac_transition = None
        ep_service_resource_decisions = 0
        ep_smdp_boundary_transitions = 0
        ep_smdp_boundary_audit = None

        for macro_idx in range(args.macro_steps):
            if warmup_episode:
                action = sample_training_warmup_action(
                    env.N,
                    hover_probability=args.td3_hover_exploration_prob,
                    max_radius=args.td3_warmup_action_radius,
                    macro_action_mode=args.macro_action_mode,
                    exploration_persistence="macro_step",
                    episode_action=warmup_action,
                )
            else:
                step_noise, step_hover = sample_training_exploration(
                    env.N,
                    0.0 if args.macro_deterministic else exploration_noise,
                    hover_probability=(
                        0.0
                        if args.macro_deterministic
                        else args.td3_hover_exploration_prob
                    ),
                    macro_action_mode=args.macro_action_mode,
                    exploration_persistence=(
                        args.td3_exploration_persistence
                    ),
                    episode_noise=episode_noise,
                    episode_hover=episode_hover,
                )
                action = noisy_macro_action(
                    agent,
                    global_state,
                    env.N,
                    episode_noise=step_noise,
                    force_hover=step_hover,
                )
            action_flat = action.reshape(-1).astype(np.float32)
            hover_mask = np.linalg.norm(action, axis=1) <= 1e-8
            ep_all_hover.append(float(np.all(hover_mask)))
            ep_uav_hover.append(float(np.mean(hover_mask)))

            _, _, macro_info = env.macro_step(action)
            if isinstance(macro_info, dict):
                ep_comm.append(
                    float(macro_info.get("projected_macro_comm_cost", 0.0))
                )
                ep_u2g.append(
                    float(macro_info.get("projected_mean_u2g_distance_m", 0.0))
                )
                ep_move.append(
                    float(macro_info.get("macro_move_distance_mean", 0.0))
                )
                ep_macro_energy.append(
                    float(macro_info.get("E_macro_total", 0.0))
                )
                ep_macro_energy_per_uav += np.asarray(
                    macro_info.get("E_macro_per_uav", np.zeros(env.N)),
                    dtype=np.float64,
                )
                ep_deployment_transient_energy.append(
                    float(macro_info.get("E_deployment_transient", 0.0))
                )
                ep_projected_uplink_time.append(
                    float(
                        macro_info.get(
                            "projected_comm_uplink_time_s",
                            0.0,
                        )
                    )
                )
                ep_projected_u2u_time.append(
                    float(
                        macro_info.get(
                            "projected_comm_u2u_time_s",
                            0.0,
                        )
                    )
                )
                ep_projected_u2u_rate.append(
                    float(
                        macro_info.get(
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
                for micro_info in frame["infos"]:
                    record_micro_reward_diagnostics(
                        ep_micro_diagnostics,
                        micro_info,
                    )
                    ep_j.append(float(micro_info.get("J_step", 0.0)))
                    ep_delay_objective.append(
                        float(micro_info.get("delay_objective_step", 0.0))
                    )
                    ep_service_energy.append(
                        float(micro_info.get("service_energy_step_j", 0.0))
                    )
                    ep_service_battery_energy_per_uav += np.asarray(
                        micro_info.get(
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
                frozen_deterministic = (
                    args.freeze_micro and not args.frozen_micro_stochastic
                )
                micro_action = (
                    sac_actor.choose_action(
                        sac_obs,
                        deterministic=frozen_deterministic,
                    )
                    if use_sac
                    else np.zeros(sac_action_dim, dtype=np.float32)
                )
                frame = env.step_resource_frame(micro_action)
                ep_micro_reward += float(frame["reward_sum"])
                for micro_info in frame["infos"]:
                    record_micro_reward_diagnostics(
                        ep_micro_diagnostics,
                        micro_info,
                    )
                    ep_j.append(float(micro_info.get("J_step", 0.0)))
                    ep_delay_objective.append(
                        float(micro_info.get("delay_objective_step", 0.0))
                    )
                    ep_service_energy.append(
                        float(micro_info.get("service_energy_step_j", 0.0))
                    )
                    ep_service_battery_energy_per_uav += np.asarray(
                        micro_info.get(
                            "uav_battery_energy_per_uav",
                            np.zeros(env.N),
                        ),
                        dtype=np.float64,
                    )
                done = done or bool(frame["done"])

                should_train_sac = joint_sac_training_enabled(
                    use_sac,
                    args.freeze_micro,
                    warmup_episode,
                )
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
            ep_macro_feedback.append(
                float(macro_reward_info.get("global_micro_signal", 0.0))
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
            _, next_global_state = env.get_macro_state()
            ep_macro_reward += macro_reward
            # Episode boundaries are rollout truncations of the continuing
            # queueing process, not physical terminal states.
            terminal = bool(done)
            if args.power_constraint_scope == "per_uav":
                macro_energy_ratio = np.asarray(
                    macro_reward_info.get(
                        "energy_constraint_ratio_per_uav_mean",
                        np.zeros(env.N),
                    ),
                    dtype=np.float32,
                )
            else:
                macro_energy_ratio = float(
                    macro_reward_info.get("energy_constraint_ratio_mean", 0.0)
                )
            if args.objective_mode == "constrained_delay":
                power_violation_penalty = macro_power_violation_penalty(
                    macro_energy_ratio,
                    args.macro_power_violation_penalty_weight,
                )
                ep_macro_power_violation_penalties.append(
                    power_violation_penalty
                )
                replay_primal_cost = float(
                    macro_reward_info.get("primal_cost_mean", 0.0)
                ) + power_violation_penalty
                replay_macro_reward = (
                    -env.micro_reward_scale
                    * replay_primal_cost
                    * float(args.td3_reward_scale)
                )
            else:
                replay_macro_reward = (
                    macro_reward * float(args.td3_reward_scale)
                )
            ep_macro_actor_reward += float(replay_macro_reward)
            replay_buffer.add(
                global_state,
                action_flat,
                replay_macro_reward,
                next_global_state,
                terminal,
                energy_ratio=macro_energy_ratio,
            )
            total_macro_steps += 1

            if (
                total_macro_steps >= int(args.td3_warmup_steps)
                and len(replay_buffer) >= int(args.td3_batch_size)
            ):
                for _ in range(max(int(args.td3_updates_per_step), 1)):
                    actor_loss, critic_loss, energy_critic_loss = agent.update(
                        replay_buffer,
                        args.td3_batch_size,
                        energy_lagrange_multiplier=(
                            env.energy_lagrange_multipliers
                            if args.power_constraint_scope == "per_uav"
                            else env.energy_lagrange_multiplier
                        ),
                    )
                    if np.isfinite(actor_loss):
                        ep_actor_losses.append(actor_loss)
                    ep_critic_losses.append(critic_loss)
                    ep_energy_critic_losses.append(energy_critic_loss)

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
        if warmup_episode:
            if args.power_constraint_scope == "per_uav":
                energy_dual_after = env.energy_lagrange_multipliers.copy()
                power_violation_ratio = np.zeros(env.N, dtype=np.float64)
                dual_update_power_w = (
                    average_battery_power_per_uav_w.copy()
                )
            else:
                energy_dual_after = float(env.energy_lagrange_multiplier)
                power_violation_ratio = 0.0
                dual_update_power_w = float(average_power_w)
        elif args.power_constraint_scope == "per_uav":
            (
                _,
                energy_dual_after,
                power_violation_ratio,
                dual_update_power_w,
            ) = update_smoothed_per_uav_energy_duals(
                env,
                args,
                dual_power_history,
                average_battery_power_per_uav_w,
            )
        else:
            (
                _,
                energy_dual_after,
                power_violation_ratio,
                dual_update_power_w,
            ) = update_smoothed_energy_dual(
                env,
                args,
                dual_power_history,
                average_power_w,
            )
        args.energy_lagrange_multiplier_current = float(
            np.mean(env.energy_lagrange_multipliers)
            if args.power_constraint_scope == "per_uav"
            else env.energy_lagrange_multiplier
        )
        if args.power_constraint_scope == "per_uav":
            args.energy_lagrange_multipliers_current = (
                env.energy_lagrange_multipliers.tolist()
            )
        mean_j = float(np.mean(ep_j)) if ep_j else 0.0
        mean_actor_loss = (
            float(np.mean(ep_actor_losses)) if ep_actor_losses else np.nan
        )
        mean_critic_loss = (
            float(np.mean(ep_critic_losses)) if ep_critic_losses else np.nan
        )
        mean_energy_critic_loss = (
            float(np.mean(ep_energy_critic_losses))
            if ep_energy_critic_losses
            else np.nan
        )
        if args.power_constraint_scope == "per_uav":
            energy_dual_before_log = float(np.mean(energy_dual_before))
            energy_dual_after_log = float(np.mean(energy_dual_after))
            power_violation_ratio_log = float(np.max(power_violation_ratio))
            dual_update_power_w_log = float(np.max(dual_update_power_w))
            constrained_power_display = np.array2string(
                average_battery_power_per_uav_w,
                precision=1,
                separator=",",
            )
            dual_display = np.array2string(
                np.asarray(energy_dual_after),
                precision=4,
                separator=",",
            )
        else:
            energy_dual_before_log = float(energy_dual_before)
            energy_dual_after_log = float(energy_dual_after)
            power_violation_ratio_log = float(power_violation_ratio)
            dual_update_power_w_log = float(dual_update_power_w)
            constrained_power_display = f"{average_power_w:.1f}"
            dual_display = f"{energy_dual_after_log:.4f}"
        print(
            f"Episode {episode + 1} | Macro R {ep_macro_reward:.2f} | "
            f"Micro R {ep_micro_reward:.2f} | J {mean_j:.4f} | "
            f"U2G {np.mean(ep_u2g) if ep_u2g else 0.0:.1f} | "
            f"Move {np.mean(ep_move) if ep_move else 0.0:.1f} | "
            f"Start {np.mean(start_offsets):.1f}m | "
            f"CPU {episode_cpu_ghz:.2f}GHz | "
            f"Psys {average_power_w:.1f}W | Pcon {constrained_power_display}W | "
            f"PemaMax {dual_update_power_w_log:.1f}W | "
            f"Dual {dual_display} | "
            f"Noise {exploration_noise:.3f} | Buffer {len(replay_buffer)}"
        )
        append_training_log(
            train_csv,
            {
                "episode": episode + 1,
                "macro_reward": ep_macro_reward,
                "macro_actor_reward": ep_macro_actor_reward,
                "micro_reward": ep_micro_reward,
                "J_train": mean_j,
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
                    np.atleast_1d(energy_dual_after)
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
                "deployment_transient_energy_episode_j": float(
                    np.sum(ep_deployment_transient_energy)
                ),
                "exploration_noise": exploration_noise,
                "td3_actor_loss": mean_actor_loss,
                "td3_critic_loss": mean_critic_loss,
                "td3_delay_critic_loss": mean_critic_loss,
                "td3_energy_critic_loss": mean_energy_critic_loss,
                "sac_actor_loss": (
                    float(np.mean(ep_sac_actor_losses))
                    if ep_sac_actor_losses
                    else np.nan
                ),
                "sac_critic_loss": (
                    float(np.mean(ep_sac_critic_losses))
                    if ep_sac_critic_losses
                    else np.nan
                ),
                "sac_actor_grad_norm": (
                    float(np.mean(ep_sac_actor_grad_norms))
                    if ep_sac_actor_grad_norms
                    else np.nan
                ),
                "sac_critic_grad_norm": (
                    float(np.mean(ep_sac_critic_grad_norms))
                    if ep_sac_critic_grad_norms
                    else np.nan
                ),
                "sac_alpha": (
                    float(sac_entropy.alpha.detach().cpu().item())
                    if sac_entropy is not None
                    else np.nan
                ),
                "sac_updates_episode": len(ep_sac_actor_losses),
                "sac_critic_updates_total": sac_critic_update_count,
                "sac_actor_updates_total": sac_actor_update_count,
                "td3_buffer_size": len(replay_buffer),
                "td3_updates": agent.update_count,
                "sac_trainable_micro_steps": sac_trainable_micro_steps,
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
                "macro_feedback_mean": float(np.mean(ep_macro_feedback))
                if ep_macro_feedback
                else 0.0,
                "macro_power_violation_penalty_mean": float(
                    np.mean(ep_macro_power_violation_penalties)
                )
                if ep_macro_power_violation_penalties
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
                "macro_reward_reconstruction_max_error": float(
                    np.max(ep_macro_reconstruction_error)
                )
                if ep_macro_reconstruction_error
                else 0.0,
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
            copy_energy_constraint_state(env, eval_env, args)
            metrics = evaluate_policy(
                eval_env,
                args,
                td3_macro_policy(agent.evaluation_actor),
                eval_micro_policy,
            )
            eval_history.append(episode + 1, metrics)
            eval_history.save_csv(eval_csv)
            print_evaluation_summary(episode + 1, metrics)
            checkpoint_payload = build_checkpoint_payload(
                args,
                agent,
                sac_actor,
                sac_critic,
                sac_entropy,
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
                    os.path.join(ckpt_dir, "best.pt"),
                )

    if not eval_history.steps or eval_history.steps[-1] != args.episodes:
        copy_energy_constraint_state(env, eval_env, args)
        metrics = evaluate_policy(
            eval_env,
            args,
            td3_macro_policy(agent.evaluation_actor),
            eval_micro_policy,
        )
        eval_history.append(args.episodes, metrics)
        print_evaluation_summary(args.episodes, metrics)
        checkpoint_payload = build_checkpoint_payload(
            args,
            agent,
            sac_actor,
            sac_critic,
            sac_entropy,
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
                os.path.join(ckpt_dir, "best.pt"),
            )

    eval_history.save_csv(eval_csv)
    eval_history.save_detail_npz(
        os.path.join(results_dirs["results_dir"], "eval_central_td3_detail.npz")
    )
    torch.save(
        build_checkpoint_payload(args, agent, sac_actor, sac_critic, sac_entropy),
        os.path.join(ckpt_dir, "final.pt"),
    )
    replay_path = save_sac_replay(
        os.path.join(ckpt_dir, "micro_replay.pt"),
        sac_memory,
    )
    if replay_path:
        print(f"Saved lower SAC replay -> {replay_path}")
    save_final_policy_trajectory(
        results_dirs,
        args,
        agent.evaluation_actor,
        eval_micro_policy,
    )
    best_checkpoint_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.isfile(best_checkpoint_path):
        best_trajectory_path = os.path.join(
            results_dirs["trajectory_dir"],
            "trajectory_best.npz",
        )
        render_td3_checkpoint_trajectory(
            best_checkpoint_path,
            best_trajectory_path,
            device=device,
        )
    print(f"Saved evaluation CSV -> {eval_csv}")
    print(f"Saved per-episode training CSV -> {train_csv}")
    print(f"Best fixed-evaluation rank -> {best_eval_rank}")
    print("Training complete.")


if __name__ == "__main__":
    main()
