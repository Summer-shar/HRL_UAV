import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.stats import t as student_t

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hrl_uav.hrl_wrappers import HRL_Wrapper
from hrl_uav.load_switching import apply_load_switch_phase
from hrl_uav.load_switching import build_load_switch_schedule
from hrl_uav.load_switching import parse_load_switch_phases
from hrl_uav.plot_utils import (
    plot_evaluation_metrics,
    plot_macro_diagnostics,
)


BASE_DIR = PROJECT_ROOT
MICRO_DT_SECONDS = 0.1

DEFAULT_EVAL_INTERVAL = 100
DEFAULT_EVAL_EPISODES = 2
DEFAULT_EVAL_BASE_SEED = 1000
EVAL_W_HIGH = 10.0
EVAL_W_LOW = 1.0
EVAL_W_ENERGY = 0.05
EVAL_W_U2G = 0.1

MacroPolicy = Callable[[HRL_Wrapper, np.ndarray, bool], np.ndarray]
MicroPolicy = Callable[[HRL_Wrapper, bool], np.ndarray]
EpisodeStartHook = Callable[[HRL_Wrapper, int], None]

LOAD_SWITCH_TRACE_FIELDS = [
    "episode",
    "episode_seed",
    "macro_index",
    "phase_index",
    "switch_flag",
    "phase_loads_mbps",
    "energy_dual",
    "macro_energy_j",
    "service_energy_j",
    "macro_average_power_w",
    "macro_reward",
    "delay_objective_mean",
    "movement_mean_m",
]


def add_common_args(
    parser: argparse.ArgumentParser,
    default_results_subdir: str,
) -> None:
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Total episodes to run",
    )
    parser.add_argument(
        "--macro_steps",
        type=int,
        default=30,
        help="Macro steps per episode",
    )
    parser.add_argument(
        "--micro_steps",
        type=int,
        default=100,
        help="Micro steps per macro step",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=3,
        help="Number of UAVs",
    )
    parser.add_argument(
        "--num_users",
        type=int,
        default=39,
        help="Number of ground users",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Global random seed",
    )
    parser.add_argument(
        "--trajectory_plot_interval",
        type=int,
        default=0,
        help="Save trajectory/macro diagnostic snapshots every N episodes; 0 disables it",
    )
    parser.add_argument(
        "--traffic_profile",
        type=str,
        choices=["easy", "normal", "hard"],
        default="normal",
        help="Traffic arrival-rate profile",
    )
    parser.add_argument(
        "--traffic_scale",
        type=float,
        default=0.5,
        help="Global multiplier for all arrival rates",
    )
    parser.add_argument(
        "--load_scale",
        type=float,
        default=1.0,
        help="Multiplier on per-zone target load (ZONE_TARGET_LOAD_MBPS); the real demand knob",
    )
    parser.add_argument(
        "--zone_loads",
        type=str,
        default="",
        help=(
            "Comma-separated base offered load per zone in Mbps, before "
            "load_scale. Empty keeps the built-in default."
        ),
    )
    parser.add_argument(
        "--load_switch_mode",
        choices=["off", "cyclic_random"],
        default="off",
        help=(
            "Episode load process. cyclic_random cycles through the frozen "
            "load phases with an independently seeded random dwell."
        ),
    )
    parser.add_argument(
        "--load_switch_phases",
        type=str,
        default="",
        help="Semicolon-separated load vectors, e.g. 12.5,5.5,3.5;...",
    )
    parser.add_argument(
        "--load_switch_dwell_min",
        type=int,
        default=8,
        help="Minimum macro periods per load phase.",
    )
    parser.add_argument(
        "--load_switch_dwell_max",
        type=int,
        default=12,
        help="Maximum macro periods per load phase.",
    )
    parser.add_argument(
        "--uav_initial_positions",
        type=str,
        default="",
        help=(
            "Optional semicolon-separated x,y start positions, for example "
            "250,250;250,750;750,500"
        ),
    )
    parser.add_argument(
        "--arrival_dt",
        type=float,
        default=0.5,
        help="Task-arrival interval in seconds; must align with resource frames",
    )
    parser.add_argument(
        "--resource_frame_dt",
        type=float,
        default=0.5,
        help="SAC/SCA decision-frame duration; actions are held within the frame",
    )
    parser.add_argument(
        "--deployment_dt",
        type=float,
        default=1.0,
        help="No-arrival/no-service deployment duration at each macro-period start",
    )
    parser.add_argument(
        "--deployment_transient_energy_mode",
        type=str,
        choices=["none", "kinetic_lower_bound"],
        default="kinetic_lower_bound",
        help=(
            "Deployment start/stop energy model. kinetic_lower_bound adds "
            "0.5*m*v^2 to the steady-flight energy of each relocation leg."
        ),
    )
    parser.add_argument(
        "--uav_mass_kg",
        type=float,
        default=20.0 / 9.81,
        help="UAV mass used by the deployment transient-energy model",
    )
    parser.add_argument(
        "--users_static",
        type=int,
        choices=[0, 1],
        default=0,
        help="Disable (1) or enable (0) Gauss-Markov ground-user mobility",
    )
    parser.add_argument(
        "--gm_mean_speed_mps",
        type=float,
        default=0.5,
        help="Mean ground-user speed in the Gauss-Markov model",
    )
    parser.add_argument(
        "--gm_mean_heading_rad",
        type=float,
        default=float(np.pi / 4.0),
        help="Mean ground-user heading in radians",
    )
    parser.add_argument(
        "--gm_heading_mode",
        choices=[
            "shared_mean",
            "independent_episode_mean",
            "reference_point_group",
        ],
        default="shared_mean",
        help=(
            "Use one shared mean heading or independently sample a persistent "
            "mean heading for every user at each episode reset"
        ),
    )
    parser.add_argument(
        "--rpgm_group_speed_mps",
        type=float,
        default=0.35,
        help="Mean speed of each moving group reference point",
    )
    parser.add_argument(
        "--rpgm_group_speed_memory",
        type=float,
        default=0.9,
        help="Temporal correlation of group-reference speed",
    )
    parser.add_argument(
        "--rpgm_group_speed_noise_std",
        type=float,
        default=0.04,
        help="Speed innovation standard deviation for group references",
    )
    parser.add_argument(
        "--rpgm_group_turn_std_rad_sqrt_s",
        type=float,
        default=0.12,
        help="Angular diffusion of group references in rad/sqrt(s)",
    )
    parser.add_argument(
        "--rpgm_local_velocity_memory",
        type=float,
        default=0.9,
        help="Temporal correlation of user-local velocity",
    )
    parser.add_argument(
        "--rpgm_local_velocity_std_mps",
        type=float,
        default=0.18,
        help="Stationary per-axis scale of local user velocity",
    )
    parser.add_argument(
        "--rpgm_local_restore_rate_s2",
        type=float,
        default=0.0002,
        help="Smooth restoring acceleration toward the moving group reference",
    )
    parser.add_argument(
        "--rpgm_local_speed_cap_mps",
        type=float,
        default=0.45,
        help="Maximum speed of a user's local motion component",
    )
    parser.add_argument(
        "--gm_speed_memory",
        type=float,
        default=0.5,
        help="Gauss-Markov speed correlation coefficient c1",
    )
    parser.add_argument(
        "--gm_heading_memory",
        type=float,
        default=0.5,
        help="Gauss-Markov heading correlation coefficient c2",
    )
    parser.add_argument(
        "--gm_speed_noise_std",
        type=float,
        default=0.1,
        help="Standard deviation of the speed innovation Phi",
    )
    parser.add_argument(
        "--gm_heading_noise_std",
        type=float,
        default=0.1,
        help="Standard deviation of the heading innovation Psi",
    )
    parser.add_argument(
        "--user_zone_radius",
        type=float,
        default=80.0,
        help="Radius constraining each user to its service zone",
    )
    parser.add_argument(
        "--uplink_bandwidth_mhz",
        type=float,
        default=20.0,
        help="Total ground-to-UAV uplink bandwidth in MHz",
    )
    parser.add_argument(
        "--u2u_bandwidth_mhz",
        type=float,
        default=3.0,
        help="Total UAV-to-UAV bandwidth in MHz",
    )
    parser.add_argument(
        "--min_uav_separation_m",
        type=float,
        default=5.0,
        help="Minimum horizontal separation between any two UAVs",
    )
    parser.add_argument(
        "--cpu_ghz",
        type=float,
        default=16.0,
        help="CPU frequency of every UAV in GHz",
    )
    parser.add_argument(
        "--cpu_alloc_mode",
        type=str,
        choices=[
            "legacy_kkt",
            "priority_kkt",
            "fixed_full",
            "fixed_average",
            "fixed_demand",
            "priority_demand",
            "priced_maxweight",
        ],
        default="priority_kkt",
        help="CPU resource allocation rule",
    )
    parser.add_argument(
        "--kkt_v",
        type=float,
        default=1.0,
        help="Lyapunov/KKT energy-delay tradeoff weight",
    )
    parser.add_argument(
        "--fixed_cpu_util",
        type=float,
        default=0.5,
        help="CPU utilization for fixed_average allocation",
    )
    parser.add_argument(
        "--cpu_power_cap_w",
        type=float,
        default=0.0,
        help=(
            "Optional system-wide CPU power cap applied by priority_demand; "
            "zero disables the deterministic feasibility projection"
        ),
    )
    parser.add_argument(
        "--micro_diag_bias",
        type=float,
        default=1.0,
        help="Diagonal retention bias before micro-action simplex projection",
    )
    parser.add_argument(
        "--micro_action_scale",
        type=float,
        default=1.0,
        help="Scale applied before routing-simplex projection",
    )
    parser.add_argument(
        "--micro_action_encoding",
        choices=["projected_full", "stick_breaking", "priority_gated"],
        default="projected_full",
        help="Continuous encoding used for the learned routing action",
    )
    parser.add_argument(
        "--micro_stick_breaking_scale",
        type=float,
        default=4.0,
        help="Logit scale for the centered stick-breaking route encoding",
    )
    parser.add_argument(
        "--micro_priority_weight_ratio",
        type=float,
        default=10.0,
        help=(
            "High-to-low delay weight ratio used only as the local-gate "
            "prior in priority_gated routing"
        ),
    )
    parser.add_argument(
        "--micro_sca_residual_scale",
        type=float,
        default=0.0,
        help=(
            "Non-negative log-ratio scale for an SAC residual restricted to "
            "the active-link support selected by SCA; zero disables it"
        ),
    )
    parser.add_argument(
        "--micro_autonomous_residual_gate",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Prepend a tanh SAC gate latent to the residual action. The "
            "environment applies g=(z_g+1)/2 to the remaining residual "
            "coordinates before the SCA support mapping."
        ),
    )
    parser.add_argument(
        "--micro_sca_support_threshold",
        type=float,
        default=1e-6,
        help="SCA routing share above which a link belongs to the residual support",
    )
    parser.add_argument(
        "--micro_sca_residual_update_interval",
        type=int,
        default=1,
        help="Resource frames between SCA support updates for residual SAC",
    )
    parser.add_argument(
        "--micro_sca_residual_max_iter",
        type=int,
        default=3,
        help="SCA approximation steps used to construct the residual support",
    )
    parser.add_argument(
        "--micro_sca_residual_tol",
        type=float,
        default=1e-3,
        help="SCA convergence tolerance used by residual SAC",
    )
    parser.add_argument(
        "--micro_sca_residual_solver",
        type=str,
        default="CLARABEL",
        help="Convex solver used to construct the residual support",
    )
    parser.add_argument(
        "--micro_sca_residual_proximal_weight",
        type=float,
        default=1e-3,
        help="SCA proximal weight used to construct the residual support",
    )
    parser.add_argument(
        "--sac_zero_mean_output_init",
        type=int,
        choices=[0, 1],
        default=0,
        help="Initialize the SAC deterministic residual action at zero",
    )
    parser.add_argument(
        "--micro_reward_scale",
        type=float,
        default=10.0,
        help="Positive numerical scale applied to the SAC reward",
    )
    parser.add_argument(
        "--delay_objective_weight",
        type=float,
        default=0.7,
        help="Normalized delay weight used by the weighted objective",
    )
    parser.add_argument(
        "--energy_objective_weight",
        type=float,
        default=0.3,
        help="Normalized energy weight used by the weighted objective",
    )
    parser.add_argument(
        "--micro_state_mode",
        type=str,
        choices=["pressure", "legacy"],
        default="pressure",
        help="Micro-state normalization mode",
    )
    parser.add_argument(
        "--u2u_state_representation",
        choices=["nominal_rate", "spectral_efficiency"],
        default="nominal_rate",
        help="U2U observation primitive used by both hierarchy levels",
    )
    parser.add_argument(
        "--u2u_goal_representation",
        choices=["match_observation", "nominal_rate"],
        default="match_observation",
        help="U2U primitive used by the derived micro routing goal",
    )
    parser.add_argument(
        "--u2u_contention_aware_pressure",
        type=int,
        choices=[-1, 0, 1],
        default=-1,
        help="Use previous active-link count in micro pressure capacity; -1 preserves legacy auto behavior",
    )
    parser.add_argument(
        "--u2u_prev_active_state",
        type=int,
        choices=[0, 1],
        default=0,
        help="Append the previous resource-frame active-link fraction",
    )
    parser.add_argument(
        "--macro_comm_weight",
        type=float,
        default=2.0,
        help="Weight of flow-aware communication cost in macro shaping",
    )
    parser.add_argument(
        "--macro_comm_flow_ema",
        type=float,
        default=0.8,
        help="EMA coefficient for recent macro routing flow",
    )
    parser.add_argument(
        "--macro_flow_state",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether macro observations include flow summary features",
    )
    parser.add_argument(
        "--macro_load_state",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether macro observations include absolute expected load features",
    )
    parser.add_argument(
        "--macro_capacity_state",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether macro observations include per-UAV CPU capacity",
    )
    parser.add_argument(
        "--macro_reward_mode",
        choices=["strict_global", "legacy_feudal"],
        default="strict_global",
        help="Exact global micro-return feedback or historical pressure proxy",
    )
    parser.add_argument(
        "--strict_hierarchy_validation",
        type=int,
        choices=[0, 1],
        default=1,
        help="Require exactly one finalized reward per complete macro slot",
    )
    parser.add_argument(
        "--pause_service_during_flight",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Use the fixed 1s/9s hierarchy: the first 1s UAV deployment "
            "phase has no U2G/U2U/CPU service. Exogenous arrivals during "
            "that phase are controlled separately."
        ),
    )
    parser.add_argument(
        "--arrivals_during_deployment",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Keep exogenous task arrivals and user mobility active during "
            "the deployment phase while all communication/CPU service is paused"
        ),
    )
    parser.add_argument(
        "--objective_mode",
        choices=["weighted", "constrained_delay"],
        default="weighted",
        help="Weighted delay-energy cost or delay minimization with a power constraint",
    )
    parser.add_argument(
        "--power_budget_w",
        type=float,
        default=0.0,
        help="Total three-UAV average power budget; required by constrained_delay",
    )
    parser.add_argument(
        "--power_constraint_scope",
        choices=["system", "per_uav"],
        default="system",
        help="Apply one system-average constraint or one battery constraint per UAV",
    )
    parser.add_argument(
        "--uav_power_budgets_w",
        type=str,
        default="",
        help="Scalar or comma-separated per-UAV average battery budgets in watts",
    )
    parser.add_argument(
        "--energy_lagrange_multipliers",
        type=str,
        default="",
        help="Optional scalar or comma-separated initial per-UAV multipliers",
    )
    parser.add_argument(
        "--energy_dual_state_norm",
        type=float,
        default=10.0,
        help="Normalization denominator for dual prices included in RL states",
    )
    parser.add_argument(
        "--system_dual_state",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Include the normalized system-level energy multiplier in both "
            "macro and micro observations"
        ),
    )
    parser.add_argument(
        "--energy_dual_initial",
        type=float,
        default=0.0,
        help="Initial non-negative Lagrange multiplier for the power constraint",
    )
    parser.add_argument(
        "--energy_dual_lr",
        type=float,
        default=0.02,
        help="Episode-level dual ascent step size",
    )
    parser.add_argument(
        "--energy_dual_max",
        type=float,
        default=20.0,
        help="Upper bound for the learned energy multiplier",
    )
    parser.add_argument(
        "--power_budget_tolerance",
        type=float,
        default=0.01,
        help="Relative feasibility tolerance used for checkpoint selection",
    )
    parser.add_argument(
        "--power_constraint_margin_w",
        type=float,
        default=0.0,
        help=(
            "Non-negative training and checkpoint-selection safety margin; "
            "the reported physical budget is unchanged"
        ),
    )
    parser.add_argument(
        "--checkpoint_power_rule",
        type=str,
        choices=["mean_target", "ucb95"],
        default="mean_target",
        help=(
            "Checkpoint feasibility rule. ucb95 requires the one-sided "
            "95%% Student-t upper confidence bound to satisfy the physical "
            "power budget."
        ),
    )
    parser.add_argument(
        "--macro_u2g_cost_weight",
        type=float,
        default=2.0,
        help="Weight of legacy centroid-distance macro cost",
    )
    parser.add_argument(
        "--macro_zone_ingress_weight",
        type=float,
        default=0.5,
        help="Weight of legacy outside-zone macro cost",
    )
    parser.add_argument(
        "--macro_action_mode",
        type=str,
        choices=[
            "polar",
            "delta",
            "waypoint",
            "centroid_waypoint",
            "cooperation_waypoint",
        ],
        default="polar",
        help="Macro action interpretation in HRL_Wrapper",
    )
    parser.add_argument(
        "--macro_action_deadzone",
        type=float,
        default=0.05,
        help="Normalized macro-action magnitude below which the UAV hovers",
    )
    parser.add_argument(
        "--macro_waypoint_radius_m",
        type=float,
        default=200.0,
        help=(
            "Maximum target offset from each episode's initial UAV position "
            "or current duty centroid, depending on the action mode"
        ),
    )
    parser.add_argument(
        "--macro_waypoint_tolerance_m",
        type=float,
        default=0.5,
        help="Distance to a learned waypoint below which the UAV hovers",
    )
    parser.add_argument(
        "--macro_ckpt",
        type=str,
        default="",
        help="Checkpoint used to initialize the macro MAPPO layer",
    )
    parser.add_argument(
        "--freeze_macro",
        action="store_true",
        help="Freeze the macro MAPPO layer and only train/evaluate the micro layer",
    )
    parser.add_argument(
        "--macro_deterministic",
        action="store_true",
        help="Use deterministic macro actions during training",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=DEFAULT_EVAL_INTERVAL,
        help="Evaluate every N episodes; 0 disables periodic evaluation",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="Number of fixed-seed evaluation episodes",
    )
    parser.add_argument(
        "--eval_base_seed",
        type=int,
        default=DEFAULT_EVAL_BASE_SEED,
        help="Base seed for evaluation episodes",
    )
    parser.add_argument(
        "--sac_batch_size",
        type=int,
        default=256,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sac_memory_warmup",
        type=int,
        default=1000,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sac_update_freq",
        type=int,
        default=100,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sac_gradient_steps",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sac_policy_delay",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sac_actor_warmup_updates",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sac_grad_clip_norm",
        type=float,
        default=0.0,
        help="SAC actor/critic gradient-norm cap; zero disables clipping",
    )
    parser.add_argument(
        "--results_subdir",
        type=str,
        default=default_results_subdir,
        help="Results subdirectory under ./results",
    )
    parser.add_argument(
        "--train_cpu_min",
        type=float,
        default=0.0,
        help="Minimum randomized training CPU GHz; <=0 disables randomization",
    )
    parser.add_argument(
        "--train_cpu_max",
        type=float,
        default=0.0,
        help="Maximum randomized training CPU GHz; <=0 disables randomization",
    )
    parser.add_argument(
        "--train_scenario_mode",
        choices=["independent", "axes"],
        default="independent",
        help="Randomize CPU/load independently or along the two paper sweep axes",
    )


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_project_path(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def parse_zone_loads(text: str) -> Optional[List[float]]:
    if text is None:
        return None
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        return None
    loads = [float(item) for item in values]
    if any((not np.isfinite(item)) or item <= 0.0 for item in loads):
        raise ValueError("zone_loads must contain finite positive Mbps values.")
    return loads


def make_episode_load_switch_schedule(args, episode_seed, num_agents):
    mode = str(getattr(args, "load_switch_mode", "off")).lower()
    if mode == "off":
        return None
    phases = parse_load_switch_phases(
        getattr(args, "load_switch_phases", ""),
        num_agents,
    )
    return build_load_switch_schedule(
        mode=mode,
        phases_mbps=phases,
        macro_steps=int(args.macro_steps),
        episode_seed=int(episode_seed),
        dwell_min=int(getattr(args, "load_switch_dwell_min", 8)),
        dwell_max=int(getattr(args, "load_switch_dwell_max", 12)),
    )


def initialize_episode_load_switch(env, args, episode_seed):
    """Set phase zero before reset without touching any shared RNG stream."""
    schedule = make_episode_load_switch_schedule(
        args,
        episode_seed,
        int(env.N),
    )
    if schedule is not None:
        apply_load_switch_phase(env, schedule, 0, force=True)
    return schedule


def advance_load_switch_before_macro(env, schedule, macro_index):
    if schedule is None:
        return False
    return apply_load_switch_phase(env, schedule, macro_index, force=False)


def initialize_load_switch_trace(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        csv.DictWriter(fp, fieldnames=LOAD_SWITCH_TRACE_FIELDS).writeheader()


def append_load_switch_trace(path, row):
    with open(path, "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=LOAD_SWITCH_TRACE_FIELDS)
        writer.writerow(
            {field: row.get(field, "") for field in LOAD_SWITCH_TRACE_FIELDS}
        )


def parse_nonnegative_vector(text: str) -> Optional[List[float]]:
    if text is None:
        return None
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        return None
    parsed = [float(item) for item in values]
    if any((not np.isfinite(item)) or item < 0.0 for item in parsed):
        raise ValueError("Expected finite non-negative numeric values.")
    return parsed


def parse_uav_initial_positions(text, num_agents):
    if text is None or not str(text).strip():
        return None
    rows = []
    for group in str(text).split(";"):
        values = [
            float(item.strip())
            for item in group.split(",")
            if item.strip()
        ]
        if len(values) != 2:
            raise ValueError(
                "Each UAV start position must contain exactly x,y."
            )
        rows.append(values)
    positions = np.asarray(rows, dtype=np.float64)
    if positions.shape != (int(num_agents), 2):
        raise ValueError(
            "uav_initial_positions must provide one x,y pair per UAV."
        )
    return positions


def torch_load_checkpoint(path: str) -> Dict:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def apply_env_config(env: HRL_Wrapper, args: argparse.Namespace) -> HRL_Wrapper:
    cpu_ghz = float(getattr(args, "cpu_ghz", 16.0))
    env.env.F_max_array[:] = cpu_ghz * 1e9
    env.macro_action_deadzone = float(
        getattr(args, "macro_action_deadzone", env.macro_action_deadzone)
    )
    return env


def make_env_from_args(
    args: argparse.Namespace,
    record_trajectory: bool = False,
) -> HRL_Wrapper:
    objective_mode = str(getattr(args, "objective_mode", "weighted"))
    constraint_scope = str(
        getattr(args, "power_constraint_scope", "system")
    )
    system_dual_state = bool(int(getattr(args, "system_dual_state", 0)))
    if (
        objective_mode == "constrained_delay"
        and constraint_scope == "system"
        and float(getattr(args, "energy_dual_lr", 0.0)) > 0.0
        and not system_dual_state
    ):
        raise ValueError(
            "A learned system energy multiplier changes priced_maxweight "
            "transitions and must be observable: set --system_dual_state 1."
        )
    env = HRL_Wrapper(
        num_uavs=args.num_agents,
        num_users=args.num_users,
        record_trajectory=record_trajectory,
        traffic_profile=args.traffic_profile,
        traffic_scale=args.traffic_scale,
        arrival_dt=float(getattr(args, "arrival_dt", 0.5)),
        resource_frame_dt=float(getattr(args, "resource_frame_dt", 0.5)),
        deployment_dt=float(getattr(args, "deployment_dt", 1.0)),
        deployment_transient_energy_mode=str(
            getattr(args, "deployment_transient_energy_mode", "none")
        ),
        uav_mass_kg=float(getattr(args, "uav_mass_kg", 20.0 / 9.81)),
        uplink_bandwidth_mhz=float(getattr(args, "uplink_bandwidth_mhz", 20.0)),
        u2u_bandwidth_mhz=float(getattr(args, "u2u_bandwidth_mhz", 3.0)),
        micro_diag_bias=float(getattr(args, "micro_diag_bias", 0.5)),
        micro_action_scale=float(getattr(args, "micro_action_scale", 1.0)),
        micro_action_encoding=str(
            getattr(args, "micro_action_encoding", "projected_full")
        ),
        micro_stick_breaking_scale=float(
            getattr(args, "micro_stick_breaking_scale", 4.0)
        ),
        micro_priority_weight_ratio=float(
            getattr(args, "micro_priority_weight_ratio", 10.0)
        ),
        micro_reward_scale=float(getattr(args, "micro_reward_scale", 10.0)),
        delay_objective_weight=float(
            getattr(args, "delay_objective_weight", 0.7)
        ),
        energy_objective_weight=float(
            getattr(args, "energy_objective_weight", 0.3)
        ),
        micro_state_mode=str(getattr(args, "micro_state_mode", "pressure")),
        u2u_state_representation=str(
            getattr(args, "u2u_state_representation", "nominal_rate")
        ),
        u2u_goal_representation=str(
            getattr(args, "u2u_goal_representation", "match_observation")
        ),
        u2u_contention_aware_pressure=(
            None
            if int(getattr(args, "u2u_contention_aware_pressure", -1)) < 0
            else bool(int(getattr(args, "u2u_contention_aware_pressure", 0)))
        ),
        u2u_prev_active_state=bool(
            int(getattr(args, "u2u_prev_active_state", 0))
        ),
        cpu_alloc_mode=str(getattr(args, "cpu_alloc_mode", "priority_kkt")),
        kkt_v=float(getattr(args, "kkt_v", 1.0)),
        fixed_cpu_util=float(getattr(args, "fixed_cpu_util", 0.5)),
        cpu_power_cap_w=float(getattr(args, "cpu_power_cap_w", 0.0)),
        load_scale=float(getattr(args, "load_scale", 1.0)),
        zone_loads=parse_zone_loads(getattr(args, "zone_loads", "")),
        uav_initial_positions=parse_uav_initial_positions(
            getattr(args, "uav_initial_positions", ""),
            args.num_agents,
        ),
        macro_comm_weight=float(getattr(args, "macro_comm_weight", 2.0)),
        macro_comm_flow_ema=float(getattr(args, "macro_comm_flow_ema", 0.8)),
        macro_flow_state=bool(int(getattr(args, "macro_flow_state", 1))),
        macro_load_state=bool(int(getattr(args, "macro_load_state", 0))),
        macro_capacity_state=bool(
            int(getattr(args, "macro_capacity_state", 0))
        ),
        macro_u2g_cost_weight=float(
            getattr(args, "macro_u2g_cost_weight", 2.0)
        ),
        macro_zone_ingress_weight=float(
            getattr(args, "macro_zone_ingress_weight", 0.5)
        ),
        macro_action_mode=str(getattr(args, "macro_action_mode", "polar")),
        macro_waypoint_radius_m=float(
            getattr(args, "macro_waypoint_radius_m", 200.0)
        ),
        macro_waypoint_tolerance_m=float(
            getattr(args, "macro_waypoint_tolerance_m", 0.5)
        ),
        macro_reward_mode=str(
            getattr(args, "macro_reward_mode", "legacy_feudal")
        ),
        strict_hierarchy_validation=bool(
            int(getattr(args, "strict_hierarchy_validation", 0))
        ),
        pause_service_during_flight=bool(
            int(getattr(args, "pause_service_during_flight", 1))
        ),
        arrivals_during_deployment=bool(
            int(getattr(args, "arrivals_during_deployment", 0))
        ),
        objective_mode=objective_mode,
        power_budget_w=float(getattr(args, "power_budget_w", 0.0)),
        energy_lagrange_multiplier=float(
            getattr(
                args,
                "energy_lagrange_multiplier_current",
                getattr(args, "energy_dual_initial", 0.0),
            )
        ),
        power_constraint_scope=constraint_scope,
        uav_power_budgets_w=parse_nonnegative_vector(
            getattr(args, "uav_power_budgets_w", "")
        ),
        energy_lagrange_multipliers=parse_nonnegative_vector(
            getattr(args, "energy_lagrange_multipliers", "")
        ),
        energy_dual_state_norm=float(
            getattr(args, "energy_dual_state_norm", 10.0)
        ),
        system_dual_state=system_dual_state,
        users_static=bool(int(getattr(args, "users_static", 0))),
        gm_mean_speed_mps=float(getattr(args, "gm_mean_speed_mps", 0.5)),
        gm_mean_heading_rad=float(
            getattr(args, "gm_mean_heading_rad", np.pi / 4.0)
        ),
        gm_heading_mode=str(getattr(args, "gm_heading_mode", "shared_mean")),
        gm_speed_memory=float(getattr(args, "gm_speed_memory", 0.5)),
        gm_heading_memory=float(getattr(args, "gm_heading_memory", 0.5)),
        gm_speed_noise_std=float(getattr(args, "gm_speed_noise_std", 0.1)),
        gm_heading_noise_std=float(
            getattr(args, "gm_heading_noise_std", 0.1)
        ),
        rpgm_group_speed_mps=float(
            getattr(args, "rpgm_group_speed_mps", 0.35)
        ),
        rpgm_group_speed_memory=float(
            getattr(args, "rpgm_group_speed_memory", 0.9)
        ),
        rpgm_group_speed_noise_std=float(
            getattr(args, "rpgm_group_speed_noise_std", 0.04)
        ),
        rpgm_group_turn_std_rad_sqrt_s=float(
            getattr(args, "rpgm_group_turn_std_rad_sqrt_s", 0.12)
        ),
        rpgm_local_velocity_memory=float(
            getattr(args, "rpgm_local_velocity_memory", 0.9)
        ),
        rpgm_local_velocity_std_mps=float(
            getattr(args, "rpgm_local_velocity_std_mps", 0.18)
        ),
        rpgm_local_restore_rate_s2=float(
            getattr(args, "rpgm_local_restore_rate_s2", 0.0002)
        ),
        rpgm_local_speed_cap_mps=float(
            getattr(args, "rpgm_local_speed_cap_mps", 0.45)
        ),
        user_zone_radius=float(getattr(args, "user_zone_radius", 80.0)),
        min_uav_separation_m=float(
            getattr(args, "min_uav_separation_m", 5.0)
        ),
    )
    return apply_env_config(env, args)


def prepare_results_dirs(results_subdir: str) -> Dict[str, str]:
    if os.path.isabs(results_subdir):
        results_dir = results_subdir
    else:
        results_dir = os.path.join(BASE_DIR, "results", results_subdir)

    trajectory_dir = os.path.join(results_dir, "trajectory_snapshots")
    macro_diag_dir = os.path.join(results_dir, "macro_diagnostics")
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    csv_dir = os.path.join(results_dir, "csv")
    fig_dir = os.path.join(results_dir, "figures")
    meta_dir = os.path.join(results_dir, "meta")

    os.makedirs(results_dir, exist_ok=True)

    return {
        "results_dir": results_dir,
        "trajectory_dir": trajectory_dir,
        "macro_diag_dir": macro_diag_dir,
        "ckpt_dir": ckpt_dir,
        "csv_dir": csv_dir,
        "fig_dir": fig_dir,
        "meta_dir": meta_dir,
    }


def save_scenario_metadata(
    results_dirs: Dict[str, str],
    args: argparse.Namespace,
    experiment_label: str = "",
) -> str:
    meta_dir = results_dirs["meta_dir"]
    os.makedirs(meta_dir, exist_ok=True)
    scenario_path = os.path.join(meta_dir, "scenario.json")
    zone_loads = parse_zone_loads(getattr(args, "zone_loads", ""))
    payload = {
        "experiment_label": experiment_label,
        "num_agents": int(getattr(args, "num_agents", 3)),
        "num_users": int(getattr(args, "num_users", 39)),
        "traffic_profile": getattr(args, "traffic_profile", "normal"),
        "traffic_scale": float(getattr(args, "traffic_scale", 1.0)),
        "load_scale": float(getattr(args, "load_scale", 1.0)),
        "zone_loads": getattr(args, "zone_loads", ""),
        "zone_target_load_mbps": zone_loads,
        "load_switch_mode": str(
            getattr(args, "load_switch_mode", "off")
        ),
        "load_switch_phases": str(
            getattr(args, "load_switch_phases", "")
        ),
        "load_switch_dwell_min": int(
            getattr(args, "load_switch_dwell_min", 8)
        ),
        "load_switch_dwell_max": int(
            getattr(args, "load_switch_dwell_max", 12)
        ),
        "load_switch_timing": "before_macro_state_observation",
        "load_switch_future_information": False,
        "uav_initial_positions": getattr(args, "uav_initial_positions", ""),
        "arrival_dt": float(getattr(args, "arrival_dt", 0.5)),
        "resource_frame_dt": float(
            getattr(args, "resource_frame_dt", 0.5)
        ),
        "internal_integration_dt_s": MICRO_DT_SECONDS,
        "resource_frame_substeps": int(
            round(
                float(getattr(args, "resource_frame_dt", 0.5))
                / MICRO_DT_SECONDS
            )
        ),
        "deployment_dt": float(getattr(args, "deployment_dt", 1.0)),
        "deployment_resource_frames": int(
            round(
                float(getattr(args, "deployment_dt", 1.0))
                / float(getattr(args, "resource_frame_dt", 0.5))
            )
        ),
        "service_resource_frames": int(
            round(
                (
                    10.0 - float(getattr(args, "deployment_dt", 1.0))
                )
                / float(getattr(args, "resource_frame_dt", 0.5))
            )
        ),
        "smdp_boundary_discount_exponent": 3,
        "users_static": int(getattr(args, "users_static", 0)),
        "gm_mean_speed_mps": float(
            getattr(args, "gm_mean_speed_mps", 0.5)
        ),
        "gm_mean_heading_rad": float(
            getattr(args, "gm_mean_heading_rad", np.pi / 4.0)
        ),
        "gm_heading_mode": str(
            getattr(args, "gm_heading_mode", "shared_mean")
        ),
        "gm_speed_memory": float(getattr(args, "gm_speed_memory", 0.5)),
        "gm_heading_memory": float(
            getattr(args, "gm_heading_memory", 0.5)
        ),
        "gm_speed_noise_std": float(
            getattr(args, "gm_speed_noise_std", 0.1)
        ),
        "gm_heading_noise_std": float(
            getattr(args, "gm_heading_noise_std", 0.1)
        ),
        "rpgm_group_speed_mps": float(
            getattr(args, "rpgm_group_speed_mps", 0.35)
        ),
        "rpgm_group_speed_memory": float(
            getattr(args, "rpgm_group_speed_memory", 0.9)
        ),
        "rpgm_group_speed_noise_std": float(
            getattr(args, "rpgm_group_speed_noise_std", 0.04)
        ),
        "rpgm_group_turn_std_rad_sqrt_s": float(
            getattr(args, "rpgm_group_turn_std_rad_sqrt_s", 0.12)
        ),
        "rpgm_local_velocity_memory": float(
            getattr(args, "rpgm_local_velocity_memory", 0.9)
        ),
        "rpgm_local_velocity_std_mps": float(
            getattr(args, "rpgm_local_velocity_std_mps", 0.18)
        ),
        "rpgm_local_restore_rate_s2": float(
            getattr(args, "rpgm_local_restore_rate_s2", 0.0002)
        ),
        "rpgm_local_speed_cap_mps": float(
            getattr(args, "rpgm_local_speed_cap_mps", 0.45)
        ),
        "user_zone_radius": float(getattr(args, "user_zone_radius", 80.0)),
        "uplink_bandwidth_mhz": float(
            getattr(args, "uplink_bandwidth_mhz", 20.0)
        ),
        "u2u_bandwidth_mhz": float(getattr(args, "u2u_bandwidth_mhz", 3.0)),
        "min_uav_separation_m": float(
            getattr(args, "min_uav_separation_m", 5.0)
        ),
        "communication_model_reference": (
            "Xu et al., IEEE TCOM 2024, "
            "DOI 10.1109/TCOMM.2024.3361536"
        ),
        "u2g_association_policy": "fixed_region_per_arrival_frame",
        "u2g_bandwidth_policy": "orthogonal_equal_per_ground_terminal",
        "u2u_bandwidth_policy": (
            "work_conserving_equal_active_directed_links"
        ),
        "interference_model": "orthogonal_u2g_u2u_no_interference",
        "downlink_result_return": "ignored_as_in_reference_model",
        "u2u_active_link_allocation": (
            "B_M / number_of_active_directed_links"
        ),
        "routing_action_mapping": str(
            getattr(args, "micro_action_encoding", "projected_full")
        ),
        "cpu_ghz": float(getattr(args, "cpu_ghz", 16.0)),
        "cpu_alloc_mode": getattr(args, "cpu_alloc_mode", "priority_kkt"),
        "kkt_v": float(getattr(args, "kkt_v", 1.0)),
        "fixed_cpu_util": float(getattr(args, "fixed_cpu_util", 0.5)),
        "cpu_power_cap_w": float(getattr(args, "cpu_power_cap_w", 0.0)),
        "cpu_energy_model": "kappa_times_total_frequency_cubed",
        "cpu_energy_price_formula": "lambda_i / average_power_budget_i",
        "micro_diag_bias": float(getattr(args, "micro_diag_bias", 0.5)),
        "micro_action_scale": float(getattr(args, "micro_action_scale", 1.0)),
        "micro_stick_breaking_scale": float(
            getattr(args, "micro_stick_breaking_scale", 4.0)
        ),
        "micro_priority_weight_ratio": float(
            getattr(args, "micro_priority_weight_ratio", 10.0)
        ),
        "micro_sca_residual_scale": float(
            getattr(args, "micro_sca_residual_scale", 0.0)
        ),
        "micro_autonomous_residual_gate": int(
            getattr(args, "micro_autonomous_residual_gate", 0)
        ),
        "micro_sca_support_threshold": float(
            getattr(args, "micro_sca_support_threshold", 1e-6)
        ),
        "micro_sca_residual_update_interval": int(
            getattr(args, "micro_sca_residual_update_interval", 1)
        ),
        "micro_sca_residual_max_iter": int(
            getattr(args, "micro_sca_residual_max_iter", 3)
        ),
        "micro_sca_residual_tol": float(
            getattr(args, "micro_sca_residual_tol", 1e-3)
        ),
        "micro_sca_residual_solver": str(
            getattr(args, "micro_sca_residual_solver", "CLARABEL")
        ),
        "micro_sca_residual_proximal_weight": float(
            getattr(args, "micro_sca_residual_proximal_weight", 1e-3)
        ),
        "sac_zero_mean_output_init": int(
            getattr(args, "sac_zero_mean_output_init", 0)
        ),
        "micro_reward_scale": float(
            getattr(args, "micro_reward_scale", 10.0)
        ),
        "delay_objective_weight": float(
            getattr(args, "delay_objective_weight", 0.7)
        ),
        "energy_objective_weight": float(
            getattr(args, "energy_objective_weight", 0.3)
        ),
        "micro_state_mode": getattr(args, "micro_state_mode", "pressure"),
        "u2u_state_representation": getattr(
            args, "u2u_state_representation", "nominal_rate"
        ),
        "u2u_goal_representation": getattr(
            args, "u2u_goal_representation", "match_observation"
        ),
        "u2u_contention_aware_pressure": int(
            getattr(args, "u2u_contention_aware_pressure", -1)
        ),
        "u2u_prev_active_state": int(
            getattr(args, "u2u_prev_active_state", 0)
        ),
        "micro_solver": getattr(args, "micro_solver", "sac"),
        "sac_actor_lr": float(getattr(args, "sac_actor_lr", 3e-4)),
        "sac_critic_lr": float(getattr(args, "sac_critic_lr", 3e-4)),
        "sac_alpha_lr": float(getattr(args, "sac_alpha_lr", 3e-4)),
        "sac_initial_alpha": float(
            getattr(args, "sac_initial_alpha", 1.0)
        ),
        "sac_grad_clip_norm": float(
            getattr(args, "sac_grad_clip_norm", 0.0)
        ),
        "resume_micro_optimizer": int(
            getattr(args, "resume_micro_optimizer", 1)
        ),
        "sca_update_interval": int(
            getattr(args, "sca_update_interval", 0)
        ),
        "sca_max_iter": int(getattr(args, "sca_max_iter", 0)),
        "sca_tol": float(getattr(args, "sca_tol", 0.0)),
        "sca_solver": getattr(args, "sca_solver", ""),
        "sca_delay_weight": float(
            getattr(args, "sca_delay_weight", 0.0)
        ),
        "sca_energy_weight": float(
            getattr(args, "sca_energy_weight", 0.0)
        ),
        "sca_proximal_weight": float(
            getattr(args, "sca_proximal_weight", 0.0)
        ),
        "macro_comm_weight": float(getattr(args, "macro_comm_weight", 2.0)),
        "macro_comm_flow_ema": float(
            getattr(args, "macro_comm_flow_ema", 0.8)
        ),
        "macro_flow_state": int(getattr(args, "macro_flow_state", 1)),
        "macro_load_state": int(getattr(args, "macro_load_state", 0)),
        "macro_capacity_state": int(
            getattr(args, "macro_capacity_state", 0)
        ),
        "objective_mode": str(getattr(args, "objective_mode", "weighted")),
        "power_budget_w": float(getattr(args, "power_budget_w", 0.0)),
        "power_constraint_scope": str(
            getattr(args, "power_constraint_scope", "system")
        ),
        "uav_power_budgets_w": parse_nonnegative_vector(
            getattr(args, "uav_power_budgets_w", "")
        ),
        "energy_lagrange_multipliers_initial": parse_nonnegative_vector(
            getattr(args, "energy_lagrange_multipliers", "")
        ),
        "energy_dual_state_norm": float(
            getattr(args, "energy_dual_state_norm", 10.0)
        ),
        "system_dual_state": int(
            getattr(args, "system_dual_state", 0)
        ),
        "uav_battery_energy_scope": [
            "propulsion",
            "u2u_transmit",
            "compute",
        ],
        "ground_uplink_energy_scope": "ground_user_only_reported_separately",
        "energy_dual_initial": float(
            getattr(args, "energy_dual_initial", 0.0)
        ),
        "energy_dual_lr": float(getattr(args, "energy_dual_lr", 0.0)),
        "energy_dual_max": float(getattr(args, "energy_dual_max", 0.0)),
        "train_scenario_mode": str(
            getattr(args, "train_scenario_mode", "independent")
        ),
        "train_start_jitter_max_m": float(
            getattr(args, "train_start_jitter_max_m", 0.0)
        ),
        "train_start_anchor_probability": float(
            getattr(args, "train_start_anchor_probability", 0.2)
        ),
        "macro_u2g_cost_weight": float(
            getattr(args, "macro_u2g_cost_weight", 2.0)
        ),
        "macro_zone_ingress_weight": float(
            getattr(args, "macro_zone_ingress_weight", 0.5)
        ),
        "macro_action_mode": getattr(args, "macro_action_mode", "polar"),
        "macro_action_deadzone": float(
            getattr(args, "macro_action_deadzone", 0.05)
        ),
        "macro_waypoint_radius_m": float(
            getattr(args, "macro_waypoint_radius_m", 200.0)
        ),
        "macro_waypoint_tolerance_m": float(
            getattr(args, "macro_waypoint_tolerance_m", 0.5)
        ),
        "td3_replay_capacity": int(
            getattr(args, "td3_replay_capacity", 0)
        ),
        "td3_warmup_steps": int(getattr(args, "td3_warmup_steps", 0)),
        "td3_warmup_action_radius": float(
            getattr(args, "td3_warmup_action_radius", 0.0)
        ),
        "td3_exploration_initial": float(
            getattr(args, "td3_exploration_initial", 0.0)
        ),
        "td3_exploration_final": float(
            getattr(args, "td3_exploration_final", 0.0)
        ),
        "td3_exploration_decay_episodes": int(
            getattr(args, "td3_exploration_decay_episodes", 0)
        ),
        "td3_exploration_persistence": str(
            getattr(args, "td3_exploration_persistence", "auto")
        ),
        "macro_slot_duration_s": 10.0,
        "macro_control_no_service_s": float(
            getattr(args, "deployment_dt", 1.0)
        ),
        "macro_control_task_arrivals": int(
            getattr(args, "arrivals_during_deployment", 0)
        ),
        "micro_service_duration_s": 10.0
        - float(getattr(args, "deployment_dt", 1.0)),
        "fixed_macro_control_phase": int(
            getattr(args, "pause_service_during_flight", 1)
        ),
        "advisor_weight": float(getattr(args, "advisor_weight", 0.0)),
        "advisor_final_weight": float(
            getattr(args, "advisor_final_weight", 0.0)
        ),
        "advisor_decay_episodes": int(
            getattr(args, "advisor_decay_episodes", 0)
        ),
        "advisor_steps": int(getattr(args, "advisor_steps", 0)),
        "advisor_lr": float(getattr(args, "advisor_lr", 0.0)),
        "advisor_min_separation_m": float(
            getattr(args, "advisor_min_separation_m", 0.0)
        ),
        "ppo_actor_lr": float(getattr(args, "ppo_actor_lr", 3e-4)),
        "ppo_gamma": float(getattr(args, "ppo_gamma", 0.99)),
        "ppo_gae_lambda": float(
            getattr(args, "ppo_gae_lambda", 0.95)
        ),
        "ppo_target_kl": float(getattr(args, "ppo_target_kl", 0.0)),
        "ppo_zero_inflated_actions": int(
            getattr(args, "ppo_zero_inflated_actions", 0)
        ),
        "ppo_hover_gate_threshold": float(
            getattr(args, "ppo_hover_gate_threshold", 0.05)
        ),
        "ppo_hover_gate_temperature": float(
            getattr(args, "ppo_hover_gate_temperature", 10.0)
        ),
        "ppo_actor_final_lr": float(
            getattr(args, "ppo_actor_final_lr", 3e-4)
        ),
        "ppo_lr_decay_episodes": int(
            getattr(args, "ppo_lr_decay_episodes", 0)
        ),
        "freeze_micro": int(bool(getattr(args, "freeze_micro", False))),
        "freeze_macro": int(bool(getattr(args, "freeze_macro", False))),
        "frozen_macro_policy": str(
            getattr(args, "frozen_macro_policy", "actor")
        ),
        "train_load_min": float(getattr(args, "train_load_min", 0.0)),
        "train_load_max": float(getattr(args, "train_load_max", 0.0)),
        "episodes": int(getattr(args, "episodes", 0)),
        "macro_steps": int(getattr(args, "macro_steps", 0)),
        "micro_steps": int(getattr(args, "micro_steps", 0)),
        "eval_interval": int(getattr(args, "eval_interval", 0)),
        "eval_episodes": int(getattr(args, "eval_episodes", 0)),
        "eval_base_seed": int(getattr(args, "eval_base_seed", 1000)),
        "seed": int(getattr(args, "seed", 0)),
    }
    with open(scenario_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return scenario_path


def compute_u2u_rate_mbps(wrapper: HRL_Wrapper, raw_state: Dict[str, np.ndarray]) -> np.ndarray:
    rate_bps = wrapper.env.compute_u2u_rate_bps(
        uav_coords=raw_state["uav_coords"],
    )
    return (rate_bps / 1e6).astype(np.float32)


@dataclass
class EpisodeTrace:
    queue_backlogs: List[float] = field(default_factory=list)
    queue_backlogs_per_uav: List[np.ndarray] = field(default_factory=list)
    energy_consumed: List[float] = field(default_factory=list)
    u2g_distance_means: List[float] = field(default_factory=list)
    arrival_high: List[float] = field(default_factory=list)
    arrival_low: List[float] = field(default_factory=list)
    arrival_total_per_uav: List[np.ndarray] = field(default_factory=list)
    q_loc_high: List[float] = field(default_factory=list)
    q_tx_high: List[float] = field(default_factory=list)
    q_loc_low: List[float] = field(default_factory=list)
    q_tx_low: List[float] = field(default_factory=list)
    q_total_per_uav: List[np.ndarray] = field(default_factory=list)
    delay_total_per_uav_ms: List[np.ndarray] = field(default_factory=list)
    task_delay_source_uav_ms: List[np.ndarray] = field(default_factory=list)
    task_completed_source_count: List[np.ndarray] = field(default_factory=list)
    task_arrived_source_count: List[np.ndarray] = field(default_factory=list)
    task_delay_source_high_ms: List[np.ndarray] = field(default_factory=list)
    task_delay_source_low_ms: List[np.ndarray] = field(default_factory=list)
    task_completed_source_high_count: List[np.ndarray] = field(default_factory=list)
    task_completed_source_low_count: List[np.ndarray] = field(default_factory=list)
    task_arrived_source_high_count: List[np.ndarray] = field(default_factory=list)
    task_arrived_source_low_count: List[np.ndarray] = field(default_factory=list)
    energy_comp_per_uav: List[np.ndarray] = field(default_factory=list)
    energy_uplink_per_uav: List[np.ndarray] = field(default_factory=list)
    energy_trans_per_uav: List[np.ndarray] = field(default_factory=list)
    energy_total_per_uav: List[np.ndarray] = field(default_factory=list)
    compute_service_per_uav_mbits: List[np.ndarray] = field(default_factory=list)
    coop_flow_mbits: List[np.ndarray] = field(default_factory=list)
    tx_service_matrix_mbits: List[np.ndarray] = field(default_factory=list)
    coop_ratio: List[float] = field(default_factory=list)
    delay_mean_ms: List[float] = field(default_factory=list)
    delay_weighted_ms: List[float] = field(default_factory=list)
    delay_max_ms: List[float] = field(default_factory=list)
    task_delay_avg_ms: List[float] = field(default_factory=list)
    task_delay_weighted_ms: List[float] = field(default_factory=list)
    task_delay_high_ms: List[float] = field(default_factory=list)
    task_delay_low_ms: List[float] = field(default_factory=list)
    task_delay_sensor_ms: List[float] = field(default_factory=list)
    task_delay_controller_ms: List[float] = field(default_factory=list)
    task_delay_vision_ms: List[float] = field(default_factory=list)
    task_completed_count: List[float] = field(default_factory=list)
    task_arrived_count: List[float] = field(default_factory=list)
    task_alive_count: List[float] = field(default_factory=list)
    task_completion_rate: List[float] = field(default_factory=list)
    task_qos_violation_rate: List[float] = field(default_factory=list)
    task_holding_cost: List[float] = field(default_factory=list)
    task_wait_cost: List[float] = field(default_factory=list)
    cost_holding: List[float] = field(default_factory=list)
    cost_wait: List[float] = field(default_factory=list)
    compute_service_high_mbits: List[float] = field(default_factory=list)
    compute_service_low_mbits: List[float] = field(default_factory=list)
    compute_service_sensor_mbits: List[float] = field(default_factory=list)
    compute_service_controller_mbits: List[float] = field(default_factory=list)
    compute_service_vision_mbits: List[float] = field(default_factory=list)
    cpu_util_per_uav: List[np.ndarray] = field(default_factory=list)
    upload_delay_cost_per_uav: List[np.ndarray] = field(default_factory=list)
    upload_delay_penalty_per_uav: List[np.ndarray] = field(default_factory=list)
    pressure_imbalance: List[float] = field(default_factory=list)
    micro_cost_raw: List[float] = field(default_factory=list)
    cpu_util_mean: List[float] = field(default_factory=list)
    macro_move_distance: List[float] = field(default_factory=list)
    macro_hover_ratio: List[float] = field(default_factory=list)
    macro_action_norm: List[float] = field(default_factory=list)
    macro_u2u_rate: List[float] = field(default_factory=list)
    macro_projected_u2g: List[float] = field(default_factory=list)
    macro_avg_upload_delay_penalty: List[float] = field(default_factory=list)
    macro_position_cost: List[float] = field(default_factory=list)
    macro_position_improvement: List[float] = field(default_factory=list)
    macro_projected_comm_cost: List[float] = field(default_factory=list)
    macro_projected_comm_uplink_time: List[float] = field(default_factory=list)
    macro_projected_comm_u2u_time: List[float] = field(default_factory=list)
    macro_rewards: List[float] = field(default_factory=list)

    def record_macro_info(self, mac_info: Optional[Dict[str, float]]) -> None:
        if isinstance(mac_info, dict):
            self.energy_consumed.append(mac_info.get("E_macro_total", 0.0))
            u2g_distances = mac_info.get("u2g_distances", None)
            if u2g_distances is not None:
                self.u2g_distance_means.append(float(np.mean(u2g_distances)))
            self.macro_move_distance.append(mac_info.get("macro_move_distance_mean", 0.0))
            self.macro_hover_ratio.append(mac_info.get("macro_hover_ratio", 1.0))
            self.macro_action_norm.append(mac_info.get("macro_action_norm_mean", 0.0))
            self.macro_u2u_rate.append(mac_info.get("projected_mean_u2u_rate_mbps", 0.0))
            self.macro_projected_u2g.append(mac_info.get("projected_mean_u2g_distance_m", 0.0))
            self.macro_projected_comm_cost.append(
                mac_info.get("projected_macro_comm_cost", 0.0)
            )
            self.macro_projected_comm_uplink_time.append(
                mac_info.get("projected_comm_uplink_time_s", 0.0)
            )
            self.macro_projected_comm_u2u_time.append(
                mac_info.get("projected_comm_u2u_time_s", 0.0)
            )
            return

        self.macro_move_distance.append(0.0)
        self.macro_hover_ratio.append(1.0)
        self.macro_action_norm.append(0.0)
        self.macro_u2u_rate.append(0.0)
        self.macro_projected_u2g.append(0.0)
        self.macro_projected_comm_cost.append(0.0)
        self.macro_projected_comm_uplink_time.append(0.0)
        self.macro_projected_comm_u2u_time.append(0.0)

    def record_micro_step(
        self,
        raw_state: Optional[Dict[str, np.ndarray]],
        mic_info: Optional[Dict[str, float]],
    ) -> None:
        metrics = build_micro_step_metrics(raw_state or {}, mic_info)
        self.queue_backlogs.append(metrics["queue_length"])
        self.queue_backlogs_per_uav.append(metrics["q_total_per_uav"].copy())
        self.energy_consumed.append(
            metrics["energy_uplink"] + metrics["energy_trans"] + metrics["energy_comp"]
        )
        self.micro_cost_raw.append(metrics["micro_cost_raw"])
        self.arrival_high.append(float(np.sum(metrics["arrival_high_per_uav"])))
        self.arrival_low.append(float(np.sum(metrics["arrival_low_per_uav"])))
        self.arrival_total_per_uav.append(metrics["arrival_total_per_uav"].copy())
        self.q_loc_high.append(float(np.sum(metrics["q_loc_high_per_uav"])))
        self.q_tx_high.append(float(np.sum(metrics["q_tx_high_per_uav"])))
        self.q_loc_low.append(float(np.sum(metrics["q_loc_low_per_uav"])))
        self.q_tx_low.append(float(np.sum(metrics["q_tx_low_per_uav"])))
        self.q_total_per_uav.append(metrics["q_total_per_uav"].copy())
        self.delay_total_per_uav_ms.append(metrics["delay_total_per_uav_ms"].copy())
        self.task_delay_source_uav_ms.append(metrics["task_delay_source_uav_ms"].copy())
        self.task_completed_source_count.append(metrics["task_completed_source_count"].copy())
        self.task_arrived_source_count.append(metrics["task_arrived_source_count"].copy())
        self.task_delay_source_high_ms.append(metrics["task_delay_source_high_ms"].copy())
        self.task_delay_source_low_ms.append(metrics["task_delay_source_low_ms"].copy())
        self.task_completed_source_high_count.append(metrics["task_completed_source_high_count"].copy())
        self.task_completed_source_low_count.append(metrics["task_completed_source_low_count"].copy())
        self.task_arrived_source_high_count.append(metrics["task_arrived_source_high_count"].copy())
        self.task_arrived_source_low_count.append(metrics["task_arrived_source_low_count"].copy())
        self.energy_comp_per_uav.append(metrics["energy_comp_per_uav"].copy())
        self.energy_uplink_per_uav.append(metrics["energy_uplink_per_uav"].copy())
        self.energy_trans_per_uav.append(metrics["energy_trans_per_uav"].copy())
        self.energy_total_per_uav.append(metrics["energy_total_per_uav"].copy())
        self.compute_service_per_uav_mbits.append(metrics["compute_service_per_uav_mbits"].copy())
        self.coop_flow_mbits.append(metrics["coop_flow_mbits"].copy())
        self.tx_service_matrix_mbits.append(metrics["tx_service_matrix_mbits"].copy())
        self.coop_ratio.append(metrics["coop_ratio"])
        self.delay_mean_ms.append(metrics["delay_mean_ms"])
        self.delay_weighted_ms.append(metrics["delay_weighted_ms"])
        self.delay_max_ms.append(metrics["delay_max_ms"])
        self.task_delay_avg_ms.append(metrics["task_delay_avg_ms"])
        self.task_delay_weighted_ms.append(metrics["task_delay_weighted_ms"])
        self.task_delay_high_ms.append(metrics["task_delay_high_ms"])
        self.task_delay_low_ms.append(metrics["task_delay_low_ms"])
        self.task_delay_sensor_ms.append(metrics["task_delay_sensor_ms"])
        self.task_delay_controller_ms.append(metrics["task_delay_controller_ms"])
        self.task_delay_vision_ms.append(metrics["task_delay_vision_ms"])
        self.task_completed_count.append(metrics["task_completed_count"])
        self.task_arrived_count.append(metrics["task_arrived_count"])
        self.task_alive_count.append(metrics["task_alive_count"])
        self.task_completion_rate.append(metrics["task_completion_rate"])
        self.task_qos_violation_rate.append(metrics["task_qos_violation_rate"])
        self.task_holding_cost.append(metrics["task_holding_cost"])
        self.task_wait_cost.append(metrics["task_wait_cost"])
        self.cost_holding.append(metrics["cost_holding"])
        self.cost_wait.append(metrics["cost_wait"])
        self.compute_service_high_mbits.append(metrics["compute_service_high_mbits"])
        self.compute_service_low_mbits.append(metrics["compute_service_low_mbits"])
        self.compute_service_sensor_mbits.append(metrics["compute_service_sensor_mbits"])
        self.compute_service_controller_mbits.append(
            metrics["compute_service_controller_mbits"]
        )
        self.compute_service_vision_mbits.append(metrics["compute_service_vision_mbits"])
        self.cpu_util_mean.append(metrics["cpu_utilization_mean"])
        self.cpu_util_per_uav.append(metrics["cpu_utilization_per_uav"].copy())
        self.upload_delay_cost_per_uav.append(metrics["upload_delay_cost_per_uav"].copy())
        self.upload_delay_penalty_per_uav.append(metrics["upload_delay_penalty_per_uav"].copy())
        self.pressure_imbalance.append(metrics["pressure_imbalance"])

    def record_macro_reward(
        self,
        macro_reward: float,
        macro_reward_info: Optional[Dict[str, float]],
    ) -> None:
        self.macro_rewards.append(float(macro_reward))
        if isinstance(macro_reward_info, dict):
            self.macro_avg_upload_delay_penalty.append(
                macro_reward_info.get("avg_upload_delay_penalty", 0.0)
            )
            self.macro_position_cost.append(macro_reward_info.get("total_cost", 0.0))
            self.macro_position_improvement.append(
                macro_reward_info.get("positioning_improvement", 0.0)
            )
            return

        self.macro_avg_upload_delay_penalty.append(0.0)
        self.macro_position_cost.append(0.0)
        self.macro_position_improvement.append(0.0)

    def summary(self) -> Dict[str, float]:
        little = compute_cumulative_little_delays(
            [loc + tx for loc, tx in zip(self.q_loc_high, self.q_tx_high)],
            [loc + tx for loc, tx in zip(self.q_loc_low, self.q_tx_low)],
            self.arrival_high,
            self.arrival_low,
            self.q_total_per_uav,
            self.arrival_total_per_uav,
            len(self.q_total_per_uav[0]) if self.q_total_per_uav else 1,
        )

        return {
            "avg_queue": float(np.mean(self.queue_backlogs)) if self.queue_backlogs else 0.0,
            "total_energy": float(np.sum(self.energy_consumed)) if self.energy_consumed else 0.0,
            "avg_u2g_distance": float(np.mean(self.u2g_distance_means))
            if self.u2g_distance_means
            else 0.0,
            "avg_arrival_high": float(np.mean(self.arrival_high)) if self.arrival_high else 0.0,
            "avg_arrival_low": float(np.mean(self.arrival_low)) if self.arrival_low else 0.0,
            "avg_q_loc_high": float(np.mean(self.q_loc_high)) if self.q_loc_high else 0.0,
            "avg_q_tx_high": float(np.mean(self.q_tx_high)) if self.q_tx_high else 0.0,
            "avg_q_loc_low": float(np.mean(self.q_loc_low)) if self.q_loc_low else 0.0,
            "avg_q_tx_low": float(np.mean(self.q_tx_low)) if self.q_tx_low else 0.0,
            "avg_micro_cost_raw": float(np.mean(self.micro_cost_raw))
            if self.micro_cost_raw
            else 0.0,
            "avg_cpu_util_mean": float(np.mean(self.cpu_util_mean))
            if self.cpu_util_mean
            else 0.0,
            "avg_delay_mean_ms": float(little["delay_mean_ms"]),
            "avg_delay_weighted_ms": float(little["delay_weighted_ms"]),
            "avg_delay_max_ms": float(little["delay_max_ms"]),
            "avg_task_delay_ms": float(self.task_delay_avg_ms[-1])
            if self.task_delay_avg_ms
            else 0.0,
            "avg_task_delay_weighted_ms": float(self.task_delay_weighted_ms[-1])
            if self.task_delay_weighted_ms
            else 0.0,
            "task_arrived_count": float(self.task_arrived_count[-1])
            if self.task_arrived_count
            else 0.0,
            "task_completed_count": float(self.task_completed_count[-1])
            if self.task_completed_count
            else 0.0,
            "task_alive_count": float(self.task_alive_count[-1])
            if self.task_alive_count
            else 0.0,
            "task_completion_rate": float(self.task_completion_rate[-1])
            if self.task_completion_rate
            else 0.0,
            "task_qos_violation_rate": float(self.task_qos_violation_rate[-1])
            if self.task_qos_violation_rate
            else 0.0,
            "avg_cost_holding": float(np.mean(self.cost_holding))
            if self.cost_holding
            else 0.0,
            "avg_cost_wait": float(np.mean(self.cost_wait))
            if self.cost_wait
            else 0.0,
            "avg_pressure_imbalance": float(np.mean(self.pressure_imbalance))
            if self.pressure_imbalance
            else 0.0,
            "sum_compute_service_high_mbits": float(np.sum(self.compute_service_high_mbits))
            if self.compute_service_high_mbits
            else 0.0,
            "sum_compute_service_low_mbits": float(np.sum(self.compute_service_low_mbits))
            if self.compute_service_low_mbits
            else 0.0,
            "sum_compute_service_sensor_mbits": float(np.sum(self.compute_service_sensor_mbits))
            if self.compute_service_sensor_mbits
            else 0.0,
            "sum_compute_service_controller_mbits": float(np.sum(self.compute_service_controller_mbits))
            if self.compute_service_controller_mbits
            else 0.0,
            "sum_compute_service_vision_mbits": float(np.sum(self.compute_service_vision_mbits))
            if self.compute_service_vision_mbits
            else 0.0,
            "avg_coop_ratio": float(self.coop_ratio[-1])
            if self.coop_ratio
            else 0.0,
            "avg_macro_move_distance": float(np.mean(self.macro_move_distance))
            if self.macro_move_distance
            else 0.0,
            "avg_macro_hover_ratio": float(np.mean(self.macro_hover_ratio))
            if self.macro_hover_ratio
            else 0.0,
            "avg_macro_action_norm": float(np.mean(self.macro_action_norm))
            if self.macro_action_norm
            else 0.0,
            "avg_macro_u2u_rate": float(np.mean(self.macro_u2u_rate))
            if self.macro_u2u_rate
            else 0.0,
            "avg_macro_position_cost": float(np.mean(self.macro_position_cost))
            if self.macro_position_cost
            else 0.0,
            "avg_macro_position_improvement": float(np.mean(self.macro_position_improvement))
            if self.macro_position_improvement
            else 0.0,
        }


@dataclass
class EvaluationHistory:
    steps: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    J: List[float] = field(default_factory=list)
    delay_objective: List[float] = field(default_factory=list)
    legacy_scores: List[float] = field(default_factory=list)
    q_high: List[float] = field(default_factory=list)
    q_low: List[float] = field(default_factory=list)
    energy: List[float] = field(default_factory=list)
    average_power_w: List[float] = field(default_factory=list)
    average_power_std_w: List[float] = field(default_factory=list)
    average_power_sem_w: List[float] = field(default_factory=list)
    power_sample_count: List[float] = field(default_factory=list)
    average_power_ucb95_w: List[float] = field(default_factory=list)
    average_power_ci95_low_w: List[float] = field(default_factory=list)
    average_power_ci95_high_w: List[float] = field(default_factory=list)
    power_budget_w: List[float] = field(default_factory=list)
    power_violation_w: List[float] = field(default_factory=list)
    power_violation_ratio: List[float] = field(default_factory=list)
    energy_feasible: List[float] = field(default_factory=list)
    power_ucb95_violation_w: List[float] = field(default_factory=list)
    energy_feasible_ucb95: List[float] = field(default_factory=list)
    average_battery_power_per_uav_w: List[np.ndarray] = field(
        default_factory=list
    )
    battery_power_ucb95_per_uav_w: List[np.ndarray] = field(
        default_factory=list
    )
    uav_power_budgets_w: List[np.ndarray] = field(default_factory=list)
    battery_power_violation_per_uav_w: List[np.ndarray] = field(
        default_factory=list
    )
    battery_power_ucb95_violation_per_uav_w: List[np.ndarray] = field(
        default_factory=list
    )
    max_battery_power_violation_w: List[float] = field(default_factory=list)
    max_battery_power_ucb95_violation_w: List[float] = field(
        default_factory=list
    )
    energy_propulsion_episode: List[float] = field(default_factory=list)
    energy_uplink_episode: List[float] = field(default_factory=list)
    energy_u2u_episode: List[float] = field(default_factory=list)
    energy_compute_episode: List[float] = field(default_factory=list)
    energy_component_residual: List[float] = field(default_factory=list)
    micro_decision_count: List[float] = field(default_factory=list)
    micro_decision_seconds: List[float] = field(default_factory=list)
    micro_decision_mean_ms: List[float] = field(default_factory=list)
    micro_decision_p95_ms: List[float] = field(default_factory=list)
    residual_gate_mean: List[float] = field(default_factory=list)
    residual_gate_std: List[float] = field(default_factory=list)
    residual_gate_q10: List[float] = field(default_factory=list)
    residual_gate_q90: List[float] = field(default_factory=list)
    u2g: List[float] = field(default_factory=list)
    cpu_util: List[float] = field(default_factory=list)
    cpu_saturation: List[float] = field(default_factory=list)
    cpu_high_util: List[float] = field(default_factory=list)
    cpu_low_util: List[float] = field(default_factory=list)
    f_high_ghz: List[float] = field(default_factory=list)
    f_low_ghz: List[float] = field(default_factory=list)
    f_total_ghz: List[float] = field(default_factory=list)
    delay_high: List[float] = field(default_factory=list)
    delay_low: List[float] = field(default_factory=list)
    delay_mean_ms: List[float] = field(default_factory=list)
    delay_weighted_ms: List[float] = field(default_factory=list)
    delay_max_ms: List[float] = field(default_factory=list)
    task_delay_avg_ms: List[float] = field(default_factory=list)
    task_delay_weighted_ms: List[float] = field(default_factory=list)
    task_delay_high_ms: List[float] = field(default_factory=list)
    task_delay_low_ms: List[float] = field(default_factory=list)
    task_delay_sensor_ms: List[float] = field(default_factory=list)
    task_delay_controller_ms: List[float] = field(default_factory=list)
    task_delay_vision_ms: List[float] = field(default_factory=list)
    task_completed_count: List[float] = field(default_factory=list)
    task_arrived_count: List[float] = field(default_factory=list)
    task_completion_rate: List[float] = field(default_factory=list)
    task_qos_violation_rate: List[float] = field(default_factory=list)
    compute_service_high_mbits: List[float] = field(default_factory=list)
    compute_service_low_mbits: List[float] = field(default_factory=list)
    compute_service_sensor_mbits: List[float] = field(default_factory=list)
    compute_service_controller_mbits: List[float] = field(default_factory=list)
    compute_service_vision_mbits: List[float] = field(default_factory=list)
    q_total_per_uav: List[np.ndarray] = field(default_factory=list)
    arrival_total_per_uav: List[np.ndarray] = field(default_factory=list)
    cpu_util_per_uav: List[np.ndarray] = field(default_factory=list)
    delay_total_per_uav_ms: List[np.ndarray] = field(default_factory=list)
    upload_delay_cost_per_uav: List[np.ndarray] = field(default_factory=list)
    upload_delay_penalty_per_uav: List[np.ndarray] = field(default_factory=list)
    task_delay_source_uav_ms: List[np.ndarray] = field(default_factory=list)
    task_completed_source_count: List[np.ndarray] = field(default_factory=list)
    task_arrived_source_count: List[np.ndarray] = field(default_factory=list)
    task_delay_source_high_ms: List[np.ndarray] = field(default_factory=list)
    task_delay_source_low_ms: List[np.ndarray] = field(default_factory=list)
    task_completed_source_high_count: List[np.ndarray] = field(default_factory=list)
    task_completed_source_low_count: List[np.ndarray] = field(default_factory=list)
    task_arrived_source_high_count: List[np.ndarray] = field(default_factory=list)
    task_arrived_source_low_count: List[np.ndarray] = field(default_factory=list)
    energy_total_episode_per_uav: List[np.ndarray] = field(default_factory=list)
    energy_comp_episode_per_uav: List[np.ndarray] = field(default_factory=list)
    energy_uplink_episode_per_uav: List[np.ndarray] = field(default_factory=list)
    energy_trans_episode_per_uav: List[np.ndarray] = field(default_factory=list)
    energy_macro_episode_per_uav: List[np.ndarray] = field(default_factory=list)
    energy_total_episode: List[float] = field(default_factory=list)
    compute_service_per_uav_mbits: List[np.ndarray] = field(default_factory=list)
    coop_flow_mbits: List[np.ndarray] = field(default_factory=list)
    tx_service_matrix_mbits: List[np.ndarray] = field(default_factory=list)
    split_ratios_high: List[np.ndarray] = field(default_factory=list)
    split_ratios_low: List[np.ndarray] = field(default_factory=list)
    pressure_per_uav: List[np.ndarray] = field(default_factory=list)
    pressure_imbalance: List[float] = field(default_factory=list)
    coop_ratio: List[float] = field(default_factory=list)

    def append(self, step: int, metrics: Dict[str, float]) -> None:
        self.steps.append(int(step))
        self.scores.append(float(metrics["score"]))
        self.J.append(float(metrics.get("J", 0.0)))
        self.delay_objective.append(
            float(metrics.get("delay_objective", 0.0))
        )
        self.legacy_scores.append(
            float(metrics.get("legacy_score", metrics["score"]))
        )
        self.q_high.append(float(metrics["q_high"]))
        self.q_low.append(float(metrics["q_low"]))
        self.energy.append(float(metrics["energy"]))
        self.average_power_w.append(float(metrics.get("average_power_w", 0.0)))
        self.average_power_std_w.append(
            float(metrics.get("average_power_std_w", 0.0))
        )
        self.average_power_sem_w.append(
            float(metrics.get("average_power_sem_w", 0.0))
        )
        self.power_sample_count.append(
            float(metrics.get("power_sample_count", 0.0))
        )
        self.average_power_ucb95_w.append(
            float(metrics.get("average_power_ucb95_w", float("inf")))
        )
        self.average_power_ci95_low_w.append(
            float(metrics.get("average_power_ci95_low_w", 0.0))
        )
        self.average_power_ci95_high_w.append(
            float(metrics.get("average_power_ci95_high_w", 0.0))
        )
        self.power_budget_w.append(float(metrics.get("power_budget_w", 0.0)))
        self.power_violation_w.append(
            float(metrics.get("power_violation_w", 0.0))
        )
        self.power_violation_ratio.append(
            float(metrics.get("power_violation_ratio", 0.0))
        )
        self.energy_feasible.append(
            float(metrics.get("energy_feasible", 1.0))
        )
        self.power_ucb95_violation_w.append(
            float(metrics.get("power_ucb95_violation_w", float("inf")))
        )
        self.energy_feasible_ucb95.append(
            float(metrics.get("energy_feasible_ucb95", 0.0))
        )
        self.average_battery_power_per_uav_w.append(
            np.asarray(
                metrics.get("average_battery_power_per_uav_w", []),
                dtype=np.float32,
            )
        )
        self.battery_power_ucb95_per_uav_w.append(
            np.asarray(
                metrics.get("battery_power_ucb95_per_uav_w", []),
                dtype=np.float32,
            )
        )
        self.uav_power_budgets_w.append(
            np.asarray(
                metrics.get("uav_power_budgets_w", []),
                dtype=np.float32,
            )
        )
        self.battery_power_violation_per_uav_w.append(
            np.asarray(
                metrics.get("battery_power_violation_per_uav_w", []),
                dtype=np.float32,
            )
        )
        self.battery_power_ucb95_violation_per_uav_w.append(
            np.asarray(
                metrics.get(
                    "battery_power_ucb95_violation_per_uav_w",
                    [],
                ),
                dtype=np.float32,
            )
        )
        self.max_battery_power_violation_w.append(
            float(metrics.get("max_battery_power_violation_w", 0.0))
        )
        self.max_battery_power_ucb95_violation_w.append(
            float(
                metrics.get(
                    "max_battery_power_ucb95_violation_w",
                    float("inf"),
                )
            )
        )
        self.energy_propulsion_episode.append(
            float(metrics.get("energy_propulsion_episode", 0.0))
        )
        self.energy_uplink_episode.append(
            float(metrics.get("energy_uplink_episode", 0.0))
        )
        self.energy_u2u_episode.append(
            float(metrics.get("energy_u2u_episode", 0.0))
        )
        self.energy_compute_episode.append(
            float(metrics.get("energy_compute_episode", 0.0))
        )
        self.energy_component_residual.append(
            float(metrics.get("energy_component_residual", 0.0))
        )
        self.micro_decision_count.append(
            float(metrics.get("micro_decision_count", 0.0))
        )
        self.micro_decision_seconds.append(
            float(metrics.get("micro_decision_seconds", 0.0))
        )
        self.micro_decision_mean_ms.append(
            float(metrics.get("micro_decision_mean_ms", 0.0))
        )
        self.micro_decision_p95_ms.append(
            float(metrics.get("micro_decision_p95_ms", 0.0))
        )
        self.residual_gate_mean.append(
            float(metrics.get("residual_gate_mean", 1.0))
        )
        self.residual_gate_std.append(
            float(metrics.get("residual_gate_std", 0.0))
        )
        self.residual_gate_q10.append(
            float(metrics.get("residual_gate_q10", 1.0))
        )
        self.residual_gate_q90.append(
            float(metrics.get("residual_gate_q90", 1.0))
        )
        self.u2g.append(float(metrics["u2g"]))
        self.cpu_util.append(float(metrics["cpu_util"]))
        self.cpu_saturation.append(float(metrics.get("cpu_saturation", 0.0)))
        self.cpu_high_util.append(float(metrics.get("cpu_high_util", 0.0)))
        self.cpu_low_util.append(float(metrics.get("cpu_low_util", 0.0)))
        self.f_high_ghz.append(float(metrics.get("f_high_ghz", 0.0)))
        self.f_low_ghz.append(float(metrics.get("f_low_ghz", 0.0)))
        self.f_total_ghz.append(float(metrics.get("f_total_ghz", 0.0)))
        self.delay_high.append(float(metrics["delay_high"]))
        self.delay_low.append(float(metrics["delay_low"]))
        self.delay_mean_ms.append(float(metrics.get("delay_mean_ms", 0.0)))
        self.delay_weighted_ms.append(float(metrics.get("delay_weighted_ms", 0.0)))
        self.delay_max_ms.append(float(metrics.get("delay_max_ms", 0.0)))
        self.task_delay_avg_ms.append(float(metrics.get("task_delay_avg_ms", 0.0)))
        self.task_delay_weighted_ms.append(float(metrics.get("task_delay_weighted_ms", 0.0)))
        self.task_delay_high_ms.append(float(metrics.get("task_delay_high_ms", 0.0)))
        self.task_delay_low_ms.append(float(metrics.get("task_delay_low_ms", 0.0)))
        self.task_delay_sensor_ms.append(float(metrics.get("task_delay_sensor_ms", 0.0)))
        self.task_delay_controller_ms.append(float(metrics.get("task_delay_controller_ms", 0.0)))
        self.task_delay_vision_ms.append(float(metrics.get("task_delay_vision_ms", 0.0)))
        self.task_completed_count.append(float(metrics.get("task_completed_count", 0.0)))
        self.task_arrived_count.append(float(metrics.get("task_arrived_count", 0.0)))
        self.task_completion_rate.append(float(metrics.get("task_completion_rate", 0.0)))
        self.task_qos_violation_rate.append(float(metrics.get("task_qos_violation_rate", 0.0)))
        self.compute_service_high_mbits.append(float(metrics.get("compute_service_high_mbits", 0.0)))
        self.compute_service_low_mbits.append(float(metrics.get("compute_service_low_mbits", 0.0)))
        self.compute_service_sensor_mbits.append(float(metrics.get("compute_service_sensor_mbits", 0.0)))
        self.compute_service_controller_mbits.append(float(metrics.get("compute_service_controller_mbits", 0.0)))
        self.compute_service_vision_mbits.append(float(metrics.get("compute_service_vision_mbits", 0.0)))
        if "q_total_per_uav" in metrics:
            self.q_total_per_uav.append(np.asarray(metrics["q_total_per_uav"], dtype=np.float32))
        if "arrival_total_per_uav" in metrics:
            self.arrival_total_per_uav.append(np.asarray(metrics["arrival_total_per_uav"], dtype=np.float32))
        if "cpu_util_per_uav" in metrics:
            self.cpu_util_per_uav.append(np.asarray(metrics["cpu_util_per_uav"], dtype=np.float32))
        if "delay_total_per_uav_ms" in metrics:
            self.delay_total_per_uav_ms.append(np.asarray(metrics["delay_total_per_uav_ms"], dtype=np.float32))
        if "upload_delay_cost_per_uav" in metrics:
            self.upload_delay_cost_per_uav.append(np.asarray(metrics["upload_delay_cost_per_uav"], dtype=np.float32))
        if "upload_delay_penalty_per_uav" in metrics:
            self.upload_delay_penalty_per_uav.append(np.asarray(metrics["upload_delay_penalty_per_uav"], dtype=np.float32))
        if "task_delay_source_uav_ms" in metrics:
            self.task_delay_source_uav_ms.append(np.asarray(metrics["task_delay_source_uav_ms"], dtype=np.float32))
        if "task_completed_source_count" in metrics:
            self.task_completed_source_count.append(np.asarray(metrics["task_completed_source_count"], dtype=np.float32))
        if "task_arrived_source_count" in metrics:
            self.task_arrived_source_count.append(np.asarray(metrics["task_arrived_source_count"], dtype=np.float32))
        if "task_delay_source_high_ms" in metrics:
            self.task_delay_source_high_ms.append(np.asarray(metrics["task_delay_source_high_ms"], dtype=np.float32))
        if "task_delay_source_low_ms" in metrics:
            self.task_delay_source_low_ms.append(np.asarray(metrics["task_delay_source_low_ms"], dtype=np.float32))
        if "task_completed_source_high_count" in metrics:
            self.task_completed_source_high_count.append(np.asarray(metrics["task_completed_source_high_count"], dtype=np.float32))
        if "task_completed_source_low_count" in metrics:
            self.task_completed_source_low_count.append(np.asarray(metrics["task_completed_source_low_count"], dtype=np.float32))
        if "task_arrived_source_high_count" in metrics:
            self.task_arrived_source_high_count.append(np.asarray(metrics["task_arrived_source_high_count"], dtype=np.float32))
        if "task_arrived_source_low_count" in metrics:
            self.task_arrived_source_low_count.append(np.asarray(metrics["task_arrived_source_low_count"], dtype=np.float32))
        if "energy_total_episode_per_uav" in metrics:
            self.energy_total_episode_per_uav.append(np.asarray(metrics["energy_total_episode_per_uav"], dtype=np.float32))
        if "energy_comp_episode_per_uav" in metrics:
            self.energy_comp_episode_per_uav.append(np.asarray(metrics["energy_comp_episode_per_uav"], dtype=np.float32))
        if "energy_uplink_episode_per_uav" in metrics:
            self.energy_uplink_episode_per_uav.append(np.asarray(metrics["energy_uplink_episode_per_uav"], dtype=np.float32))
        if "energy_trans_episode_per_uav" in metrics:
            self.energy_trans_episode_per_uav.append(np.asarray(metrics["energy_trans_episode_per_uav"], dtype=np.float32))
        if "energy_macro_episode_per_uav" in metrics:
            self.energy_macro_episode_per_uav.append(np.asarray(metrics["energy_macro_episode_per_uav"], dtype=np.float32))
        self.energy_total_episode.append(float(metrics.get("energy_total_episode", 0.0)))
        if "compute_service_per_uav_mbits" in metrics:
            self.compute_service_per_uav_mbits.append(np.asarray(metrics["compute_service_per_uav_mbits"], dtype=np.float32))
        if "coop_flow_mbits" in metrics:
            self.coop_flow_mbits.append(np.asarray(metrics["coop_flow_mbits"], dtype=np.float32))
        if "tx_service_matrix_mbits" in metrics:
            self.tx_service_matrix_mbits.append(np.asarray(metrics["tx_service_matrix_mbits"], dtype=np.float32))
        if "split_ratios_high" in metrics:
            self.split_ratios_high.append(np.asarray(metrics["split_ratios_high"], dtype=np.float32))
        if "split_ratios_low" in metrics:
            self.split_ratios_low.append(np.asarray(metrics["split_ratios_low"], dtype=np.float32))
        if "pressure_per_uav" in metrics:
            self.pressure_per_uav.append(np.asarray(metrics["pressure_per_uav"], dtype=np.float32))
        self.pressure_imbalance.append(float(metrics.get("pressure_imbalance", 0.0)))
        self.coop_ratio.append(float(metrics.get("coop_ratio", 0.0)))

    def save_csv(self, save_path: str) -> None:
        with open(save_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "episode",
                    "score",
                    "J",
                    "delay_objective",
                    "legacy_score",
                    "q_high",
                    "q_low",
                    "energy",
                    "average_power_w",
                    "average_power_std_w",
                    "average_power_sem_w",
                    "power_sample_count",
                    "average_power_ucb95_w",
                    "average_power_ci95_low_w",
                    "average_power_ci95_high_w",
                    "power_budget_w",
                    "power_violation_w",
                    "power_violation_ratio",
                    "energy_feasible",
                    "power_ucb95_violation_w",
                    "energy_feasible_ucb95",
                    "average_battery_power_per_uav_w",
                    "battery_power_ucb95_per_uav_w",
                    "uav_power_budgets_w",
                    "battery_power_violation_per_uav_w",
                    "battery_power_ucb95_violation_per_uav_w",
                    "max_battery_power_violation_w",
                    "max_battery_power_ucb95_violation_w",
                    "energy_total_episode",
                    "energy_propulsion_episode",
                    "energy_uplink_episode",
                    "energy_u2u_episode",
                    "energy_compute_episode",
                    "energy_component_residual",
                    "micro_decision_count",
                    "micro_decision_seconds",
                    "micro_decision_mean_ms",
                    "micro_decision_p95_ms",
                    "residual_gate_mean",
                    "residual_gate_std",
                    "residual_gate_q10",
                    "residual_gate_q90",
                    "u2g",
                    "cpu_util",
                    "cpu_saturation",
                    "cpu_high_util",
                    "cpu_low_util",
                    "f_high_ghz",
                    "f_low_ghz",
                    "f_total_ghz",
                    "delay_high_ms",
                    "delay_low_ms",
                    "delay_mean_ms",
                    "delay_weighted_ms",
                    "delay_max_ms",
                    "task_delay_avg_ms",
                    "task_delay_weighted_ms",
                    "task_delay_high_ms",
                    "task_delay_low_ms",
                    "task_delay_sensor_ms",
                    "task_delay_controller_ms",
                    "task_delay_vision_ms",
                    "task_completed_count",
                    "task_arrived_count",
                    "task_completion_rate",
                    "task_qos_violation_rate",
                    "pressure_imbalance",
                    "coop_ratio",
                    "compute_service_high_mbits",
                    "compute_service_low_mbits",
                    "compute_service_sensor_mbits",
                    "compute_service_controller_mbits",
                    "compute_service_vision_mbits",
                ]
            )
            for idx in range(len(self.steps)):
                writer.writerow(
                    [
                        self.steps[idx],
                        self.scores[idx],
                        self.J[idx],
                        self.delay_objective[idx],
                        self.legacy_scores[idx],
                        self.q_high[idx],
                        self.q_low[idx],
                        self.energy[idx],
                        self.average_power_w[idx],
                        self.average_power_std_w[idx],
                        self.average_power_sem_w[idx],
                        self.power_sample_count[idx],
                        self.average_power_ucb95_w[idx],
                        self.average_power_ci95_low_w[idx],
                        self.average_power_ci95_high_w[idx],
                        self.power_budget_w[idx],
                        self.power_violation_w[idx],
                        self.power_violation_ratio[idx],
                        self.energy_feasible[idx],
                        self.power_ucb95_violation_w[idx],
                        self.energy_feasible_ucb95[idx],
                        json.dumps(
                            self.average_battery_power_per_uav_w[idx].tolist(),
                            separators=(",", ":"),
                        ),
                        json.dumps(
                            self.battery_power_ucb95_per_uav_w[idx].tolist(),
                            separators=(",", ":"),
                        ),
                        json.dumps(
                            self.uav_power_budgets_w[idx].tolist(),
                            separators=(",", ":"),
                        ),
                        json.dumps(
                            self.battery_power_violation_per_uav_w[idx].tolist(),
                            separators=(",", ":"),
                        ),
                        json.dumps(
                            self.battery_power_ucb95_violation_per_uav_w[
                                idx
                            ].tolist(),
                            separators=(",", ":"),
                        ),
                        self.max_battery_power_violation_w[idx],
                        self.max_battery_power_ucb95_violation_w[idx],
                        self.energy_total_episode[idx],
                        self.energy_propulsion_episode[idx],
                        self.energy_uplink_episode[idx],
                        self.energy_u2u_episode[idx],
                        self.energy_compute_episode[idx],
                        self.energy_component_residual[idx],
                        self.micro_decision_count[idx],
                        self.micro_decision_seconds[idx],
                        self.micro_decision_mean_ms[idx],
                        self.micro_decision_p95_ms[idx],
                        self.residual_gate_mean[idx],
                        self.residual_gate_std[idx],
                        self.residual_gate_q10[idx],
                        self.residual_gate_q90[idx],
                        self.u2g[idx],
                        self.cpu_util[idx],
                        self.cpu_saturation[idx],
                        self.cpu_high_util[idx],
                        self.cpu_low_util[idx],
                        self.f_high_ghz[idx],
                        self.f_low_ghz[idx],
                        self.f_total_ghz[idx],
                        self.delay_high[idx],
                        self.delay_low[idx],
                        self.delay_mean_ms[idx],
                        self.delay_weighted_ms[idx],
                        self.delay_max_ms[idx],
                        self.task_delay_avg_ms[idx],
                        self.task_delay_weighted_ms[idx],
                        self.task_delay_high_ms[idx],
                        self.task_delay_low_ms[idx],
                        self.task_delay_sensor_ms[idx],
                        self.task_delay_controller_ms[idx],
                        self.task_delay_vision_ms[idx],
                        self.task_completed_count[idx],
                        self.task_arrived_count[idx],
                        self.task_completion_rate[idx],
                        self.task_qos_violation_rate[idx],
                        self.pressure_imbalance[idx],
                        self.coop_ratio[idx],
                        self.compute_service_high_mbits[idx],
                        self.compute_service_low_mbits[idx],
                        self.compute_service_sensor_mbits[idx],
                        self.compute_service_controller_mbits[idx],
                        self.compute_service_vision_mbits[idx],
                    ]
                )

    def save_detail_npz(self, save_path: str) -> None:
        payload = {
            "steps": np.asarray(self.steps, dtype=np.int32),
            "scores": np.asarray(self.scores, dtype=np.float32),
            "J": np.asarray(self.J, dtype=np.float32),
            "legacy_scores": np.asarray(self.legacy_scores, dtype=np.float32),
            "delay_high_ms": np.asarray(self.delay_high, dtype=np.float32),
            "delay_low_ms": np.asarray(self.delay_low, dtype=np.float32),
            "delay_mean_ms": np.asarray(self.delay_mean_ms, dtype=np.float32),
            "delay_weighted_ms": np.asarray(self.delay_weighted_ms, dtype=np.float32),
            "delay_max_ms": np.asarray(self.delay_max_ms, dtype=np.float32),
            "task_delay_avg_ms": np.asarray(self.task_delay_avg_ms, dtype=np.float32),
            "task_delay_weighted_ms": np.asarray(self.task_delay_weighted_ms, dtype=np.float32),
            "task_delay_high_ms": np.asarray(self.task_delay_high_ms, dtype=np.float32),
            "task_delay_low_ms": np.asarray(self.task_delay_low_ms, dtype=np.float32),
            "task_delay_sensor_ms": np.asarray(self.task_delay_sensor_ms, dtype=np.float32),
            "task_delay_controller_ms": np.asarray(self.task_delay_controller_ms, dtype=np.float32),
            "task_delay_vision_ms": np.asarray(self.task_delay_vision_ms, dtype=np.float32),
            "task_completed_count": np.asarray(self.task_completed_count, dtype=np.float32),
            "task_arrived_count": np.asarray(self.task_arrived_count, dtype=np.float32),
            "task_completion_rate": np.asarray(self.task_completion_rate, dtype=np.float32),
            "task_qos_violation_rate": np.asarray(self.task_qos_violation_rate, dtype=np.float32),
            "compute_service_high_mbits": np.asarray(self.compute_service_high_mbits, dtype=np.float32),
            "compute_service_low_mbits": np.asarray(self.compute_service_low_mbits, dtype=np.float32),
            "compute_service_sensor_mbits": np.asarray(self.compute_service_sensor_mbits, dtype=np.float32),
            "compute_service_controller_mbits": np.asarray(self.compute_service_controller_mbits, dtype=np.float32),
            "compute_service_vision_mbits": np.asarray(self.compute_service_vision_mbits, dtype=np.float32),
            "service_energy_per_micro_step_j": np.asarray(self.energy, dtype=np.float32),
            "residual_gate_mean": np.asarray(
                self.residual_gate_mean, dtype=np.float32
            ),
            "residual_gate_std": np.asarray(
                self.residual_gate_std, dtype=np.float32
            ),
            "residual_gate_q10": np.asarray(
                self.residual_gate_q10, dtype=np.float32
            ),
            "residual_gate_q90": np.asarray(
                self.residual_gate_q90, dtype=np.float32
            ),
            "average_power_w": np.asarray(self.average_power_w, dtype=np.float32),
            "average_power_std_w": np.asarray(
                self.average_power_std_w, dtype=np.float32
            ),
            "average_power_sem_w": np.asarray(
                self.average_power_sem_w, dtype=np.float32
            ),
            "power_sample_count": np.asarray(
                self.power_sample_count, dtype=np.float32
            ),
            "average_power_ucb95_w": np.asarray(
                self.average_power_ucb95_w, dtype=np.float32
            ),
            "average_power_ci95_low_w": np.asarray(
                self.average_power_ci95_low_w, dtype=np.float32
            ),
            "average_power_ci95_high_w": np.asarray(
                self.average_power_ci95_high_w, dtype=np.float32
            ),
            "power_budget_w": np.asarray(self.power_budget_w, dtype=np.float32),
            "power_violation_w": np.asarray(self.power_violation_w, dtype=np.float32),
            "power_violation_ratio": np.asarray(self.power_violation_ratio, dtype=np.float32),
            "energy_feasible": np.asarray(self.energy_feasible, dtype=np.float32),
            "power_ucb95_violation_w": np.asarray(
                self.power_ucb95_violation_w, dtype=np.float32
            ),
            "energy_feasible_ucb95": np.asarray(
                self.energy_feasible_ucb95, dtype=np.float32
            ),
            "max_battery_power_violation_w": np.asarray(
                self.max_battery_power_violation_w,
                dtype=np.float32,
            ),
            "max_battery_power_ucb95_violation_w": np.asarray(
                self.max_battery_power_ucb95_violation_w,
                dtype=np.float32,
            ),
            "delay_objective": np.asarray(self.delay_objective, dtype=np.float32),
            "energy_total_episode": np.asarray(self.energy_total_episode, dtype=np.float32),
            "energy_propulsion_episode": np.asarray(self.energy_propulsion_episode, dtype=np.float32),
            "energy_uplink_episode": np.asarray(self.energy_uplink_episode, dtype=np.float32),
            "energy_u2u_episode": np.asarray(self.energy_u2u_episode, dtype=np.float32),
            "energy_compute_episode": np.asarray(self.energy_compute_episode, dtype=np.float32),
            "energy_component_residual": np.asarray(self.energy_component_residual, dtype=np.float32),
            "micro_decision_count": np.asarray(self.micro_decision_count, dtype=np.float32),
            "micro_decision_seconds": np.asarray(self.micro_decision_seconds, dtype=np.float32),
            "micro_decision_mean_ms": np.asarray(self.micro_decision_mean_ms, dtype=np.float32),
            "micro_decision_p95_ms": np.asarray(self.micro_decision_p95_ms, dtype=np.float32),
            "coop_ratio": np.asarray(self.coop_ratio, dtype=np.float32),
            "cpu_saturation": np.asarray(self.cpu_saturation, dtype=np.float32),
            "cpu_high_util": np.asarray(self.cpu_high_util, dtype=np.float32),
            "cpu_low_util": np.asarray(self.cpu_low_util, dtype=np.float32),
            "f_high_ghz": np.asarray(self.f_high_ghz, dtype=np.float32),
            "f_low_ghz": np.asarray(self.f_low_ghz, dtype=np.float32),
            "f_total_ghz": np.asarray(self.f_total_ghz, dtype=np.float32),
            "pressure_imbalance": np.asarray(self.pressure_imbalance, dtype=np.float32),
        }
        if self.q_total_per_uav:
            payload["q_total_per_uav"] = np.stack(self.q_total_per_uav, axis=0)
        if (
            self.average_battery_power_per_uav_w
            and self.average_battery_power_per_uav_w[0].size > 0
        ):
            payload["average_battery_power_per_uav_w"] = np.stack(
                self.average_battery_power_per_uav_w,
                axis=0,
            )
            payload["battery_power_ucb95_per_uav_w"] = np.stack(
                self.battery_power_ucb95_per_uav_w,
                axis=0,
            )
            payload["uav_power_budgets_w"] = np.stack(
                self.uav_power_budgets_w,
                axis=0,
            )
            payload["battery_power_violation_per_uav_w"] = np.stack(
                self.battery_power_violation_per_uav_w,
                axis=0,
            )
            payload[
                "battery_power_ucb95_violation_per_uav_w"
            ] = np.stack(
                self.battery_power_ucb95_violation_per_uav_w,
                axis=0,
            )
        if self.arrival_total_per_uav:
            payload["arrival_total_per_uav"] = np.stack(self.arrival_total_per_uav, axis=0)
        if self.cpu_util_per_uav:
            payload["cpu_util_per_uav"] = np.stack(self.cpu_util_per_uav, axis=0)
        if self.delay_total_per_uav_ms:
            payload["delay_total_per_uav_ms"] = np.stack(self.delay_total_per_uav_ms, axis=0)
        if self.upload_delay_cost_per_uav:
            payload["upload_delay_cost_per_uav"] = np.stack(self.upload_delay_cost_per_uav, axis=0)
        if self.upload_delay_penalty_per_uav:
            payload["upload_delay_penalty_per_uav"] = np.stack(self.upload_delay_penalty_per_uav, axis=0)
        if self.task_delay_source_uav_ms:
            payload["task_delay_source_uav_ms"] = np.stack(self.task_delay_source_uav_ms, axis=0)
        if self.task_completed_source_count:
            payload["task_completed_source_count"] = np.stack(self.task_completed_source_count, axis=0)
        if self.task_arrived_source_count:
            payload["task_arrived_source_count"] = np.stack(self.task_arrived_source_count, axis=0)
        if self.task_delay_source_high_ms:
            payload["task_delay_source_high_ms"] = np.stack(self.task_delay_source_high_ms, axis=0)
        if self.task_delay_source_low_ms:
            payload["task_delay_source_low_ms"] = np.stack(self.task_delay_source_low_ms, axis=0)
        if self.task_completed_source_high_count:
            payload["task_completed_source_high_count"] = np.stack(self.task_completed_source_high_count, axis=0)
        if self.task_completed_source_low_count:
            payload["task_completed_source_low_count"] = np.stack(self.task_completed_source_low_count, axis=0)
        if self.task_arrived_source_high_count:
            payload["task_arrived_source_high_count"] = np.stack(self.task_arrived_source_high_count, axis=0)
        if self.task_arrived_source_low_count:
            payload["task_arrived_source_low_count"] = np.stack(self.task_arrived_source_low_count, axis=0)
        if self.energy_total_episode_per_uav:
            payload["energy_total_episode_per_uav"] = np.stack(self.energy_total_episode_per_uav, axis=0)
        if self.energy_comp_episode_per_uav:
            payload["energy_comp_episode_per_uav"] = np.stack(self.energy_comp_episode_per_uav, axis=0)
        if self.energy_uplink_episode_per_uav:
            payload["energy_uplink_episode_per_uav"] = np.stack(self.energy_uplink_episode_per_uav, axis=0)
        if self.energy_trans_episode_per_uav:
            payload["energy_trans_episode_per_uav"] = np.stack(self.energy_trans_episode_per_uav, axis=0)
        if self.energy_macro_episode_per_uav:
            payload["energy_macro_episode_per_uav"] = np.stack(self.energy_macro_episode_per_uav, axis=0)
        if self.compute_service_per_uav_mbits:
            payload["compute_service_per_uav_mbits"] = np.stack(self.compute_service_per_uav_mbits, axis=0)
        if self.coop_flow_mbits:
            payload["coop_flow_mbits"] = np.stack(self.coop_flow_mbits, axis=0)
        if self.tx_service_matrix_mbits:
            payload["tx_service_matrix_mbits"] = np.stack(self.tx_service_matrix_mbits, axis=0)
        if self.split_ratios_high:
            payload["split_ratios_high_mean"] = np.stack(self.split_ratios_high, axis=0)
        if self.split_ratios_low:
            payload["split_ratios_low_mean"] = np.stack(self.split_ratios_low, axis=0)
        if self.pressure_per_uav:
            payload["pressure_per_uav"] = np.stack(self.pressure_per_uav, axis=0)

        np.savez_compressed(save_path, **payload)

    def plot(self, save_dir: str) -> None:
        if not self.steps:
            return

        plot_evaluation_metrics(
            self.steps,
            self.scores,
            self.q_high,
            self.q_low,
            self.energy,
            self.u2g,
            self.cpu_util,
            self.delay_high,
            self.delay_low,
            self.delay_mean_ms,
            self.delay_weighted_ms,
            self.delay_max_ms,
            save_dir,
        )


def save_trajectory_data_with_verification(
    env: HRL_Wrapper,
    save_path: str,
) -> None:
    """Save raw trajectory arrays and a PNG trajectory preview."""
    traj_env = getattr(env, "env", env)
    if not getattr(traj_env, "record_trajectory", False):
        print(f"  [SKIP] Trajectory recording disabled; nothing saved to {save_path}")
        return

    uav_traj = np.asarray(getattr(traj_env, "uav_trajectory", []), dtype=np.float32)
    centroid_traj = np.asarray(getattr(traj_env, "centroid_trajectory", []), dtype=np.float32)
    users_traj = np.asarray(getattr(traj_env, "users_trajectory", []), dtype=np.float32)
    anchor_points = np.asarray(getattr(traj_env, "anchor_points", []), dtype=np.float32)
    user_allocations = np.asarray(getattr(traj_env, "user_allocations", []), dtype=np.int32)
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


def save_macro_diagnostics_with_verification(
    save_path: str,
    move_distance: List[float],
    action_norm: List[float],
    hover_ratio: List[float],
    projected_u2g_distance: List[float],
    projected_u2u_rate: List[float],
    avg_upload_delay_penalty: List[float],
    position_cost: List[float],
    position_improvement: List[float],
    macro_reward: List[float],
    projected_comm_cost: Optional[List[float]] = None,
    projected_comm_uplink_time: Optional[List[float]] = None,
    projected_comm_u2u_time: Optional[List[float]] = None,
) -> None:
    macro_steps = np.arange(1, len(move_distance) + 1, dtype=np.int32)
    if len(macro_steps) == 0:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if projected_comm_cost is None:
        projected_comm_cost = np.zeros_like(macro_steps, dtype=np.float64)
    if projected_comm_uplink_time is None:
        projected_comm_uplink_time = np.zeros_like(macro_steps, dtype=np.float64)
    if projected_comm_u2u_time is None:
        projected_comm_u2u_time = np.zeros_like(macro_steps, dtype=np.float64)

    plot_macro_diagnostics(
        macro_steps=macro_steps,
        move_distance=move_distance,
        action_norm=action_norm,
        hover_ratio=hover_ratio,
        projected_u2g_distance=projected_u2g_distance,
        projected_u2u_rate=projected_u2u_rate,
        avg_upload_delay_penalty=avg_upload_delay_penalty,
        position_cost=position_cost,
        position_improvement=position_improvement,
        macro_reward=macro_reward,
        save_path=save_path,
    )

    csv_path = os.path.splitext(save_path)[0] + ".csv"
    data = np.column_stack(
        [
            macro_steps,
            np.asarray(move_distance, dtype=np.float64),
            np.asarray(action_norm, dtype=np.float64),
            np.asarray(hover_ratio, dtype=np.float64),
            np.asarray(projected_u2g_distance, dtype=np.float64),
            np.asarray(projected_u2u_rate, dtype=np.float64),
            np.asarray(avg_upload_delay_penalty, dtype=np.float64),
            np.asarray(position_cost, dtype=np.float64),
            np.asarray(position_improvement, dtype=np.float64),
            np.asarray(macro_reward, dtype=np.float64),
            np.asarray(projected_comm_cost, dtype=np.float64),
            np.asarray(projected_comm_uplink_time, dtype=np.float64),
            np.asarray(projected_comm_u2u_time, dtype=np.float64),
        ]
    )
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header=(
            "macro_step,move_distance_m,action_norm,hover_ratio,"
            "projected_mean_u2g_distance_m,projected_mean_u2u_rate_mbps,"
            "avg_upload_delay_penalty,position_cost,position_improvement,"
            "macro_reward,projected_comm_cost,"
            "projected_comm_uplink_time_s,projected_comm_u2u_time_s"
        ),
        comments="",
    )

    if os.path.exists(save_path):
        file_size_kb = os.path.getsize(save_path) / 1024.0
        print(f"  Saved macro diagnostics -> {save_path} ({file_size_kb:.1f} KB)")
    else:
        print(f"  [WARN] Macro diagnostics plot was not found after saving: {save_path}")

    if os.path.exists(csv_path):
        csv_size_kb = os.path.getsize(csv_path) / 1024.0
        print(f"  Saved macro diagnostics CSV -> {csv_path} ({csv_size_kb:.1f} KB)")
    else:
        print(f"  [WARN] Macro diagnostics CSV was not found after saving: {csv_path}")


def maybe_save_episode_snapshots(
    env: HRL_Wrapper,
    results_dirs: Dict[str, str],
    trajectory_plot_interval: int,
    episode_idx: int,
    trace: EpisodeTrace,
) -> None:
    if trajectory_plot_interval <= 0:
        return
    if (episode_idx + 1) % trajectory_plot_interval != 0:
        return

    trajectory_path = os.path.join(
        results_dirs["trajectory_dir"],
        f"trajectory_ep_{episode_idx + 1:04d}.npz",
    )
    save_trajectory_data_with_verification(env, trajectory_path)

    diagnostics_path = os.path.join(
        results_dirs["macro_diag_dir"],
        f"macro_diag_ep_{episode_idx + 1:04d}.png",
    )
    save_macro_diagnostics_with_verification(
        save_path=diagnostics_path,
        move_distance=trace.macro_move_distance,
        action_norm=trace.macro_action_norm,
        hover_ratio=trace.macro_hover_ratio,
        projected_u2g_distance=trace.macro_projected_u2g,
        projected_u2u_rate=trace.macro_u2u_rate,
        avg_upload_delay_penalty=trace.macro_avg_upload_delay_penalty,
        position_cost=trace.macro_position_cost,
        position_improvement=trace.macro_position_improvement,
        macro_reward=trace.macro_rewards,
        projected_comm_cost=trace.macro_projected_comm_cost,
        projected_comm_uplink_time=trace.macro_projected_comm_uplink_time,
        projected_comm_u2u_time=trace.macro_projected_comm_u2u_time,
    )


def compute_delay_ms(avg_q: float, avg_arrival_step: float) -> float:
    lambda_mbps = avg_arrival_step / MICRO_DT_SECONDS
    return (avg_q / (lambda_mbps + 1e-9)) * 1000.0


def compute_delay_ms_vector(avg_q: np.ndarray, avg_arrival_step: np.ndarray) -> np.ndarray:
    avg_q = np.asarray(avg_q, dtype=np.float64)
    avg_arrival_step = np.asarray(avg_arrival_step, dtype=np.float64)
    numerator = avg_q * MICRO_DT_SECONDS * 1000.0
    return np.divide(
        numerator,
        avg_arrival_step,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=avg_arrival_step > 1e-12,
    ).astype(np.float64)


def compute_cumulative_little_delays(
    q_high_steps: List[float],
    q_low_steps: List[float],
    arrival_high_steps: List[float],
    arrival_low_steps: List[float],
    q_total_per_uav_steps: List[np.ndarray],
    arrival_total_per_uav_steps: List[np.ndarray],
    num_agents: int,
) -> Dict[str, np.ndarray]:
    """Compute W = integral(Q) / total arrivals over a full evaluation window."""
    q_high_sum = float(np.sum(q_high_steps)) if q_high_steps else 0.0
    q_low_sum = float(np.sum(q_low_steps)) if q_low_steps else 0.0
    arrival_high_sum = float(np.sum(arrival_high_steps)) if arrival_high_steps else 0.0
    arrival_low_sum = float(np.sum(arrival_low_steps)) if arrival_low_steps else 0.0

    delay_high_ms = (
        q_high_sum * MICRO_DT_SECONDS * 1000.0 / max(arrival_high_sum, 1e-12)
        if arrival_high_sum > 1e-12
        else 0.0
    )
    delay_low_ms = (
        q_low_sum * MICRO_DT_SECONDS * 1000.0 / max(arrival_low_sum, 1e-12)
        if arrival_low_sum > 1e-12
        else 0.0
    )

    if q_total_per_uav_steps:
        q_total_per_uav_sum = np.sum(
            np.stack(q_total_per_uav_steps, axis=0),
            axis=0,
        ).astype(np.float64)
        num_steps = len(q_total_per_uav_steps)
    else:
        q_total_per_uav_sum = np.zeros(num_agents, dtype=np.float64)
        num_steps = 0

    if arrival_total_per_uav_steps:
        arrival_total_per_uav_sum = np.sum(
            np.stack(arrival_total_per_uav_steps, axis=0),
            axis=0,
        ).astype(np.float64)
    else:
        arrival_total_per_uav_sum = np.zeros(num_agents, dtype=np.float64)

    delay_total_per_uav_ms = np.divide(
        q_total_per_uav_sum * MICRO_DT_SECONDS * 1000.0,
        arrival_total_per_uav_sum,
        out=np.zeros_like(q_total_per_uav_sum, dtype=np.float64),
        where=arrival_total_per_uav_sum > 1e-12,
    )
    delay_mean_ms = (
        float(np.mean(delay_total_per_uav_ms))
        if delay_total_per_uav_ms.size
        else 0.0
    )
    delay_weighted_ms = (
        float(
            np.sum(q_total_per_uav_sum) * MICRO_DT_SECONDS * 1000.0
            / max(float(np.sum(arrival_total_per_uav_sum)), 1e-12)
        )
        if np.sum(arrival_total_per_uav_sum) > 1e-12
        else 0.0
    )
    delay_max_ms = (
        float(np.max(delay_total_per_uav_ms))
        if delay_total_per_uav_ms.size
        else 0.0
    )
    avg_q_total_per_uav = (
        q_total_per_uav_sum / max(float(num_steps), 1.0)
        if num_steps > 0
        else np.zeros(num_agents, dtype=np.float64)
    )
    avg_arrival_total_per_uav = (
        arrival_total_per_uav_sum / max(float(num_steps), 1.0)
        if num_steps > 0
        else np.zeros(num_agents, dtype=np.float64)
    )

    return {
        "delay_high_ms": float(delay_high_ms),
        "delay_low_ms": float(delay_low_ms),
        "delay_mean_ms": delay_mean_ms,
        "delay_weighted_ms": delay_weighted_ms,
        "delay_max_ms": delay_max_ms,
        "delay_total_per_uav_ms": delay_total_per_uav_ms.astype(np.float32),
        "avg_q_total_per_uav": avg_q_total_per_uav.astype(np.float32),
        "avg_arrival_total_per_uav": avg_arrival_total_per_uav.astype(np.float32),
        "q_total_per_uav_sum": q_total_per_uav_sum.astype(np.float32),
        "arrival_total_per_uav_sum": arrival_total_per_uav_sum.astype(np.float32),
        "q_high_sum": q_high_sum,
        "q_low_sum": q_low_sum,
        "arrival_high_sum": arrival_high_sum,
        "arrival_low_sum": arrival_low_sum,
    }


def build_micro_step_metrics(
    raw_state: Dict[str, np.ndarray],
    mic_info: Optional[Dict[str, float]],
) -> Dict[str, np.ndarray]:
    raw_state = raw_state or {}
    mic_info = mic_info or {}

    q_loc_high = np.asarray(raw_state.get("Q_loc_high", []), dtype=np.float64)
    q_loc_low = np.asarray(raw_state.get("Q_loc_low", []), dtype=np.float64)
    q_tx_high = np.asarray(raw_state.get("Q_tx_high", []), dtype=np.float64)
    q_tx_low = np.asarray(raw_state.get("Q_tx_low", []), dtype=np.float64)
    # Raw A_t may retain the latest service-frame batch while deployment pauses
    # service. Step info records an arrival only when it enters upload service,
    # so a retained batch is not counted repeatedly during the transition.
    arrival_high = np.asarray(
        mic_info.get("arrival_high_per_uav", raw_state.get("A_t_high", [])),
        dtype=np.float64,
    )
    arrival_low = np.asarray(
        mic_info.get("arrival_low_per_uav", raw_state.get("A_t_low", [])),
        dtype=np.float64,
    )

    if q_loc_high.size == 0:
        num_uavs = int(np.asarray(mic_info.get("cpu_utilization_per_uav", []), dtype=np.float64).size)
        if num_uavs <= 0:
            num_uavs = 1
        zeros = np.zeros(num_uavs, dtype=np.float64)
        q_loc_high = zeros.copy()
        q_loc_low = zeros.copy()
        q_tx_high = np.zeros((num_uavs, num_uavs), dtype=np.float64)
        q_tx_low = np.zeros((num_uavs, num_uavs), dtype=np.float64)
        arrival_high = zeros.copy()
        arrival_low = zeros.copy()

    q_tx_high_out = np.sum(q_tx_high, axis=1)
    q_tx_low_out = np.sum(q_tx_low, axis=1)
    q_high_per_uav = q_loc_high + q_tx_high_out
    q_low_per_uav = q_loc_low + q_tx_low_out
    q_total_per_uav = q_high_per_uav + q_low_per_uav
    arrival_total_per_uav = arrival_high + arrival_low

    delay_high_per_uav_ms = compute_delay_ms_vector(q_high_per_uav, arrival_high)
    delay_low_per_uav_ms = compute_delay_ms_vector(q_low_per_uav, arrival_low)
    delay_total_per_uav_ms = compute_delay_ms_vector(q_total_per_uav, arrival_total_per_uav)

    cpu_util_per_uav = np.asarray(
        mic_info.get("cpu_utilization_per_uav", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    cpu_high_util_per_uav = np.asarray(
        mic_info.get("cpu_high_utilization_per_uav", np.zeros_like(cpu_util_per_uav)),
        dtype=np.float64,
    )
    cpu_low_util_per_uav = np.asarray(
        mic_info.get("cpu_low_utilization_per_uav", np.zeros_like(cpu_util_per_uav)),
        dtype=np.float64,
    )
    f_high_per_uav = np.asarray(
        mic_info.get("f_high_per_uav", np.zeros_like(cpu_util_per_uav)),
        dtype=np.float64,
    )
    f_low_per_uav = np.asarray(
        mic_info.get("f_low_per_uav", np.zeros_like(cpu_util_per_uav)),
        dtype=np.float64,
    )
    f_total_per_uav = np.asarray(
        mic_info.get("f_total_per_uav", np.zeros_like(cpu_util_per_uav)),
        dtype=np.float64,
    )
    cpu_saturation_ratio = float(
        np.asarray(mic_info.get("cpu_saturation_ratio", 0.0), dtype=np.float64)
    )
    upload_delay_cost_per_uav = np.asarray(
        mic_info.get("upload_delay_cost_per_uav", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    upload_delay_penalty_per_uav = np.asarray(
        mic_info.get("upload_delay_penalty_per_uav", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    energy_comp_per_uav = np.asarray(
        mic_info.get("energy_comp_per_uav", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    energy_uplink_per_uav = np.asarray(
        mic_info.get("energy_uplink_per_uav", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    energy_trans_per_uav = np.asarray(
        mic_info.get("energy_trans_per_uav", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    energy_total_per_uav = np.asarray(
        mic_info.get(
            "energy_total_per_uav",
            energy_comp_per_uav + energy_uplink_per_uav + energy_trans_per_uav,
        ),
        dtype=np.float64,
    )
    num_uavs = int(arrival_total_per_uav.size)
    compute_service_per_uav_mbits = np.asarray(
        mic_info.get("compute_service_per_uav_mbits", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    task_delay_source_uav_ms = np.asarray(
        mic_info.get("task_delay_source_uav_ms", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    task_completed_source_count = np.asarray(
        mic_info.get("task_completed_source_count", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    task_arrived_source_count = np.asarray(
        mic_info.get("task_arrived_source_count", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    task_delay_source_high_ms = np.asarray(
        mic_info.get("task_delay_source_high_ms", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    task_delay_source_low_ms = np.asarray(
        mic_info.get("task_delay_source_low_ms", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    task_completed_source_high_count = np.asarray(
        mic_info.get("task_completed_source_high_count", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    task_completed_source_low_count = np.asarray(
        mic_info.get("task_completed_source_low_count", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    task_arrived_source_high_count = np.asarray(
        mic_info.get("task_arrived_source_high_count", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    task_arrived_source_low_count = np.asarray(
        mic_info.get("task_arrived_source_low_count", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )
    coop_flow_mbits = np.asarray(
        mic_info.get("coop_flow_mbits", np.zeros((num_uavs, num_uavs))),
        dtype=np.float64,
    )
    tx_service_matrix_mbits = np.asarray(
        mic_info.get("tx_service_matrix_mbits", np.zeros((num_uavs, num_uavs))),
        dtype=np.float64,
    )
    split_ratios_high = np.asarray(
        mic_info.get("split_ratios_high", np.zeros((num_uavs, num_uavs))),
        dtype=np.float64,
    )
    split_ratios_low = np.asarray(
        mic_info.get("split_ratios_low", np.zeros((num_uavs, num_uavs))),
        dtype=np.float64,
    )
    pressure_per_uav = np.asarray(
        mic_info.get("pressure_per_uav", np.zeros_like(arrival_total_per_uav)),
        dtype=np.float64,
    )

    total_arrival = float(np.sum(arrival_total_per_uav))
    weighted_delay_ms = float(
        np.sum(delay_total_per_uav_ms * arrival_total_per_uav)
        / max(total_arrival, 1e-12)
    )

    return {
        "queue_length": float(np.sum(q_total_per_uav)),
        "q_high_tot": float(np.sum(q_high_per_uav)),
        "q_low_tot": float(np.sum(q_low_per_uav)),
        "q_loc_high_per_uav": q_loc_high.astype(np.float32),
        "q_loc_low_per_uav": q_loc_low.astype(np.float32),
        "q_tx_high_per_uav": q_tx_high_out.astype(np.float32),
        "q_tx_low_per_uav": q_tx_low_out.astype(np.float32),
        "q_high_per_uav": q_high_per_uav.astype(np.float32),
        "q_low_per_uav": q_low_per_uav.astype(np.float32),
        "q_total_per_uav": q_total_per_uav.astype(np.float32),
        "arrival_high_per_uav": arrival_high.astype(np.float32),
        "arrival_low_per_uav": arrival_low.astype(np.float32),
        "arrival_total_per_uav": arrival_total_per_uav.astype(np.float32),
        "delay_high_per_uav_ms": delay_high_per_uav_ms.astype(np.float32),
        "delay_low_per_uav_ms": delay_low_per_uav_ms.astype(np.float32),
        "delay_total_per_uav_ms": delay_total_per_uav_ms.astype(np.float32),
        "delay_mean_ms": float(np.mean(delay_total_per_uav_ms)) if delay_total_per_uav_ms.size else 0.0,
        "delay_weighted_ms": weighted_delay_ms,
        "delay_max_ms": float(np.max(delay_total_per_uav_ms)) if delay_total_per_uav_ms.size else 0.0,
        "cpu_utilization_mean": float(np.mean(cpu_util_per_uav)) if cpu_util_per_uav.size else 0.0,
        "cpu_utilization_per_uav": cpu_util_per_uav.astype(np.float32),
        "cpu_high_utilization_mean": float(np.mean(cpu_high_util_per_uav)) if cpu_high_util_per_uav.size else 0.0,
        "cpu_low_utilization_mean": float(np.mean(cpu_low_util_per_uav)) if cpu_low_util_per_uav.size else 0.0,
        "cpu_high_utilization_per_uav": cpu_high_util_per_uav.astype(np.float32),
        "cpu_low_utilization_per_uav": cpu_low_util_per_uav.astype(np.float32),
        "cpu_saturation_ratio": cpu_saturation_ratio,
        "f_high_mean_ghz": float(np.mean(f_high_per_uav) / 1e9) if f_high_per_uav.size else 0.0,
        "f_low_mean_ghz": float(np.mean(f_low_per_uav) / 1e9) if f_low_per_uav.size else 0.0,
        "f_total_mean_ghz": float(np.mean(f_total_per_uav) / 1e9) if f_total_per_uav.size else 0.0,
        "f_high_per_uav": f_high_per_uav.astype(np.float32),
        "f_low_per_uav": f_low_per_uav.astype(np.float32),
        "f_total_per_uav": f_total_per_uav.astype(np.float32),
        "upload_delay_cost_per_uav": upload_delay_cost_per_uav.astype(np.float32),
        "upload_delay_penalty_per_uav": upload_delay_penalty_per_uav.astype(np.float32),
        "upload_delay_cost": float(np.asarray(mic_info.get("upload_delay_cost", 0.0), dtype=np.float64)),
        "upload_delay_penalty": float(np.asarray(mic_info.get("upload_delay_penalty", 0.0), dtype=np.float64)),
        "energy_uplink": float(np.asarray(mic_info.get("energy_uplink", 0.0), dtype=np.float64)),
        "energy_trans": float(np.asarray(mic_info.get("energy_trans", 0.0), dtype=np.float64)),
        "energy_comp": float(np.asarray(mic_info.get("energy_comp", 0.0), dtype=np.float64)),
        "energy_comp_per_uav": energy_comp_per_uav.astype(np.float32),
        "energy_uplink_per_uav": energy_uplink_per_uav.astype(np.float32),
        "energy_trans_per_uav": energy_trans_per_uav.astype(np.float32),
        "energy_total_per_uav": energy_total_per_uav.astype(np.float32),
        "compute_service_per_uav_mbits": compute_service_per_uav_mbits.astype(np.float32),
        "task_delay_source_uav_ms": task_delay_source_uav_ms.astype(np.float32),
        "task_completed_source_count": task_completed_source_count.astype(np.float32),
        "task_arrived_source_count": task_arrived_source_count.astype(np.float32),
        "task_delay_source_high_ms": task_delay_source_high_ms.astype(np.float32),
        "task_delay_source_low_ms": task_delay_source_low_ms.astype(np.float32),
        "task_completed_source_high_count": task_completed_source_high_count.astype(np.float32),
        "task_completed_source_low_count": task_completed_source_low_count.astype(np.float32),
        "task_arrived_source_high_count": task_arrived_source_high_count.astype(np.float32),
        "task_arrived_source_low_count": task_arrived_source_low_count.astype(np.float32),
        "coop_flow_mbits": coop_flow_mbits.astype(np.float32),
        "tx_service_matrix_mbits": tx_service_matrix_mbits.astype(np.float32),
        "split_ratios_high": split_ratios_high.astype(np.float32),
        "split_ratios_low": split_ratios_low.astype(np.float32),
        "pressure_per_uav": pressure_per_uav.astype(np.float32),
        "pressure_imbalance": float(np.asarray(mic_info.get("pressure_imbalance", 0.0), dtype=np.float64)),
        "coop_total_flow_mbits": float(np.asarray(mic_info.get("coop_total_flow_mbits", np.sum(coop_flow_mbits)), dtype=np.float64)),
        "coop_cross_flow_mbits": float(np.asarray(mic_info.get("coop_cross_flow_mbits", np.sum(coop_flow_mbits) - np.trace(coop_flow_mbits)), dtype=np.float64)),
        "coop_ratio": float(np.asarray(mic_info.get("coop_ratio", 0.0), dtype=np.float64)),
        "total_energy": float(np.asarray(mic_info.get("total_energy", 0.0), dtype=np.float64)),
        "micro_cost_raw": float(np.asarray(mic_info.get("micro_cost_raw", 0.0), dtype=np.float64)),
        "cost_holding": float(np.asarray(mic_info.get("cost_holding", mic_info.get("cost_wait", 0.0)), dtype=np.float64)),
        "cost_wait": float(np.asarray(mic_info.get("cost_wait", 0.0), dtype=np.float64)),
        "cost_queue": float(np.asarray(mic_info.get("cost_queue", 0.0), dtype=np.float64)),
        "cost_energy": float(np.asarray(mic_info.get("cost_energy", 0.0), dtype=np.float64)),
        "compute_service_high_mbits": float(np.asarray(mic_info.get("compute_service_high_mbits", 0.0), dtype=np.float64)),
        "compute_service_low_mbits": float(np.asarray(mic_info.get("compute_service_low_mbits", 0.0), dtype=np.float64)),
        "compute_service_sensor_mbits": float(np.asarray(mic_info.get("compute_service_sensor_mbits", 0.0), dtype=np.float64)),
        "compute_service_controller_mbits": float(np.asarray(mic_info.get("compute_service_controller_mbits", 0.0), dtype=np.float64)),
        "compute_service_vision_mbits": float(np.asarray(mic_info.get("compute_service_vision_mbits", 0.0), dtype=np.float64)),
        "compute_service_high_cycles": float(np.asarray(mic_info.get("compute_service_high_cycles", 0.0), dtype=np.float64)),
        "compute_service_low_cycles": float(np.asarray(mic_info.get("compute_service_low_cycles", 0.0), dtype=np.float64)),
        "compute_service_sensor_cycles": float(np.asarray(mic_info.get("compute_service_sensor_cycles", 0.0), dtype=np.float64)),
        "compute_service_controller_cycles": float(np.asarray(mic_info.get("compute_service_controller_cycles", 0.0), dtype=np.float64)),
        "compute_service_vision_cycles": float(np.asarray(mic_info.get("compute_service_vision_cycles", 0.0), dtype=np.float64)),
        "task_holding_cost": float(np.asarray(mic_info.get("task_holding_cost", mic_info.get("task_wait_cost", 0.0)), dtype=np.float64)),
        "task_wait_cost": float(np.asarray(mic_info.get("task_wait_cost", mic_info.get("task_holding_cost", 0.0)), dtype=np.float64)),
        "task_alive_count": float(np.asarray(mic_info.get("task_alive_count", 0.0), dtype=np.float64)),
        "task_alive_high_count": float(np.asarray(mic_info.get("task_alive_high_count", 0.0), dtype=np.float64)),
        "task_alive_low_count": float(np.asarray(mic_info.get("task_alive_low_count", 0.0), dtype=np.float64)),
        "task_arrived_count": float(np.asarray(mic_info.get("task_arrived_count", 0.0), dtype=np.float64)),
        "task_delay_avg_ms": float(np.asarray(mic_info.get("task_delay_avg_ms", 0.0), dtype=np.float64)),
        "task_delay_weighted_ms": float(np.asarray(mic_info.get("task_delay_weighted_ms", 0.0), dtype=np.float64)),
        "task_delay_high_ms": float(np.asarray(mic_info.get("task_delay_high_ms", 0.0), dtype=np.float64)),
        "task_delay_low_ms": float(np.asarray(mic_info.get("task_delay_low_ms", 0.0), dtype=np.float64)),
        "task_delay_sensor_ms": float(np.asarray(mic_info.get("task_delay_sensor_ms", 0.0), dtype=np.float64)),
        "task_delay_controller_ms": float(np.asarray(mic_info.get("task_delay_controller_ms", 0.0), dtype=np.float64)),
        "task_delay_vision_ms": float(np.asarray(mic_info.get("task_delay_vision_ms", 0.0), dtype=np.float64)),
        "task_completed_count": float(np.asarray(mic_info.get("task_completed_count", 0.0), dtype=np.float64)),
        "task_completion_rate": float(np.asarray(mic_info.get("task_completion_rate", 0.0), dtype=np.float64)),
        "task_qos_violation_rate": float(np.asarray(mic_info.get("task_qos_violation_rate", 0.0), dtype=np.float64)),
        "task_qos_violation_high_rate": float(np.asarray(mic_info.get("task_qos_violation_high_rate", 0.0), dtype=np.float64)),
        "task_qos_violation_low_rate": float(np.asarray(mic_info.get("task_qos_violation_low_rate", 0.0), dtype=np.float64)),
    }


def print_episode_summary(
    experiment_label: str,
    episode_idx: int,
    episode_macro_reward: float,
    episode_micro_reward: float,
    trace: EpisodeTrace,
) -> None:
    summary = trace.summary()
    avg_micro_reward_per_step = episode_micro_reward / max(len(trace.queue_backlogs), 1)
    avg_macro_reward_per_step = episode_macro_reward / max(len(trace.macro_rewards), 1)

    print(
        f"{experiment_label} Episode {episode_idx} | "
        f"Macro R: {episode_macro_reward:.2f} "
        f"(avg {avg_macro_reward_per_step:.2f}/macro) | "
        f"Micro R: {episode_micro_reward:.2f} "
        f"(avg {avg_micro_reward_per_step:.2f}/micro)"
    )
    print(
        f"  Raw micro cost: {summary['avg_micro_cost_raw']:.4f} | "
        f"CPU util: {summary['avg_cpu_util_mean']:.3f} | "
        f"Dq(w): {summary['avg_delay_weighted_ms']:.1f} ms | "
        f"Ttask: {summary['avg_task_delay_ms']:.1f} ms | "
        f"Done: {summary['task_completed_count']:.0f}/"
        f"{summary['task_arrived_count']:.0f} "
        f"({summary['task_completion_rate']:.2f}) | "
        f"Alive: {summary['task_alive_count']:.0f} | "
        f"Dq(max): {summary['avg_delay_max_ms']:.1f} ms | "
        f"U2G mean: {summary['avg_u2g_distance']:.2f} m | "
        f"Ah: {summary['avg_arrival_high']:.2f} | "
        f"Qh_loc: {summary['avg_q_loc_high']:.2f} | "
        f"Qh_tx: {summary['avg_q_tx_high']:.2f} | "
        f"Al: {summary['avg_arrival_low']:.2f} | "
        f"Ql_loc: {summary['avg_q_loc_low']:.2f} | "
        f"Ql_tx: {summary['avg_q_tx_low']:.2f}"
    )
    print(
        f"  Macro motion: {summary['avg_macro_move_distance']:.2f} m | "
        f"Action norm: {summary['avg_macro_action_norm']:.3f} | "
        f"Hover ratio: {summary['avg_macro_hover_ratio']:.2f} | "
        f"Nominal U2U: {summary['avg_macro_u2u_rate']:.2f} Mbps | "
        f"Pos cost: {summary['avg_macro_position_cost']:.3f} | "
        f"Improve: {summary['avg_macro_position_improvement']:.3f}"
    )
    print(
        f"  Task delay(ms) H/L/S/C/V: "
        f"{trace.task_delay_high_ms[-1] if trace.task_delay_high_ms else 0.0:.1f}/"
        f"{trace.task_delay_low_ms[-1] if trace.task_delay_low_ms else 0.0:.1f}/"
        f"{trace.task_delay_sensor_ms[-1] if trace.task_delay_sensor_ms else 0.0:.1f}/"
        f"{trace.task_delay_controller_ms[-1] if trace.task_delay_controller_ms else 0.0:.1f}/"
        f"{trace.task_delay_vision_ms[-1] if trace.task_delay_vision_ms else 0.0:.1f} | "
        f"CPU service(Mb) H/L/S/C/V: "
        f"{summary['sum_compute_service_high_mbits']:.2f}/"
        f"{summary['sum_compute_service_low_mbits']:.2f}/"
        f"{summary['sum_compute_service_sensor_mbits']:.2f}/"
        f"{summary['sum_compute_service_controller_mbits']:.2f}/"
        f"{summary['sum_compute_service_vision_mbits']:.2f}"
    )


def print_evaluation_summary(step: int, metrics: Dict[str, float]) -> None:
    if metrics.get("power_constraint_scope", "system") == "per_uav":
        battery_power = np.asarray(
            metrics.get("average_battery_power_per_uav_w", []),
            dtype=np.float64,
        )
        battery_ucb = np.asarray(
            metrics.get("battery_power_ucb95_per_uav_w", []),
            dtype=np.float64,
        )
        constrained_power_summary = (
            f"Psys(W): {metrics.get('average_power_w', 0.0):.2f} | "
            "Pbatt(W): "
            f"{np.array2string(battery_power, precision=1, separator=',')} | "
            "Pbatt-UCB95(W): "
            f"{np.array2string(battery_ucb, precision=1, separator=',')} | "
            "Pviol-max(W): "
            f"{metrics.get('max_battery_power_ucb95_violation_w', 0.0):.2f}"
        )
    else:
        constrained_power_summary = (
            f"Pavg(W): {metrics.get('average_power_w', 0.0):.2f} | "
            "Pucb95(W): "
            f"{metrics.get('average_power_ucb95_w', float('inf')):.2f} | "
            f"Pvio(W): {metrics.get('power_violation_w', 0.0):.2f}"
        )
    print(
        f"  Eval@{step} | "
        f"Score: {metrics['score']:.2f} | "
        f"J: {metrics.get('J', 0.0):.4f} | "
        f"Dh(ms): {metrics['delay_high']:.1f} | "
        f"Dl(ms): {metrics['delay_low']:.1f} | "
        f"Dw(ms): {metrics.get('delay_weighted_ms', 0.0):.1f} | "
        f"Ttask(ms): {metrics.get('task_delay_avg_ms', 0.0):.1f} | "
        f"TwE2E(ms): {metrics.get('task_delay_weighted_ms', 0.0):.1f} | "
        f"Done: {metrics.get('task_completed_count', 0.0):.0f}/"
        f"{metrics.get('task_arrived_count', 0.0):.0f} | "
        f"Dmax(ms): {metrics.get('delay_max_ms', 0.0):.1f} | "
        f"Eservice/step(J): {metrics['energy']:.2f} | "
        f"{constrained_power_summary} | "
        f"Eep: {metrics.get('energy_total_episode', 0.0):.2f} | "
        f"Coop: {metrics.get('coop_ratio', 0.0):.2f} | "
        f"CPU sat: {metrics.get('cpu_saturation', 0.0):.2f} | "
        f"Press imbal: {metrics.get('pressure_imbalance', 0.0):.2f} | "
        f"U2G: {metrics['u2g']:.2f} | "
        f"Decision: {metrics.get('micro_decision_mean_ms', 0.0):.3f} ms"
    )
    print(
        f"  Eval task delay(ms) H/L/S/C/V: "
        f"{metrics.get('task_delay_high_ms', 0.0):.1f}/"
        f"{metrics.get('task_delay_low_ms', 0.0):.1f}/"
        f"{metrics.get('task_delay_sensor_ms', 0.0):.1f}/"
        f"{metrics.get('task_delay_controller_ms', 0.0):.1f}/"
        f"{metrics.get('task_delay_vision_ms', 0.0):.1f} | "
        f"CPU service(Mb) H/L/S/C/V: "
        f"{metrics.get('compute_service_high_mbits', 0.0):.2f}/"
        f"{metrics.get('compute_service_low_mbits', 0.0):.2f}/"
        f"{metrics.get('compute_service_sensor_mbits', 0.0):.2f}/"
        f"{metrics.get('compute_service_controller_mbits', 0.0):.2f}/"
        f"{metrics.get('compute_service_vision_mbits', 0.0):.2f}"
    )


def evaluate_policy(
    eval_env: HRL_Wrapper,
    args: argparse.Namespace,
    macro_policy: MacroPolicy,
    micro_policy: MicroPolicy,
    on_episode_start: Optional[EpisodeStartHook] = None,
) -> Dict[str, float]:
    gate_values = getattr(micro_policy, "residual_gate_values", None)
    if gate_values is not None:
        gate_values.clear()
    expected_micro_steps = int(eval_env.env.micro_steps_per_macro)
    if (
        bool(int(getattr(args, "strict_hierarchy_validation", 0)))
        and int(args.micro_steps) != expected_micro_steps
    ):
        raise ValueError(
            f"micro_steps={args.micro_steps} does not match physical macro "
            f"slot length {expected_micro_steps}."
        )
    eval_scores: List[float] = []
    eval_J: List[float] = []
    eval_delay_objective: List[float] = []
    eval_legacy_scores: List[float] = []
    eval_q_high: List[float] = []
    eval_q_low: List[float] = []
    eval_energy: List[float] = []
    eval_average_power_w: List[float] = []
    eval_u2g: List[float] = []
    eval_cpu_util: List[float] = []
    eval_cpu_saturation: List[float] = []
    eval_cpu_high_util: List[float] = []
    eval_cpu_low_util: List[float] = []
    eval_f_high_ghz: List[float] = []
    eval_f_low_ghz: List[float] = []
    eval_f_total_ghz: List[float] = []
    eval_delay_high: List[float] = []
    eval_delay_low: List[float] = []
    eval_delay_mean_ms: List[float] = []
    eval_delay_weighted_ms: List[float] = []
    eval_delay_max_ms: List[float] = []
    eval_task_delay_avg_ms: List[float] = []
    eval_task_delay_weighted_ms: List[float] = []
    eval_task_delay_high_ms: List[float] = []
    eval_task_delay_low_ms: List[float] = []
    eval_task_delay_sensor_ms: List[float] = []
    eval_task_delay_controller_ms: List[float] = []
    eval_task_delay_vision_ms: List[float] = []
    eval_task_completed_count: List[float] = []
    eval_task_arrived_count: List[float] = []
    eval_task_completion_rate: List[float] = []
    eval_task_qos_violation_rate: List[float] = []
    eval_compute_service_high_mbits: List[float] = []
    eval_compute_service_low_mbits: List[float] = []
    eval_compute_service_sensor_mbits: List[float] = []
    eval_compute_service_controller_mbits: List[float] = []
    eval_compute_service_vision_mbits: List[float] = []
    eval_q_total_per_uav: List[np.ndarray] = []
    eval_arrival_total_per_uav: List[np.ndarray] = []
    eval_cpu_util_per_uav: List[np.ndarray] = []
    eval_delay_total_per_uav_ms: List[np.ndarray] = []
    eval_upload_delay_cost_per_uav: List[np.ndarray] = []
    eval_upload_delay_penalty_per_uav: List[np.ndarray] = []
    eval_task_delay_source_uav_ms: List[np.ndarray] = []
    eval_task_completed_source_count: List[np.ndarray] = []
    eval_task_arrived_source_count: List[np.ndarray] = []
    eval_task_delay_source_high_ms: List[np.ndarray] = []
    eval_task_delay_source_low_ms: List[np.ndarray] = []
    eval_task_completed_source_high_count: List[np.ndarray] = []
    eval_task_completed_source_low_count: List[np.ndarray] = []
    eval_task_arrived_source_high_count: List[np.ndarray] = []
    eval_task_arrived_source_low_count: List[np.ndarray] = []
    eval_energy_total_episode_per_uav: List[np.ndarray] = []
    eval_energy_comp_episode_per_uav: List[np.ndarray] = []
    eval_energy_uplink_episode_per_uav: List[np.ndarray] = []
    eval_energy_trans_episode_per_uav: List[np.ndarray] = []
    eval_energy_macro_episode_per_uav: List[np.ndarray] = []
    eval_energy_total_episode: List[float] = []
    eval_average_battery_power_per_uav_w: List[np.ndarray] = []
    eval_compute_service_per_uav_mbits: List[np.ndarray] = []
    eval_coop_flow_mbits: List[np.ndarray] = []
    eval_tx_service_matrix_mbits: List[np.ndarray] = []
    eval_split_ratios_high: List[np.ndarray] = []
    eval_split_ratios_low: List[np.ndarray] = []
    eval_pressure_per_uav: List[np.ndarray] = []
    eval_pressure_imbalance: List[float] = []
    eval_coop_ratio: List[float] = []
    micro_decision_count = 0
    micro_decision_seconds = 0.0
    micro_decision_durations_s: List[float] = []

    for eval_idx in range(args.eval_episodes):
        seed = args.eval_base_seed + eval_idx
        load_switch_schedule = initialize_episode_load_switch(
            eval_env,
            args,
            seed,
        )
        eval_env.reset(seed=seed)
        if load_switch_schedule is not None:
            apply_load_switch_phase(
                eval_env,
                load_switch_schedule,
                0,
                force=True,
            )
        if on_episode_start is not None:
            on_episode_start(eval_env, seed)

        ep_q_high: List[float] = []
        ep_q_low: List[float] = []
        ep_J: List[float] = []
        ep_delay_objective: List[float] = []
        ep_energy: List[float] = []
        ep_u2g: List[float] = []
        ep_cpu_util: List[float] = []
        ep_cpu_saturation: List[float] = []
        ep_cpu_high_util: List[float] = []
        ep_cpu_low_util: List[float] = []
        ep_f_high_ghz: List[float] = []
        ep_f_low_ghz: List[float] = []
        ep_f_total_ghz: List[float] = []
        ep_arrival_high: List[float] = []
        ep_arrival_low: List[float] = []
        ep_q_total_per_uav: List[np.ndarray] = []
        ep_arrival_total_per_uav: List[np.ndarray] = []
        ep_cpu_util_per_uav: List[np.ndarray] = []
        ep_delay_total_per_uav_ms: List[np.ndarray] = []
        ep_upload_delay_cost_per_uav: List[np.ndarray] = []
        ep_upload_delay_penalty_per_uav: List[np.ndarray] = []
        ep_energy_comp_per_uav: List[np.ndarray] = []
        ep_energy_uplink_per_uav: List[np.ndarray] = []
        ep_energy_trans_per_uav: List[np.ndarray] = []
        ep_energy_macro_per_uav: List[np.ndarray] = []
        ep_compute_service_per_uav_mbits: List[np.ndarray] = []
        ep_coop_flow_mbits: List[np.ndarray] = []
        ep_tx_service_matrix_mbits: List[np.ndarray] = []
        ep_split_ratios_high: List[np.ndarray] = []
        ep_split_ratios_low: List[np.ndarray] = []
        ep_pressure_per_uav: List[np.ndarray] = []
        ep_pressure_imbalance: List[float] = []
        ep_coop_ratio: List[float] = []
        ep_compute_service_high_mbits: List[float] = []
        ep_compute_service_low_mbits: List[float] = []
        ep_compute_service_sensor_mbits: List[float] = []
        ep_compute_service_controller_mbits: List[float] = []
        ep_compute_service_vision_mbits: List[float] = []

        mappo_local_state, _ = eval_env.get_macro_state()
        done = False

        for macro_index in range(args.macro_steps):
            switched = advance_load_switch_before_macro(
                eval_env,
                load_switch_schedule,
                macro_index,
            )
            if switched:
                mappo_local_state, _ = eval_env.get_macro_state()
            local_obs_batch = np.vstack(mappo_local_state).astype(np.float32)
            macro_action = np.asarray(
                macro_policy(eval_env, local_obs_batch, True),
                dtype=np.float32,
            )

            _, _, mac_info = eval_env.macro_step(macro_action)

            if isinstance(mac_info, dict):
                ep_energy_macro_per_uav.append(
                    np.asarray(
                        mac_info.get("E_macro_per_uav", np.zeros(args.num_agents)),
                        dtype=np.float32,
                    )
                )
                u2g_distances = mac_info.get("u2g_distances", None)
                if u2g_distances is not None:
                    ep_u2g.append(float(np.mean(u2g_distances)))

            for _ in range(eval_env.env.resource_frames_per_macro):
                service_paused = bool(
                    getattr(eval_env.env, "pause_service_during_flight", False)
                    and eval_env.env.macro_service_pause_steps_remaining > 0
                )
                if service_paused:
                    micro_action = np.zeros(
                        eval_env.micro_action_space.shape[0],
                        dtype=np.float32,
                    )
                else:
                    eval_env.prepare_resource_frame()
                    decision_started = time.perf_counter()
                    micro_action = np.asarray(
                        micro_policy(eval_env, True),
                        dtype=np.float32,
                    )
                    decision_duration_s = time.perf_counter() - decision_started
                    micro_decision_seconds += decision_duration_s
                    micro_decision_durations_s.append(decision_duration_s)
                    micro_decision_count += 1
                frame = eval_env.step_resource_frame(micro_action)
                done = bool(frame["done"])
                for raw_state, mic_info in zip(
                    frame["raw_states"],
                    frame["infos"],
                ):
                    micro_metrics = build_micro_step_metrics(raw_state, mic_info)

                    ep_q_high.append(micro_metrics["q_high_tot"])
                    ep_q_low.append(micro_metrics["q_low_tot"])
                    ep_J.append(float(mic_info.get("J_step", 0.0)))
                    ep_delay_objective.append(
                        float(mic_info.get("delay_objective_step", 0.0))
                    )
                    ep_energy.append(
                        micro_metrics["energy_uplink"]
                        + micro_metrics["energy_trans"]
                        + micro_metrics["energy_comp"]
                    )
                    ep_cpu_util.append(micro_metrics["cpu_utilization_mean"])
                    ep_cpu_saturation.append(micro_metrics["cpu_saturation_ratio"])
                    ep_cpu_high_util.append(micro_metrics["cpu_high_utilization_mean"])
                    ep_cpu_low_util.append(micro_metrics["cpu_low_utilization_mean"])
                    ep_f_high_ghz.append(micro_metrics["f_high_mean_ghz"])
                    ep_f_low_ghz.append(micro_metrics["f_low_mean_ghz"])
                    ep_f_total_ghz.append(micro_metrics["f_total_mean_ghz"])
                    ep_arrival_high.append(float(np.sum(micro_metrics["arrival_high_per_uav"])))
                    ep_arrival_low.append(float(np.sum(micro_metrics["arrival_low_per_uav"])))
                    ep_q_total_per_uav.append(micro_metrics["q_total_per_uav"])
                    ep_arrival_total_per_uav.append(micro_metrics["arrival_total_per_uav"])
                    ep_cpu_util_per_uav.append(micro_metrics["cpu_utilization_per_uav"])
                    ep_delay_total_per_uav_ms.append(micro_metrics["delay_total_per_uav_ms"])
                    ep_upload_delay_cost_per_uav.append(micro_metrics["upload_delay_cost_per_uav"])
                    ep_upload_delay_penalty_per_uav.append(micro_metrics["upload_delay_penalty_per_uav"])
                    ep_energy_comp_per_uav.append(micro_metrics["energy_comp_per_uav"])
                    ep_energy_uplink_per_uav.append(micro_metrics["energy_uplink_per_uav"])
                    ep_energy_trans_per_uav.append(micro_metrics["energy_trans_per_uav"])
                    ep_compute_service_per_uav_mbits.append(
                        micro_metrics["compute_service_per_uav_mbits"]
                    )
                    ep_coop_flow_mbits.append(micro_metrics["coop_flow_mbits"])
                    ep_tx_service_matrix_mbits.append(
                        micro_metrics["tx_service_matrix_mbits"]
                    )
                    ep_split_ratios_high.append(micro_metrics["split_ratios_high"])
                    ep_split_ratios_low.append(micro_metrics["split_ratios_low"])
                    ep_pressure_per_uav.append(micro_metrics["pressure_per_uav"])
                    ep_pressure_imbalance.append(micro_metrics["pressure_imbalance"])
                    ep_coop_ratio.append(micro_metrics["coop_ratio"])
                    ep_compute_service_high_mbits.append(micro_metrics["compute_service_high_mbits"])
                    ep_compute_service_low_mbits.append(micro_metrics["compute_service_low_mbits"])
                    ep_compute_service_sensor_mbits.append(micro_metrics["compute_service_sensor_mbits"])
                    ep_compute_service_controller_mbits.append(
                        micro_metrics["compute_service_controller_mbits"]
                    )
                    ep_compute_service_vision_mbits.append(micro_metrics["compute_service_vision_mbits"])

                if done:
                    break

            # Finalizing the macro reward also updates the realized-flow EMA.
            # It must happen before constructing s_{k+1}; otherwise training
            # sees a one-slot lag and evaluation never updates flow state.
            eval_env.get_macro_reward()
            mappo_local_state, _ = eval_env.get_macro_state()
            if done:
                break

        avg_q_high = float(np.mean(ep_q_high)) if ep_q_high else 0.0
        avg_q_low = float(np.mean(ep_q_low)) if ep_q_low else 0.0
        avg_energy = float(np.mean(ep_energy)) if ep_energy else 0.0
        avg_u2g = float(np.mean(ep_u2g)) if ep_u2g else 0.0
        avg_J = float(np.mean(ep_J)) if ep_J else 0.0
        avg_delay_objective = (
            float(np.mean(ep_delay_objective))
            if ep_delay_objective
            else 0.0
        )
        avg_cpu = float(np.mean(ep_cpu_util)) if ep_cpu_util else 0.0
        avg_cpu_saturation = (
            float(np.mean(ep_cpu_saturation)) if ep_cpu_saturation else 0.0
        )
        avg_cpu_high_util = (
            float(np.mean(ep_cpu_high_util)) if ep_cpu_high_util else 0.0
        )
        avg_cpu_low_util = (
            float(np.mean(ep_cpu_low_util)) if ep_cpu_low_util else 0.0
        )
        avg_f_high_ghz = float(np.mean(ep_f_high_ghz)) if ep_f_high_ghz else 0.0
        avg_f_low_ghz = float(np.mean(ep_f_low_ghz)) if ep_f_low_ghz else 0.0
        avg_f_total_ghz = float(np.mean(ep_f_total_ghz)) if ep_f_total_ghz else 0.0
        little = compute_cumulative_little_delays(
            ep_q_high,
            ep_q_low,
            ep_arrival_high,
            ep_arrival_low,
            ep_q_total_per_uav,
            ep_arrival_total_per_uav,
            args.num_agents,
        )
        delay_high_ms = float(little["delay_high_ms"])
        delay_low_ms = float(little["delay_low_ms"])
        avg_q_total_per_uav = little["avg_q_total_per_uav"]
        avg_arrival_total_per_uav = little["avg_arrival_total_per_uav"]
        avg_cpu_util_per_uav = (
            np.mean(np.stack(ep_cpu_util_per_uav, axis=0), axis=0)
            if ep_cpu_util_per_uav
            else np.zeros(args.num_agents, dtype=np.float32)
        )
        avg_delay_total_per_uav_ms = little["delay_total_per_uav_ms"]
        avg_upload_delay_cost_per_uav = (
            np.mean(np.stack(ep_upload_delay_cost_per_uav, axis=0), axis=0)
            if ep_upload_delay_cost_per_uav
            else np.zeros(args.num_agents, dtype=np.float32)
        )
        avg_upload_delay_penalty_per_uav = (
            np.mean(np.stack(ep_upload_delay_penalty_per_uav, axis=0), axis=0)
            if ep_upload_delay_penalty_per_uav
            else np.zeros(args.num_agents, dtype=np.float32)
        )
        sum_energy_comp_per_uav = (
            np.sum(np.stack(ep_energy_comp_per_uav, axis=0), axis=0)
            if ep_energy_comp_per_uav
            else np.zeros(args.num_agents, dtype=np.float32)
        )
        sum_energy_uplink_per_uav = (
            np.sum(np.stack(ep_energy_uplink_per_uav, axis=0), axis=0)
            if ep_energy_uplink_per_uav
            else np.zeros(args.num_agents, dtype=np.float32)
        )
        sum_energy_trans_per_uav = (
            np.sum(np.stack(ep_energy_trans_per_uav, axis=0), axis=0)
            if ep_energy_trans_per_uav
            else np.zeros(args.num_agents, dtype=np.float32)
        )
        sum_energy_macro_per_uav = (
            np.sum(np.stack(ep_energy_macro_per_uav, axis=0), axis=0)
            if ep_energy_macro_per_uav
            else np.zeros(args.num_agents, dtype=np.float32)
        )
        sum_energy_total_per_uav = (
            sum_energy_comp_per_uav
            + sum_energy_uplink_per_uav
            + sum_energy_trans_per_uav
            + sum_energy_macro_per_uav
        )
        sum_uav_battery_energy_per_uav = (
            sum_energy_comp_per_uav
            + sum_energy_trans_per_uav
            + sum_energy_macro_per_uav
        )
        sum_compute_service_per_uav_mbits = (
            np.sum(np.stack(ep_compute_service_per_uav_mbits, axis=0), axis=0)
            if ep_compute_service_per_uav_mbits
            else np.zeros(args.num_agents, dtype=np.float32)
        )
        final_coop_flow_mbits = (
            ep_coop_flow_mbits[-1]
            if ep_coop_flow_mbits
            else np.zeros((args.num_agents, args.num_agents), dtype=np.float32)
        )
        sum_tx_service_matrix_mbits = (
            np.sum(np.stack(ep_tx_service_matrix_mbits, axis=0), axis=0)
            if ep_tx_service_matrix_mbits
            else np.zeros((args.num_agents, args.num_agents), dtype=np.float32)
        )
        avg_split_ratios_high = (
            np.mean(np.stack(ep_split_ratios_high, axis=0), axis=0)
            if ep_split_ratios_high
            else np.zeros((args.num_agents, args.num_agents), dtype=np.float32)
        )
        avg_split_ratios_low = (
            np.mean(np.stack(ep_split_ratios_low, axis=0), axis=0)
            if ep_split_ratios_low
            else np.zeros((args.num_agents, args.num_agents), dtype=np.float32)
        )
        avg_pressure_per_uav = (
            np.mean(np.stack(ep_pressure_per_uav, axis=0), axis=0)
            if ep_pressure_per_uav
            else np.zeros(args.num_agents, dtype=np.float32)
        )
        avg_pressure_imbalance = (
            float(np.mean(ep_pressure_imbalance))
            if ep_pressure_imbalance
            else 0.0
        )
        final_coop_ratio = float(ep_coop_ratio[-1]) if ep_coop_ratio else 0.0
        delay_mean_ms = float(little["delay_mean_ms"])
        delay_weighted_ms = float(little["delay_weighted_ms"])
        delay_max_ms = float(little["delay_max_ms"])
        task_stats = eval_env.env.get_task_delay_stats()

        legacy_score = -(
            EVAL_W_HIGH * avg_q_high
            + EVAL_W_LOW * avg_q_low
            + EVAL_W_ENERGY * avg_energy
            + EVAL_W_U2G * avg_u2g
        )
        if str(getattr(eval_env, "objective_mode", "weighted")) == "constrained_delay":
            score = -avg_delay_objective * 100.0
        else:
            score = -avg_J * 100.0

        eval_scores.append(score)
        eval_J.append(avg_J)
        eval_delay_objective.append(avg_delay_objective)
        eval_legacy_scores.append(legacy_score)
        eval_q_high.append(avg_q_high)
        eval_q_low.append(avg_q_low)
        eval_energy.append(avg_energy)
        eval_u2g.append(avg_u2g)
        eval_cpu_util.append(avg_cpu)
        eval_cpu_saturation.append(avg_cpu_saturation)
        eval_cpu_high_util.append(avg_cpu_high_util)
        eval_cpu_low_util.append(avg_cpu_low_util)
        eval_f_high_ghz.append(avg_f_high_ghz)
        eval_f_low_ghz.append(avg_f_low_ghz)
        eval_f_total_ghz.append(avg_f_total_ghz)
        eval_delay_high.append(delay_high_ms)
        eval_delay_low.append(delay_low_ms)
        eval_delay_mean_ms.append(delay_mean_ms)
        eval_delay_weighted_ms.append(delay_weighted_ms)
        eval_delay_max_ms.append(delay_max_ms)
        eval_task_delay_avg_ms.append(task_stats.get("task_delay_avg_ms", 0.0))
        eval_task_delay_weighted_ms.append(task_stats.get("task_delay_weighted_ms", 0.0))
        eval_task_delay_high_ms.append(task_stats.get("task_delay_high_ms", 0.0))
        eval_task_delay_low_ms.append(task_stats.get("task_delay_low_ms", 0.0))
        eval_task_delay_sensor_ms.append(task_stats.get("task_delay_sensor_ms", 0.0))
        eval_task_delay_controller_ms.append(task_stats.get("task_delay_controller_ms", 0.0))
        eval_task_delay_vision_ms.append(task_stats.get("task_delay_vision_ms", 0.0))
        eval_task_completed_count.append(task_stats.get("task_completed_count", 0))
        eval_task_arrived_count.append(task_stats.get("task_arrived_count", 0))
        eval_task_completion_rate.append(task_stats.get("task_completion_rate", 0.0))
        eval_task_qos_violation_rate.append(task_stats.get("task_qos_violation_rate", 0.0))
        eval_compute_service_high_mbits.append(float(np.sum(ep_compute_service_high_mbits)))
        eval_compute_service_low_mbits.append(float(np.sum(ep_compute_service_low_mbits)))
        eval_compute_service_sensor_mbits.append(float(np.sum(ep_compute_service_sensor_mbits)))
        eval_compute_service_controller_mbits.append(
            float(np.sum(ep_compute_service_controller_mbits))
        )
        eval_compute_service_vision_mbits.append(float(np.sum(ep_compute_service_vision_mbits)))
        eval_q_total_per_uav.append(avg_q_total_per_uav)
        eval_arrival_total_per_uav.append(avg_arrival_total_per_uav)
        eval_cpu_util_per_uav.append(avg_cpu_util_per_uav)
        eval_delay_total_per_uav_ms.append(avg_delay_total_per_uav_ms)
        eval_upload_delay_cost_per_uav.append(avg_upload_delay_cost_per_uav)
        eval_upload_delay_penalty_per_uav.append(avg_upload_delay_penalty_per_uav)
        eval_task_delay_source_uav_ms.append(
            np.asarray(
                task_stats.get("task_delay_source_uav_ms", np.zeros(args.num_agents)),
                dtype=np.float32,
            )
        )
        eval_task_completed_source_count.append(
            np.asarray(
                task_stats.get("task_completed_source_count", np.zeros(args.num_agents)),
                dtype=np.float32,
            )
        )
        eval_task_arrived_source_count.append(
            np.asarray(
                task_stats.get("task_arrived_source_count", np.zeros(args.num_agents)),
                dtype=np.float32,
            )
        )
        eval_task_delay_source_high_ms.append(
            np.asarray(
                task_stats.get("task_delay_source_high_ms", np.zeros(args.num_agents)),
                dtype=np.float32,
            )
        )
        eval_task_delay_source_low_ms.append(
            np.asarray(
                task_stats.get("task_delay_source_low_ms", np.zeros(args.num_agents)),
                dtype=np.float32,
            )
        )
        eval_task_completed_source_high_count.append(
            np.asarray(
                task_stats.get("task_completed_source_high_count", np.zeros(args.num_agents)),
                dtype=np.float32,
            )
        )
        eval_task_completed_source_low_count.append(
            np.asarray(
                task_stats.get("task_completed_source_low_count", np.zeros(args.num_agents)),
                dtype=np.float32,
            )
        )
        eval_task_arrived_source_high_count.append(
            np.asarray(
                task_stats.get("task_arrived_source_high_count", np.zeros(args.num_agents)),
                dtype=np.float32,
            )
        )
        eval_task_arrived_source_low_count.append(
            np.asarray(
                task_stats.get("task_arrived_source_low_count", np.zeros(args.num_agents)),
                dtype=np.float32,
            )
        )
        eval_energy_total_episode_per_uav.append(sum_energy_total_per_uav)
        eval_energy_comp_episode_per_uav.append(sum_energy_comp_per_uav)
        eval_energy_uplink_episode_per_uav.append(sum_energy_uplink_per_uav)
        eval_energy_trans_episode_per_uav.append(sum_energy_trans_per_uav)
        eval_energy_macro_episode_per_uav.append(sum_energy_macro_per_uav)
        total_episode_energy = float(np.sum(sum_energy_total_per_uav))
        eval_energy_total_episode.append(total_episode_energy)
        episode_duration_s = max(
            float(args.macro_steps) * float(eval_env.env.macro_dt),
            1e-8,
        )
        eval_average_power_w.append(total_episode_energy / episode_duration_s)
        eval_average_battery_power_per_uav_w.append(
            sum_uav_battery_energy_per_uav / episode_duration_s
        )
        eval_compute_service_per_uav_mbits.append(sum_compute_service_per_uav_mbits)
        eval_coop_flow_mbits.append(final_coop_flow_mbits)
        eval_tx_service_matrix_mbits.append(sum_tx_service_matrix_mbits)
        eval_split_ratios_high.append(avg_split_ratios_high)
        eval_split_ratios_low.append(avg_split_ratios_low)
        eval_pressure_per_uav.append(avg_pressure_per_uav)
        eval_pressure_imbalance.append(avg_pressure_imbalance)
        eval_coop_ratio.append(final_coop_ratio)

    average_power_w = (
        float(np.mean(eval_average_power_w))
        if eval_average_power_w
        else 0.0
    )
    power_sample_count = len(eval_average_power_w)
    average_power_std_w = (
        float(np.std(eval_average_power_w, ddof=1))
        if power_sample_count > 1
        else 0.0
    )
    average_power_sem_w = average_power_std_w / np.sqrt(
        max(power_sample_count, 1)
    )
    if power_sample_count > 1:
        degrees_freedom = power_sample_count - 1
        average_power_ci95_half_w = float(
            student_t.ppf(0.975, degrees_freedom)
        ) * average_power_sem_w
        average_power_ucb95_w = average_power_w + float(
            student_t.ppf(0.95, degrees_freedom)
        ) * average_power_sem_w
    else:
        average_power_ci95_half_w = float("inf")
        average_power_ucb95_w = float("inf")
    power_budget_w = float(getattr(eval_env, "power_budget_w", 0.0))
    power_violation_w = (
        max(average_power_w - power_budget_w, 0.0)
        if power_budget_w > 0.0
        else 0.0
    )
    power_violation_ratio = (
        power_violation_w / power_budget_w
        if power_budget_w > 0.0
        else 0.0
    )
    power_ucb95_violation_w = (
        max(average_power_ucb95_w - power_budget_w, 0.0)
        if power_budget_w > 0.0
        else 0.0
    )
    if eval_average_battery_power_per_uav_w:
        battery_power_samples = np.stack(
            eval_average_battery_power_per_uav_w,
            axis=0,
        ).astype(np.float64)
        average_battery_power_per_uav_w = np.mean(
            battery_power_samples,
            axis=0,
        )
        if battery_power_samples.shape[0] > 1:
            battery_power_sem_per_uav_w = np.std(
                battery_power_samples,
                axis=0,
                ddof=1,
            ) / np.sqrt(battery_power_samples.shape[0])
            battery_power_ucb95_per_uav_w = (
                average_battery_power_per_uav_w
                + float(
                    student_t.ppf(
                        0.95,
                        battery_power_samples.shape[0] - 1,
                    )
                )
                * battery_power_sem_per_uav_w
            )
        else:
            battery_power_ucb95_per_uav_w = np.full(
                args.num_agents,
                np.inf,
                dtype=np.float64,
            )
    else:
        average_battery_power_per_uav_w = np.zeros(
            args.num_agents,
            dtype=np.float64,
        )
        battery_power_ucb95_per_uav_w = np.full(
            args.num_agents,
            np.inf,
            dtype=np.float64,
        )
    uav_power_budgets_w = np.asarray(
        getattr(eval_env, "uav_power_budgets_w", np.zeros(args.num_agents)),
        dtype=np.float64,
    )
    battery_power_violation_per_uav_w = np.maximum(
        average_battery_power_per_uav_w - uav_power_budgets_w,
        0.0,
    )
    battery_power_ucb95_violation_per_uav_w = np.maximum(
        battery_power_ucb95_per_uav_w - uav_power_budgets_w,
        0.0,
    )
    max_battery_power_violation_w = float(
        np.max(battery_power_violation_per_uav_w)
    )
    max_battery_power_ucb95_violation_w = float(
        np.max(battery_power_ucb95_violation_per_uav_w)
    )
    if str(getattr(eval_env, "power_constraint_scope", "system")) == "per_uav":
        energy_feasible = float(max_battery_power_violation_w <= 1e-8)
        energy_feasible_ucb95 = float(
            max_battery_power_ucb95_violation_w <= 1e-8
        )
    else:
        energy_feasible = float(power_violation_w <= 1e-8)
        energy_feasible_ucb95 = float(power_ucb95_violation_w <= 1e-8)
    energy_propulsion_episode = (
        float(np.mean([np.sum(item) for item in eval_energy_macro_episode_per_uav]))
        if eval_energy_macro_episode_per_uav
        else 0.0
    )
    energy_uplink_episode = (
        float(np.mean([np.sum(item) for item in eval_energy_uplink_episode_per_uav]))
        if eval_energy_uplink_episode_per_uav
        else 0.0
    )
    energy_u2u_episode = (
        float(np.mean([np.sum(item) for item in eval_energy_trans_episode_per_uav]))
        if eval_energy_trans_episode_per_uav
        else 0.0
    )
    energy_compute_episode = (
        float(np.mean([np.sum(item) for item in eval_energy_comp_episode_per_uav]))
        if eval_energy_comp_episode_per_uav
        else 0.0
    )
    mean_total_episode_energy = (
        float(np.mean(eval_energy_total_episode))
        if eval_energy_total_episode
        else 0.0
    )
    energy_component_residual = mean_total_episode_energy - (
        energy_propulsion_episode
        + energy_uplink_episode
        + energy_u2u_episode
        + energy_compute_episode
    )
    residual_gate_values = np.asarray(
        getattr(micro_policy, "residual_gate_values", []),
        dtype=np.float64,
    )
    if residual_gate_values.size:
        residual_gate_mean = float(np.mean(residual_gate_values))
        residual_gate_std = float(np.std(residual_gate_values))
        residual_gate_q10 = float(np.quantile(residual_gate_values, 0.10))
        residual_gate_q90 = float(np.quantile(residual_gate_values, 0.90))
    else:
        residual_gate_mean = 1.0
        residual_gate_std = 0.0
        residual_gate_q10 = 1.0
        residual_gate_q90 = 1.0

    return {
        "score": float(np.mean(eval_scores)),
        "J": float(np.mean(eval_J)),
        "delay_objective": float(np.mean(eval_delay_objective)),
        "legacy_score": float(np.mean(eval_legacy_scores)),
        "q_high": float(np.mean(eval_q_high)),
        "q_low": float(np.mean(eval_q_low)),
        "energy": float(np.mean(eval_energy)),
        "service_energy_per_micro_step_j": float(np.mean(eval_energy)),
        "average_power_w": average_power_w,
        "average_power_std_w": average_power_std_w,
        "average_power_sem_w": average_power_sem_w,
        "power_sample_count": float(power_sample_count),
        "average_power_ucb95_w": average_power_ucb95_w,
        "average_power_ci95_low_w": (
            average_power_w - average_power_ci95_half_w
        ),
        "average_power_ci95_high_w": (
            average_power_w + average_power_ci95_half_w
        ),
        "power_budget_w": power_budget_w,
        "power_violation_w": power_violation_w,
        "power_violation_ratio": power_violation_ratio,
        "energy_feasible": energy_feasible,
        "power_ucb95_violation_w": power_ucb95_violation_w,
        "energy_feasible_ucb95": energy_feasible_ucb95,
        "average_battery_power_per_uav_w": (
            average_battery_power_per_uav_w.astype(np.float32)
        ),
        "battery_power_ucb95_per_uav_w": (
            battery_power_ucb95_per_uav_w.astype(np.float32)
        ),
        "uav_power_budgets_w": uav_power_budgets_w.astype(np.float32),
        "battery_power_violation_per_uav_w": (
            battery_power_violation_per_uav_w.astype(np.float32)
        ),
        "battery_power_ucb95_violation_per_uav_w": (
            battery_power_ucb95_violation_per_uav_w.astype(np.float32)
        ),
        "max_battery_power_violation_w": max_battery_power_violation_w,
        "max_battery_power_ucb95_violation_w": (
            max_battery_power_ucb95_violation_w
        ),
        "power_constraint_scope": str(
            getattr(eval_env, "power_constraint_scope", "system")
        ),
        "energy_propulsion_episode": energy_propulsion_episode,
        "energy_uplink_episode": energy_uplink_episode,
        "energy_u2u_episode": energy_u2u_episode,
        "energy_compute_episode": energy_compute_episode,
        "energy_component_residual": energy_component_residual,
        "micro_decision_count": float(micro_decision_count),
        "micro_decision_seconds": float(micro_decision_seconds),
        "micro_decision_mean_ms": (
            1000.0 * micro_decision_seconds / max(micro_decision_count, 1)
        ),
        "micro_decision_p95_ms": (
            1000.0 * float(np.percentile(micro_decision_durations_s, 95.0))
            if micro_decision_durations_s
            else 0.0
        ),
        "residual_gate_mean": residual_gate_mean,
        "residual_gate_std": residual_gate_std,
        "residual_gate_q10": residual_gate_q10,
        "residual_gate_q90": residual_gate_q90,
        "u2g": float(np.mean(eval_u2g)),
        "cpu_util": float(np.mean(eval_cpu_util)),
        "cpu_saturation": float(np.mean(eval_cpu_saturation)),
        "cpu_high_util": float(np.mean(eval_cpu_high_util)),
        "cpu_low_util": float(np.mean(eval_cpu_low_util)),
        "f_high_ghz": float(np.mean(eval_f_high_ghz)),
        "f_low_ghz": float(np.mean(eval_f_low_ghz)),
        "f_total_ghz": float(np.mean(eval_f_total_ghz)),
        "delay_high": float(np.mean(eval_delay_high)),
        "delay_low": float(np.mean(eval_delay_low)),
        "delay_mean_ms": float(np.mean(eval_delay_mean_ms)),
        "delay_weighted_ms": float(np.mean(eval_delay_weighted_ms)),
        "delay_max_ms": float(np.mean(eval_delay_max_ms)),
        "task_delay_avg_ms": float(np.mean(eval_task_delay_avg_ms)),
        "task_delay_weighted_ms": float(np.mean(eval_task_delay_weighted_ms)),
        "task_delay_high_ms": float(np.mean(eval_task_delay_high_ms)),
        "task_delay_low_ms": float(np.mean(eval_task_delay_low_ms)),
        "task_delay_sensor_ms": float(np.mean(eval_task_delay_sensor_ms)),
        "task_delay_controller_ms": float(np.mean(eval_task_delay_controller_ms)),
        "task_delay_vision_ms": float(np.mean(eval_task_delay_vision_ms)),
        "task_completed_count": float(np.mean(eval_task_completed_count)),
        "task_arrived_count": float(np.mean(eval_task_arrived_count)),
        "task_completion_rate": float(np.mean(eval_task_completion_rate)),
        "task_qos_violation_rate": float(np.mean(eval_task_qos_violation_rate)),
        "compute_service_high_mbits": float(np.mean(eval_compute_service_high_mbits)),
        "compute_service_low_mbits": float(np.mean(eval_compute_service_low_mbits)),
        "compute_service_sensor_mbits": float(np.mean(eval_compute_service_sensor_mbits)),
        "compute_service_controller_mbits": float(np.mean(eval_compute_service_controller_mbits)),
        "compute_service_vision_mbits": float(np.mean(eval_compute_service_vision_mbits)),
        "q_total_per_uav": np.mean(np.stack(eval_q_total_per_uav, axis=0), axis=0)
        if eval_q_total_per_uav
        else np.zeros(args.num_agents, dtype=np.float32),
        "arrival_total_per_uav": np.mean(np.stack(eval_arrival_total_per_uav, axis=0), axis=0)
        if eval_arrival_total_per_uav
        else np.zeros(args.num_agents, dtype=np.float32),
        "cpu_util_per_uav": np.mean(np.stack(eval_cpu_util_per_uav, axis=0), axis=0)
        if eval_cpu_util_per_uav
        else np.zeros(args.num_agents, dtype=np.float32),
        "delay_total_per_uav_ms": np.mean(np.stack(eval_delay_total_per_uav_ms, axis=0), axis=0)
        if eval_delay_total_per_uav_ms
        else np.zeros(args.num_agents, dtype=np.float32),
        "upload_delay_cost_per_uav": np.mean(np.stack(eval_upload_delay_cost_per_uav, axis=0), axis=0)
        if eval_upload_delay_cost_per_uav
        else np.zeros(args.num_agents, dtype=np.float32),
        "upload_delay_penalty_per_uav": np.mean(np.stack(eval_upload_delay_penalty_per_uav, axis=0), axis=0)
        if eval_upload_delay_penalty_per_uav
        else np.zeros(args.num_agents, dtype=np.float32),
        "task_delay_source_uav_ms": np.mean(np.stack(eval_task_delay_source_uav_ms, axis=0), axis=0)
        if eval_task_delay_source_uav_ms
        else np.zeros(args.num_agents, dtype=np.float32),
        "task_completed_source_count": np.mean(np.stack(eval_task_completed_source_count, axis=0), axis=0)
        if eval_task_completed_source_count
        else np.zeros(args.num_agents, dtype=np.float32),
        "task_arrived_source_count": np.mean(np.stack(eval_task_arrived_source_count, axis=0), axis=0)
        if eval_task_arrived_source_count
        else np.zeros(args.num_agents, dtype=np.float32),
        "task_delay_source_high_ms": np.mean(np.stack(eval_task_delay_source_high_ms, axis=0), axis=0)
        if eval_task_delay_source_high_ms
        else np.zeros(args.num_agents, dtype=np.float32),
        "task_delay_source_low_ms": np.mean(np.stack(eval_task_delay_source_low_ms, axis=0), axis=0)
        if eval_task_delay_source_low_ms
        else np.zeros(args.num_agents, dtype=np.float32),
        "task_completed_source_high_count": np.mean(np.stack(eval_task_completed_source_high_count, axis=0), axis=0)
        if eval_task_completed_source_high_count
        else np.zeros(args.num_agents, dtype=np.float32),
        "task_completed_source_low_count": np.mean(np.stack(eval_task_completed_source_low_count, axis=0), axis=0)
        if eval_task_completed_source_low_count
        else np.zeros(args.num_agents, dtype=np.float32),
        "task_arrived_source_high_count": np.mean(np.stack(eval_task_arrived_source_high_count, axis=0), axis=0)
        if eval_task_arrived_source_high_count
        else np.zeros(args.num_agents, dtype=np.float32),
        "task_arrived_source_low_count": np.mean(np.stack(eval_task_arrived_source_low_count, axis=0), axis=0)
        if eval_task_arrived_source_low_count
        else np.zeros(args.num_agents, dtype=np.float32),
        "energy_total_episode_per_uav": np.mean(np.stack(eval_energy_total_episode_per_uav, axis=0), axis=0)
        if eval_energy_total_episode_per_uav
        else np.zeros(args.num_agents, dtype=np.float32),
        "energy_comp_episode_per_uav": np.mean(np.stack(eval_energy_comp_episode_per_uav, axis=0), axis=0)
        if eval_energy_comp_episode_per_uav
        else np.zeros(args.num_agents, dtype=np.float32),
        "energy_uplink_episode_per_uav": np.mean(np.stack(eval_energy_uplink_episode_per_uav, axis=0), axis=0)
        if eval_energy_uplink_episode_per_uav
        else np.zeros(args.num_agents, dtype=np.float32),
        "energy_trans_episode_per_uav": np.mean(np.stack(eval_energy_trans_episode_per_uav, axis=0), axis=0)
        if eval_energy_trans_episode_per_uav
        else np.zeros(args.num_agents, dtype=np.float32),
        "energy_macro_episode_per_uav": np.mean(np.stack(eval_energy_macro_episode_per_uav, axis=0), axis=0)
        if eval_energy_macro_episode_per_uav
        else np.zeros(args.num_agents, dtype=np.float32),
        "energy_total_episode": mean_total_episode_energy,
        "compute_service_per_uav_mbits": np.mean(np.stack(eval_compute_service_per_uav_mbits, axis=0), axis=0)
        if eval_compute_service_per_uav_mbits
        else np.zeros(args.num_agents, dtype=np.float32),
        "coop_flow_mbits": np.mean(np.stack(eval_coop_flow_mbits, axis=0), axis=0)
        if eval_coop_flow_mbits
        else np.zeros((args.num_agents, args.num_agents), dtype=np.float32),
        "tx_service_matrix_mbits": np.mean(np.stack(eval_tx_service_matrix_mbits, axis=0), axis=0)
        if eval_tx_service_matrix_mbits
        else np.zeros((args.num_agents, args.num_agents), dtype=np.float32),
        "split_ratios_high": np.mean(np.stack(eval_split_ratios_high, axis=0), axis=0)
        if eval_split_ratios_high
        else np.zeros((args.num_agents, args.num_agents), dtype=np.float32),
        "split_ratios_low": np.mean(np.stack(eval_split_ratios_low, axis=0), axis=0)
        if eval_split_ratios_low
        else np.zeros((args.num_agents, args.num_agents), dtype=np.float32),
        "pressure_per_uav": np.mean(np.stack(eval_pressure_per_uav, axis=0), axis=0)
        if eval_pressure_per_uav
        else np.zeros(args.num_agents, dtype=np.float32),
        "pressure_imbalance": float(np.mean(eval_pressure_imbalance))
        if eval_pressure_imbalance
        else 0.0,
        "coop_ratio": float(np.mean(eval_coop_ratio)) if eval_coop_ratio else 0.0,
    }


def maybe_append_final_evaluation(
    eval_history: EvaluationHistory,
    args: argparse.Namespace,
    eval_env: HRL_Wrapper,
    macro_policy: MacroPolicy,
    micro_policy: MicroPolicy,
    on_episode_start: Optional[EpisodeStartHook] = None,
) -> None:
    if eval_history.steps and eval_history.steps[-1] == args.episodes:
        return

    metrics = evaluate_policy(
        eval_env,
        args,
        macro_policy,
        micro_policy,
        on_episode_start=on_episode_start,
    )
    eval_history.append(args.episodes, metrics)
    print_evaluation_summary(args.episodes, metrics)


def save_experiment_artifacts(
    results_dir: str,
    csv_filename: str,
    eval_history: EvaluationHistory,
    queue_history: List[float],
    energy_history: List[float],
) -> str:
    csv_path = os.path.join(results_dir, csv_filename)
    eval_history.save_csv(csv_path)
    detail_path = os.path.splitext(csv_path)[0] + "_detail.npz"
    eval_history.save_detail_npz(detail_path)
    print(f"Saved evaluation CSV -> {csv_path}")
    print(f"Saved evaluation detail -> {detail_path}")
    return csv_path


def run_fixed_policy_benchmark(
    experiment_label: str,
    args: argparse.Namespace,
    csv_filename: str,
    macro_policy: MacroPolicy,
    micro_policy: MicroPolicy,
    on_train_episode_start: Optional[EpisodeStartHook] = None,
    on_eval_episode_start: Optional[EpisodeStartHook] = None,
) -> Dict[str, str]:
    set_global_seeds(args.seed)
    results_dirs = prepare_results_dirs(args.results_subdir)
    save_scenario_metadata(results_dirs, args, experiment_label)

    env = make_env_from_args(
        args,
        record_trajectory=False,
    )
    eval_env = make_env_from_args(args, record_trajectory=False)

    history_avg_queue: List[float] = []
    history_total_energy: List[float] = []
    eval_history = EvaluationHistory()

    for episode in range(args.episodes):
        should_record_trajectory = (
            args.trajectory_plot_interval > 0
            and (episode + 1) % args.trajectory_plot_interval == 0
        )
        env.env.record_trajectory = should_record_trajectory
        env.reset(seed=args.seed + episode)
        if on_train_episode_start is not None:
            on_train_episode_start(env, args.seed + episode)

        trace = EpisodeTrace()
        episode_macro_reward = 0.0
        episode_micro_reward = 0.0
        done = False

        mappo_local_state, _ = env.get_macro_state()

        for _ in range(args.macro_steps):
            local_obs_batch = np.vstack(mappo_local_state).astype(np.float32)
            macro_action = np.asarray(
                macro_policy(env, local_obs_batch, False),
                dtype=np.float32,
            )
            _, _, mac_info = env.macro_step(macro_action)
            trace.record_macro_info(mac_info)

            for _ in range(env.env.resource_frames_per_macro):
                service_paused = bool(
                    getattr(env.env, "pause_service_during_flight", False)
                    and env.env.macro_service_pause_steps_remaining > 0
                )
                if service_paused:
                    micro_action = np.zeros(
                        env.micro_action_space.shape[0],
                        dtype=np.float32,
                    )
                else:
                    env.prepare_resource_frame()
                    micro_action = np.asarray(
                        micro_policy(env, False),
                        dtype=np.float32,
                    )
                frame = env.step_resource_frame(micro_action)
                done = bool(frame["done"])
                episode_micro_reward += float(frame["reward_sum"])
                for raw_state, mic_info in zip(
                    frame["raw_states"],
                    frame["infos"],
                ):
                    trace.record_micro_step(raw_state, mic_info)
                if done:
                    break

            macro_reward = env.get_macro_reward()
            mappo_local_state, _ = env.get_macro_state()
            episode_macro_reward += float(macro_reward)
            trace.record_macro_reward(macro_reward, env.get_last_macro_reward_info())

            if done:
                break

        summary = trace.summary()
        history_avg_queue.append(summary["avg_queue"])
        history_total_energy.append(summary["total_energy"])

        print_episode_summary(
            experiment_label=experiment_label,
            episode_idx=episode,
            episode_macro_reward=episode_macro_reward,
            episode_micro_reward=episode_micro_reward,
            trace=trace,
        )

        maybe_save_episode_snapshots(
            env,
            results_dirs,
            args.trajectory_plot_interval,
            episode,
            trace,
        )

        if args.eval_interval > 0 and (episode + 1) % args.eval_interval == 0:
            eval_metrics = evaluate_policy(
                eval_env,
                args,
                macro_policy,
                micro_policy,
                on_episode_start=on_eval_episode_start,
            )
            eval_history.append(episode + 1, eval_metrics)
            print_evaluation_summary(episode + 1, eval_metrics)

    maybe_append_final_evaluation(
        eval_history,
        args,
        eval_env,
        macro_policy,
        micro_policy,
        on_episode_start=on_eval_episode_start,
    )
    csv_path = save_experiment_artifacts(
        results_dirs["results_dir"],
        csv_filename,
        eval_history,
        history_avg_queue,
        history_total_energy,
    )

    return {
        "results_dir": results_dirs["results_dir"],
        "csv_path": csv_path,
    }
