"""Run the clean two-timescale joint HRL experiment protocol."""

import argparse
import json
import math
import os
import subprocess
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hrl_uav.env import UAV_MEC_Env

DEFAULT_CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "configs",
    "main_experiment.json",
)
config_parser = argparse.ArgumentParser(add_help=False)
config_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
config_args, _ = config_parser.parse_known_args()
CONFIG_PATH = os.path.abspath(
    config_args.config
    if os.path.isabs(config_args.config)
    else os.path.join(PROJECT_ROOT, config_args.config)
)
with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
    PROTOCOL = json.load(config_file)

RESULTS_VERSION = str(PROTOCOL["results_version"])
MICRO_PRETRAIN_SOURCE_VERSION = str(
    PROTOCOL.get("micro_pretrain_source_results_version", RESULTS_VERSION)
)
_micro_pretrain_source_stage = PROTOCOL.get("micro_pretrain_source_stage", None)
MICRO_PRETRAIN_SOURCE_STAGE = (
    None
    if _micro_pretrain_source_stage is None
    else str(_micro_pretrain_source_stage).lower()
)
if MICRO_PRETRAIN_SOURCE_STAGE not in {
    None,
    "smoke",
    "diagnostic",
    "pilot",
    "formal",
}:
    raise ValueError(
        "micro_pretrain_source_stage must be smoke, diagnostic, pilot, or formal"
    )
_micro_pretrain_source_seed = PROTOCOL.get("micro_pretrain_source_seed", None)
MICRO_PRETRAIN_SOURCE_SEED = (
    None
    if _micro_pretrain_source_seed is None
    else int(_micro_pretrain_source_seed)
)
MICRO_PRETRAIN_SOURCE_METHOD = str(
    PROTOCOL.get("micro_pretrain_source_method", "micro_pretrain")
).strip()
if MICRO_PRETRAIN_SOURCE_METHOD not in {"micro_pretrain", "joint_td3_sac"}:
    raise ValueError(
        "micro_pretrain_source_method must be micro_pretrain or joint_td3_sac"
    )
FREEZE_MICRO_IN_JOINT = bool(
    PROTOCOL.get("freeze_micro_in_joint", False)
)
INDEPENDENT_MICRO_PRETRAINING = bool(
    PROTOCOL.get("independent_micro_pretraining", False)
)
MICRO_PRETRAIN_MACRO_POLICIES = {
    str(method): str(policy).lower()
    for method, policy in PROTOCOL.get(
        "micro_pretrain_macro_policies",
        {},
    ).items()
}
if any(
    policy not in {"zero", "fixed"}
    for policy in MICRO_PRETRAIN_MACRO_POLICIES.values()
):
    raise ValueError(
        "micro_pretrain_macro_policies values must be 'zero' or 'fixed'"
    )
FORMAL_SEEDS = [int(value) for value in PROTOCOL["formal_seeds"]]
TRAINING_HORIZON = int(PROTOCOL["training_episodes"])
MICRO_PRETRAIN_HORIZON = int(PROTOCOL["micro_pretrain_episodes"])
MICRO_PRETRAIN_MACRO_POLICY = str(
    PROTOCOL.get("micro_pretrain_macro_policy", "actor")
).lower()
if MICRO_PRETRAIN_MACRO_POLICY not in {"actor", "zero"}:
    raise ValueError(
        "micro_pretrain_macro_policy must be either 'actor' or 'zero'"
    )
MICRO_PRETRAIN_CHECKPOINT = str(
    PROTOCOL.get("micro_pretrain_checkpoint", "best")
).lower()
if MICRO_PRETRAIN_CHECKPOINT not in {"best", "final"}:
    raise ValueError("micro_pretrain_checkpoint must be either 'best' or 'final'")
MICRO_PRETRAIN_ACTOR_INIT_CHECKPOINT = str(
    PROTOCOL.get("micro_pretrain_actor_init_checkpoint", "")
).strip()
MICRO_PRETRAIN_PREFILL_EPISODES = int(
    PROTOCOL.get("micro_pretrain_prefill_episodes", 0)
)
MICRO_PRETRAIN_ACTOR_FREEZE_EPISODES = int(
    PROTOCOL.get("micro_pretrain_actor_freeze_episodes", 0)
)
if (
    MICRO_PRETRAIN_PREFILL_EPISODES < 0
    or MICRO_PRETRAIN_ACTOR_FREEZE_EPISODES < MICRO_PRETRAIN_PREFILL_EPISODES
):
    raise ValueError(
        "micro warm-start requires 0 <= prefill_episodes <= "
        "actor_freeze_episodes"
    )
if (
    MICRO_PRETRAIN_PREFILL_EPISODES > 0
    or MICRO_PRETRAIN_ACTOR_FREEZE_EPISODES > 0
) and not MICRO_PRETRAIN_ACTOR_INIT_CHECKPOINT:
    raise ValueError(
        "micro warm-start episodes require "
        "micro_pretrain_actor_init_checkpoint"
    )
MAIN_LOAD = float(PROTOCOL["load_scale"])
ZONE_LOADS_MBPS = [float(value) for value in PROTOCOL["zone_loads_mbps"]]
LOAD_SWITCH_MODE = str(
    PROTOCOL.get("load_switch_mode", "off")
).lower()
LOAD_SWITCH_PHASES_MBPS = [
    [float(item) for item in phase]
    for phase in PROTOCOL.get(
        "load_switch_phases_mbps",
        [ZONE_LOADS_MBPS],
    )
]
LOAD_SWITCH_DWELL_MIN = int(
    PROTOCOL.get("load_switch_dwell_min", 8)
)
LOAD_SWITCH_DWELL_MAX = int(
    PROTOCOL.get("load_switch_dwell_max", 12)
)
CPU_GHZ = float(PROTOCOL["cpu_ghz"])
NUM_UAVS = int(PROTOCOL["num_uavs"])
NUM_USERS = int(PROTOCOL["num_users"])
OBJECTIVE_MODE = str(
    PROTOCOL.get("objective_mode", "constrained_delay")
).lower()
CONSTRAINT_FORM = str(PROTOCOL.get("constraint_form", "legacy")).lower()
POWER_CONSTRAINT_SCOPE = str(
    PROTOCOL.get(
        "power_constraint_scope",
        "per_uav" if "uav_power_budget_w" in PROTOCOL else "system",
    )
).lower()
if POWER_CONSTRAINT_SCOPE not in {"system", "per_uav"}:
    raise ValueError(
        "power_constraint_scope must be either 'system' or 'per_uav'"
    )
UAV_POWER_BUDGET_W = float(PROTOCOL.get("uav_power_budget_w", 0.0))
SYSTEM_POWER_BUDGET_W = float(
    PROTOCOL.get(
        "power_budget_w",
        UAV_POWER_BUDGET_W * NUM_UAVS,
    )
)
MAX_DEPLOYMENT_DISTANCE_M = float(PROTOCOL["max_deployment_distance_m"])
MACRO_ACTION_MODE = str(PROTOCOL.get("macro_action_mode", "delta"))
MACRO_WAYPOINT_RADIUS_M = float(
    PROTOCOL.get("macro_waypoint_radius_m", 200.0)
)
MACRO_WAYPOINT_TOLERANCE_M = float(
    PROTOCOL.get("macro_waypoint_tolerance_m", 0.5)
)
MACRO_STEPS = int(PROTOCOL["macro_steps"])
MACRO_PERIOD_S = float(PROTOCOL["macro_period_s"])
DEPLOYMENT_S = float(PROTOCOL["deployment_s"])
RESOURCE_FRAME_S = float(PROTOCOL["resource_frame_s"])
INTEGRATION_STEP_S = float(PROTOCOL["integration_step_s"])
USER_ZONE_RADIUS_M = float(PROTOCOL["user_zone_radius_m"])
USERS_STATIC = bool(PROTOCOL.get("users_static", False))
GM_MEAN_SPEED_MPS = float(PROTOCOL.get("gm_mean_speed_mps", 0.5))
GM_MEAN_HEADING_RAD = float(
    PROTOCOL.get("gm_mean_heading_rad", math.pi / 4.0)
)
GM_HEADING_MODE = str(
    PROTOCOL.get("gm_heading_mode", "shared_mean")
).lower()
GM_SPEED_MEMORY = float(PROTOCOL.get("gm_speed_memory", 0.5))
GM_HEADING_MEMORY = float(PROTOCOL.get("gm_heading_memory", 0.5))
GM_SPEED_NOISE_STD = float(PROTOCOL.get("gm_speed_noise_std", 0.1))
GM_HEADING_NOISE_STD = float(PROTOCOL.get("gm_heading_noise_std", 0.1))
RPGM_GROUP_SPEED_MPS = float(PROTOCOL.get("rpgm_group_speed_mps", 0.35))
RPGM_GROUP_SPEED_MEMORY = float(
    PROTOCOL.get("rpgm_group_speed_memory", 0.9)
)
RPGM_GROUP_SPEED_NOISE_STD = float(
    PROTOCOL.get("rpgm_group_speed_noise_std", 0.04)
)
RPGM_GROUP_TURN_STD_RAD_SQRT_S = float(
    PROTOCOL.get("rpgm_group_turn_std_rad_sqrt_s", 0.12)
)
RPGM_LOCAL_VELOCITY_MEMORY = float(
    PROTOCOL.get("rpgm_local_velocity_memory", 0.9)
)
RPGM_LOCAL_VELOCITY_STD_MPS = float(
    PROTOCOL.get("rpgm_local_velocity_std_mps", 0.18)
)
RPGM_LOCAL_RESTORE_RATE_S2 = float(
    PROTOCOL.get("rpgm_local_restore_rate_s2", 0.0002)
)
RPGM_LOCAL_SPEED_CAP_MPS = float(
    PROTOCOL.get("rpgm_local_speed_cap_mps", 0.45)
)
CPU_ALLOC_MODE = str(PROTOCOL.get("cpu_alloc_mode", "priced_maxweight"))
DELAY_OBJECTIVE_WEIGHT = float(
    PROTOCOL.get("delay_objective_weight", 0.7)
)
ENERGY_OBJECTIVE_WEIGHT = float(
    PROTOCOL.get("energy_objective_weight", 0.3)
)
TD3_REPLAY_CAPACITY = int(PROTOCOL.get("td3_replay_capacity", 12000))
TD3_BATCH_SIZE = int(PROTOCOL.get("td3_batch_size", 128))
SAVE_MACRO_REPLAY = bool(PROTOCOL.get("save_macro_replay", False))
TD3_WARMUP_STEPS = int(PROTOCOL.get("td3_warmup_steps", 300))
TD3_UPDATES_PER_STEP = int(PROTOCOL.get("td3_updates_per_step", 1))
TD3_DELAY_N_STEP = int(PROTOCOL.get("td3_delay_n_step", 1))
TD3_REPLAY_SEQUENCE_SPAN = int(
    PROTOCOL.get("td3_replay_sequence_span", 1)
)
TD3_POLICY_DELAY = int(PROTOCOL.get("td3_policy_delay", 2))
TD3_ACTOR_LR = float(PROTOCOL.get("td3_actor_lr", 3e-5))
TD3_CRITIC_LR = float(PROTOCOL.get("td3_critic_lr", 3e-4))
TD3_ACTOR_EMA_DECAY = float(
    PROTOCOL.get("td3_actor_ema_decay", 0.0)
)
MACRO_INIT_CHECKPOINT = str(
    PROTOCOL.get("macro_init_checkpoint", "")
).strip()
MACRO_INIT_CHECKPOINTS = {
    str(method): str(path).strip()
    for method, path in PROTOCOL.get("macro_init_checkpoints", {}).items()
    if str(path).strip()
}
_KNOWN_METHODS = {
    "joint_td3_sac",
    "joint_ppo_sac",
    "joint_ppo_gated_sac",
    "cppo_sca",
    "no_coop",
    "centroid_sac",
    "centroid_gated_sac",
    "fixed_sac",
}
_unknown_macro_init_methods = set(MACRO_INIT_CHECKPOINTS) - _KNOWN_METHODS
if _unknown_macro_init_methods:
    raise ValueError(
        "macro_init_checkpoints contains unknown methods: "
        f"{sorted(_unknown_macro_init_methods)}"
    )
MACRO_INIT_CRITICS = bool(PROTOCOL.get("macro_init_critics", True))
RESUME_MACRO_OPTIMIZER = bool(
    PROTOCOL.get("resume_macro_optimizer", False)
)
TD3_ACTOR_WARMUP_UPDATES = int(
    PROTOCOL.get("td3_actor_warmup_updates", 0)
)
TD3_ACTOR_DELAY_AGGREGATION = str(
    PROTOCOL.get("td3_actor_delay_aggregation", "q1")
).lower()
if TD3_ACTOR_DELAY_AGGREGATION not in {"q1", "min"}:
    raise ValueError("td3_actor_delay_aggregation must be 'q1' or 'min'")
TD3_POLICY_NOISE = float(PROTOCOL.get("td3_policy_noise", 0.2))
TD3_NOISE_CLIP = float(PROTOCOL.get("td3_noise_clip", 0.5))
TD3_HOVER_EXPLORATION_PROB = float(
    PROTOCOL.get("td3_hover_exploration_probability", 0.0)
)
TD3_NEUTRAL_EPISODE_PROB = float(
    PROTOCOL.get("td3_neutral_episode_probability", 0.0)
)
TD3_EXPLORATION_PERSISTENCE = str(
    PROTOCOL.get("td3_exploration_persistence", "episode")
)
TD3_WARMUP_ACTION_RADIUS_M = float(
    PROTOCOL.get("td3_warmup_action_radius_m", 6.0)
)
TD3_EXPLORATION_INITIAL_M = float(
    PROTOCOL.get("td3_exploration_initial_m", 4.0)
)
TD3_EXPLORATION_FINAL_M = float(
    PROTOCOL.get("td3_exploration_final_m", 0.5)
)
TD3_EXPLORATION_DECAY_EPISODES = int(
    PROTOCOL.get("td3_exploration_decay_episodes", TRAINING_HORIZON)
)
TD3_REWARD_SCALE = float(PROTOCOL.get("td3_reward_scale", 0.01))
PILOT_EVAL_INTERVAL = int(PROTOCOL.get("pilot_eval_interval", 20))
MACRO_POWER_VIOLATION_PENALTY_WEIGHT = float(
    PROTOCOL.get("macro_power_violation_penalty_weight", 0.0)
)
ENERGY_DUAL_LR = float(PROTOCOL.get("energy_dual_lr", 0.02))
SYSTEM_DUAL_STATE = bool(PROTOCOL.get("system_dual_state", False))
PPO_EXPLORATION_INITIAL_M = float(
    PROTOCOL.get("ppo_exploration_initial_m", 2.0)
)
PPO_EXPLORATION_FINAL_M = float(
    PROTOCOL.get("ppo_exploration_final_m", 0.1)
)
JOINT_SAC_ACTOR_LR = float(PROTOCOL.get("joint_sac_actor_lr", 3e-4))
JOINT_SAC_CRITIC_LR = float(PROTOCOL.get("joint_sac_critic_lr", 3e-4))
JOINT_SAC_ALPHA_LR = float(PROTOCOL.get("joint_sac_alpha_lr", 3e-4))
JOINT_SAC_INITIAL_ALPHA = float(
    PROTOCOL.get("joint_sac_initial_alpha", 1.0)
)
JOINT_SAC_POLICY_DELAY = int(
    PROTOCOL.get("joint_sac_policy_delay", 1)
)
JOINT_SAC_ACTOR_WARMUP_UPDATES = int(
    PROTOCOL.get("joint_sac_actor_warmup_updates", 0)
)
JOINT_TRAIN_MICRO_DURING_MACRO_WARMUP = bool(
    PROTOCOL.get("joint_train_micro_during_macro_warmup", False)
)
MICRO_PRETRAIN_SAC_ACTOR_LR = float(
    PROTOCOL.get("micro_pretrain_sac_actor_lr", 3e-4)
)
MICRO_PRETRAIN_SAC_CRITIC_LR = float(
    PROTOCOL.get("micro_pretrain_sac_critic_lr", 3e-4)
)
MICRO_PRETRAIN_SAC_ALPHA_LR = float(
    PROTOCOL.get("micro_pretrain_sac_alpha_lr", 3e-4)
)
MICRO_PRETRAIN_SAC_INITIAL_ALPHA = float(
    PROTOCOL.get("micro_pretrain_sac_initial_alpha", 1.0)
)
MICRO_REWARD_SCALE = float(PROTOCOL.get("micro_reward_scale", 10.0))
U2U_STATE_REPRESENTATION = str(
    PROTOCOL.get("u2u_state_representation", "nominal_rate")
).lower()
if U2U_STATE_REPRESENTATION not in {
    "nominal_rate",
    "spectral_efficiency",
}:
    raise ValueError("Unsupported u2u_state_representation")
U2U_GOAL_REPRESENTATION = str(
    PROTOCOL.get("u2u_goal_representation", "match_observation")
).lower()
if U2U_GOAL_REPRESENTATION not in {"match_observation", "nominal_rate"}:
    raise ValueError("Unsupported u2u_goal_representation")
_contention_pressure = PROTOCOL.get("u2u_contention_aware_pressure", None)
U2U_CONTENTION_AWARE_PRESSURE = (
    -1 if _contention_pressure is None else int(bool(_contention_pressure))
)
U2U_PREV_ACTIVE_STATE = bool(
    PROTOCOL.get("u2u_prev_active_state", False)
)

if LOAD_SWITCH_MODE not in {"off", "cyclic_random"}:
    raise ValueError("Unsupported load_switch_mode")
if LOAD_SWITCH_MODE != "off":
    if len(LOAD_SWITCH_PHASES_MBPS) < 2:
        raise ValueError("Load switching requires at least two phases")
    if any(len(phase) != NUM_UAVS for phase in LOAD_SWITCH_PHASES_MBPS):
        raise ValueError("Every load-switch phase must match num_uavs")
    phase_totals = [sum(phase) for phase in LOAD_SWITCH_PHASES_MBPS]
    if any(abs(total - phase_totals[0]) > 1e-9 for total in phase_totals):
        raise ValueError("Load-switch phases must preserve total load")
    if not (0 < LOAD_SWITCH_DWELL_MIN <= LOAD_SWITCH_DWELL_MAX):
        raise ValueError("Invalid load-switch dwell range")
SAC_GRAD_CLIP_NORM = float(PROTOCOL.get("sac_grad_clip_norm", 0.0))
MICRO_ACTION_ENCODING = str(
    PROTOCOL.get("micro_action_encoding", "projected_full")
)
MICRO_STICK_BREAKING_SCALE = float(
    PROTOCOL.get("micro_stick_breaking_scale", 4.0)
)
MICRO_PRIORITY_WEIGHT_RATIO = float(
    PROTOCOL.get("micro_priority_weight_ratio", 10.0)
)
MICRO_SCA_RESIDUAL_SCALE = float(
    PROTOCOL.get("micro_sca_residual_scale", 0.0)
)
MICRO_SCA_SUPPORT_THRESHOLD = float(
    PROTOCOL.get("micro_sca_support_threshold", 1e-6)
)
MICRO_SCA_RESIDUAL_UPDATE_INTERVAL = int(
    PROTOCOL.get("micro_sca_residual_update_interval", 1)
)
MICRO_SCA_RESIDUAL_MAX_ITER = int(
    PROTOCOL.get("micro_sca_residual_max_iter", 3)
)
MICRO_SCA_RESIDUAL_TOL = float(
    PROTOCOL.get("micro_sca_residual_tol", 1e-3)
)
MICRO_SCA_RESIDUAL_SOLVER = str(
    PROTOCOL.get("micro_sca_residual_solver", "CLARABEL")
)
MICRO_SCA_RESIDUAL_PROXIMAL_WEIGHT = float(
    PROTOCOL.get("micro_sca_residual_proximal_weight", 1e-3)
)
ENABLE_AUTONOMOUS_RESIDUAL_GATE_METHODS = bool(
    PROTOCOL.get("enable_autonomous_residual_gate_methods", False)
)
ENABLED_METHODS = tuple(
    str(method) for method in PROTOCOL.get("enabled_methods", [])
)
if set(ENABLED_METHODS) - _KNOWN_METHODS:
    raise ValueError(
        "enabled_methods contains unknown methods: "
        f"{sorted(set(ENABLED_METHODS) - _KNOWN_METHODS)}"
    )
SAC_ZERO_MEAN_OUTPUT_INIT = bool(
    PROTOCOL.get("sac_zero_mean_output_init", False)
)
if MICRO_SCA_RESIDUAL_SCALE < 0.0:
    raise ValueError("micro_sca_residual_scale must be non-negative")
if MICRO_SCA_RESIDUAL_SCALE > 0.0 and MICRO_ACTION_ENCODING != "priority_gated":
    raise ValueError(
        "SCA-guided residual SAC requires micro_action_encoding='priority_gated'"
    )
RESUME_MICRO_REPLAY = bool(PROTOCOL.get("resume_micro_replay", False))
RESUME_MICRO_OPTIMIZER_IN_JOINT = bool(
    PROTOCOL.get("resume_micro_optimizer_in_joint", False)
)
ENERGY_DUAL_POWER_WINDOW = int(
    PROTOCOL.get("energy_dual_power_window", 5)
)

if CONSTRAINT_FORM == "pure_lagrangian":
    if OBJECTIVE_MODE != "constrained_delay":
        raise ValueError(
            "pure_lagrangian requires objective_mode='constrained_delay'"
        )
    if MACRO_POWER_VIOLATION_PENALTY_WEIGHT != 0.0:
        raise ValueError(
            "pure_lagrangian cannot also use the fixed macro power penalty"
        )
if (
    OBJECTIVE_MODE == "constrained_delay"
    and POWER_CONSTRAINT_SCOPE == "system"
    and ENERGY_DUAL_LR > 0.0
    and not SYSTEM_DUAL_STATE
):
    raise ValueError(
        "A nonzero learned system dual requires system_dual_state=true"
    )


def normalized_macro_distance(distance_m, action_mode=None):
    mode = MACRO_ACTION_MODE if action_mode is None else str(action_mode)
    scale_m = (
        MACRO_WAYPOINT_RADIUS_M
        if mode in ("waypoint", "centroid_waypoint", "cooperation_waypoint")
        else MAX_DEPLOYMENT_DISTANCE_M
    )
    return float(distance_m) / scale_m


def ppo_log_std_for_distance(distance_m):
    return math.log(normalized_macro_distance(distance_m))


def compute_capacity_audit():
    env = UAV_MEC_Env(
        num_uavs=NUM_UAVS,
        num_users=NUM_USERS,
        zone_loads=ZONE_LOADS_MBPS,
        load_scale=MAIN_LOAD,
        arrival_dt=RESOURCE_FRAME_S,
        resource_frame_dt=RESOURCE_FRAME_S,
        deployment_dt=DEPLOYMENT_S,
        arrivals_during_deployment=True,
        users_static=USERS_STATIC,
        user_zone_radius=USER_ZONE_RADIUS_M,
    )
    env.F_max_array[:] = CPU_GHZ * 1e9
    phase_audits = []
    phases = (
        LOAD_SWITCH_PHASES_MBPS
        if LOAD_SWITCH_MODE != "off"
        else [ZONE_LOADS_MBPS]
    )
    for phase_index, phase_loads in enumerate(phases):
        env.set_zone_target_loads(phase_loads)
        audit = env.get_compute_capacity_audit()
        phase_audits.append(
            {
                "phase_index": int(phase_index),
                "offered_load_mbps": audit["offered_load_mbps"].tolist(),
                "locally_overloaded": audit["locally_overloaded"].tolist(),
                "globally_feasible": bool(audit["globally_feasible"]),
                "global_margin_mbits": float(audit["global_margin_mbits"]),
            }
        )
    env.set_zone_target_loads(phases[0])
    audit = env.get_compute_capacity_audit()
    return {
        "load_scale": MAIN_LOAD,
        "arrival_mbits_per_macro": audit[
            "arrival_mbits_per_macro"
        ].tolist(),
        "compute_capacity_mbits_per_macro": audit[
            "compute_capacity_mbits_per_macro"
        ].tolist(),
        "locally_overloaded": audit["locally_overloaded"].tolist(),
        "globally_feasible": bool(audit["globally_feasible"]),
        "global_margin_mbits": float(audit["global_margin_mbits"]),
        "phase_audits": phase_audits,
    }


def parse_seed_list(value):
    seeds = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def stage_config(stage):
    configs = {
        "smoke": {
            "episodes": 2,
            "seeds": [1],
            "macro_steps": int(PROTOCOL.get("smoke_macro_steps", 2)),
            "eval_interval": 1,
            "eval_episodes": 2,
            "trajectory_interval": 0,
            "micro_pretrain_episodes": 2,
            "sac_batch_size": int(PROTOCOL.get("smoke_sac_batch_size", 256)),
            "sac_memory_warmup": int(
                PROTOCOL.get("smoke_sac_memory_warmup", 1000)
            ),
            "joint_sac_policy_delay": int(
                PROTOCOL.get(
                    "smoke_joint_sac_policy_delay",
                    JOINT_SAC_POLICY_DELAY,
                )
            ),
            "joint_sac_actor_warmup_updates": int(
                PROTOCOL.get(
                    "smoke_joint_sac_actor_warmup_updates",
                    JOINT_SAC_ACTOR_WARMUP_UPDATES,
                )
            ),
            "td3_warmup_steps": int(
                PROTOCOL.get("smoke_td3_warmup_steps", TD3_WARMUP_STEPS)
            ),
            "td3_batch_size": int(
                PROTOCOL.get("smoke_td3_batch_size", TD3_BATCH_SIZE)
            ),
        },
        "diagnostic": {
            "episodes": 20,
            "seeds": [1],
            "macro_steps": MACRO_STEPS,
            "eval_interval": 5,
            "eval_episodes": 5,
            "trajectory_interval": 20,
            "micro_pretrain_episodes": 20,
        },
        "pilot": {
            "episodes": 100,
            "seeds": [1],
            "macro_steps": MACRO_STEPS,
            "eval_interval": PILOT_EVAL_INTERVAL,
            "eval_episodes": 20,
            "trajectory_interval": 100,
            "micro_pretrain_episodes": MICRO_PRETRAIN_HORIZON,
        },
        "formal": {
            "episodes": TRAINING_HORIZON,
            "seeds": FORMAL_SEEDS,
            "macro_steps": MACRO_STEPS,
            "eval_interval": int(PROTOCOL.get("formal_eval_interval", 50)),
            "eval_episodes": int(PROTOCOL.get("formal_eval_episodes", 20)),
            "trajectory_interval": 100,
            "micro_pretrain_episodes": MICRO_PRETRAIN_HORIZON,
        },
    }
    return dict(configs[stage])


def common_args(config, device):
    budgets = ",".join([f"{UAV_POWER_BUDGET_W:.1f}"] * NUM_UAVS)
    duals = ",".join(["0"] * NUM_UAVS)
    if POWER_CONSTRAINT_SCOPE == "system":
        constraint_args = [
            "--power_constraint_scope", "system",
            "--power_budget_w", str(SYSTEM_POWER_BUDGET_W),
            "--energy_dual_initial", "0",
        ]
    else:
        constraint_args = [
            "--power_constraint_scope", "per_uav",
            "--uav_power_budgets_w", budgets,
            "--power_budget_w", str(UAV_POWER_BUDGET_W * NUM_UAVS),
            "--energy_lagrange_multipliers", duals,
            "--energy_dual_initial", "0",
        ]
    return [
        "--episodes", str(config["episodes"]),
        "--macro_steps", str(config["macro_steps"]),
        "--micro_steps", str(int(round(MACRO_PERIOD_S / INTEGRATION_STEP_S))),
        "--num_agents", str(NUM_UAVS),
        "--num_users", str(NUM_USERS),
        "--eval_interval", str(config["eval_interval"]),
        "--eval_episodes", str(config["eval_episodes"]),
        "--trajectory_plot_interval", str(config["trajectory_interval"]),
        "--zone_loads", ",".join(str(value) for value in ZONE_LOADS_MBPS),
        "--load_switch_mode", LOAD_SWITCH_MODE,
        "--load_switch_phases", ";".join(
            ",".join(str(value) for value in phase)
            for phase in LOAD_SWITCH_PHASES_MBPS
        ),
        "--load_switch_dwell_min", str(LOAD_SWITCH_DWELL_MIN),
        "--load_switch_dwell_max", str(LOAD_SWITCH_DWELL_MAX),
        "--load_scale", str(MAIN_LOAD),
        "--traffic_profile", "normal",
        "--traffic_scale", "0.5",
        "--arrival_dt", str(RESOURCE_FRAME_S),
        "--resource_frame_dt", str(RESOURCE_FRAME_S),
        "--deployment_dt", str(DEPLOYMENT_S),
        "--pause_service_during_flight", "1",
        "--arrivals_during_deployment", "1",
        "--deployment_transient_energy_mode", "kinetic_lower_bound",
        "--uav_mass_kg", "2.038735983690112",
        "--users_static", "1" if USERS_STATIC else "0",
        "--gm_mean_speed_mps", str(GM_MEAN_SPEED_MPS),
        "--gm_mean_heading_rad", str(GM_MEAN_HEADING_RAD),
        "--gm_heading_mode", GM_HEADING_MODE,
        "--gm_speed_memory", str(GM_SPEED_MEMORY),
        "--gm_heading_memory", str(GM_HEADING_MEMORY),
        "--gm_speed_noise_std", str(GM_SPEED_NOISE_STD),
        "--gm_heading_noise_std", str(GM_HEADING_NOISE_STD),
        "--rpgm_group_speed_mps", str(RPGM_GROUP_SPEED_MPS),
        "--rpgm_group_speed_memory", str(RPGM_GROUP_SPEED_MEMORY),
        "--rpgm_group_speed_noise_std", str(RPGM_GROUP_SPEED_NOISE_STD),
        "--rpgm_group_turn_std_rad_sqrt_s", str(
            RPGM_GROUP_TURN_STD_RAD_SQRT_S
        ),
        "--rpgm_local_velocity_memory", str(RPGM_LOCAL_VELOCITY_MEMORY),
        "--rpgm_local_velocity_std_mps", str(RPGM_LOCAL_VELOCITY_STD_MPS),
        "--rpgm_local_restore_rate_s2", str(RPGM_LOCAL_RESTORE_RATE_S2),
        "--rpgm_local_speed_cap_mps", str(RPGM_LOCAL_SPEED_CAP_MPS),
        "--user_zone_radius", str(USER_ZONE_RADIUS_M),
        "--uplink_bandwidth_mhz", "20",
        "--u2u_bandwidth_mhz", "3",
        "--cpu_ghz", str(CPU_GHZ),
        "--cpu_alloc_mode", CPU_ALLOC_MODE,
        "--cpu_power_cap_w", "0",
        "--micro_diag_bias", "0",
        "--micro_action_scale", "1",
        "--micro_action_encoding", MICRO_ACTION_ENCODING,
        "--micro_stick_breaking_scale", str(MICRO_STICK_BREAKING_SCALE),
        "--micro_priority_weight_ratio", str(MICRO_PRIORITY_WEIGHT_RATIO),
        "--micro_sca_residual_scale", str(MICRO_SCA_RESIDUAL_SCALE),
        "--micro_sca_support_threshold", str(MICRO_SCA_SUPPORT_THRESHOLD),
        "--micro_sca_residual_update_interval", str(
            MICRO_SCA_RESIDUAL_UPDATE_INTERVAL
        ),
        "--micro_sca_residual_max_iter", str(MICRO_SCA_RESIDUAL_MAX_ITER),
        "--micro_sca_residual_tol", str(MICRO_SCA_RESIDUAL_TOL),
        "--micro_sca_residual_solver", MICRO_SCA_RESIDUAL_SOLVER,
        "--micro_sca_residual_proximal_weight", str(
            MICRO_SCA_RESIDUAL_PROXIMAL_WEIGHT
        ),
        "--sac_zero_mean_output_init", "1" if SAC_ZERO_MEAN_OUTPUT_INIT else "0",
        "--micro_reward_scale", str(MICRO_REWARD_SCALE),
        "--delay_objective_weight", str(DELAY_OBJECTIVE_WEIGHT),
        "--energy_objective_weight", str(ENERGY_OBJECTIVE_WEIGHT),
        "--sac_grad_clip_norm", str(SAC_GRAD_CLIP_NORM),
        "--sac_batch_size", str(int(config.get("sac_batch_size", 256))),
        "--sac_memory_warmup", str(
            int(config.get("sac_memory_warmup", 1000))
        ),
        "--micro_state_mode", "pressure",
        "--u2u_state_representation", U2U_STATE_REPRESENTATION,
        "--u2u_goal_representation", U2U_GOAL_REPRESENTATION,
        "--u2u_contention_aware_pressure", str(U2U_CONTENTION_AWARE_PRESSURE),
        "--u2u_prev_active_state", "1" if U2U_PREV_ACTIVE_STATE else "0",
        "--objective_mode", OBJECTIVE_MODE,
    ] + constraint_args + [
        "--energy_dual_lr", str(ENERGY_DUAL_LR),
        "--energy_dual_max", "20",
        "--energy_dual_state_norm", "10",
        "--system_dual_state", "1" if SYSTEM_DUAL_STATE else "0",
        "--checkpoint_power_rule", "ucb95",
        "--power_budget_tolerance", "0",
        "--power_constraint_margin_w", "0",
        "--macro_reward_mode", "strict_global",
        "--strict_hierarchy_validation", "1",
        "--macro_action_mode", MACRO_ACTION_MODE,
        "--macro_action_deadzone", "0.05",
        "--macro_waypoint_radius_m", str(MACRO_WAYPOINT_RADIUS_M),
        "--macro_waypoint_tolerance_m", str(MACRO_WAYPOINT_TOLERANCE_M),
        "--macro_flow_state", "1",
        "--macro_load_state", "1",
        "--macro_capacity_state", "1",
        "--macro_comm_weight", "0",
        "--seed", "{seed}",
        "--device", str(device),
    ]


def micro_pretrain_method_name(target_method=None):
    if INDEPENDENT_MICRO_PRETRAINING:
        if not target_method:
            raise ValueError(
                "Independent micro pretraining requires a target method"
            )
        return f"micro_pretrain_{target_method}"
    return MICRO_PRETRAIN_SOURCE_METHOD


def micro_checkpoint_path(stage, seed, target_method=None):
    source_stage = MICRO_PRETRAIN_SOURCE_STAGE or str(stage)
    source_seed = (
        int(seed)
        if MICRO_PRETRAIN_SOURCE_SEED is None
        else MICRO_PRETRAIN_SOURCE_SEED
    )
    return os.path.join(
        "results",
        MICRO_PRETRAIN_SOURCE_VERSION,
        source_stage,
        f"seed{source_seed}",
        micro_pretrain_method_name(target_method),
        "checkpoints",
        f"{MICRO_PRETRAIN_CHECKPOINT}.pt",
    )


def micro_replay_path(stage, seed, target_method=None):
    source_stage = MICRO_PRETRAIN_SOURCE_STAGE or str(stage)
    source_seed = (
        int(seed)
        if MICRO_PRETRAIN_SOURCE_SEED is None
        else MICRO_PRETRAIN_SOURCE_SEED
    )
    return os.path.join(
        "results",
        MICRO_PRETRAIN_SOURCE_VERSION,
        source_stage,
        f"seed{source_seed}",
        micro_pretrain_method_name(target_method),
        "checkpoints",
        "micro_replay.pt",
    )


def macro_checkpoint_path(method, stage, seed):
    template = MACRO_INIT_CHECKPOINTS.get(str(method), "")
    if not template and str(method) == "joint_td3_sac":
        template = MACRO_INIT_CHECKPOINT
    return template.format(stage=stage, seed=int(seed)) if template else ""


def build_micro_pretrain(stage, seed, config, device, target_method=None):
    setup = dict(config)
    setup["episodes"] = int(config["micro_pretrain_episodes"])
    setup["eval_interval"] = int(PROTOCOL.get(
        "micro_pretrain_eval_interval",
        min(20, max(setup["episodes"], 1)),
    ))
    setup["eval_episodes"] = int(PROTOCOL.get(
        "micro_pretrain_eval_episodes",
        setup["eval_episodes"],
    ))
    artifact_method = micro_pretrain_method_name(target_method)
    subdir = f"{RESULTS_VERSION}/{stage}/seed{seed}/{artifact_method}"
    frozen_policy = MICRO_PRETRAIN_MACRO_POLICIES.get(
        str(target_method),
        str(
            config.get(
                "micro_pretrain_macro_policy",
                MICRO_PRETRAIN_MACRO_POLICY,
            )
        ),
    )
    args = [item.format(seed=seed) for item in common_args(setup, device)]
    args += [
        "--micro_solver", "sac",
        "--freeze_macro",
        "--frozen_macro_policy", frozen_policy,
        "--ppo_zero_inflated_actions", "0",
        "--ppo_initial_log_std", str(ppo_log_std_for_distance(2.0)),
        "--macro_shaping_initial", "0",
        "--macro_shaping_final", "0",
        "--macro_move_penalty_weight", "0",
        "--macro_propulsion_tiebreak_weight", "0",
        "--advisor_weight", "0",
        "--advisor_final_weight", "0",
        "--energy_dual_power_window", str(ENERGY_DUAL_POWER_WINDOW),
        "--sac_actor_lr", str(MICRO_PRETRAIN_SAC_ACTOR_LR),
        "--sac_critic_lr", str(MICRO_PRETRAIN_SAC_CRITIC_LR),
        "--sac_alpha_lr", str(MICRO_PRETRAIN_SAC_ALPHA_LR),
        "--sac_initial_alpha", str(MICRO_PRETRAIN_SAC_INITIAL_ALPHA),
        "--results_subdir", subdir,
    ]
    if target_method in {
        "joint_ppo_gated_sac",
        "centroid_gated_sac",
    }:
        args += ["--micro_autonomous_residual_gate", "1"]
    if MICRO_PRETRAIN_ACTOR_INIT_CHECKPOINT:
        args += [
            "--micro_actor_ckpt",
            MICRO_PRETRAIN_ACTOR_INIT_CHECKPOINT.format(
                stage=stage,
                seed=seed,
            ),
            "--sac_prefill_episodes",
            str(MICRO_PRETRAIN_PREFILL_EPISODES),
            "--sac_actor_freeze_episodes",
            str(MICRO_PRETRAIN_ACTOR_FREEZE_EPISODES),
        ]
    return {
        "name": f"{artifact_method}_s{seed}",
        "method": artifact_method,
        "target_method": target_method,
        "seed": int(seed),
        "script": os.path.join(SCRIPT_DIR, "train_centralized_macro.py"),
        "args": args,
        "results_subdir": subdir,
    }


def method_specs():
    td3_common = [
        "--td3_replay_capacity", str(TD3_REPLAY_CAPACITY),
        "--td3_batch_size", str(TD3_BATCH_SIZE),
        "--save_macro_replay", str(int(SAVE_MACRO_REPLAY)),
        "--td3_warmup_steps", str(TD3_WARMUP_STEPS),
        "--td3_updates_per_step", str(TD3_UPDATES_PER_STEP),
        "--td3_delay_n_step", str(TD3_DELAY_N_STEP),
        "--td3_replay_sequence_span", str(TD3_REPLAY_SEQUENCE_SPAN),
        "--td3_policy_delay", str(TD3_POLICY_DELAY),
        "--td3_actor_warmup_updates", str(TD3_ACTOR_WARMUP_UPDATES),
        "--td3_actor_delay_aggregation", TD3_ACTOR_DELAY_AGGREGATION,
        "--td3_exploration_decay_episodes",
        str(TD3_EXPLORATION_DECAY_EPISODES),
        "--td3_exploration_persistence", TD3_EXPLORATION_PERSISTENCE,
        "--td3_warmup_action_radius",
        str(normalized_macro_distance(TD3_WARMUP_ACTION_RADIUS_M)),
        "--td3_exploration_initial",
        str(normalized_macro_distance(TD3_EXPLORATION_INITIAL_M)),
        "--td3_exploration_final",
        str(normalized_macro_distance(TD3_EXPLORATION_FINAL_M)),
        "--td3_policy_noise", str(TD3_POLICY_NOISE),
        "--td3_noise_clip", str(TD3_NOISE_CLIP),
        "--td3_actor_lr", str(TD3_ACTOR_LR),
        "--td3_critic_lr", str(TD3_CRITIC_LR),
        "--td3_actor_ema_decay", str(TD3_ACTOR_EMA_DECAY),
        "--td3_gamma", "0.99",
        "--td3_reward_scale", str(TD3_REWARD_SCALE),
        "--macro_power_violation_penalty_weight",
        str(MACRO_POWER_VIOLATION_PENALTY_WEIGHT),
        "--td3_hover_exploration_prob",
        str(TD3_HOVER_EXPLORATION_PROB),
        "--td3_neutral_episode_prob",
        str(TD3_NEUTRAL_EPISODE_PROB),
        "--td3_dual_power_window", str(ENERGY_DUAL_POWER_WINDOW),
        "--macro_shaping_initial", "0",
        "--macro_shaping_final", "0",
        "--macro_move_penalty_weight", "0",
        "--macro_propulsion_tiebreak_weight", "0",
    ]
    ppo_common = [
        "--ppo_update_episode_window", "10",
        "--ppo_gamma", "0.99",
        "--ppo_gae_lambda", "0.95",
        "--ppo_zero_inflated_actions", "0",
        "--ppo_initial_log_std",
        str(ppo_log_std_for_distance(PPO_EXPLORATION_INITIAL_M)),
        "--ppo_final_log_std",
        str(ppo_log_std_for_distance(PPO_EXPLORATION_FINAL_M)),
        "--ppo_log_std_decay_episodes", str(TRAINING_HORIZON),
        "--ppo_actor_lr", "3e-5",
        "--ppo_target_kl", "0.01",
        "--ppo_actor_final_lr", "1e-5",
        "--ppo_lr_decay_episodes", str(TRAINING_HORIZON),
        "--energy_dual_power_window", str(ENERGY_DUAL_POWER_WINDOW),
        "--macro_shaping_initial", "0",
        "--macro_shaping_final", "0",
        "--macro_move_penalty_weight", "0",
        "--macro_propulsion_tiebreak_weight", "0",
        "--advisor_weight", "0",
        "--advisor_final_weight", "0",
    ]
    specs = [
        {
            "method": "joint_td3_sac",
            "script": "train_centralized_td3.py",
            "args": ["--micro_solver", "sac"] + td3_common,
            "uses_micro_checkpoint": True,
        },
        {
            "method": "joint_ppo_sac",
            "script": "train_centralized_macro.py",
            "args": ["--micro_solver", "sac"] + ppo_common,
            "uses_micro_checkpoint": True,
        },
        {
            "method": "cppo_sca",
            "script": "train_centralized_macro.py",
            "args": [
                "--micro_solver", "sca",
                "--micro_sca_residual_scale", "0",
                "--sca_update_interval", "1",
                "--sca_max_iter", "3",
                "--sca_tol", "1e-3",
                "--sca_proximal_weight", "1e-3",
                "--sca_delay_weight", str(DELAY_OBJECTIVE_WEIGHT),
                "--sca_energy_weight", str(ENERGY_OBJECTIVE_WEIGHT),
            ] + ppo_common,
            "uses_micro_checkpoint": False,
        },
        {
            "method": "no_coop",
            "script": "train_centralized_td3.py",
            "args": [
                "--micro_solver", "local",
                "--micro_sca_residual_scale", "0",
            ] + td3_common,
            "uses_micro_checkpoint": False,
        },
        {
            "method": "centroid_sac",
            "script": "train_centralized_macro.py",
            "args": [
                "--micro_solver", "sac",
                "--freeze_macro",
                "--frozen_macro_policy", "zero",
            ] + ppo_common,
            "uses_micro_checkpoint": True,
        },
        {
            "method": "fixed_sac",
            "script": "train_centralized_macro.py",
            "args": [
                "--micro_solver", "sac",
                "--freeze_macro",
                "--frozen_macro_policy", "fixed",
            ] + ppo_common,
            "uses_micro_checkpoint": True,
        },
    ]
    if ENABLE_AUTONOMOUS_RESIDUAL_GATE_METHODS:
        specs.extend(
            [
                {
                    "method": "joint_ppo_gated_sac",
                    "script": "train_centralized_macro.py",
                    "args": [
                        "--micro_solver", "sac",
                        "--micro_autonomous_residual_gate", "1",
                    ] + ppo_common,
                    "uses_micro_checkpoint": True,
                },
                {
                    "method": "centroid_gated_sac",
                    "script": "train_centralized_macro.py",
                    "args": [
                        "--micro_solver", "sac",
                        "--micro_autonomous_residual_gate", "1",
                        "--freeze_macro",
                        "--frozen_macro_policy", "zero",
                    ] + ppo_common,
                    "uses_micro_checkpoint": True,
                },
            ]
        )
    if ENABLED_METHODS:
        specs = [spec for spec in specs if spec["method"] in ENABLED_METHODS]
    return specs


def build_experiments(stage, seeds, config, device):
    experiments = []
    for seed in seeds:
        shared = [item.format(seed=seed) for item in common_args(config, device)]
        for spec in method_specs():
            subdir = f"{RESULTS_VERSION}/{stage}/seed{seed}/{spec['method']}"
            method_args = list(spec["args"])
            if spec["method"] in {"joint_td3_sac", "no_coop"}:
                td3_overrides = {
                    "--td3_warmup_steps": int(config.get(
                        "td3_warmup_steps",
                        TD3_WARMUP_STEPS,
                    )),
                    "--td3_batch_size": int(config.get(
                        "td3_batch_size",
                        TD3_BATCH_SIZE,
                    )),
                }
                for flag, value in td3_overrides.items():
                    flag_index = method_args.index(flag)
                    method_args[flag_index + 1] = str(value)
            if spec["uses_micro_checkpoint"]:
                method_args += [
                    "--micro_ckpt", micro_checkpoint_path(
                        stage,
                        seed,
                        spec["method"],
                    ),
                    "--sac_actor_lr", str(JOINT_SAC_ACTOR_LR),
                    "--sac_critic_lr", str(JOINT_SAC_CRITIC_LR),
                    "--sac_alpha_lr", str(JOINT_SAC_ALPHA_LR),
                    "--sac_initial_alpha", str(JOINT_SAC_INITIAL_ALPHA),
                    "--sac_policy_delay", str(int(config.get(
                        "joint_sac_policy_delay",
                        JOINT_SAC_POLICY_DELAY,
                    ))),
                    "--sac_actor_warmup_updates",
                    str(int(config.get(
                        "joint_sac_actor_warmup_updates",
                        JOINT_SAC_ACTOR_WARMUP_UPDATES,
                    ))),
                ]
                if (
                    JOINT_TRAIN_MICRO_DURING_MACRO_WARMUP
                    and spec["method"] == "joint_td3_sac"
                ):
                    method_args += ["--train_micro_during_macro_warmup", "1"]
                if RESUME_MICRO_REPLAY:
                    method_args += [
                        "--micro_replay", micro_replay_path(
                            stage,
                            seed,
                            spec["method"],
                        ),
                    ]
                method_args += [
                    "--resume_micro_optimizer",
                    "1" if RESUME_MICRO_OPTIMIZER_IN_JOINT else "0",
                ]
                if FREEZE_MICRO_IN_JOINT:
                    method_args += ["--freeze_micro"]
            macro_checkpoint = macro_checkpoint_path(
                spec["method"], stage, seed
            )
            if macro_checkpoint:
                if spec["method"] in {"joint_td3_sac", "no_coop"}:
                    method_args += [
                        "--macro_init_ckpt",
                        macro_checkpoint,
                        "--macro_init_critics",
                        "1" if MACRO_INIT_CRITICS else "0",
                        "--resume_macro_optimizer",
                        "1" if RESUME_MACRO_OPTIMIZER else "0",
                    ]
                else:
                    method_args += ["--macro_ckpt", macro_checkpoint]
            experiments.append(
                {
                    "name": f"{spec['method']}_s{seed}",
                    "method": spec["method"],
                    "seed": int(seed),
                    "script": os.path.join(SCRIPT_DIR, spec["script"]),
                    "args": shared
                    + method_args
                    + ["--results_subdir", subdir],
                    "results_subdir": subdir,
                }
            )
    return experiments


def run_experiment(exp, python_cmd, stage, dry_run=False, skip_completed=False):
    final_checkpoint = os.path.join(
        PROJECT_ROOT,
        "results",
        exp["results_subdir"],
        "checkpoints",
        "final.pt",
    )
    if skip_completed and os.path.isfile(final_checkpoint):
        print(f"[SKIP] {exp['name']}: final checkpoint already exists")
        return 0

    command = [python_cmd, "-u", exp["script"]] + exp["args"]
    printable = subprocess.list2cmdline(command)
    if dry_run:
        print(f"[DRY-RUN] {exp['name']}: {printable}")
        return 0

    log_dir = os.path.join(PROJECT_ROOT, "results", RESULTS_VERSION, "_logs", stage)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{exp['name']}.log")
    print(f"\n[START] {exp['name']} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    started = time.time()
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"# CMD: {printable}\n\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    elapsed = time.time() - started
    status = "OK" if return_code == 0 else f"FAIL({return_code})"
    print(f"[{status}] {exp['name']} in {elapsed:.1f}s; log={log_path}")
    return return_code


def main():
    parser = argparse.ArgumentParser(
        description="Run the no-hidden-cap long-term HRL research protocol"
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument(
        "--stage",
        choices=["smoke", "diagnostic", "pilot", "formal"],
        default="smoke",
    )
    parser.add_argument(
        "--method",
        choices=[
            "joint_td3_sac",
            "joint_ppo_sac",
            "joint_ppo_gated_sac",
            "cppo_sca",
            "no_coop",
            "centroid_sac",
            "centroid_gated_sac",
            "fixed_sac",
        ],
        default="",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=parse_seed_list, default=None)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()

    config = stage_config(args.stage)
    seeds = args.seeds if args.seeds is not None else config["seeds"]
    if args.seed:
        seeds = [int(args.seed)]
    if args.episodes > 0:
        config["episodes"] = int(args.episodes)

    snapshot_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        RESULTS_VERSION,
        args.stage,
    )
    os.makedirs(snapshot_dir, exist_ok=True)
    with open(
        os.path.join(snapshot_dir, "protocol_snapshot.json"),
        "w",
        encoding="utf-8",
    ) as snapshot_file:
        json.dump(
            {
                "protocol": PROTOCOL,
                "stage": args.stage,
                "stage_config": config,
                "requested_seeds": list(seeds),
            },
            snapshot_file,
            ensure_ascii=False,
            indent=2,
        )

    experiments = build_experiments(args.stage, seeds, config, args.device)
    if args.method:
        experiments = [exp for exp in experiments if exp["method"] == args.method]

    setup_experiments = []
    setup_targets = sorted({
        exp["method"]
        for exp in experiments
        if exp["method"] in {
            "joint_td3_sac",
            "joint_ppo_sac",
            "joint_ppo_gated_sac",
            "centroid_sac",
            "centroid_gated_sac",
            "fixed_sac",
        }
    })
    if args.setup_only and not setup_targets:
        setup_targets = [
            "joint_td3_sac",
            "joint_ppo_sac",
            "centroid_sac",
            "fixed_sac",
        ]
        if ENABLE_AUTONOMOUS_RESIDUAL_GATE_METHODS:
            setup_targets += [
                "joint_ppo_gated_sac",
                "centroid_gated_sac",
            ]
    if setup_targets:
        for seed in seeds:
            artifact_targets = (
                setup_targets
                if INDEPENDENT_MICRO_PRETRAINING
                else [None]
            )
            for target_method in artifact_targets:
                checkpoint = os.path.join(
                    PROJECT_ROOT,
                    micro_checkpoint_path(
                        args.stage,
                        seed,
                        target_method,
                    ),
                )
                replay = os.path.join(
                    PROJECT_ROOT,
                    micro_replay_path(
                        args.stage,
                        seed,
                        target_method,
                    ),
                )
                setup_ready = os.path.isfile(checkpoint) and (
                    not RESUME_MICRO_REPLAY or os.path.isfile(replay)
                )
                if args.setup_only or not setup_ready:
                    if MICRO_PRETRAIN_SOURCE_VERSION != RESULTS_VERSION:
                        raise FileNotFoundError(
                            "Configured external micro pretraining artifacts do "
                            f"not both exist: checkpoint={checkpoint}, replay={replay}"
                        )
                    setup_experiments.append(
                        build_micro_pretrain(
                            args.stage,
                            seed,
                            config,
                            args.device,
                            target_method=target_method,
                        )
                    )
    if args.setup_only:
        experiments = []

    audit_summary = compute_capacity_audit()
    phase_audits = audit_summary.get("phase_audits", [])
    if any(not item["globally_feasible"] for item in phase_audits):
        raise RuntimeError(
            "Main protocol is globally infeasible according to the current "
            f"environment: {audit_summary}"
        )
    if LOAD_SWITCH_MODE == "off" and (
        audit_summary["locally_overloaded"] != [True, False, False]
    ):
        raise RuntimeError(
            "Main protocol no longer has exactly one overloaded source zone: "
            f"{audit_summary}"
        )
    if LOAD_SWITCH_MODE != "off" and any(
        sum(bool(value) for value in item["locally_overloaded"]) != 1
        for item in phase_audits
    ):
        raise RuntimeError(
            "Every load-switch phase must have exactly one overloaded zone: "
            f"{audit_summary}"
        )
    print("Candidate long-term research protocol")
    print(f"  stage={args.stage}, episodes={config['episodes']}, seeds={list(seeds)}")
    print(f"  methods={[exp['method'] for exp in experiments]}")
    print(f"  capacity_audit={audit_summary}")
    print(
        "  load_switch="
        f"{LOAD_SWITCH_MODE}, phases={LOAD_SWITCH_PHASES_MBPS}, "
        f"dwell=[{LOAD_SWITCH_DWELL_MIN},{LOAD_SWITCH_DWELL_MAX}]"
    )
    print(
        f"  CPU={CPU_ALLOC_MODE}, aggregate_cpu_cap=disabled, "
        f"power_constraint={POWER_CONSTRAINT_SCOPE}, "
        f"system_average_budget={SYSTEM_POWER_BUDGET_W:.1f}W, "
        f"constraint_form={CONSTRAINT_FORM}, "
        f"system_dual_state={int(SYSTEM_DUAL_STATE)}"
    )
    print(
        "  objective="
        f"{OBJECTIVE_MODE}, delay_weight={DELAY_OBJECTIVE_WEIGHT:.3f}, "
        f"energy_weight={ENERGY_OBJECTIVE_WEIGHT:.3f}, "
        f"mobility_heading={GM_HEADING_MODE}"
    )
    print("  deployment=arrivals+mobility, service=paused for exactly 1s")

    results = []
    for exp in setup_experiments + experiments:
        code = run_experiment(
            exp,
            args.python,
            args.stage,
            dry_run=args.dry_run,
            skip_completed=args.skip_completed,
        )
        results.append((exp["name"], code))
        if code != 0 and not args.dry_run:
            break

    print("\nSummary")
    for name, code in results:
        print(f"  [{'OK' if code == 0 else 'FAIL'}] {name}")
    if any(code != 0 for _, code in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
