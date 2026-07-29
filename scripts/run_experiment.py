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
FORMAL_SEEDS = [int(value) for value in PROTOCOL["formal_seeds"]]
TRAINING_HORIZON = int(PROTOCOL["training_episodes"])
MICRO_PRETRAIN_HORIZON = int(PROTOCOL["micro_pretrain_episodes"])
MAIN_LOAD = float(PROTOCOL["load_scale"])
ZONE_LOADS_MBPS = [float(value) for value in PROTOCOL["zone_loads_mbps"]]
CPU_GHZ = float(PROTOCOL["cpu_ghz"])
NUM_UAVS = int(PROTOCOL["num_uavs"])
NUM_USERS = int(PROTOCOL["num_users"])
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
TD3_REPLAY_CAPACITY = int(PROTOCOL.get("td3_replay_capacity", 12000))
TD3_WARMUP_STEPS = int(PROTOCOL.get("td3_warmup_steps", 300))
TD3_UPDATES_PER_STEP = int(PROTOCOL.get("td3_updates_per_step", 1))
TD3_POLICY_DELAY = int(PROTOCOL.get("td3_policy_delay", 2))
TD3_ACTOR_WARMUP_UPDATES = int(
    PROTOCOL.get("td3_actor_warmup_updates", 0)
)
TD3_POLICY_NOISE = float(PROTOCOL.get("td3_policy_noise", 0.2))
TD3_NOISE_CLIP = float(PROTOCOL.get("td3_noise_clip", 0.5))
TD3_HOVER_EXPLORATION_PROB = float(
    PROTOCOL.get("td3_hover_exploration_probability", 0.0)
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
MACRO_POWER_VIOLATION_PENALTY_WEIGHT = float(
    PROTOCOL.get("macro_power_violation_penalty_weight", 0.0)
)
ENERGY_DUAL_LR = float(PROTOCOL.get("energy_dual_lr", 0.02))
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
SAC_GRAD_CLIP_NORM = float(PROTOCOL.get("sac_grad_clip_norm", 0.0))
MICRO_ACTION_ENCODING = str(
    PROTOCOL.get("micro_action_encoding", "projected_full")
)
MICRO_STICK_BREAKING_SCALE = float(
    PROTOCOL.get("micro_stick_breaking_scale", 4.0)
)
RESUME_MICRO_REPLAY = bool(PROTOCOL.get("resume_micro_replay", False))
ENERGY_DUAL_POWER_WINDOW = int(
    PROTOCOL.get("energy_dual_power_window", 5)
)


def normalized_macro_distance(distance_m, action_mode=None):
    mode = MACRO_ACTION_MODE if action_mode is None else str(action_mode)
    scale_m = (
        MACRO_WAYPOINT_RADIUS_M
        if mode in ("waypoint", "centroid_waypoint")
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
            "macro_steps": 2,
            "eval_interval": 1,
            "eval_episodes": 2,
            "trajectory_interval": 0,
            "micro_pretrain_episodes": 2,
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
            "eval_interval": 20,
            "eval_episodes": 20,
            "trajectory_interval": 100,
            "micro_pretrain_episodes": MICRO_PRETRAIN_HORIZON,
        },
        "formal": {
            "episodes": TRAINING_HORIZON,
            "seeds": FORMAL_SEEDS,
            "macro_steps": MACRO_STEPS,
            "eval_interval": 50,
            "eval_episodes": 20,
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
        "--gm_mean_speed_mps", "0.5",
        "--gm_mean_heading_rad", "0.7853981633974483",
        "--gm_speed_memory", "0.5",
        "--gm_heading_memory", "0.5",
        "--gm_speed_noise_std", "0.1",
        "--gm_heading_noise_std", "0.1",
        "--user_zone_radius", str(USER_ZONE_RADIUS_M),
        "--uplink_bandwidth_mhz", "20",
        "--u2u_bandwidth_mhz", "3",
        "--cpu_ghz", str(CPU_GHZ),
        "--cpu_alloc_mode", "priced_maxweight",
        "--cpu_power_cap_w", "0",
        "--micro_diag_bias", "0",
        "--micro_action_scale", "1",
        "--micro_action_encoding", MICRO_ACTION_ENCODING,
        "--micro_stick_breaking_scale", str(MICRO_STICK_BREAKING_SCALE),
        "--micro_reward_scale", str(MICRO_REWARD_SCALE),
        "--sac_grad_clip_norm", str(SAC_GRAD_CLIP_NORM),
        "--micro_state_mode", "pressure",
        "--objective_mode", "constrained_delay",
    ] + constraint_args + [
        "--energy_dual_lr", str(ENERGY_DUAL_LR),
        "--energy_dual_max", "20",
        "--energy_dual_state_norm", "10",
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


def micro_checkpoint_path(stage, seed):
    return os.path.join(
        "results",
        MICRO_PRETRAIN_SOURCE_VERSION,
        stage,
        f"seed{int(seed)}",
        "micro_pretrain",
        "checkpoints",
        "best.pt",
    )


def micro_replay_path(stage, seed):
    return os.path.join(
        "results",
        MICRO_PRETRAIN_SOURCE_VERSION,
        stage,
        f"seed{int(seed)}",
        "micro_pretrain",
        "checkpoints",
        "micro_replay.pt",
    )


def build_micro_pretrain(stage, seed, config, device):
    setup = dict(config)
    setup["episodes"] = int(config["micro_pretrain_episodes"])
    setup["eval_interval"] = min(20, max(setup["episodes"], 1))
    subdir = f"{RESULTS_VERSION}/{stage}/seed{seed}/micro_pretrain"
    args = [item.format(seed=seed) for item in common_args(setup, device)]
    args += [
        "--micro_solver", "sac",
        "--freeze_macro",
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
    return {
        "name": f"micro_pretrain_s{seed}",
        "method": "micro_pretrain",
        "seed": int(seed),
        "script": os.path.join(SCRIPT_DIR, "train_centralized_macro.py"),
        "args": args,
        "results_subdir": subdir,
    }


def method_specs():
    td3_common = [
        "--td3_replay_capacity", str(TD3_REPLAY_CAPACITY),
        "--td3_warmup_steps", str(TD3_WARMUP_STEPS),
        "--td3_updates_per_step", str(TD3_UPDATES_PER_STEP),
        "--td3_policy_delay", str(TD3_POLICY_DELAY),
        "--td3_actor_warmup_updates", str(TD3_ACTOR_WARMUP_UPDATES),
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
        "--td3_actor_lr", "3e-5",
        "--td3_critic_lr", "3e-4",
        "--td3_gamma", "0.99",
        "--td3_reward_scale", str(TD3_REWARD_SCALE),
        "--macro_power_violation_penalty_weight",
        str(MACRO_POWER_VIOLATION_PENALTY_WEIGHT),
        "--td3_hover_exploration_prob",
        str(TD3_HOVER_EXPLORATION_PROB),
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
    return [
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
                "--sca_update_interval", "1",
                "--sca_max_iter", "3",
                "--sca_tol", "1e-3",
                "--sca_proximal_weight", "1e-3",
            ] + ppo_common,
            "uses_micro_checkpoint": False,
        },
        {
            "method": "no_coop",
            "script": "train_centralized_td3.py",
            "args": ["--micro_solver", "local"] + td3_common,
            "uses_micro_checkpoint": False,
        },
    ]


def build_experiments(stage, seeds, config, device):
    experiments = []
    for seed in seeds:
        shared = [item.format(seed=seed) for item in common_args(config, device)]
        for spec in method_specs():
            subdir = f"{RESULTS_VERSION}/{stage}/seed{seed}/{spec['method']}"
            method_args = list(spec["args"])
            if spec["uses_micro_checkpoint"]:
                method_args += [
                    "--micro_ckpt", micro_checkpoint_path(stage, seed),
                    "--sac_actor_lr", str(JOINT_SAC_ACTOR_LR),
                    "--sac_critic_lr", str(JOINT_SAC_CRITIC_LR),
                    "--sac_alpha_lr", str(JOINT_SAC_ALPHA_LR),
                    "--sac_initial_alpha", str(JOINT_SAC_INITIAL_ALPHA),
                    "--sac_policy_delay", str(JOINT_SAC_POLICY_DELAY),
                    "--sac_actor_warmup_updates",
                    str(JOINT_SAC_ACTOR_WARMUP_UPDATES),
                ]
                if RESUME_MICRO_REPLAY:
                    method_args += [
                        "--micro_replay", micro_replay_path(stage, seed),
                    ]
                method_args += ["--resume_micro_optimizer", "0"]
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
            "cppo_sca",
            "no_coop",
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
    if args.setup_only or any(
        exp["method"] in {"joint_td3_sac", "joint_ppo_sac"}
        for exp in experiments
    ):
        for seed in seeds:
            checkpoint = os.path.join(PROJECT_ROOT, micro_checkpoint_path(args.stage, seed))
            replay = os.path.join(PROJECT_ROOT, micro_replay_path(args.stage, seed))
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
                    build_micro_pretrain(args.stage, seed, config, args.device)
                )
    if args.setup_only:
        experiments = []

    audit_summary = compute_capacity_audit()
    if not audit_summary["globally_feasible"]:
        raise RuntimeError(
            "Main protocol is globally infeasible according to the current "
            f"environment: {audit_summary}"
        )
    if audit_summary["locally_overloaded"] != [True, False, False]:
        raise RuntimeError(
            "Main protocol no longer has exactly one overloaded source zone: "
            f"{audit_summary}"
        )
    print("Candidate long-term research protocol")
    print(f"  stage={args.stage}, episodes={config['episodes']}, seeds={list(seeds)}")
    print(f"  methods={[exp['method'] for exp in experiments]}")
    print(f"  capacity_audit={audit_summary}")
    print(
        "  CPU=priced_maxweight, aggregate_cpu_cap=disabled, "
        f"power_constraint={POWER_CONSTRAINT_SCOPE}, "
        f"system_average_budget={SYSTEM_POWER_BUDGET_W:.1f}W"
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
