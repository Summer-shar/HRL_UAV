import argparse
import csv
import json
import os

import numpy as np
import torch

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

PROJECT_ROOT = bootstrap()

from compare_common import build_micro_step_metrics
from compare_common import compute_cumulative_little_delays
from compare_common import load_macro_checkpoint
from compare_common import make_env_from_args
from compare_common import prepare_results_dirs
from hrl_uav.plot_utils import (
    plot_macro_diagnostics,
)
import hrl_uav.agents.ppo_agent as ppo_agent_mod
import hrl_uav.agents.sac_agent as sac_agent_mod
from hrl_uav.agents.ppo_agent import Actor as PPO_Actor
from hrl_uav.agents.ppo_agent import Critic as PPO_Critic
from hrl_uav.agents.ppo_agent import GAMMA as PPO_GAMMA

PPO_LAMBDA = 0.95  # [GAE] generalized advantage estimation 的 λ
from hrl_uav.agents.sac_agent import Actor as SAC_Actor
from hrl_uav.agents.sac_agent import Critic as SAC_Critic
from hrl_uav.agents.sac_agent import Entroy as SAC_Entropy
from hrl_uav.agents.sac_agent import Memory as SAC_Memory


BASE_DIR = PROJECT_ROOT


# ==================== SAC Hyperparameters ====================
# [HYPERPARAMETER FIX] 灏?SAC 鐨勮閲庢姌鎵ｄ粠 0.9 鎻愬崌鍒?0.99锛?# 鍥犱负 0.1s 鐨勬帶鍒堕鐜囦笅锛?.9 鐨勮閲庡お鐭紙浠?绉掞級锛屾棤娉曢瑙佹帓闃熷欢杩熷甫鏉ョ殑闀挎湡鐏鹃毦銆?SAC_GAMMA = 0.99
SAC_GAMMA = 0.99
SAC_BATCH_SIZE = 512
SAC_MEMORY_WARMUP = 1000
SAC_UPDATE_FREQ = 40   # 姣?0姝ユ洿鏂颁竴娆★紙鍘?0锛夛紝鍑忓皯姊害鏇存柊娆℃暟绾?x
SAC_MEMORY_CAPACITY = 500000
SAC_RECENT_WINDOW = 500000
SAC_RECENT_RATIO = 0.0
SAC_BATCH_SIZE = 256
SAC_UPDATE_FREQ = 100
SAC_GRADIENT_STEPS = 1


# ==================== Evaluation Hyperparameters ====================
EVAL_INTERVAL = 25   # 鍘?0锛屽噺灏戣瘎浼板紑閿€
EVAL_EPISODES = 5
EVAL_BASE_SEED = 1000
EVAL_INTERVAL = 100
EVAL_EPISODES = 2
EVAL_W_HIGH = 10.0
EVAL_W_LOW = 1.0
EVAL_W_ENERGY = 0.05
EVAL_W_U2G = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description="HRL UAV-MEC Training")
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Total training episodes",
    )
    parser.add_argument(
        "--macro_steps",
        type=int,
        default=30,
        help="Macro steps per episode (each = 10s)",
    )
    parser.add_argument(
        "--micro_steps",
        type=int,
        default=100,
        help="Micro steps per macro step (each = 0.1s)",
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
        help="Random seed",
    )
    parser.add_argument(
        "--trajectory_plot_interval",
        type=int,
        default=100,   # 鍘?5锛屽噺灏憁atplotlib娓叉煋寮€閿€
        help="Save one trajectory plot every N episodes; 0 disables it",
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
        default=1.0,
        help="Global multiplier for all arrival rates",
    )
    parser.add_argument(
        "--load_scale",
        type=float,
        default=1.0,
        help="Multiplier on per-zone target load (ZONE_TARGET_LOAD_MBPS); the real demand knob",
    )
    parser.add_argument(
        "--arrival_dt",
        type=float,
        default=0.1,
        help="Task-arrival interval in seconds; rates remain per-second",
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
        "--cpu_ghz",
        type=float,
        default=16.0,
        help="CPU frequency of every UAV in GHz",
    )
    parser.add_argument(
        "--cpu_alloc_mode",
        type=str,
        choices=["legacy_kkt", "priority_kkt", "fixed_full", "fixed_average"],
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
        "--micro_diag_bias",
        type=float,
        default=0.5,
        help="Diagonal retention bias before micro-action softmax",
    )
    parser.add_argument(
        "--micro_action_scale",
        type=float,
        default=1.0,
        help="Scale applied to continuous micro actions before softmax",
    )
    parser.add_argument(
        "--micro_state_mode",
        type=str,
        choices=["pressure", "legacy"],
        default="pressure",
        help="Micro-state normalization mode",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=EVAL_INTERVAL,
        help="Evaluate every N episodes; 0 disables periodic evaluation",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=EVAL_EPISODES,
        help="Number of fixed-seed evaluation episodes",
    )
    parser.add_argument(
        "--eval_base_seed",
        type=int,
        default=EVAL_BASE_SEED,
        help="Base seed for evaluation episodes",
    )
    parser.add_argument(
        "--sac_batch_size",
        type=int,
        default=SAC_BATCH_SIZE,
        help="Mini-batch size for each SAC gradient update",
    )
    parser.add_argument(
        "--sac_update_freq",
        type=int,
        default=SAC_UPDATE_FREQ,
        help="Run one SAC update every N trainable micro steps",
    )
    parser.add_argument(
        "--sac_gradient_steps",
        type=int,
        default=SAC_GRADIENT_STEPS,
        help="Number of SAC gradient steps at each update point",
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
        help="Freeze the macro MAPPO layer and only train SAC micro actions",
    )
    parser.add_argument(
        "--macro_deterministic",
        action="store_true",
        help="Use deterministic macro actions during training",
    )
    parser.add_argument(
        "--results_subdir",
        type=str,
        default="sac",
        help="Results subdirectory under ./results",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help=(
            "Compute device. Default cpu: for these small nets the env-loop "
            "batch-1 inference is far faster on CPU than GPU round-trips."
        ),
    )
    return parser.parse_args()


def choose_macro_actions(ppo_actors, local_obs_batch, deterministic=False):
    actions = []
    log_probs = []

    for actor_idx, actor in enumerate(ppo_actors):
        local_obs = local_obs_batch[actor_idx]
        if deterministic:
            action = actor.choose_action_deterministic(local_obs)
            actions.append(np.asarray(action, dtype=np.float32))
        else:
            action, log_prob = actor.choose_action(local_obs)
            actions.append(np.asarray(action, dtype=np.float32))
            log_probs.append(np.asarray(log_prob, dtype=np.float32).reshape(1))

    actions = np.asarray(actions, dtype=np.float32)
    if deterministic:
        return actions

    log_probs = np.asarray(log_probs, dtype=np.float32).reshape(len(ppo_actors), 1)
    return actions, log_probs


def evaluate_policy(eval_env, ppo_actors, sac_actor, args):
    """
    [MAPPO UPGRADE]
    浣跨敤鍙傛暟鍏变韩 MAPPO Actor 鍋氱‘瀹氭€ц瘎浼般€?    """
    eval_scores = []
    eval_q_high = []
    eval_q_low = []
    eval_energy = []
    eval_u2g = []
    eval_J = []
    eval_cpu_util = []
    eval_cpu_saturation = []
    eval_cpu_high_util = []
    eval_cpu_low_util = []
    eval_f_high_ghz = []
    eval_f_low_ghz = []
    eval_f_total_ghz = []
    eval_delay_high = []
    eval_delay_low = []
    eval_delay_mean_ms = []
    eval_delay_weighted_ms = []
    eval_delay_max_ms = []
    eval_task_delay_avg_ms = []
    eval_task_delay_weighted_ms = []
    eval_task_delay_high_ms = []
    eval_task_delay_low_ms = []
    eval_task_delay_sensor_ms = []
    eval_task_delay_controller_ms = []
    eval_task_delay_vision_ms = []
    eval_task_completed_count = []
    eval_task_arrived_count = []
    eval_task_completion_rate = []
    eval_task_qos_violation_rate = []
    eval_compute_service_high_mbits = []
    eval_compute_service_low_mbits = []
    eval_compute_service_sensor_mbits = []
    eval_compute_service_controller_mbits = []
    eval_compute_service_vision_mbits = []
    eval_task_delay_source_uav_ms = []
    eval_task_completed_source_count = []
    eval_task_arrived_source_count = []
    eval_task_delay_source_high_ms = []
    eval_task_delay_source_low_ms = []
    eval_task_completed_source_high_count = []
    eval_task_completed_source_low_count = []
    eval_task_arrived_source_high_count = []
    eval_task_arrived_source_low_count = []
    eval_energy_total_episode_per_uav = []
    eval_energy_comp_episode_per_uav = []
    eval_energy_uplink_episode_per_uav = []
    eval_energy_trans_episode_per_uav = []
    eval_energy_macro_episode_per_uav = []
    eval_energy_total_episode = []
    eval_compute_service_per_uav_mbits = []
    eval_coop_flow_mbits = []
    eval_tx_service_matrix_mbits = []
    eval_split_ratios_high = []
    eval_split_ratios_low = []
    eval_pressure_per_uav = []
    eval_pressure_imbalance = []
    eval_coop_ratio = []

    for eval_idx in range(args.eval_episodes):
        eval_env.reset(seed=args.eval_base_seed + eval_idx)

        ep_q_high = []
        ep_q_low = []
        ep_energy = []
        ep_u2g = []
        ep_J = []  # [UNIFIED J] 评估打分对齐训练目标 J
        ep_cpu_util = []
        ep_cpu_saturation = []
        ep_cpu_high_util = []
        ep_cpu_low_util = []
        ep_f_high_ghz = []
        ep_f_low_ghz = []
        ep_f_total_ghz = []
        ep_arrival_high = []
        ep_arrival_low = []
        ep_q_total_per_uav = []
        ep_arrival_total_per_uav = []
        ep_cpu_util_per_uav = []
        ep_delay_total_per_uav_ms = []
        ep_energy_comp_per_uav = []
        ep_energy_uplink_per_uav = []
        ep_energy_trans_per_uav = []
        ep_energy_macro_per_uav = []
        ep_compute_service_per_uav_mbits = []
        ep_coop_flow_mbits = []
        ep_tx_service_matrix_mbits = []
        ep_split_ratios_high = []
        ep_split_ratios_low = []
        ep_pressure_per_uav = []
        ep_pressure_imbalance = []
        ep_coop_ratio = []
        ep_compute_service_high_mbits = []
        ep_compute_service_low_mbits = []
        ep_compute_service_sensor_mbits = []
        ep_compute_service_controller_mbits = []
        ep_compute_service_vision_mbits = []

        mappo_local_state, _ = eval_env.get_macro_state()
        done = False

        for _ in range(args.macro_steps):
            # [MAPPO UPGRADE] 灏?N 涓眬閮ㄨ娴嬫嫾鎴?batch锛屽苟琛岃緭鍑?N 涓姩浣?            local_obs_batch = np.vstack(mappo_local_state).astype(np.float32)
            local_obs_batch = np.vstack(mappo_local_state).astype(np.float32)
            mac_action = choose_macro_actions(
                ppo_actors,
                local_obs_batch,
                deterministic=True,
            )

            _, _, mac_info = eval_env.macro_step(mac_action)

            if isinstance(mac_info, dict):
                ep_energy.append(mac_info.get("E_macro_total", 0.0))
                ep_energy_macro_per_uav.append(
                    np.asarray(
                        mac_info.get("E_macro_per_uav", np.zeros(args.num_agents)),
                        dtype=np.float32,
                    )
                )
                u2g_distances = mac_info.get("u2g_distances", None)
                if u2g_distances is not None:
                    ep_u2g.append(float(np.mean(u2g_distances)))

            for _ in range(args.micro_steps):
                if eval_env.env.macro_fly_steps_remaining > 0:
                    sac_action = np.zeros(
                        2 * eval_env.N * eval_env.N,
                        dtype=np.float32,
                    )
                else:
                    sac_obs = eval_env.get_sac_state()
                    sac_action = sac_actor.choose_action(
                        sac_obs,
                        deterministic=True,
                    )

                _, _, done, _, mic_info = eval_env.micro_step(sac_action)
                raw_state = eval_env.last_raw_state
                micro_metrics = build_micro_step_metrics(raw_state, mic_info)

                ep_q_high.append(micro_metrics["q_high_tot"])
                ep_q_low.append(micro_metrics["q_low_tot"])
                ep_J.append(float(mic_info.get("J_step", 0.0)))
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
                ep_compute_service_high_mbits.append(
                    micro_metrics["compute_service_high_mbits"]
                )
                ep_compute_service_low_mbits.append(
                    micro_metrics["compute_service_low_mbits"]
                )
                ep_compute_service_sensor_mbits.append(
                    micro_metrics["compute_service_sensor_mbits"]
                )
                ep_compute_service_controller_mbits.append(
                    micro_metrics["compute_service_controller_mbits"]
                )
                ep_compute_service_vision_mbits.append(
                    micro_metrics["compute_service_vision_mbits"]
                )

                if done:
                    break

            mappo_local_state, _ = eval_env.get_macro_state()

            if done:
                break

        avg_q_high = float(np.mean(ep_q_high)) if ep_q_high else 0.0
        avg_q_low = float(np.mean(ep_q_low)) if ep_q_low else 0.0
        avg_energy = float(np.mean(ep_energy)) if ep_energy else 0.0
        avg_u2g = float(np.mean(ep_u2g)) if ep_u2g else 0.0
        avg_J = float(np.mean(ep_J)) if ep_J else 0.0
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

        # [UNIFIED J] 评估打分 = 训练目标 J 的负值 (×100 仅为可读性, 不影响 argmax)。
        # 这彻底消除“训练信号涨而评估指标恶化”的错位:reward 曲线与该 score 同源。
        score = -avg_J * 100.0

        eval_scores.append(score)
        eval_q_high.append(avg_q_high)
        eval_q_low.append(avg_q_low)
        eval_energy.append(avg_energy)
        eval_u2g.append(avg_u2g)
        eval_J.append(avg_J)
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
        eval_energy_total_episode.append(float(np.sum(sum_energy_total_per_uav)))
        eval_compute_service_per_uav_mbits.append(sum_compute_service_per_uav_mbits)
        eval_coop_flow_mbits.append(final_coop_flow_mbits)
        eval_tx_service_matrix_mbits.append(sum_tx_service_matrix_mbits)
        eval_split_ratios_high.append(avg_split_ratios_high)
        eval_split_ratios_low.append(avg_split_ratios_low)
        eval_pressure_per_uav.append(avg_pressure_per_uav)
        eval_pressure_imbalance.append(avg_pressure_imbalance)
        eval_coop_ratio.append(final_coop_ratio)

    return {
        "score": float(np.mean(eval_scores)),
        "J": float(np.mean(eval_J)),
        "q_high": float(np.mean(eval_q_high)),
        "q_low": float(np.mean(eval_q_low)),
        "energy": float(np.mean(eval_energy)),
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
        "q_total_per_uav": avg_q_total_per_uav,
        "arrival_total_per_uav": avg_arrival_total_per_uav,
        "cpu_util_per_uav": avg_cpu_util_per_uav,
        "delay_total_per_uav_ms": avg_delay_total_per_uav_ms,
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
        "energy_total_episode": float(np.mean(eval_energy_total_episode))
        if eval_energy_total_episode
        else 0.0,
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


def save_trajectory_with_verification(env, save_path):
    """
    Save raw trajectory arrays and a PNG trajectory preview.
    """
    traj_env = getattr(env, "env", env)
    uav_traj = np.asarray(getattr(traj_env, "uav_trajectory", []), dtype=np.float32)
    centroid_traj = np.asarray(getattr(traj_env, "centroid_trajectory", []), dtype=np.float32)
    users_traj = np.asarray(getattr(traj_env, "users_trajectory", []), dtype=np.float32)
    if uav_traj.size == 0 or centroid_traj.size == 0 or users_traj.size == 0:
        print(f"  [SKIP] No trajectory samples available for {save_path}")
        return

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
        print(
            f"  Saved trajectory data -> {save_path} "
            f"({file_size_kb:.1f} KB)"
        )
    else:
        print(
            f"  [WARN] Trajectory data was not found after saving: "
            f"{save_path}"
        )

    png_path = os.path.splitext(save_path)[0] + ".png"
    try:
        traj_env.render_trajectory(save_path=png_path, show=False)
    except Exception as exc:
        print(f"  [WARN] Failed to render trajectory PNG: {exc}")
        return

    if os.path.exists(png_path):
        file_size_kb = os.path.getsize(png_path) / 1024.0
        print(
            f"  Saved trajectory figure -> {png_path} "
            f"({file_size_kb:.1f} KB)"
        )
    else:
        print(f"  [WARN] Trajectory figure was not found after saving: {png_path}")


def save_macro_diagnostics_with_verification(
    save_path,
    move_distance,
    action_norm,
    hover_ratio,
    projected_u2g_distance,
    projected_u2u_rate,
    avg_upload_delay_penalty,
    position_cost,
    position_improvement,
    macro_reward,
):
    """Save per-macro-step diagnostics plot and CSV for one episode."""
    macro_steps = np.arange(1, len(move_distance) + 1, dtype=np.int32)
    if len(macro_steps) == 0:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
            "macro_reward"
        ),
        comments="",
    )

    if os.path.exists(save_path):
        file_size_kb = os.path.getsize(save_path) / 1024.0
        print(
            f"  Saved macro diagnostics -> {save_path} "
            f"({file_size_kb:.1f} KB)"
        )
    else:
        print(
            f"  [WARN] Macro diagnostics plot was not found after saving: "
            f"{save_path}"
        )

    if os.path.exists(csv_path):
        csv_size_kb = os.path.getsize(csv_path) / 1024.0
        print(
            f"  Saved macro diagnostics CSV -> {csv_path} "
            f"({csv_size_kb:.1f} KB)"
        )
    else:
        print(
            f"  [WARN] Macro diagnostics CSV was not found after saving: "
            f"{csv_path}"
        )


def main():
    args = parse_args()

    # Setup Device
    # 默认 CPU:小网络 + 海量 batch-1 推理时, CPU 比 GPU 单样本往返快得多。
    # --device cuda 可切回 GPU。覆盖两个 agent 模块的全局 device, 使其方法
    # (choose_action / learn 等) 在调用时拾取到正确设备。
    requested = args.device
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU")
        requested = "cpu"
    device = torch.device(requested)
    sac_agent_mod.device = device
    ppo_agent_mod.device = device
    print(f"[INFO] Training on {str(device).upper()}")

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # ==================== 1. Initialize Environment ====================
    print("\n[INFO] Initializing HRL Environment...")
    env = make_env_from_args(
        args,
        record_trajectory=False,
    )
    eval_env = make_env_from_args(args, record_trajectory=False)

    # ==================== 2. Initialize SAC (Micro-Layer) ====================
    print("[INFO] Initializing SAC Agent (Micro-Layer)...")
    sac_state_dim = env.sac_obs_dim
    sac_action_dim = env.micro_action_space.shape[0]

    sac_actor = SAC_Actor(sac_state_dim, sac_action_dim)
    sac_critic = SAC_Critic(sac_state_dim, sac_action_dim)
    sac_entropy = SAC_Entropy(sac_action_dim)
    
    sac_memory = SAC_Memory(
        SAC_MEMORY_CAPACITY,
        2 * sac_state_dim + sac_action_dim + 1,
        recent_window=SAC_RECENT_WINDOW,
        recent_ratio=SAC_RECENT_RATIO,
    )

    # ==================== 3. Initialize MAPPO (Macro-Layer) ====================
    print("[INFO] Initializing MAPPO Agent (Macro-Layer)...")

    # [MAPPO UPGRADE] Actor 鍙湅鍗曟満灞€閮ㄧ姸鎬侊紝Critic 鐪嬪叏灞€鎷兼帴鐘舵€?    ppo_local_state_dim = env.macro_local_obs_dim
    ppo_local_state_dim = env.macro_local_obs_dim
    ppo_global_state_dim = env.macro_global_obs_dim
    ppo_action_dim = 2

    ppo_actors = [
        PPO_Actor(ppo_local_state_dim, ppo_action_dim)
        for _ in range(args.num_agents)
    ]
    ppo_critic = PPO_Critic(ppo_global_state_dim, args.num_agents)

    loaded_macro_path = load_macro_checkpoint(
        ppo_actors,
        ppo_critic,
        args.macro_ckpt,
    )
    if loaded_macro_path is not None:
        print(f"[INFO] Loaded macro MAPPO checkpoint: {loaded_macro_path}")
    if args.freeze_macro and loaded_macro_path is None:
        raise ValueError("--freeze_macro requires --macro_ckpt.")

    print("\n========== Dual Algorithm Architecture Initialized ==========")
    print(f"[INFO] Traffic profile: {args.traffic_profile}")
    print("Beginning HRL Training Loop...")

    results_dirs = prepare_results_dirs(args.results_subdir)
    results_dir = results_dirs["results_dir"]
    trajectory_dir = results_dirs["trajectory_dir"]
    macro_diag_dir = results_dirs["macro_diag_dir"]
    ckpt_dir = results_dirs["ckpt_dir"]
    meta_dir = results_dirs["meta_dir"]

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    scenario_path = os.path.join(meta_dir, "scenario.json")
    with open(scenario_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "num_agents": args.num_agents,
                "num_users": args.num_users,
                "traffic_profile": args.traffic_profile,
                "traffic_scale": args.traffic_scale,
                "arrival_dt": args.arrival_dt,
                "uplink_bandwidth_mhz": args.uplink_bandwidth_mhz,
                "u2u_bandwidth_mhz": args.u2u_bandwidth_mhz,
                "cpu_ghz": args.cpu_ghz,
                "cpu_alloc_mode": args.cpu_alloc_mode,
                "kkt_v": args.kkt_v,
                "fixed_cpu_util": args.fixed_cpu_util,
                "micro_diag_bias": args.micro_diag_bias,
                "micro_action_scale": args.micro_action_scale,
                "micro_state_mode": args.micro_state_mode,
                "episodes": args.episodes,
                "macro_steps": args.macro_steps,
                "micro_steps": args.micro_steps,
                "eval_interval": args.eval_interval,
                "eval_episodes": args.eval_episodes,
                "eval_base_seed": args.eval_base_seed,
                "seed": args.seed,
            },
            fp,
            indent=2,
        )

    # ==================== Metrics Tracking ====================
    history_macro_reward = []
    history_micro_reward = []
    history_macro_reward_per_step = []
    history_micro_reward_per_step = []
    history_micro_cost_raw = []
    history_cost_holding = []
    history_cost_queue = []
    history_task_delay_avg_ms = []
    history_task_delay_high_ms = []
    history_task_delay_low_ms = []
    history_task_completion_rate = []
    history_task_alive_count = []
    history_avg_queue = []
    history_total_energy = []
    history_energy_per_step = []

    eval_history_steps = []
    eval_history_scores = []
    eval_history_q_high = []
    eval_history_q_low = []
    eval_history_energy = []
    eval_history_u2g = []
    eval_history_cpu_util = []
    eval_history_cpu_saturation = []
    eval_history_cpu_high_util = []
    eval_history_cpu_low_util = []
    eval_history_f_high_ghz = []
    eval_history_f_low_ghz = []
    eval_history_f_total_ghz = []
    eval_history_delay_high = []
    eval_history_delay_low = []
    eval_history_delay_mean_ms = []
    eval_history_delay_weighted_ms = []
    eval_history_delay_max_ms = []
    eval_history_task_delay_avg_ms = []
    eval_history_task_delay_weighted_ms = []
    eval_history_task_delay_high_ms = []
    eval_history_task_delay_low_ms = []
    eval_history_task_delay_sensor_ms = []
    eval_history_task_delay_controller_ms = []
    eval_history_task_delay_vision_ms = []
    eval_history_task_completed_count = []
    eval_history_task_arrived_count = []
    eval_history_task_completion_rate = []
    eval_history_task_qos_violation_rate = []
    eval_history_compute_service_high_mbits = []
    eval_history_compute_service_low_mbits = []
    eval_history_compute_service_sensor_mbits = []
    eval_history_compute_service_controller_mbits = []
    eval_history_compute_service_vision_mbits = []
    eval_history_energy_total_episode = []
    eval_history_coop_ratio = []
    eval_history_task_delay_source_uav_ms = []
    eval_history_task_completed_source_count = []
    eval_history_task_arrived_source_count = []
    eval_history_task_delay_source_high_ms = []
    eval_history_task_delay_source_low_ms = []
    eval_history_task_completed_source_high_count = []
    eval_history_task_completed_source_low_count = []
    eval_history_task_arrived_source_high_count = []
    eval_history_task_arrived_source_low_count = []
    eval_history_energy_total_episode_per_uav = []
    eval_history_energy_comp_episode_per_uav = []
    eval_history_energy_uplink_episode_per_uav = []
    eval_history_energy_trans_episode_per_uav = []
    eval_history_energy_macro_episode_per_uav = []
    eval_history_compute_service_per_uav_mbits = []
    eval_history_coop_flow_mbits = []
    eval_history_tx_service_matrix_mbits = []
    eval_history_split_ratios_high = []
    eval_history_split_ratios_low = []
    eval_history_pressure_per_uav = []
    eval_history_pressure_imbalance = []

    ppo_actor_losses = []
    ppo_critic_losses = []
    sac_actor_losses = []
    sac_critic_losses = []
    best_eval_score = -np.inf
    best_eval_metric = np.inf
    best_eval_metric_name = "task_delay_avg_ms"
    best_saved_metric_name = best_eval_metric_name

    # [PPO STABILITY FIX] 鍏ㄥ眬 PPO buffer锛岀敤浜庤法 Episode 绉敀澶?Batch
    ppo_buffer_local_s = []
    ppo_buffer_global_s = []
    ppo_buffer_a = []
    ppo_buffer_a_logp = []
    ppo_buffer_discounted_r = []
    ppo_buffer_adv = []  # [GAE] 与 returns 并行存储的 GAE advantage
    sac_trainable_micro_steps = 0

    # ==================== 4. HRL Training Loop ====================
    for episode in range(args.episodes):
        should_record_trajectory = (
            args.trajectory_plot_interval > 0
            and (episode + 1) % args.trajectory_plot_interval == 0
        )
        env.env.record_trajectory = should_record_trajectory
        env.reset()

        episode_macro_reward = 0.0
        episode_micro_reward = 0.0
        done = False

        ep_queue_backlogs = []
        ep_energy_consumed = []
        ep_u2g_distance_means = []
        ep_arrival_high = []
        ep_arrival_low = []
        ep_q_loc_high = []
        ep_q_tx_high = []
        ep_q_loc_low = []
        ep_q_tx_low = []
        ep_micro_cost_raw = []
        ep_cost_holding = []
        ep_cost_queue = []
        ep_cost_wait = []
        ep_task_delay_avg_ms = []
        ep_task_completed_count = []
        ep_task_arrived_count = []
        ep_task_alive_count = []
        ep_task_completion_rate = []
        ep_task_qos_violation_rate = []
        ep_compute_service_high_mbits = []
        ep_compute_service_low_mbits = []
        ep_compute_service_sensor_mbits = []
        ep_compute_service_controller_mbits = []
        ep_compute_service_vision_mbits = []
        ep_cpu_util_mean = []
        ep_macro_move_distance = []
        ep_macro_hover_ratio = []
        ep_macro_action_norm = []
        ep_macro_u2u_rate = []
        ep_macro_projected_u2g = []
        ep_macro_avg_upload_delay_penalty = []
        ep_macro_position_cost = []
        ep_macro_position_improvement = []
        ep_macro_rewards = []

        # 褰撳墠 Episode 鐨勪复鏃?buffer
        ep_buffer_local_s = []
        ep_buffer_global_s = []
        ep_buffer_a = []
        ep_buffer_r = []
        ep_buffer_a_logp = []

        # [MAPPO UPGRADE] 鑾峰彇鍒濆瀹忚灞€閮?鍏ㄥ眬鐘舵€?        mappo_local_state, mappo_global_state = env.get_macro_state()
        mappo_local_state, mappo_global_state = env.get_macro_state()

        for macro_step in range(args.macro_steps):
            # ========== [MACRO: MAPPO Decision] ==========
            # [MAPPO UPGRADE] 灏?local obs list 杞负 (N, local_obs_dim) batch
            local_obs_batch = np.vstack(mappo_local_state).astype(np.float32)

            # [MAPPO UPGRADE] 鍙傛暟鍏变韩 Actor 涓烘瘡鏋?UAV 鐙珛杈撳嚭鍔ㄤ綔
            if args.freeze_macro or args.macro_deterministic:
                mac_action = choose_macro_actions(
                    ppo_actors,
                    local_obs_batch,
                    deterministic=True,
                )
                mac_action_logprob = np.zeros(
                    (args.num_agents, 1),
                    dtype=np.float32,
                )
            else:
                mac_action, mac_action_logprob = choose_macro_actions(
                    ppo_actors,
                    local_obs_batch,
                )

            mac_action = np.asarray(mac_action, dtype=np.float32)
            mac_action_logprob = np.asarray(
                mac_action_logprob,
                dtype=np.float32,
            ).reshape(args.num_agents, 1)

            _, _, mac_info = env.macro_step(mac_action)

            if isinstance(mac_info, dict):
                ep_energy_consumed.append(mac_info.get("E_macro_total", 0.0))
                u2g_distances = mac_info.get("u2g_distances", None)
                if u2g_distances is not None:
                    ep_u2g_distance_means.append(
                        float(np.mean(u2g_distances))
                    )
                ep_macro_move_distance.append(
                    mac_info.get("macro_move_distance_mean", 0.0)
                )
                ep_macro_hover_ratio.append(
                    mac_info.get("macro_hover_ratio", 0.0)
                )
                ep_macro_action_norm.append(
                    mac_info.get("macro_action_norm_mean", 0.0)
                )
                ep_macro_u2u_rate.append(
                    mac_info.get("projected_mean_u2u_rate_mbps", 0.0)
                )
                ep_macro_projected_u2g.append(
                    mac_info.get("projected_mean_u2g_distance_m", 0.0)
                )
            else:
                ep_macro_move_distance.append(0.0)
                ep_macro_hover_ratio.append(1.0)
                ep_macro_action_norm.append(0.0)
                ep_macro_u2u_rate.append(0.0)
                ep_macro_projected_u2g.append(0.0)

            # ========== [MICRO: SAC Control] ==========
            for micro_step in range(args.micro_steps):
                trainable_micro_step = env.env.macro_fly_steps_remaining <= 0
                if trainable_micro_step:
                    # --- SAC Observe ---
                    sac_obs = env.get_sac_state()

                    # --- SAC Act ---
                    sac_action = sac_actor.choose_action(sac_obs)
                else:
                    sac_obs = None
                    sac_action = np.zeros(sac_action_dim, dtype=np.float32)

                # --- Environment Step ---
                next_sac_obs, mic_reward, done, _, mic_info = env.micro_step(
                    sac_action
                )

                should_train_sac_step = (
                    trainable_micro_step
                    and not mic_info.get("flight_phase", False)
                )
                if should_train_sac_step:
                    sac_trainable_micro_steps += 1
                    # --- SAC Store ---
                    sac_memory.store_transition(
                        sac_obs,
                        sac_action,
                        mic_reward,
                        next_sac_obs,
                    )

                # --- SAC Train ---
                if (
                    should_train_sac_step
                    and
                    sac_memory.memory_counter > SAC_MEMORY_WARMUP
                    and sac_trainable_micro_steps % args.sac_update_freq == 0
                ):
                    for _ in range(args.sac_gradient_steps):

                        b_M = sac_memory.sample(args.sac_batch_size)

                        # Split the flat memory row: [s | a | r | s_]
                        b_s = b_M[:, :sac_state_dim]
                        b_a = b_M[
                            :,
                            sac_state_dim:sac_state_dim + sac_action_dim,
                        ]
                        b_r = b_M[
                            :,
                            sac_state_dim + sac_action_dim:
                            sac_state_dim + sac_action_dim + 1,
                        ]
                        b_s_ = b_M[:, -sac_state_dim:]

                        b_s = torch.FloatTensor(b_s).to(device)
                        b_a = torch.FloatTensor(b_a).to(device)
                        b_r = torch.FloatTensor(b_r).to(device)
                        b_s_ = torch.FloatTensor(b_s_).to(device)

                        # 1. Compute Target Q
                        new_action, log_prob_ = sac_actor.evaluate(b_s_)
                        target_q1, target_q2 = sac_critic.target_critic_v(
                            b_s_,
                            new_action,
                        )
                        target_q = b_r + SAC_GAMMA * (
                            torch.min(target_q1, target_q2)
                            - sac_entropy.alpha.detach().to(device) * log_prob_
                        )

                        # 2. Critic Update
                        current_q1, current_q2 = sac_critic.get_v(b_s, b_a)
                        sac_critic.learn(
                            current_q1,
                            current_q2,
                            target_q.detach(),
                        )

                        # 3. Actor Update
                        a_eval, log_prob = sac_actor.evaluate(b_s)
                        q1_eval, q2_eval = sac_critic.get_v(b_s, a_eval)
                        q_eval = torch.min(q1_eval, q2_eval)

                        actor_loss = (
                            sac_entropy.alpha.detach().to(device) * log_prob
                            - q_eval
                        ).mean()
                        sac_actor.learn(actor_loss)

                        # 4. Entropy Update
                        alpha_loss = -(
                            sac_entropy.log_alpha.exp().to(device)
                            * (
                                log_prob
                                + sac_entropy.target_entropy
                            ).detach()
                        ).mean()
                        sac_entropy.learn(alpha_loss)
                        sac_entropy.alpha = sac_entropy.log_alpha.exp().to(
                            device
                        )

                        # 5. Soft Update Target Network
                        sac_critic.soft_update()

                        # Record losses
                        sac_actor_losses.append(actor_loss.item())
                        sac_critic_losses.append(
                            (
                                torch.nn.functional.mse_loss(
                                    current_q1,
                                    target_q.detach(),
                                )
                                + torch.nn.functional.mse_loss(
                                    current_q2,
                                    target_q.detach(),
                                )
                            ).item()
                        )

                # --- Metrics ---
                episode_micro_reward += mic_reward

                if isinstance(mic_info, dict):
                    micro_metrics = build_micro_step_metrics(
                        env.last_raw_state,
                        mic_info,
                    )
                    ep_queue_backlogs.append(micro_metrics["queue_length"])
                    ep_energy_consumed.append(
                        micro_metrics["energy_uplink"]
                        + micro_metrics["energy_trans"]
                        + micro_metrics["energy_comp"]
                    )
                    ep_micro_cost_raw.append(micro_metrics["micro_cost_raw"])
                    ep_cost_holding.append(micro_metrics["cost_holding"])
                    ep_cost_queue.append(micro_metrics["cost_queue"])
                    ep_cost_wait.append(micro_metrics["cost_wait"])
                    ep_task_delay_avg_ms.append(
                        micro_metrics["task_delay_avg_ms"]
                    )
                    ep_task_completed_count.append(
                        micro_metrics["task_completed_count"]
                    )
                    ep_task_arrived_count.append(
                        micro_metrics["task_arrived_count"]
                    )
                    ep_task_alive_count.append(
                        micro_metrics["task_alive_count"]
                    )
                    ep_task_completion_rate.append(
                        micro_metrics["task_completion_rate"]
                    )
                    ep_task_qos_violation_rate.append(
                        micro_metrics["task_qos_violation_rate"]
                    )
                    ep_compute_service_high_mbits.append(
                        micro_metrics["compute_service_high_mbits"]
                    )
                    ep_compute_service_low_mbits.append(
                        micro_metrics["compute_service_low_mbits"]
                    )
                    ep_compute_service_sensor_mbits.append(
                        micro_metrics["compute_service_sensor_mbits"]
                    )
                    ep_compute_service_controller_mbits.append(
                        micro_metrics["compute_service_controller_mbits"]
                    )
                    ep_compute_service_vision_mbits.append(
                        micro_metrics["compute_service_vision_mbits"]
                    )
                    ep_arrival_high.append(float(np.sum(micro_metrics["arrival_high_per_uav"])))
                    ep_arrival_low.append(float(np.sum(micro_metrics["arrival_low_per_uav"])))
                    ep_q_loc_high.append(float(np.sum(micro_metrics["q_loc_high_per_uav"])))
                    ep_q_tx_high.append(float(np.sum(micro_metrics["q_tx_high_per_uav"])))
                    ep_q_loc_low.append(float(np.sum(micro_metrics["q_loc_low_per_uav"])))
                    ep_q_tx_low.append(float(np.sum(micro_metrics["q_tx_low_per_uav"])))
                    ep_cpu_util_mean.append(micro_metrics["cpu_utilization_mean"])

                if done:
                    break

            # ========== End of 10s Macro Step ==========
            next_mappo_local_state, next_mappo_global_state = (
                env.get_macro_state()
            )

            mac_reward = env.get_macro_reward()
            macro_reward_info = env.get_last_macro_reward_info()
            episode_macro_reward += mac_reward
            mac_reward_per_uav = np.full(
                args.num_agents,
                mac_reward,
                dtype=np.float32,
            )

            if isinstance(macro_reward_info, dict):
                ep_macro_avg_upload_delay_penalty.append(
                    macro_reward_info.get("avg_upload_delay_penalty", 0.0)
                )
                ep_macro_position_cost.append(
                    macro_reward_info.get("total_cost", 0.0)
                )
                ep_macro_position_improvement.append(
                    macro_reward_info.get("positioning_improvement", 0.0)
                )
                mac_reward_per_uav = np.asarray(
                    macro_reward_info.get(
                        "macro_rewards_per_uav",
                        mac_reward_per_uav,
                    ),
                    dtype=np.float32,
                ).reshape(args.num_agents)

            # 淇濆瓨澶氭櫤鑳戒綋杞ㄨ抗
            else:
                ep_macro_avg_upload_delay_penalty.append(0.0)
                ep_macro_position_cost.append(0.0)
                ep_macro_position_improvement.append(0.0)

            ep_macro_rewards.append(mac_reward)
            
            ep_buffer_local_s.append(local_obs_batch.copy())
            ep_buffer_global_s.append(mappo_global_state.copy())
            ep_buffer_a.append(mac_action.copy())
            ep_buffer_r.append(mac_reward_per_uav.copy())
            ep_buffer_a_logp.append(mac_action_logprob.copy())

            # 鐘舵€佽浆绉?            mappo_local_state = next_mappo_local_state
            mappo_global_state = next_mappo_global_state

            if done:
                break

        # ==================== Episode End: GAE Calculation ====================
        # [GAE] critic 在 20-episode 累积窗口内是冻结的, 因此在回合结束时用当前
        # critic 估计本回合各步价值来算 GAE(λ) 是自洽的。
        # δ_t = r_t + γV(s_{t+1}) - V(s_t);  A_t = δ_t + γλ A_{t+1};  R_t = A_t + V(s_t)
        if (not args.freeze_macro) and len(ep_buffer_global_s) > 0:
            T = len(ep_buffer_global_s)
            ep_global_arr = np.vstack(ep_buffer_global_s).astype(np.float32)
            ep_values = (
                ppo_critic.get_v(ep_global_arr)
                .detach()
                .cpu()
                .numpy()
                .reshape(T, args.num_agents)
            )
            v_next = (
                ppo_critic.get_v(mappo_global_state)
                .detach()
                .cpu()
                .numpy()
                .reshape(args.num_agents)
            )
            ep_rewards = np.asarray(ep_buffer_r, dtype=np.float32).reshape(
                T, args.num_agents
            )

            ep_adv = np.zeros((T, args.num_agents), dtype=np.float32)
            lastgae = np.zeros(args.num_agents, dtype=np.float32)
            for t in range(T - 1, -1, -1):
                next_v = v_next if t == T - 1 else ep_values[t + 1]
                delta = ep_rewards[t] + PPO_GAMMA * next_v - ep_values[t]
                lastgae = delta + PPO_GAMMA * PPO_LAMBDA * lastgae
                ep_adv[t] = lastgae
            ep_returns = ep_adv + ep_values  # critic 回归目标

            ppo_buffer_local_s.extend(ep_buffer_local_s)
            ppo_buffer_global_s.extend(ep_buffer_global_s)
            ppo_buffer_a.extend(ep_buffer_a)
            ppo_buffer_a_logp.extend(ep_buffer_a_logp)
            ppo_buffer_discounted_r.extend(list(ep_returns))
            ppo_buffer_adv.extend(list(ep_adv))

        # ==================== MAPPO Update ====================
        # [PPO STABILITY FIX] 绉敀 20 涓?Episode (200姝? 鍚堝苟鏇存柊锛屾秷闄ゅ崟闆嗛殢鏈烘父璧扮殑楂樻柟宸櫔闊?        if (not args.freeze_macro) and (
        if (not args.freeze_macro) and (
            len(ppo_buffer_global_s) >= args.macro_steps * 20
            or (episode == args.episodes - 1 and len(ppo_buffer_global_s) > 0)
        ):
            # [GAE] critic 目标用未标准化的 GAE returns;
            # advantage 用并行存储的 GAE advantage(标准化交给 actor.learn 内部完成)。
            returns_arr = np.array(ppo_buffer_discounted_r, dtype=np.float32)
            advantage = np.array(ppo_buffer_adv, dtype=np.float32)

            # Critic batch: (T, global_obs_dim)
            bs_global = np.vstack(ppo_buffer_global_s).astype(np.float32)
            br = returns_arr.astype(np.float32)

            # 用 GAE returns 训练 critic; 忽略其内部返回的 (returns - V) advantage。
            _, critic_loss = ppo_critic.learn(bs_global, br)

            # Actor batch: 灏?(T, N, dim) 灞曞钩涓?(T * N, dim)
            bs_local = np.asarray(ppo_buffer_local_s, dtype=np.float32)
            ba = np.asarray(ppo_buffer_a, dtype=np.float32)
            bap = np.asarray(ppo_buffer_a_logp, dtype=np.float32)

            actor_losses = []
            for agent_idx, ppo_actor in enumerate(ppo_actors):
                actor_bs = bs_local[:, agent_idx, :].astype(np.float32)
                actor_ba = ba[:, agent_idx, :].astype(np.float32)
                actor_bap = bap[:, agent_idx, :].astype(np.float32)
                actor_advantage = advantage[:, agent_idx:agent_idx + 1].astype(
                    np.float32
                )

                actor_loss = ppo_actor.learn(
                    actor_bs,
                    actor_ba,
                    actor_advantage,
                    actor_bap,
                )

                ppo_actor.update_oldpi()
                actor_losses.append(actor_loss)

            ppo_actor_losses.append(float(np.mean(actor_losses)))

            if False:
                actor_bs = bs_local.reshape(-1, ppo_local_state_dim)
                actor_ba = ba.reshape(-1, ppo_action_dim)
                actor_bap = bap.reshape(-1, 1)

            # [MAPPO UPGRADE] 鍥㈤槦浼樺娍澶嶅埗缁欐瘡鏋舵棤浜烘満
                actor_advantage = advantage.reshape(-1, 1).astype(np.float32)

                actor_loss = ppo_actor.learn(
                    actor_bs,
                    actor_ba,
                    actor_advantage,
                    actor_bap,
                )

                ppo_actor.update_oldpi()

                ppo_actor_losses.append(actor_loss)
            ppo_critic_losses.append(critic_loss)

            # 娓呯┖璺ㄩ泦鍏ㄥ眬 buffer
            ppo_buffer_local_s.clear()
            ppo_buffer_global_s.clear()
            ppo_buffer_a.clear()
            ppo_buffer_a_logp.clear()
            ppo_buffer_discounted_r.clear()
            ppo_buffer_adv.clear()

        # ==================== Record Metrics ====================
        history_macro_reward.append(episode_macro_reward)
        history_micro_reward.append(episode_micro_reward)
        history_avg_queue.append(
            np.mean(ep_queue_backlogs) if ep_queue_backlogs else 0.0
        )
        history_total_energy.append(
            np.sum(ep_energy_consumed) if ep_energy_consumed else 0.0
        )

        avg_micro_reward_per_step = episode_micro_reward / max(
            len(ep_queue_backlogs),
            1,
        )
        avg_macro_reward_per_step = episode_macro_reward / max(
            len(ep_buffer_r),
            1,
        )
        avg_u2g_distance = (
            np.mean(ep_u2g_distance_means)
            if ep_u2g_distance_means
            else 0.0
        )
        avg_arrival_high = np.mean(ep_arrival_high) if ep_arrival_high else 0.0
        avg_arrival_low = np.mean(ep_arrival_low) if ep_arrival_low else 0.0
        avg_q_loc_high = np.mean(ep_q_loc_high) if ep_q_loc_high else 0.0
        avg_q_tx_high = np.mean(ep_q_tx_high) if ep_q_tx_high else 0.0
        avg_q_loc_low = np.mean(ep_q_loc_low) if ep_q_loc_low else 0.0
        avg_q_tx_low = np.mean(ep_q_tx_low) if ep_q_tx_low else 0.0
        avg_micro_cost_raw = (
            np.mean(ep_micro_cost_raw) if ep_micro_cost_raw else 0.0
        )
        avg_cost_holding = (
            np.mean(ep_cost_holding)
            if ep_cost_holding
            else np.mean(ep_cost_wait) if ep_cost_wait else 0.0
        )
        avg_cost_queue = (
            np.mean(ep_cost_queue) if ep_cost_queue else 0.0
        )
        last_task_delay_avg_ms = (
            ep_task_delay_avg_ms[-1] if ep_task_delay_avg_ms else 0.0
        )
        task_stats = env.env.get_task_delay_stats()
        last_task_delay_high_ms = task_stats.get("task_delay_high_ms", 0.0)
        last_task_delay_low_ms = task_stats.get("task_delay_low_ms", 0.0)
        last_task_completed_count = (
            ep_task_completed_count[-1] if ep_task_completed_count else 0.0
        )
        last_task_arrived_count = (
            ep_task_arrived_count[-1] if ep_task_arrived_count else 0.0
        )
        last_task_alive_count = (
            ep_task_alive_count[-1] if ep_task_alive_count else 0.0
        )
        last_task_completion_rate = (
            ep_task_completion_rate[-1] if ep_task_completion_rate else 0.0
        )
        last_task_qos_violation_rate = (
            ep_task_qos_violation_rate[-1]
            if ep_task_qos_violation_rate
            else 0.0
        )
        sum_compute_service_high_mbits = (
            np.sum(ep_compute_service_high_mbits)
            if ep_compute_service_high_mbits
            else 0.0
        )
        sum_compute_service_low_mbits = (
            np.sum(ep_compute_service_low_mbits)
            if ep_compute_service_low_mbits
            else 0.0
        )
        sum_compute_service_sensor_mbits = (
            np.sum(ep_compute_service_sensor_mbits)
            if ep_compute_service_sensor_mbits
            else 0.0
        )
        sum_compute_service_controller_mbits = (
            np.sum(ep_compute_service_controller_mbits)
            if ep_compute_service_controller_mbits
            else 0.0
        )
        sum_compute_service_vision_mbits = (
            np.sum(ep_compute_service_vision_mbits)
            if ep_compute_service_vision_mbits
            else 0.0
        )
        avg_cpu_util_mean = (
            np.mean(ep_cpu_util_mean) if ep_cpu_util_mean else 0.0
        )
        avg_macro_move_distance = (
            np.mean(ep_macro_move_distance) if ep_macro_move_distance else 0.0
        )
        avg_macro_hover_ratio = (
            np.mean(ep_macro_hover_ratio) if ep_macro_hover_ratio else 0.0
        )
        avg_macro_action_norm = (
            np.mean(ep_macro_action_norm) if ep_macro_action_norm else 0.0
        )
        avg_macro_u2u_rate = (
            np.mean(ep_macro_u2u_rate) if ep_macro_u2u_rate else 0.0
        )
        avg_macro_position_cost = (
            np.mean(ep_macro_position_cost) if ep_macro_position_cost else 0.0
        )
        avg_macro_position_improvement = (
            np.mean(ep_macro_position_improvement)
            if ep_macro_position_improvement
            else 0.0
        )
        avg_energy_per_step = (
            np.mean(ep_energy_consumed) if ep_energy_consumed else 0.0
        )

        history_macro_reward_per_step.append(avg_macro_reward_per_step)
        history_micro_reward_per_step.append(avg_micro_reward_per_step)
        history_micro_cost_raw.append(avg_micro_cost_raw)
        history_cost_holding.append(avg_cost_holding)
        history_cost_queue.append(avg_cost_queue)
        history_task_delay_avg_ms.append(last_task_delay_avg_ms)
        history_task_delay_high_ms.append(last_task_delay_high_ms)
        history_task_delay_low_ms.append(last_task_delay_low_ms)
        history_task_completion_rate.append(last_task_completion_rate)
        history_task_alive_count.append(last_task_alive_count)
        history_energy_per_step.append(avg_energy_per_step)

        print(
            f"Episode {episode} | Macro R: {episode_macro_reward:.2f} "
            f"(avg {avg_macro_reward_per_step:.2f}/macro) | "
            f"Micro R: {episode_micro_reward:.2f} "
            f"(avg {avg_micro_reward_per_step:.2f}/micro)"
        )
        print(
            f"  Raw micro cost: {avg_micro_cost_raw:.4f} | "
            f"Queue cost: {avg_cost_queue:.4f} | "
            f"Holding cost: {avg_cost_holding:.4f} | "
            f"CPU util: {avg_cpu_util_mean:.3f} | "
            f"Ttask: {last_task_delay_avg_ms:.1f} ms | "
            f"Done: {last_task_completed_count:.0f}/"
            f"{last_task_arrived_count:.0f} "
            f"({last_task_completion_rate:.2f}) | "
            f"Alive: {last_task_alive_count:.0f} | "
            f"QoS vio: {last_task_qos_violation_rate:.2f} | "
            f"U2G mean: {avg_u2g_distance:.2f} m | "
            f"Ah: {avg_arrival_high:.2f} | "
            f"Qh_loc: {avg_q_loc_high:.2f} | "
            f"Qh_tx: {avg_q_tx_high:.2f} | "
            f"Al: {avg_arrival_low:.2f} | "
            f"Ql_loc: {avg_q_loc_low:.2f} | "
            f"Ql_tx: {avg_q_tx_low:.2f}"
        )
        print(
            f"  Macro motion: {avg_macro_move_distance:.2f} m | "
            f"Action norm: {avg_macro_action_norm:.3f} | "
            f"Hover ratio: {avg_macro_hover_ratio:.2f} | "
            f"Nominal U2U: {avg_macro_u2u_rate:.2f} Mbps | "
            f"Pos cost: {avg_macro_position_cost:.3f} | "
            f"Improve: {avg_macro_position_improvement:.3f}"
        )
        print(
            f"  Task delay(ms) H/L/S/C/V: "
            f"{task_stats.get('task_delay_high_ms', 0.0):.1f}/"
            f"{task_stats.get('task_delay_low_ms', 0.0):.1f}/"
            f"{task_stats.get('task_delay_sensor_ms', 0.0):.1f}/"
            f"{task_stats.get('task_delay_controller_ms', 0.0):.1f}/"
            f"{task_stats.get('task_delay_vision_ms', 0.0):.1f} | "
            f"CPU service(Mb) H/L/S/C/V: "
            f"{sum_compute_service_high_mbits:.2f}/"
            f"{sum_compute_service_low_mbits:.2f}/"
            f"{sum_compute_service_sensor_mbits:.2f}/"
            f"{sum_compute_service_controller_mbits:.2f}/"
            f"{sum_compute_service_vision_mbits:.2f}"
        )

        # ==================== Periodic Trajectory Visualization ====================
        if (
            args.trajectory_plot_interval > 0
            and (episode + 1) % args.trajectory_plot_interval == 0
        ):
            periodic_traj_path = os.path.join(
                trajectory_dir,
                f"trajectory_ep_{episode + 1:04d}.npz",
            )
            save_trajectory_with_verification(
                env,
                periodic_traj_path,
            )
            periodic_macro_diag_path = os.path.join(
                macro_diag_dir,
                f"macro_diag_ep_{episode + 1:04d}.png",
            )
            save_macro_diagnostics_with_verification(
                save_path=periodic_macro_diag_path,
                move_distance=ep_macro_move_distance,
                action_norm=ep_macro_action_norm,
                hover_ratio=ep_macro_hover_ratio,
                projected_u2g_distance=ep_macro_projected_u2g,
                projected_u2u_rate=ep_macro_u2u_rate,
                avg_upload_delay_penalty=ep_macro_avg_upload_delay_penalty,
                position_cost=ep_macro_position_cost,
                position_improvement=ep_macro_position_improvement,
                macro_reward=ep_macro_rewards,
            )

        if args.eval_interval > 0 and (episode + 1) % args.eval_interval == 0:
            eval_metrics = evaluate_policy(
                eval_env,
                ppo_actors,
                sac_actor,
                args,
            )

            eval_history_steps.append(episode + 1)
            eval_history_scores.append(eval_metrics["score"])
            eval_history_q_high.append(eval_metrics["q_high"])
            eval_history_q_low.append(eval_metrics["q_low"])
            eval_history_energy.append(eval_metrics["energy"])
            eval_history_u2g.append(eval_metrics["u2g"])
            eval_history_cpu_util.append(eval_metrics["cpu_util"])
            eval_history_cpu_saturation.append(eval_metrics["cpu_saturation"])
            eval_history_cpu_high_util.append(eval_metrics["cpu_high_util"])
            eval_history_cpu_low_util.append(eval_metrics["cpu_low_util"])
            eval_history_f_high_ghz.append(eval_metrics["f_high_ghz"])
            eval_history_f_low_ghz.append(eval_metrics["f_low_ghz"])
            eval_history_f_total_ghz.append(eval_metrics["f_total_ghz"])
            eval_history_delay_high.append(eval_metrics["delay_high"])
            eval_history_delay_low.append(eval_metrics["delay_low"])
            eval_history_delay_mean_ms.append(eval_metrics["delay_mean_ms"])
            eval_history_delay_weighted_ms.append(eval_metrics["delay_weighted_ms"])
            eval_history_delay_max_ms.append(eval_metrics["delay_max_ms"])
            eval_history_task_delay_avg_ms.append(eval_metrics["task_delay_avg_ms"])
            eval_history_task_delay_weighted_ms.append(
                eval_metrics["task_delay_weighted_ms"]
            )
            eval_history_task_delay_high_ms.append(eval_metrics["task_delay_high_ms"])
            eval_history_task_delay_low_ms.append(eval_metrics["task_delay_low_ms"])
            eval_history_task_delay_sensor_ms.append(eval_metrics["task_delay_sensor_ms"])
            eval_history_task_delay_controller_ms.append(eval_metrics["task_delay_controller_ms"])
            eval_history_task_delay_vision_ms.append(eval_metrics["task_delay_vision_ms"])
            eval_history_task_completed_count.append(eval_metrics["task_completed_count"])
            eval_history_task_arrived_count.append(eval_metrics["task_arrived_count"])
            eval_history_task_completion_rate.append(eval_metrics["task_completion_rate"])
            eval_history_task_qos_violation_rate.append(
                eval_metrics["task_qos_violation_rate"]
            )
            eval_history_compute_service_high_mbits.append(
                eval_metrics["compute_service_high_mbits"]
            )
            eval_history_compute_service_low_mbits.append(
                eval_metrics["compute_service_low_mbits"]
            )
            eval_history_compute_service_sensor_mbits.append(
                eval_metrics["compute_service_sensor_mbits"]
            )
            eval_history_compute_service_controller_mbits.append(
                eval_metrics["compute_service_controller_mbits"]
            )
            eval_history_compute_service_vision_mbits.append(
                eval_metrics["compute_service_vision_mbits"]
            )
            eval_history_energy_total_episode.append(
                eval_metrics["energy_total_episode"]
            )
            eval_history_coop_ratio.append(eval_metrics["coop_ratio"])
            eval_history_task_delay_source_uav_ms.append(
                np.asarray(eval_metrics["task_delay_source_uav_ms"], dtype=np.float32)
            )
            eval_history_task_completed_source_count.append(
                np.asarray(eval_metrics["task_completed_source_count"], dtype=np.float32)
            )
            eval_history_task_arrived_source_count.append(
                np.asarray(eval_metrics["task_arrived_source_count"], dtype=np.float32)
            )
            eval_history_task_delay_source_high_ms.append(
                np.asarray(eval_metrics["task_delay_source_high_ms"], dtype=np.float32)
            )
            eval_history_task_delay_source_low_ms.append(
                np.asarray(eval_metrics["task_delay_source_low_ms"], dtype=np.float32)
            )
            eval_history_task_completed_source_high_count.append(
                np.asarray(eval_metrics["task_completed_source_high_count"], dtype=np.float32)
            )
            eval_history_task_completed_source_low_count.append(
                np.asarray(eval_metrics["task_completed_source_low_count"], dtype=np.float32)
            )
            eval_history_task_arrived_source_high_count.append(
                np.asarray(eval_metrics["task_arrived_source_high_count"], dtype=np.float32)
            )
            eval_history_task_arrived_source_low_count.append(
                np.asarray(eval_metrics["task_arrived_source_low_count"], dtype=np.float32)
            )
            eval_history_energy_total_episode_per_uav.append(
                np.asarray(eval_metrics["energy_total_episode_per_uav"], dtype=np.float32)
            )
            eval_history_energy_comp_episode_per_uav.append(
                np.asarray(eval_metrics["energy_comp_episode_per_uav"], dtype=np.float32)
            )
            eval_history_energy_uplink_episode_per_uav.append(
                np.asarray(eval_metrics["energy_uplink_episode_per_uav"], dtype=np.float32)
            )
            eval_history_energy_trans_episode_per_uav.append(
                np.asarray(eval_metrics["energy_trans_episode_per_uav"], dtype=np.float32)
            )
            eval_history_energy_macro_episode_per_uav.append(
                np.asarray(eval_metrics["energy_macro_episode_per_uav"], dtype=np.float32)
            )
            eval_history_compute_service_per_uav_mbits.append(
                np.asarray(eval_metrics["compute_service_per_uav_mbits"], dtype=np.float32)
            )
            eval_history_coop_flow_mbits.append(
                np.asarray(eval_metrics["coop_flow_mbits"], dtype=np.float32)
            )
            eval_history_tx_service_matrix_mbits.append(
                np.asarray(eval_metrics["tx_service_matrix_mbits"], dtype=np.float32)
            )
            eval_history_split_ratios_high.append(
                np.asarray(eval_metrics["split_ratios_high"], dtype=np.float32)
            )
            eval_history_split_ratios_low.append(
                np.asarray(eval_metrics["split_ratios_low"], dtype=np.float32)
            )
            eval_history_pressure_per_uav.append(
                np.asarray(eval_metrics["pressure_per_uav"], dtype=np.float32)
            )
            eval_history_pressure_imbalance.append(
                eval_metrics["pressure_imbalance"]
            )

            current_best_metric = float(
                eval_metrics.get(best_eval_metric_name, np.inf)
            )
            if (
                not np.isfinite(current_best_metric)
                or current_best_metric <= 0.0
            ):
                current_best_metric = -float(eval_metrics["score"])
                current_best_metric_name = "negative_score"
            else:
                current_best_metric_name = best_eval_metric_name

            if current_best_metric < best_eval_metric:
                best_eval_metric = current_best_metric
                best_saved_metric_name = current_best_metric_name
                best_eval_score = eval_metrics["score"]
                torch.save(
                    {
                        "episode": episode + 1,
                        "best_eval_score": best_eval_score,
                        "best_eval_metric": best_eval_metric,
                        "best_eval_metric_name": current_best_metric_name,
                        "best_task_delay_avg_ms": eval_metrics.get("task_delay_avg_ms", None),
                        "best_task_delay_weighted_ms": eval_metrics.get("task_delay_weighted_ms", None),
                        "best_eval_energy": eval_metrics.get("energy", None),
                        "args": vars(args),
                        "ppo_actors": [actor.old_pi.state_dict() for actor in ppo_actors],
                        "ppo_critic": ppo_critic.critic_v.state_dict(),
                        "sac_actor": sac_actor.action_net.state_dict(),
                        "sac_critic": sac_critic.critic_v.state_dict(),
                        "sac_target_critic": sac_critic.target_critic_v.state_dict(),
                        "sac_entropy": {
                            "log_alpha": sac_entropy.log_alpha.detach().cpu(),
                            "alpha": sac_entropy.alpha.detach().cpu(),
                        },
                        "sac_memory_counter": sac_memory.memory_counter,
                    },
                    os.path.join(ckpt_dir, "best.pt"),
                )

            print(
                f"  Eval@{episode + 1} | "
                f"Score: {eval_metrics['score']:.2f} | "
                f"Dh(ms): {eval_metrics['delay_high']:.1f} | "
                f"Dl(ms): {eval_metrics['delay_low']:.1f} | "
                f"Dw(ms): {eval_metrics['delay_weighted_ms']:.1f} | "
                f"Ttask(ms): {eval_metrics['task_delay_avg_ms']:.1f} | "
                f"TwE2E(ms): {eval_metrics['task_delay_weighted_ms']:.1f} | "
                f"Done: {eval_metrics['task_completed_count']:.0f}/"
                f"{eval_metrics['task_arrived_count']:.0f} | "
                f"Dmax(ms): {eval_metrics['delay_max_ms']:.1f} | "
                f"Energy: {eval_metrics['energy']:.2f} | "
                f"Eep: {eval_metrics['energy_total_episode']:.2f} | "
                f"Coop: {eval_metrics['coop_ratio']:.2f} | "
                f"CPU sat: {eval_metrics['cpu_saturation']:.2f} | "
                f"Press imbal: {eval_metrics['pressure_imbalance']:.2f} | "
                f"KKT fh/fl(GHz): {eval_metrics['f_high_ghz']:.2f}/"
                f"{eval_metrics['f_low_ghz']:.2f} | "
                f"U2G: {eval_metrics['u2g']:.2f}"
            )
            print(
                f"  Eval task delay(ms) H/L/S/C/V: "
                f"{eval_metrics['task_delay_high_ms']:.1f}/"
                f"{eval_metrics['task_delay_low_ms']:.1f}/"
                f"{eval_metrics['task_delay_sensor_ms']:.1f}/"
                f"{eval_metrics['task_delay_controller_ms']:.1f}/"
                f"{eval_metrics['task_delay_vision_ms']:.1f} | "
                f"CPU service(Mb) H/L/S/C/V: "
                f"{eval_metrics['compute_service_high_mbits']:.2f}/"
                f"{eval_metrics['compute_service_low_mbits']:.2f}/"
                f"{eval_metrics['compute_service_sensor_mbits']:.2f}/"
                f"{eval_metrics['compute_service_controller_mbits']:.2f}/"
                f"{eval_metrics['compute_service_vision_mbits']:.2f}"
            )

    # ==================== 5. Save Artifacts ====================
    print("Training Complete! Saving artifacts...")

    train_csv_path = os.path.join(results_dir, "train_sac.csv")
    with open(train_csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "episode",
                "macro_reward",
                "micro_reward",
                "macro_reward_per_step",
                "micro_reward_per_step",
                "micro_cost_raw",
                "cost_holding",
                "cost_queue",
                "queue_backlog",
                "energy_total_episode",
                "energy_per_step",
                "task_delay_avg_ms",
                "task_delay_high_ms",
                "task_delay_low_ms",
                "task_completion_rate",
                "task_alive_count",
            ]
        )
        for idx in range(len(history_micro_reward)):
            writer.writerow(
                [
                    idx + 1,
                    history_macro_reward[idx],
                    history_micro_reward[idx],
                    history_macro_reward_per_step[idx],
                    history_micro_reward_per_step[idx],
                    history_micro_cost_raw[idx],
                    history_cost_holding[idx],
                    history_cost_queue[idx],
                    history_avg_queue[idx],
                    history_total_energy[idx],
                    history_energy_per_step[idx],
                    history_task_delay_avg_ms[idx],
                    history_task_delay_high_ms[idx],
                    history_task_delay_low_ms[idx],
                    history_task_completion_rate[idx],
                    history_task_alive_count[idx],
                ]
            )

    eval_csv_path = os.path.join(results_dir, "eval_sac.csv")
    with open(eval_csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "episode",
                "score",
                "q_high",
                "q_low",
                "energy",
                "energy_total_episode",
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
        for idx in range(len(eval_history_steps)):
            writer.writerow(
                [
                    eval_history_steps[idx],
                    eval_history_scores[idx],
                    eval_history_q_high[idx],
                    eval_history_q_low[idx],
                    eval_history_energy[idx],
                    eval_history_energy_total_episode[idx],
                    eval_history_u2g[idx],
                    eval_history_cpu_util[idx],
                    eval_history_cpu_saturation[idx],
                    eval_history_cpu_high_util[idx],
                    eval_history_cpu_low_util[idx],
                    eval_history_f_high_ghz[idx],
                    eval_history_f_low_ghz[idx],
                    eval_history_f_total_ghz[idx],
                    eval_history_delay_high[idx],
                    eval_history_delay_low[idx],
                    eval_history_delay_mean_ms[idx],
                    eval_history_delay_weighted_ms[idx],
                    eval_history_delay_max_ms[idx],
                    eval_history_task_delay_avg_ms[idx],
                    eval_history_task_delay_weighted_ms[idx],
                    eval_history_task_delay_high_ms[idx],
                    eval_history_task_delay_low_ms[idx],
                    eval_history_task_delay_sensor_ms[idx],
                    eval_history_task_delay_controller_ms[idx],
                    eval_history_task_delay_vision_ms[idx],
                    eval_history_task_completed_count[idx],
                    eval_history_task_arrived_count[idx],
                    eval_history_task_completion_rate[idx],
                    eval_history_task_qos_violation_rate[idx],
                    eval_history_pressure_imbalance[idx],
                    eval_history_coop_ratio[idx],
                    eval_history_compute_service_high_mbits[idx],
                    eval_history_compute_service_low_mbits[idx],
                    eval_history_compute_service_sensor_mbits[idx],
                    eval_history_compute_service_controller_mbits[idx],
                    eval_history_compute_service_vision_mbits[idx],
                ]
            )

    eval_detail_path = os.path.join(results_dir, "eval_sac_detail.npz")
    eval_detail_payload = {
        "steps": np.asarray(eval_history_steps, dtype=np.int32),
        "score": np.asarray(eval_history_scores, dtype=np.float32),
        "energy_per_step": np.asarray(eval_history_energy, dtype=np.float32),
        "energy_total_episode": np.asarray(
            eval_history_energy_total_episode,
            dtype=np.float32,
        ),
        "coop_ratio": np.asarray(eval_history_coop_ratio, dtype=np.float32),
        "task_delay_avg_ms": np.asarray(
            eval_history_task_delay_avg_ms,
            dtype=np.float32,
        ),
        "task_delay_weighted_ms": np.asarray(
            eval_history_task_delay_weighted_ms,
            dtype=np.float32,
        ),
        "task_completion_rate": np.asarray(
            eval_history_task_completion_rate,
            dtype=np.float32,
        ),
        "task_qos_violation_rate": np.asarray(
            eval_history_task_qos_violation_rate,
            dtype=np.float32,
        ),
        "pressure_imbalance": np.asarray(
            eval_history_pressure_imbalance,
            dtype=np.float32,
        ),
    }
    if eval_history_task_delay_source_uav_ms:
        eval_detail_payload["task_delay_source_uav_ms"] = np.stack(
            eval_history_task_delay_source_uav_ms,
            axis=0,
        )
    if eval_history_task_completed_source_count:
        eval_detail_payload["task_completed_source_count"] = np.stack(
            eval_history_task_completed_source_count,
            axis=0,
        )
    if eval_history_task_arrived_source_count:
        eval_detail_payload["task_arrived_source_count"] = np.stack(
            eval_history_task_arrived_source_count,
            axis=0,
        )
    if eval_history_task_delay_source_high_ms:
        eval_detail_payload["task_delay_source_high_ms"] = np.stack(
            eval_history_task_delay_source_high_ms,
            axis=0,
        )
    if eval_history_task_delay_source_low_ms:
        eval_detail_payload["task_delay_source_low_ms"] = np.stack(
            eval_history_task_delay_source_low_ms,
            axis=0,
        )
    if eval_history_task_completed_source_high_count:
        eval_detail_payload["task_completed_source_high_count"] = np.stack(
            eval_history_task_completed_source_high_count,
            axis=0,
        )
    if eval_history_task_completed_source_low_count:
        eval_detail_payload["task_completed_source_low_count"] = np.stack(
            eval_history_task_completed_source_low_count,
            axis=0,
        )
    if eval_history_task_arrived_source_high_count:
        eval_detail_payload["task_arrived_source_high_count"] = np.stack(
            eval_history_task_arrived_source_high_count,
            axis=0,
        )
    if eval_history_task_arrived_source_low_count:
        eval_detail_payload["task_arrived_source_low_count"] = np.stack(
            eval_history_task_arrived_source_low_count,
            axis=0,
        )
    if eval_history_energy_total_episode_per_uav:
        eval_detail_payload["energy_total_episode_per_uav"] = np.stack(
            eval_history_energy_total_episode_per_uav,
            axis=0,
        )
    if eval_history_energy_comp_episode_per_uav:
        eval_detail_payload["energy_comp_episode_per_uav"] = np.stack(
            eval_history_energy_comp_episode_per_uav,
            axis=0,
        )
    if eval_history_energy_uplink_episode_per_uav:
        eval_detail_payload["energy_uplink_episode_per_uav"] = np.stack(
            eval_history_energy_uplink_episode_per_uav,
            axis=0,
        )
    if eval_history_energy_trans_episode_per_uav:
        eval_detail_payload["energy_trans_episode_per_uav"] = np.stack(
            eval_history_energy_trans_episode_per_uav,
            axis=0,
        )
    if eval_history_energy_macro_episode_per_uav:
        eval_detail_payload["energy_macro_episode_per_uav"] = np.stack(
            eval_history_energy_macro_episode_per_uav,
            axis=0,
        )
    if eval_history_compute_service_per_uav_mbits:
        eval_detail_payload["compute_service_per_uav_mbits"] = np.stack(
            eval_history_compute_service_per_uav_mbits,
            axis=0,
        )
    if eval_history_coop_flow_mbits:
        eval_detail_payload["coop_flow_mbits"] = np.stack(
            eval_history_coop_flow_mbits,
            axis=0,
        )
    if eval_history_tx_service_matrix_mbits:
        eval_detail_payload["tx_service_matrix_mbits"] = np.stack(
            eval_history_tx_service_matrix_mbits,
            axis=0,
        )
    if eval_history_split_ratios_high:
        eval_detail_payload["split_ratios_high_mean"] = np.stack(
            eval_history_split_ratios_high,
            axis=0,
        )
    if eval_history_split_ratios_low:
        eval_detail_payload["split_ratios_low_mean"] = np.stack(
            eval_history_split_ratios_low,
            axis=0,
        )
    if eval_history_pressure_per_uav:
        eval_detail_payload["pressure_per_uav"] = np.stack(
            eval_history_pressure_per_uav,
            axis=0,
        )
    np.savez_compressed(eval_detail_path, **eval_detail_payload)

    final_ckpt_path = os.path.join(ckpt_dir, "final.pt")
    torch.save(
        {
            "episode": args.episodes,
            "best_eval_score": best_eval_score,
            "best_eval_metric": best_eval_metric,
            "best_eval_metric_name": best_saved_metric_name,
            "args": vars(args),
            "ppo_actors": [actor.old_pi.state_dict() for actor in ppo_actors],
            "ppo_critic": ppo_critic.critic_v.state_dict(),
            "sac_actor": sac_actor.action_net.state_dict(),
            "sac_critic": sac_critic.critic_v.state_dict(),
            "sac_target_critic": sac_critic.target_critic_v.state_dict(),
            "sac_entropy": {
                "log_alpha": sac_entropy.log_alpha.detach().cpu(),
                "alpha": sac_entropy.alpha.detach().cpu(),
            },
            "sac_memory_counter": sac_memory.memory_counter,
        },
        final_ckpt_path,
    )

    traj_path = os.path.join(trajectory_dir, "trajectory_final.npz")
    save_trajectory_with_verification(env, traj_path)

    print(f"Saved evaluation CSV -> {eval_csv_path}")
    print(f"Saved evaluation detail -> {eval_detail_path}")
    print(f"Saved checkpoint -> {final_ckpt_path}")
    print(f"Saved trajectory data -> {traj_path}")
    print("All tracking data saved successfully.")


if __name__ == "__main__":
    main()
