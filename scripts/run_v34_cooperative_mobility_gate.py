"""Zero-training gate for whether cooperative UAV motion adds value in v34."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import t as student_t
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_common import make_env_from_args, torch_load_checkpoint
from hrl_uav.agents.sac_agent import Actor as SACActor
import hrl_uav.agents.sac_agent as sac_agent_mod
from hrl_uav.load_switching import build_load_switch_schedule
from run_load_switch_gate import run_gate_episode, weighted_node_queue_scores
from search_position_oracle import target_centroid_waypoint_action
from train_centralized_macro import configure_macro_objective, sac_micro_policy


DEFAULT_CONFIG = ROOT / "configs" / "tcom_v34_cooperative_mobility.json"
DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v33_soft_energy_random_mobility"
    / "formal"
    / "seed1"
    / "micro_pretrain"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_OUTPUT = ROOT / "result_archive" / "v34_cooperative_mobility_gate"
POLICIES = ("fixed", "centroid", "cooperative")
POLICY_LABELS = {
    "fixed": "固定初始站位",
    "centroid": "动态质心跟踪",
    "cooperative": "协作感知移动",
}
POLICY_COLORS = {
    "fixed": "#7F7F7F",
    "centroid": "#F28E2B",
    "cooperative": "#2E7D6E",
}
ZONE_COLORS = ("#2878B5", "#F28E2B", "#2CA02C")


def parse_seed_list(value: str) -> list[int]:
    seeds: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(item) for item in token.split("-", 1))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one evaluation seed is required.")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed, centroid-tracking, and cooperative-residual UAV "
            "motion with a frozen micro SAC policy."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="1000-1019")
    parser.add_argument("--offset_m", type=float, default=60.0)
    parser.add_argument("--movement_trigger_m", type=float, default=10.0)
    parser.add_argument("--load_scale", type=float, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--practical_gain_ms", type=float, default=5.0)
    parser.add_argument(
        "--queue_noninferiority_fraction", type=float, default=0.05
    )
    parser.add_argument("--plot_only", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def apply_v34_config(checkpoint_args: argparse.Namespace, config: dict) -> None:
    direct_keys = (
        "load_scale",
        "cpu_ghz",
        "cpu_alloc_mode",
        "objective_mode",
        "delay_objective_weight",
        "energy_objective_weight",
        "power_constraint_scope",
        "power_budget_w",
        "system_dual_state",
        "macro_power_violation_penalty_weight",
        "energy_dual_lr",
        "macro_steps",
        "users_static",
        "gm_mean_speed_mps",
        "gm_mean_heading_rad",
        "gm_heading_mode",
        "gm_speed_memory",
        "gm_heading_memory",
        "gm_speed_noise_std",
        "gm_heading_noise_std",
        "rpgm_group_speed_mps",
        "rpgm_group_speed_memory",
        "rpgm_group_speed_noise_std",
        "rpgm_group_turn_std_rad_sqrt_s",
        "rpgm_local_velocity_memory",
        "rpgm_local_velocity_std_mps",
        "rpgm_local_restore_rate_s2",
        "rpgm_local_speed_cap_mps",
        "macro_action_mode",
        "macro_waypoint_radius_m",
        "macro_waypoint_tolerance_m",
        "micro_action_encoding",
        "micro_reward_scale",
        "u2u_state_representation",
        "u2u_goal_representation",
        "u2u_contention_aware_pressure",
        "u2u_prev_active_state",
    )
    for key in direct_keys:
        if key in config:
            setattr(checkpoint_args, key, config[key])

    checkpoint_args.num_agents = int(config["num_uavs"])
    checkpoint_args.num_users = int(config["num_users"])
    checkpoint_args.zone_loads = ",".join(
        str(value) for value in config["zone_loads_mbps"]
    )
    checkpoint_args.load_switch_mode = str(config["load_switch_mode"])
    checkpoint_args.load_switch_phases = ";".join(
        ",".join(str(value) for value in phase)
        for phase in config["load_switch_phases_mbps"]
    )
    checkpoint_args.load_switch_dwell_min = int(
        config["load_switch_dwell_min"]
    )
    checkpoint_args.load_switch_dwell_max = int(
        config["load_switch_dwell_max"]
    )
    checkpoint_args.arrival_dt = float(config["resource_frame_s"])
    checkpoint_args.resource_frame_dt = float(config["resource_frame_s"])
    checkpoint_args.deployment_dt = float(config["deployment_s"])
    checkpoint_args.user_zone_radius = float(config["user_zone_radius_m"])
    checkpoint_args.macro_flow_state = 1
    checkpoint_args.macro_load_state = 1
    checkpoint_args.macro_capacity_state = 1
    checkpoint_args.pause_service_during_flight = 1
    checkpoint_args.arrivals_during_deployment = 1
    checkpoint_args.energy_lagrange_multiplier_current = 0.0
    checkpoint_args.energy_lagrange_multipliers_current = None
    checkpoint_args.energy_lagrange_multipliers = ""
    checkpoint_args.cpu_power_cap_w = 0.0
    checkpoint_args.micro_diag_bias = 0.0


def load_frozen_micro_policy(
    checkpoint_path: Path,
    config: dict,
    requested_device: str,
):
    checkpoint = torch_load_checkpoint(str(checkpoint_path))
    if str(checkpoint.get("micro_solver", "")) != "sac":
        raise ValueError("The v34 gate requires a SAC micro checkpoint.")
    if "args" not in checkpoint or "sac_actor" not in checkpoint:
        raise KeyError("Checkpoint is missing args or sac_actor.")
    checkpoint_args = argparse.Namespace(**checkpoint["args"])
    apply_v34_config(checkpoint_args, config)
    use_cuda = requested_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sac_agent_mod.device = device

    probe_env = make_env_from_args(checkpoint_args, record_trajectory=False)
    micro_actor = SACActor(
        int(probe_env.sac_obs_dim),
        int(probe_env.micro_action_space.shape[0]),
    )
    micro_actor.action_net.load_state_dict(checkpoint["sac_actor"])
    micro_actor.action_net.eval()
    return checkpoint_args, sac_micro_policy(micro_actor), device


def make_gate_env(checkpoint_args: argparse.Namespace):
    env = make_env_from_args(checkpoint_args, record_trajectory=True)
    configure_macro_objective(
        env,
        checkpoint_args,
        getattr(checkpoint_args, "macro_shaping_final", 0.0),
    )
    env.set_energy_lagrange_multipliers(0.0)
    return env


class V34MacroPolicy:
    """Causal fixed/centroid/cooperative waypoint policies for the gate."""

    def __init__(self, name: str, offset_m: float, movement_trigger_m: float):
        if name not in POLICIES:
            raise ValueError(f"Unknown v34 policy: {name}")
        self.name = name
        self.offset_m = float(offset_m)
        self.movement_trigger_m = float(movement_trigger_m)
        self.initial_centroids = None
        self.last_heavy = -1
        self.last_support = -1
        self.last_residual = np.zeros((3, 2), dtype=np.float64)
        self.last_event = "initial"

    def _support_uav(self, env, heavy: int) -> int:
        base_env = env.env
        loads = np.asarray(
            base_env.zone_expected_offered_load_mbps, dtype=np.float64
        )
        queues = weighted_node_queue_scores(base_env, env.macro_queue_norm)
        candidates = [index for index in range(base_env.N) if index != heavy]
        return min(
            candidates,
            key=lambda index: (float(loads[index]), float(queues[index]), index),
        )

    def __call__(self, env, _local_obs_batch, _eval_mode):
        raw_state = env.env.get_raw_state()
        centroids = np.asarray(raw_state["weighted_centroids"], dtype=np.float64)
        if self.initial_centroids is None:
            self.initial_centroids = centroids.copy()

        residual = np.zeros_like(centroids)
        if self.name == "fixed":
            targets = self.initial_centroids.copy()
            self.last_heavy = -1
            self.last_support = -1
            self.last_event = "initial_centroid_hover"
        elif self.name == "centroid":
            targets = centroids.copy()
            self.last_heavy = -1
            self.last_support = -1
            self.last_event = "dynamic_centroid_tracking"
        else:
            loads = np.asarray(
                env.env.zone_expected_offered_load_mbps, dtype=np.float64
            )
            heavy = int(np.argmax(loads))
            support = self._support_uav(env, heavy)
            direction = (
                np.asarray(raw_state["uav_coords"][heavy], dtype=np.float64)
                - centroids[support]
            )
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12:
                residual[support] = direction / norm * self.offset_m
            targets = centroids + residual
            self.last_event = (
                "load_role_switched"
                if heavy != self.last_heavy or support != self.last_support
                else "maintain_cooperative_offset"
            )
            self.last_heavy = heavy
            self.last_support = support

        self.last_residual = residual.copy()
        targets = np.clip(targets, 0.0, 1000.0)
        uav_coords = np.asarray(raw_state["uav_coords"], dtype=np.float64)
        target_distance = np.linalg.norm(targets - uav_coords, axis=1)
        targets = np.where(
            (target_distance > self.movement_trigger_m)[:, None],
            targets,
            uav_coords,
        )
        return target_centroid_waypoint_action(
            targets,
            centroids,
            env.macro_waypoint_radius_m,
        )

    def metadata(self) -> dict:
        return {
            "heavy_uav": int(self.last_heavy),
            "support_uav": int(self.last_support),
            "event": self.last_event,
            "residual_command_m": self.last_residual.copy(),
        }


def queue_metrics(trace: dict, macro_period_s: float) -> dict:
    queue_total = np.asarray(
        trace["queue_high_trajectory_mbits"]
        + trace["queue_low_trajectory_mbits"],
        dtype=np.float64,
    )
    tail_start = max(0, int(math.floor(2.0 * (queue_total.size - 1) / 3.0)))
    tail = queue_total[tail_start:]
    tail_time = np.arange(tail.size, dtype=np.float64) * float(macro_period_s)
    tail_slope = (
        float(np.polyfit(tail_time, tail, 1)[0]) if tail.size >= 2 else 0.0
    )
    return {
        "mean_queue_mbits": float(np.mean(queue_total)),
        "terminal_queue_mbits": float(queue_total[-1]),
        "tail_queue_slope_mbps": tail_slope,
        "peak_queue_mbits": float(np.max(queue_total)),
    }


def mean_ci(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if values.size < 2:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "std": 0.0}
    std = float(np.std(values, ddof=1))
    half = float(
        student_t.ppf(0.975, values.size - 1) * std / math.sqrt(values.size)
    )
    return {
        "mean": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "std": std,
    }


def paired_summary(rows: list[dict], treatment: str, control: str, field: str) -> dict:
    by_policy = {
        policy: {int(row["seed"]): float(row[field]) for row in rows if row["policy"] == policy}
        for policy in (treatment, control)
    }
    common = sorted(set(by_policy[treatment]) & set(by_policy[control]))
    delta = np.asarray(
        [by_policy[treatment][seed] - by_policy[control][seed] for seed in common],
        dtype=np.float64,
    )
    summary = mean_ci(delta)
    summary.update(
        {
            "treatment": treatment,
            "control": control,
            "field": field,
            "n": len(common),
            "mean_gain": -summary["mean"],
            "seeds": common,
            "deltas": delta.tolist(),
        }
    )
    return summary


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )


def policy_values(rows: list[dict], policy: str, field: str) -> np.ndarray:
    return np.asarray(
        [float(row[field]) for row in rows if row["policy"] == policy],
        dtype=np.float64,
    )


def plot_performance(rows: list[dict], output: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    fields = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms"),
        ("mean_queue_mbits", "时间平均队列积压 / Mbit"),
    )
    rng = np.random.default_rng(20260812)
    for axis, (field, ylabel) in zip(axes, fields):
        for index, policy in enumerate(POLICIES):
            values = policy_values(rows, policy, field)
            stats = mean_ci(values)
            axis.bar(
                index,
                stats["mean"],
                width=0.62,
                color=POLICY_COLORS[policy],
                alpha=0.88,
                yerr=[[stats["mean"] - stats["ci_low"]], [stats["ci_high"] - stats["mean"]]],
                capsize=4,
                edgecolor="white",
                linewidth=0.8,
            )
            jitter = rng.uniform(-0.10, 0.10, size=values.size)
            axis.scatter(
                index + jitter,
                values,
                s=18,
                facecolors="white",
                edgecolors="#303030",
                linewidths=0.7,
                zorder=4,
            )
        axis.set_xticks(range(len(POLICIES)), [POLICY_LABELS[p] for p in POLICIES])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("端到端时延（均值与 95% CI）")
    axes[1].set_title("队列积压（均值与 95% CI）")
    fig.tight_layout()
    fig.savefig(output / "01_三策略时延与队列.png", dpi=300, facecolor="white")
    fig.savefig(output / "01_三策略时延与队列.pdf", facecolor="white")
    plt.close(fig)


def plot_queue_traces(traces_by_policy: dict[str, list[dict]], output: Path, macro_period_s: float) -> None:
    configure_plotting()
    fig, axis = plt.subplots(figsize=(7.5, 4.4))
    for policy in POLICIES:
        stacked = np.stack(
            [
                trace["queue_high_trajectory_mbits"]
                + trace["queue_low_trajectory_mbits"]
                for trace in traces_by_policy[policy]
            ],
            axis=0,
        )
        times = np.arange(stacked.shape[1]) * float(macro_period_s)
        mean = np.mean(stacked, axis=0)
        if stacked.shape[0] >= 2:
            sem = np.std(stacked, axis=0, ddof=1) / math.sqrt(stacked.shape[0])
            half = student_t.ppf(0.975, stacked.shape[0] - 1) * sem
        else:
            half = np.zeros_like(mean)
        axis.plot(
            times,
            mean,
            color=POLICY_COLORS[policy],
            linewidth=2.0,
            label=POLICY_LABELS[policy],
        )
        axis.fill_between(times, mean - half, mean + half, color=POLICY_COLORS[policy], alpha=0.14)
    axis.set_xlabel("仿真时间 / s")
    axis.set_ylabel("系统队列积压 / Mbit")
    axis.set_title("队列积压随时间变化")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "02_队列积压时序.png", dpi=300, facecolor="white")
    fig.savefig(output / "02_队列积压时序.pdf", facecolor="white")
    plt.close(fig)


def plot_paired_gains(rows: list[dict], output: Path) -> None:
    configure_plotting()
    comparisons = (
        ("centroid", "fixed", "质心跟踪 相对 固定站位", "#F28E2B"),
        ("cooperative", "centroid", "协作移动 相对 质心跟踪", "#2E7D6E"),
        ("cooperative", "fixed", "协作移动 相对 固定站位", "#2878B5"),
    )
    fig, axis = plt.subplots(figsize=(8.0, 4.6))
    rng = np.random.default_rng(20260813)
    for y_index, (treatment, control, label, color) in enumerate(comparisons):
        by_policy = {
            policy: {
                int(row["seed"]): float(row["task_delay_weighted_ms"])
                for row in rows
                if row["policy"] == policy
            }
            for policy in (treatment, control)
        }
        common = sorted(set(by_policy[treatment]) & set(by_policy[control]))
        gains = np.asarray(
            [
                by_policy[control][seed] - by_policy[treatment][seed]
                for seed in common
            ],
            dtype=np.float64,
        )
        stats = mean_ci(gains)
        jitter = rng.uniform(-0.10, 0.10, size=gains.size)
        axis.scatter(
            gains,
            y_index + jitter,
            s=24,
            facecolors="white",
            edgecolors=color,
            linewidths=0.9,
            alpha=0.9,
            zorder=3,
        )
        axis.errorbar(
            stats["mean"],
            y_index,
            xerr=[
                [stats["mean"] - stats["ci_low"]],
                [stats["ci_high"] - stats["mean"]],
            ],
            fmt="D",
            markersize=6,
            color=color,
            capsize=4,
            linewidth=2.0,
            zorder=4,
        )
        axis.annotate(
            f"平均 {stats['mean']:.2f} ms",
            (stats["mean"], y_index),
            xytext=(7, 9),
            textcoords="offset points",
            color=color,
            fontsize=9.5,
        )
    axis.axvline(0.0, color="#333333", linewidth=1.0)
    axis.set_yticks(range(len(comparisons)), [item[2] for item in comparisons])
    axis.set_xlabel("端到端时延配对收益 / ms（正值表示更优）")
    axis.set_title("20 个共同随机种子下的移动收益")
    axis.grid(axis="x", linestyle="--", alpha=0.35)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(output / "03_逐种子配对收益.png", dpi=300, facecolor="white")
    fig.savefig(output / "03_逐种子配对收益.pdf", facecolor="white")
    plt.close(fig)


def plot_representative_trajectories(
    representative: dict[str, dict], output: Path, seed: int
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharex=True, sharey=True)
    all_xy = []
    for policy in POLICIES:
        users = representative[policy]["users_trajectory_m"]
        uavs = representative[policy]["uav_trajectory_m"]
        all_xy.extend((users.reshape(-1, 2), uavs.reshape(-1, 2)))
    all_xy_array = np.concatenate(all_xy, axis=0)
    x_min, y_min = np.min(all_xy_array, axis=0) - 25.0
    x_max, y_max = np.max(all_xy_array, axis=0) + 25.0

    for axis, policy in zip(axes, POLICIES):
        trace = representative[policy]
        users = trace["users_trajectory_m"]
        allocations = trace["user_allocations"]
        uavs = trace["uav_trajectory_m"]
        centroids = trace["centroid_trajectory_m"]
        stride = max(1, users.shape[0] // 300)
        for zone, color in enumerate(ZONE_COLORS):
            for user_index in np.flatnonzero(allocations == zone):
                axis.plot(
                    users[::stride, user_index, 0],
                    users[::stride, user_index, 1],
                    color=color,
                    alpha=0.18,
                    linewidth=0.55,
                )
            axis.plot(
                centroids[:, zone, 0],
                centroids[:, zone, 1],
                color=color,
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
            )
            axis.plot(
                uavs[:, zone, 0],
                uavs[:, zone, 1],
                color=color,
                linewidth=3.0,
                marker="o",
                markersize=2.8,
                markevery=max(1, uavs.shape[0] // 8),
                label=f"UAV-{zone + 1}",
            )
            axis.scatter(
                uavs[0, zone, 0],
                uavs[0, zone, 1],
                marker="s",
                s=40,
                facecolors="white",
                edgecolors=color,
                linewidths=1.2,
                zorder=5,
            )
            axis.scatter(
                uavs[-1, zone, 0],
                uavs[-1, zone, 1],
                marker="*",
                s=105,
                color=color,
                edgecolors="#222222",
                linewidths=0.7,
                zorder=6,
            )
        axis.set_title(POLICY_LABELS[policy])
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, linestyle="--", alpha=0.28)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("X 坐标 / m")
    axes[0].set_ylabel("Y 坐标 / m")
    handles = [
        Line2D([0], [0], color="#777777", linewidth=0.8, alpha=0.55, label="地面终端轨迹"),
        Line2D([0], [0], color="#333333", linewidth=1.5, linestyle="--", label="区域业务质心"),
    ] + [
        Line2D([0], [0], color=color, linewidth=3.0, label=f"UAV-{index + 1}")
        for index, color in enumerate(ZONE_COLORS)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=5,
        frameon=False,
        handlelength=2.4,
    )
    fig.suptitle(f"相同随机场景下的 UAV 与地面终端轨迹（seed={seed}）", fontweight="bold")
    fig.text(
        0.5,
        0.025,
        "方块和星形分别表示 UAV 起点与终点；三幅图使用完全相同的用户移动、任务到达和信道随机轨迹",
        ha="center",
        color="#444444",
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.22, wspace=0.14)
    fig.savefig(output / "04_三策略轨迹对比.png", dpi=300, facecolor="white")
    fig.savefig(output / "04_三策略轨迹对比.pdf", facecolor="white")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_trace(path: Path, trace: dict, base_env, include_users: bool) -> dict:
    payload = {
        "uav_trajectory_m": trace["uav_trajectory_m"],
        "centroid_trajectory_m": trace["centroid_trajectory_m"],
        "trajectory_phases": trace["trajectory_phases"],
        "queue_high_trajectory_mbits": trace["queue_high_trajectory_mbits"],
        "queue_low_trajectory_mbits": trace["queue_low_trajectory_mbits"],
        "queue_per_uav_trajectory_mbits": trace["queue_per_uav_trajectory_mbits"],
        "phase_flow_mbits": trace["phase_flow_mbits"],
        "total_flow_mbits": trace["total_flow_mbits"],
    }
    if include_users:
        payload.update(
            {
                "users_trajectory_m": np.asarray(base_env.users_trajectory),
                "group_center_trajectory_m": np.asarray(
                    base_env.group_center_trajectory
                ),
                "user_allocations": np.asarray(base_env.user_allocations),
            }
        )
    np.savez_compressed(path, **payload)
    return payload


def make_report(
    rows: list[dict],
    seeds: list[int],
    practical_gain_ms: float,
    queue_noninferiority_fraction: float,
) -> dict:
    summaries = {
        "cooperative_vs_centroid_delay": paired_summary(
            rows, "cooperative", "centroid", "task_delay_weighted_ms"
        ),
        "cooperative_vs_fixed_delay": paired_summary(
            rows, "cooperative", "fixed", "task_delay_weighted_ms"
        ),
        "centroid_vs_fixed_delay": paired_summary(
            rows, "centroid", "fixed", "task_delay_weighted_ms"
        ),
        "cooperative_vs_centroid_queue": paired_summary(
            rows, "cooperative", "centroid", "mean_queue_mbits"
        ),
    }
    centroid_queue = float(np.mean(policy_values(rows, "centroid", "mean_queue_mbits")))
    queue_margin = float(queue_noninferiority_fraction * centroid_queue)
    delay_centroid = summaries["cooperative_vs_centroid_delay"]
    delay_fixed = summaries["cooperative_vs_fixed_delay"]
    queue = summaries["cooperative_vs_centroid_queue"]
    checks = {
        "cooperative_beats_centroid_statistically": delay_centroid["ci_high"] < 0.0,
        "cooperative_beats_centroid_practically": delay_centroid["mean_gain"] > practical_gain_ms,
        "cooperative_beats_fixed_statistically": delay_fixed["ci_high"] < 0.0,
        "cooperative_beats_fixed_practically": delay_fixed["mean_gain"] > practical_gain_ms,
        "queue_noninferior_to_centroid": queue["ci_high"] < queue_margin,
    }
    means = {
        policy: {
            field: mean_ci(policy_values(rows, policy, field))
            for field in (
                "task_delay_weighted_ms",
                "mean_queue_mbits",
                "terminal_queue_mbits",
                "tail_queue_slope_mbps",
                "average_power_w",
                "coop_ratio",
            )
        }
        for policy in POLICIES
    }
    return {
        "seeds": seeds,
        "policy_means": means,
        "paired": summaries,
        "pre_registered_thresholds": {
            "practical_delay_gain_ms": practical_gain_ms,
            "queue_noninferiority_fraction": queue_noninferiority_fraction,
            "queue_noninferiority_margin_mbits": queue_margin,
            "power_role": "descriptive_only_no_hard_threshold",
        },
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def write_markdown_report(report: dict, output: Path) -> None:
    means = report["policy_means"]
    cvc = report["paired"]["cooperative_vs_centroid_delay"]
    cvf = report["paired"]["cooperative_vs_fixed_delay"]
    lines = [
        "# v34 协作移动零训练门禁",
        "",
        f"**判决：{'PASS' if report['pass'] else 'FAIL'}**",
        "",
        "## 三策略结果",
        "",
        "| 策略 | 加权端到端时延/ms | 平均队列/Mbit | 终端队列/Mbit | 平均功率/W |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        lines.append(
            "| {label} | {delay:.3f} | {queue:.3f} | {terminal:.3f} | {power:.3f} |".format(
                label=POLICY_LABELS[policy],
                delay=means[policy]["task_delay_weighted_ms"]["mean"],
                queue=means[policy]["mean_queue_mbits"]["mean"],
                terminal=means[policy]["terminal_queue_mbits"]["mean"],
                power=means[policy]["average_power_w"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## 配对判据",
            "",
            (
                "- 协作感知相对质心跟踪：平均收益 "
                f"{cvc['mean_gain']:.3f} ms，差值 95% CI "
                f"[{cvc['ci_low']:.3f}, {cvc['ci_high']:.3f}] ms。"
            ),
            (
                "- 协作感知相对固定站位：平均收益 "
                f"{cvf['mean_gain']:.3f} ms，差值 95% CI "
                f"[{cvf['ci_low']:.3f}, {cvf['ci_high']:.3f}] ms。"
            ),
            "- 功率仅作描述性指标，不设硬阈值。",
            "",
            "## 判据逐项",
            "",
        ]
    )
    for name, passed in report["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    (output / "gate_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def replot_saved_results(output: Path, macro_period_s: float) -> None:
    with (output / "per_seed_raw.csv").open(
        "r", newline="", encoding="utf-8-sig"
    ) as stream:
        rows = list(csv.DictReader(stream))
    seeds = sorted({int(row["seed"]) for row in rows})
    traces_by_policy = {policy: [] for policy in POLICIES}
    representative = {}
    for seed in seeds:
        for policy in POLICIES:
            path = output / "traces" / f"seed{seed}_{policy}.npz"
            with np.load(path) as archive:
                trace = {key: archive[key].copy() for key in archive.files}
            traces_by_policy[policy].append(trace)
            if seed == seeds[0]:
                representative[policy] = trace
    plot_performance(rows, output)
    plot_queue_traces(traces_by_policy, output, macro_period_s)
    plot_paired_gains(rows, output)
    plot_representative_trajectories(representative, output, seeds[0])


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "traces").mkdir(exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.load_scale is not None:
        config["load_scale"] = float(args.load_scale)
    if args.plot_only:
        replot_saved_results(output, float(config["macro_period_s"]))
        print(f"plots regenerated; output={output}")
        return
    seeds = parse_seed_list(args.seeds)
    checkpoint_args, micro_policy, device = load_frozen_micro_policy(
        checkpoint_path, config, args.device
    )
    phases = np.asarray(config["load_switch_phases_mbps"], dtype=np.float64)
    macro_steps = int(config["macro_steps"])
    macro_period_s = float(config["macro_period_s"])
    rows: list[dict] = []
    traces_by_policy: dict[str, list[dict]] = {policy: [] for policy in POLICIES}
    representative: dict[str, dict] = {}

    print(
        f"checkpoint={checkpoint_path}; device={device}; seeds={seeds}; "
        f"offset={args.offset_m:.1f}m"
    )
    for seed in seeds:
        schedule = build_load_switch_schedule(
            mode=config["load_switch_mode"],
            phases_mbps=phases,
            macro_steps=macro_steps,
            episode_seed=seed,
            dwell_min=int(config["load_switch_dwell_min"]),
            dwell_max=int(config["load_switch_dwell_max"]),
        )
        seed_hashes = {}
        for policy_name in POLICIES:
            env = make_gate_env(checkpoint_args)
            macro_policy = V34MacroPolicy(
                policy_name,
                args.offset_m,
                args.movement_trigger_m,
            )
            result, trace = run_gate_episode(
                env=env,
                checkpoint_args=checkpoint_args,
                micro_policy=micro_policy,
                policy_name=policy_name,
                seed=seed,
                macro_steps=macro_steps,
                load_switch_schedule=schedule,
                macro_policy=macro_policy,
                macro_metadata_provider=macro_policy.metadata,
            )
            result.update(queue_metrics(trace, macro_period_s))
            rows.append(result)
            traces_by_policy[policy_name].append(trace)
            seed_hashes[policy_name] = result["external_trace_hash"]
            include_users = seed == seeds[0]
            payload = save_trace(
                output / "traces" / f"seed{seed}_{policy_name}.npz",
                trace,
                env.env,
                include_users,
            )
            if include_users:
                representative[policy_name] = payload
            write_csv(output / "per_seed_raw.csv", rows)
            print(
                f"seed={seed} policy={policy_name}: "
                f"delay={result['task_delay_weighted_ms']:.2f}ms, "
                f"queue={result['mean_queue_mbits']:.2f}Mbit, "
                f"power={result['average_power_w']:.2f}W"
            )
        if len(set(seed_hashes.values())) != 1:
            raise RuntimeError(
                f"CRN external trace mismatch for seed {seed}: {seed_hashes}"
            )

    report = make_report(
        rows,
        seeds,
        practical_gain_ms=float(args.practical_gain_ms),
        queue_noninferiority_fraction=float(args.queue_noninferiority_fraction),
    )
    report.update(
        {
            "config": str(config_path),
            "micro_checkpoint": str(checkpoint_path),
            "device": str(device),
            "cooperative_offset_m": float(args.offset_m),
            "movement_trigger_m": float(args.movement_trigger_m),
            "load_scale": float(config["load_scale"]),
            "fixed_policy_definition": (
                "move once toward the initial business centroid using the common "
                "event trigger, then hover"
            ),
            "centroid_policy_definition": (
                "causal event-triggered dynamic weighted-business-centroid tracking"
            ),
            "cooperative_policy_definition": (
                "centroid tracking plus a "
                f"{float(args.offset_m):g} m causal residual from the lightest "
                "support UAV toward the current heavy UAV"
            ),
        }
    )
    (output / "gate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown_report(report, output)
    plot_performance(rows, output)
    plot_queue_traces(traces_by_policy, output, macro_period_s)
    plot_paired_gains(rows, output)
    plot_representative_trajectories(representative, output, seeds[0])
    print(f"verdict={'PASS' if report['pass'] else 'FAIL'}; output={output}")


if __name__ == "__main__":
    main()
