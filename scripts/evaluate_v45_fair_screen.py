"""Evaluate the fair seed-1 method screen with common random numbers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_v42_macro_only_pilot import load_policies as load_td3_policies  # noqa: E402
from compare_common import evaluate_policy, make_env_from_args  # noqa: E402
from compare_sca import solve_sca_micro_action  # noqa: E402
from hrl_uav.agents.ppo_agent import Actor as PPOActor  # noqa: E402
from hrl_uav.agents.sac_agent import Actor as SACActor  # noqa: E402
import hrl_uav.agents.ppo_agent as ppo_agent_mod  # noqa: E402
import hrl_uav.agents.sac_agent as sac_agent_mod  # noqa: E402
from run_v34_cooperative_mobility_gate import V34MacroPolicy  # noqa: E402
from search_position_oracle import make_centroid_offset_policy  # noqa: E402
from train_centralized_macro import (  # noqa: E402
    centralized_macro_policy,
    configure_macro_objective,
    configure_ppo_hover_policy,
    configure_ppo_log_std_schedule,
    sac_micro_policy,
    sca_micro_policy,
    torch_load_checkpoint,
)
from train_centralized_td3 import MacroActor, td3_macro_policy  # noqa: E402


DEFAULT_OUTPUT = ROOT / "result_archive" / "v45_fair_method_screen"
DEFAULT_CHECKPOINTS = {
    "td3_sac": (
        ROOT
        / "results"
        / "tcom_v45_fair_screen_stage2"
        / "formal"
        / "seed1"
        / "joint_td3_sac"
        / "checkpoints"
        / "final.pt"
    ),
    "ppo_sac": (
        ROOT
        / "results"
        / "tcom_v45_fair_screen_stage2"
        / "formal"
        / "seed1"
        / "joint_ppo_sac"
        / "checkpoints"
        / "final.pt"
    ),
    "ppo_sca": (
        ROOT
        / "results"
        / "tcom_v45_fair_screen_stage2"
        / "formal"
        / "seed1"
        / "cppo_sca"
        / "checkpoints"
        / "final.pt"
    ),
    "no_coop": (
        ROOT
        / "results"
        / "tcom_v45_fair_screen_stage2"
        / "formal"
        / "seed1"
        / "no_coop"
        / "checkpoints"
        / "final.pt"
    ),
}
METHOD_LABELS = {
    "td3_sac": "TD3-SAC",
    "ppo_sac": "PPO-SAC",
    "ppo_sca": "PPO-SCA",
    "no_coop": "No-Coop",
    "fixed": "固定初始站位",
    "centroid": "动态质心",
}
METHOD_ORDER = tuple(METHOD_LABELS)
COLORS = {
    "td3_sac": "#2878B5",
    "ppo_sac": "#E07B39",
    "ppo_sca": "#7A65A8",
    "no_coop": "#D95F59",
    "fixed": "#777777",
    "centroid": "#2E8B57",
}
SCENARIO_KEYS = (
    "num_agents",
    "num_users",
    "macro_steps",
    "micro_steps",
    "zone_loads",
    "load_switch_mode",
    "load_switch_phases",
    "load_switch_dwell_min",
    "load_switch_dwell_max",
    "load_scale",
    "cpu_ghz",
    "cpu_alloc_mode",
    "objective_mode",
    "delay_objective_weight",
    "energy_objective_weight",
    "macro_action_mode",
    "macro_waypoint_radius_m",
    "gm_heading_mode",
    "rpgm_group_speed_mps",
    "rpgm_local_velocity_std_mps",
)


def parse_seed_spec(value: str) -> list[int]:
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
        raise ValueError("At least one evaluation seed is required")
    return seeds


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_arg(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [normalized_arg(item) for item in value]
    if isinstance(value, float):
        return round(value, 12)
    return value


def scenario_signature(args) -> dict:
    return {
        key: normalized_arg(getattr(args, key, None))
        for key in SCENARIO_KEYS
    }


def make_eval_env(args, *, record_trajectory: bool):
    env = make_env_from_args(args, record_trajectory=record_trajectory)
    configure_macro_objective(
        env,
        args,
        float(getattr(args, "macro_shaping_final", 0.0)),
    )
    return env


def load_ppo_policy(checkpoint_path: Path, requested_device: str):
    checkpoint = torch_load_checkpoint(str(checkpoint_path))
    args = argparse.Namespace(**checkpoint["args"])
    use_cuda = requested_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    ppo_agent_mod.device = device
    sac_agent_mod.device = device

    probe_env = make_eval_env(args, record_trajectory=False)
    _, global_state = probe_env.reset(seed=int(args.seed))[0]
    macro_actor = PPOActor(
        int(np.asarray(global_state).size),
        int(args.num_agents) * 2,
    )
    configure_ppo_hover_policy(macro_actor, args, initialize_hover=False)
    configure_ppo_log_std_schedule(macro_actor, args)
    macro_actor.old_pi.load_state_dict(checkpoint["macro_actor"])
    macro_actor.new_pi.load_state_dict(checkpoint["macro_actor"])
    macro_actor.old_pi.eval()
    macro_actor.new_pi.eval()
    macro_policy = centralized_macro_policy(macro_actor, deterministic=True)

    micro_solver = str(checkpoint.get("micro_solver", args.micro_solver)).lower()
    if micro_solver == "sac":
        micro_action_dim = int(probe_env.micro_action_space.shape[0]) + int(
            bool(getattr(args, "micro_autonomous_residual_gate", 0))
        )
        micro_actor = SACActor(
            int(probe_env.sac_obs_dim),
            micro_action_dim,
        )
        micro_actor.action_net.load_state_dict(checkpoint["sac_actor"])
        micro_actor.action_net.eval()
        micro_policy = sac_micro_policy(micro_actor, args)
    elif micro_solver == "sca":
        micro_policy = sca_micro_policy(solve_sca_micro_action, args)
    else:
        micro_policy = lambda env, _eval: np.zeros(
            env.micro_action_space.shape[0], dtype=np.float32
        )
    return args, macro_policy, micro_policy, micro_solver


def load_method_policy(name: str, path: Path, requested_device: str):
    checkpoint = torch_load_checkpoint(str(path))
    macro_solver = str(checkpoint.get("macro_solver", "td3")).lower()
    if macro_solver == "ppo":
        args, macro_policy, micro_policy, micro_solver = load_ppo_policy(
            path, requested_device
        )
    elif str(checkpoint.get("micro_solver", "sac")).lower() == "local":
        args = argparse.Namespace(**checkpoint["args"])
        use_cuda = requested_device == "cuda" and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        probe_env = make_eval_env(args, record_trajectory=False)
        _, global_state = probe_env.reset(seed=int(args.seed))[0]
        macro_actor = MacroActor(
            int(np.asarray(global_state).size),
            int(args.num_agents) * 2,
            hidden_dim=int(args.td3_hidden_dim),
        ).to(device)
        macro_actor.load_state_dict(checkpoint["macro_actor"])
        macro_actor.eval()
        macro_policy = td3_macro_policy(macro_actor)
        micro_policy = lambda env, _eval: np.zeros(
            env.micro_action_space.shape[0], dtype=np.float32
        )
        micro_solver = "local"
    else:
        args, macro_policy, micro_policy = load_td3_policies(
            path, requested_device
        )
        micro_solver = str(checkpoint.get("micro_solver", args.micro_solver)).lower()
    return {
        "name": name,
        "path": path,
        "args": args,
        "macro_policy": macro_policy,
        "micro_policy": micro_policy,
        "micro_solver": micro_solver,
        "macro_solver": macro_solver,
    }


def evaluate_one(policy: dict, seed: int, *, record_trajectory: bool = False):
    args = policy["args"]
    env = make_eval_env(args, record_trajectory=record_trajectory)
    if policy["micro_solver"] == "local":
        env.force_local_routing = True
    eval_args = argparse.Namespace(**vars(args))
    eval_args.eval_episodes = 1
    eval_args.eval_base_seed = int(seed)
    metrics = evaluate_policy(
        env,
        eval_args,
        policy["macro_policy"],
        policy["micro_policy"],
    )
    row = {
        "seed": int(seed),
        "method": policy["name"],
        "task_delay_weighted_ms": float(metrics["task_delay_weighted_ms"]),
        "task_delay_high_ms": float(metrics["task_delay_high_ms"]),
        "task_delay_low_ms": float(metrics["task_delay_low_ms"]),
        "queue_total_mbits": float(metrics["q_high"] + metrics["q_low"]),
        "queue_high_mbits": float(metrics["q_high"]),
        "queue_low_mbits": float(metrics["q_low"]),
        "soft_objective_j": float(metrics["J"]),
        "average_power_w": float(metrics["average_power_w"]),
        "coop_ratio": float(metrics["coop_ratio"]),
        "pressure_imbalance": float(metrics["pressure_imbalance"]),
        "u2g_distance_m": float(metrics["u2g"]),
        "micro_decision_mean_ms": float(metrics["micro_decision_mean_ms"]),
        "residual_gate_mean": float(metrics.get("residual_gate_mean", 1.0)),
        "residual_gate_std": float(metrics.get("residual_gate_std", 0.0)),
        "residual_gate_q10": float(metrics.get("residual_gate_q10", 1.0)),
        "residual_gate_q90": float(metrics.get("residual_gate_q90", 1.0)),
    }
    if record_trajectory:
        base = getattr(env, "env", env)
        row["trajectory"] = {
            "uavs": np.asarray(base.uav_trajectory, dtype=np.float32),
            "users": np.asarray(base.users_trajectory, dtype=np.float32),
            "centroids": np.asarray(base.centroid_trajectory, dtype=np.float32),
            "user_allocations": np.asarray(base.user_allocations, dtype=np.int64),
        }
    return row


def mean_ci95(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    critical = (
        float(student_t.ppf(0.975, array.size - 1))
        if array.size > 1
        else 0.0
    )
    half = critical * std / np.sqrt(array.size) if array.size > 1 else 0.0
    return {
        "mean": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "std": std,
        "n": int(array.size),
    }


def summarize(rows: list[dict], seeds: list[int]) -> dict:
    numeric = [
        key for key in rows[0] if key not in {"seed", "method"}
    ]
    by_method = {
        method: {
            int(row["seed"]): row
            for row in rows
            if row["method"] == method
        }
        for method in METHOD_ORDER
    }
    summaries = {
        method: {
            field: mean_ci95(
                [by_method[method][seed][field] for seed in seeds]
            )
            for field in numeric
        }
        for method in METHOD_ORDER
    }
    paired_vs_td3 = {}
    for method in METHOD_ORDER:
        if method == "td3_sac":
            continue
        paired_vs_td3[method] = {
            field: mean_ci95(
                [
                    by_method[method][seed][field]
                    - by_method["td3_sac"][seed][field]
                    for seed in seeds
                ]
            )
            for field in numeric
        }
    delay_ranking = sorted(
        METHOD_ORDER,
        key=lambda method: summaries[method]["task_delay_weighted_ms"]["mean"],
    )
    objective_ranking = sorted(
        METHOD_ORDER,
        key=lambda method: summaries[method]["soft_objective_j"]["mean"],
    )
    return {
        "evaluation_seeds": seeds,
        "method_summaries": summaries,
        "paired_baseline_minus_td3": paired_vs_td3,
        "delay_ranking": delay_ranking,
        "objective_ranking": objective_ranking,
        "scope": (
            "One training seed per learned method and 20 paired evaluation seeds; "
            "this is a screening result, not the final multi-training-seed claim."
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
    })


def plot_ranking(report: dict, output: Path) -> None:
    configure_plotting()
    methods = list(METHOD_ORDER)
    x = np.arange(len(methods))
    labels = [METHOD_LABELS[item] for item in methods]
    colors = [COLORS[item] for item in methods]
    metrics = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms", "(a) 端到端时延"),
        ("queue_total_mbits", "平均队列 / Mbit", "(b) 平均队列"),
        ("average_power_w", "平均功率 / W", "(c) 功率账本"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    for axis, (field, ylabel, title) in zip(axes, metrics):
        stats = [report["method_summaries"][m][field] for m in methods]
        means = np.asarray([item["mean"] for item in stats])
        errors = np.asarray([
            item["ci95_high"] - item["mean"] for item in stats
        ])
        axis.bar(x, means, yerr=errors, capsize=4, color=colors, width=0.68)
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        for index, value in enumerate(means):
            axis.text(index, value + max(means) * 0.012, f"{value:.1f}", ha="center")
    fig.suptitle(
        "同一场景下的方法筛查（学习方法均训练500轮，20个共同评估种子）"
    )
    fig.tight_layout()
    fig.savefig(output / "01_方法排名.png", dpi=300, facecolor="white")
    plt.close(fig)


def convergence_sources() -> dict[str, tuple[Path, Path]]:
    return {
        "td3_sac": (
            ROOT / "results/tcom_v43_cooperation_aligned_macro/formal/seed1/joint_td3_sac/eval_central_td3.csv",
            ROOT / "results/tcom_v45_fair_screen_stage2/formal/seed1/joint_td3_sac/eval_central_td3.csv",
        ),
        "ppo_sac": (
            ROOT / "results/tcom_v45_fair_screen_stage1/formal/seed1/joint_ppo_sac/eval_central_macro.csv",
            ROOT / "results/tcom_v45_fair_screen_stage2/formal/seed1/joint_ppo_sac/eval_central_macro.csv",
        ),
        "ppo_sca": (
            ROOT / "results/tcom_v45_fair_screen_stage1/formal/seed1/cppo_sca/eval_central_sca.csv",
            ROOT / "results/tcom_v45_fair_screen_stage2/formal/seed1/cppo_sca/eval_central_sca.csv",
        ),
        "no_coop": (
            ROOT / "results/tcom_v45_fair_screen_stage1/formal/seed1/no_coop/eval_central_td3.csv",
            ROOT / "results/tcom_v45_fair_screen_stage2/formal/seed1/no_coop/eval_central_td3.csv",
        ),
    }


def read_eval_curve(path: Path, offset: int) -> tuple[list[int], list[float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        [offset + int(float(row["episode"])) for row in rows],
        [float(row["task_delay_weighted_ms"]) for row in rows],
    )


def plot_convergence(output: Path) -> None:
    configure_plotting()
    curves = {}
    for method, (stage1, stage2) in convergence_sources().items():
        x1, y1 = read_eval_curve(stage1, 0)
        x2, y2 = read_eval_curve(stage2, 300)
        curves[method] = (x1 + x2, y1 + y2)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    for method, (x_values, y_values) in curves.items():
        axes[0].plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    for method in ("td3_sac", "ppo_sac"):
        x_values, y_values = curves[method]
        axes[1].plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.2,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
    for axis in axes:
        axis.axvline(300, color="#444444", linestyle="--", linewidth=1.2)
        axis.set_xlabel("累计训练轮次")
        axis.grid(linestyle="--", alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("固定评估端到端时延 / ms")
    axes[0].set_title("(a) 四方法全尺度")
    axes[1].set_title("(b) TD3-SAC 与 PPO-SAC 局部放大")
    axes[1].set_ylim(580, 1150)
    axes[1].text(305, 1125, "联合阶段开始", va="top")
    fig.suptitle("四种方法的真实固定评估曲线（训练种子 seed=1）")
    fig.tight_layout()
    fig.savefig(output / "02_四方法收敛曲线.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_paired_td3_ppo(rows: list[dict], output: Path) -> None:
    configure_plotting()
    td3 = {
        int(row["seed"]): float(row["task_delay_weighted_ms"])
        for row in rows if row["method"] == "td3_sac"
    }
    ppo = {
        int(row["seed"]): float(row["task_delay_weighted_ms"])
        for row in rows if row["method"] == "ppo_sac"
    }
    seeds = sorted(set(td3) & set(ppo))
    deltas = np.asarray([td3[seed] - ppo[seed] for seed in seeds])
    stats = mean_ci95(deltas)
    fig, axis = plt.subplots(figsize=(8.4, 4.3))
    axis.axhline(0, color="#444444", linewidth=1.0)
    axis.scatter(
        np.arange(1, len(seeds) + 1),
        deltas,
        color=COLORS["ppo_sac"],
        edgecolor="white",
        s=52,
        zorder=3,
    )
    axis.axhspan(
        stats["ci95_low"], stats["ci95_high"],
        color=COLORS["ppo_sac"], alpha=0.14,
    )
    axis.axhline(
        stats["mean"],
        color=COLORS["ppo_sac"],
        linewidth=2.0,
        label=(
            f"平均优势 {stats['mean']:.1f} ms，"
            f"95% CI [{stats['ci95_low']:.1f}, {stats['ci95_high']:.1f}]"
        ),
    )
    axis.set_xlabel("共同评估种子序号")
    axis.set_ylabel("TD3-SAC时延 - PPO-SAC时延 / ms")
    axis.set_title("TD3-SAC 与 PPO-SAC 的逐种子配对差异")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "04_TD3与PPO配对差异.png", dpi=300, facecolor="white")
    plt.close(fig)


def save_and_plot_trajectories(policies: dict, output: Path, seed: int) -> None:
    configure_plotting()
    trace_dir = output / "trajectories"
    trace_dir.mkdir(exist_ok=True)
    saved_metrics = {}
    raw_path = output / "per_seed_raw.csv"
    if raw_path.is_file():
        with raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["seed"]) == int(seed):
                    saved_metrics[row["method"]] = {
                        key: (float(value) if key not in {"seed", "method"} else value)
                        for key, value in row.items()
                    }
    traces = {}
    for method in METHOD_ORDER:
        trace_path = trace_dir / f"{method}_seed{seed}.npz"
        if trace_path.is_file() and method in saved_metrics:
            result = saved_metrics[method]
            with np.load(trace_path) as archive:
                trace = {key: archive[key] for key in archive.files}
        else:
            result = evaluate_one(policies[method], seed, record_trajectory=True)
            trace = result.pop("trajectory")
            np.savez_compressed(trace_path, **trace)
        traces[method] = (result, trace)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2), sharex=True, sharey=True)
    uav_colors = ("#2878B5", "#E67E22", "#2E9D55")
    for axis, method in zip(axes.ravel(), METHOD_ORDER):
        result, trace = traces[method]
        users = trace["users"]
        uavs = trace["uavs"]
        for user in range(users.shape[1]):
            zone = user % 3
            axis.plot(
                users[:, user, 0], users[:, user, 1],
                color=uav_colors[zone], alpha=0.12, linewidth=0.65,
            )
        for index, color in enumerate(uav_colors):
            axis.plot(
                uavs[:, index, 0], uavs[:, index, 1],
                color=color, linewidth=2.4,
            )
            axis.scatter(
                uavs[0, index, 0], uavs[0, index, 1],
                marker="s", s=34, facecolor="white", edgecolor=color, zorder=4,
            )
            axis.scatter(
                uavs[-1, index, 0], uavs[-1, index, 1],
                marker="*", s=95, facecolor=color, edgecolor="#222222", zorder=5,
            )
        axis.set_title(
            f"{METHOD_LABELS[method]}\n"
            f"时延 {result['task_delay_weighted_ms']:.1f} ms，"
            f"队列 {result['queue_total_mbits']:.1f} Mbit",
            fontsize=10,
        )
        axis.grid(linestyle="--", alpha=0.22)
        axis.set_aspect("equal", adjustable="box")
    for axis in axes[:, 0]:
        axis.set_ylabel("Y坐标 / m")
    for axis in axes[-1, :]:
        axis.set_xlabel("X坐标 / m")
    fig.suptitle(f"六种策略代表轨迹（共同随机种子 {seed}）", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.6, w_pad=1.2)
    fig.savefig(output / "03_六策略轨迹对比.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_trajectory_differences(output: Path, seed: int) -> None:
    """Show macro-policy differences hidden by the full-scene map scale."""
    configure_plotting()
    trace_dir = output / "trajectories"
    traces = {}
    for method in METHOD_ORDER:
        with np.load(trace_dir / f"{method}_seed{seed}.npz") as archive:
            traces[method] = {key: archive[key] for key in archive.files}

    raw_path = output / "per_seed_raw.csv"
    with raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metrics = {
            row["method"]: row
            for row in csv.DictReader(handle)
            if int(row["seed"]) == int(seed)
        }

    fig = plt.figure(figsize=(13.8, 9.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.12, 0.88), hspace=0.28)
    uav_colors = ("#2878B5", "#E67E22", "#2E9D55")
    for axis, method, panel in (
        (fig.add_subplot(grid[0, 0]), "td3_sac", "(a) TD3-SAC"),
        (fig.add_subplot(grid[0, 1]), "ppo_sac", "(b) PPO-SAC"),
    ):
        trace = traces[method]
        users = trace["users"]
        uavs = trace["uavs"]
        centroids = trace["centroids"]
        for user in range(users.shape[1]):
            zone = user % 3
            axis.plot(
                users[:, user, 0], users[:, user, 1],
                color=uav_colors[zone], alpha=0.10, linewidth=0.65,
            )
        for index, color in enumerate(uav_colors):
            axis.plot(
                centroids[:, index, 0], centroids[:, index, 1],
                color=color, linestyle="--", linewidth=1.2, alpha=0.65,
            )
            axis.plot(
                uavs[:, index, 0], uavs[:, index, 1],
                color=color, linewidth=2.5, label=f"UAV-{index + 1}",
            )
            axis.scatter(
                uavs[0, index, 0], uavs[0, index, 1],
                marker="s", s=38, facecolor="white", edgecolor=color, zorder=4,
            )
            axis.scatter(
                uavs[-1, index, 0], uavs[-1, index, 1],
                marker="*", s=105, facecolor=color,
                edgecolor="#222222", zorder=5,
            )
        axis.set_title(
            f"{panel}：时延 {float(metrics[method]['task_delay_weighted_ms']):.1f} ms"
        )
        axis.set_xlabel("X坐标 / m")
        axis.set_ylabel("Y坐标 / m")
        axis.grid(linestyle="--", alpha=0.22)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="best", ncol=3, fontsize=9)

    curve_axis = fig.add_subplot(grid[1, 0])
    curve_methods = ("td3_sac", "ppo_sac", "ppo_sca", "no_coop", "fixed", "centroid")
    for method in curve_methods:
        trace = traces[method]
        distance = np.linalg.norm(trace["uavs"] - trace["centroids"], axis=2).mean(axis=1)
        window = min(100, distance.size)
        kernel = np.ones(window, dtype=np.float64) / window
        smooth = np.convolve(distance, kernel, mode="valid")
        time_s = np.linspace(0.0, 300.0, distance.size)[window - 1:]
        curve_axis.plot(
            time_s, smooth, color=COLORS[method], linewidth=2.0,
            label=METHOD_LABELS[method],
        )
    curve_axis.set_title("(c) 三机相对动态质心的平均偏移（10 s平滑）")
    curve_axis.set_xlabel("时间 / s")
    curve_axis.set_ylabel("平均偏移 / m")
    curve_axis.grid(linestyle="--", alpha=0.25)
    curve_axis.legend(ncol=2, fontsize=8.5)

    bar_axis = fig.add_subplot(grid[1, 1])
    mean_offsets = []
    for method in curve_methods:
        trace = traces[method]
        mean_offsets.append(float(np.linalg.norm(
            trace["uavs"] - trace["centroids"], axis=2
        ).mean()))
    x = np.arange(len(curve_methods))
    bar_axis.bar(
        x, mean_offsets,
        color=[COLORS[method] for method in curve_methods], width=0.68,
    )
    bar_axis.set_xticks(
        x, [METHOD_LABELS[method] for method in curve_methods],
        rotation=22, ha="right",
    )
    bar_axis.set_ylabel("平均偏移 / m")
    bar_axis.set_title("(d) 轨迹差异的放大量化")
    bar_axis.grid(axis="y", linestyle="--", alpha=0.25)
    for index, value in enumerate(mean_offsets):
        bar_axis.text(index, value + 0.9, f"{value:.2f}", ha="center", fontsize=9)
    bar_axis.text(
        0.98, 0.96,
        "PPO-SAC、PPO-SCA与动态质心接近重合\n是策略结果，不是重复绘图",
        transform=bar_axis.transAxes, ha="right", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
    )

    fig.suptitle(f"轨迹策略差异解析（共同随机种子 {seed}）", y=0.99)
    fig.savefig(output / "05_轨迹差异放大.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def write_report(report: dict, output: Path) -> None:
    lines = [
        "# v45 公平方法筛查",
        "",
        "> 口径：每种学习方法 1 个训练种子，20 个共同随机评估种子。",
        "> 本结果只用于决定正式多训练种子矩阵，不替代最终统计结论。",
        "",
        "| 排名 | 方法 | 端到端时延/ms | 95% CI | 平均队列/Mbit | 平均功率/W | 软目标 J |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, method in enumerate(report["delay_ranking"], start=1):
        stats = report["method_summaries"][method]
        delay = stats["task_delay_weighted_ms"]
        lines.append(
            f"| {rank} | {METHOD_LABELS[method]} | {delay['mean']:.2f} | "
            f"[{delay['ci95_low']:.2f}, {delay['ci95_high']:.2f}] | "
            f"{stats['queue_total_mbits']['mean']:.2f} | "
            f"{stats['average_power_w']['mean']:.2f} | "
            f"{stats['soft_objective_j']['mean']:.5f} |"
        )
    lines.extend(["", "## TD3-SAC 配对比较", ""])
    for method, values in report["paired_baseline_minus_td3"].items():
        delay = values["task_delay_weighted_ms"]
        direction = "TD3-SAC更快" if delay["mean"] > 0 else "TD3-SAC更慢"
        lines.append(
            f"- {METHOD_LABELS[method]} - TD3-SAC = {delay['mean']:+.2f} ms，"
            f"95% CI [{delay['ci95_low']:+.2f}, {delay['ci95_high']:+.2f}]，"
            f"{direction}。"
        )
    ppo_delta = report["paired_baseline_minus_td3"]["ppo_sac"][
        "task_delay_weighted_ms"
    ]
    lines.extend([
        "",
        "## 筛查结论",
        "",
        (
            f"- 本轮第一名为 PPO-SAC；相对 TD3-SAC 平均快 "
            f"{-ppo_delta['mean']:.2f} ms，95% CI "
            f"[{-ppo_delta['ci95_high']:.2f}, {-ppo_delta['ci95_low']:.2f}]。"
        ),
        "- TD3-SAC 排名第二，但显著优于固定初始站位、动态质心、PPO-SCA 和 No-Coop。",
        "- 该结论基于每种学习方法 1 个训练种子；正式论文强主张仍需补多个训练种子。",
        "- PPO-SAC、PPO-SCA 与动态质心的绝对轨迹接近重合，但轨迹数组和网络参数均不相同；三者都学成了近似质心跟踪，差异见 `05_轨迹差异放大.png`。",
    ])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="8000-8019")
    parser.add_argument("--trajectory_seed", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    seeds = parse_seed_spec(args.seeds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    missing = [str(path) for path in DEFAULT_CHECKPOINTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing final checkpoints: {missing}")

    policies = {
        method: load_method_policy(method, path, args.device)
        for method, path in DEFAULT_CHECKPOINTS.items()
    }
    reference = scenario_signature(policies["td3_sac"]["args"])
    mismatches = {
        method: scenario_signature(policy["args"])
        for method, policy in policies.items()
        if scenario_signature(policy["args"]) != reference
    }
    if mismatches:
        raise RuntimeError(
            "Checkpoint scenarios are not identical: "
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )

    td3 = policies["td3_sac"]
    policies["fixed"] = {
        **td3,
        "name": "fixed",
        "macro_policy": V34MacroPolicy("fixed", 0.0, 0.0),
    }
    policies["centroid"] = {
        **td3,
        "name": "centroid",
        "macro_policy": make_centroid_offset_policy(np.zeros((3, 2))),
    }

    manifest = {
        "evaluation_seeds": seeds,
        "trajectory_seed": int(args.trajectory_seed),
        "scenario_signature": reference,
        "checkpoints": {
            method: {"path": str(path.resolve()), "sha256": sha256(path)}
            for method, path in DEFAULT_CHECKPOINTS.items()
        },
        "training_budget": {
            "stage1_macro_episodes": 300,
            "stage2_joint_or_continuation_episodes": 200,
            "checkpoint_rule": "final only",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows: list[dict] = []
    for seed in seeds:
        for method in METHOD_ORDER:
            row = evaluate_one(policies[method], seed)
            rows.append(row)
            print(
                f"seed={seed} method={method} "
                f"delay={row['task_delay_weighted_ms']:.2f}ms "
                f"queue={row['queue_total_mbits']:.2f}Mbit",
                flush=True,
            )
    write_csv(output / "per_seed_raw.csv", rows)
    report = summarize(rows, seeds)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(report, output)
    plot_ranking(report, output)
    plot_convergence(output)
    plot_paired_td3_ppo(rows, output)
    save_and_plot_trajectories(policies, output, int(args.trajectory_seed))
    plot_trajectory_differences(output, int(args.trajectory_seed))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
