"""Evaluate the seed-1 v48 priority-gated TD3-SAC pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_v45_fair_screen as v45  # noqa: E402


RESULTS_VERSION = "tcom_v48_priority_gated_synchronous"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v48_priority_gated_seed1"


def parse_seeds(value: str) -> list[int]:
    return v45.parse_seed_spec(value)


def mean_ci95(values) -> dict:
    return v45.mean_ci95(values)


def checkpoint(stage: str, train_seed: int) -> Path:
    return (
        ROOT
        / "results"
        / RESULTS_VERSION
        / stage
        / f"seed{train_seed}"
        / "joint_td3_sac"
        / "checkpoints"
        / "final.pt"
    )


def evaluate_detailed(policy: dict, seed: int, *, trajectory: bool = False):
    args = policy["args"]
    env = v45.make_eval_env(args, record_trajectory=trajectory)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.eval_episodes = 1
    eval_args.eval_base_seed = int(seed)
    metrics = v45.evaluate_policy(
        env,
        eval_args,
        policy["macro_policy"],
        policy["micro_policy"],
    )
    high_raw = np.asarray(metrics["split_ratios_high"], dtype=np.float64)
    low_raw = np.asarray(metrics["split_ratios_low"], dtype=np.float64)
    high_duty = np.sum(high_raw, axis=1, keepdims=True)
    low_duty = np.sum(low_raw, axis=1, keepdims=True)
    high = high_raw / np.maximum(high_duty, 1e-12)
    low = low_raw / np.maximum(low_duty, 1e-12)
    row = {
        "seed": int(seed),
        "task_delay_weighted_ms": float(metrics["task_delay_weighted_ms"]),
        "task_delay_high_ms": float(metrics["task_delay_high_ms"]),
        "task_delay_low_ms": float(metrics["task_delay_low_ms"]),
        "task_completion_rate": float(metrics["task_completion_rate"]),
        "queue_total_mbits": float(metrics["q_high"] + metrics["q_low"]),
        "queue_high_mbits": float(metrics["q_high"]),
        "queue_low_mbits": float(metrics["q_low"]),
        "average_power_w": float(metrics["average_power_w"]),
        "soft_objective_j": float(metrics["J"]),
        "coop_ratio": float(metrics["coop_ratio"]),
        "u2g_distance_m": float(metrics["u2g"]),
        "high_local_ratio": float(np.mean(np.diag(high))),
        "low_local_ratio": float(np.mean(np.diag(low))),
        "routing_service_duty": float(
            0.5 * (np.mean(high_duty) + np.mean(low_duty))
        ),
    }
    trace = None
    if trajectory:
        base = getattr(env, "env", env)
        trace = {
            "uavs": np.asarray(base.uav_trajectory, dtype=np.float32),
            "users": np.asarray(base.users_trajectory, dtype=np.float32),
            "centroids": np.asarray(base.centroid_trajectory, dtype=np.float32),
            "user_allocations": np.asarray(base.user_allocations, dtype=np.int64),
        }
    return row, high, low, trace


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict], output: Path) -> None:
    v45.configure_plotting()
    fields = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms", "(a) 端到端时延"),
        ("queue_total_mbits", "平均队列积压 / Mbit", "(b) 队列积压"),
        ("high_local_ratio", "高优先级本地处理比例", "(c) 高优路由"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))
    for axis, (field, ylabel, title) in zip(axes, fields):
        values = np.asarray([float(row[field]) for row in rows])
        stats = mean_ci95(values)
        axis.bar([0], [stats["mean"]], color="#2878B5", width=0.55)
        axis.errorbar(
            [0],
            [stats["mean"]],
            yerr=[[stats["mean"] - stats["ci95_low"]], [stats["ci95_high"] - stats["mean"]]],
            color="#222222",
            capsize=5,
        )
        axis.scatter(np.zeros_like(values), values, facecolors="white", edgecolors="#2878B5", zorder=3)
        axis.set_xticks([0], ["v48 TD3-SAC"])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle("v48 优先级门控路由：20个共同评估种子")
    fig.tight_layout()
    fig.savefig(output / "01_v48时延队列路由.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_trajectory(trace: dict, delay_ms: float, output: Path) -> None:
    v45.configure_plotting()
    fig, axis = plt.subplots(figsize=(7.2, 7.0))
    colors = ("#2878B5", "#E67E22", "#2E8B57")
    users = trace["users"]
    for user in range(users.shape[1]):
        zone = user % 3
        axis.plot(users[:, user, 0], users[:, user, 1], color=colors[zone], alpha=0.25, linewidth=0.75)
    for index, color in enumerate(colors):
        axis.plot(trace["centroids"][:, index, 0], trace["centroids"][:, index, 1], linestyle="--", color=color, linewidth=1.3)
        axis.plot(trace["uavs"][:, index, 0], trace["uavs"][:, index, 1], color=color, linewidth=2.8, label=f"UAV {index + 1}")
        axis.scatter(*trace["uavs"][0, index], marker="s", s=42, facecolor="white", edgecolor=color, zorder=4)
        axis.scatter(*trace["uavs"][-1, index], marker="*", s=110, facecolor=color, edgecolor="#222222", zorder=5)
    axis.set_title(f"v48 最终策略轨迹（端到端时延 {delay_ms:.1f} ms）")
    axis.set_xlabel("X坐标 / m")
    axis.set_ylabel("Y坐标 / m")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(linestyle="--", alpha=0.22)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "02_v48最终轨迹.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--trajectory-seed", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = checkpoint(args.stage, args.train_seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    policy = v45.load_method_policy("joint_td3_sac", path, args.device)
    rows = []
    high_matrices = []
    low_matrices = []
    for seed in parse_seeds(args.eval_seeds):
        row, high, low, _ = evaluate_detailed(policy, seed)
        rows.append(row)
        high_matrices.append(high)
        low_matrices.append(low)
        print(
            f"seed={seed} delay={row['task_delay_weighted_ms']:.2f}ms "
            f"high_local={row['high_local_ratio']:.3f}",
            flush=True,
        )
    trajectory_row, _, _, trace = evaluate_detailed(
        policy,
        args.trajectory_seed,
        trajectory=True,
    )
    write_csv(output / "per_seed_raw.csv", rows)
    numeric_fields = [field for field in rows[0] if field != "seed"]
    report = {
        "checkpoint": str(path),
        "summaries": {
            field: mean_ci95([float(row[field]) for row in rows])
            for field in numeric_fields
        },
        "mean_split_ratios_high": np.mean(high_matrices, axis=0).tolist(),
        "mean_split_ratios_low": np.mean(low_matrices, axis=0).tolist(),
        "v47_seed1_reference_ms": {
            "joint_td3_sac": 710.9259,
            "cppo_sca": 649.46,
            "fixed_sac": 668.71,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "trajectory_seed8000.npz",
        **trace,
    )
    plot_summary(rows, output)
    plot_trajectory(trace, trajectory_row["task_delay_weighted_ms"], output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
