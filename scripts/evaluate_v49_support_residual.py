"""Evaluate and plot the seed-1 v49 support-preserving residual HRL pilot."""

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
import evaluate_v48_priority_gated as v48  # noqa: E402


RESULTS_VERSION = "tcom_v49_support_residual_synchronous"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v49_support_residual_seed1"


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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_ci95(values) -> dict:
    return v45.mean_ci95(values)


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
            yerr=[
                [stats["mean"] - stats["ci95_low"]],
                [stats["ci95_high"] - stats["mean"]],
            ],
            color="#222222",
            capsize=5,
        )
        axis.scatter(
            np.zeros_like(values),
            values,
            facecolors="white",
            edgecolors="#2878B5",
            zorder=3,
        )
        axis.set_xticks([0], ["TD3-残差SAC"])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle("支撑集残差分层策略：20个共同评估种子")
    fig.tight_layout()
    fig.savefig(output / "01_时延队列路由.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_trajectory(trace: dict, delay_ms: float, output: Path) -> None:
    v45.configure_plotting()
    fig, axis = plt.subplots(figsize=(8.2, 7.4))
    colors = ("#2878B5", "#E67E22", "#2E8B57")
    users = trace["users"]
    allocations = np.asarray(trace["user_allocations"], dtype=np.int64)
    for user in range(users.shape[1]):
        zone = int(allocations[user])
        axis.plot(
            users[:, user, 0],
            users[:, user, 1],
            color=colors[zone],
            alpha=0.18,
            linewidth=0.65,
        )
    for index, color in enumerate(colors):
        axis.plot(
            trace["centroids"][:, index, 0],
            trace["centroids"][:, index, 1],
            linestyle="--",
            color=color,
            linewidth=1.4,
            alpha=0.85,
            label=f"区域{index + 1}业务质心",
        )
        axis.plot(
            trace["uavs"][:, index, 0],
            trace["uavs"][:, index, 1],
            color=color,
            linewidth=2.8,
            label=f"UAV {index + 1}",
        )
        axis.scatter(
            *trace["uavs"][0, index],
            marker="s",
            s=42,
            facecolor="white",
            edgecolor=color,
            zorder=4,
        )
        axis.scatter(
            *trace["uavs"][-1, index],
            marker="*",
            s=110,
            facecolor=color,
            edgecolor="#222222",
            zorder=5,
        )
    axis.set_title(f"无人机、用户与动态业务质心轨迹（{delay_ms:.1f} ms）")
    axis.set_xlabel("X坐标 / m")
    axis.set_ylabel("Y坐标 / m")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(linestyle="--", alpha=0.22)
    axis.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "02_联合轨迹.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_convergence(stage: str, train_seed: int, output: Path) -> None:
    v45.configure_plotting()
    root = ROOT / "results" / RESULTS_VERSION / stage / f"seed{train_seed}"
    phase_paths = (
        (
            root / "micro_pretrain_joint_td3_sac" / "eval_central_macro.csv",
            root / "micro_pretrain_joint_td3_sac" / "train_centralized.csv",
        ),
        (
            root / "joint_td3_sac" / "eval_central_td3.csv",
            root / "joint_td3_sac" / "train_td3.csv",
        ),
    )
    eval_series = []
    train_series = []
    offset = 0
    for eval_path, train_path in phase_paths:
        with eval_path.open("r", encoding="utf-8-sig", newline="") as handle:
            eval_rows = list(csv.DictReader(handle))
        with train_path.open("r", encoding="utf-8-sig", newline="") as handle:
            train_rows = list(csv.DictReader(handle))
        eval_x = np.asarray([int(row["episode"]) + offset for row in eval_rows])
        eval_y = np.asarray(
            [float(row["task_delay_weighted_ms"]) for row in eval_rows]
        )
        train_x = np.arange(1, len(train_rows) + 1, dtype=np.int64) + offset
        train_y = np.asarray([float(row["J_train"]) for row in train_rows])
        smooth_y = np.asarray(
            [
                float(np.mean(train_y[max(0, index - 24) : index + 1]))
                for index in range(train_y.size)
            ]
        )
        eval_series.append((eval_x, eval_y))
        train_series.append((train_x, smooth_y))
        offset += len(train_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0))
    colors = ("#2878B5", "#D9534F")
    labels = ("下层独立训练", "上下层同步训练")
    for (x, y), color, label in zip(eval_series, colors, labels):
        axes[0].plot(
            x,
            y,
            marker="o",
            markersize=3.5,
            linewidth=2.0,
            color=color,
            label=label,
        )
    for (x, y), color, label in zip(train_series, colors, labels):
        axes[1].plot(x, y, linewidth=1.8, color=color, label=label)
    boundary = len(train_series[0][0])
    for axis in axes:
        axis.axvline(boundary, color="#444444", linestyle="--", linewidth=1.2)
        axis.grid(linestyle="--", alpha=0.25)
        axis.set_xlabel("累计训练轮次")
        axis.legend()
    axes[0].set_ylabel("优先级加权端到端时延 / ms")
    axes[0].set_title("(a) 固定种子评估时延")
    axes[1].set_ylabel("逐轮真实训练目标 J（25轮滑动平均）")
    axes[1].set_title("(b) 训练实际目标")
    axes[0].text(
        boundary,
        0.98,
        "  联训开始",
        transform=axes[0].get_xaxis_transform(),
        va="top",
        ha="left",
    )
    fig.suptitle("分层强化学习完整训练过程")
    fig.tight_layout()
    fig.savefig(output / "03_完整收敛曲线.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("formal", "diagnostic", "smoke"), default="formal")
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
    for seed in v45.parse_seed_spec(args.eval_seeds):
        row, high, low, _ = v48.evaluate_detailed(policy, seed)
        rows.append(row)
        high_matrices.append(high)
        low_matrices.append(low)
        print(
            f"seed={seed} delay={row['task_delay_weighted_ms']:.2f}ms "
            f"queue={row['queue_total_mbits']:.2f}Mbit",
            flush=True,
        )
    trajectory_row, _, _, trace = v48.evaluate_detailed(
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
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(output / "trajectory_seed8000.npz", **trace)
    plot_summary(rows, output)
    plot_trajectory(trace, trajectory_row["task_delay_weighted_ms"], output)
    plot_convergence(args.stage, args.train_seed, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
