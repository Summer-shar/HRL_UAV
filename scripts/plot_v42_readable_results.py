"""Create reader-facing v42 trajectory and convergence figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_v42_macro_only_pilot import load_policies
from compare_common import evaluate_policy, make_env_from_args
from run_v34_cooperative_mobility_gate import V34MacroPolicy
from train_centralized_macro import configure_macro_objective


DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v42_macro_only_pilot"
    / "formal"
    / "seed1"
    / "joint_td3_sac"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_TRAIN_CSV = DEFAULT_CHECKPOINT.parents[1] / "train_td3.csv"
DEFAULT_EVAL_CSV = DEFAULT_CHECKPOINT.parents[1] / "eval_central_td3.csv"
DEFAULT_AUDIT_REPORT = (
    ROOT / "result_archive" / "v42_macro_only_pilot_audit" / "report.json"
)
DEFAULT_OUTPUT = ROOT / "result_archive" / "v42_macro_only_pilot_audit"
COLORS = ("#2878B5", "#F28E2B", "#2CA02C")


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
    })


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if values.size >= window:
        result[window - 1 :] = np.convolve(
            values, np.ones(window, dtype=np.float64) / window, mode="valid"
        )
    return result


def replay_one(args, macro_policy, micro_policy, seed: int) -> dict:
    env = make_env_from_args(args, record_trajectory=True)
    configure_macro_objective(env, args, getattr(args, "macro_shaping_final", 0.0))
    eval_args = argparse.Namespace(**vars(args))
    eval_args.eval_episodes = 1
    eval_args.eval_base_seed = int(seed)
    metrics = evaluate_policy(env, eval_args, macro_policy, micro_policy)
    base_env = getattr(env, "env", env)
    return {
        "metrics": metrics,
        "uavs": np.asarray(base_env.uav_trajectory, dtype=np.float32),
        "centroids": np.asarray(base_env.centroid_trajectory, dtype=np.float32),
        "users": np.asarray(base_env.users_trajectory, dtype=np.float32),
        "allocations": np.asarray(base_env.user_allocations, dtype=np.int32),
    }


def plot_trajectory_comparison(traces: dict[str, dict], output: Path, seed: int) -> None:
    configure_plotting()
    max_user_diff = float(np.max(np.abs(
        traces["centroid"]["users"] - traces["td3"]["users"]
    )))
    if max_user_diff > 1e-6:
        raise ValueError("The paired trajectory replay did not preserve CRN user motion")

    all_points = []
    for trace in traces.values():
        all_points.extend((trace["users"].reshape(-1, 2), trace["uavs"].reshape(-1, 2)))
    extent = np.concatenate(all_points, axis=0)
    lower = np.min(extent, axis=0) - 25.0
    upper = np.max(extent, axis=0) + 25.0

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), sharex=True, sharey=True)
    labels = {
        "centroid": "(a) 动态质心跟踪",
        "td3": "(b) v42 宏观 TD3 最终策略",
    }
    for axis, name in zip(axes, ("centroid", "td3")):
        trace = traces[name]
        stride = max(1, trace["users"].shape[0] // 350)
        for zone, color in enumerate(COLORS):
            for user_index in np.flatnonzero(trace["allocations"] == zone):
                axis.plot(
                    trace["users"][::stride, user_index, 0],
                    trace["users"][::stride, user_index, 1],
                    color=color,
                    linewidth=0.55,
                    alpha=0.18,
                )
            axis.plot(
                trace["centroids"][:, zone, 0],
                trace["centroids"][:, zone, 1],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
            )
            axis.plot(
                trace["uavs"][:, zone, 0],
                trace["uavs"][:, zone, 1],
                color=color,
                linewidth=3.2,
            )
            axis.scatter(
                trace["uavs"][0, zone, 0],
                trace["uavs"][0, zone, 1],
                marker="s",
                s=48,
                facecolors="white",
                edgecolors=color,
                linewidths=1.5,
                zorder=5,
            )
            axis.scatter(
                trace["uavs"][-1, zone, 0],
                trace["uavs"][-1, zone, 1],
                marker="*",
                s=130,
                color=color,
                edgecolors="#222222",
                linewidths=0.8,
                zorder=6,
            )
        delay = float(trace["metrics"]["task_delay_weighted_ms"])
        u2g = float(trace["metrics"]["u2g"])
        axis.set_title(
            f"{labels[name]}\n端到端时延 {delay:.1f} ms，"
            f"UAV-质心平均偏离 {u2g:.1f} m"
        )
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("X 坐标 / m")
        axis.grid(linestyle="--", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Y 坐标 / m")
    handles = [
        Line2D([0], [0], color="#777777", alpha=0.35, label="地面用户轨迹"),
        Line2D([0], [0], color="#333333", linestyle="--", label="动态业务质心"),
        Line2D([0], [0], color="#333333", linewidth=3.0, label="UAV 轨迹"),
        Line2D([0], [0], marker="s", color="none", markeredgecolor="#333333", label="UAV 起点"),
        Line2D([0], [0], marker="*", color="none", markeredgecolor="#333333", markersize=11, label="UAV 终点"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(f"同一地面用户随机轨迹下的对照（seed={seed}）", fontweight="bold")
    fig.text(
        0.5,
        0.125,
        "TD3 将 UAV 带离业务质心；本例中移动没有换来协作收益，反而增加上传距离和时延。",
        ha="center",
        color="#B23A3A",
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.25, wspace=0.12)
    fig.savefig(output / "02_轨迹直观对比.png", dpi=300, facecolor="white")
    plt.close(fig)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_convergence(
    train_csv: Path,
    eval_csv: Path,
    audit_report: Path,
    output: Path,
) -> None:
    configure_plotting()
    train = read_csv(train_csv)
    evaluation = read_csv(eval_csv)
    report = json.loads(audit_report.read_text(encoding="utf-8"))
    episodes = np.asarray([int(row["episode"]) for row in train])
    train_j = np.asarray([float(row["J_train"]) for row in train])
    eval_episodes = np.asarray([int(row["episode"]) for row in evaluation])
    eval_delay = np.asarray([
        float(row["task_delay_weighted_ms"]) for row in evaluation
    ])
    centroid_delay = float(
        report["policy_means"]["centroid"]["task_delay_weighted_ms"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    axes[0].plot(episodes, train_j, color="#7AA6C2", alpha=0.25, linewidth=0.8, label="每轮真实值")
    axes[0].plot(
        episodes,
        rolling_mean(train_j, 30),
        color="#2878B5",
        linewidth=2.4,
        label="30 轮滑动平均",
    )
    axes[0].set_title("(a) 训练目标没有稳定下降")
    axes[0].set_xlabel("宏观训练轮次")
    axes[0].set_ylabel("训练软目标 J（越低越好）")
    axes[0].legend(frameon=False)
    axes[0].grid(linestyle="--", alpha=0.25)

    axes[1].plot(
        eval_episodes,
        eval_delay,
        color="#E45756",
        marker="o",
        linewidth=2.4,
        label="TD3 确定性评估",
    )
    axes[1].axhline(
        centroid_delay,
        color="#2E7D6E",
        linestyle="--",
        linewidth=2.0,
        label=f"质心基线 {centroid_delay:.1f} ms",
    )
    axes[1].annotate(
        f"最终 {eval_delay[-1]:.1f} ms\n仍高于质心基线",
        xy=(eval_episodes[-1], eval_delay[-1]),
        xytext=(-82, -42),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#B23A3A"},
        color="#B23A3A",
    )
    axes[1].set_title("(b) 确定性评估后期反而恶化")
    axes[1].set_xlabel("宏观训练轮次")
    axes[1].set_ylabel("优先级加权端到端时延 / ms")
    axes[1].legend(frameon=False)
    axes[1].grid(linestyle="--", alpha=0.25)
    fig.suptitle("v42 收敛判断：未收敛到优于质心的轨迹策略", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "03_收敛判断.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    cli = parser.parse_args()
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    args, td3_policy, micro_policy = load_policies(
        cli.checkpoint.resolve(), cli.device
    )
    traces = {
        "centroid": replay_one(
            args, V34MacroPolicy("centroid", 0.0, 0.5), micro_policy, cli.seed
        ),
        "td3": replay_one(args, td3_policy, micro_policy, cli.seed),
    }
    np.savez_compressed(
        output / f"representative_trajectory_seed{cli.seed}.npz",
        centroid_uavs=traces["centroid"]["uavs"],
        td3_uavs=traces["td3"]["uavs"],
        centroids=traces["centroid"]["centroids"],
        users=traces["centroid"]["users"],
        allocations=traces["centroid"]["allocations"],
    )
    plot_trajectory_comparison(traces, output, cli.seed)
    plot_convergence(
        cli.train_csv.resolve(),
        cli.eval_csv.resolve(),
        cli.audit_report.resolve(),
        output,
    )
    print(
        json.dumps(
            {
                name: {
                    "delay_ms": float(trace["metrics"]["task_delay_weighted_ms"]),
                    "u2g_m": float(trace["metrics"]["u2g"]),
                }
                for name, trace in traces.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
