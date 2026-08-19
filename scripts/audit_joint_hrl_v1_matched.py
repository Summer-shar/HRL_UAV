"""Matched replay of the historical v1 TD3 trajectory policy.

Both macro policies use the final micro SAC stored in the same v1 checkpoint.
This isolates the trajectory policy from the micro-policy training gain that was
mixed into the original 200-vs-600 episode comparison.
"""

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
from search_position_oracle import make_centroid_offset_policy
from train_centralized_macro import configure_macro_objective


DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "joint_hrl_v1"
    / "formal"
    / "seed1"
    / "joint_td3_sac"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_OUTPUT = ROOT / "result_archive" / "joint_hrl_v1_matched_replay"
DEFAULT_SEEDS = tuple(range(1000, 1020))
COLORS = ("#2878B5", "#F28E2B", "#2CA02C")


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


def mean_ci95(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    critical = 2.093 if array.size == 20 else 1.96
    half_width = critical * std / np.sqrt(array.size) if array.size > 1 else 0.0
    mean = float(np.mean(array))
    return {
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "std": std,
        "n": int(array.size),
    }


def make_eval_env(args, record_trajectory: bool):
    env = make_env_from_args(args, record_trajectory=record_trajectory)
    configure_macro_objective(env, args, getattr(args, "macro_shaping_final", 0.0))
    return env


def replay(args, macro_policy, micro_policy, seed: int, record: bool = False) -> dict:
    env = make_eval_env(args, record_trajectory=record)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.eval_episodes = 1
    eval_args.eval_base_seed = int(seed)
    metrics = evaluate_policy(env, eval_args, macro_policy, micro_policy)
    result = {
        "task_delay_weighted_ms": float(metrics["task_delay_weighted_ms"]),
        "J": float(metrics["J"]),
        "queue_mbits": float(metrics["q_high"] + metrics["q_low"]),
        "average_power_w": float(metrics["average_power_w"]),
        "u2g_distance_m": float(metrics["u2g"]),
        "coop_ratio": float(metrics["coop_ratio"]),
    }
    if record:
        base_env = getattr(env, "env", env)
        result["uavs"] = np.asarray(base_env.uav_trajectory, dtype=np.float32)
        result["centroids"] = np.asarray(
            base_env.centroid_trajectory, dtype=np.float32
        )
        result["users"] = np.asarray(base_env.users_trajectory, dtype=np.float32)
        result["allocations"] = np.asarray(
            base_env.user_allocations, dtype=np.int32
        )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
    })


def plot_paired_delay(rows: list[dict], report: dict, output: Path) -> None:
    configure_plotting()
    td3 = [row for row in rows if row["policy"] == "td3"]
    centroid = {int(row["seed"]): row for row in rows if row["policy"] == "centroid"}
    deltas = np.asarray([
        float(row["task_delay_weighted_ms"])
        - float(centroid[int(row["seed"])]["task_delay_weighted_ms"])
        for row in td3
    ])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    means = [
        report["policy_means"]["centroid"]["task_delay_weighted_ms"],
        report["policy_means"]["td3"]["task_delay_weighted_ms"],
    ]
    axes[0].bar(
        ["动态质心追踪", "旧版TD3轨迹"],
        means,
        color=["#2E7D6E", "#2878B5"],
        width=0.58,
    )
    axes[0].set_ylabel("优先级加权端到端时延 / ms")
    axes[0].set_title("同一最终微观SAC下的平均时延")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    for index, value in enumerate(means):
        axes[0].text(index, value + 3.0, f"{value:.1f}", ha="center")

    colors = np.where(deltas < 0.0, "#2E7D6E", "#D95F59")
    axes[1].bar(np.arange(deltas.size) + 1, deltas, color=colors)
    axes[1].axhline(0.0, color="#333333", linewidth=1.0)
    axes[1].set_xlabel("共同随机种子序号")
    axes[1].set_ylabel("TD3 - 质心 / ms（负值表示TD3更快）")
    axes[1].set_title("逐种子公平配对差")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)
    summary = report["paired"][
        "td3_minus_centroid_task_delay_weighted_ms"
    ]
    axes[1].text(
        0.03,
        0.96,
        f"均值 {summary['mean']:+.2f} ms\n"
        f"95% CI [{summary['ci95_low']:+.2f}, {summary['ci95_high']:+.2f}]",
        transform=axes[1].transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(output / "01_matched_delay.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_trajectories(
    traces: dict[str, dict], output: Path, seed: int, label: str
) -> None:
    configure_plotting()
    user_difference = float(np.max(np.abs(
        traces["centroid"]["users"] - traces["td3"]["users"]
    )))
    if user_difference > 1e-6:
        raise ValueError(
            f"CRN user trajectories differ by {user_difference:.3e} m"
        )
    extent = np.concatenate([
        traces[name]["users"].reshape(-1, 2)
        for name in ("centroid", "td3")
    ] + [
        traces[name]["uavs"].reshape(-1, 2)
        for name in ("centroid", "td3")
    ])
    lower = np.min(extent, axis=0) - 25.0
    upper = np.max(extent, axis=0) + 25.0
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), sharex=True, sharey=True)
    labels = {
        "centroid": "(a) 动态业务质心追踪",
        "td3": f"(b) {label} TD3轨迹",
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
            )
            axis.plot(
                trace["uavs"][:, zone, 0],
                trace["uavs"][:, zone, 1],
                color=color,
                linewidth=3.0,
            )
            axis.scatter(
                *trace["uavs"][0, zone], marker="s", s=48,
                facecolors="white", edgecolors=color, linewidths=1.5, zorder=5
            )
            axis.scatter(
                *trace["uavs"][-1, zone], marker="*", s=125,
                color=color, edgecolors="#222222", linewidths=0.8, zorder=6
            )
        axis.set_title(
            f"{labels[name]}\n端到端时延 {trace['task_delay_weighted_ms']:.1f} ms"
        )
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("X坐标 / m")
        axis.grid(linestyle="--", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Y坐标 / m")
    handles = [
        Line2D([0], [0], color="#777777", alpha=0.35, label="地面用户轨迹"),
        Line2D([0], [0], color="#333333", linestyle="--", label="动态业务质心"),
        Line2D([0], [0], color="#333333", linewidth=3.0, label="UAV轨迹"),
        Line2D([0], [0], marker="s", color="none", markeredgecolor="#333333", label="起点"),
        Line2D([0], [0], marker="*", color="none", markeredgecolor="#333333", markersize=11, label="终点"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        f"{label}同微观策略公平轨迹对比（seed={seed}）",
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.18, wspace=0.12)
    fig.savefig(output / "02_matched_trajectory.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="1000-1019")
    parser.add_argument("--trajectory-seed", type=int, default=None)
    parser.add_argument("--label", default="历史v1")
    parser.add_argument("--reuse-raw", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args_cli = parser.parse_args()

    output = args_cli.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_spec(args_cli.seeds)
    checkpoint_args, td3_policy, micro_policy = load_policies(
        args_cli.checkpoint.resolve(), args_cli.device
    )
    centroid_policy = make_centroid_offset_policy(np.zeros((3, 2)))

    raw_path = output / "per_seed_raw.csv"
    if args_cli.reuse_raw:
        rows = read_csv(raw_path)
        observed_seeds = sorted({int(row["seed"]) for row in rows})
        if observed_seeds != seeds or len(rows) != 2 * len(seeds):
            raise ValueError("Existing raw data do not match the requested seeds")
    else:
        rows: list[dict] = []
        for seed in seeds:
            for name, policy in (
                ("centroid", centroid_policy),
                ("td3", td3_policy),
            ):
                metrics = replay(checkpoint_args, policy, micro_policy, seed)
                rows.append({"seed": seed, "policy": name, **metrics})
                print(
                    f"seed={seed} policy={name} "
                    f"delay={metrics['task_delay_weighted_ms']:.2f}ms "
                    f"J={metrics['J']:.5f}"
                )
        write_csv(raw_path, rows)

    by_policy = {
        name: {int(row["seed"]): row for row in rows if row["policy"] == name}
        for name in ("centroid", "td3")
    }
    fields = (
        "task_delay_weighted_ms",
        "J",
        "queue_mbits",
        "average_power_w",
        "u2g_distance_m",
        "coop_ratio",
    )
    policy_means = {
        name: {
            field: float(np.mean([
                float(by_policy[name][seed][field]) for seed in seeds
            ]))
            for field in fields
        }
        for name in ("centroid", "td3")
    }
    paired = {}
    paired_rows = []
    for field in fields:
        differences = [
            float(by_policy["td3"][seed][field])
            - float(by_policy["centroid"][seed][field])
            for seed in seeds
        ]
        paired[f"td3_minus_centroid_{field}"] = mean_ci95(differences)
    for seed in seeds:
        paired_rows.append({
            "seed": seed,
            **{
                f"td3_minus_centroid_{field}": (
                    float(by_policy["td3"][seed][field])
                    - float(by_policy["centroid"][seed][field])
                )
                for field in fields
            },
        })
    write_csv(output / "paired_deltas.csv", paired_rows)
    delay_deltas = np.asarray([
        row["td3_minus_centroid_task_delay_weighted_ms"]
        for row in paired_rows
    ])
    report = {
        "comparison": f"{args_cli.label}_final_td3_vs_dynamic_centroid",
        "micro_policy": "same final SAC from selected joint checkpoint",
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "macro_action_mode": str(checkpoint_args.macro_action_mode),
        "seeds": seeds,
        "policy_means": policy_means,
        "paired": paired,
        "td3_faster_seed_count": int(np.sum(delay_deltas < 0.0)),
        "centroid_faster_seed_count": int(np.sum(delay_deltas > 0.0)),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_paired_delay(rows, report, output)

    if args_cli.trajectory_seed is None:
        mean_delta = float(np.mean(delay_deltas))
        trajectory_seed = int(
            paired_rows[int(np.argmin(np.abs(delay_deltas - mean_delta)))]["seed"]
        )
    else:
        trajectory_seed = int(args_cli.trajectory_seed)
    traces = {
        "centroid": replay(
            checkpoint_args, centroid_policy, micro_policy, trajectory_seed, True
        ),
        "td3": replay(
            checkpoint_args, td3_policy, micro_policy, trajectory_seed, True
        ),
    }
    np.savez_compressed(
        output / f"trajectory_seed{trajectory_seed}.npz",
        centroid_uavs=traces["centroid"]["uavs"],
        td3_uavs=traces["td3"]["uavs"],
        centroids=traces["centroid"]["centroids"],
        users=traces["centroid"]["users"],
        allocations=traces["centroid"]["allocations"],
    )
    plot_trajectories(traces, output, trajectory_seed, str(args_cli.label))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
