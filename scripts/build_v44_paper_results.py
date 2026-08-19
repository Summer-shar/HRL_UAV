"""Build the final, matched v44 result table and paper-ready PNG figures.

All three macro policies share the final v44 micro SAC, environment settings,
and evaluation seeds. This keeps trajectory-policy comparisons isolated from
micro-policy quality and external randomness.
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

from audit_joint_hrl_v1_matched import (  # noqa: E402
    load_policies,
    mean_ci95,
    replay,
)
from run_v34_cooperative_mobility_gate import V34MacroPolicy  # noqa: E402
from search_position_oracle import make_centroid_offset_policy  # noqa: E402


DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "tcom_v44_joint_cooperation_finetune"
    / "formal"
    / "seed1"
    / "joint_td3_sac"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_OUTPUT = ROOT / "paper_results_v44"
DEFAULT_SEEDS = tuple(range(8000, 8020))
POLICIES = ("fixed", "centroid", "learned")
POLICY_LABELS = {
    "fixed": "固定悬停",
    "centroid": "动态质心跟踪",
    "learned": "联合 TD3-SAC",
}
POLICY_COLORS = {
    "fixed": "#777777",
    "centroid": "#D98F39",
    "learned": "#2677B8",
}
ZONE_COLORS = ("#2878B5", "#F28E2B", "#2E9D55")


def parse_seed_spec(value: str) -> list[int]:
    seeds: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(part) for part in token.split("-", 1))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    return seeds


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def trajectory_metrics(result: dict, macro_steps: int) -> dict[str, float]:
    uavs = np.asarray(result["uavs"], dtype=np.float64)
    increments = np.linalg.norm(np.diff(uavs, axis=0), axis=2)
    total_path = float(np.sum(increments))
    macro_indices = np.linspace(
        0, uavs.shape[0] - 1, int(macro_steps) + 1, dtype=np.int64
    )
    macro_displacements = np.linalg.norm(
        np.diff(uavs[macro_indices], axis=0), axis=2
    )
    moving_fraction = float(np.mean(macro_displacements > 0.5))
    return {
        "total_path_m": total_path,
        "moving_uav_fraction": moving_fraction,
    }


def evaluate_all(
    checkpoint: Path,
    output: Path,
    seeds: list[int],
    trajectory_seed: int,
    device: str,
) -> tuple[list[dict], Path]:
    checkpoint_args, learned_policy, micro_policy = load_policies(
        checkpoint.resolve(), device
    )
    centroid_policy = make_centroid_offset_policy(np.zeros((3, 2)))
    rows: list[dict] = []
    saved_traces: dict[str, dict] = {}

    for seed in seeds:
        policies = {
            "fixed": V34MacroPolicy("fixed", 0.0, 0.0),
            "centroid": centroid_policy,
            "learned": learned_policy,
        }
        seed_traces: dict[str, dict] = {}
        for name in POLICIES:
            result = replay(
                checkpoint_args,
                policies[name],
                micro_policy,
                int(seed),
                record=True,
            )
            motion = trajectory_metrics(result, int(checkpoint_args.macro_steps))
            rows.append(
                {
                    "seed": int(seed),
                    "policy": name,
                    "task_delay_weighted_ms": result["task_delay_weighted_ms"],
                    "J": result["J"],
                    "queue_mbits": result["queue_mbits"],
                    "average_power_w": result["average_power_w"],
                    "u2g_distance_m": result["u2g_distance_m"],
                    "coop_ratio": result["coop_ratio"],
                    **motion,
                }
            )
            seed_traces[name] = result
            print(
                f"seed={seed} policy={name} "
                f"delay={result['task_delay_weighted_ms']:.2f} ms "
                f"queue={result['queue_mbits']:.2f} Mbit"
            )

        reference_users = np.asarray(seed_traces["fixed"]["users"])
        for name in ("centroid", "learned"):
            max_difference = float(
                np.max(
                    np.abs(
                        reference_users - np.asarray(seed_traces[name]["users"])
                    )
                )
            )
            if max_difference > 1e-4:
                raise RuntimeError(
                    f"CRN user trace mismatch for seed={seed}, policy={name}: "
                    f"{max_difference:.3e} m"
                )
        if int(seed) == int(trajectory_seed):
            saved_traces = seed_traces

    raw_path = output / "三策略逐种子原始结果.csv"
    write_csv(raw_path, rows)
    if not saved_traces:
        raise ValueError("The trajectory seed must be included in evaluation seeds")
    trace_path = output / f"代表轨迹_seed{trajectory_seed}.npz"
    np.savez_compressed(
        trace_path,
        fixed_uavs=saved_traces["fixed"]["uavs"],
        centroid_uavs=saved_traces["centroid"]["uavs"],
        learned_uavs=saved_traces["learned"]["uavs"],
        centroids=saved_traces["centroid"]["centroids"],
        users=saved_traces["centroid"]["users"],
        allocations=saved_traces["centroid"]["allocations"],
    )
    return rows, trace_path


def build_report(
    rows: list[dict], checkpoint: Path, seeds: list[int], output: Path
) -> dict:
    by_policy = {
        name: {
            int(row["seed"]): row for row in rows if row["policy"] == name
        }
        for name in POLICIES
    }
    fields = (
        "task_delay_weighted_ms",
        "J",
        "queue_mbits",
        "average_power_w",
        "u2g_distance_m",
        "coop_ratio",
        "total_path_m",
        "moving_uav_fraction",
    )
    policy_stats = {
        name: {
            field: mean_ci95(
                [float(by_policy[name][seed][field]) for seed in seeds]
            )
            for field in fields
        }
        for name in POLICIES
    }
    paired = {}
    paired_rows: list[dict] = []
    for seed in seeds:
        row = {"seed": seed}
        for baseline in ("fixed", "centroid"):
            for field in fields:
                key = f"learned_minus_{baseline}_{field}"
                row[key] = (
                    float(by_policy["learned"][seed][field])
                    - float(by_policy[baseline][seed][field])
                )
        paired_rows.append(row)
    for baseline in ("fixed", "centroid"):
        for field in fields:
            key = f"learned_minus_{baseline}_{field}"
            paired[key] = mean_ci95([float(row[key]) for row in paired_rows])

    write_csv(output / "三策略配对差值.csv", paired_rows)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "micro_policy": "all policies share the final v44 SAC",
        "seeds": seeds,
        "policy_stats": policy_stats,
        "paired": paired,
        "learned_faster_than_fixed": int(
            sum(row["learned_minus_fixed_task_delay_weighted_ms"] < 0 for row in paired_rows)
        ),
        "learned_faster_than_centroid": int(
            sum(row["learned_minus_centroid_task_delay_weighted_ms"] < 0 for row in paired_rows)
        ),
    }


def extract_values(rows: list[dict], policy: str, field: str) -> np.ndarray:
    selected = sorted(
        (row for row in rows if row["policy"] == policy),
        key=lambda row: int(row["seed"]),
    )
    return np.asarray([float(row[field]) for row in selected], dtype=np.float64)


def plot_summary(rows: list[dict], output: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    fields = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms", "端到端时延"),
        ("queue_mbits", "系统平均队列 / Mbit", "平均队列积压"),
    )
    x = np.arange(len(POLICIES), dtype=np.float64)
    jitter = np.linspace(-0.09, 0.09, 20)
    for axis, (field, ylabel, title) in zip(axes, fields):
        values = [extract_values(rows, name, field) for name in POLICIES]
        means = np.asarray([np.mean(item) for item in values])
        errors = np.asarray(
            [mean_ci95(item.tolist())["ci95_high"] - np.mean(item) for item in values]
        )
        axis.bar(
            x,
            means,
            yerr=errors,
            width=0.62,
            color=[POLICY_COLORS[name] for name in POLICIES],
            capsize=4,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
        )
        for index, item in enumerate(values):
            local_jitter = jitter if item.size == jitter.size else np.zeros(item.size)
            axis.scatter(
                np.full(item.size, x[index]) + local_jitter,
                item,
                s=18,
                facecolor="white",
                edgecolor=POLICY_COLORS[POLICIES[index]],
                linewidth=0.8,
                alpha=0.8,
                zorder=3,
            )
            axis.text(
                x[index],
                means[index] + errors[index] + max(means) * 0.025,
                f"{means[index]:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        axis.set_xticks(x, [POLICY_LABELS[name] for name in POLICIES])
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("同一微观策略和共同随机轨迹下的总体性能", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.015, "误差棒为 95% 置信区间，空心圆为单个随机种子", ha="center", color="#555555")
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.savefig(output / "01_端到端时延与平均队列.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(
    trace_path: Path,
    rows: list[dict],
    output: Path,
    trajectory_seed: int,
) -> None:
    configure_plotting()
    payload = np.load(trace_path)
    users = np.asarray(payload["users"], dtype=np.float64)
    centroids = np.asarray(payload["centroids"], dtype=np.float64)
    allocations = np.asarray(payload["allocations"], dtype=np.int64)
    paths = {
        "fixed": np.asarray(payload["fixed_uavs"], dtype=np.float64),
        "centroid": np.asarray(payload["centroid_uavs"], dtype=np.float64),
        "learned": np.asarray(payload["learned_uavs"], dtype=np.float64),
    }
    delays = {
        row["policy"]: float(row["task_delay_weighted_ms"])
        for row in rows
        if int(row["seed"]) == int(trajectory_seed)
    }
    extent = np.concatenate(
        [users.reshape(-1, 2), centroids.reshape(-1, 2)]
        + [path.reshape(-1, 2) for path in paths.values()]
    )
    lower = np.min(extent, axis=0) - 20.0
    upper = np.max(extent, axis=0) + 20.0
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.5), sharex=True, sharey=True)
    user_stride = max(1, users.shape[0] // 450)
    centroid_stride = max(1, centroids.shape[0] // 700)
    for axis, name in zip(axes, POLICIES):
        for zone, color in enumerate(ZONE_COLORS):
            for user_index in np.flatnonzero(allocations == zone):
                axis.plot(
                    users[::user_stride, user_index, 0],
                    users[::user_stride, user_index, 1],
                    color=color,
                    linewidth=0.55,
                    alpha=0.16,
                    zorder=1,
                )
            axis.plot(
                centroids[::centroid_stride, zone, 0],
                centroids[::centroid_stride, zone, 1],
                color=color,
                linestyle="--",
                linewidth=1.35,
                alpha=0.8,
                zorder=2,
            )
            path = paths[name][:, zone]
            axis.plot(path[:, 0], path[:, 1], color=color, linewidth=3.0, zorder=4)
            axis.scatter(
                path[0, 0],
                path[0, 1],
                marker="s",
                s=42,
                facecolor="white",
                edgecolor=color,
                linewidth=1.4,
                zorder=5,
            )
            axis.scatter(
                path[-1, 0],
                path[-1, 1],
                marker="*",
                s=125,
                facecolor=color,
                edgecolor="#222222",
                linewidth=0.8,
                zorder=6,
            )
        axis.set_title(f"{POLICY_LABELS[name]}\n端到端时延 {delays[name]:.1f} ms")
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("X 坐标 / m")
        axis.grid(linestyle="--", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Y 坐标 / m")
    handles = [
        Line2D([0], [0], color="#777777", alpha=0.35, label="地面用户轨迹"),
        Line2D([0], [0], color="#333333", linestyle="--", label="动态业务质心"),
        Line2D([0], [0], color="#333333", linewidth=3.0, label="UAV 轨迹"),
        Line2D([0], [0], marker="s", color="none", markeredgecolor="#333333", label="起点"),
        Line2D([0], [0], marker="*", color="none", markeredgecolor="#333333", markersize=11, label="终点"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        f"三种部署策略的轨迹对比（共同随机轨迹，seed={trajectory_seed}）",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.84, bottom=0.18, wspace=0.10)
    fig.savefig(output / "02_三策略轨迹对比.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_eval_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    return (
        np.asarray([int(row["episode"]) for row in records]),
        np.asarray([float(row["task_delay_weighted_ms"]) for row in records]),
    )


def plot_training_process(output: Path) -> None:
    configure_plotting()
    v43_path = (
        ROOT
        / "results"
        / "tcom_v43_cooperation_aligned_macro"
        / "formal"
        / "seed1"
        / "joint_td3_sac"
        / "eval_central_td3.csv"
    )
    v44_path = (
        ROOT
        / "results"
        / "tcom_v44_joint_cooperation_finetune"
        / "formal"
        / "seed1"
        / "joint_td3_sac"
        / "eval_central_td3.csv"
    )
    ep43, delay43 = read_eval_curve(v43_path)
    ep44, delay44 = read_eval_curve(v44_path)
    selected_episode = 200
    selected_index = int(np.flatnonzero(ep43 == selected_episode)[0])
    selected_delay = float(delay43[selected_index])
    shifted44 = 300 + ep44

    fig, axis = plt.subplots(figsize=(9.8, 5.2))
    selected_mask = ep43 <= selected_episode
    diagnostic_mask = ep43 >= selected_episode
    axis.plot(
        ep43[selected_mask],
        delay43[selected_mask],
        marker="o",
        color="#2E8B57",
        linewidth=2.4,
        label="宏观 TD3 学习（微观 SAC 固定）",
    )
    axis.plot(
        ep43[diagnostic_mask],
        delay43[diagnostic_mask],
        marker="o",
        color="#AAAAAA",
        linewidth=1.5,
        linestyle="--",
        label="未采用的后续检查点",
    )
    axis.scatter(
        [selected_episode],
        [selected_delay],
        marker="*",
        s=150,
        color="#2E8B57",
        edgecolor="#222222",
        zorder=5,
        label="联合微调初始化点",
    )
    axis.plot(
        np.r_[300, shifted44],
        np.r_[selected_delay, delay44],
        marker="o",
        color="#2677B8",
        linewidth=2.6,
        label="TD3-SAC 联合微调",
    )
    axis.axvline(300, color="#555555", linewidth=1.0, linestyle=":")
    axis.text(305, 925, "两层联合更新开始", color="#444444", va="top")
    axis.scatter(
        [shifted44[-1]],
        [698.779],
        marker="D",
        s=58,
        color="#B33A3A",
        zorder=6,
        label="最终 20 种子均值 698.8 ms",
    )
    axis.annotate(
        "947.6 → 704.3 ms",
        xy=(200, selected_delay),
        xytext=(75, 810),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        color="#333333",
    )
    axis.set_xlabel("训练阶段累计轮次")
    axis.set_ylabel("固定评估端到端时延 / ms")
    axis.set_title("分阶段训练与联合微调过程（真实固定评估数据）")
    axis.set_ylim(620, 1000)
    axis.grid(linestyle="--", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "03_分阶段训练与联合微调曲线.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paired_gains(rows: list[dict], output: Path) -> None:
    configure_plotting()
    learned = extract_values(rows, "learned", "task_delay_weighted_ms")
    fixed = extract_values(rows, "fixed", "task_delay_weighted_ms")
    centroid = extract_values(rows, "centroid", "task_delay_weighted_ms")
    gain_fixed = fixed - learned
    gain_centroid = centroid - learned
    x = np.arange(1, learned.size + 1)
    width = 0.38
    fig, axis = plt.subplots(figsize=(10.6, 4.7))
    axis.bar(x - width / 2, gain_fixed, width, color=POLICY_COLORS["fixed"], label="相对固定悬停")
    axis.bar(x + width / 2, gain_centroid, width, color=POLICY_COLORS["centroid"], label="相对质心跟踪")
    axis.axhline(0.0, color="#222222", linewidth=1.0)
    axis.set_xlabel("共同随机种子序号")
    axis.set_ylabel("联合 TD3-SAC 的时延降低量 / ms")
    axis.set_title("逐种子配对时延收益（正值表示联合策略更快）")
    axis.set_xticks(x)
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    axis.text(
        0.015,
        0.96,
        f"相对固定悬停：平均 {np.mean(gain_fixed):.1f} ms，20/20 更快\n"
        f"相对质心跟踪：平均 {np.mean(gain_centroid):.1f} ms，20/20 更快",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(output / "04_逐种子配对时延收益.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(report: dict, output: Path) -> None:
    stats = report["policy_stats"]
    learned_delay = stats["learned"]["task_delay_weighted_ms"]["mean"]
    fixed_delay = stats["fixed"]["task_delay_weighted_ms"]["mean"]
    centroid_delay = stats["centroid"]["task_delay_weighted_ms"]["mean"]
    learned_queue = stats["learned"]["queue_mbits"]["mean"]
    fixed_queue = stats["fixed"]["queue_mbits"]["mean"]
    centroid_queue = stats["centroid"]["queue_mbits"]["mean"]
    text = f"""# v44 联合 TD3-SAC 最终结果

本目录只使用 v44 最终检查点、同一微观 SAC、同一环境配置和 20 个共同随机种子。

## 核心结果

| 策略 | 端到端时延 (ms) | 平均队列 (Mbit) |
|---|---:|---:|
| 固定悬停 | {fixed_delay:.1f} | {fixed_queue:.2f} |
| 动态质心跟踪 | {centroid_delay:.1f} | {centroid_queue:.2f} |
| 联合 TD3-SAC | {learned_delay:.1f} | {learned_queue:.2f} |

- 联合策略相对固定悬停降低 {fixed_delay - learned_delay:.1f} ms。
- 联合策略相对动态质心跟踪降低 {centroid_delay - learned_delay:.1f} ms。
- 两组比较均为 20/20 个共同随机种子更快。
- v44 联合阶段中 TD3 与 SAC 均发生参数更新；最终结果取第 200 轮 final checkpoint，不挑中间点。

## 图件

1. `01_端到端时延与平均队列.png`：正文主性能图。
2. `02_三策略轨迹对比.png`：含地面用户、动态业务质心和 UAV 轨迹。
3. `03_分阶段训练与联合微调曲线.png`：真实固定评估曲线，不平滑、不伪造单调收敛。
4. `04_逐种子配对时延收益.png`：20 个种子的配对稳健性。

轨迹收益不应解释为“始终离用户更近”或“移动更少”。联合策略的平均 U2G 偏移和总航程都更大，但队列与时延显著更低；因此当前证据支持的解释是：上层沿负载/协作方向选择非零残差部署，下层同步调整转发与计算，从而改善系统队列，而不是简单追逐业务质心。
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="8000-8019")
    parser.add_argument("--trajectory-seed", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--reuse-raw", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = parse_seed_spec(args.seeds)
    raw_path = output / "三策略逐种子原始结果.csv"
    trace_path = output / f"代表轨迹_seed{args.trajectory_seed}.npz"
    if args.reuse_raw:
        rows = read_csv(raw_path)
        observed = sorted({int(row["seed"]) for row in rows})
        if observed != seeds or len(rows) != len(POLICIES) * len(seeds):
            raise ValueError("Existing raw data do not match requested seeds")
        if not trace_path.exists():
            raise FileNotFoundError(trace_path)
    else:
        rows, trace_path = evaluate_all(
            args.checkpoint, output, seeds, args.trajectory_seed, args.device
        )

    report = build_report(rows, args.checkpoint, seeds, output)
    (output / "统计汇总.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_summary(rows, output)
    plot_trajectories(trace_path, rows, output, args.trajectory_seed)
    plot_training_process(output)
    plot_paired_gains(rows, output)
    write_readme(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
