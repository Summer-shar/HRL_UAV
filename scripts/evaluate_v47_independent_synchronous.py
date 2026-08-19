"""Evaluate v47 independently trained synchronous HRL methods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import evaluate_v45_fair_screen as v45
from train_centralized_macro import fixed_macro_policy, zero_macro_policy


ROOT = Path(__file__).resolve().parents[1]
RESULTS_VERSION = "tcom_v47_independent_synchronous"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v47_independent_synchronous"
METHODS = (
    "joint_td3_sac",
    "joint_ppo_sac",
    "cppo_sca",
    "no_coop",
    "centroid_sac",
    "fixed_sac",
)
LABELS = {
    "joint_td3_sac": "TD3-SAC（本文）",
    "joint_ppo_sac": "PPO-SAC",
    "cppo_sca": "PPO-SCA",
    "no_coop": "无协作",
    "centroid_sac": "动态质心-SAC",
    "fixed_sac": "固定站位-SAC",
}
COLORS = {
    "joint_td3_sac": "#2878B5",
    "joint_ppo_sac": "#E07B39",
    "cppo_sca": "#7A65A8",
    "no_coop": "#D95F59",
    "centroid_sac": "#2E8B57",
    "fixed_sac": "#666666",
}
EVAL_FILENAMES = {
    "joint_td3_sac": "eval_central_td3.csv",
    "joint_ppo_sac": "eval_central_macro.csv",
    "cppo_sca": "eval_central_sca.csv",
    "no_coop": "eval_central_td3.csv",
    "centroid_sac": "eval_central_macro.csv",
    "fixed_sac": "eval_central_macro.csv",
}
TRAIN_FILENAMES = {
    "joint_td3_sac": "train_td3.csv",
    "joint_ppo_sac": "train_centralized.csv",
    "cppo_sca": "train_centralized.csv",
    "no_coop": "train_td3.csv",
    "centroid_sac": "train_centralized.csv",
    "fixed_sac": "train_centralized.csv",
}
SAC_METHODS = (
    "joint_td3_sac",
    "joint_ppo_sac",
    "centroid_sac",
    "fixed_sac",
)
METRICS = (
    "task_delay_weighted_ms",
    "task_delay_high_ms",
    "task_delay_low_ms",
    "queue_total_mbits",
    "queue_high_mbits",
    "queue_low_mbits",
    "average_power_w",
    "soft_objective_j",
    "coop_ratio",
    "u2g_distance_m",
)


def checkpoint(stage: str, train_seed: int, method: str) -> Path:
    return (
        ROOT
        / "results"
        / RESULTS_VERSION
        / stage
        / f"seed{train_seed}"
        / method
        / "checkpoints"
        / "final.pt"
    )


def method_policy(stage: str, train_seed: int, method: str, device: str) -> dict:
    policy = v45.load_method_policy(
        method,
        checkpoint(stage, train_seed, method),
        device,
    )
    if method == "centroid_sac":
        policy["macro_policy"] = zero_macro_policy
    elif method == "fixed_sac":
        policy["macro_policy"] = fixed_macro_policy
    return policy


def load_policies(stage: str, train_seed: int, device: str) -> dict[str, dict]:
    policies = {
        method: method_policy(stage, train_seed, method, device)
        for method in METHODS
    }
    reference = v45.scenario_signature(policies[METHODS[0]]["args"])
    for method in METHODS[1:]:
        signature = v45.scenario_signature(policies[method]["args"])
        if signature != reference:
            raise RuntimeError(
                f"Scenario mismatch for train seed {train_seed}: {method}"
            )
    return policies


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def per_training_seed(rows: list[dict], method: str, field: str) -> dict[int, float]:
    output = {}
    seeds = sorted({int(row["train_seed"]) for row in rows})
    for train_seed in seeds:
        values = [
            float(row[field])
            for row in rows
            if row["method"] == method and int(row["train_seed"]) == train_seed
        ]
        output[train_seed] = float(np.mean(values))
    return output


def hierarchical_summary(rows: list[dict]) -> dict:
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for field in METRICS:
            values = per_training_seed(rows, method, field)
            summary[method][field] = {
                **v45.mean_ci95(list(values.values())),
                "per_training_seed": values,
            }
    return summary


def paired_difference(rows: list[dict], left: str, right: str, field: str) -> dict:
    deltas = {}
    for train_seed in sorted({int(row["train_seed"]) for row in rows}):
        left_values = {
            int(row["seed"]): float(row[field])
            for row in rows
            if row["method"] == left and int(row["train_seed"]) == train_seed
        }
        right_values = {
            int(row["seed"]): float(row[field])
            for row in rows
            if row["method"] == right and int(row["train_seed"]) == train_seed
        }
        seeds = sorted(set(left_values) & set(right_values))
        deltas[train_seed] = float(
            np.mean([left_values[seed] - right_values[seed] for seed in seeds])
        )
    return {**v45.mean_ci95(list(deltas.values())), "per_training_seed": deltas}


def build_report(rows: list[dict]) -> dict:
    summary = hierarchical_summary(rows)
    comparisons = {
        f"{method}_minus_joint_td3_sac": paired_difference(
            rows,
            method,
            "joint_td3_sac",
            "task_delay_weighted_ms",
        )
        for method in METHODS
        if method != "joint_td3_sac"
    }
    ranking = sorted(
        METHODS,
        key=lambda method: summary[method]["task_delay_weighted_ms"]["mean"],
    )
    return {
        "summaries": summary,
        "paired_baseline_minus_td3_delay_ms": comparisons,
        "delay_ranking": ranking,
        "statistics_unit": "training_seed",
    }


def configure_plotting() -> None:
    v45.configure_plotting()
    plt.rcParams.update({"axes.titleweight": "bold"})


def plot_method_summary(report: dict, output: Path) -> None:
    configure_plotting()
    x = np.arange(len(METHODS))
    panels = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms", "(a) 端到端时延"),
        ("queue_total_mbits", "平均终端队列 / Mbit", "(b) 队列积压"),
        ("average_power_w", "平均功率 / W", "(c) 功率账本"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    for axis, (field, ylabel, title) in zip(axes, panels):
        stats = [report["summaries"][method][field] for method in METHODS]
        means = np.asarray([item["mean"] for item in stats])
        errors = np.asarray([item["ci95_high"] - item["mean"] for item in stats])
        bars = axis.bar(
            x,
            means,
            yerr=errors,
            capsize=4,
            width=0.68,
            color=[COLORS[method] for method in METHODS],
        )
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in means], padding=3)
        axis.set_xticks(x, [LABELS[method] for method in METHODS], rotation=24, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.24)
    fig.suptitle("独立训练后的六种方法对比")
    fig.tight_layout()
    fig.savefig(output / "01_方法时延队列功率对比.png", dpi=300, facecolor="white")
    plt.close(fig)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def eval_curve(path: Path, offset: int) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(path)
    x = np.asarray(
        [int(float(row["episode"])) + int(offset) for row in rows],
        dtype=np.int64,
    )
    y = np.asarray([float(row["task_delay_weighted_ms"]) for row in rows])
    return x, y


def train_curve(path: Path, offset: int) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv_rows(path)
    x = np.arange(1, len(rows) + 1, dtype=np.int64) + int(offset)
    y = np.asarray([float(row["J_train"]) for row in rows])
    return x, y


def complete_curve(
    stage: str,
    train_seed: int,
    method: str,
    curve_reader,
) -> tuple[np.ndarray, np.ndarray]:
    base = ROOT / "results" / RESULTS_VERSION / stage / f"seed{train_seed}"
    fallback_pretrain = base / "micro_pretrain_joint_td3_sac" / "train_centralized.csv"
    pretrain_horizon = len(read_csv_rows(fallback_pretrain))
    parts = []
    if method in SAC_METHODS:
        pretrain = base / f"micro_pretrain_{method}"
        filename = (
            "eval_central_macro.csv"
            if curve_reader is eval_curve
            else "train_centralized.csv"
        )
        parts.append(curve_reader(pretrain / filename, 0))
    method_offset = int(pretrain_horizon)
    filename = EVAL_FILENAMES[method] if curve_reader is eval_curve else TRAIN_FILENAMES[method]
    parts.append(curve_reader(base / method / filename, method_offset))
    return (
        np.concatenate([part[0] for part in parts]),
        np.concatenate([part[1] for part in parts]),
    )


def aggregate_curves(curves: list[tuple[np.ndarray, np.ndarray]]):
    common = sorted(set.intersection(*(set(x.tolist()) for x, _ in curves)))
    values = np.asarray([
        [float(y[np.where(x == episode)[0][0]]) for episode in common]
        for x, y in curves
    ])
    std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(len(common))
    return np.asarray(common), values.mean(axis=0), std


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    output = np.empty_like(values, dtype=np.float64)
    for index in range(values.size):
        start = max(0, index - window + 1)
        output[index] = float(np.mean(values[start : index + 1]))
    return output


def plot_convergence(stage: str, train_seeds: list[int], output: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharey=True)
    groups = (METHODS[:3], METHODS[3:])
    for axis, methods in zip(axes, groups):
        for method in methods:
            curves = [
                complete_curve(stage, seed, method, eval_curve)
                for seed in train_seeds
            ]
            x, mean, std = aggregate_curves(curves)
            axis.plot(x, mean, color=COLORS[method], linewidth=2.1, marker="o", label=LABELS[method])
            axis.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.13)
        axis.axvline(200, color="#333333", linestyle="--", linewidth=1.1)
        axis.grid(linestyle="--", alpha=0.24)
        axis.set_xlabel("累计训练轮次")
        axis.legend(fontsize=9)
    axes[0].set_ylabel("固定评估端到端时延 / ms")
    axes[0].set_title("(a) 学习型与优化型方法")
    axes[1].set_title("(b) 消融与部署基线")
    axes[0].text(
        208,
        0.96,
        "联训开始",
        transform=axes[0].get_xaxis_transform(),
        va="top",
        ha="left",
    )
    fig.suptitle("各方法独立训练的连续收敛曲线（阴影为训练种子标准差）")
    fig.tight_layout()
    fig.savefig(output / "02_独立训练连续收敛曲线.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_training_objective(stage: str, train_seeds: list[int], output: Path) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharey=True)
    groups = (METHODS[:3], METHODS[3:])
    for axis, methods in zip(axes, groups):
        for method in methods:
            curves = []
            for seed in train_seeds:
                x, y = complete_curve(stage, seed, method, train_curve)
                curves.append((x, rolling_mean(y, 25)))
            x, mean, std = aggregate_curves(curves)
            axis.plot(x, mean, color=COLORS[method], linewidth=2.0, label=LABELS[method])
            axis.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.13)
        axis.axvline(200, color="#333333", linestyle="--", linewidth=1.1)
        axis.grid(linestyle="--", alpha=0.24)
        axis.set_xlabel("累计训练轮次")
        axis.legend(fontsize=9)
    axes[0].set_ylabel("逐轮真实训练目标 J（25轮滑动平均）")
    axes[0].set_title("(a) 学习型与优化型方法")
    axes[1].set_title("(b) 消融与部署基线")
    fig.suptitle("训练过程中实际产生的目标值")
    fig.tight_layout()
    fig.savefig(output / "03_逐轮真实训练目标.png", dpi=300, facecolor="white")
    plt.close(fig)


def trajectory_signature(trace: dict) -> str:
    array = np.asarray(trace["uavs"], dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def save_trajectories(policies: dict, output: Path, eval_seed: int) -> None:
    configure_plotting()
    traces = {}
    signatures = {}
    for method in METHODS:
        result = v45.evaluate_one(policies[method], eval_seed, record_trajectory=True)
        trace = result.pop("trajectory")
        traces[method] = (result, trace)
        signatures[method] = trajectory_signature(trace)
    duplicates = []
    for left_index, left in enumerate(METHODS):
        for right in METHODS[left_index + 1 :]:
            if signatures[left] == signatures[right]:
                duplicates.append([left, right])
    (output / "trajectory_uniqueness.json").write_text(
        json.dumps(
            {"eval_seed": eval_seed, "signatures": signatures, "duplicates": duplicates},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    learned_pairs = {
        frozenset(("joint_td3_sac", "joint_ppo_sac")),
        frozenset(("joint_td3_sac", "cppo_sca")),
        frozenset(("joint_ppo_sac", "cppo_sca")),
    }
    if any(frozenset(pair) in learned_pairs for pair in duplicates):
        raise RuntimeError(f"Learned methods produced identical UAV trajectories: {duplicates}")

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 10.0), sharex=True, sharey=True)
    zone_colors = ("#2878B5", "#E67E22", "#2E8B57")
    for axis, method in zip(axes.flat, METHODS):
        result, trace = traces[method]
        users = trace["users"]
        allocations = np.asarray(trace["user_allocations"], dtype=np.int64)
        for user in range(users.shape[1]):
            zone = int(allocations[user])
            axis.plot(
                users[:, user, 0],
                users[:, user, 1],
                color=zone_colors[zone],
                alpha=0.32,
                linewidth=0.8,
            )
        for index, color in enumerate(zone_colors):
            axis.plot(trace["centroids"][:, index, 0], trace["centroids"][:, index, 1], color=color, linestyle="--", linewidth=1.15, alpha=0.8)
            axis.plot(trace["uavs"][:, index, 0], trace["uavs"][:, index, 1], color=color, linewidth=2.6)
            axis.scatter(*trace["uavs"][0, index], marker="s", s=34, facecolor="white", edgecolor=color, zorder=4)
            axis.scatter(*trace["uavs"][-1, index], marker="*", s=86, facecolor=color, edgecolor="#222222", zorder=5)
        axis.set_title(f"{LABELS[method]}\n时延 {result['task_delay_weighted_ms']:.1f} ms")
        axis.set_xlabel("X坐标 / m")
        axis.set_ylabel("Y坐标 / m")
        axis.grid(linestyle="--", alpha=0.2)
        axis.set_aspect("equal", adjustable="box")
    fig.suptitle("最终策略轨迹：细线为用户，虚线为业务质心，粗线为UAV")
    fig.tight_layout()
    fig.savefig(output / "04_六种方法最终轨迹.png", dpi=300, facecolor="white")
    plt.close(fig)


def write_report(report: dict, output: Path) -> None:
    lines = [
        f"# {RESULTS_VERSION} 独立预训练与同步联训结果",
        "",
        "> 每种方法独立初始化；SAC方法先各自预训练200轮，再与上层同步训练800轮。",
        "> 统计单位是训练种子；环境种子只用于每个训练种子内部的共同随机数配对。",
        "",
        "| 排名 | 方法 | 端到端时延/ms | 训练种子95% CI | 队列/Mbit | 功率/W |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, method in enumerate(report["delay_ranking"], start=1):
        stats = report["summaries"][method]
        delay = stats["task_delay_weighted_ms"]
        lines.append(
            f"| {rank} | {LABELS[method]} | {delay['mean']:.2f} | "
            f"[{delay['ci95_low']:.2f}, {delay['ci95_high']:.2f}] | "
            f"{stats['queue_total_mbits']['mean']:.2f} | "
            f"{stats['average_power_w']['mean']:.2f} |"
        )
    lines.extend(["", "## 相对本文方法的训练种子层级配对差", ""])
    for name, stats in report["paired_baseline_minus_td3_delay_ms"].items():
        lines.append(
            f"- `{name}`：{stats['mean']:+.2f} ms，95% CI "
            f"[{stats['ci95_low']:+.2f}, {stats['ci95_high']:+.2f}]，"
            f"逐训练种子={stats['per_training_seed']}。"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global RESULTS_VERSION, DEFAULT_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "diagnostic", "formal"),
        default="formal",
    )
    parser.add_argument("--results-version", default=RESULTS_VERSION)
    parser.add_argument("--main-label", default=LABELS["joint_td3_sac"])
    parser.add_argument("--train-seeds", default="1,7,42,99,123")
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    RESULTS_VERSION = str(args.results_version)
    LABELS["joint_td3_sac"] = str(args.main_label)
    if args.output is None:
        DEFAULT_OUTPUT = ROOT / "result_archive" / RESULTS_VERSION
        args.output = DEFAULT_OUTPUT
    train_seeds = v45.parse_seed_spec(args.train_seeds)
    eval_seeds = v45.parse_seed_spec(args.eval_seeds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        plot_method_summary(report, output)
        plot_convergence(args.stage, train_seeds, output)
        plot_training_objective(args.stage, train_seeds, output)
        write_report(report, output)
        return

    rows = []
    representative = None
    for train_seed in train_seeds:
        policies = load_policies(args.stage, train_seed, args.device)
        if representative is None:
            representative = policies
        for eval_seed in eval_seeds:
            for method in METHODS:
                row = v45.evaluate_one(policies[method], eval_seed)
                row["train_seed"] = train_seed
                rows.append(row)
                print(
                    f"train={train_seed} eval={eval_seed} {method}: "
                    f"delay={row['task_delay_weighted_ms']:.2f} ms",
                    flush=True,
                )
    write_csv(output / "per_seed_raw.csv", rows)
    report = build_report(rows)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(report, output)
    plot_method_summary(report, output)
    plot_convergence(args.stage, train_seeds, output)
    plot_training_objective(args.stage, train_seeds, output)
    save_trajectories(representative, output, eval_seeds[0])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
