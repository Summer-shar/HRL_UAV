"""Evaluate v46 common-micro TD3/PPO runs with training-seed hierarchy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import evaluate_v45_fair_screen as v45


ROOT = Path(__file__).resolve().parents[1]
RESULTS_VERSION = "tcom_v46_common_micro_joint"
DEFAULT_OUTPUT = ROOT / "result_archive" / "v46_common_micro_joint"
METHODS = (
    "td3_sac",
    "ppo_sac",
    "td3_fixed",
    "td3_centroid",
    "ppo_fixed",
    "ppo_centroid",
)
LABELS = {
    "td3_sac": "TD3-SAC",
    "ppo_sac": "PPO-SAC",
    "td3_fixed": "TD3微观+固定站位",
    "td3_centroid": "TD3微观+动态质心",
    "ppo_fixed": "PPO微观+固定站位",
    "ppo_centroid": "PPO微观+动态质心",
}
COLORS = {
    "td3_sac": "#2878B5",
    "ppo_sac": "#E07B39",
    "td3_fixed": "#777777",
    "td3_centroid": "#2E8B57",
    "ppo_fixed": "#A0A0A0",
    "ppo_centroid": "#6DAA7B",
}
FIELDS = (
    "task_delay_weighted_ms",
    "task_delay_high_ms",
    "task_delay_low_ms",
    "queue_total_mbits",
    "average_power_w",
    "soft_objective_j",
    "coop_ratio",
    "u2g_distance_m",
)


def checkpoint(train_seed: int, method: str) -> Path:
    return (
        ROOT / "results" / RESULTS_VERSION / "formal" / f"seed{train_seed}"
        / method / "checkpoints" / "final.pt"
    )


def make_policies(train_seed: int, device: str) -> dict:
    td3 = v45.load_method_policy(
        "td3_sac", checkpoint(train_seed, "joint_td3_sac"), device
    )
    ppo = v45.load_method_policy(
        "ppo_sac", checkpoint(train_seed, "joint_ppo_sac"), device
    )
    if v45.scenario_signature(td3["args"]) != v45.scenario_signature(ppo["args"]):
        raise RuntimeError(f"Scenario mismatch for training seed {train_seed}")
    centroid = v45.make_centroid_offset_policy(np.zeros((3, 2)))
    return {
        "td3_sac": td3,
        "ppo_sac": ppo,
        "td3_fixed": {
            **td3,
            "name": "td3_fixed",
            "macro_policy": v45.V34MacroPolicy("fixed", 0.0, 0.0),
        },
        "td3_centroid": {**td3, "name": "td3_centroid", "macro_policy": centroid},
        "ppo_fixed": {
            **ppo,
            "name": "ppo_fixed",
            "macro_policy": v45.V34MacroPolicy("fixed", 0.0, 0.0),
        },
        "ppo_centroid": {**ppo, "name": "ppo_centroid", "macro_policy": centroid},
    }


def mean_by_training_seed(rows: list[dict], method: str, field: str) -> dict[int, float]:
    values = {}
    for train_seed in sorted({int(row["train_seed"]) for row in rows}):
        sample = [
            float(row[field]) for row in rows
            if int(row["train_seed"]) == train_seed and row["method"] == method
        ]
        values[train_seed] = float(np.mean(sample))
    return values


def hierarchical_summary(rows: list[dict]) -> dict:
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for field in FIELDS:
            per_train_seed = mean_by_training_seed(rows, method, field)
            summary[method][field] = {
                **v45.mean_ci95(list(per_train_seed.values())),
                "per_training_seed": per_train_seed,
            }
    return summary


def paired_hierarchical(rows: list[dict], left: str, right: str, field: str) -> dict:
    per_training_seed = {}
    for train_seed in sorted({int(row["train_seed"]) for row in rows}):
        left_values = {
            int(row["seed"]): float(row[field]) for row in rows
            if int(row["train_seed"]) == train_seed and row["method"] == left
        }
        right_values = {
            int(row["seed"]): float(row[field]) for row in rows
            if int(row["train_seed"]) == train_seed and row["method"] == right
        }
        seeds = sorted(set(left_values) & set(right_values))
        per_training_seed[train_seed] = float(np.mean([
            left_values[seed] - right_values[seed] for seed in seeds
        ]))
    return {
        **v45.mean_ci95(list(per_training_seed.values())),
        "per_training_seed": per_training_seed,
    }


def build_report(rows: list[dict]) -> dict:
    comparisons = {
        "td3_minus_ppo": paired_hierarchical(
            rows, "td3_sac", "ppo_sac", "task_delay_weighted_ms"
        ),
        "td3_minus_own_fixed": paired_hierarchical(
            rows, "td3_sac", "td3_fixed", "task_delay_weighted_ms"
        ),
        "td3_minus_own_centroid": paired_hierarchical(
            rows, "td3_sac", "td3_centroid", "task_delay_weighted_ms"
        ),
        "ppo_minus_own_fixed": paired_hierarchical(
            rows, "ppo_sac", "ppo_fixed", "task_delay_weighted_ms"
        ),
        "ppo_minus_own_centroid": paired_hierarchical(
            rows, "ppo_sac", "ppo_centroid", "task_delay_weighted_ms"
        ),
    }
    return {"summaries": hierarchical_summary(rows), "paired_delay_ms": comparisons}


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: dict, output: Path) -> None:
    lines = [
        "# v46 公共微观预训练后的联合训练评估",
        "",
        "> 统计单位为训练种子；每个训练种子内部使用20个共同环境种子。",
        "> 全部方法固定使用最终checkpoint，不挑best。",
        "",
        "| 方法 | 端到端时延/ms | 训练种子95% CI | 队列/Mbit | 功率/W |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        stats = report["summaries"][method]
        delay = stats["task_delay_weighted_ms"]
        lines.append(
            f"| {LABELS[method]} | {delay['mean']:.2f} | "
            f"[{delay['ci95_low']:.2f}, {delay['ci95_high']:.2f}] | "
            f"{stats['queue_total_mbits']['mean']:.2f} | "
            f"{stats['average_power_w']['mean']:.2f} |"
        )
    lines.extend(["", "## 训练种子分层配对差值", ""])
    for name, stats in report["paired_delay_ms"].items():
        lines.append(
            f"- `{name}`：{stats['mean']:+.2f} ms，95% CI "
            f"[{stats['ci95_low']:+.2f}, {stats['ci95_high']:+.2f}]，"
            f"逐训练种子={stats['per_training_seed']}。"
        )
    td3_vs_ppo = report["paired_delay_ms"]["td3_minus_ppo"]
    td3_vs_centroid = report["paired_delay_ms"]["td3_minus_own_centroid"]
    ppo_vs_centroid = report["paired_delay_ms"]["ppo_minus_own_centroid"]
    td3_seed_gains = [-float(value) for value in td3_vs_centroid["per_training_seed"].values()]
    lines.extend([
        "",
        "## 当前判决",
        "",
        f"- TD3-SAC 相对 PPO-SAC 平均快 {-td3_vs_ppo['mean']:.2f} ms，三个训练种子方向一致；"
        f"但训练种子层级 95% CI [{td3_vs_ppo['ci95_low']:+.2f}, "
        f"{td3_vs_ppo['ci95_high']:+.2f}] ms 仍跨 0，因此只能判为当前更优候选，不能宣称显著胜出。",
        f"- TD3 学习轨迹相对同一微观策略的动态质心平均快 {-td3_vs_centroid['mean']:.2f} ms；"
        f"逐训练种子净收益为 {[round(value, 2) for value in td3_seed_gains]} ms，"
        "收益主要由一个训练种子贡献，尚未形成稳定的轨迹优势。",
        f"- PPO 学习轨迹相对自身动态质心的差值为 {ppo_vs_centroid['mean']:+.2f} ms，"
        "可视为宏观策略基本退化成质心跟踪。",
        "- 结论：保留 TD3-SAC 作为后续主方法候选；先修复连续联合更新造成的微观策略遗忘，"
        "并重新验证轨迹净收益，再开展 CPU 与任务负载矩阵。当前结果不支持直接启动大规模对比实验。",
    ])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(report: dict, output: Path) -> None:
    v45.configure_plotting()
    methods = METHODS
    y = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0), sharey=True)
    panels = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms", "(a) 端到端时延"),
        ("queue_total_mbits", "终端队列总量 / Mbit", "(b) 队列积压"),
    )
    for axis, (field, xlabel, title) in zip(axes, panels):
        stats = [report["summaries"][method][field] for method in methods]
        means = np.asarray([item["mean"] for item in stats])
        lower = np.asarray([item["mean"] - item["ci95_low"] for item in stats])
        upper = np.asarray([item["ci95_high"] - item["mean"] for item in stats])
        for index, method in enumerate(methods):
            axis.errorbar(
                means[index], y[index],
                xerr=np.asarray([[lower[index]], [upper[index]]]),
                fmt="o", markersize=8, capsize=4, linewidth=1.8,
                color=COLORS[method],
            )
            axis.annotate(
                f"{means[index]:.1f}", (means[index], y[index]),
                xytext=(7, 0), textcoords="offset points", va="center", fontsize=9,
            )
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(axis="x", linestyle="--", alpha=0.25)
    axes[0].set_yticks(y, [LABELS[method] for method in methods])
    axes[0].invert_yaxis()
    fig.suptitle("公共SAC起点下的三训练种子结果")
    fig.tight_layout()
    fig.savefig(output / "01_三训练种子方法对比.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_macro_gain(report: dict, output: Path) -> None:
    v45.configure_plotting()
    train_seeds = (1, 42, 123)
    td3_delta = report["paired_delay_ms"]["td3_minus_own_centroid"]["per_training_seed"]
    ppo_delta = report["paired_delay_ms"]["ppo_minus_own_centroid"]["per_training_seed"]
    td3_gain = np.asarray([-float(td3_delta[str(seed)] if str(seed) in td3_delta else td3_delta[seed]) for seed in train_seeds])
    ppo_gain = np.asarray([-float(ppo_delta[str(seed)] if str(seed) in ppo_delta else ppo_delta[seed]) for seed in train_seeds])
    x = np.arange(len(train_seeds))
    width = 0.34
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    td3_bars = axis.bar(x - width / 2, td3_gain, width, color=COLORS["td3_sac"], label="TD3-SAC")
    ppo_bars = axis.bar(x + width / 2, ppo_gain, width, color=COLORS["ppo_sac"], label="PPO-SAC")
    axis.bar_label(td3_bars, labels=[f"{value:+.2f}" for value in td3_gain], padding=3)
    axis.bar_label(ppo_bars, labels=[f"{value:+.2f}" for value in ppo_gain], padding=3)
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set_xticks(x, [f"训练种子 {seed}" for seed in train_seeds])
    axis.set_ylabel("相对同微观动态质心的时延改善 / ms")
    axis.set_title("学习轨迹的逐训练种子净收益（正值表示更快）")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "05_轨迹相对质心净收益.png", dpi=300, facecolor="white")
    plt.close(fig)


def read_curve(path: Path, offset: int) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([offset + int(float(row["episode"])) for row in rows]),
        np.asarray([float(row["task_delay_weighted_ms"]) for row in rows]),
    )


def aggregate_curves(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = sorted(set.intersection(*(set(x.tolist()) for x, _ in curves)))
    values = np.asarray([
        [float(y[np.where(x == episode)[0][0]]) for episode in common]
        for x, y in curves
    ])
    return np.asarray(common), values.mean(axis=0), values.std(axis=0, ddof=1)


def plot_convergence(train_seeds: list[int], output: Path) -> None:
    v45.configure_plotting()
    pretrain_curves = []
    method_curves = {"td3_sac": [], "ppo_sac": []}
    for seed in train_seeds:
        base = ROOT / "results" / RESULTS_VERSION / "formal" / f"seed{seed}"
        pretrain_curves.append(read_curve(base / "micro_pretrain/eval_central_macro.csv", 0))
        method_curves["td3_sac"].append(read_curve(base / "joint_td3_sac/eval_central_td3.csv", 300))
        method_curves["ppo_sac"].append(read_curve(base / "joint_ppo_sac/eval_central_macro.csv", 300))
    all_curves = {"pretrain": aggregate_curves(pretrain_curves)}
    all_curves.update({method: aggregate_curves(curves) for method, curves in method_curves.items()})
    fig, (upper, lower) = plt.subplots(
        2, 1, figsize=(10.8, 6.8), sharex=True,
        gridspec_kw={"height_ratios": (0.8, 1.5), "hspace": 0.05},
    )
    for axis in (upper, lower):
        x, mean, std = all_curves["pretrain"]
        axis.plot(x, mean, color="#555555", linewidth=2.2, marker="o", label="公共SAC预训练")
        axis.fill_between(x, mean - std, mean + std, color="#777777", alpha=0.14)
        for method in ("td3_sac", "ppo_sac"):
            x, mean, std = all_curves[method]
            axis.plot(x, mean, color=COLORS[method], linewidth=2.2, marker="o", label=LABELS[method])
            axis.fill_between(x, mean - std, mean + std, color=COLORS[method], alpha=0.14)
        axis.axvline(300, color="#333333", linestyle="--", linewidth=1.1)
        axis.grid(linestyle="--", alpha=0.25)
    upper.set_ylim(1200, 4550)
    lower.set_ylim(620, 1100)
    upper.spines["bottom"].set_visible(False)
    lower.spines["top"].set_visible(False)
    upper.tick_params(labeltop=False, bottom=False)
    lower.xaxis.tick_bottom()
    gap = 0.008
    kwargs = dict(color="#333333", clip_on=False, linewidth=1.1)
    upper.plot((-gap, +gap), (-gap, +gap), transform=upper.transAxes, **kwargs)
    upper.plot((1 - gap, 1 + gap), (-gap, +gap), transform=upper.transAxes, **kwargs)
    lower.plot((-gap, +gap), (1 - gap, 1 + gap), transform=lower.transAxes, **kwargs)
    lower.plot((1 - gap, 1 + gap), (1 - gap, 1 + gap), transform=lower.transAxes, **kwargs)
    lower.text(307, 1065, "联合训练开始", va="top")
    upper.legend(loc="upper right")
    lower.set_xlabel("累计训练轮次")
    fig.supylabel("固定评估端到端时延 / ms")
    fig.suptitle("SAC预训练到TD3/PPO联合训练的连续收敛曲线")
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.10, top=0.90, hspace=0.06)
    fig.savefig(output / "02_连续联合收敛曲线.png", dpi=300, facecolor="white")
    plt.close(fig)


def save_trajectory_figures(policies: dict, output: Path, eval_seed: int) -> None:
    v45.configure_plotting()
    selected = ("td3_sac", "ppo_sac", "td3_centroid")
    traces = {}
    for method in selected:
        result = v45.evaluate_one(policies[method], eval_seed, record_trajectory=True)
        traces[method] = (result, result.pop("trajectory"))
    fig, axes = plt.subplots(1, 3, figsize=(14.3, 6.0), sharex=True, sharey=True)
    uav_colors = ("#2878B5", "#E67E22", "#2E8B57")
    for axis, method in zip(axes, selected):
        result, trace = traces[method]
        for user in range(trace["users"].shape[1]):
            zone = user % 3
            axis.plot(
                trace["users"][:, user, 0], trace["users"][:, user, 1],
                color=uav_colors[zone], alpha=0.25, linewidth=0.75,
            )
        for index, color in enumerate(uav_colors):
            axis.plot(
                trace["centroids"][:, index, 0], trace["centroids"][:, index, 1],
                color=color, linestyle="--", linewidth=1.25, alpha=0.78,
            )
            axis.plot(trace["uavs"][:, index, 0], trace["uavs"][:, index, 1], color=color, linewidth=2.7)
            axis.scatter(*trace["uavs"][0, index], marker="s", s=35, facecolor="white", edgecolor=color, zorder=4)
            axis.scatter(*trace["uavs"][-1, index], marker="*", s=90, facecolor=color, edgecolor="#222222", zorder=5)
        axis.set_title(f"{LABELS[method]}\n时延 {result['task_delay_weighted_ms']:.1f} ms")
        axis.set_xlabel("X坐标 / m")
        axis.grid(linestyle="--", alpha=0.22)
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Y坐标 / m")
    fig.suptitle(f"最终策略代表轨迹（训练种子1，评估种子{eval_seed}）", fontsize=16, y=0.98)
    fig.text(
        0.5, 0.91, "细线：地面用户轨迹；虚线：动态业务质心；粗实线：UAV轨迹",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.79))
    fig.savefig(output / "03_最终策略轨迹.png", dpi=300, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.8), sharex=True, sharey=True)
    for axis, method in zip(axes, selected):
        result, trace = traces[method]
        residual = trace["uavs"] - trace["centroids"]
        for index, color in enumerate(uav_colors):
            axis.plot(
                residual[:, index, 0], residual[:, index, 1],
                color=color, linewidth=2.0, label=f"UAV{index + 1}",
            )
            axis.scatter(
                residual[0, index, 0], residual[0, index, 1], marker="s",
                s=34, facecolor="white", edgecolor=color, zorder=4,
            )
            axis.scatter(
                residual[-1, index, 0], residual[-1, index, 1], marker="*",
                s=86, facecolor=color, edgecolor="#222222", zorder=5,
            )
        axis.axhline(0.0, color="#999999", linewidth=0.8)
        axis.axvline(0.0, color="#999999", linewidth=0.8)
        mean_offset = float(np.linalg.norm(residual, axis=2).mean())
        axis.set_title(
            f"{LABELS[method]}\n平均偏移 {mean_offset:.1f} m，时延 {result['task_delay_weighted_ms']:.1f} ms"
        )
        axis.set_xlabel("相对质心X偏移 / m")
        axis.grid(linestyle="--", alpha=0.22)
        axis.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("相对质心Y偏移 / m")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"最终策略相对业务质心的轨迹残差（评估种子{eval_seed}）")
    fig.tight_layout()
    fig.savefig(output / "04_相对质心轨迹残差.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seeds", default="1,42,123")
    parser.add_argument("--eval-seeds", default="8000-8019")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-only", action="store_true", help="Reuse report.json and training CSVs without rerunning evaluation")
    args = parser.parse_args()
    train_seeds = v45.parse_seed_spec(args.train_seeds)
    eval_seeds = v45.parse_seed_spec(args.eval_seeds)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        write_report(report, output)
        plot_summary(report, output)
        plot_convergence(train_seeds, output)
        plot_macro_gain(report, output)
        print(f"Plots refreshed in {output}")
        return

    rows = []
    representative = None
    for train_seed in train_seeds:
        policies = make_policies(train_seed, args.device)
        if train_seed == train_seeds[0]:
            representative = policies
        for eval_seed in eval_seeds:
            for method in METHODS:
                row = v45.evaluate_one(policies[method], eval_seed)
                row["train_seed"] = train_seed
                rows.append(row)
                print(
                    f"train_seed={train_seed} eval_seed={eval_seed} method={method} "
                    f"delay={row['task_delay_weighted_ms']:.2f}ms",
                    flush=True,
                )
    write_rows(output / "per_seed_raw.csv", rows)
    report = build_report(rows)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report, output)
    plot_summary(report, output)
    plot_convergence(train_seeds, output)
    save_trajectory_figures(representative, output, eval_seeds[0])
    plot_macro_gain(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
