"""Audit the fixed-budget micro experiment; never select a checkpoint."""

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prefix_check(reference, current):
    if len(current) < len(reference):
        return {"equal": False, "reason": "Incomplete prefix"}
    for index, (left, right) in enumerate(zip(reference, current)):
        if left != right:
            return {"equal": False, "row": index,
                    "different_fields": sorted(k for k in left.keys() | right.keys()
                                               if left.get(k) != right.get(k))}
    return {"equal": True, "rows": len(reference)}


def budget_difference_only(old, new):
    restored = copy.deepcopy(new)
    restored["training"]["pretrain_windows"] = old["training"]["pretrain_windows"]
    restored["qualification"]["max_stage_seconds"] = old["qualification"]["max_stage_seconds"]
    return restored == old


def workload(root, report):
    per_seed = []
    for row in report["final"]:
        rows = read_csv(root / f"{report['final_label']}_{row['seed']}_uavs.csv")
        duration = max(float(r["time_s"]) for r in rows)
        per_seed.append([{
            "assigned_mbps": sum(float(r["assigned_destination_mbit"]) for r in rows if int(r["uav"]) == uav) / duration,
            "computed_mbps": sum(float(r["computed_mbit"]) for r in rows if int(r["uav"]) == uav) / duration,
            "terminal_compute_mbit": next(float(r["queue_compute_mbit"]) for r in reversed(rows) if int(r["uav"]) == uav),
        } for uav in range(3)])
    return [{key: float(np.mean([rows[uav][key] for rows in per_seed]))
             for key in per_seed[0][uav]} for uav in range(3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=ROOT / "results/split_mdp_actor_lr3e5_seed1")
    parser.add_argument("--run", type=Path, default=ROOT / "results/split_mdp_actor_lr3e5_budget300_seed1")
    args = parser.parse_args()
    roots = [args.reference, args.run]
    manifests = [read_json(root / "manifest.json") for root in roots]
    reports = [read_json(root / "report.json") for root in roots]
    budgets = [m["config"]["training"]["pretrain_windows"] for m in manifests]
    assert 0 < budgets[0] < budgets[1], budgets
    training = [read_csv(root / "training.csv") for root in roots]
    validation = [read_csv(root / "validation.csv") for root in roots]
    prefix = {"training": prefix_check(*training), "validation": prefix_check(*validation)}
    assert budget_difference_only(*(m["config"] for m in manifests))
    assert all(item["equal"] for item in prefix.values()), prefix
    assert reports[1]["source_hashes_unchanged"]
    assert not reports[1]["joint_training_started"]
    assert reports[1]["stage_audits"][0]["ppo_updates"] == 0
    assert all(report["stage_audits"][0]["completed_windows"] == budget
               for report, budget in zip(reports, budgets))
    assert manifests[1]["old_checkpoint"] is None and not manifests[1]["teacher"]
    crn = []
    for seed in manifests[0]["config"]["test_seeds"]:
        with np.load(roots[0] / f"{reports[0]['final_label']}_{seed}_trajectory.npz") as left, \
             np.load(roots[1] / f"{reports[1]['final_label']}_{seed}_trajectory.npz") as right:
            equal = all(np.array_equal(left[key], right[key]) for key in ("uav", "users"))
        arrivals = [[r["task_arrived_count"] for r in read_csv(root / f"{report['final_label']}_{seed}_queues.csv")]
                    for root, report in zip(roots, reports)]
        assert equal and arrivals[0] == arrivals[1]
        crn.append({"seed": seed, "users_uavs_arrival_counts_equal": True})
    work = [workload(root, report) for root, report in zip(roots, reports)]
    paired = []
    for row in reports[1]["final"]:
        old = next(r for r in reports[0]["final"] if r["seed"] == row["seed"])
        paired.append({"seed": row["seed"], **{
            key: row[key] - old[key] for key in ("task_delay_weighted_ms", "task_completion_rate", "queue_total_mbit", "J")}})
    tail = [row for row in validation[1] if int(row["window"]) >= budgets[1] - 60]
    result = {
        "scope": "One training seed; reused development evaluation scenes. Not independent confirmatory evidence.",
        "config_changes": {"pretrain_windows": budgets, "max_stage_seconds": [
            m["config"]["qualification"]["max_stage_seconds"] for m in manifests]},
        "prefix": prefix, "crn": crn,
        "old_mean": reports[0]["mean"], "new_mean": reports[1]["mean"],
        "uniform_mean": reports[1]["uniform_mean"], "paired_new_minus_old": paired,
        "old_workload_by_fixed_uav": work[0], "new_workload_by_fixed_uav": work[1],
        "qualification_passed": reports[1]["pretrain_admission"]["passed"],
        "qualification": reports[1]["pretrain_admission"],
        "stage_audits": reports[1]["stage_audits"],
        "tail_validation_range_by_seed": {str(seed): {
            key: [min(float(r[key]) for r in tail if int(r["seed"]) == seed),
                  max(float(r[key]) for r in tail if int(r["seed"]) == seed)]
            for key in ("task_delay_weighted_ms", "task_completion_rate", "queue_total_mbit")}
            for seed in manifests[1]["config"]["validation_seeds"]},
        "interpretation_limit": "Fixed final weights only; finite queue traces do not prove infinite-horizon stability or optimality.",
    }
    (args.run / "budget_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(item.name == font for item in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False
    target = args.run / "figures"
    target.mkdir(exist_ok=True)
    colors = ("#b95338", "#14758c")
    labels = tuple(f"训练至{budget}窗口" for budget in budgets)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for index, (rows, val, color, label) in enumerate(zip(training, validation, colors, labels)):
        axes[0, 0].plot([int(r["window"]) for r in rows], [float(r["window_reward"]) for r in rows],
                        color=color, label=label, alpha=.8)
        windows = sorted({int(r["window"]) for r in val})
        for ax, key, scale in ((axes[0, 1], "task_delay_weighted_ms", .001),
                               (axes[1, 0], "task_completion_rate", 100),
                               (axes[1, 1], "queue_total_mbit", 1)):
            values = [scale * np.mean([float(r[key]) for r in val if int(r["window"]) == window]) for window in windows]
            ax.plot(windows, values, "o-", color=color, label=label, markersize=3, alpha=.8)
    settings = (("训练原始奖励（未平滑）", "窗口平均奖励，越高越好"),
                ("固定验证：已完成任务加权时延", "时延 / 秒"),
                ("固定验证：任务完成率", "完成率 / %"),
                ("固定验证：900秒期末积压", "积压 / Mbit"))
    for ax, (title, ylabel) in zip(axes.flat, settings):
        ax.set(title=title, xlabel="训练窗口（每窗口300秒仿真）", ylabel=ylabel)
        ax.axvline(budgets[0], color="#777777", linestyle=":", linewidth=1)
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=9)
    axes[1, 0].set_ylim(0, 102)
    fig.suptitle(f"纯 SAC 训练预算对照：{budgets[0]} → {budgets[1]}窗口\n"
                 f"同种子、同奖励、同学习率；前{budgets[0]}窗口逐项复现；未启动PPO", fontsize=14)
    fig.savefig(target / "01_training_budget.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    variants = [(roots[i], reports[i]["final_label"], reports[i]["mean"], colors[i], labels[i]) for i in range(2)]
    variants.append((roots[1], "uniform_test", reports[1]["uniform_mean"], "#53763f", "均分规则（仅诊断）"))
    for root, label, mean, color, title in variants:
        curves = [read_csv(root / f"{label}_{seed}_queues.csv") for seed in manifests[1]["config"]["test_seeds"]]
        values = np.array([[float(row["queue_total_mbit"]) for row in curve] for curve in curves])
        times = [float(row["time_s"]) for row in curves[0]]
        axes[0].plot(times, values.mean(0), color=color, label=title)
        axes[0].fill_between(times, values.min(0), values.max(0), color=color, alpha=.12)
    x = np.arange(3)
    for i, (_, _, mean, color, label) in enumerate(variants):
        values = [mean[key]/1000 for key in ("task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms")]
        bars = axes[1].bar(x + (i-1)*.25, values, width=.24, color=color, label=label)
        axes[1].bar_label(bars, fmt="%.2f", fontsize=8)
    axes[1].set_xticks(x, ["加权", "高优", "低优"])
    for i, (label, color) in enumerate(zip(labels, colors)):
        bars = axes[2].bar(x + (i-.5)*.34, [u["assigned_mbps"] for u in work[i]], width=.32, color=color, label=label)
        axes[2].bar_label(bars, fmt="%.2f", fontsize=8,
                          bbox={"facecolor": "white", "edgecolor": "none", "pad": .4})
    axes[2].axhline(9, color="#555555", linestyle="--", label="名义周期平均处理容量9 Mbps")
    axes[2].set_xticks(x, ["UAV 1", "UAV 2", "UAV 3"])
    axes[0].set(title="末次权重：连续900秒队列", xlabel="连续运行 / 秒", ylabel="总积压 / Mbit", ylim=(0, None))
    axes[1].set(title="完成时延：必须与完成率一起读", ylabel="时延 / 秒", ylim=(0, None))
    axes[2].set(title="末次策略：分给各计算节点的任务量", ylabel="按总运行时长平均 / Mbps", ylim=(0, 10.5))
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("固定末次权重，3个配对评估场景；队列阴影是场景范围，不是置信区间", fontsize=13)
    fig.savefig(target / "02_final_budget_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    tail_windows = sorted({int(row["window"]) for row in tail})
    for seed in manifests[1]["config"]["validation_seeds"]:
        values = [float(next(row["task_delay_weighted_ms"] for row in tail
                             if int(row["seed"]) == seed and int(row["window"]) == window))
                  for window in tail_windows]
        axes[0].plot(tail_windows, values, "o-", markersize=4, label=f"验证场景 {seed}")
    baseline_delay = np.mean([float(row["task_delay_weighted_ms"])
                              for row in read_csv(args.run / "baseline_validation.csv")])
    axes[0].axhline(baseline_delay, color="#53763f", linestyle="--", label="均分诊断：验证场景均值")
    for root, label, mean, color, title in variants[1:]:
        curves = [read_csv(root / f"{label}_{seed}_queues.csv") for seed in manifests[1]["config"]["test_seeds"]]
        values = np.array([[float(row["queue_total_mbit"]) for row in curve] for curve in curves])
        times = [float(row["time_s"]) for row in curves[0]]
        axes[1].plot(times, values.mean(0), color=color, label=title)
        axes[1].fill_between(times, values.min(0), values.max(0), color=color, alpha=.12)
    tail_delays = [float(row["task_delay_weighted_ms"]) for row in tail] + [baseline_delay]
    padding = max(20, .1 * np.ptp(tail_delays))
    axes[0].set(title="最后60窗口：逐验证场景时延", xlabel="训练窗口", ylabel="完成任务加权时延 / 毫秒",
                ylim=(max(0, min(tail_delays) - padding), max(tail_delays) + padding))
    axes[1].set(title=f"{budgets[1]}窗口末次权重：队列放大图", xlabel="连续运行 / 秒", ylabel="全部积压 / Mbit", ylim=(0, None))
    for ax in axes:
        ax.grid(alpha=.15)
        ax.legend(fontsize=9)
    fig.suptitle("局部放大：不能把大坐标轴下的平坦外观当成稳定；阴影为3场景范围", fontsize=12)
    fig.savefig(target / "03_tail_and_queue_detail.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
