"""Chinese figures for the fixed-budget convergence watch; no training/evaluation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/convergence_watch1600to1800_seed1"
BLUE, RED, GRAY = "#1976a3", "#c35440", "#70777d"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def style(ax):
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def build_figures(run):
    run = Path(run).resolve()
    result = read_json(run / "comparison.json")
    start, end = result["protocol"]["start_global"], result["protocol"]["end_global"]
    points = result["validation_points"]
    out = run / "figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    x = np.asarray([p["global_window"] for p in points])
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8), layout="constrained")
    for ax, key, title in zip(axes.flat[:3], ("task_delay_weighted_ms", "J", "task_delay_low_ms"),
                              ("加权完成时延 / ms（越低越好）", "软目标 J（越低越好）", "低优先级完成时延 / ms")):
        for policy, label, color in (("full", "PPO + Direct SAC", BLUE),
                                      ("centroid", "同一 SAC + 动态质心", RED)):
            values = np.asarray([[row[key] for row in p[policy]] for p in points])
            for seed_values in values.T:
                ax.plot(x, seed_values, color=color, alpha=.23, linewidth=.9)
            average = values.mean(axis=1)
            ax.plot(x, average, "o-" if policy == "full" else "s--", color=color, label=label, linewidth=2)
        ax.set(title=title, xlabel="总训练窗口（含 200 轮微观预训练）", xticks=x)
        style(ax)
    ax = axes[1, 1]
    delta = np.asarray([[row["task_delay_weighted_ms"] for row in p["macro_deltas"]] for p in points])
    for values in delta.T:
        ax.plot(x, values, color=BLUE, alpha=.25, linewidth=1)
    ax.plot(x, delta.mean(axis=1), "o-", color=BLUE, linewidth=2)
    ax.axhline(0, color=GRAY, linestyle="--")
    ax.set(title="PPO 宏观效应：完整方法减同一 SAC 质心 / ms\n低于零才是宏观收益", xlabel="总训练窗口", xticks=x)
    style(ax)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("固定场景的真实检查点曲线，不是训练奖励\n1 个训练初始化；2 个开发验证场景各 900 s；细线逐场景、粗线均值，无平滑、无置信区间", fontsize=12)
    fig.savefig(out / "01_fixed_validation.png", dpi=165)
    plt.close(fig)

    rows = read_rows(run / f"at{end}/training.csv")
    assert len(rows) == result["protocol"]["additional_windows"]
    x_train = np.asarray([int(r["global_window"]) for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.5), layout="constrained")
    for ax, key, title in zip(axes.flat,
                              ("window_reward", "window_J", "critic_loss", "alpha"),
                              ("训练奖励（越高越好）", "训练窗口软目标 J（越低越好）", "Critic 第一次拟合损失", "SAC 熵温度")):
        y = np.asarray([float(r[key]) for r in rows])
        ax.plot(x_train, y, color=BLUE, linewidth=.8, alpha=.28)
        ax.plot(x_train[19:], np.convolve(y, np.ones(20)/20, mode="valid"), color=BLUE, linewidth=2)
        ax.set(title=title, xlabel="总训练窗口")
        style(ax)
    axes[1, 0].set_yscale("log")
    fig.suptitle(f"{start} 至 {end} 轮训练记录\n细线为原始记录，粗线为 20 窗口后向均值；训练含探索与场景变化，不据平滑线单独判收敛", fontsize=12)
    fig.savefig(out / "02_training_trace.png", dpi=165)
    plt.close(fig)

    seeds = result["protocol"]["config"]["test_seeds"]
    fig, axes = plt.subplots(2, len(seeds), figsize=(max(6, 3.1*len(seeds)), 7.5), squeeze=False, layout="constrained")
    for col, seed in enumerate(seeds):
        for policy, label, color in (("final", "PPO + Direct SAC", BLUE),
                                      ("centroid_same_sac", "同一 SAC + 质心", RED),
                                      ("uniform_test", "质心 + 固定均分（非 RL）", GRAY)):
            rows = read_rows(run / f"at{end}/{policy}_{seed}_queues.csv")
            t = [float(r["time_s"]) for r in rows]
            axes[0, col].plot(t, [float(r["queue_total_mbit"]) for r in rows], label=label, color=color, alpha=.8, linewidth=1)
            axes[1, col].plot(t, [float(r["task_alive_low_count"]) for r in rows], color=color, alpha=.8, linewidth=1)
        axes[0, col].set(title=f"开发场景 {seed}", ylabel="总积压 / Mbit", xlabel="物理时间 / s")
        axes[1, col].set(ylabel="未完成低优先级任务 / 个", xlabel="物理时间 / s")
        style(axes[0, col])
        style(axes[1, col])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{end} 轮固定策略的连续 900 s 队列与未完成任务\n原始曲线、不取跨种子平均；每 10 s 宏观末采样，有限时间读数不构成长期稳定性证明", fontsize=12)
    fig.savefig(out / "03_queue_and_unfinished.png", dpi=165)
    plt.close(fig)
    print(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    assert read_json(args.run / "status.json")["status"] == "COMPLETE"
    build_figures(args.run)


if __name__ == "__main__":
    main()
