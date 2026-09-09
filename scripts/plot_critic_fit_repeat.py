"""Plot the completed paired experiment without additional evaluation or training."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/critic_fit_repeat_1200to1400_seed1"
COLORS = ("#7c8188", "#1976a3", "#c35440", "#34834c", "#947135")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    result = read_json(RUN / "comparison.json")
    assert read_json(RUN / "status.json")["status"] == "COMPLETE"
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    out = RUN / "figures"
    out.mkdir(exist_ok=True)
    baseline = read_json(Path(result["protocol"]["source"]).parent / "report.json")
    reports = {name: read_json(RUN / name / "report.json") for name in ("control", "two_fits")}
    series = [baseline["final"], reports["control"]["final"], reports["two_fits"]["final"],
              reports["two_fits"]["matched_centroid"], reports["two_fits"]["uniform"]]
    labels = ["1200 轮 PPO + SAC", "1400 轮 PPO + SAC\n原更新", "1400 轮 PPO + SAC\nCritic 两次拟合",
              "1400 轮同一 SAC + 质心\n两次拟合，零训练重放", "质心 + 固定均分\n非 RL 诊断"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), layout="constrained", sharey=True)
    for ax, key, title, fmt in zip(axes, ("task_delay_weighted_ms", "J"),
                                  ("加权完成时延 / ms（越低越好）", "软目标 J（越低越好）"),
                                  (".2f", ".5f")):
        all_values = np.asarray([[row[key] for row in rows] for rows in series])
        span = float(np.ptp(all_values))
        for i, (values, color) in enumerate(zip(all_values, COLORS)):
            ax.scatter(values, i + np.linspace(-.11, .11, len(values)), color=color, alpha=.45, s=23)
            mean = float(values.mean())
            ax.scatter([mean], [i], color=color, marker="D", s=45, zorder=3)
            ax.annotate(format(mean, fmt), (mean, i), xytext=(0, 13), textcoords="offset points",
                        ha="center", color=color, fontsize=11)
        ax.set(xlabel=title, yticks=range(len(labels)), yticklabels=labels,
               xlim=(all_values.min() - .12 * span, all_values.max() + .12 * span), ylim=(4.5, -.55))
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("同一起点，各续训 200 轮：最终结果\n1 个训练初始化；3 个开发场景各连续 900 s；圆点为逐场景，菱形为均值，不是置信区间", fontsize=12)
    fig.savefig(out / "01_final_comparison.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5), layout="constrained")
    panels = [("window_reward", "训练奖励（越高越好）", "平均宏观奖励"),
              ("critic_loss", "Critic 第一次拟合前的损失", "双 Q 均方误差，对数坐标"),
              ("window_delay_ms", "训练窗口内完成任务的加权时延", "时延 / ms"),
              ("queue_total_mbit", "训练窗口末的未完成积压", "总队列 / Mbit")]
    for name, label, color in (("control", "原更新", COLORS[1]), ("two_fits", "两次拟合", COLORS[2])):
        with (RUN / name / "training.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 200
        x = np.arange(1, len(rows) + 1)
        for ax, (key, title, unit) in zip(axes.flat, panels):
            y = np.asarray([float(row[key]) for row in rows])
            assert np.isfinite(y).all()
            ax.plot(x, y, color=color, alpha=.22, linewidth=.7)
            ax.plot(x[19:], np.convolve(y, np.ones(20)/20, mode="valid"), color=color, label=label)
            ax.set(title=title, xlabel="本次续训窗口（1 至 200）", ylabel=unit)
            ax.grid(alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 1].set_yscale("log")
    axes[0, 0].legend(fontsize=10)
    fig.suptitle("本次续训记录：细线为全部原始值，粗线为 20 窗口后向均值\n训练含随机动作和场景切换；不能与确定性末次评估混作一条收敛曲线", fontsize=12)
    fig.savefig(out / "02_training_trace.png", dpi=170)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
