"""Readable Chinese plots for the matched SAC/TQC pilot and its continuations."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"sac": "#1878a4", "tqc": "#ca5541"}
LABELS = {"sac": "PPO + Direct SAC", "tqc": "PPO + Direct TQC"}


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def history_rows(run, result, name, filename):
    rows = []
    for folder in result["protocol"].get("history_runs", []) + [str(run)]:
        path = Path(folder) / name / filename
        if path.exists():
            rows.extend(read_rows(path))
    if filename == "training.csv":
        windows = [int(r["global_window"]) for r in rows]
        if not windows or windows != sorted(set(windows)):
            raise ValueError("Training histories must contain unique increasing global windows")
    return rows


def decorate(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.18)


def build_figures(run):
    run = Path(run)
    result = json.loads((run / "comparison.json").read_text(encoding="utf-8"))
    notice = "工程冒烟测试：仅核验程序，不作性能结论\n" if result["protocol"]["smoke"] else ""
    output = run / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    pretrain = result["protocol"]["configs"]["sac"]["training"]["pretrain_windows"]
    continuation = result["protocol"].get("continuation")
    schedule = (f"两臂从各自 {continuation['start_global']} 窗口终点续到 {continuation['end_global']}，完整恢复两层训练状态"
                if continuation else "两臂各自从头训练：微观预训练后开启 PPO，微观继续更新")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), layout="constrained")
    for name in COLORS:
        rows = history_rows(run, result, name, "training.csv")
        x = np.asarray([int(r["global_window"]) for r in rows])
        for ax, key, title in zip(axes.flat,
                ("window_reward", "window_J", "window_delay_ms", "window_low_ms"),
                ("训练奖励（越高越好）", "训练软目标 J（越低越好）", "训练窗口加权完成时延 / ms", "训练窗口低优先级完成时延 / ms")):
            values = np.asarray([float(r[key]) for r in rows])
            ax.plot(x, values, color=COLORS[name], alpha=.22, linewidth=.6)
            width = min(20, len(x))
            ax.plot(x[width-1:], np.convolve(values, np.ones(width)/width, mode="valid"),
                    color=COLORS[name], label=LABELS[name], linewidth=1.6)
            ax.set(title=title, xlabel="总训练窗口")
    for ax in axes.flat:
        ax.axvline(pretrain, linestyle="--", color="#777777", linewidth=1)
        if continuation:
            ax.axvline(continuation["start_global"], linestyle=":", color="#555555", linewidth=1)
        decorate(ax)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(notice + schedule + "\n细线为原始值，粗线为最多 20 窗口后向均值；含探索，不据平滑线单独判断收敛", fontsize=12)
    fig.savefig(output / "01_training.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for name in COLORS:
        points = []
        raw = history_rows(run, result, name, "validation.csv")
        for window in sorted({int(r["window"]) for r in raw if r["stage"] == "pretrain"}):
            group = [r for r in raw if r["stage"] == "pretrain" and int(r["window"]) == window]
            points.append((window, {k: np.mean([float(r[k]) for r in group]) for k in
                                     ("task_delay_weighted_ms", "J", "task_delay_low_ms")}))
        points.extend((p["global_window"], p["mean"]) for p in result["curves"][name])
        for ax, key, title in zip(axes, ("task_delay_weighted_ms", "J", "task_delay_low_ms"),
                                  ("固定场景加权完成时延 / ms", "固定场景软目标 J", "固定场景低优先级时延 / ms")):
            ax.plot([p[0] for p in points], [p[1][key] for p in points], "o-", label=LABELS[name], color=COLORS[name])
            ax.set(title=title, xlabel="总训练窗口")
    for ax in axes:
        ax.axvline(pretrain, linestyle="--", color="#777777", linewidth=1)
        if continuation:
            ax.axvline(continuation["start_global"], linestyle=":", color="#555555", linewidth=1)
        decorate(ax)
    axes[0].legend(fontsize=9)
    fig.suptitle(notice + "固定开发场景的确定性评估：原始检查点均值、无平滑\n联训期间只保存权重，训练结束后回放；不按这条曲线选择最终权重", fontsize=12)
    fig.savefig(output / "02_fixed_validation.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), layout="constrained")
    for ax, key, title in zip(axes.flat, ("task_delay_weighted_ms", "task_delay_high_ms", "task_delay_low_ms", "J"),
                              ("加权完成时延 / ms", "高优先级完成时延 / ms", "低优先级完成时延 / ms", "软目标 J")):
        paired = result["tqc_minus_sac"]
        for pair in paired:
            values = [next(r[key] for r in result["arms"][n]["final"] if r["seed"] == pair["seed"]) for n in COLORS]
            ax.plot([0, 1], values, "o-", color="#969b9e", alpha=.5, linewidth=.9, markersize=4)
        for x, name in enumerate(COLORS):
            ax.scatter(x, result["arms"][name]["mean"][key], color=COLORS[name], s=85, zorder=4)
        difference = result["paired_summary"][key]
        ci = difference["paired_t95"]
        label = f"TQC − SAC = {difference['mean']:+.4f}" if key == "J" else f"TQC − SAC = {difference['mean']:+.2f} ms"
        if ci is not None:
            label += f"\n配对 95% 区间 [{ci[0]:.4f}, {ci[1]:.4f}]" if key == "J" else f"\n配对 95% 区间 [{ci[0]:.2f}, {ci[1]:.2f}] ms"
        ax.set(title=f"{title}（越低越好）\n{label}", xticks=[0, 1], xticklabels=["PPO + SAC", "PPO + TQC"], xlim=(-.4, 1.4))
        decorate(ax)
    fig.suptitle(notice + "最终权重的同场景配对比较：灰线为逐场景，彩点为均值\n仅 1 个训练初始化；区间只描述评估场景变异，不代表多训练种子稳健性", fontsize=12)
    fig.savefig(output / "03_final_comparison.png", dpi=160)
    plt.close(fig)

    seeds = [r["seed"] for r in result["arms"]["sac"]["final"]]
    fig, axes = plt.subplots(2, len(seeds), figsize=(max(6, 3.1*len(seeds)), 6.8), squeeze=False, layout="constrained")
    for col, seed in enumerate(seeds):
        for name in COLORS:
            rows = read_rows(run / name / f"final_{seed}_queues.csv")
            times = [float(r["time_s"]) for r in rows]
            axes[0, col].plot(times, [float(r["queue_total_mbit"]) for r in rows], color=COLORS[name], label=LABELS[name], linewidth=1)
            axes[1, col].plot(times, [float(r["task_alive_low_count"]) for r in rows], color=COLORS[name], linewidth=1)
        axes[0, col].set(title=f"场景 {seed}", xlabel="物理时间 / s", ylabel="总积压 / Mbit")
        axes[1, col].set(xlabel="物理时间 / s", ylabel="未完成低优先级任务 / 个")
        decorate(axes[0, col])
        decorate(axes[1, col])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(notice + "最终策略的连续队列与未完成任务，逐场景原始曲线\n每 10 s 宏观末采样；有限时域队列不爆炸不等于无限时域稳定性证明", fontsize=12)
    fig.savefig(output / "04_queues.png", dpi=160)
    plt.close(fig)

    title = "SAC 与 TQC 等预算续训对比" if continuation else "SAC 与 TQC 从头训练对比"
    origin = ("原始实验两臂从头训练；本次仅从各自最终权重续训，不混用权重，不重跑预训练。无教师、无 SCA 锚定，两层继续按各自频率共同更新。"
              if continuation else "两臂均无教师、无 SCA 锚定、无旧权重；微观预训练后，两层按各自频率共同更新。")
    lines = [f"# {title}", "", f"状态：{result['status']}。本次耗时 {result['minutes']:.1f} 分钟。", "",
             origin, "", schedule, "",
             "| 方法 | 加权时延/ms | 高优/ms | 低优/ms | J | 完成率 | 期末队列/Mbit |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in COLORS:
        m = result["arms"][name]["mean"]
        lines.append(f"| {LABELS[name]} | {m['task_delay_weighted_ms']:.3f} | {m['task_delay_high_ms']:.3f} | {m['task_delay_low_ms']:.3f} | {m['J']:.6f} | {m['task_completion_rate']:.6f} | {m['queue_total_mbit']:.3f} |")
    lines.extend(["", "差值均定义为 TQC 减 SAC；时延和 J 的负值表示 TQC 较好。完成率则越高越好。", "",
                  "## 上层贡献", "", "以下是各自最终微观策略不变时，PPO 相对动态质心的配对效应，不是独立重训比较。"])
    for name in COLORS:
        d = result["macro_effect"][name]["task_delay_weighted_ms"]
        lines.append(f"- {LABELS[name]}：PPO 减质心 {d['mean']:+.3f} ms；配对区间 {d['paired_t95']}。")
    if continuation:
        lines.extend(["", "## 续训增量", "", "下列为新终点减各自旧终点，同场景配对；时延和 J 的负值表示改善。"])
        for name in COLORS:
            d = result["endpoint_change"][name]["task_delay_weighted_ms"]
            j = result["endpoint_change"][name]["J"]
            lines.append(f"- {LABELS[name]}：加权时延 {d['mean']:+.3f} ms，区间 {d['paired_t95']}；J {j['mean']:+.6f}，区间 {j['paired_t95']}。")
        lines.extend(["", "本次续训是在查看旧终点后授权的探索性延长。评估场景沿用旧场景，不称为全新留出集。曲线包含旧历史，点虚线标出本次续训起点。"])
    budget = result["protocol"]["configs"]["sac"]["training"]["joint_windows"] + pretrain
    lines.extend(["", "## 解释边界", "", "- 仅 1 个匹配训练初始化；5 个评估场景不等于 5 个训练种子。",
                  "- TQC 是既有算法在当前路由框架下的适配，不能将其名称或基础机制声称为本文原创。",
                  f"- 本次终点为总计 {budget} 窗口。曲线仍变化时，应报告未收敛，不能将有限预算阴性判作方法上界。",
                  "- 两臂从第一轮起都用两次 Critic 拟合，不直接对比历史中途更改拟合次数的 1800/2000 轮结果。",
                  "- SAC 的 MSE 与 TQC 的分位数 Huber 损失不是同一标度，不能用损失数值大小判优。",
                  "- 完成时延仅统计已完成任务，必须同时看完成率、未完成任务与队列图。", "",
                  "![训练曲线](figures/01_training.png)", "![固定验证](figures/02_fixed_validation.png)",
                  "![最终配对](figures/03_final_comparison.png)", "![队列](figures/04_queues.png)"])
    (run / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    build_figures(parser.parse_args().run)
