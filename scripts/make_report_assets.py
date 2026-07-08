import csv
from pathlib import Path

try:
    from ._bootstrap import bootstrap
except ImportError:
    from _bootstrap import bootstrap

ROOT = Path(bootstrap())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 13


def read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def style(axis):
    axis.grid(True, alpha=0.28)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def annotate_points(axis, x, y, fmt="{:.1f}", yoffset=6):
    for x_i, y_i in zip(x, y):
        axis.annotate(
            fmt.format(y_i),
            (x_i, y_i),
            textcoords="offset points",
            xytext=(0, yoffset),
            ha="center",
            fontsize=10,
        )


def rows_for(rows, *, cpu=None, load=None):
    selected = rows
    if cpu is not None:
        selected = [row for row in selected if abs(f(row, "cpu_ghz") - cpu) < 1e-9]
    if load is not None:
        selected = [row for row in selected if abs(f(row, "load_scale") - load) < 1e-9]
    return sorted(selected, key=lambda row: (f(row, "cpu_ghz"), f(row, "load_scale")))


def main():
    result_dir = ROOT / "results" / "robust_loadstate_prioritydemand_load08_12_p300"
    out_dir = ROOT / "results" / "report_ready"
    out_dir.mkdir(parents=True, exist_ok=True)

    cross_rows = read_rows(result_dir / "cross_cpu_load_eval_best.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.2), dpi=180)
    axes = axes.reshape(-1)

    loads = np.array([1.0, 1.1, 1.2, 1.3], dtype=float)
    demand = 21.0 * loads
    cpus = [8.0, 10.0, 12.0, 14.0]
    caps = [3.0 * cpu for cpu in cpus]
    colors = {
        8.0: "#d62728",
        10.0: "#2ca02c",
        12.0: "#1f77b4",
        14.0: "#9467bd",
    }

    ax = axes[0]
    ax.plot(loads, demand, marker="o", linewidth=2.4, color="#111111", label="Offered load")
    for cpu, cap in zip(cpus, caps):
        ax.hlines(
            cap,
            loads.min(),
            loads.max(),
            colors=colors[cpu],
            linestyles="--",
            linewidth=1.6,
            label=f"{cpu:g}GHz service",
        )
    ax.fill_between(
        loads,
        demand,
        caps[0],
        where=demand > caps[0],
        color="#d62728",
        alpha=0.14,
        label="8GHz overloaded",
    )
    ax.set_title("(a) Compute capacity explains non-convergence at 8GHz")
    ax.set_xlabel("Load scale")
    ax.set_ylabel("Mbits/s")
    ax.legend(frameon=False, fontsize=7.8, ncol=2, loc="upper left")
    style(ax)

    ax = axes[1]
    for cpu in cpus:
        cpu_rows = rows_for(cross_rows, cpu=cpu)
        x = np.array([f(row, "load_scale") for row in cpu_rows])
        y = np.array([f(row, "task_delay_low_ms") for row in cpu_rows])
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.1,
            color=colors[cpu],
            label=f"{cpu:g}GHz",
        )
    ax.set_yscale("log")
    ax.set_title("(b) Low-priority delay explodes only in overloaded 8GHz")
    ax.set_xlabel("Load scale")
    ax.set_ylabel("Low-priority E2E delay (ms, log scale)")
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    ax = axes[2]
    load_rows = rows_for(cross_rows, load=1.2)
    x = np.array([f(row, "cpu_ghz") for row in load_rows])
    high = np.array([f(row, "task_delay_high_ms") for row in load_rows])
    low = np.array([f(row, "task_delay_low_ms") for row in load_rows])
    ax.plot(x, high, marker="o", linewidth=2.2, label="High priority delay")
    ax.plot(x, low, marker="o", linewidth=2.2, label="Low priority delay")
    ax.set_yscale("log")
    ax.set_title("(c) Raising CPU restores low-priority service at load=1.2")
    ax.set_xlabel("CPU capacity per UAV (GHz)")
    ax.set_ylabel("E2E delay (ms, log scale)")
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    ax = axes[3]
    width = 0.25
    high_comp = np.array([f(row, "task_completion_high_rate") for row in load_rows])
    low_comp = np.array([f(row, "task_completion_low_rate") for row in load_rows])
    qos = np.array([f(row, "task_qos_violation_rate") for row in load_rows])
    idx = np.arange(len(x))
    ax.bar(idx - width, high_comp, width=width, label="High completion")
    ax.bar(idx, low_comp, width=width, label="Low completion")
    ax.bar(idx + width, qos, width=width, label="QoS violation")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{cpu:g}" for cpu in x])
    ax.set_ylim(0.0, 1.08)
    ax.set_title("(d) Priority protection and QoS at load=1.2")
    ax.set_xlabel("CPU capacity per UAV (GHz)")
    ax.set_ylabel("Rate")
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    fig.suptitle("UAV-MEC HRL Results: Capacity, Load, Priority and Cooperation", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "main_result_summary.png")
    plt.close(fig)

    eval_path = ROOT / "results" / "auto_prioritydemand_base_best300" / "eval_central_macro.csv"
    eval_rows = read_rows(eval_path)
    episodes = np.array([f(row, "episode") for row in eval_rows])
    j_values = np.array([f(row, "J") for row in eval_rows])
    low_delay = np.array([f(row, "task_delay_low_ms") for row in eval_rows])
    high_delay = np.array([f(row, "task_delay_high_ms") for row in eval_rows])
    best_idx = int(np.argmin(j_values))
    best_so_far = np.minimum.accumulate(j_values)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2), dpi=180)
    ax = axes[0]
    ax.plot(episodes, j_values, marker="o", color="0.66", label="Evaluation J")
    ax.plot(episodes, best_so_far, linewidth=2.4, color="#d62728", label="Best-so-far")
    ax.scatter(
        [episodes[best_idx]],
        [j_values[best_idx]],
        s=58,
        color="#d62728",
        zorder=5,
        label=f"Best ep {int(episodes[best_idx])}",
    )
    ax.set_title("(a) Feasible-load training convergence")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Objective J (lower is better)")
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    ax = axes[1]
    ax.plot(episodes, high_delay, marker="o", linewidth=2.1, label="High priority")
    ax.plot(episodes, low_delay, marker="o", linewidth=2.1, label="Low priority")
    ax.set_title("(b) Priority delay during convergence")
    ax.set_xlabel("Episode")
    ax.set_ylabel("E2E delay (ms)")
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    fig.suptitle("Training Evidence: Best Checkpoint Selection Under Feasible Load", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_evidence.png")
    plt.close(fig)

    summary_rows = []
    for cpu in cpus:
        row = rows_for(cross_rows, cpu=cpu, load=1.2)[0]
        summary_rows.append(
            {
                "cpu_ghz": f"{cpu:g}",
                "load_scale": "1.2",
                "J": f(row, "J"),
                "high_delay_ms": f(row, "task_delay_high_ms"),
                "low_delay_ms": f(row, "task_delay_low_ms"),
                "completion_rate": f(row, "task_completion_rate"),
                "qos_violation_rate": f(row, "task_qos_violation_rate"),
                "energy": f(row, "energy"),
            }
        )
    with (out_dir / "load12_cpu_summary.csv").open("w", newline="", encoding="utf-8") as fp:
        fields = [
            "cpu_ghz",
            "load_scale",
            "J",
            "high_delay_ms",
            "low_delay_ms",
            "completion_rate",
            "qos_violation_rate",
            "energy",
        ]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(out_dir / "main_result_summary.png")
    print(out_dir / "convergence_evidence.png")
    print(out_dir / "load12_cpu_summary.csv")

    # Chinese, presentation-ready figures with larger labels.
    fig, axes = plt.subplots(2, 2, figsize=(17.2, 12.4), dpi=180)
    axes = axes.reshape(-1)

    ax = axes[0]
    ax.plot(loads, demand, marker="o", linewidth=2.8, color="#111111", label="任务到达需求")
    for cpu, cap in zip(cpus, caps):
        ax.hlines(
            cap,
            loads.min(),
            loads.max(),
            colors=colors[cpu],
            linestyles="--",
            linewidth=2.0,
            label=f"{cpu:g}GHz 总计算服务能力",
        )
    ax.fill_between(
        loads,
        demand,
        caps[0],
        where=demand > caps[0],
        color="#d62728",
        alpha=0.14,
        label="8GHz 过载区域",
    )
    ax.scatter([1.2, 1.3], [25.2, 27.3], s=72, color="#d62728", zorder=5)
    ax.annotate(
        "load=1.2 时 25.2Mbps > 24Mbps",
        xy=(1.2, 25.2),
        xytext=(1.205, 29.2),
        arrowprops=dict(arrowstyle="->", lw=1.1),
        fontsize=12,
    )
    ax.set_title("（a）8GHz 高负载不收敛的容量原因", fontsize=16)
    ax.set_xlabel("负载系数", fontsize=13)
    ax.set_ylabel("任务量 / 服务能力（Mbps）", fontsize=13)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper left")
    style(ax)

    ax = axes[1]
    for cpu in cpus:
        cpu_rows = rows_for(cross_rows, cpu=cpu)
        x_load = np.array([f(row, "load_scale") for row in cpu_rows])
        y_low = np.array([f(row, "task_delay_low_ms") for row in cpu_rows])
        ax.plot(
            x_load,
            y_low,
            marker="o",
            linewidth=2.5,
            color=colors[cpu],
            label=f"{cpu:g}GHz",
        )
    ax.set_yscale("log")
    ax.set_title("（b）不同负载下低优先级端到端时延", fontsize=16)
    ax.set_xlabel("负载系数", fontsize=13)
    ax.set_ylabel("低优先级时延（ms，对数坐标）", fontsize=13)
    ax.legend(frameon=False, fontsize=11)
    style(ax)

    ax = axes[2]
    ax.plot(x, high, marker="o", linewidth=2.6, label="高优先级时延")
    ax.plot(x, low, marker="o", linewidth=2.6, label="低优先级时延")
    ax.set_yscale("log")
    annotate_points(ax, x, high, "{:.0f}", -22)
    annotate_points(ax, x, low, "{:.0f}", 8)
    ax.set_title("（c）负载系数=1.2 时提高 CPU 后低优先级恢复", fontsize=16)
    ax.set_xlabel("单架 UAV 计算能力（GHz）", fontsize=13)
    ax.set_ylabel("端到端时延（ms，对数坐标）", fontsize=13)
    ax.legend(frameon=False, fontsize=11)
    style(ax)

    ax = axes[3]
    width = 0.25
    ax.bar(idx - width, high_comp, width=width, label="高优先级完成率")
    ax.bar(idx, low_comp, width=width, label="低优先级完成率")
    ax.bar(idx + width, qos, width=width, label="QoS 违约率")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{cpu:g}" for cpu in x])
    ax.set_ylim(0.0, 1.08)
    ax.set_title("（d）负载系数=1.2 时任务完成率与 QoS", fontsize=16)
    ax.set_xlabel("单架 UAV 计算能力（GHz）", fontsize=13)
    ax.set_ylabel("比例", fontsize=13)
    ax.legend(frameon=False, fontsize=11)
    style(ax)

    fig.suptitle("UAV-MEC 分层强化学习实验结果：算力、负载与优先级保护", fontsize=20)
    fig.tight_layout()
    fig.savefig(out_dir / "main_result_summary_cn.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=180)
    ax.plot(loads, demand, marker="o", linewidth=3.0, color="#111111", label="任务到达需求")
    for cpu, cap in zip(cpus, caps):
        ax.hlines(
            cap,
            loads.min(),
            loads.max(),
            colors=colors[cpu],
            linestyles="--",
            linewidth=2.0,
            label=f"{cpu:g}GHz 总服务能力 ({cap:.0f}Mbps)",
        )
    ax.fill_between(loads, demand, caps[0], where=demand > caps[0], color="#d62728", alpha=0.16)
    ax.annotate(
        "过载：需求超过 8GHz 服务能力",
        xy=(1.2, 25.2),
        xytext=(1.205, 29.0),
        arrowprops=dict(arrowstyle="->", lw=1.1),
        fontsize=12,
    )
    ax.set_title("算力可行性分析：8GHz 在高负载下不满足稳定条件", fontsize=17)
    ax.set_xlabel("负载系数", fontsize=14)
    ax.set_ylabel("Mbps", fontsize=14)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "capacity_feasibility_cn.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.5), dpi=180)
    axes[0].plot(x, high, marker="o", linewidth=2.6, label="高优先级")
    axes[0].plot(x, low, marker="o", linewidth=2.6, label="低优先级")
    axes[0].set_yscale("log")
    annotate_points(axes[0], x, high, "{:.0f}", -22)
    annotate_points(axes[0], x, low, "{:.0f}", 8)
    axes[0].set_title("CPU 增大后低优先级时延明显下降", fontsize=15)
    axes[0].set_xlabel("单架 UAV 计算能力（GHz）", fontsize=13)
    axes[0].set_ylabel("端到端时延（ms，对数坐标）", fontsize=13)
    axes[0].legend(frameon=False, fontsize=11)
    style(axes[0])

    axes[1].bar(idx - width, high_comp, width=width, label="高优先级完成率")
    axes[1].bar(idx, low_comp, width=width, label="低优先级完成率")
    axes[1].bar(idx + width, qos, width=width, label="QoS 违约率")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels([f"{cpu:g}" for cpu in x])
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("10GHz 以上 QoS 违约率降为 0", fontsize=15)
    axes[1].set_xlabel("单架 UAV 计算能力（GHz）", fontsize=13)
    axes[1].set_ylabel("比例", fontsize=13)
    axes[1].legend(frameon=False, fontsize=11)
    style(axes[1])
    fig.suptitle("负载系数=1.2 下不同 CPU 算力对比", fontsize=18)
    fig.tight_layout()
    fig.savefig(out_dir / "cpu_comparison_load12_cn.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.5), dpi=180)
    cpu8_rows = rows_for(cross_rows, cpu=8.0)
    x_load = np.array([f(row, "load_scale") for row in cpu8_rows])
    y_low = np.array([f(row, "task_delay_low_ms") for row in cpu8_rows])
    y_qos = np.array([f(row, "task_qos_violation_rate") for row in cpu8_rows])
    axes[0].plot(x_load, y_low, marker="o", linewidth=2.8, color="#d62728")
    axes[0].set_yscale("log")
    annotate_points(axes[0], x_load, y_low, "{:.0f}", 7)
    axes[0].set_title("8GHz 下低优先级时延随负载急剧上升", fontsize=15)
    axes[0].set_xlabel("负载系数", fontsize=13)
    axes[0].set_ylabel("低优先级时延（ms，对数坐标）", fontsize=13)
    style(axes[0])
    axes[1].plot(x_load, y_qos, marker="o", linewidth=2.8, color="#d62728")
    for x_i, y_i in zip(x_load, y_qos):
        axes[1].annotate(
            f"{y_i * 100.0:.1f}%",
            (x_i, y_i),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=10,
        )
    axes[1].set_ylim(-0.03, 0.62)
    axes[1].set_title("8GHz 下高负载 QoS 违约率升高", fontsize=15)
    axes[1].set_xlabel("负载系数", fontsize=13)
    axes[1].set_ylabel("QoS 违约率", fontsize=13)
    style(axes[1])
    fig.suptitle("固定 8GHz 时的负载敏感性", fontsize=18)
    fig.tight_layout()
    fig.savefig(out_dir / "load_sweep_8ghz_cn.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4), dpi=180)
    axes[0].plot(episodes, j_values, marker="o", color="0.62", linewidth=2.0, label="评估 J")
    axes[0].plot(episodes, best_so_far, linewidth=3.0, color="#d62728", label="历史最优 J")
    axes[0].scatter(
        [episodes[best_idx]],
        [j_values[best_idx]],
        s=70,
        color="#d62728",
        zorder=5,
        label=f"最优第 {int(episodes[best_idx])} 轮",
    )
    axes[0].set_title("可行负载下目标函数下降并趋稳", fontsize=15)
    axes[0].set_xlabel("训练轮数", fontsize=13)
    axes[0].set_ylabel("目标函数 J（越低越好）", fontsize=13)
    axes[0].legend(frameon=False, fontsize=11)
    style(axes[0])
    axes[1].plot(episodes, high_delay, marker="o", linewidth=2.6, label="高优先级")
    axes[1].plot(episodes, low_delay, marker="o", linewidth=2.6, label="低优先级")
    axes[1].set_title("训练过程中高/低优先级端到端时延", fontsize=15)
    axes[1].set_xlabel("训练轮数", fontsize=13)
    axes[1].set_ylabel("端到端时延（ms）", fontsize=13)
    axes[1].legend(frameon=False, fontsize=11)
    style(axes[1])
    fig.suptitle("收敛性证据：选择验证集上 J 最小的检查点", fontsize=18)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_evidence_cn.png")
    plt.close(fig)

    print(out_dir / "main_result_summary_cn.png")
    print(out_dir / "capacity_feasibility_cn.png")
    print(out_dir / "cpu_comparison_load12_cn.png")
    print(out_dir / "load_sweep_8ghz_cn.png")
    print(out_dir / "convergence_evidence_cn.png")

    # Revised figures for the more defensible setting:
    # local overload exists, but the whole 3-UAV system is still feasible.
    target_load = 1.1
    zone_base = np.array([10.2, 7.0, 3.8], dtype=float)
    zone_demand = zone_base * target_load
    total_demand = float(np.sum(zone_demand))
    local_cpu8_cap = 8.0
    total_cpu8_cap = 24.0

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.2), dpi=180)
    zones = np.arange(3)
    bar_colors = ["#d62728", "#1f77b4", "#1f77b4"]
    axes[0].bar(zones, zone_demand, color=bar_colors, width=0.58)
    axes[0].axhline(
        local_cpu8_cap,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label="单架 8GHz 计算能力 8Mbps",
    )
    for idx_z, val in enumerate(zone_demand):
        axes[0].annotate(
            f"{val:.2f}",
            (idx_z, val),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=12,
        )
    axes[0].set_xticks(zones)
    axes[0].set_xticklabels(["区域1", "区域2", "区域3"])
    axes[0].set_ylabel("区域任务需求（Mbps）")
    axes[0].set_title("局部过载：区域1超过单架 UAV 计算能力")
    axes[0].legend(frameon=False, fontsize=11)
    style(axes[0])

    total_labels = ["总任务需求", "8GHz总服务能力"]
    total_vals = [total_demand, total_cpu8_cap]
    axes[1].bar(
        np.arange(2),
        total_vals,
        color=["#2ca02c", "#1f77b4"],
        width=0.52,
    )
    for idx_t, val in enumerate(total_vals):
        axes[1].annotate(
            f"{val:.1f}",
            (idx_t, val),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=12,
        )
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(total_labels)
    axes[1].set_ylim(0, 27)
    axes[1].set_ylabel("总量（Mbps）")
    axes[1].set_title("全局可行：总需求仍低于三架 UAV 总能力")
    style(axes[1])
    fig.suptitle(
        "主实验场景选择：局部过载但全局可行（负载系数=1.1，CPU=8GHz）",
        fontsize=18,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "local_overload_global_feasible_cn.png")
    plt.close(fig)

    load11_rows = rows_for(cross_rows, load=target_load)
    x11 = np.array([f(row, "cpu_ghz") for row in load11_rows])
    high11 = np.array([f(row, "task_delay_high_ms") for row in load11_rows])
    low11 = np.array([f(row, "task_delay_low_ms") for row in load11_rows])
    qos11 = np.array([f(row, "task_qos_violation_rate") for row in load11_rows])
    comp11 = np.array([f(row, "task_completion_rate") for row in load11_rows])
    energy11 = np.array([f(row, "energy") for row in load11_rows])
    j11 = np.array([f(row, "J") for row in load11_rows])

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.2), dpi=180)
    axes[0].plot(x11, high11, marker="o", linewidth=2.6, label="高优先级")
    axes[0].set_ylim(315, 350)
    annotate_points(axes[0], x11, high11, "{:.1f}", 8)
    axes[0].set_xlabel("单架 UAV 计算能力（GHz）")
    axes[0].set_ylabel("高优先级时延（ms）")
    axes[0].set_title("高优先级时延基本稳定")
    axes[0].legend(frameon=False, fontsize=11)
    style(axes[0])

    axes[1].plot(x11, low11, marker="o", linewidth=2.6, color="#ff7f0e", label="低优先级")
    axes[1].set_ylim(0, max(low11) * 1.18)
    annotate_points(axes[1], x11, low11, "{:.0f}", 8)
    axes[1].set_xlabel("单架 UAV 计算能力（GHz）")
    axes[1].set_ylabel("低优先级时延（ms）")
    axes[1].set_title("低优先级受 CPU 算力影响更明显")
    axes[1].legend(frameon=False, fontsize=11)
    style(axes[1])
    fig.suptitle("负载系数=1.1 时 CPU 算力对高/低优先级时延的影响", fontsize=18)
    fig.tight_layout()
    fig.savefig(out_dir / "cpu_delay_comparison_load11_cn.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.2, 3.6), dpi=180)
    ax.axis("off")
    table_rows = []
    for row in load11_rows:
        table_rows.append(
            [
                f"{f(row, 'cpu_ghz'):.0f}",
                f"{f(row, 'J'):.3f}",
                f"{f(row, 'task_delay_high_ms'):.1f}",
                f"{f(row, 'task_delay_low_ms'):.1f}",
                f"{f(row, 'task_completion_rate') * 100:.2f}%",
                f"{f(row, 'task_qos_violation_rate') * 100:.2f}%",
                f"{f(row, 'energy'):.2f}",
            ]
        )
    col_labels = [
        "CPU(GHz)",
        "J↓",
        "高优时延(ms)",
        "低优时延(ms)",
        "完成率",
        "QoS违约率",
        "能耗",
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.10, 0.10, 0.16, 0.16, 0.13, 0.14, 0.11],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.55)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D5DD")
        if row_idx == 0:
            cell.set_facecolor("#E8EEF5")
            cell.set_text_props(weight="bold")
        elif col_idx == 0:
            cell.set_facecolor("#F7F9FC")
        if row_idx == 1 and col_idx in (3, 5):
            cell.set_facecolor("#FDE8E8")
    ax.set_title(
        "负载系数=1.1 的结果表：8GHz 仍可行但明显更紧张",
        fontsize=17,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "result_table_load11_cn.png")
    plt.close(fig)

    with (out_dir / "load11_cpu_summary.csv").open("w", newline="", encoding="utf-8") as fp:
        fields = [
            "cpu_ghz",
            "load_scale",
            "J",
            "high_delay_ms",
            "low_delay_ms",
            "completion_rate",
            "qos_violation_rate",
            "energy",
        ]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in load11_rows:
            writer.writerow(
                {
                    "cpu_ghz": f(row, "cpu_ghz"),
                    "load_scale": f(row, "load_scale"),
                    "J": f(row, "J"),
                    "high_delay_ms": f(row, "task_delay_high_ms"),
                    "low_delay_ms": f(row, "task_delay_low_ms"),
                    "completion_rate": f(row, "task_completion_rate"),
                    "qos_violation_rate": f(row, "task_qos_violation_rate"),
                    "energy": f(row, "energy"),
                }
            )

    print(out_dir / "local_overload_global_feasible_cn.png")
    print(out_dir / "cpu_delay_comparison_load11_cn.png")
    print(out_dir / "result_table_load11_cn.png")
    print(out_dir / "load11_cpu_summary.csv")


if __name__ == "__main__":
    main()
