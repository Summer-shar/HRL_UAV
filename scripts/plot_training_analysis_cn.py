import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def setup_chinese_font():
    font_paths = [
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def value(row, key):
    return float(row[key])


def fmt_ms(x):
    if x >= 10000:
        return f"{x / 1000:.1f}s"
    if x >= 1000:
        return f"{x:.0f}"
    return f"{x:.1f}"


def save_convergence(eval_rows, out_dir):
    episodes = [int(value(r, "episode")) for r in eval_rows]
    js = [value(r, "J") for r in eval_rows]
    high = [value(r, "task_delay_high_ms") for r in eval_rows]
    low = [value(r, "task_delay_low_ms") for r in eval_rows]
    completion = [value(r, "task_completion_rate") * 100 for r in eval_rows]

    best_idx = min(range(len(js)), key=lambda i: js[i])

    fig, axes = plt.subplots(2, 1, figsize=(9.8, 7.4), dpi=180, sharex=True)
    ax = axes[0]
    ax.plot(episodes, js, marker="o", linewidth=2.5, color="#1f77b4", label="评估目标 J")
    ax.scatter([episodes[best_idx]], [js[best_idx]], s=100, color="#d62728", zorder=4, label="当前最优")
    ax.annotate(
        f"最优：第{episodes[best_idx]}轮\nJ={js[best_idx]:.4f}",
        xy=(episodes[best_idx], js[best_idx]),
        xytext=(episodes[best_idx] + 8, js[best_idx] + 0.018),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.95),
    )
    ax.set_title("新场景收敛结果：CPU=8GHz，分区负载=(12.5, 5.5, 3.5)", fontsize=15, weight="bold")
    ax.set_ylabel("综合目标 J（越低越好）")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(episodes, high, marker="o", linewidth=2.2, color="#2ca02c", label="高优先级时延")
    ax.plot(episodes, low, marker="o", linewidth=2.2, color="#ff7f0e", label="低优先级时延")
    ax.set_yscale("log")
    ax.set_ylabel("端到端时延（ms，对数轴）")
    ax.set_xlabel("训练轮次")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")

    for ep, c in zip(episodes, completion):
        ax.text(ep, min(low) * 0.82, f"{c:.2f}%", ha="center", va="top", fontsize=8, color="#555555")
    ax.text(episodes[0], min(low) * 0.58, "下方数字为任务完成率", fontsize=8.5, color="#555555")

    fig.tight_layout()
    fig.savefig(out_dir / "convergence_new_scene_cn.png", bbox_inches="tight")
    plt.close(fig)


def save_cpu_comparison(rows, out_dir, target_load):
    subset = sorted(
        [r for r in rows if abs(value(r, "load_scale") - target_load) < 1e-9],
        key=lambda r: value(r, "cpu_ghz"),
    )
    if not subset:
        return

    cpus = [value(r, "cpu_ghz") for r in subset]
    js = [value(r, "J") for r in subset]
    high = [value(r, "task_delay_high_ms") for r in subset]
    low = [value(r, "task_delay_low_ms") for r in subset]
    energy = [value(r, "energy") for r in subset]
    best_idx = min(range(len(js)), key=lambda i: js[i])

    fig = plt.figure(figsize=(10.4, 7.0), dpi=180)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0])
    ax = fig.add_subplot(gs[0])
    ax.plot(cpus, high, marker="o", linewidth=2.2, color="#2ca02c", label="高优先级时延")
    ax.plot(cpus, low, marker="o", linewidth=2.2, color="#ff7f0e", label="低优先级时延")
    ax.set_yscale("log")
    ax.set_xlabel("单架 UAV 计算能力（GHz）")
    ax.set_ylabel("端到端时延（ms，对数轴）")
    ax.set_title(f"负载系数={target_load:.1f} 时 CPU 算力对比", fontsize=15, weight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    for x, y in zip(cpus, low):
        ax.text(x, y * 1.08, fmt_ms(y), ha="center", va="bottom", fontsize=8.5, color="#b45f06")
    ax.scatter([cpus[best_idx]], [low[best_idx]], s=95, color="#d62728", zorder=4)

    table_ax = fig.add_subplot(gs[1])
    table_ax.axis("off")
    headers = ["CPU", "J", "高优先级(ms)", "低优先级(ms)", "能耗"]
    table_data = [
        [
            f"{int(c)}GHz",
            f"{j:.4f}",
            fmt_ms(h),
            fmt_ms(l),
            f"{e:.2f}",
        ]
        for c, j, h, l, e in zip(cpus, js, high, low, energy)
    ]
    table = table_ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colColours=["#f2f2f2"] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.3)
    table.scale(1, 1.35)
    for col in range(len(headers)):
        table[(best_idx + 1, col)].set_facecolor("#fff2cc")

    fig.tight_layout()
    suffix = str(target_load).replace(".", "_")
    fig.savefig(out_dir / f"cpu_comparison_load{suffix}_cn.png", bbox_inches="tight")
    plt.close(fig)


def save_load_sweep(rows, out_dir):
    cpus = sorted({value(r, "cpu_ghz") for r in rows})
    loads = sorted({value(r, "load_scale") for r in rows})
    fig, ax = plt.subplots(figsize=(10.2, 6.0), dpi=180)
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b", "#17becf"]
    for idx, cpu in enumerate(cpus):
        subset = sorted([r for r in rows if abs(value(r, "cpu_ghz") - cpu) < 1e-9], key=lambda r: value(r, "load_scale"))
        ax.plot(
            [value(r, "load_scale") for r in subset],
            [value(r, "task_delay_low_ms") for r in subset],
            marker="o",
            linewidth=2.2,
            color=colors[idx % len(colors)],
            label=f"{int(cpu)}GHz",
        )
    ax.set_yscale("log")
    ax.set_xticks(loads)
    ax.set_xlabel("负载系数")
    ax.set_ylabel("低优先级端到端时延（ms，对数轴）")
    ax.set_title("不同 CPU 算力下的低优先级压力变化", fontsize=15, weight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(title="单架 UAV 算力", ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "load_sweep_low_delay_cn.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(eval_rows, cross_rows, out_path):
    best_eval = min(eval_rows, key=lambda r: value(r, "J"))
    final_eval = eval_rows[-1]

    def best_for_load(load):
        subset = [r for r in cross_rows if abs(value(r, "load_scale") - load) < 1e-9]
        return min(subset, key=lambda r: value(r, "J"))

    selected_loads = [1.0, 1.1, 1.2]
    lines = [
        "# 新场景训练结果分析",
        "",
        "训练设置：CPU=8GHz，基础分区负载为 (12.5, 5.5, 3.5) Mbps，训练负载系数在 0.9 到 1.1 之间随机变化，训练 300 轮，每 50 轮评估一次。",
        "",
        "这个场景的意义是：区域1本地需求 12.5Mbps 大于单架 UAV 的 8GHz 服务能力约 8Mbps，但系统总需求 21.5Mbps 小于三架 UAV 总服务能力约 24Mbps。因此它不是简单的全局不可行，而是必须依赖跨 UAV 协作才能处理高负载区域。",
        "",
        "## 收敛结果",
        "",
        f"当前评估最优出现在第 {int(value(best_eval, 'episode'))} 轮：J={value(best_eval, 'J'):.4f}，高优先级端到端时延={value(best_eval, 'task_delay_high_ms'):.1f}ms，低优先级端到端时延={value(best_eval, 'task_delay_low_ms'):.1f}ms，任务完成率={value(best_eval, 'task_completion_rate') * 100:.2f}%。",
        f"第 300 轮结果为：J={value(final_eval, 'J'):.4f}，高优先级端到端时延={value(final_eval, 'task_delay_high_ms'):.1f}ms，低优先级端到端时延={value(final_eval, 'task_delay_low_ms'):.1f}ms，任务完成率={value(final_eval, 'task_completion_rate') * 100:.2f}%。",
        "",
        "| 轮次 | J | 高优先级时延(ms) | 低优先级时延(ms) | 完成率 | QoS违约率 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in eval_rows:
        lines.append(
            f"| {int(value(r, 'episode'))} | {value(r, 'J'):.4f} | {value(r, 'task_delay_high_ms'):.1f} | {value(r, 'task_delay_low_ms'):.1f} | {value(r, 'task_completion_rate') * 100:.2f}% | {value(r, 'task_qos_violation_rate') * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## CPU 与负载对比",
            "",
            "| 负载系数 | 当前 J 最优 CPU | J | 高优先级时延(ms) | 低优先级时延(ms) | 能耗 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for load in selected_loads:
        r = best_for_load(load)
        lines.append(
            f"| {load:.1f} | {int(value(r, 'cpu_ghz'))}GHz | {value(r, 'J'):.4f} | {value(r, 'task_delay_high_ms'):.1f} | {value(r, 'task_delay_low_ms'):.1f} | {value(r, 'energy'):.2f} |"
        )

    lines.extend(
        [
            "",
            "## 可用于汇报的结论",
            "",
            "1. 300 轮基本够用：J 在第 100 轮后已经明显下降，第 200 轮达到当前最优，第 300 轮与最优值非常接近，后续主要是随机评估波动。",
            "2. 高优先级保护有效：跨 CPU 和负载测试中，高优先级端到端时延基本稳定在 322 到 331ms 左右。",
            "3. 8GHz 是高压但不完全失效的边界：load=1.0 时仍可用；load=1.1 时总需求接近 24Mbps，低优先级等待明显增加；load=1.2 时总需求超过 24Mbps，低优先级时延爆炸，这属于容量不足导致的自然结果。",
            "4. CPU 不是越高越好：目标函数同时考虑时延和能耗。load=1.0 时 8GHz 的 J 最低；load=1.1 时 9GHz 最优；load=1.2 时 10GHz 最优。继续提高到 12/14GHz 虽能降低部分时延，但能耗惩罚使综合目标变差。",
            "5. 协作确实被触发：协作比例大多在 0.52 到 0.58 之间，说明高负载区域没有只靠本地 UAV，而是通过空中转发利用其他 UAV 的计算能力。",
            "",
            "对应图片在 analysis_figures 文件夹中：收敛图、load=1.0/1.1/1.2 的 CPU 对比图、以及负载扫描图。",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="results/central_zone125_55_35_cpu8_load09_11_p300")
    args = parser.parse_args()

    setup_chinese_font()
    result_dir = Path(args.result_dir)
    out_dir = result_dir / "analysis_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = read_csv(result_dir / "eval_central_macro.csv")
    cross_rows = read_csv(result_dir / "cross_cpu_load_eval_best.csv")

    save_convergence(eval_rows, out_dir)
    for load in [1.0, 1.1, 1.2]:
        save_cpu_comparison(cross_rows, out_dir, load)
    save_load_sweep(cross_rows, out_dir)
    write_summary(eval_rows, cross_rows, result_dir / "analysis_summary.md")
    print(f"Saved analysis to {result_dir}")


if __name__ == "__main__":
    main()
