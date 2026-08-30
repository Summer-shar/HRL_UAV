"""Generate detailed Chinese analysis and mechanism figures for v64."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_v64_causal_fair as v64  # noqa: E402


ARCHIVE = ROOT / "result_archive" / "v64_causal_fair_hrl"
TRAIN_ROOT = ROOT / "results" / v64.RESULTS_VERSION / "formal" / "seed1"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def method_values(rows: list[dict], method: str, metric: str) -> dict[int, float]:
    return {
        int(row["seed"]): float(row[metric])
        for row in rows
        if row["method"] == method
    }


def plot_zoomed_results(report: dict) -> None:
    v64.configure_plotting()
    methods = ("qa_psca_causal", "hrl_residual", "hrl_direct")
    panels = (
        ("task_delay_weighted_ms", "优先级加权端到端时延 / ms", "(a) 时延"),
        ("queue_total_mbits", "平均队列积压 / Mbit", "(b) 队列"),
        ("coop_ratio", "协作转发比例", "(c) 协作比例"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    x = np.arange(len(methods))
    labels = [v64.METHOD_LABELS[method] for method in methods]
    for axis, (metric, ylabel, title) in zip(axes, panels):
        stats = [report["method_summaries"][method][metric] for method in methods]
        means = np.asarray([item["mean"] for item in stats])
        errors = np.asarray([item["ci95_high"] - item["mean"] for item in stats])
        bars = axis.bar(
            x,
            means,
            yerr=errors,
            capsize=4,
            color=[v64.COLORS[method] for method in methods],
        )
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    fig.suptitle("三个可行方法的局部放大（均值与95% CI）")
    fig.tight_layout()
    fig.savefig(ARCHIVE / "05_可行方法局部放大.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_routing_matrices(report: dict) -> None:
    v64.configure_plotting()
    methods = ("qa_psca_causal", "hrl_residual", "hrl_direct")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
    last_image = None
    for row_index, priority in enumerate(("high", "low")):
        matrices = report[f"mean_routing_{priority}"]
        for column_index, method in enumerate(methods):
            axis = axes[row_index, column_index]
            matrix = np.asarray(matrices[method], dtype=float)
            last_image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlGnBu")
            for source in range(matrix.shape[0]):
                for destination in range(matrix.shape[1]):
                    color = "white" if matrix[source, destination] > 0.55 else "#222222"
                    axis.text(
                        destination,
                        source,
                        f"{100.0 * matrix[source, destination]:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=color,
                    )
            if row_index == 0:
                axis.set_title(v64.METHOD_LABELS[method])
            if column_index == 0:
                axis.set_ylabel(("高优：" if priority == "high" else "低优：") + "源UAV")
            axis.set_xticks(range(3), ("UAV1", "UAV2", "UAV3"))
            axis.set_yticks(range(3), ("UAV1", "UAV2", "UAV3"))
            axis.set_xlabel("目的UAV")
    color_axis = fig.add_axes([0.925, 0.15, 0.015, 0.68])
    fig.colorbar(last_image, cax=color_axis, label="路由比例")
    fig.suptitle("微观路由机制：残差SAC接近QA-PSCA，纯SAC存在过度转发")
    fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.09, wspace=0.28, hspace=0.30)
    fig.savefig(ARCHIVE / "06_微观路由矩阵机制.png", dpi=300, facecolor="white")
    plt.close(fig)


def plot_paired_differences(rows: list[dict]) -> None:
    v64.configure_plotting()
    residual = method_values(rows, "hrl_residual", "task_delay_weighted_ms")
    direct = method_values(rows, "hrl_direct", "task_delay_weighted_ms")
    qa = method_values(rows, "qa_psca_causal", "task_delay_weighted_ms")
    seeds = sorted(residual)
    comparisons = (
        ("残差HRL - 纯SAC HRL", np.asarray([residual[s] - direct[s] for s in seeds]), "#2A9D8F"),
        ("残差HRL - 因果QA-PSCA", np.asarray([residual[s] - qa[s] for s in seeds]), "#D1495B"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.7), sharex=True)
    for axis, (title, delta, color) in zip(axes, comparisons):
        axis.bar(np.arange(len(seeds)), delta, color=color, alpha=0.9)
        axis.axhline(0.0, color="#222222", linewidth=1.0)
        axis.axhline(delta.mean(), color="#222222", linestyle="--", linewidth=1.2)
        axis.set_ylabel("配对时延差 / ms")
        axis.set_title(f"{title}（均值 {delta.mean():+.2f} ms）")
        axis.grid(axis="y", linestyle="--", alpha=0.25)
    axes[-1].set_xticks(np.arange(len(seeds)), [str(seed) for seed in seeds], rotation=55)
    axes[-1].set_xlabel("共同环境种子")
    fig.suptitle("逐种子配对差：负值表示残差HRL更快")
    fig.tight_layout()
    fig.savefig(ARCHIVE / "07_残差HRL配对差.png", dpi=300, facecolor="white")
    plt.close(fig)


def combined_curve(method: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    folders = {
        "hrl_residual": ("micro_pretrain_joint_ppo_sac", "joint_ppo_sac"),
        "hrl_direct": ("micro_pretrain_joint_ppo_direct_sac", "joint_ppo_direct_sac"),
    }
    pretrain_name, joint_name = folders[method]
    pretrain = read_csv(TRAIN_ROOT / pretrain_name / "eval_central_macro.csv")
    joint = read_csv(TRAIN_ROOT / joint_name / "eval_central_macro.csv")

    def value(row: dict) -> float:
        if metric == "queue_total_mbits":
            return float(row["q_high"]) + float(row["q_low"])
        return float(row[metric])

    x = np.asarray(
        [int(float(row["episode"])) for row in pretrain]
        + [200 + int(float(row["episode"])) for row in joint]
    )
    y = np.asarray([value(row) for row in pretrain] + [value(row) for row in joint])
    return x, y


def plot_convergence_and_queue() -> None:
    v64.configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.7))
    for method in ("hrl_residual", "hrl_direct"):
        x_delay, y_delay = combined_curve(method, "task_delay_weighted_ms")
        x_queue, y_queue = combined_curve(method, "queue_total_mbits")
        axes[0].plot(x_delay, y_delay, marker="o", markersize=3, linewidth=2, color=v64.COLORS[method], label=v64.METHOD_LABELS[method])
        axes[1].plot(x_queue, y_queue, marker="o", markersize=3, linewidth=2, color=v64.COLORS[method], label=v64.METHOD_LABELS[method])
    for axis in axes:
        axis.axvline(200, color="#333333", linestyle="--", linewidth=1.1)
        axis.grid(linestyle="--", alpha=0.25)
        axis.set_xlabel("累计训练轮次")
    axes[0].set_ylabel("固定策略评估端到端时延 / ms")
    axes[0].set_title("(a) 时延收敛")
    axes[1].set_ylabel("固定策略评估队列 / Mbit")
    axes[1].set_title("(b) 队列稳定性")
    axes[0].legend()
    fig.suptitle("微观预训练（0–200）与上下层联合训练（200–600）")
    fig.tight_layout()
    fig.savefig(ARCHIVE / "02b_收敛与队列稳定性.png", dpi=300, facecolor="white")
    plt.close(fig)


def write_analysis(report: dict, macro: dict, layer: dict) -> None:
    summaries = report["method_summaries"]
    contrasts = report["paired_contrasts"]
    rq = contrasts["hrl_residual_minus_qa_psca_causal"]
    rd = contrasts["hrl_residual_minus_hrl_direct"]
    rn = contrasts["hrl_residual_minus_no_coop"]
    rx = contrasts["hrl_residual_minus_xu_cvx"]
    macro_delta = macro["paired_hrl_minus_centroid"]
    layer_effects = layer["effects"]
    residual_with_ppo = layer_effects[
        "residual_effect_with_ppo_residual_minus_qa"
    ]["task_delay_weighted_ms"]
    macro_with_qa = layer_effects[
        "macro_effect_with_qa_ppo_minus_centroid"
    ]["task_delay_weighted_ms"]
    interaction = layer["interaction"]["task_delay_weighted_ms"]

    def ci(item: dict, digits: int = 3) -> str:
        return (
            f"{item['mean']:.{digits}f} "
            f"[{item['ci95_low']:.{digits}f}, {item['ci95_high']:.{digits}f}]"
        )

    qa_delay = summaries["qa_psca_causal"]["task_delay_weighted_ms"]["mean"]
    residual_delay = summaries["hrl_residual"]["task_delay_weighted_ms"]["mean"]
    relative_gap = 100.0 * (residual_delay - qa_delay) / qa_delay
    lines = [
        "# v64 公平因果口径：完整结果与分层机制分析",
        "",
        "## 1. 本轮到底改了什么",
        "",
        "- 物理场景保持 v49：3 架 UAV、39 名用户、RPGM 移动、CPU 10 GHz、负载 [12.5, 5.5, 3.5] Mbps 循环切换。",
        "- QA-PSCA 不再读取真实期望到达率，只使用当帧已经观测到的任务到达量。",
        "- Xu-CVX 的空到达帧不再用真实到达率伪造任务。",
        "- PPO+残差 SAC 与 PPO+纯 SAC 均独立从头训练：200 轮微观预训练 + 400 轮上下层同步联合训练。",
        "- 正式结果固定 final checkpoint，并用 8000–8019 共 20 个 CRN 环境种子配对评估。",
        "",
        "## 2. 主结果",
        "",
        "| 方法 | 时延/ms | 95% CI | 队列/Mbit | 功率/W | 协作比例 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in report["delay_ranking"]:
        item = summaries[method]
        delay = item["task_delay_weighted_ms"]
        lines.append(
            f"| {v64.METHOD_LABELS[method]} | {delay['mean']:.3f} | "
            f"[{delay['ci95_low']:.3f}, {delay['ci95_high']:.3f}] | "
            f"{item['queue_total_mbits']['mean']:.3f} | "
            f"{item['average_power_w']['mean']:.3f} | "
            f"{item['coop_ratio']['mean']:.4f} |"
        )
    lines += [
        "",
        f"当前最优是因果 QA-PSCA（{qa_delay:.3f} ms）。本文残差 HRL 为 {residual_delay:.3f} ms，"
        f"慢 {ci(rq['task_delay_weighted_ms'])} ms，相对差距 {relative_gap:.2f}%。因此不能声称本文方法超过所有已测方法。",
        "",
        "## 3. 每一层到底产生了什么效果",
        "",
        "### 3.1 宏观 PPO 轨迹层",
        "",
        f"固定最终残差 SAC，仅把宏观动作从 PPO 改为动态质心后，PPO 相对质心的配对时延差为 "
        f"+{ci(macro_delta['task_delay_weighted_ms'])} ms。PPO 平均多偏离质心 "
        f"{macro_delta['u2g_distance_m']['mean']:.3f} m，但没有形成收益；当前上层贡献是小幅负值。",
        "",
        "这也解释了轨迹图为什么几乎一样：动态质心本身随用户云移动，PPO 只在其附近增加约 2.24 m 残差；在数百米坐标范围内肉眼难以区分。轨迹不是没执行，而是学到的非零残差很小且无益。",
        "",
        "### 3.2 微观动作结构：QA锚定残差 vs 从零学习完整路由",
        "",
        f"残差 HRL 相对 pure/direct SAC 的配对改善为 {ci(rd['task_delay_weighted_ms'])} ms；"
        f"队列减少 {-rd['queue_total_mbits']['mean']:.3f} Mbit，低优任务时延减少 {-rd['task_delay_low_ms']['mean']:.1f} ms，"
        f"高优任务时延减少 {-rd['task_delay_high_ms']['mean']:.1f} ms。",
        "",
        f"pure SAC 的协作比例比残差 SAC 高 {-rd['coop_ratio']['mean'] * 100.0:.2f} 个百分点。"
        "它不是协作不够，而是转发过多、方向不准：尤其低优 UAV3 本地比例只有 60.6%，"
        "而 QA-PSCA/残差 SAC 约为 76.9%/77.4%。过度转发增加 U2U 与目的端排队，最终主要放大低优时延。",
        "",
        "残差架构的路由矩阵与 QA-PSCA 很接近，因为执行动作以 QA-PSCA 为锚点；pure SAC 必须从零发现 18 维 simplex 路由结构，虽已收敛，仍停在较差策略。",
        "",
        "但这项比较不能证明学习残差本身有益。完整 2×2 同-checkpoint 审计显示，在相同 PPO 轨迹下，"
        f"加入 SAC 残差相对纯 QA 锚点反而变慢 {ci(residual_with_ppo)} ms；"
        f"在纯 QA 微观下，PPO 相对质心的宏观效应为 {ci(macro_with_qa)} ms，CI 跨零。"
        f"宏微观交互项为 {ci(interaction)} ms，CI 也跨零。",
        "",
        "因此相对 direct SAC 的 112 ms 优势主要来自 QA-PSCA 锚点提供的结构先验，不应归功于 SAC 学习修正。当前 learned residual 是小幅有害的。",
        "",
        "### 3.3 协作机制本身",
        "",
        f"残差 HRL 相对不协作降低时延 {-rn['task_delay_weighted_ms']['mean']:.1f} ms，"
        f"队列减少 {-rn['queue_total_mbits']['mean']:.1f} Mbit。主要收益来自低优任务：低优时延降低 "
        f"{-rn['task_delay_low_ms']['mean']:.0f} ms；作为代价，高优时延增加 {rn['task_delay_high_ms']['mean']:.1f} ms。",
        "",
        "因此协作的具体作用不是让所有任务都更快，而是把部分任务转给空闲 UAV，阻止低优队列持续积压，使整个优先级加权端到端时延大幅下降。",
        "",
        "### 3.4 残差 HRL 与因果 QA-PSCA 的剩余差距",
        "",
        f"残差 HRL 比 QA-PSCA 慢 {rq['task_delay_weighted_ms']['mean']:.3f} ms，其中高优仅慢 "
        f"{rq['task_delay_high_ms']['mean']:.3f} ms，低优慢 {rq['task_delay_low_ms']['mean']:.3f} ms，"
        f"队列多 {rq['queue_total_mbits']['mean']:.3f} Mbit；功率差 "
        f"{ci(rq['average_power_w'])} W，CI 跨零。",
        "",
        f"同 checkpoint 宏观 PPO 只解释约 {macro_delta['task_delay_weighted_ms']['mean']:.3f} ms 损失，"
        "其余差距主要来自 SAC 残差对 QA 动作的有害修正。QA-PSCA 每帧利用显式队列/信道模型做优化；残差策略没有超过锚点。",
        "",
        "### 3.5 Xu-CVX 适配版为什么很慢",
        "",
        f"残差 HRL 相对 Xu-CVX 快 {-rx['task_delay_weighted_ms']['mean']:.1f} ms，"
        f"队列少 {-rx['queue_total_mbits']['mean']:.1f} Mbit。Xu-CVX 高优时延仍为 "
        f"{summaries['xu_cvx']['task_delay_high_ms']['mean']:.1f} ms，但低优时延达到 "
        f"{summaries['xu_cvx']['task_delay_low_ms']['mean']:.0f} ms。",
        "",
        "原因不是 CVX 数值求解失败，而是参考论文的单帧任务语义直接接到本仓库持续排队环境后，不会主动清理跨帧低优积压。本结果只能称为 Xu-CVX 适配基线，不能称为逐项复现原论文。",
        "",
        "## 4. 是否收敛、队列是否稳定",
        "",
        "- 残差 SAC 在预训练早期已进入约 623–628 ms 区间；联合训练 final 约 630 ms，没有发散，队列约 10 Mbit。",
        "- pure SAC 从约 3.6 s 降到预训练 final 约 769 ms，联合 final 约 753 ms，说明它真实学习并收敛，但上界明显差于残差结构。",
        "- PPO 在联合阶段没有把残差方法继续推低，最终宏观反事实表明其轨迹比质心慢约 0.46 ms。",
        "",
        "## 5. 当前可以写与不能写的结论",
        "",
        "可以写：",
        "",
        "1. 分层训练流程真实运行：先独立训练微观 SAC，再让 PPO/SAC 以不同频率同步联合更新。",
        "2. QA锚定残差架构显著优于从零学习完整路由的 pure SAC，并显著优于 Xu-CVX 适配、随机和不协作。",
        "3. 因果 QA-PSCA 是当前最强方法；残差 HRL 距其约 1.7%。残差方法虽不运行 Xu-CVX，但部署时仍需每帧运行 QA-PSCA 锚点。",
        "4. 协作的主要价值是消除低优队列积压，路由矩阵给出了可解释机制。",
        "",
        "不能写：",
        "",
        "1. 不能写本文方法在全部已测方法中最优，因为 QA-PSCA 显著更快。",
        "2. 不能写 PPO 学出了优于质心的协作轨迹；同 checkpoint 反事实显示 PPO 小幅有害。",
        "3. 不能写 SAC 残差超过了 QA 锚点；2×2 审计显示残差修正使时延增加约 10.48 ms。",
        "4. 不能把 20 个环境种子写成 20 个训练种子；学习方法目前只有训练 seed=1。",
        "5. 不能把 QA-PSCA 写成 Xu et al. 的 SCA；两者目标、队列语义和求解流程不同。",
        "",
        "## 6. 文件说明",
        "",
        "- `per_seed_raw.csv`：6 方法 × 20 环境种子的原始数据。",
        "- `macro_counterfactual_per_seed.csv`：同一残差 SAC 下 PPO 与动态质心的配对重放。",
        "- `report.json`、`macro_counterfactual_report.json`、`layer_2x2_report.json`：均值、95% CI 与分层配对差。",
        "- `01`/`05`：主结果与可行方法放大；`02b`：时延收敛和队列；`03`：轨迹；`04`：宏观贡献；`06`：路由矩阵；`07`：逐种子配对差；`08`：完整2×2分层归因。",
        "",
        "## 7. 证据边界",
        "",
        "本轮用于筛选与机制定量，训练随机性仍只有 seed=1。若论文主表要作稳定性主张，仍需在冻结代码和场景后补独立训练种子。",
    ]
    (ARCHIVE / "COMPLETE_ANALYSIS_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    report = load_json(ARCHIVE / "report.json")
    macro = load_json(ARCHIVE / "macro_counterfactual_report.json")
    layer = load_json(ARCHIVE / "layer_2x2_report.json")
    rows = read_csv(ARCHIVE / "per_seed_raw.csv")
    plot_zoomed_results(report)
    plot_routing_matrices(report)
    plot_paired_differences(rows)
    plot_convergence_and_queue()
    write_analysis(report, macro, layer)


if __name__ == "__main__":
    main()
