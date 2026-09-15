"""Offline figures and a descriptive report; never invokes an environment or policy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("SAC", "Local", "Uniform", "Random")
LABELS = {"SAC": "SAC", "Local": "本地", "Uniform": "均分", "Random": "随机"}
COLORS = {"SAC": "#b32430", "Local": "#555555", "Uniform": "#2868a0", "Random": "#19806a"}
MARKERS = dict(zip(METHODS, ("o", "s", "^", "D")))
SEEDS = (201, 202, 203)
SCENES = (33600, 33601)
CONDITIONS = ("base", "load080", "load110", "bandwidth3", "bandwidth45")


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (ValueError, TypeError):
                pass
    return rows


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_case(row):
    path = Path(row["directory"])
    reference = path / "reused.json"
    return Path(read_json(reference)["source"]) if reference.exists() else path


def get(summary, condition, micro, macro="PPO120"):
    return next(r for r in summary if (r["condition"], r["micro"], r["macro"]) == (condition, micro, macro))


def mean_metric(rows, metric):
    return float(np.mean([r[metric] for r in rows]))


def setup():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "lines.linewidth": 1.6, "savefig.dpi": 220})


def finish(fig, title, note, name, directory, pdf, top=.94):
    fig.suptitle(title, fontsize=13, y=.985)
    fig.text(.5, .023, note, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(.015, .078, .985, top), h_pad=2.1, w_pad=2.2)
    pdf.savefig(fig)
    fig.savefig(directory / f"{name}.png")
    fig.savefig(directory / f"{name}.svg")
    plt.close(fig)


def method_figure(summary, directory, pdf):
    fig, axes = plt.subplots(2, 3, figsize=(8, 6.5))
    specifications = (("J", "(a) 原物理代价J", True, METHODS),
                      ("J", "(b) J局部放大（Local见a）", False, ("SAC", "Uniform", "Random")),
                      ("task_delay_weighted_ms", "(c) 加权完成时延 / ms", True, METHODS),
                      ("task_completion_low_rate", "(d) 低优任务完成率 / %", False, METHODS),
                      ("queue_total_mbit", "(e) 900秒期末总积压 / Mbit", True, METHODS),
                      ("power_w", "(f) 完整平均功率 / W", False, METHODS))
    for ax, (key, label, log, methods) in zip(axes.flat, specifications):
        factor = 100 if key == "task_completion_low_rate" else 1
        for x, micro in enumerate(methods):
            row = get(summary, "base", micro)
            ax.errorbar(x, row[key]*factor, yerr=row["std_"+key]*factor, fmt=MARKERS[micro],
                        color=COLORS[micro], capsize=4, markersize=6)
        ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods])
        ax.set_title(label, loc="left")
        ax.grid(axis="y", alpha=.22)
        ax.margins(x=.18, y=.15)
        if log:
            ax.set_yscale("log")
            ax.set_ylabel("对数刻度")
    finish(fig, "同一冻结PPO策略下的微观方法比较", "误差棒：三个PPO初始化的两场景均值之样本标准差；SAC仅来源101。各微观规则闭环运行。",
           "01_methods", directory, pdf)


def sensitivity_figure(summary, kind, directory, pdf):
    if kind == "load":
        conditions, x = ("load080", "base", "load110"), (.8, 1., 1.1)
        xlabel, title, name = "总负载倍率", "固定权重的负载敏感性", "02_load"
        panels = (("J", "(a) 原物理代价J", True),
                  ("task_delay_weighted_ms", "(b) 加权完成时延 / ms", True),
                  ("task_unfinished_low_count", "(c) 期末未完成低优任务数", False),
                  ("mean_queue_compute_mbit", "(d) 平均计算积压 / Mbit", True))
    else:
        conditions, x = ("bandwidth3", "bandwidth45", "base"), (3., 4.5, 6.)
        xlabel, title, name = "U2U总带宽 / MHz", "固定权重的带宽敏感性", "03_bandwidth"
        panels = (("J", "(a) 原物理代价J", True),
                  ("task_delay_weighted_ms", "(b) 加权完成时延 / ms", True),
                  ("mean_queue_transfer_mbit", "(c) 平均转发积压 / Mbit", False),
                  ("tx_throughput_mbps_wall", "(d) 实际U2U吞吐量 / Mbps", False))
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.8))
    for ax, (metric, label, log) in zip(axes.flat, panels):
        for micro in METHODS:
            selected = [get(summary, c, micro) for c in conditions]
            ax.errorbar(x, [r[metric] for r in selected], yerr=[r["std_"+metric] for r in selected],
                        color=COLORS[micro], marker=MARKERS[micro], capsize=3, label=LABELS[micro])
        reference = [get(summary, c, "SAC", "Predictive")[metric] for c in conditions]
        ax.plot(x, reference, ":", color="#8f6b00", label="Predictive+SAC参考", marker="x")
        ax.set_title(label, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_xticks(x)
        ax.grid(alpha=.2)
        if log:
            ax.set_yscale("log")
            ax.set_ylabel("对数刻度")
        elif metric == "mean_queue_transfer_mbit":
            ax.set_yscale("symlog", linthresh=1)
            ax.set_ylabel("0~1线性，其余对数")
        if not log:
            ax.set_ylim(bottom=0)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .946), ncol=5, fontsize=8)
    finish(fig, title, "两开发场景；PPO120与SAC101权重均不变，未逐参数点重训；不是各配置最优性能。误差棒仅反映PPO初始化差异。",
           name, directory, pdf, top=.89)


def learning_figure(directory, pdf):
    initial = ROOT / "results/mission_ppo_stage4a_qualification"
    continuation = ROOT / "results/mission_ppo_continue120to600/final/learning_curve.csv"
    later = read_csv(continuation)
    groups = []
    for seed in SEEDS:
        rows = []
        for window in (0, 30, 60, 90, 120):
            r = [read_json(initial / f"seed{seed}/monitor{window:03d}/{s}/summary.json") for s in SCENES]
            rows.append((window, mean_metric(r, "J")))
        for window in sorted({r["window"] for r in later if r["window"] > 120}):
            r = [r for r in later if r["ppo_seed"] == seed and r["window"] == window]
            if len(r) != 2:
                raise ValueError("Incomplete learning monitor")
            rows.append((window, mean_metric(r, "J")))
        groups.append(np.asarray(rows))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.8))
    for ax, maximum, title in zip(axes, (120, 600), ("(a) 原120窗口资格训练", "(b) 已完成的120至600窗口续训")):
        for g, seed, color in zip(groups, SEEDS, ("#2868a0", "#19806a", "#a65025")):
            select = g[g[:, 0] <= maximum]
            ax.plot(select[:, 0], select[:, 1], marker="o", color=color, label=f"PPO {seed}", alpha=.8)
        x = groups[0][:, 0]
        values = np.array([g[:, 1] for g in groups])
        mask = x <= maximum
        ax.plot(x[mask], values.mean(0)[mask], "k-", linewidth=2, label="三初始化均值")
        for controller, text, style in (("hover", "Hover+SAC", "--"), ("predictive_centroid", "Predictive+SAC", ":")):
            refs = [read_json(ROOT / f"results/mission_rpgm_v1_pilot/{controller}_{s}/summary.json") for s in SCENES]
            ax.axhline(mean_metric(refs, "J"), color="#777777", linestyle=style, label=text)
        ax.set_title(title, loc="left")
        ax.set_xlabel("累计训练窗口（每窗口30个宏观决策）")
        ax.set_ylabel("开发场景原物理代价J")
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .943), ncol=3, fontsize=7)
    finish(fig, "已有PPO训练过程：保留全部初始化与后期回退", "每个点是33600/33601的均值，不是训练奖励；不平滑、不选最优checkpoint，不据此声称已经收敛。",
           "04_learning", directory, pdf, top=.86)


def trajectory_figure(rows, directory, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
    labels = [("PPO120", seed, f"PPO初始化{seed}") for seed in SEEDS] + [("Predictive", 0, "Predictive规则参考")]
    palette = ("#2868a0", "#a65025", "#19806a")
    for ax, (macro, seed, title) in zip(axes.flat, labels):
        row = next(r for r in rows if (r["condition"], r["macro"], r["ppo_seed"], r["micro"], r["seed"])
                   == ("base", macro, seed, "SAC", 33600))
        with np.load(source_case(row) / "trajectory.npz") as data:
            q, groups, users = data["uav"], data["groups"], data["users"]
        for i, color in enumerate(palette):
            ax.plot(groups[:, i, 0], groups[:, i, 1], "--", color=color, alpha=.7, linewidth=1.2,
                    label=f"群{i+1}中心")
            ax.plot(q[:, i, 0], q[:, i, 1], color=color, label=f"UAV{i+1}")
            ax.scatter(q[0, i, 0], q[0, i, 1], marker="o", facecolors="none", edgecolors=color, s=30)
            ax.scatter(q[-1, i, 0], q[-1, i, 1], marker="*", color=color, s=75)
            for t in (3000, 6000):
                a, b = q[t-100, i], q[t, i]
                ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=color, lw=1.1))
        ax.scatter(users[0, :, 0], users[0, :, 1], s=5, color="#b8b8b8", alpha=.45)
        ax.scatter(users[-1, :, 0], users[-1, :, 1], s=5, color="#b8b8b8", alpha=.45)
        ax.set(xlim=(200, 990), ylim=(250, 780), xlabel="世界坐标x / m", ylabel="世界坐标y / m")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, loc="left")
        ax.grid(alpha=.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .052), ncol=6, fontsize=7)
    finish(fig, "同一开发场景的真实900秒轨迹", "场景33600预先固定；圆点为UAV起点，星号为终点；灰点只显示用户初末位置。实线UAV，虚线群中心。",
           "05_trajectories", directory, pdf)


def route_figure(rows, directory, pdf):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.8))
    groupings = [("PPO120", seed, f"PPO{seed}+SAC") for seed in SEEDS] + [("Predictive", 0, "Predictive+SAC参考")]
    for ax, (macro, seed, title) in zip(axes.flat, groupings):
        selected = [r for r in rows if (r["condition"], r["macro"], r["ppo_seed"], r["micro"])
                    == ("base", macro, seed, "SAC")]
        records = [r for item in selected for r in read_csv(source_case(item) / "routes.csv")]
        x = np.arange(3)
        for priority, shift, color, hatch in (("high", -.18, "#2868a0", ""), ("low", .18, "#a65025", "//")):
            values = [100*np.mean([r["local_ratio"] for r in records if r["role"] == role and r["priority"] == priority])
                      for role in ("heavy", "medium", "light")]
            ax.bar(x+shift, values, width=.34, color=color, hatch=hatch, label="高优" if priority=="high" else "低优")
        ax.set_xticks(x, ("重载源", "中载源", "轻载源"))
        ax.set_ylim(0, 105)
        ax.set_ylabel("本地计算比例 / %")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .944), ncol=2)
    finish(fig, "优先级与负载角色对应的实际执行路由", "两开发场景全部服务帧的源任务路由比例均值；角色随负载轮换，不等同于固定UAV编号。",
           "06_routing", directory, pdf, top=.89)


def queue_figure(rows, directory, pdf):
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    for micro in METHODS:
        row = next(r for r in rows if (r["condition"], r["macro"], r["ppo_seed"], r["micro"], r["seed"])
                   == ("base", "PPO120", 201, micro, 33600))
        records = read_csv(source_case(row) / "queues.csv")
        times = np.array([r["time_s"] for r in records])
        for ax, stage, label in zip(axes, ("upload", "compute", "transfer"), ("上传", "计算", "转发")):
            values = np.array([r[f"queue_{stage}_mbit"] for r in records])
            ax.plot(times, values, color=COLORS[micro], linewidth=.8, alpha=.85, label=LABELS[micro])
            ax.set_ylabel(f"{label}积压 / Mbit")
            ax.set_yscale("symlog", linthresh=1)
            ax.grid(alpha=.2)
            ax.set_xlim(0, 900)
    for ax in axes:
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .945), ncol=4)
    axes[-1].set_xlabel("时间 / s")
    finish(fig, "闭环运行中的各阶段积压", "预先固定PPO201、场景33600；全部0.5秒帧末记录，无平滑。纵轴0~1 Mbit线性，其余为对数。",
           "07_queues", directory, pdf, top=.89)


def markdown_table(headers, lines):
    return ["|"+"|".join(headers)+"|", "|"+"|".join("---" for _ in headers)+"|"] + ["|"+"|".join(map(str, r))+"|" for r in lines]


def write_report(output, summary, rows, manifest):
    contents = ["# Mission论文数据：固定策略消融与参数敏感性", "",
        "## 先看结论", "",
        "本轮完成122次新增900秒评价，复用8条已有记录，共130条；没有训练或调参。",
        "下面比较均为同一PPO120权重下替换微观方法，PPO会根据各自队列观测闭环决策，并非强制相同轨迹。",
        "三个PPO初始化201/202/203共享SAC101；33600/33601均为已用开发场景。不能当作完整三seed HRL最终测试。", ""]
    base = get(summary, "base", "SAC")
    for other in METHODS[1:]:
        ref = get(summary, "base", other)
        pct = 100*(base["J"]-ref["J"])/ref["J"]
        dt = base["task_delay_weighted_ms"]-ref["task_delay_weighted_ms"]
        contents.append(f"- SAC相对{other}：J差{base['J']-ref['J']:+.6f}（{pct:+.2f}%），加权完成时延差{dt:+.2f} ms。")
    ref = get(summary, "base", "SAC", "Predictive")
    contents += [f"- PPO120+SAC相对Predictive+SAC：J差{base['J']-ref['J']:+.6f}（{100*(base['J']-ref['J'])/ref['J']:+.2f}%）。",
                 "- SCA、TD3和异频联合训练尚未补齐；本轮不能证明本文完整方法优于这些未运行基线。", "",
                 "## 中心配置方法比较", ""]
    table = []
    for micro in METHODS:
        r = get(summary, "base", micro)
        table.append([micro, f"{r['J']:.6f} +/- {r['std_J']:.6f}", f"{r['task_delay_weighted_ms']:.2f}",
                      f"{r['task_delay_high_ms']:.2f}", f"{r['task_delay_low_ms']:.2f}",
                      f"{r['task_completion_high_rate']*100:.3f}/{r['task_completion_low_rate']*100:.3f}",
                      f"{r['task_unfinished_high_count']:.2f}/{r['task_unfinished_low_count']:.2f}",
                      f"{r['queue_total_mbit']:.2f}", f"{r['power_w']:.2f}"])
    contents += markdown_table(["微观方法", "J：均值+/-标准差", "加权/ms", "高优/ms", "低优/ms", "高/低完成率/%", "高/低未完数", "期末队列/Mbit", "完整功率/W"], table)
    contents += ["", "标准差：先对每个PPO初始化的两场景求均值，再对三个均值求样本标准差。不是六次独立训练。", "",
                 "## 全部固定参数点", "", "每点未重训；下表保留所有结果，不依照优劣删点。", ""]
    table = []
    for condition in CONDITIONS:
        r = get(summary, condition, "SAC")
        deltas = [100*(r["J"]-get(summary, condition, m)["J"])/get(summary, condition, m)["J"] for m in METHODS[1:]]
        table.append([condition, *[f"{get(summary, condition, m)['J']:.6f}" for m in METHODS],
                      f"{get(summary, condition, 'SAC', 'Predictive')['J']:.6f}",
                      "/".join(f"{d:+.2f}%" for d in deltas)])
    contents += markdown_table(["配置", "SAC J", "Local J", "Uniform J", "Random J", "Predictive+SAC J", "SAC相对Local/Uniform/Random"], table)
    contents += ["", "base=原负载/6MHz；load080、load110为原负载0.8、1.1倍；bandwidth3、bandwidth45为3、4.5MHz。",
        "这些是冻结策略的参数敏感性，不是逐配置训练后的最优性能；跨规则外生过程配对不等于队列或UAV轨迹相同。", "",
        "## 3MHz与6MHz：完成情况和实际服务", ""]
    resource_table, resource_rows = [], []
    for condition in ("base", "bandwidth3"):
        for micro in METHODS:
            r = get(summary, condition, micro)
            selected = [read_json(Path(item["directory"])/"summary.json") for item in rows
                        if (item["condition"], item["macro"], item["micro"]) == (condition, "PPO120", micro)]
            active = mean_metric(selected, "mean_active_links_service")
            link_rate = mean_metric(selected, "mean_rate_per_active_link_mbps")
            resource_rows.append(dict(condition=condition, micro=micro, active_links=active,
                                      active_link_rate_mbps=link_rate, **{k: r[k] for k in
                                      ("J", "J_holding", "J_energy", "task_delay_weighted_ms",
                                       "task_delay_high_ms", "task_delay_low_ms", "task_completion_high_rate",
                                       "task_completion_low_rate", "task_unfinished_high_count", "task_unfinished_low_count",
                                       "mean_queue_transfer_mbit", "mean_queue_compute_mbit", "queue_total_mbit",
                                       "tx_throughput_mbps_wall", "u2u_tx_time_s", "power_w")}))
            resource_table.append(["6" if condition=="base" else "3", micro, f"{r['task_delay_weighted_ms']:.2f}",
                f"{r['task_completion_high_rate']*100:.3f}/{r['task_completion_low_rate']*100:.3f}",
                f"{r['task_unfinished_high_count']:.2f}/{r['task_unfinished_low_count']:.2f}",
                f"{r['mean_queue_transfer_mbit']:.2f}/{r['mean_queue_compute_mbit']:.2f}",
                f"{r['tx_throughput_mbps_wall']:.2f}", f"{active:.2f}", f"{link_rate:.2f}", f"{r['power_w']:.2f}"])
    contents += markdown_table(["MHz", "微观", "加权/ms", "高/低完成率/%", "高/低未完数", "平均转发/计算积压/Mbit",
                                "U2U吞吐/Mbps", "活跃链路数", "活跃链路均速/Mbps", "完整功率/W"], resource_table)
    with open(output / "resource_summary.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(resource_rows[0]))
        writer.writeheader()
        writer.writerows(resource_rows)
    contents += ["", "吞吐分母为900秒；活跃链路数按服务物理步平均，活跃链路速率按活跃链路-步平均。不同分母不得混用。",
        "这组结果检验的是既有规则在通信资源收紧后的表现。路由及实际服务记录可用于解释积压变化，不能把实测吞吐反推为固定网络容量。",
        "3MHz使用原6MHz权重，没有把它写成3MHz独立训练结果。", "", "## 可直接查看的图", ""]
    for name, title in (("01_methods", "方法比较"), ("02_load", "负载敏感性"), ("03_bandwidth", "带宽敏感性"),
                        ("04_learning", "训练过程含600窗口续训"), ("05_trajectories", "真实轨迹"),
                        ("06_routing", "优先级路由"), ("07_queues", "各阶段积压")):
        contents += [f"### {title}", f"![{title}](figures/{name}.png)", ""]
    contents += ["## 结果解释边界", "",
        "- J是原驻留代价与能量代价的物理步均值；不是训练折扣回报。",
        "- 时延仅统计已完成父任务。完成率、未完成数、期末队列并列报告，不能把未完成全部叫超时。",
        "- SAC是否更好按明确比较对象及指定J判断，不要求每个优先级、功率、完成率都同时最优。",
        "- 保留Predictive参考及三个PPO的600窗口回退，不能用有利seed替代完整重复。",
        "- 所有曲线只连接实际参数点，无拟合伪点；实际U2U吞吐不是固定网络容量。", "",
        "## 尚缺实验及SCA接口问题", "",
        "历史f6b20dd:hrl_uav/xu2024_sca.py实现了逐任务rho与CPU联合求解；旧compare_xu2024_sca.py负责聚合执行。",
        "当前mission固定CPU服务规则，候选无线的实际速率依赖活跃链路及每机功率分摊。旧适配器不能未经核验直接移植。",
        "还需要同信息、同可执行动作的SCA适配核验及实际结果；本轮未创建一个新优化器来冒充论文SCA。",
        "TD3对比、两层异频联合训练、最终保留场景评价仍缺。本轮没有触碰33700-33709。", "",
        "## 文件与核验", "",
        f"运行HEAD：`{manifest['actual_head']}`。物理实现与原mission锁定一致。",
        "`manifest.json`含协议、有效配置、源码和来源指纹；`per_seed.csv`保留130条记录；",
        "`per_training_seed.csv`、`summary.csv`、`paired_deltas.csv`保留层级汇总及逐场景差。",
        "每个cases目录保留真实轨迹、0.5秒队列、实际链路量、路由、宏观动作及CRN；复用项记录来源路径和哈希。",
        f"smoke通过：{manifest['smoke_passed']}；外生过程配对：{manifest['crn_all_equal']}；源码未变：{manifest['source_unchanged']}。",
        f"本轮评价用时：{manifest['elapsed_seconds']/60:.1f}分钟；学习、反向传播、replay写入/采样均为0。", ""]
    (output / "result.md").write_text("\n".join(contents), encoding="utf-8")


def run(output, rerender=False):
    output = Path(output).resolve()
    manifest = read_json(output / "manifest.json")
    if manifest["status"] != "COMPLETE_STOPPED" or not manifest["source_unchanged"]:
        raise ValueError("Do not plot partial or source-mismatched results")
    directory = output / "figures"
    inputs = [output / f for f in ("manifest.json", "per_seed.csv", "summary.csv")]
    input_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    if rerender:
        previous = read_json(directory / "figure_manifest.json")
        if previous["inputs"] != input_hashes:
            raise ValueError("Figure inputs changed; refusing to overwrite the previous figures")
    directory.mkdir(exist_ok=rerender)
    rows, summary = read_csv(output / "per_seed.csv"), read_csv(output / "summary.csv")
    if len(rows) != 130 or len(summary) != 25:
        raise ValueError("Incomplete comparison data")
    setup()
    with PdfPages(directory / "comparison_figures.pdf") as pdf:
        method_figure(summary, directory, pdf)
        sensitivity_figure(summary, "load", directory, pdf)
        sensitivity_figure(summary, "bandwidth", directory, pdf)
        learning_figure(directory, pdf)
        trajectory_figure(rows, directory, pdf)
        route_figure(rows, directory, pdf)
        queue_figure(rows, directory, pdf)
    write_report(output, summary, rows, manifest)
    if input_hashes != {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}:
        raise ValueError("Input data changed during offline rendering")
    evidence = dict(builder=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    inputs=input_hashes,
                    figure_count=7, original_data_modified=False, environments_run=False)
    (directory / "figure_manifest.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(output / "result.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--rerender", action="store_true", help="Replace only this builder's figures/report after verifying unchanged inputs")
    args = parser.parse_args()
    run(args.output, args.rerender)
