"""Render Chinese PNGs directly from this run's CSV/NPZ, without smoothing results."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numbers(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def plot_bandwidth(root, report, figures):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    colors = ["#cb593b", "#1976a3"]
    cpu_check = report["status"] == "FROZEN_CPU_DIAGNOSTIC_NOT_TRAINING"
    levels = (10, 12) if cpu_check else (3, 6)
    axis = "cpu_ghz" if cpu_check else "bandwidth_mhz"
    unit = "GHz" if cpu_check else "MHz"
    seeds = sorted({row["seed"] for row in report["rows"]})
    x = np.arange(len(seeds))
    for index, (level, color) in enumerate(zip(levels, colors)):
        rows = sorted((row for row in report["rows"] if row[axis] == level), key=lambda row: row["seed"])
        label = f"{level} {unit}"
        prefix = f"b6c{level}" if cpu_check else f"b{level}"
        for ax, key in ((axes[0,0], "task_delay_weighted_ms"), (axes[1,0], "task_delay_low_ms")):
            bars = ax.bar(x+(index-.5)*.34, numbers(rows,key)/1000, width=.34, color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
            ax.set_xticks(x, seeds)
            ax.set_xlabel("评估场景种子")
        queues = [read_csv(root/f"{prefix}_{seed}_queues.csv") for seed in seeds]
        curves = np.asarray([numbers(series, "queue_total_mbit") for series in queues])
        times = numbers(queues[0], "time_s")
        axes[1,1].plot(times, curves.mean(axis=0), color=color, label=f"{label}：3 场景均值")
        axes[1,1].fill_between(times, curves.min(axis=0), curves.max(axis=0), color=color, alpha=.12)
    bottom = np.zeros(2)
    for key, label, color in (("compute", "等待计算", "#69797b"), ("transfer", "等待 U2U 转发", "#be8a31"), ("upload", "等待上传", "#4e9561")):
        values = np.asarray([report["means"][str(level)][f"mean_queue_{key}_mbit"] for level in levels])
        axes[0,1].bar([0,1], values, bottom=bottom, color=color, width=.55, label=label)
        for index, value in enumerate(values):
            if value > 10:
                axes[0,1].text(index, bottom[index]+value/2, f"{value:.1f}", ha="center", va="center", color="white")
        bottom += values
    axes[0,1].set_xticks([0,1], [f"{level} {unit}" for level in levels])
    axes[0,0].set(title="完成任务的加权端到端时延", ylabel="时延 / s")
    axes[1,0].set(title="完成的低优先级任务时延", ylabel="时延 / s")
    axes[0,1].set(title="整段运行的平均积压位置（非时延比例）", ylabel="0.5 s 采样时间均值 / Mbit")
    axes[1,1].set(title="连续积压：阴影为 3 场景最小至最大值", xlabel="运行时间 / s", ylabel="上传 + 转发 + 计算队列 / Mbit")
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.margins(y=.13)
    axes[0,0].legend()
    axes[0,1].legend(fontsize=9)
    axes[1,1].legend(fontsize=9)
    title = "CPU 诊断：同一冻结 SAC、固定 6 MHz，仅每架 CPU 10 → 12 GHz" if cpu_check else "U2U 带宽诊断：同一冻结 SAC、同一质心轨迹，仅总带宽 3 → 6 MHz"
    fig.suptitle(title+"\n零训练；每场景连续 900 s；完成任务时延须结合未完成积压解读", fontsize=13)
    fig.savefig(figures/("01_cpu_diagnostic.png" if cpu_check else "01_bandwidth_diagnostic.png"), dpi=160)
    plt.close(fig)


def plot_routing_control(root, report, figures):
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    reference = Path(manifest["reference"])
    learned = json.loads((reference/"report.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2,2,figsize=(13,8),layout="constrained")
    colors = ["#cb593b", "#1976a3"]
    for index,(current,label,color) in enumerate(((learned,"训练后 SAC",colors[0]),(report,"固定均分（不学习）",colors[1]))):
        for ax,key,scale in ((axes[0,0],"task_delay_weighted_ms",1000),(axes[0,1],"queue_total_mbit",1)):
            vals = [current["means"][str(ghz)][key]/scale for ghz in (10,12)]
            bars = ax.bar(np.arange(2)+(index-.5)*.32, vals, width=.32, color=color, label=label)
            ax.bar_label(bars,fmt="%.2f",padding=3)
            ax.set_xticks([0,1],["10 GHz / UAV","12 GHz / UAV"])
        seeds = sorted({row["seed"] for row in report["rows"]})
        directory,prefix = (reference,"b6c10") if index==0 else (root,"uniform_c10")
        queues = [read_csv(directory/f"{prefix}_{seed}_queues.csv") for seed in seeds]
        curves = np.asarray([numbers(rows,"queue_total_mbit") for rows in queues])
        times = numbers(queues[0],"time_s")
        axes[1,1].plot(times,curves.mean(axis=0),color=color,label=label)
        axes[1,1].fill_between(times,curves.min(axis=0),curves.max(axis=0),color=color,alpha=.12)
        flow = []
        for role in ("heavy","medium","light"):
            rates = []
            for seed in seeds:
                rows = [row for row in read_csv(directory/f"{prefix}_{seed}_uavs.csv") if row["role"]==role]
                rates.append(numbers(rows,"assigned_destination_mbit").sum()/(len(rows)*.5))
            flow.append(np.mean(rates))
        bars = axes[1,0].bar(np.arange(3)+(index-.5)*.32,flow,width=.32,color=color,label=label)
        axes[1,0].bar_label(bars,fmt="%.2f",padding=3)
    axes[0,0].set(title="完成任务加权时延：均分反而更低",ylabel="3 场景均值 / s")
    axes[0,1].set(title="900 秒末的未完成积压",ylabel="3 场景均值 / Mbit")
    axes[1,0].set_xticks([0,1,2],["原始重载区 UAV","原始中载区 UAV","原始轻载区 UAV"])
    axes[1,0].axhline(9,color="#555555",linestyle="--",label="10 GHz 含部署暂停的平均容量")
    axes[1,0].set(title="10 GHz：实际被分配的计算工作量",ylabel="按当时负载角色分组 / Mbps",ylim=(0,12))
    axes[1,1].set(title="10 GHz：连续队列（阴影为场景范围）",ylabel="总积压 / Mbit",xlabel="运行时间 / s")
    axes[0,0].legend(fontsize=9)
    axes[1,0].legend(fontsize=8,loc="lower left")
    axes[1,1].legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(axis="y",alpha=.15)
        ax.set_axisbelow(True)
        ax.margins(y=.15)
    fig.suptitle("分配诊断：固定 6 MHz、相同场景和质心轨迹；均分不是新 RL 成果\n每个源的两类任务均按 1/3 分到各目的地，无教师、无 SCA、无训练",fontsize=13)
    fig.savefig(figures/"01_routing_diagnostic.png",dpi=160)
    plt.close(fig)


def plot_reference(root, report, config, figures):
    methods = [("native", "原更新实现（同预算）", "#cb593b"),
               ("sb3", "SB3 原生更新（同预算）", "#1976a3")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for key, label, color in methods:
        training = read_csv(root/key/"training.csv")
        validation = read_csv(root/key/"validation.csv")
        x = numbers(training, "window")
        axes[0,0].plot(x, numbers(training,"window_reward"), color=color, label=label)
        axes[1,0].plot(x, numbers(training,"queue_total_mbit"), color=color)
        windows = sorted(set(int(row["window"]) for row in validation))
        delay, completion = [], []
        for window in windows:
            selected = [row for row in validation if int(row["window"]) == window]
            delay.append(numbers(selected,"task_delay_weighted_ms").mean())
            completion.append(100*numbers(selected,"task_completion_rate").mean())
        axes[0,1].plot(windows, delay, "o-", color=color)
        axes[1,1].plot(windows, completion, "o-", color=color)
    for ax in axes.flat:
        ax.set_xlabel("预训练窗口（每个 300 s）")
        ax.grid(alpha=.15)
    axes[0,0].set(title="原始训练奖励（越高越好）", ylabel="平均宏观奖励")
    axes[0,0].legend(fontsize=9)
    axes[0,1].set(title="固定验证场景：完成任务加权时延", ylabel="时延 / ms")
    axes[1,0].set(title="训练积压（每 900 s 换场景）", ylabel="窗口末队列 / Mbit")
    axes[1,1].set(title="固定验证场景：任务完成率", ylabel="已完成 / 已到达任务 / %", ylim=(0, 102))
    fig.suptitle("纯 SAC 微观对照：相同 MDP、初始网络、数据量和更新预算\n上层均为质心；原始曲线未平滑，训练队列的锯齿不等于恢复", fontsize=13)
    fig.savefig(figures/"01_sac_training.png", dpi=160)
    plt.close(fig)
    seeds = config["test_seeds"]
    fig, axes = plt.subplots(len(seeds),2,figsize=(13,3*len(seeds)),layout="constrained",squeeze=False)
    for index, seed in enumerate(seeds):
        for key,label,color in methods:
            rows = read_csv(root/key/f"final_{seed}_queues.csv")
            times = numbers(rows,"time_s")
            axes[index,0].plot(times,numbers(rows,"queue_total_mbit"),label=label,color=color)
            axes[index,1].plot(times,100*numbers(rows,"task_completion_rate"),color=color)
        axes[index,0].set(title=f"场景 {seed}：未完成积压",ylabel="上传 + 转发 + 计算队列 / Mbit")
        axes[index,1].set(title=f"场景 {seed}：累计任务完成率",ylabel="完成率 / %",ylim=(0,102))
        for ax in axes[index]:
            ax.set_xlabel("连续运行时间 / s")
            ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=9)
    fig.suptitle("最终权重：每个测试场景逐一展示，不在 300/600 s 清空队列\n一训练种子的实现对照，不是多训练种子论文结论",fontsize=13)
    fig.savefig(figures/"02_sac_continuous_queues.png",dpi=160)
    plt.close(fig)
    print(json.dumps({"figures":str(figures),"status":report["status"]},ensure_ascii=False))


def plot_mainline(root, report, config, figures):
    training = read_csv(root/"training.csv")
    validation = read_csv(root/"validation.csv")
    baseline = read_csv(root/"baseline_validation.csv")
    colors = ["#1976a3", "#c35440", "#34834c"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    x = np.arange(1, len(training)+1)
    axes[0,0].plot(x, numbers(training, "window_reward"), color=colors[0])
    axes[0,0].set(title="原始训练奖励：越高越好", ylabel="每宏观周期奖励")
    axes[1,0].plot(x, numbers(training, "critic_loss"), color=colors[0])
    axes[1,0].set_yscale("symlog", linthresh=1)
    axes[1,0].set(title="Critic 回归损失（对称对数坐标）", ylabel="最后一次双 Q 损失")
    for index, seed in enumerate(config["validation_seeds"]):
        rows = [row for row in validation if int(row["seed"]) == seed]
        vx = [int(row["window"])+(config["training"]["pretrain_windows"] if row["stage"] == "joint" else 0) for row in rows]
        for ax, key in ((axes[0,1], "task_delay_weighted_ms"), (axes[1,1], "queue_total_mbit")):
            ax.plot(vx, numbers(rows, key), "o-", color=colors[index], label=f"场景 {seed}")
            value = next(float(row[key]) for row in baseline if int(row["seed"]) == seed)
            ax.axhline(value, color=colors[index], ls="--", lw=1)
    axes[0,1].set(title="固定 900 s 验证：完成任务加权时延", ylabel="时延 / ms")
    axes[1,1].set(title="固定 900 s 验证：期末全部积压", ylabel="上传 + 转发 + 计算 / Mbit")
    axes[0,1].legend(fontsize=9)
    for ax in axes.flat:
        ax.set_xlabel("训练窗口（每个 300 s）")
        ax.grid(alpha=.15)
        if report["joint_training_started"]:
            ax.axvline(config["training"]["pretrain_windows"]+.5, color="#777777", ls=":")
    stage = "已进入 PPO + SAC 同步联训" if report["joint_training_started"] else "仅微观 SAC；未启动 PPO"
    fig.suptitle(f"6 MHz / 10 GHz：从零训练，{stage}\n原始曲线未平滑；虚线为只读均分对照，不是教师；训练奖励与确定性验证分开解释", fontsize=12)
    fig.savefig(figures/"01_training.png", dpi=160)
    plt.close(fig)

    seeds, label = config["test_seeds"], report["final_label"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    name = "PPO + SAC 最后权重" if report["joint_training_started"] else "SAC 预训练最后权重"
    for index, (rows, current_label, current_name, color) in enumerate((
        (report["final"], label, name, colors[0]),
        (report["uniform"], "uniform_test", "固定均分（不学习）", colors[1]),
    )):
        for ax, key in ((axes[0,0], "task_delay_weighted_ms"), (axes[0,1], "task_delay_low_ms")):
            vals = [next(row[key] for row in rows if row["seed"] == seed) for seed in seeds]
            bars = ax.bar(np.arange(len(seeds))+(index-.5)*.32, vals, width=.32, color=color, label=current_name)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
            ax.set_xticks(range(len(seeds)), seeds)
            ax.set_xlabel("独立测试场景种子")
        rates = []
        for role in ("heavy", "medium", "light"):
            values = []
            for seed in seeds:
                data = [r for r in read_csv(root/f"{current_label}_{seed}_uavs.csv") if r["role"] == role]
                values.append(numbers(data, "assigned_destination_mbit").sum()/(len(data)*.5))
            rates.append(np.mean(values))
        bars = axes[1,0].bar(np.arange(3)+(index-.5)*.32, rates, width=.32, color=color, label=current_name)
        axes[1,0].bar_label(bars, fmt="%.2f", padding=3)
        for k, seed in enumerate(seeds):
            queues = read_csv(root/f"{current_label}_{seed}_queues.csv")
            axes[1,1].plot(numbers(queues,"time_s"), numbers(queues,"queue_total_mbit"),
                           color=color, ls=("-", "--", ":")[k % 3], label=f"{current_name} / {seed}", lw=1)
    axes[0,0].set(title="已完成任务：10:1 加权端到端时延", ylabel="时延 / ms")
    axes[0,1].set(title="已完成的低优先级任务时延", ylabel="时延 / ms")
    axes[1,0].set_xticks([0,1,2], ["原始重载角色", "原始中载角色", "原始轻载角色"])
    axes[1,0].axhline(9, color="#555555", ls="--", label="含部署暂停的单架平均容量")
    axes[1,0].set(title="实际分配到目的 UAV 的计算工作量", ylabel="按当时角色分组 / Mbps")
    axes[1,1].set(title="全部测试场景：连续队列，不清空重置", ylabel="全部积压 / Mbit", xlabel="时间 / s")
    axes[0,0].legend(fontsize=9)
    axes[1,0].legend(fontsize=8)
    axes[1,1].legend(fontsize=7, ncol=2)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.margins(y=.17)
    complete = report["mean"]["task_completion_rate"]*100
    axes[1,1].set_ylim(bottom=0)
    fig.suptitle(f"同场景只读诊断：{name} vs 均分；不是论文方法排名\nSAC 完成率 {complete:.2f}%；已完成任务时延必须结合积压读；仅一个训练种子", fontsize=12)
    fig.savefig(figures/"02_outcome.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    root = args.result.resolve()
    report = json.loads((root/"report.json").read_text(encoding="utf-8"))
    config = json.loads((root/"manifest.json").read_text(encoding="utf-8"))["config"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.sans-serif":[f for f in ("Microsoft YaHei","SimHei","Noto Sans CJK SC","DejaVu Sans") if f in available],
                         "axes.unicode_minus":False,"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    figures = root/"figures"
    figures.mkdir(exist_ok=True)
    if "pretrain_admission" in report:
        plot_mainline(root, report, config, figures)
        print(str(figures))
        return
    if report["status"] == "FIXED_UNIFORM_ROUTING_DIAGNOSTIC_NOT_TRAINING":
        plot_routing_control(root,report,figures)
        print(str(figures))
        return
    if report["status"] in ("FROZEN_BANDWIDTH_DIAGNOSTIC_NOT_TRAINING", "FROZEN_CPU_DIAGNOSTIC_NOT_TRAINING"):
        plot_bandwidth(root, report, figures)
        print(str(figures))
        return
    if "arms" in report:
        plot_reference(root, report, config, figures)
        return
    colors = ["#1976a3","#cb593b","#34834c"]
    training = read_csv(root/"training.csv")
    validation = read_csv(root/"validation.csv")
    pretrain = config["training"]["pretrain_windows"]
    x = np.arange(1,len(training)+1)
    fig, axes = plt.subplots(2,2,figsize=(13,8),layout="constrained")
    ax = axes[0,0]
    ax.plot(x,numbers(training,"window_reward"),color=colors[0],lw=1.2)
    ax.set(title="训练奖励（越高越好，原始值）",ylabel="每宏观周期平均折扣奖励")
    ax = axes[0,1]
    labels = list(dict.fromkeys(r["policy"] for r in validation))
    vx, vy = [], []
    for label in labels:
        records = [r for r in validation if r["policy"]==label]
        stage, window = label.split("_")[-2:]
        vx.append(int(window)+(pretrain if stage=="joint" else 0))
        vy.append(numbers(records,"task_delay_weighted_ms").mean())
    ax.plot(vx,vy,"o-",color=colors[1])
    ax.set(title="同一验证种子的确定性评估",ylabel="加权完成任务时延 / ms")
    ax = axes[1,0]
    ax.plot(x,numbers(training,"queue_total_mbit"),color=colors[1],lw=1.2)
    ax.set(title="每个 300 s 窗口末的真实积压",ylabel="上传 + 转发 + 计算队列 / Mbit")
    ax = axes[1,1]
    ax.plot(x,numbers(training,"critic_loss"),color=colors[2],lw=1.2)
    ax.set_yscale("symlog",linthresh=1)
    ax.set(title="SAC Critic 损失（对称对数坐标）",ylabel="最后一次更新的双 Q 回归损失")
    for ax in axes.flat:
        ax.axvline(pretrain+.5,color="#666666",ls="--",lw=.9)
        ax.set_xlabel("训练窗口（每个 300 s）")
        ax.grid(alpha=.15)
    fig.suptitle("短 pilot 训练记录：虚线后 PPO + SAC 同步更新\n每 3 个窗口换一个新场景；队列锯齿下降可能来自换场景，不能据此宣称收敛",fontsize=13)
    fig.savefig(figures/"01_training.png",dpi=160)
    plt.close(fig)

    evaluation = read_csv(root/"final_evaluation.csv")
    fig, axes = plt.subplots(1,3,figsize=(13,4.4),layout="constrained")
    metrics = [("task_delay_high_ms","高优先级时延 / ms"),("task_delay_low_ms","低优先级时延 / ms"),("task_delay_weighted_ms","10:1 加权时延 / ms")]
    for ax,(key,title) in zip(axes,metrics):
        final = numbers([r for r in evaluation if r["policy"]=="final"],key)
        centroid = numbers([r for r in evaluation if r["policy"]=="centroid_same_sac"],key)
        for a,b in zip(final,centroid):
            ax.plot([0,1],[a,b],"o-",color="#a6a6a6",alpha=.75,lw=1)
        ax.scatter([0,1],[final.mean(),centroid.mean()],s=85,color=colors[:2],marker="D",zorder=5)
        ax.set_xticks([0,1],["PPO + Direct SAC","同 SAC 强制质心"])
        ax.set_xlim(-.3,1.3)
        ax.set(title=title,ylabel="完成任务均值 / ms")
        ax.grid(axis="y",alpha=.15)
    fig.suptitle(f"最终权重：{len(config['test_seeds'])} 个 900 s 场景的配对读数\n只代表这一次训练；灰线为单个场景，菱形为均值，不是独立方法排名",fontsize=12)
    fig.savefig(figures/"02_delay.png",dpi=160)
    plt.close(fig)

    seeds = config["test_seeds"]
    fig, axes = plt.subplots(len(seeds),2,figsize=(13,3.0*len(seeds)),layout="constrained",squeeze=False)
    for i,seed in enumerate(seeds):
        for policy,color,label in [("final",colors[0],"PPO + SAC"),("centroid_same_sac",colors[1],"同 SAC 强制质心")]:
            rows = read_csv(root/f"{policy}_{seed}_queues.csv")
            times = numbers(rows,"time_s")
            axes[i,0].plot(times,numbers(rows,"queue_total_mbit"),color=color,label=label)
            axes[i,1].plot(times,numbers(rows,"oldest_age_s"),color=color,label=label)
        for ax in axes[i]:
            ax.grid(alpha=.15)
            ax.set_xlabel("连续运行时间 / s")
            for boundary in (300,600):
                if boundary < config["test_macros"]*10:
                    ax.axvline(boundary,color="#aaaaaa",ls=":",lw=1)
        axes[i,0].set(title=f"场景 {seed}：全部队列",ylabel="积压 / Mbit")
        axes[i,1].set(title=f"场景 {seed}：最老未完成任务",ylabel="等待年龄 / s")
    axes[0,0].legend(fontsize=9)
    fig.suptitle("稳定性检查：300/600 s 不重置；时延必须和积压、删失一起读",fontsize=13)
    fig.savefig(figures/"03_continuous_queues.png",dpi=160)
    plt.close(fig)

    # Always use the first predeclared test seed, never select by rank or beauty.
    seed = seeds[0]
    final = np.load(root/f"final_{seed}_trajectory.npz")
    centroid = np.load(root/f"centroid_same_sac_{seed}_trajectory.npz")
    same_users = np.array_equal(final["users"],centroid["users"])
    fig, axes = plt.subplots(1,2,figsize=(13,6.0),layout="constrained")
    ax = axes[0]
    for user in range(final["users"].shape[1]):
        ax.plot(*final["users"][:,user,:].T,color="#9b9b9b",alpha=.35,lw=.5)
    for uav,color in enumerate(colors):
        ax.plot(*final["uav"][:,uav,:].T,color=color,lw=1.8,label=f"UAV {uav+1}：PPO")
        ax.plot(*centroid["uav"][:,uav,:].T,color=color,lw=1.3,ls="--",label=f"UAV {uav+1}：质心")
        ax.scatter(*final["uav"][-1,uav,:],color=color,s=35,marker="s")
    ax.set(xlim=(0,1000),ylim=(0,1000),xlabel="水平位置 x / m",ylabel="水平位置 y / m",title=f"实测轨迹：预先指定场景 {seed}")
    ax.set_aspect("equal")
    ax.legend(fontsize=8,ncol=2,loc="upper left")
    ax = axes[1]
    times = np.arange(1,len(final["uav"])+1)*10
    for uav,color in enumerate(colors):
        gap = np.linalg.norm(final["uav"][:,uav]-centroid["uav"][:,uav],axis=1)
        ax.plot(times,gap,color=color,label=f"UAV {uav+1}")
    ax.set(title="同一时刻的两条 UAV 轨迹相距多远",xlabel="时间 / s",ylabel="PPO 与强制质心的位置差 / m")
    ax.legend()
    ax.grid(alpha=.15)
    fig.suptitle(f"灰细线为地面用户；虚线为质心对照；用户轨迹逐点一致：{'是' if same_users else '否，不能视为精确配对'}\n本图是单场景机制展示，不代表平均排名；轨迹不同不等于轨迹有收益",fontsize=12)
    fig.savefig(figures/"04_trajectory.png",dpi=160)
    plt.close(fig)
    print(json.dumps({"figures":str(figures),"trajectory_seed":seed,"same_user_trace":same_users,"status":report["status"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
