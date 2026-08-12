import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#2878B5", "#F28E2B", "#2CA02C")


def parse_args():
    parser = argparse.ArgumentParser(description="绘制 v33 中文收敛与轨迹结果图")
    parser.add_argument(
        "--run-dir",
        default="results/tcom_v33_soft_energy_random_mobility/formal/seed1",
    )
    parser.add_argument(
        "--output-dir",
        default="result_archive/v33_seed1_summary",
    )
    return parser.parse_args()


def resolve(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def configure_plotting():
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
        }
    )


def read_eval_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"评估文件为空: {path}")
    return {
        "episode": np.asarray([float(row["episode"]) for row in rows]),
        "delay": np.asarray([float(row["task_delay_weighted_ms"]) for row in rows]),
        "queue": np.asarray(
            [float(row["q_high"]) + float(row["q_low"]) for row in rows]
        ),
        "power": np.asarray([float(row["average_power_w"]) for row in rows]),
        "completed": np.asarray([float(row["task_completed_count"]) for row in rows]),
        "arrived": np.asarray([float(row["task_arrived_count"]) for row in rows]),
    }


def style_axis(axis):
    axis.grid(True, color="#D6D6D6", linestyle="--", linewidth=0.7, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)


def shade_stages(axis, total_episode):
    axis.axvspan(0, 300, color="#DCEAF7", alpha=0.48, zorder=0)
    axis.axvspan(300, total_episode, color="#F7E5D2", alpha=0.36, zorder=0)
    axis.axvline(300, color="#555555", linestyle="--", linewidth=1.1)


def add_inset(axis, x, y, ylabel, y_floor=0.0):
    inset = inset_axes(axis, width="31%", height="41%", loc="upper right", borderpad=1.0)
    inset.plot(x, y, color="#2878B5", marker="o", markersize=2.8, linewidth=1.2)
    inset.set_xlim(0, 310)
    upper = max(float(np.max(y)) * 1.08, y_floor + 1.0)
    inset.set_ylim(y_floor, upper)
    inset.set_title("阶段 I 全尺度", fontsize=8.5, pad=3)
    inset.set_xlabel("训练轮数", fontsize=7.5)
    inset.set_ylabel(ylabel, fontsize=7.5)
    inset.tick_params(labelsize=7)
    inset.grid(True, linestyle=":", alpha=0.5)


def plot_convergence(micro, joint, output_dir):
    micro_x = micro["episode"]
    joint_x = 300.0 + joint["episode"]
    total_episode = float(joint_x[-1])
    stage_boundary_delay = float(micro["delay"][-1])
    joint_tail = joint["episode"] >= 600
    joint_tail_delay = float(np.mean(joint["delay"][joint_tail]))

    fig, axes = plt.subplots(3, 1, figsize=(10.6, 11.0), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.08, hspace=0.25)
    fig.suptitle("v33 分层强化学习完整训练过程（固定测试集）", fontsize=16, fontweight="bold")

    for axis in axes:
        shade_stages(axis, total_episode)
        style_axis(axis)

    axes[0].plot(
        micro_x,
        micro["delay"],
        color="#2878B5",
        marker="o",
        markersize=4,
        linewidth=2.0,
        label="阶段 I：SAC 学习，轨迹采用动态质心跟踪",
    )
    axes[0].plot(
        joint_x,
        joint["delay"],
        color="#D55E00",
        marker="s",
        markersize=4,
        linewidth=2.0,
        label="阶段 II：TD3-SAC 联合学习",
    )
    axes[0].axhline(
        stage_boundary_delay,
        color="#2878B5",
        linestyle=":",
        linewidth=1.3,
        label=f"阶段 I 终点：{stage_boundary_delay:.1f} ms",
    )
    axes[0].hlines(
        joint_tail_delay,
        float(joint_x[joint_tail][0]),
        float(joint_x[joint_tail][-1]),
        color="#8C2D04",
        linestyle="--",
        linewidth=1.4,
        label=f"阶段 II 后半程均值：{joint_tail_delay:.1f} ms",
    )
    visible_delay = np.concatenate((micro["delay"][1:], joint["delay"]))
    delay_low = max(0.0, float(np.min(visible_delay)) * 0.92)
    delay_high = max(900.0, float(np.max(visible_delay)) * 1.08)
    axes[0].set_ylim(delay_low, delay_high)
    axes[0].set_ylabel("优先级加权端到端时延 / ms")
    axes[0].set_title("(a) 平均端到端时延（越低越好）", loc="left", fontweight="bold")
    axes[0].legend(loc="upper left", frameon=False, ncol=1)
    add_inset(axes[0], micro_x, micro["delay"], "时延/ms")

    axes[1].plot(
        micro_x,
        micro["queue"],
        color="#2878B5",
        marker="o",
        markersize=4,
        linewidth=2.0,
    )
    axes[1].plot(
        joint_x,
        joint["queue"],
        color="#D55E00",
        marker="s",
        markersize=4,
        linewidth=2.0,
    )
    visible_queue = np.concatenate((micro["queue"][1:], joint["queue"]))
    axes[1].set_ylim(0.0, max(25.0, float(np.max(visible_queue)) * 1.12))
    axes[1].set_ylabel("平均总队列积压 / Mbit")
    axes[1].set_title("(b) 队列积压（越低且不持续增长越好）", loc="left", fontweight="bold")
    add_inset(axes[1], micro_x, micro["queue"], "队列/Mbit")

    axes[2].plot(
        micro_x,
        micro["power"],
        color="#2878B5",
        marker="o",
        markersize=4,
        linewidth=2.0,
    )
    axes[2].plot(
        joint_x,
        joint["power"],
        color="#D55E00",
        marker="s",
        markersize=4,
        linewidth=2.0,
    )
    all_power = np.concatenate((micro["power"], joint["power"]))
    padding = max(8.0, float(np.ptp(all_power)) * 0.22)
    axes[2].set_ylim(float(np.min(all_power)) - padding, float(np.max(all_power)) + padding)
    axes[2].set_ylabel("系统平均功率 / W")
    axes[2].set_xlabel("统一训练轮数")
    axes[2].set_title("(c) 系统平均功率（软能耗目标，无硬阈值）", loc="left", fontweight="bold")

    axes[2].text(
        150,
        axes[2].get_ylim()[0] + 0.05 * np.ptp(axes[2].get_ylim()),
        "阶段 I\n仅学习微观资源分配",
        ha="center",
        va="bottom",
        color="#1F5A89",
        fontweight="bold",
    )
    axes[2].text(
        (300 + total_episode) / 2,
        axes[2].get_ylim()[0] + 0.05 * np.ptp(axes[2].get_ylim()),
        "阶段 II\n两层同时更新",
        ha="center",
        va="bottom",
        color="#A24500",
        fontweight="bold",
    )

    png = output_dir / "v33_完整收敛曲线.png"
    pdf = output_dir / "v33_完整收敛曲线.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def latest_trajectory_npz(directory):
    final = directory / "trajectory_final.npz"
    if final.exists():
        return final
    candidates = sorted(directory.glob("trajectory_ep_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"没有找到轨迹文件: {directory}")
    return candidates[-1]


def load_trajectory(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def plot_trajectory_panel(axis, data, title):
    uav = data["uav_traj"]
    centroid = data["centroid_traj"]
    users = data["users_traj"]
    allocations = data["user_allocations"].astype(int)
    sample_stride = max(1, users.shape[0] // 300)

    for zone, color in enumerate(COLORS):
        user_indices = np.flatnonzero(allocations == zone)
        for user_index in user_indices:
            path = users[::sample_stride, user_index]
            axis.plot(path[:, 0], path[:, 1], color=color, alpha=0.18, linewidth=0.65)
        axis.scatter(
            users[0, user_indices, 0],
            users[0, user_indices, 1],
            s=12,
            facecolors="white",
            edgecolors=color,
            linewidths=0.7,
            alpha=0.8,
        )
        axis.scatter(
            users[-1, user_indices, 0],
            users[-1, user_indices, 1],
            s=15,
            color=color,
            alpha=0.72,
        )
        axis.plot(
            centroid[::sample_stride, zone, 0],
            centroid[::sample_stride, zone, 1],
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
        axis.plot(
            uav[:, zone, 0],
            uav[:, zone, 1],
            color=color,
            linewidth=2.5,
            label=f"UAV-{zone + 1}",
        )
        axis.scatter(
            uav[0, zone, 0],
            uav[0, zone, 1],
            marker="s",
            s=42,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
        )
        axis.scatter(
            uav[-1, zone, 0],
            uav[-1, zone, 1],
            marker="*",
            s=125,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )
        axis.scatter(
            centroid[-1, zone, 0],
            centroid[-1, zone, 1],
            marker="+",
            s=105,
            color=color,
            linewidth=2.3,
            zorder=5,
        )

    axis.set_title(title, fontweight="bold")
    axis.set_xlabel("X 坐标 / m")
    axis.set_ylabel("Y 坐标 / m")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle="--", alpha=0.45)
    axis.spines[["top", "right"]].set_visible(False)


def plot_trajectories(micro_path, joint_path, output_dir):
    micro = load_trajectory(micro_path)
    joint = load_trajectory(joint_path)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.19, wspace=0.22)
    plot_trajectory_panel(axes[0], micro, "(a) 阶段 I 结束：动态质心跟踪")
    plot_trajectory_panel(axes[1], joint, "(b) 阶段 II 结束：TD3-SAC 联合策略")

    all_xy = np.concatenate(
        (
            micro["users_traj"].reshape(-1, 2),
            micro["uav_traj"].reshape(-1, 2),
            joint["users_traj"].reshape(-1, 2),
            joint["uav_traj"].reshape(-1, 2),
        ),
        axis=0,
    )
    mins = np.min(all_xy, axis=0) - 45.0
    maxs = np.max(all_xy, axis=0) + 45.0
    for axis in axes:
        axis.set_xlim(mins[0], maxs[0])
        axis.set_ylim(mins[1], maxs[1])

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.055),
    )
    fig.suptitle(
        "地面用户独立随机漂移下的 UAV 轨迹",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "细线：地面用户轨迹；虚线：业务质心；方块：UAV 起点；星形：UAV 终点；加号：最终业务质心",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )

    png = output_dir / "v33_用户与无人机轨迹对比.png"
    pdf = output_dir / "v33_用户与无人机轨迹对比.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def linear_slope_per_100(x, y):
    if len(x) < 2:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0] * 100.0)


def trajectory_stats(path):
    data = load_trajectory(path)
    uav = data["uav_traj"]
    centroid = data["centroid_traj"]
    return {
        "final_distance": np.linalg.norm(uav[-1] - centroid[-1], axis=1),
        "path_length": np.linalg.norm(np.diff(uav, axis=0), axis=2).sum(axis=0),
    }


def write_summary(micro, joint, micro_trajectory, joint_trajectory, output_dir):
    final_delay = float(joint["delay"][-1])
    baseline_delay = float(micro["delay"][-1])
    final_queue = float(joint["queue"][-1])
    final_power = float(joint["power"][-1])
    tail = joint["episode"] >= 600
    tail_delay = joint["delay"][tail]
    tail_queue = joint["queue"][tail]
    tail_power = joint["power"][tail]
    micro_traj = trajectory_stats(micro_trajectory)
    joint_traj = trajectory_stats(joint_trajectory)

    fmt_vector = lambda values: "/".join(f"{float(value):.1f}" for value in values)
    text = (
        "# v33 seed=1 最终结果\n\n"
        "## 判决\n\n"
        "- 微观 SAC 已收敛：阶段 I 后段的时延和队列进入稳定平台。\n"
        "- 系统队列未发散，但宏观 TD3 未形成稳定收益：联合阶段后半程时延呈轻微上升，最终结果劣于阶段 I 终点。\n"
        "- 因 seed=1 未通过宏观收敛门禁，本轮不继续消耗算力跑更多训练种子。\n"
        "- 地面用户采用逐用户独立随机漂移；功率仅作为 0.1 权重的软优化项，不设硬功率阈值。\n\n"
        "## 数值摘要\n\n"
        f"- 阶段 I 终点优先级加权端到端时延：{baseline_delay:.3f} ms\n"
        f"- 阶段 II 最终优先级加权端到端时延：{final_delay:.3f} ms\n"
        f"- 阶段 II 相对阶段 I 终点变化：{final_delay - baseline_delay:+.3f} ms（正值表示变差）\n"
        f"- 阶段 II 后半程（600--1000 轮）时延：{np.mean(tail_delay):.3f} ± {np.std(tail_delay, ddof=1):.3f} ms，趋势 {linear_slope_per_100(joint['episode'][tail], tail_delay):+.3f} ms/100轮\n"
        f"- 阶段 II 后半程总队列：{np.mean(tail_queue):.3f} ± {np.std(tail_queue, ddof=1):.3f} Mbit，趋势 {linear_slope_per_100(joint['episode'][tail], tail_queue):+.3f} Mbit/100轮\n"
        f"- 阶段 II 最终总队列：{final_queue:.3f} Mbit\n"
        f"- 阶段 II 后半程系统平均功率：{np.mean(tail_power):.3f} ± {np.std(tail_power, ddof=1):.3f} W；最终 {final_power:.3f} W\n"
        f"- 阶段 I 终点 UAV--质心距离（UAV-1/2/3）：{fmt_vector(micro_traj['final_distance'])} m\n"
        f"- 阶段 II 终点 UAV--质心距离（UAV-1/2/3）：{fmt_vector(joint_traj['final_distance'])} m\n"
        f"- 阶段 II UAV 累计航程（UAV-1/2/3）：{fmt_vector(joint_traj['path_length'])} m\n\n"
        "## 图表口径\n\n"
        "收敛图全部使用固定 20 个测试种子的确定性评估结果，不使用训练回报冒充测试性能。轨迹图保留全部地面用户轨迹、动态业务质心和三架 UAV 的真实轨迹。\n"
    )
    path = output_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    args = parse_args()
    configure_plotting()
    run_dir = resolve(args.run_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    micro_dir = run_dir / "micro_pretrain"
    joint_dir = run_dir / "joint_td3_sac"
    micro = read_eval_csv(micro_dir / "eval_central_macro.csv")
    joint = read_eval_csv(joint_dir / "eval_central_td3.csv")
    convergence = plot_convergence(micro, joint, output_dir)

    micro_trajectory = latest_trajectory_npz(micro_dir / "trajectory_snapshots")
    joint_trajectory = latest_trajectory_npz(joint_dir / "trajectory_snapshots")
    trajectories = plot_trajectories(micro_trajectory, joint_trajectory, output_dir)
    summary = write_summary(
        micro, joint, micro_trajectory, joint_trajectory, output_dir
    )

    for source in (micro_trajectory, joint_trajectory):
        shutil.copy2(source, output_dir / source.name.replace("trajectory_", f"{source.parent.parent.name}_"))

    print("Generated:")
    for path in (*convergence, *trajectories, summary):
        print(path)


if __name__ == "__main__":
    main()
